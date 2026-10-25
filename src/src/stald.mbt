// Stability Limit Detection (STALD) for the BDF family.
//
// Ported from OrdinaryDiffEqBDF/src/stald.jl, which is itself a transcription of
// CVODE's `cvSLdet` (Hindmarsh, 1992/1995; cvode_impl.h).
//
// Why it exists
// -------------
// BDF1 and BDF2 are A-stable: their stability regions contain the whole left
// half-plane. BDF3..BDF5 are only alpha-stable — their stability regions exclude
// a wedge around the imaginary axis — so a problem whose dominant eigenvalue sits
// in that wedge can go unstable even though the local error estimate stays small.
//
// The detector watches the *scaled derivative norms* of the last five steps. When
// the step size is near the stability boundary those norms decay geometrically
// with a characteristic ratio, and that ratio shows up in three places at once
// (the direct ratio sequence, a quartic the data satisfy, and a second relation
// through the `sigsq` cross-check). `rr` is that characteristic ratio; `rr > 1`
// means the geometric growth the data imply would push the solution outside the
// stability region, so the order is cut by one.
//
// Upstream gates collection and checking on `order >= 3`, which is why orders 1-2
// never even reach the kernel.

///|
/// CVODE's `StabilityLimitDetectionState`, reduced to `Double`.
///
/// `ssdat` is `5 x 3` row-major, `ssdat[i * 3 + k]`: row `i` is a step (row 0 the
/// newest) and column `k` a derivative level (`k = 0` is the `order - 2`
/// estimate, `k = 2` the `order` estimate). Every other matrix (`rat`, `qc`,
/// `qco`) keeps upstream's own row-major `(first, k)` layout, so the transcription
/// is a pure index shift with no reordering.
pub(all) struct StaldState {
  pub mut ssdat : Array[Double]
  /// Consecutive steps at the same order.
  pub mut nscon : Int32
  /// Total number of STALD-triggered order reductions (diagnostic only).
  pub mut nor : Int32
  pub mut enabled : Bool
  /// Order used on the previous check.
  pub mut last_order : Int32
  // CVODE tuning constants (algorithms.jl:547-552).
  pub mut rrcut : Double
  pub mut vrrtol : Double
  pub mut vrrt2 : Double
  pub mut sqtol : Double
  pub mut rrtol : Double
  pub mut tiny : Double
  // Workspace, so the kernel allocates nothing.
  pub mut smax : Array[Double]
  pub mut ssmax : Array[Double]
  pub mut rav : Array[Double]
  pub mut vrat : Array[Double]
  pub mut rat : Array[Double]
  pub mut qc : Array[Double]
  pub mut qco : Array[Double]
  pub mut drr : Array[Double]
  pub mut rrc : Array[Double]
  pub mut sigsq : Array[Double]
}

///|
/// An empty, disabled-by-default-`enabled = true` state with CVODE's own tuning
/// constants. `StaldState::new()` takes them from `FBDF`'s defaults
/// (algorithms.jl:543-552): `rrcut = 0.98`, `vrrtol = 1e-4`, `vrrt2 = 5e-4`,
/// `sqtol = 1e-3`, `rrtol = 1e-2`, `tiny = 1e-90`.
pub fn StaldState::new() -> StaldState {
  {
    ssdat: Array::make(15, 0.0),
    nscon: 0,
    nor: 0,
    enabled: true,
    last_order: 0,
    rrcut: 0.98,
    vrrtol: 0.0001,
    vrrt2: 0.0005,
    sqtol: 0.001,
    rrtol: 0.01,
    tiny: 1.0e-90,
    smax: Array::make(3, 0.0),
    ssmax: Array::make(3, 0.0),
    rav: Array::make(3, 0.0),
    vrat: Array::make(3, 0.0),
    rat: Array::make(12, 0.0),
    qc: Array::make(15, 0.0),
    qco: Array::make(15, 0.0),
    drr: Array::make(3, 0.0),
    rrc: Array::make(3, 0.0),
    sigsq: Array::make(3, 0.0),
  }
}

///|
/// `stald_collect_data!` (stald.jl:107): record the three scaled derivative
/// norms of the step that just finished.
///
/// The shift makes row 0 the newest sample, so the ratio sequence
/// `ssdat[i] / ssdat[i+1]` reads "newer over older" — the same direction CVODE
/// uses. Collection is gated on `order >= 3` upstream, and that gate matters:
/// A-stable orders have no boundary to be near, and feeding them in would only
/// poison the window with meaningless ratios.
pub fn stald_collect_data(
  stald : StaldState,
  order : Int32,
  terkm2 : Double,
  terkm1 : Double,
  terk : Double,
) -> Unit {
  if !stald.enabled || order < 3 {
    return
  }
  let mut i = 4
  while i >= 1 {
    let mut k = 0
    while k < 3 {
      stald.ssdat.unsafe_set(i * 3 + k, stald.ssdat.unsafe_get((i - 1) * 3 + k))
      k += 1
    }
    i -= 1
  }
  stald.ssdat.unsafe_set(0 * 3 + 0, terkm2 * terkm2)
  stald.ssdat.unsafe_set(0 * 3 + 1, terkm1 * terkm1)
  stald.ssdat.unsafe_set(0 * 3 + 2, terk * terk)
}

///|
/// `stald_check!` (stald.jl:133): decide whether the order that produced this
/// step is too close to the stability boundary.
///
/// Returns true when the order must be reduced by one. The counter discipline is
/// upstream's: an order change resets `nscon` (the window is only meaningful at a
/// fixed order), and the detector needs `order + 5` consecutive samples before it
/// will say anything at all — CVODE's own guard against reading noise.
pub fn stald_check(stald : StaldState, order : Int32) -> Bool {
  if !stald.enabled {
    return false
  }
  if order < 3 {
    stald.last_order = order
    return false
  }
  if order != stald.last_order {
    stald.nscon = 0
    stald.last_order = order
    return false
  }
  stald.nscon += 1
  if stald.nscon < order + 5 {
    return false
  }
  let ldflag = stald_detect(stald, order)
  if ldflag > 3 {
    stald.nor += 1
    stald.nscon = 0
    true
  } else {
    false
  }
}

///|
/// `stald_reset!` (stald.jl:165): drop the whole window. Called after a
/// derivative discontinuity and by `stepsize_controller!` on the filter path,
/// where the history has been rewritten and the old ratios no longer describe
/// anything.
pub fn stald_reset(stald : StaldState) -> Unit {
  let mut i = 0
  while i < stald.ssdat.length() {
    stald.ssdat.unsafe_set(i, 0.0)
    i += 1
  }
  stald.nscon = 0
  stald.last_order = 0
  let names = [
    stald.smax,
    stald.ssmax,
    stald.rav,
    stald.vrat,
    stald.rat,
    stald.qc,
    stald.qco,
    stald.drr,
    stald.rrc,
    stald.sigsq,
  ]
  for buf in names {
    let mut j = 0
    while j < buf.length() {
      buf.unsafe_set(j, 0.0)
      j += 1
    }
  }
}

///|
/// `_stald_detect` (stald.jl:186) — the port of CVODE's `cvSLdet`.
///
/// Returns an integer status rather than a boolean because the caller needs to
/// tell "no characteristic root could be established" (negative) from "root
/// found, magnitude below the cut" (`kflag`) from "stability violation"
/// (`kflag + 3`). Upstream keeps the same encoding for exactly that reason.
///
/// Two routes to `rr`:
///
///   * **Low variance** (`vmin < vrrtol²`): the four consecutive ratios are
///     consistent, so `rr` is just their mean. CVODE takes this path when the
///     data are clean — which is the common case far from the boundary.
///   * **Higher variance**: solve a quartic. Each derivative level `k` gives a
///     quartic in `rr` built from `ssdat[.,k]`; Gaussian elimination reduces the
///     three of them to one equation, and Newton corrections polish the root
///     when the raw solve is not accurate enough.
///
/// Either way `rr` is then cross-checked against a second relation through
/// `sigsq` (Phase 4) before it is trusted, and only then compared to `rrcut`.
fn stald_detect(stald : StaldState, q : Int32) -> Int32 {
  let ssdat = stald.ssdat
  let tiny = stald.tiny
  // Phase 1 — per-level statistics.
  let mut k = 0
  while k < 3 {
    let mut smink = ssdat.unsafe_get(0 * 3 + k)
    let mut smaxk = 0.0
    let mut i = 0
    while i < 5 {
      let v = ssdat.unsafe_get(i * 3 + k)
      if v < smink {
        smink = v
      }
      if v > smaxk {
        smaxk = v
      }
      i += 1
    }
    if smink < tiny * smaxk {
      return -1
    }
    stald.smax.unsafe_set(k, smaxk)
    stald.ssmax.unsafe_set(k, smaxk * smaxk)
    let mut sumrat = 0.0
    let mut sumrsq = 0.0
    let mut i2 = 0
    while i2 < 4 {
      let r = ssdat.unsafe_get(i2 * 3 + k) / ssdat.unsafe_get((i2 + 1) * 3 + k)
      stald.rat.unsafe_set(i2 * 3 + k, r)
      sumrat = sumrat + r
      sumrsq = sumrsq + r * r
      i2 += 1
    }
    stald.rav.unsafe_set(k, 0.25 * sumrat)
    stald.vrat.unsafe_set(
      k,
      (0.25 * sumrsq - stald.rav.unsafe_get(k) * stald.rav.unsafe_get(k)).abs(),
    )
    // Quartic coefficients, in Julia's descending-power order:
    // qc[4] rr^4 + qc[3] rr^3 + qc[2] rr^2 + qc[1] rr + qc[0] = 0.
    stald.qc.unsafe_set(
      4 * 3 + k,
      ssdat.unsafe_get(0 * 3 + k) * ssdat.unsafe_get(2 * 3 + k) -
      ssdat.unsafe_get(1 * 3 + k) * ssdat.unsafe_get(1 * 3 + k),
    )
    stald.qc.unsafe_set(
      3 * 3 + k,
      ssdat.unsafe_get(1 * 3 + k) * ssdat.unsafe_get(2 * 3 + k) -
      ssdat.unsafe_get(0 * 3 + k) * ssdat.unsafe_get(3 * 3 + k),
    )
    stald.qc.unsafe_set(2 * 3 + k, 0.0)
    stald.qc.unsafe_set(
      1 * 3 + k,
      ssdat.unsafe_get(1 * 3 + k) * ssdat.unsafe_get(4 * 3 + k) -
      ssdat.unsafe_get(2 * 3 + k) * ssdat.unsafe_get(3 * 3 + k),
    )
    stald.qc.unsafe_set(
      0 * 3 + k,
      ssdat.unsafe_get(3 * 3 + k) * ssdat.unsafe_get(3 * 3 + k) -
      ssdat.unsafe_get(2 * 3 + k) * ssdat.unsafe_get(4 * 3 + k),
    )
    k += 1
  }
  // Phase 2 — the characteristic root rr.
  let vmin = dmin3(
    stald.vrat.unsafe_get(0),
    stald.vrat.unsafe_get(1),
    stald.vrat.unsafe_get(2),
  )
  let vmax = dmax3(
    stald.vrat.unsafe_get(0),
    stald.vrat.unsafe_get(1),
    stald.vrat.unsafe_get(2),
  )
  let mut rr = 0.0
  let mut kflag = 0
  if vmin < stald.vrrtol * stald.vrrtol {
    if vmax > stald.vrrt2 * stald.vrrt2 {
      return -2
    }
    rr = (
        stald.rav.unsafe_get(0) +
        stald.rav.unsafe_get(1) +
        stald.rav.unsafe_get(2)
      ) /
      3.0
    let mut drrmax = 0.0
    let mut k2 = 0
    while k2 < 3 {
      let adrr = (stald.rav.unsafe_get(k2) - rr).abs()
      if adrr > drrmax {
        drrmax = adrr
      }
      k2 += 1
    }
    if drrmax > stald.vrrt2 {
      return -3
    }
    kflag = 1
  } else {
    // Higher variance: eliminate between the three quartics.
    let qc = stald.qc
    let qco = stald.qco
    let mut j = 0
    while j < 15 {
      qco.unsafe_set(j, qc.unsafe_get(j))
      j += 1
    }
    if qco.unsafe_get(3 * 3 + 0).abs() >
      tiny * stald.smax.unsafe_get(0) * stald.smax.unsafe_get(0) {
      let r1 = qco.unsafe_get(3 * 3 + 1) / qco.unsafe_get(3 * 3 + 0)
      let r2 = qco.unsafe_get(3 * 3 + 2) / qco.unsafe_get(3 * 3 + 0)
      let mut j2 = 4
      while j2 >= 0 {
        if j2 != 3 {
          qco.unsafe_set(
            j2 * 3 + 1,
            qco.unsafe_get(j2 * 3 + 1) - r1 * qco.unsafe_get(j2 * 3 + 0),
          )
          qco.unsafe_set(
            j2 * 3 + 2,
            qco.unsafe_get(j2 * 3 + 2) - r2 * qco.unsafe_get(j2 * 3 + 0),
          )
        }
        j2 -= 1
      }
    } else {
      return -4
    }
    if qco.unsafe_get(4 * 3 + 1).abs() >
      tiny * stald.smax.unsafe_get(1) * stald.smax.unsafe_get(1) {
      let r1 = qco.unsafe_get(4 * 3 + 2) / qco.unsafe_get(4 * 3 + 1)
      let mut j3 = 4
      while j3 >= 0 {
        if j3 != 3 && j3 != 4 {
          qco.unsafe_set(
            j3 * 3 + 2,
            qco.unsafe_get(j3 * 3 + 2) - r1 * qco.unsafe_get(j3 * 3 + 1),
          )
        }
        j3 -= 1
      }
    } else if qco.unsafe_get(4 * 3 + 2).abs() >
      tiny * stald.smax.unsafe_get(2) * stald.smax.unsafe_get(2) {
      return -4
    }
    if qco.unsafe_get(3 * 3 + 2).abs() < tiny {
      return -4
    }
    rr = -qco.unsafe_get(4 * 3 + 2) / qco.unsafe_get(3 * 3 + 2)
    if rr < tiny || rr > 100.0 {
      return -5
    }
    let mut sqmax = 0.0
    let mut k3 = 0
    while k3 < 3 {
      let qkr = qc.unsafe_get(4 * 3 + k3) +
        rr *
        (
          qc.unsafe_get(3 * 3 + k3) +
          rr * rr * (qc.unsafe_get(1 * 3 + k3) + rr * qc.unsafe_get(0 * 3 + k3))
        )
      let saqk = qkr.abs() / stald.ssmax.unsafe_get(k3)
      if saqk > sqmax {
        sqmax = saqk
      }
      k3 += 1
    }
    if sqmax < stald.sqtol {
      kflag = 2
    } else {
      // Newton corrections, at most three, each picking the level whose
      // corrected root leaves the smallest residual. `kmin` persists across
      // iterations exactly as upstream's does (it is initialized once, before
      // the loop), so a level selected on one pass is still the fallback on
      // the next even if that pass selects nothing; `-1` is the "nothing
      // selected yet" sentinel, because `0` is a legitimate 0-based index.
      let mut kmin = -1
      let mut it = 0
      while it < 3 {
        let mut k4 = 0
        while k4 < 3 {
          stald.drr.unsafe_set(k4, 0.0)
          k4 += 1
        }
        let mut sqmin = @double.infinity
        k4 = 0
        while k4 < 3 {
          let qkr = qc.unsafe_get(4 * 3 + k4) +
            rr *
            (
              qc.unsafe_get(3 * 3 + k4) +
              rr *
              rr *
              (qc.unsafe_get(1 * 3 + k4) + rr * qc.unsafe_get(0 * 3 + k4))
            )
          let qp = qc.unsafe_get(3 * 3 + k4) +
            rr *
            rr *
            (
              3.0 * qc.unsafe_get(1 * 3 + k4) +
              rr * 4.0 * qc.unsafe_get(0 * 3 + k4)
            )
          if qp.abs() > tiny * stald.ssmax.unsafe_get(k4) {
            stald.drr.unsafe_set(k4, -qkr / qp)
          }
          stald.rrc.unsafe_set(k4, rr + stald.drr.unsafe_get(k4))
          k4 += 1
        }
        let mut k5 = 0
        while k5 < 3 {
          let rk = stald.rrc.unsafe_get(k5)
          if rk > 0.0 {
            let qkr = qc.unsafe_get(4 * 3 + k5) +
              rk *
              (
                qc.unsafe_get(3 * 3 + k5) +
                rk *
                rk *
                (qc.unsafe_get(1 * 3 + k5) + rk * qc.unsafe_get(0 * 3 + k5))
              )
            let saqk = qkr.abs() / stald.ssmax.unsafe_get(k5)
            if saqk < sqmin {
              sqmin = saqk
              kmin = k5
            }
          }
          k5 += 1
        }
        if kmin < 0 {
          return -6
        }
        rr = stald.rrc.unsafe_get(kmin)
        if sqmin < stald.sqtol {
          kflag = 3
          break
        }
        it += 1
      }
      if kflag == 0 {
        return -6
      }
    }
  }
  // Phase 3 — the sigsq sequence, and a second estimate of rr from it.
  let mut k6 = 0
  while k6 < 3 {
    let rsa = ssdat.unsafe_get(0 * 3 + k6)
    let rsb = ssdat.unsafe_get(1 * 3 + k6) * rr
    let rsc = ssdat.unsafe_get(2 * 3 + k6) * rr * rr
    let rsd = ssdat.unsafe_get(3 * 3 + k6) * rr * rr * rr
    let rd1a = rsa - rsb
    let rd1b = rsb - rsc
    let rd1c = rsc - rsd
    let rd2a = rd1a - rd1b
    let rd2b = rd1b - rd1c
    let rd3a = rd2a - rd2b
    if rd1b.abs() < tiny * stald.smax.unsafe_get(k6) {
      return -7
    }
    let cest1 = -rd3a / rd1b
    if cest1 < tiny || cest1 > 4.0 {
      return -7
    }
    let corr1 = rd2b / cest1 / (rr * rr)
    stald.sigsq.unsafe_set(k6, ssdat.unsafe_get(2 * 3 + k6) + corr1)
    k6 += 1
  }
  if stald.sigsq.unsafe_get(1) < tiny {
    return -8
  }
  let ratp = stald.sigsq.unsafe_get(2) / stald.sigsq.unsafe_get(1)
  let ratm = stald.sigsq.unsafe_get(0) / stald.sigsq.unsafe_get(1)
  let qd = Double::from_int(q)
  let qfac1 = 0.25 * (qd * qd - 1.0)
  let qfac2 = 2.0 / (qd - 1.0)
  let bb = ratp * ratm - 1.0 - qfac1 * ratp
  let tem = 1.0 - qfac2 * bb
  if tem.abs() < tiny {
    return -8
  }
  let rrb = 1.0 / tem
  if (rrb - rr).abs() > stald.rrtol {
    return -9
  }
  if rr > stald.rrcut {
    kflag + 3
  } else {
    kflag
  }
}

///|
/// Three-argument minimum (Julia's `min(a, b, c)`).
fn dmin3(a : Double, b : Double, c : Double) -> Double {
  dmin(dmin(a, b), c)
}

///|
/// Three-argument maximum.
fn dmax3(a : Double, b : Double, c : Double) -> Double {
  dmax(dmax(a, b), c)
}

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

///|
/// Fill `stald.ssdat` from a flat 15-entry row-major list (row 0 newest, three
/// derivative levels per row), for the tests below.
fn stald_load(stald : StaldState, rows : Array[Double]) -> Unit {
  let mut i = 0
  while i < 15 {
    stald.ssdat.unsafe_set(i, rows.unsafe_get(i))
    i += 1
  }
}

///|
/// A two-mode solution: `h^k y^(k) = (hλ₁)^k c₁ e^{λ₁t} + (hλ₂)^k c₂ e^{λ₂t}`.
/// This is the shape the detector is built for — a *single* geometric sequence
/// always cancels exactly in Phase 3's difference sequence and bails at -7, so
/// the interesting branches are only reachable with at least two modes.
///
/// Reference values come from `tools/fbdf_expected.py::twomode_ssdat`.
fn twomode_rows(
  lam1 : Double,
  lam2 : Double,
  c1 : Double,
  c2 : Double,
  h : Double,
  q : Int32,
) -> Array[Double] {
  let out : Array[Double] = []
  let mut i = 0
  while i < 5 {
    let t = Double::from_int(6 - i) * h
    let mut k = 0
    while k < 3 {
      let lvl = Double::from_int(q - 2 + k)
      let v = fastpower(h * lam1, lvl) * c1 * @math.exp(lam1 * t) +
        fastpower(h * lam2, lvl) * c2 * @math.exp(lam2 * t)
      out.push(v * v)
      k += 1
    }
    i += 1
  }
  out
}

///|
/// A decaying two-mode solution at a well-resolved step size: the characteristic
/// ratio comes out far below `rrcut`, so the detector reports "stable" (status
/// `kflag = 1`) rather than a violation.
///
/// `λ₁ = -50, λ₂ = -20, h = 0.5, q = 3` — the fast mode is fully damped, so the
/// ratios are consistent (`vrat ≈ 0`) and the low-variance path is taken.
test "stald reports a stable two-mode window" {
  let stald = StaldState::new()
  stald_load(stald, twomode_rows(-50.0, -20.0, 10.0, 1.0, 0.5, 3))
  assert_eq(stald_detect(stald, 3), 1)
}

///|
/// The violation branch: `rr` lands just above `rrcut = 0.98`, so the status is
/// `kflag + 3 = 4` and `stepsize_controller!` cuts the order.
///
/// `λ₁ = -50, λ₂ = -0.02, h = 0.5, q = 5` — the slow mode decays by only ~1% per
/// step, which is exactly the "barely decaying" signature of a step size sitting
/// near the stability boundary.
test "stald flags a stability violation" {
  let stald = StaldState::new()
  stald_load(stald, twomode_rows(-50.0, -0.02, 1.0, 1.0, 0.5, 5))
  assert_eq(stald_detect(stald, 5), 4)
}

///|
/// A *single* geometric sequence is the degenerate case: Phase 3's difference
/// sequence cancels exactly, `cest1` comes out negative, and the detector bails
/// at -7 rather than guessing. That conservatism is the pinned behavior — the
/// detector refuses to act on data it cannot cross-check.
test "stald refuses a single geometric sequence" {
  let stald = StaldState::new()
  // ss[i] = 0.25^-i, i.e. a per-step ratio of exactly 0.25 at all three levels.
  let g = 0.25
  let rows : Array[Double] = []
  let mut i = 0
  while i < 5 {
    let mut k = 0
    while k < 3 {
      rows.push(fastpower(g, -Double::from_int(i)))
      k += 1
    }
    i += 1
  }
  stald_load(stald, rows)
  assert_eq(stald_detect(stald, 5), -7)
}

///|
/// Orders below 3 never reach the kernel: they are A-stable, and upstream's
/// early return also pins `last_order` so a later order-3 step starts a fresh
/// window.
test "stald ignores A-stable orders" {
  let stald = StaldState::new()
  assert_false(stald_check(stald, 2))
  assert_eq(stald.last_order, 2)
  assert_eq(stald.nscon, 0)
}

///|
/// The window needs `order + 5` consecutive samples at one order, so the first
/// `order + 5 - 1` checks are silent no matter what the data say. This is
/// CVODE's guard against reading noise out of a freshly started window.
///
/// Note the first call does not count: `last_order` starts at 0, so that call
/// takes the `order != last_order` branch and only re-arms the counter. Ten
/// more are then needed for order 5.
test "stald waits for order+5 consecutive steps" {
  let stald = StaldState::new()
  // A window that *would* report a violation, so the silence is attributable
  // to the counter rather than to the data.
  stald_load(stald, twomode_rows(-50.0, -0.02, 1.0, 1.0, 0.5, 5))
  let mut n = 0
  while n < 10 {
    assert_false(stald_check(stald, 5))
    n += 1
  }
  assert_eq(stald.nscon, 9)
  assert_true(stald_check(stald, 5))
  assert_eq(stald.nscon, 0)
  assert_eq(stald.nor, 1)
}

///|
/// An order change invalidates the window: `nscon` resets and the check goes
/// silent again, because the three derivative levels are now different
/// polynomials and the old ratios describe nothing.
test "stald resets its window on an order change" {
  let stald = StaldState::new()
  stald_load(stald, twomode_rows(-50.0, -0.02, 1.0, 1.0, 0.5, 5))
  // Eight checks at order 5: one to arm the counter, seven increments — not
  // yet the ten order 5 needs, so the window is still silent.
  let mut n = 0
  while n < 8 {
    ignore(stald_check(stald, 5))
    n += 1
  }
  assert_eq(stald.nscon, 7)
  // Switching to order 4 re-arms the counter and starts over.
  assert_false(stald_check(stald, 4))
  assert_eq(stald.last_order, 4)
  assert_eq(stald.nscon, 0)
}

///|
/// `stald_reset!` drops the whole window, which is what
/// `stepsize_controller!`'s filter path and a derivative discontinuity both do:
/// the history has been rewritten, so the old ratios are meaningless.
test "stald_reset clears the window" {
  let stald = StaldState::new()
  stald_load(stald, twomode_rows(-50.0, -0.02, 1.0, 1.0, 0.5, 5))
  stald.nscon = 4
  stald.nor = 2
  stald_reset(stald)
  assert_eq(stald.nscon, 0)
  assert_eq(stald.nor, 2)
  assert_eq(stald.last_order, 0)
  let mut i = 0
  while i < 15 {
    assert_eq(stald.ssdat.unsafe_get(i), 0.0)
    i += 1
  }
}

///|
/// `stald_collect_data!` shifts the window down and writes the newest sample
/// into row 0, and it is a no-op below order 3 — the A-stable orders must not
/// poison the window.
test "stald_collect_data shifts and gates on order" {
  let stald = StaldState::new()
  stald_load(stald, twomode_rows(-50.0, -0.02, 1.0, 1.0, 0.5, 5))
  let before = stald.ssdat.unsafe_get(0)
  // Order 2: no-op.
  stald_collect_data(stald, 2, 1.0, 2.0, 3.0)
  assert_eq(stald.ssdat.unsafe_get(0), before)
  // Order 5: row 0 becomes the new sample, everything else moves down.
  stald_collect_data(stald, 5, 1.0, 2.0, 3.0)
  assert_eq(stald.ssdat.unsafe_get(0 * 3 + 0), 1.0)
  assert_eq(stald.ssdat.unsafe_get(0 * 3 + 1), 4.0)
  assert_eq(stald.ssdat.unsafe_get(0 * 3 + 2), 9.0)
  assert_eq(stald.ssdat.unsafe_get(1 * 3 + 0), before)
}
