// Shared machinery for the multistep BDF family (FBDF, and later QNDF/Nordsieck).
//
// Ported from OrdinaryDiffEqBDF/src/bdf_utils.jl:
//   * `_make_bdf_coeffs_fbdf` — the fixed BDF coefficients.
//   * `calc_finite_difference_weights!` — the Fornberg weights that turn a
//     solution history into `h^k y^(k)` estimates.
//   * `reinitFBDF!` — the history-queue maintenance for a variable-step,
//     variable-order method.
//   * `backdiff!` / `_filt_*` — the divided-difference filters of DeCaria et
//     al. (arXiv:1810.06670). FBDF's `time_filter` is off by default
//     (algorithms.jl:556), so the filter kernels below are built but not yet
//     reached; they are kept next to the rest of the divided-difference
//     machinery because they share `backdiff!`.
//   * `_CHEB_NODES` / `_BARY_WEIGHTS` — the fixed Chebyshev reference nodes
//     and barycentric weights the dense output is resampled onto.

// ---------------------------------------------------------------------------
// Fixed BDF coefficients
// ---------------------------------------------------------------------------

///|
/// `bdf_coeffs[k, i]`, the fixed coefficients of the `k`-th order BDF for
/// equally spaced history points, as rationals reduced to `Double`.
///
/// Row `k` is `[α₀, α₁, …, α_k]` with
///
///     (1/dt) Σ_i αᵢ y(t_i) = f(t_{k+1})
///
/// over `t_i = t + (1-i) dt`, so `α₀` (the coefficient of the new point) is
/// the leading coefficient and `α₀(k+1)` is the local-error constant.
///
/// These are the *DFBDF* coefficients of `_make_bdf_coeffs_fbdf`
/// (bdf_utils.jl:76) — note that FBDF and DFBDF share one table in upstream
/// (`FBDFCache` reads `_make_bdf_coeffs_fbdf` at bdf_caches.jl:662), and the
/// `k = 1` row is `1, -1`, i.e. implicit Euler, not the `3/2, -2, 1/2` of the
/// classical fixed-coefficient BDF2. The difference is the `1/α₀` scaling
/// `β₀ = inv(bdf_coeffs[k, 1])` applies, which makes the *corrector*
/// `nlsolver.tmp = (Σ αᵢ u_i) / dt` come out right for either table.
///
/// Stored row-major, `6 * 6`, with `bdf_coeffs_row(k)` returning row `k` for
/// `k = 1..5`; row 0 is unused padding.
pub const BDF_COEFFS_ROWS : Int32 = 6

///|
pub const BDF_COEFFS_COLS : Int32 = 6

///|
/// Row `k` (1-based order) of the BDF coefficient table, flattened to a
/// 6-entry array so callers can index without a second indirection.
pub fn bdf_coeffs_row(k : Int32) -> Array[Double] {
  [
    // k = 0: unused padding.
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // k = 1: 1 -1
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    // k = 2: 3/2 -2 1/2
    [1.5, -2.0, 0.5, 0.0, 0.0, 0.0],
    // k = 3: 11/6 -3 3/2 -1/3
    [1.8333333333333333, -3.0, 1.5, -0.3333333333333333, 0.0, 0.0],
    // k = 4: 25/12 -4 3 -4/3 1/4
    [2.0833333333333335, -4.0, 3.0, -1.3333333333333333, 0.25, 0.0],
    // k = 5: 137/60 -5 5 -10/3 5/4 -1/5
    [2.283333333333333, -5.0, 5.0, -3.3333333333333335, 1.25, -0.2],
  ].unsafe_get(k)
}

///|
/// The leading coefficient `α₀ = bdf_coeffs[k, 1]`. The Newton solve's
/// `β₀ = 1/α₀` is what makes the corrector solve for the new value directly,
/// and the CVODE step-size formula (controllers.jl:445) reads the same entry.
pub fn bdf_leading_coeff(k : Int32) -> Double {
  bdf_coeffs_row(k).unsafe_get(0)
}

// ---------------------------------------------------------------------------
// Finite-difference weights (Fornberg's algorithm)
// ---------------------------------------------------------------------------

///|
/// `calc_finite_difference_weights!` (bdf_utils.jl:161): fill the leading
/// `(order+1) x (order+1)` block of `c` with the weights that approximate the
/// `order`-th derivative at `t` from the values at `ts[0..order]`.
///
/// On return `c[i*stride + order]` is the weight of `ts[i]`, so
///
///     y^(order)(t) ≈ Σ_i c[i*stride + order] * y(ts[i])
///
/// which is what `estimate_terk` needs: it forms `|dt^(k-1)| * Σ c[i,k] * y_i`,
/// an estimate of `h^(k-1) y^(k-1)` on which the order-selection chain
/// `terkm2 > terkm1 > terk > terkp1` compares.
///
/// The recurrence is the standard divided-difference one (Fornberg 1988,
/// "Generation of finite difference formulas on arbitrarily spaced grids").
/// It is written here as the direct transcription of the upstream loop so the
/// two agree operation for operation — the tabulation is only used with
/// `n <= 6` points, so clarity beats speed.
///
/// `stride` is the row pitch of `c` and must be at least `order + 1`; the
/// block outside it is left untouched (upstream's `fill!` covers the whole
/// matrix, but no reader ever looks past the block). The tabulation is only
/// used with `n <= 6` points, so clarity beats speed.
pub fn calc_finite_difference_weights(
  c : Array[Double],
  ts : Array[Double],
  t : Double,
  order : Int32,
  stride : Int32,
) -> Unit {
  let n = order + 1
  let mut i = 0
  while i < n {
    let mut j = 0
    while j < n {
      c.unsafe_set(i * stride + j, 0.0)
      j += 1
    }
    i += 1
  }
  let mut c1 = 1.0
  let mut c4 = ts.unsafe_get(0) - t
  c.unsafe_set(0 * stride + 0, 1.0)
  let mut i2 = 1
  while i2 < n {
    let mut c2 = 1.0
    let c5 = c4
    c4 = ts.unsafe_get(i2) - t
    let mut j = 0
    while j < i2 {
      let c3 = ts.unsafe_get(i2) - ts.unsafe_get(j)
      c2 = c2 * c3
      if j == i2 - 1 {
        // `for k in i:-1:2` with 1-based columns; column `k` is 0-based `kk`,
        // and the Julia multiplier `(k - 1)` is therefore `kk` itself.
        let mut kk = i2
        while kk >= 1 {
          let prev_k = c.unsafe_get((i2 - 1) * stride + (kk - 1))
          let prev_k1 = c.unsafe_get((i2 - 1) * stride + kk)
          c.unsafe_set(
            i2 * stride + kk,
            c1 * (Double::from_int(kk) * prev_k - c5 * prev_k1) / c2,
          )
          kk -= 1
        }
        c.unsafe_set(i2 * stride + 0, 0.0)
      }
      let mut k2 = i2
      while k2 >= 1 {
        let cur = c.unsafe_get(j * stride + k2)
        let cur_m1 = c.unsafe_get(j * stride + (k2 - 1))
        c.unsafe_set(
          j * stride + k2,
          (c4 * cur - Double::from_int(k2) * cur_m1) / c3,
        )
        k2 -= 1
      }
      c.unsafe_set(j * stride + 0, 0.0)
      j += 1
    }
    c1 = c2
    i2 += 1
  }
}

// ---------------------------------------------------------------------------
// Chebyshev reference nodes and barycentric weights
// ---------------------------------------------------------------------------

///|
/// Chebyshev nodes of the first kind on `[0, 1]`: `(1 + cos(π(2j-1)/(2n)))/2`
/// for `j = 1..n` (`_CHEB_NODES`, bdf_utils.jl:280).
///
/// Flattened: `cheb_nodes(n)` returns the `n` nodes for `n = 1..6`. Upstream's
/// `ntuple` runs `j = 1..n`, and `cos` is decreasing on `[0, π]`, so its rows
/// come out **descending** (j = 1 gives the node closest to 1). This table is
/// stored **ascending** — the same set, reversed — because the port indexes the
/// dense-output stages `k[0..n-1]` from the oldest end. The two orders describe
/// identical node sets, so every consumer that pairs node `i` with stage `i` is
/// unaffected; only `bary_weights` has to be reversed to match (see there).
///
/// Values are the correctly-rounded doubles of the exact expressions, matching
/// Julia's `cospi`-based constants to the last bit.
pub fn cheb_nodes(n : Int32) -> Array[Double] {
  [
    [0.5],
    [0.14644660940672624, 0.8535533905932737],
    [0.06698729810778067, 0.5, 0.9330127018922193],
    [
      0.038060233744356624, 0.30865828381745514, 0.6913417161825449, 0.9619397662556434,
    ],
    [
      0.024471741852423214, 0.20610737385376343, 0.5, 0.7938926261462366, 0.9755282581475768,
    ],
    [
      0.017037086855465858, 0.14644660940672624, 0.37059047744873963, 0.6294095225512604,
      0.8535533905932737, 0.9829629131445341,
    ],
  ].unsafe_get(n - 1)
}

///|
/// Barycentric weights for the same nodes: `(-1)^(j-1) sin(π(2j-1)/(2n))`
/// (`_BARY_WEIGHTS`, bdf_utils.jl:284). These need no normalization because
/// the barycentric formula divides by their sum.
///
/// **The order is the port's ascending node order, not upstream's.** Upstream
/// pairs weight `w_j` with its *descending* node `j`; here node `i` (0-based,
/// ascending) is upstream's node `j = n - i`, so the weight that belongs with
/// it is
///
///     w_i = (-1)^(n-i-1) · sin(π(2(n-i)-1)/(2n)) = (-1)^(n-i-1) · sin(π(2i+1)/(2n))
///
/// The pairing is what matters: the barycentric quotient is invariant under a
/// common rescaling of the weights but *not* under reordering them, and a
/// table that keeps upstream's signs while the nodes are reversed silently
/// attaches each weight to the wrong node (the interpolant then reproduces
/// nothing). Values are the correctly-rounded doubles of the exact sines.
pub fn bary_weights(n : Int32) -> Array[Double] {
  [
    [1.0],
    [-0.7071067811865476, 0.7071067811865476],
    [0.5, -1.0, 0.5],
    [
      -0.3826834323650898, 0.9238795325112867, -0.9238795325112867, 0.3826834323650898,
    ],
    [
      0.30901699437494745, -0.8090169943749475, 1.0, -0.8090169943749475, 0.30901699437494745,
    ],
    [
      -0.25881904510252074, 0.7071067811865476, -0.9659258262890683, 0.9659258262890683,
      -0.7071067811865476, 0.25881904510252074,
    ],
  ].unsafe_get(n - 1)
}

// ---------------------------------------------------------------------------
// Lagrange interpolation on the reference nodes
// ---------------------------------------------------------------------------

///|
/// `L_j(xi)` for the `j`-th of `n` Chebyshev nodes (`_lagrange_basis_scalar`,
/// bdf_utils.jl:291).
pub fn lagrange_basis_scalar(
  xi : Double,
  j : Int32,
  thetas : Array[Double],
  n : Int32,
) -> Double {
  let theta_j = thetas.unsafe_get(j)
  let mut l = 1.0
  let mut m = 0
  while m < n {
    if m != j {
      l = l * (xi - thetas.unsafe_get(m)) / (theta_j - thetas.unsafe_get(m))
    }
    m += 1
  }
  l
}

///|
/// Evaluate the Lagrange interpolant through `(thetas[j], values[j])` at
/// scalar `xi` (`_eval_lagrange_oop`, bdf_utils.jl:303). Used for the
/// predictor, whose nodes are the *actual* (variable-step) history times, not
/// the fixed reference nodes.
pub fn eval_lagrange(
  out : Array[Double],
  xi : Double,
  thetas : Array[Double],
  values : Array[Array[Double]],
  n : Int32,
) -> Unit {
  let dim = out.length()
  let l1 = lagrange_basis_scalar(xi, 0, thetas, n)
  let mut i = 0
  while i < dim {
    out.unsafe_set(i, l1 * values.unsafe_get(0).unsafe_get(i))
    i += 1
  }
  let mut j = 1
  while j < n {
    let lj = lagrange_basis_scalar(xi, j, thetas, n)
    i = 0
    while i < dim {
      out.unsafe_set(
        i,
        out.unsafe_get(i) + lj * values.unsafe_get(j).unsafe_get(i),
      )
      i += 1
    }
    j += 1
  }
}

///|
/// `L'_i(Θ)` on the fixed nodes (`_fbdf_lagrange_basis_deriv`,
/// bdf_interpolants.jl:314): the product-rule form, one factor omitted per
/// outer term.
pub fn lagrange_basis_deriv(theta : Double, i : Int32, n : Int32) -> Double {
  let nodes = cheb_nodes(n)
  let ni = nodes.unsafe_get(i)
  let mut result = 0.0
  let mut m = 0
  while m < n {
    if m != i {
      let nm = nodes.unsafe_get(m)
      let mut term = 1.0 / (ni - nm)
      let mut l = 0
      while l < n {
        if l != i && l != m {
          term = term *
            (theta - nodes.unsafe_get(l)) /
            (ni - nodes.unsafe_get(l))
        }
        l += 1
      }
      result = result + term
    }
    m += 1
  }
  result
}

// ---------------------------------------------------------------------------
// Divided-difference table (backdiff!) — used by the FBDF filters
// ---------------------------------------------------------------------------

///|
/// `backdiff!` (bdf_utils.jl:443): tabulate divided differences over the
/// ascending points `ts[0..n-1]` into `c` and the `n x n` workspace `d`.
///
/// On return `c[q*n + i]` is the coefficient of `ts[i]` in the `q`-th divided
/// difference over the newest `q+1` points:
///
///     δ^q y = Σ_i c[q*n + i] * y(ts[i]),    q = 0..n-1
///
/// Only the leading block of `c` is touched, and both may be larger than
/// `n x n` upstream; here they are sized exactly.
pub fn backdiff(
  c : Array[Double],
  d : Array[Double],
  ts : Array[Double],
  n : Int32,
) -> Unit {
  let mut j = 0
  while j < n {
    let mut i = 0
    while i < n {
      // Julia `i == n + 1 - j` (1-based) becomes `i == n - 1 - j`.
      let v = if i == n - 1 - j { 1.0 } else { 0.0 }
      d.unsafe_set(j * n + i, v)
      i += 1
    }
    j += 1
  }
  let mut i2 = 0
  while i2 < n {
    // Julia `c[1, i]` is 0-based row 0.
    c.unsafe_set(0 * n + i2, d.unsafe_get(0 * n + i2))
    i2 += 1
  }
  let mut q = 1
  while q < n {
    let mut j2 = 0
    while j2 < n - q {
      let denom = ts.unsafe_get(n - j2) - ts.unsafe_get(n - j2 - q)
      let mut i3 = 0
      while i3 < n {
        d.unsafe_set(
          j2 * n + i3,
          (d.unsafe_get(j2 * n + i3) - d.unsafe_get((j2 + 1) * n + i3)) / denom,
        )
        i3 += 1
      }
      j2 += 1
    }
    let mut i4 = 0
    while i4 < n {
      // Julia `c[q + 1, i]` is 0-based row `q`.
      c.unsafe_set(q * n + i4, d.unsafe_get(0 * n + i4))
      i4 += 1
    }
    q += 1
  }
}

// ---------------------------------------------------------------------------
// Filter helpers (DeCaria et al., arXiv:1810.06670)
// ---------------------------------------------------------------------------

///|
/// `ts_asc[0..n-2] = ts[n-2], …, ts[0]` and `ts_asc[n-1] = tdt`
/// (`_filt_fill_ts_asc!`, bdf_utils.jl:479): the cache history (newest-first)
/// turned ascending, with the point being solved for appended.
fn filt_fill_ts_asc(
  ts_asc : Array[Double],
  ts : Array[Double],
  tdt : Double,
  n : Int32,
) -> Unit {
  let mut i = 0
  while i < n - 1 {
    ts_asc.unsafe_set(i, ts.unsafe_get(n - 1 - i))
    i += 1
  }
  ts_asc.unsafe_set(n - 1, tdt)
}

///|
/// `δ = Σ_i c[row*n + i] * y_i` where `y` is the value at the newest point
/// and the older ones come from the newest-first history (`_filt_divided_diff`,
/// bdf_utils.jl:489).
fn filt_divided_diff(
  out : Array[Double],
  c : Array[Double],
  row : Int32,
  n : Int32,
  y : Array[Double],
  u_history : Array[Array[Double]],
) -> Unit {
  let dim = y.length()
  let cn = c.unsafe_get(row * n + (n - 1))
  let mut i = 0
  while i < dim {
    out.unsafe_set(i, cn * y.unsafe_get(i))
    i += 1
  }
  let mut j = 0
  while j < n - 1 {
    let cj = c.unsafe_get(row * n + j)
    let hist = u_history.unsafe_get(n - 1 - j)
    i = 0
    while i < dim {
      out.unsafe_set(i, out.unsafe_get(i) + cj * hist.unsafe_get(i))
      i += 1
    }
    j += 1
  }
}

///|
/// `bdf3stab_coeff(c, n) = µ / c[4*n + n-1]` (bdf_utils.jl:475) with
/// `µ = 9/125`, the G-stability-optimal constant of DeCaria et al. Weight of
/// the BDF3-Stab filter step `y² = y³ + w·δ³y³`.
fn bdf3stab_coeff(c : Array[Double], n : Int32) -> Double {
  let mu = 9.0 / 125.0
  mu / c.unsafe_get(3 * n + (n - 1))
}

///|
/// `_fbdf_filter_weight` (bdf_utils.jl:540): the weight that cancels the
/// fixed-coefficient solve's error on a monic polynomial of degree `k+1`.
///
/// The history interpolant of such a polynomial is `θ^(k+1) - ∏ⱼ(θ-θⱼ)` with
/// `θ = (t - tdt)/dt`, and the defect is the amount of that polynomial the
/// BDF formula fails to annihilate.
fn fbdf_filter_weight(
  c : Array[Double],
  ts : Array[Double],
  n : Int32,
  k : Int32,
  dt : Double,
) -> Double {
  let coeffs = bdf_coeffs_row(k)
  let mut defect = 0.0
  let mut i = 1
  while i <= k {
    let theta = -Double::from_int(i)
    let mut product = 1.0
    let mut j = 0
    while j < n - 1 {
      product = product *
        (theta - (ts.unsafe_get(j) - ts.unsafe_get(n - 1)) / dt)
      j += 1
    }
    let pk = fastpower(theta, Double::from_int(k + 1))
    defect = defect - coeffs.unsafe_get(i) * (pk - product)
    i += 1
  }
  let error = defect *
    fastpower(dt, Double::from_int(k + 1)) /
    coeffs.unsafe_get(0)
  error / (1.0 + c.unsafe_get((n - 1) * n + (n - 1)) * error)
}

///|
/// `_filt_bdf_coefficients!` (bdf_utils.jl:554): the `(k+1)`-th order BDF
/// coefficients on the *actual* (variable) history nodes, for the residual
/// test that decides whether the filtered value is admissible.
fn filt_bdf_coefficients(
  alpha_bar : Array[Double],
  c : Array[Double],
  ts : Array[Double],
  n : Int32,
  p : Int32,
) -> Unit {
  let mut i = 0
  while i < n {
    alpha_bar.unsafe_set(i, 0.0)
    let mut prefactor = 1.0
    let mut j = 1
    while j <= p {
      alpha_bar.unsafe_set(
        i,
        alpha_bar.unsafe_get(i) + prefactor * c.unsafe_get((j + 1 - 1) * n + i),
      )
      // Julia `ts[n] - ts[n - j]` (1-based) is 0-based `ts[n-1] - ts[n-1-j]`.
      prefactor = prefactor * (ts.unsafe_get(n - 1) - ts.unsafe_get(n - 1 - j))
      j += 1
    }
    i += 1
  }
}

///|
/// `_filt_bdf_residual` (bdf_utils.jl:514): `Σ ᾱ_i y_i - f(y)` on the actual
/// history nodes — the BDF residual whose scaled norm gates the filter.
fn filt_bdf_residual(
  out : Array[Double],
  alpha_bar : Array[Double],
  n : Int32,
  y : Array[Double],
  u_history : Array[Array[Double]],
  fy : Array[Double],
) -> Unit {
  let dim = y.length()
  let an = alpha_bar.unsafe_get(n - 1)
  let mut i = 0
  while i < dim {
    out.unsafe_set(i, an * y.unsafe_get(i))
    i += 1
  }
  let mut j = 0
  while j < n - 1 {
    let aj = alpha_bar.unsafe_get(j)
    let hist = u_history.unsafe_get(n - 1 - j)
    i = 0
    while i < dim {
      out.unsafe_set(i, out.unsafe_get(i) + aj * hist.unsafe_get(i))
      i += 1
    }
    j += 1
  }
  i = 0
  while i < dim {
    out.unsafe_set(i, out.unsafe_get(i) - fy.unsafe_get(i))
    i += 1
  }
}

///|
/// `_filt_step_ratio` (bdf_utils.jl:566): the step-size ratio a filter
/// candidate at order `p` would produce, used to compare candidates against
/// the unfiltered step. `Inf` for a zero estimate (exact step → unbounded
/// growth), `0` for a non-positive or non-finite one (never chosen).
fn filt_step_ratio(est : Double, p : Int32) -> Double {
  if est == 0.0 {
    return 1.0 / 0.0
  }
  if !(est > 0.0) || !is_finite(est) {
    return 0.0
  }
  fastpower(est, -1.0 / Double::from_int(p + 1))
}

// ---------------------------------------------------------------------------
// History maintenance
// ---------------------------------------------------------------------------

///|
/// `reinitFBDF!` (bdf_utils.jl:196): maintain the newest-first history queue
/// `ts` / `u_history` and the order bookkeeping for a variable-step,
/// variable-order method.
///
/// Three cases, in upstream's order:
///
///   * `iters_from_event == 0` — the first step after a (re)start: the queue
///     holds one point.
///   * `iters_from_event == 1 && t != ts[1]` — the second: shift the one point
///     down so the queue has two.
///   * `consfailcnt == 0` — an ordinary successful step: shift everything
///     from index `order + 1` down by one, so the newest `order + 1` points
///     are the live history.
///
/// The `consfailcnt > 0` case is the interesting one: a rejected step must
/// *not* corrupt the history, so the shift is skipped entirely and the
/// previous state is preserved. That is what makes a rejected BDF step
/// retryable — the history it would have overwritten is still the history of
/// the last accepted step.
///
/// Upstream's fourth branch, the `derivative_discontinuity` reset that zeroes
/// the queue and restarts at order 1, has no field of its own here. It is
/// reached two ways upstream: `initialize!` brackets its forced
/// `reinitFBDF!` with the flag set — which `FBDFCache::ensure` performs
/// equivalently, once, before the first step — and
/// `bdf_step_reject_controller!` arms it after repeated order-1 failures, for
/// which this port calls [`bdf_restart`] directly.
pub fn reinit_fbdf(integ : Integrator, cache : FBDFCache) -> Unit {
  let ts = cache.ts
  let hist = cache.u_history
  if cache.iters_from_event == 0 {
    ts.unsafe_set(0, integ.t)
    copy_into(hist.unsafe_get(0), integ.uprev)
  } else if cache.iters_from_event == 1 && integ.t != ts.unsafe_get(0) {
    ts.unsafe_set(1, ts.unsafe_get(0))
    copy_into(hist.unsafe_get(1), hist.unsafe_get(0))
    ts.unsafe_set(0, integ.t)
    copy_into(hist.unsafe_get(0), integ.uprev)
  } else if cache.consfailcnt == 0 {
    let mut i = cache.order + 1
    while i >= 1 {
      ts.unsafe_set(i, ts.unsafe_get(i - 1))
      copy_into(hist.unsafe_get(i), hist.unsafe_get(i - 1))
      i -= 1
    }
    ts.unsafe_set(0, integ.t)
    copy_into(hist.unsafe_get(0), integ.uprev)
  }
}

///|
/// The number of history slots FBDF keeps: `max_order + 2`, so the corrector
/// can look back `k` points at order `k` and still have room for the
/// predictor's `k + 1` nodes.
pub const FBDF_HISTORY_SLOTS : Int32 = 7

///|
/// `bdf_restart_estimates!` (controllers.jl:181) plus the
/// `derivative_discontinuity` branch of `reinitFBDF!` (bdf_utils.jl:206):
/// drop the whole history and start over at order 1.
///
/// Upstream's comment says why: "the BDF step-size and order logic is based on
/// the history of the solution, and we have effectively entered a new regime
/// where old estimates no longer apply". `bdf_step_reject_controller!` arms it
/// after `consfailcnt > 3` consecutive failures at order 1 — the method has
/// been unable to take *any* step with even the lowest order, so the history
/// it is extrapolating from is not describing the solution any more.
///
/// The field set is upstream's, exactly: `order`, `consfailcnt`, `nconsteps`,
/// `qwait`, `iters_from_event` and the three history buffers. Notably absent
/// are `prev_order`, the `terk` family and `filter_order` — upstream's
/// `derivative_discontinuity` branch does not touch them, and `prev_order`'s
/// stale value is harmless because the restart's `qwait = 3` coincides with the
/// `order + 2` that `step_accept_controller!` would re-arm it to anyway.
///
/// The next `reinit_fbdf` call then takes its `iters_from_event == 0` branch
/// and rebuilds a one-point history at the current `(t, uprev)`, which is the
/// last accepted state. Upstream zeroes the buffers inside `reinitFBDF!` at
/// the top of the next step; this port does it here, in the controller, which
/// is unobservable because nothing reads the history in between.
pub fn bdf_restart(cache : FBDFCache) -> Unit {
  cache.order = 1
  cache.consfailcnt = 0
  cache.nconsteps = 0
  cache.qwait = 3
  cache.iters_from_event = 0
  cache.restart_pending = true
  let mut i = 0
  while i < cache.ts.length() {
    cache.ts.unsafe_set(i, 0.0)
    i += 1
  }
  for h in cache.u_history {
    i = 0
    while i < h.length() {
      h.unsafe_set(i, 0.0)
      i += 1
    }
  }
  for h in cache.u_corrector {
    i = 0
    while i < h.length() {
      h.unsafe_set(i, 0.0)
      i += 1
    }
  }
  stald_reset(cache.stald)
}

///|
/// The default `max_order` (`FBDF(; max_order = Val{5}())`, algorithms.jl:543).
pub const FBDF_DEFAULT_MAX_ORDER : Int32 = 5

// ---------------------------------------------------------------------------
// The FBDF cache
// ---------------------------------------------------------------------------

///|
/// Everything a FBDF solve carries between steps — Julia's `FBDFConstantCache`
/// plus the `NLSolver`/`NLNewtonCache` pair the corrector is driven through.
///
/// The split matters for reading the port: upstream keeps the *method* state
/// (`ts`, `u_history`, `order`, the `terk` family) on the cache and the
/// *nonlinear-solver* state (`z`, `tmp`, `ztmp`, `γ`, `α`, `κ`, `ηold`,
/// `prev_θ`, `new_W`, `W_γdt`, `J_t`) on a separate `NLSolver` object that
/// points at an `NLNewtonCache`. They are merged here because this port has no
/// generic nonlinear-solver layer to plug into, and the merge is a pure
/// relocation — every field keeps the name and the meaning it has upstream.
///
/// Allocation is lazy (`FBDFCache::ensure`) so explicit-only solves never pay
/// for the `dim`-vectors, mirroring `StiffCache::ensure`.
pub(all) struct FBDFCache {
  /// History times, **newest first**: `ts[0]` is the most recent accepted
  /// point. `reinit_fbdf` maintains the queue.
  pub mut ts : Array[Double]
  /// History states, same order as `ts`.
  pub mut u_history : Array[Array[Double]]
  /// Lagrange interpolants of the history at the equidistant past points
  /// `-1, …, -(k-1)`, filled once per step and consumed by the corrector.
  pub mut u_corrector : Array[Array[Double]]
  /// Working order. Starts at 1 and is moved only by the controller.
  pub mut order : Int32
  /// The order the previous step used; `step_accept_controller!` compares the
  /// two to decide whether `qwait` must be re-armed.
  pub mut prev_order : Int32
  /// `max_order`, the `Val` parameter of `FBDFConstantCache{max_order}`.
  pub mut max_order : Int32
  /// Consecutive *rejected* steps. Non-zero freezes the history shift in
  /// `reinit_fbdf`, which is what makes a rejected step retryable.
  pub mut consfailcnt : Int32
  /// Consecutive accepted steps (Julia's comment: "Successful Consecutive Step
  /// with the same step size").
  pub mut nconsteps : Int32
  /// Steps since the last (re)start. Gates the predictor: at 0 there is no
  /// history to interpolate, so `u₀ = uprev`.
  pub mut iters_from_event : Int32
  /// CVODE's order-change countdown. An order change arms it to `order + 2`;
  /// it then counts down once per accepted step, and `terkp1` (the estimate an
  /// order increase would be judged on) is only computed once it reaches 0.
  pub mut qwait : Int32
  /// Tolerance-scaled norms of `dt^{k-2} y^{(k-2)}`, `dt^{k-1} y^{(k-1)}`,
  /// `dt^k y^k` — the order-selection ladder `terkm2 > terkm1 > terk > terkp1`.
  pub mut terkm2 : Double
  pub mut terkm1 : Double
  pub mut terk : Double
  pub mut terkp1 : Double
  /// Order a filter step selected. `time_filter` is off by default, so this
  /// stays 0 and the filter branches in the controllers stay unreachable.
  pub mut filter_order : Int32
  /// Set by [`bdf_restart`]: this port's stand-in for upstream's
  /// `integrator.derivative_discontinuity` flag. `reinit_fbdf` and
  /// `fbdf_do_new_jw` read it during the step that follows a restart, and
  /// `fbdf_perform_step` clears it at the end — the same window in which
  /// upstream's `loopfooter_reset!` (integrator_utils.jl:725) clears the flag.
  pub mut restart_pending : Bool
  /// The flat BDF coefficient table, `6 * 6` row-major, row `k` for order `k`.
  /// A copy of `bdf_coeffs_row`'s literal so the hot path does not rebuild it.
  pub mut bdf_coeffs : Array[Double]
  /// `r[j]`, the LTE weights of the fixed-coefficient formula (index 2..k).
  pub mut r : Array[Double]
  /// `ts_tmp[0] = tdt`, `ts_tmp[j] = ts[j-1]` — the divided-difference nodes
  /// `estimate_terk` and `choose_order!` differentiate over.
  pub mut ts_tmp : Array[Double]
  /// Fornberg weights, `(max_order + 1)^2` row-major.
  pub mut fd_weights : Array[Double]
  /// `(ts[j] - t) / dt` for the `k + 1` history nodes, the predictor's and
  /// corrector's node set.
  pub mut pred_thetas : Array[Double]
  /// `u₀`, the predictor: the Lagrange interpolant of the history at `Θ = 1`,
  /// or the current state while no history exists (upstream's `cache.u₀`,
  /// bdf_caches.jl:753). It seeds the Newton iterate and is the reference
  /// `terkp1` measures the corrector's displacement against.
  pub mut u0 : Array[Double]
  /// `terkp1_tmp`: the raw `(u - u₀) · ∏ⱼ j·dt/(tdt - ts[j])` product whose
  /// scaled norm is `terkp1` (upstream's `cache.terkp1_tmp`). Kept separate
  /// from `terk_tmp` because the corrector's Newton solve runs between the two.
  pub mut terkp1_tmp : Array[Double]
  /// `dim`-vector scratch for the `terk` estimates.
  pub mut terk_tmp : Array[Double]
  // --- nonlinear solver state (upstream's NLSolver + NLNewtonCache) ---
  /// The iterate. For `COEFFICIENT_MULTISTEP` it *is* the solution value, not
  /// an increment — `compute_ustep` returns `z` unchanged.
  pub mut z : Array[Double]
  /// `nlsolver.tmp`, the corrector's history term divided by `dt`.
  pub mut tmp : Array[Double]
  /// The residual buffer; also holds the next iterate until `apply_step!`
  /// copies it back into `z` (upstream's `nlsolver.ztmp`).
  pub mut ztmp : Array[Double]
  /// The next iterate, `z - dz`, computed at the end of `compute_step!`
  /// (upstream writes it into `ztmp` too; a second buffer keeps the residual
  /// readable while `apply_step!` is pending).
  pub mut ztmp2 : Array[Double]
  /// The Newton increment `dz`.
  pub mut dz : Array[Double]
  /// The tolerance-scaled residual whose norm is `ndz`.
  pub mut atmp : Array[Double]
  /// `f(z, tdt)`, upstream's `NLNewtonCache.k`.
  pub mut nlk : Array[Double]
  /// `β₀ = 1/α₀`, the `nlsolver.γ` slot. `γ·dt = β₀·dt` is the `dtgamma` `W`
  /// is assembled at.
  pub mut beta0 : Double
  /// `α₀ = 1` for FBDF (`nlsolver.α`).
  pub mut alpha : Double
  /// `NLNewton.κ`, the convergence threshold on `η·ndz`.
  pub mut kappa : Double
  /// `NLNewton.max_iter`, default 10.
  pub mut max_iter : Int32
  /// `NLNewton.new_W_dt_cutoff`, default 0.2: the relative change in
  /// `1/(γ·dt)` tolerated before `W` is refactored.
  pub mut new_w_dt_cutoff : Double
  /// Remembered convergence-rate estimate (`nlsolver.ηold`), reused when `W`
  /// was not rebuilt.
  pub mut etaold : Double
  /// `nlsolver.prev_θ`, carried across steps. Non-trivial because
  /// `NLSolver` has the field, so `has_prev_θ` is true.
  pub mut prev_theta : Double
  /// `nlsolver.iter`, the iteration counter of the last solve.
  pub mut nliter : Int32
  /// `NLStatus`, encoded as `1` = Convergence, `-2` = Divergence,
  /// `-3` = TryAgain. `nlsolvefail` is `status <= 0`.
  pub mut nlstatus : Int32
  /// `nlsolver.nfails`.
  pub mut nfails : Int32
  /// `NLNewtonCache.new_W`: whether `W` was rebuilt on the last solve, which
  /// is what makes the linear solver refactorize.
  pub mut new_w : Bool
  /// `NLNewtonCache.W_γdt`, the `γ·dt` the current `W` was built at.
  pub mut w_gdt : Double
  /// `NLNewtonCache.J_t`, the `t` `J` was evaluated at; `isJcurrent` compares
  /// it against the integrator's `t`.
  pub mut j_t : Double
  /// `NLNewtonCache.firstcall`, cleared by the first solve's postamble.
  pub mut firstcall : Bool
  /// `NLNewtonCache.firststage`, set by `markfirststage!` at the top of every
  /// `perform_step!` and cleared by the postamble.
  pub mut firststage : Bool
  /// The dense Jacobian and the factorization of `W = J - I/(β₀·dt)`.
  pub mut stiff : StiffCache
  /// CVODE stability-limit detection state.
  pub mut stald : StaldState
  /// Buffers are allocated and sized for `dim`.
  pub mut ready : Bool
  pub mut dim : Int32
}

///|
/// An empty cache. `FBDFCache::ensure` allocates it on first use.
pub fn FBDFCache::new() -> FBDFCache {
  {
    ts: [],
    u_history: [],
    u_corrector: [],
    order: 1,
    prev_order: 1,
    max_order: FBDF_DEFAULT_MAX_ORDER,
    consfailcnt: 0,
    nconsteps: 0,
    iters_from_event: 0,
    qwait: 3,
    // `alg_cache` seeds the order-selection ladder at 1
    // (bdf_caches.jl:677-680): an unset estimate reads as "no evidence", which
    // is above zero so the first `iszero(terk)` test in the controller takes
    // the CVODE formula rather than the unbounded-growth branch.
    terkm2: 1.0,
    terkm1: 1.0,
    terk: 1.0,
    terkp1: 1.0,
    filter_order: 0,
    restart_pending: false,
    bdf_coeffs: [],
    r: [],
    ts_tmp: [],
    fd_weights: [],
    pred_thetas: [],
    u0: [],
    terkp1_tmp: [],
    terk_tmp: [],
    z: [],
    tmp: [],
    ztmp: [],
    ztmp2: [],
    dz: [],
    atmp: [],
    nlk: [],
    beta0: 1.0,
    alpha: 1.0,
    kappa: 0.01,
    max_iter: 10,
    new_w_dt_cutoff: 0.2,
    etaold: 1.0,
    prev_theta: 1.0,
    nliter: 0,
    nlstatus: -2,
    nfails: 0,
    new_w: false,
    w_gdt: 0.0,
    j_t: 0.0,
    firstcall: true,
    firststage: false,
    stiff: StiffCache::new(),
    stald: StaldState::new(),
    ready: false,
    dim: 0,
  }
}

///|
/// Allocate (or resize) the cache for a problem of `dim` components.
///
/// Returns true when the buffers were (re)allocated, i.e. the Jacobian and the
/// history are no longer valid. Upstream's cache constructor does the same
/// sizing once per solve from `size(u)`.
pub fn FBDFCache::ensure(self : FBDFCache, dim : Int32) -> Bool {
  if self.ready && self.dim == dim {
    return false
  }
  let slots = FBDF_HISTORY_SLOTS
  let mo = self.max_order
  self.ts = Array::make(slots, 0.0)
  self.u_history = Array::make(slots, Array::make(0, 0.0))
  let mut i = 0
  while i < slots {
    self.u_history.unsafe_set(i, Array::make(dim, 0.0))
    i += 1
  }
  self.u_corrector = Array::make(mo, Array::make(0, 0.0))
  i = 0
  while i < mo {
    self.u_corrector.unsafe_set(i, Array::make(dim, 0.0))
    i += 1
  }
  // The BDF coefficient table, copied out of the literal rows once.
  self.bdf_coeffs = Array::make(6 * 6, 0.0)
  let mut k = 1
  while k <= 5 {
    let row = bdf_coeffs_row(k)
    let mut j = 0
    while j < 6 {
      self.bdf_coeffs.unsafe_set(k * 6 + j, row.unsafe_get(j))
      j += 1
    }
    k += 1
  }
  self.r = Array::make(mo + 2, 0.0)
  self.ts_tmp = Array::make(mo + 2, 0.0)
  self.fd_weights = Array::make((mo + 1) * (mo + 1), 0.0)
  self.pred_thetas = Array::make(mo + 1, 0.0)
  self.u0 = Array::make(dim, 0.0)
  self.terkp1_tmp = Array::make(dim, 0.0)
  self.terk_tmp = Array::make(dim, 0.0)
  self.z = Array::make(dim, 0.0)
  self.tmp = Array::make(dim, 0.0)
  self.ztmp = Array::make(dim, 0.0)
  self.ztmp2 = Array::make(dim, 0.0)
  self.dz = Array::make(dim, 0.0)
  self.atmp = Array::make(dim, 0.0)
  self.nlk = Array::make(dim, 0.0)
  ignore(StiffCache::ensure(self.stiff, dim))
  // A fresh solve starts at order 1 with a one-point history, exactly as
  // `initialize!`'s forced `reinitFBDF!` leaves it.
  self.order = 1
  self.prev_order = 1
  self.consfailcnt = 0
  self.nconsteps = 0
  self.iters_from_event = 0
  self.qwait = 3
  self.terkm2 = 1.0
  self.terkm1 = 1.0
  self.terk = 1.0
  self.terkp1 = 1.0
  self.filter_order = 0
  self.restart_pending = false
  self.beta0 = 1.0
  self.alpha = 1.0
  self.etaold = 1.0
  self.prev_theta = 1.0
  self.nliter = 0
  self.nlstatus = -2
  self.nfails = 0
  self.new_w = false
  self.w_gdt = 0.0
  self.j_t = 0.0
  self.firstcall = true
  self.firststage = false
  stald_reset(self.stald)
  self.ready = true
  self.dim = dim
  true
}

///|
/// `bdf_coeffs[k, i]` from the cache's copy of the table (0-based both).
pub fn FBDFCache::coeff(self : FBDFCache, k : Int32, i : Int32) -> Double {
  self.bdf_coeffs.unsafe_get(k * 6 + i)
}
