// Adaptive step-size control: the PI controller and the automatic initial
// step. This is the M2 piece the stepping loop was wired for since M1a.
//
// Ported from OrdinaryDiffEqCore:
//   * integrators/controllers.jl — `PIController` (the default controller for
//     every OrdinaryDiffEq algorithm: `default_controller(QT, alg)`), with
//     `stepsize_controller!` / `step_accept_controller!` /
//     `step_reject_controller!` / `accept_step_controller`.
//   * initdt.jl — `_ode_initdt_iip`, the Hairer-Nørsett-Wanner initial step
//     heuristic, reached through `handle_dt!` -> `auto_dt_reset!` ->
//     `_determine_initdt` (solve.jl:1108, integrator_interface.jl:679).
//   * integrator_utils.jl — `calc_dt_propose!` (1215).
//
// Reduced to what a scalar, pure ODE needs: no composite algorithms (so no
// `get_current_adaptive_order` indirection — the order is the algorithm's
// classical order), no discontinuity detection, no Newton failures, no
// `dtnew_modification`, no `isoutofdomain`/`step_limiter!` hooks (both
// default to no-ops upstream), no unitful time (every `oneunit(t)` factor is
// 1 and drops out).

///|
/// The proportional-integral step-size controller, holding both the resolved
/// knobs and the per-solve mutable state (Julia splits these between
/// `PIController` and `PIControllerCache`; this port keeps one struct).
///
/// The update is `dt_{n+1} = EEst_{n+1}^beta1 * EEst_n^beta2 * dt_n`, with the
/// factor clipped to `[1/qmax, 1/qmin]` after dividing by the safety factor
/// `gamma`. Defaults for an algorithm of classical order `p` (alg_utils.jl:
/// 798, 774):
///
///     beta2 = 2 / (5p)      beta1 = 7 / (10p)
///     qmin = 1/5            qmax = 10
///     qmax_first_step = 1e4 (looser cap until the first step is accepted)
///     gamma = 9/10          qsteady_min = qsteady_max = 1 (deadband)
///     qoldinit = 1e-4       (floor on the remembered error)
pub struct PIController {
  /// Cached `EEst^beta1` from the last `stepsize_controller` call; the reject
  /// path needs it after the accept path computed it.
  pub mut q11 : Double
  /// The previous step's (floored) error estimate.
  pub mut errold : Double
  /// Proportional gain.
  pub beta1 : Double
  /// Integral gain.
  pub beta2 : Double
  /// Floor applied to `errold` when remembering an error.
  pub qoldinit : Double
  /// Lower bound on the per-step shrink factor.
  pub qmin : Double
  /// Upper bound on the per-step growth factor.
  pub qmax : Double
  /// Looser `qmax` used until the first step is accepted
  /// (`get_current_qmax`, controllers.jl:288).
  pub qmax_first_step : Double
  /// Safety factor applied to the controller's predicted change.
  pub gamma : Double
  /// Deadband edges: a proposed factor inside `[qsteady_min, qsteady_max]`
  /// holds `dt` constant.
  pub qsteady_min : Double
  pub qsteady_max : Double
}

///|
/// Build the default PI controller for an algorithm of classical order
/// `order` (Julia's `PIController(QT, alg)` route: `default_controller`
/// alg_utils.jl:740, beta defaults alg_utils.jl:774/798, knob defaults
/// alg_utils.jl:417-889).
pub fn PIController::for_order(order : Int32) -> PIController {
  let p = Double::from_int(order)
  {
    q11: 1.0,
    errold: 0.0001,
    beta1: 7.0 / (10.0 * p),
    beta2: 2.0 / (5.0 * p),
    qoldinit: 0.0001,
    qmin: 0.2,
    qmax: 10.0,
    qmax_first_step: 10000.0,
    gamma: 0.9,
    qsteady_min: 1.0,
    qsteady_max: 1.0,
  }
}

///|
/// `stepsize_controller!` (controllers.jl:805): compute the step-size factor
/// `q` for the step that just finished.
///
/// A zero error estimate means "exact step": grow by the full `qmax` (and
/// leave `q11` untouched, matching upstream). Otherwise
/// `q = EEst^beta1 / errold^beta2`, divided by `gamma` and clipped into
/// `[1/qmax_eff, 1/qmin]`, where `qmax_eff` is `qmax_first_step` until the
/// first step has been accepted.
pub fn stepsize_controller(integ : Integrator) -> Double {
  let ctrl = integ.controller
  let qmax_eff = if integ.success_iter == 0 {
    ctrl.qmax_first_step
  } else {
    ctrl.qmax
  }
  if integ.est == 0.0 {
    1.0 / qmax_eff
  } else {
    let q11 = fastpower(integ.est, ctrl.beta1)
    let q = q11 / fastpower(ctrl.errold, ctrl.beta2)
    ctrl.q11 = q11
    clamp(q / ctrl.gamma, 1.0 / qmax_eff, 1.0 / ctrl.qmin)
  }
}

///|
/// `step_accept_controller!` (controllers.jl:823): turn an accepted step's
/// factor `q` into the next step size, applying the steady-state deadband
/// and remembering the error (floored at `qoldinit`) for the next PI update.
/// Returns the proposed step length (unsigned; the caller re-signs it).
pub fn step_accept_controller(integ : Integrator, q : Double) -> Double {
  let ctrl = integ.controller
  let mut qq = q
  if ctrl.qsteady_min <= qq && qq <= ctrl.qsteady_max {
    qq = 1.0
  }
  ctrl.errold = dmax(integ.est, ctrl.qoldinit)
  integ.dt / qq
}

///|
/// `step_reject_controller!` (controllers.jl:838): shrink `dt` after a
/// rejected step, by `min(1/qmin, q11/gamma)`.
pub fn step_reject_controller(integ : Integrator) -> Unit {
  let ctrl = integ.controller
  integ.dt = integ.dt / dmin(1.0 / ctrl.qmin, ctrl.q11 / ctrl.gamma)
}

///|
/// `accept_step_controller` (controllers.jl:245): accept when the scaled
/// error estimate is at most 1. `isoutofdomain` and `force_dtmin` are false
/// in this port, so the upstream `||` fallbacks never fire.
pub fn accept_step_controller(integ : Integrator) -> Bool {
  integ.est <= 1.0
}

///|
/// `calc_dt_propose!` (integrator_utils.jl:1215): store the controller's
/// proposal as the next `dtpropose`, after the floating-point clock fix
/// (`dtnew = (t + dtnew) - t`) and the `dtmax` / `timedepentdtmin` clamps.
pub fn calc_dt_propose(integ : Integrator, dtnew : Double) -> Unit {
  let signed = integ.tdir * dtnew.abs()
  // Round the step onto the floating-point grid anchored at `t`, so the
  // state update and the clock agree (upstream comment at 1220).
  let rounded = integ.t + signed - integ.t
  let dtpropose = integ.tdir * dmin(integ.opts.dtmax.abs(), rounded.abs())
  integ.dtpropose = integ.tdir * dmax(dtpropose.abs(), Integrator::dtmin(integ))
}

// ---------------------------------------------------------------------------
// Automatic initial step (initdt.jl `_ode_initdt_iip`)
// ---------------------------------------------------------------------------

///|
/// `ode_determine_initdt` for the in-place ODE path: the Hairer-Nørsett-Wanner
/// heuristic. Two derivative evaluations beyond the pre-start FSAL one, so a
/// solve that auto-selects `dt` starts with `nf == 3` (Julia:
/// `auto_dt_reset!` adds 2, integrator_interface.jl:695).
///
/// The mass-matrix, unitful-time and diffusion (`g`) branches of the original
/// do not exist here. Scratch buffers: the scale `sk` uses `integ.atmp`,
/// `f0` uses `integ.tmp`, the trial state `u1` uses `integ.utilde` and its
/// derivative `f1` uses `integ.fsallast` — all four are free before the first
/// step. The scaled norms are computed inline (no quotient buffer), which is
/// arithmetically identical to Julia's `internalnorm(u0 ./ sk, t)`.
pub fn determine_initdt(integ : Integrator, order : Int32) -> Double {
  let tdir = integ.tdir
  let t = integ.t
  let u0 = integ.u
  let n = u0.length()
  // dtmin is time-dependent upstream (`max(opts.dtmin, eps*|t|)`), and the
  // fallback default is `max(dtmin, 1e-6)`.
  let dtmin = Integrator::dtmin(integ)
  let smalldt = dmax(dtmin, 0.000001)
  if n == 0 {
    return tdir * smalldt
  }
  // The step-size cap: `dtmax` shortened to the next tstop (Julia's
  // `_determine_initdt`, integrator_interface.jl:680).
  let dtmax = if Integrator::has_tstop(integ) {
    let dist = (Integrator::first_tstop(integ) - tdir * t).abs()
    dmin(integ.opts.dtmax.abs(), dist)
  } else {
    integ.opts.dtmax.abs()
  }
  let sk = integ.atmp
  let mut i = 0
  while i < n {
    sk.unsafe_set(
      i,
      integ.opts.abstol + u0.unsafe_get(i).abs() * integ.opts.reltol,
    )
    i += 1
  }
  // f0 = f(t0, u0). Upstream reuses the FSAL slot and re-evaluates; this port
  // runs the heuristic before the pre-start FSAL, so the count matches (2
  // evaluations here + 1 there == Julia's 1 + 2).
  let f0 = integ.tmp
  Integrator::eval_f(integ, t, u0, f0)
  let d0 = rms_norm_scaled(u0, sk)
  let d1 = rms_norm_scaled(f0, sk)
  if d1 != d1 {
    // NaN in the first derivative: upstream bails with `tdir * dtmin`.
    return tdir * dtmin
  }
  let dt0 = if d0 < 0.00001 || d1 < 0.00001 { smalldt } else { d0 / d1 / 100.0 }
  let dt0 = dmin(dt0, dtmax)
  let tiny_first = dt0 < 10.0 * eps
  let dt0_tdir = tdir * dt0
  // Trial state u1 = u0 + dt0 * f0 and its derivative f1 = f(t + dt0, u1).
  let u1 = integ.utilde
  i = 0
  while i < n {
    u1.unsafe_set(i, u0.unsafe_get(i) + dt0_tdir * f0.unsafe_get(i))
    i += 1
  }
  let f1 = integ.fsallast
  Integrator::eval_f(integ, t + dt0_tdir, u1, f1)
  if vectors_equal(f0, f1) {
    // Constant vector field near u0: the first guess is already fine.
    return fallback_if_tiny(
      tdir * dmax(dtmin, 100.0 * dt0),
      tiny_first,
      tdir,
      smalldt,
      dtmin,
    )
  }
  // d2 = ||(f1 - f0) / sk|| / dt0.
  let d2 = rms_norm_scaled_diff(f1, f0, sk) / dt0
  let maxd1d2 = dmax(d1, d2)
  let dt1 = if maxd1d2 <= 0.000000000000001 {
    dmax(0.000001, dt0 * 0.001)
  } else {
    fastpower(10.0, -(2.0 + log10(maxd1d2)) / Double::from_int(order))
  }
  fallback_if_tiny(
    tdir * dmax(dtmin, dmin(dmin(100.0 * dt0, dt1), dtmax)),
    tiny_first,
    tdir,
    smalldt,
    dtmin,
  )
}

///|
/// `_fallback_if_tiny` (initdt.jl:19): a first guess that came out below
/// machine epsilon is only trusted if the refined step is not tiny too.
fn fallback_if_tiny(
  dt : Double,
  tiny_first : Bool,
  tdir : Double,
  smalldt : Double,
  dtmin : Double,
) -> Double {
  if tiny_first && (!is_finite(dt) || dt.abs() < 10.0 * eps) {
    tdir * dmax(smalldt, dtmin)
  } else {
    dt
  }
}

///|
/// RMS norm of the element-wise quotient `x_i / s_i` — Julia's
/// `internalnorm(x ./ s, t)` without materializing the quotient.
fn rms_norm_scaled(x : Array[Double], s : Array[Double]) -> Double {
  let n = x.length()
  if n == 0 {
    return 0.0
  }
  let mut acc = 0.0
  let mut i = 0
  while i < n {
    let q = x.unsafe_get(i) / s.unsafe_get(i)
    acc = acc + q * q
    i += 1
  }
  sqrt(acc / Double::from_int(n))
}

///|
/// RMS norm of the element-wise quotient `(a_i - b_i) / s_i`.
fn rms_norm_scaled_diff(
  a : Array[Double],
  b : Array[Double],
  s : Array[Double],
) -> Double {
  let n = a.length()
  if n == 0 {
    return 0.0
  }
  let mut acc = 0.0
  let mut i = 0
  while i < n {
    let q = (a.unsafe_get(i) - b.unsafe_get(i)) / s.unsafe_get(i)
    acc = acc + q * q
    i += 1
  }
  sqrt(acc / Double::from_int(n))
}

///|
/// Component-wise equality (Julia's `f₀ == f₁` on vectors, guarded by
/// `_bool_equal` for array wrappers — OrdinaryDiffEq.jl#1402).
fn vectors_equal(a : Array[Double], b : Array[Double]) -> Bool {
  if a.length() != b.length() {
    return false
  }
  let mut i = 0
  while i < a.length() {
    if a.unsafe_get(i) != b.unsafe_get(i) {
      return false
    }
    i += 1
  }
  true
}

///|
/// `clamp(x, lo, hi)` (Julia's `clamp`).
fn clamp(x : Double, lo : Double, hi : Double) -> Double {
  dmin(dmax(x, lo), hi)
}

///|
/// `log10(x)`. MoonBit's `@math` has `ln` but no `log10`, and the init
/// heuristic's `10^(-(2 + log10(m))/order)` is the one place base 10 appears.
fn log10(x : Double) -> Double {
  @math.ln(x) / @math.ln(10.0)
}

// ---------------------------------------------------------------------------
// Controller invariants
// ---------------------------------------------------------------------------

///|
/// The full PI update, driven by hand-set integrator state. This lives inside
/// the package (an internal test) because the MoonBit version exports struct
/// fields as read-only to other packages: a black-box test cannot write
/// `est` / `dt` / `success_iter`. Expected values were computed with an
/// independent IEEE-754 reference (tools/m2_expected.py), not from a run.
test "PI controller matches the Julia formulas" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_dtmax(SolveOptions::new(), 1.0)
  let integ = Integrator::new(prob, opts, 7, 5)
  let c = integ.controller
  assert_true(c.beta1 == 0.14)
  assert_true(c.beta2 == 0.08)
  assert_true(c.qmin == 0.2)
  assert_true(c.qmax == 10.0)
  assert_true(c.qmax_first_step == 10000.0)
  assert_true(c.gamma == 0.9)
  assert_true(c.qsteady_min == 1.0 && c.qsteady_max == 1.0)
  assert_true(c.qoldinit == 0.0001)
  assert_true(c.errold == 0.0001)
  assert_true(c.q11 == 1.0)

  // Case A: EEst = 4, dt = 0.1, errold = 1e-4, success_iter = 0, so the
  // growth cap is qmax_first_step. q11 = 4^0.14 = 1.214194884395047;
  // q = 1.214194884395047 / 1e-4^0.08 / 0.9 = 2.818680748965933 (the upper
  // bound 1/qmin = 5 is not reached).
  integ.est = 4.0
  integ.dt = 0.1
  integ.success_iter = 0
  let q = stepsize_controller(integ)
  assert_true((q - 2.818680748965933).abs() < 0.000000001)
  assert_true((integ.controller.q11 - 1.214194884395047).abs() < 0.000000001)
  // Reject: dt /= min(1/qmin, q11/gamma) = min(5, 1.3491054271056078).
  step_reject_controller(integ)
  assert_true((integ.dt - 0.07412319155408158).abs() < 0.000000000001)

  // Case B: EEst = 0.5, success_iter = 1 (cap qmax = 10).
  // q11 = 0.5^0.14 = 0.9075191553171609;
  // q = 0.9075191553171609 / 1e-4^0.08 / 0.9 = 2.106751399866745.
  let integ2 = Integrator::new(prob, opts, 7, 5)
  integ2.est = 0.5
  integ2.dt = 0.1
  integ2.success_iter = 1
  let q2 = stepsize_controller(integ2)
  assert_true((q2 - 2.106751399866745).abs() < 0.000000001)
  // Accept: the deadband [1, 1] only holds q == 1, so q stays and
  // dtnew = dt/q = 0.04746644526085287; errold is lifted to
  // max(EEst, qoldinit) = 0.5.
  let dtnew = step_accept_controller(integ2, q2)
  assert_true((dtnew - 0.04746644526085287).abs() < 0.000000001)
  assert_true(integ2.controller.errold == 0.5)
  // calc_dt_propose at t = 0: (0 + dtnew) - 0 is exact, dtmax = 1 does not
  // clip, and timedepentdtmin = max(0, eps*0) = 0.
  calc_dt_propose(integ2, dtnew)
  assert_true((integ2.dtpropose - 0.04746644526085287).abs() < 0.000000001)
  // Second PI update with the remembered error: EEst = 2, errold = 0.5 gives
  // q = 2^0.14 / 0.5^0.08 / 0.9 = 1.2941484294093952, hence
  // dtnew = 0.1 / 1.2941484294093952 = 0.07727088927939785.
  integ2.est = 2.0
  let q3 = stepsize_controller(integ2)
  assert_true((q3 - 1.2941484294093952).abs() < 0.000000001)
  let dtnew3 = step_accept_controller(integ2, q3)
  assert_true((dtnew3 - 0.07727088927939785).abs() < 0.000000001)

  // Case C: EEst = 0 is the "exact step" special case: grow by the full
  // qmax (1/10 here) and leave q11 untouched.
  let integ3 = Integrator::new(prob, opts, 7, 5)
  integ3.est = 0.0
  integ3.success_iter = 1
  let q4 = stepsize_controller(integ3)
  assert_true(q4 == 0.1)
  assert_true(integ3.controller.q11 == 1.0)

  // Case D: the floating-point clock anchor and the dtmax clip. t = 0.5 and
  // tdir = +1 come from the problem's own time span.
  let prob_mid = OdeProblem::iip(0.5, 1.5, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let integ4 = Integrator::new(prob_mid, opts, 7, 5)
  assert_true(integ4.t == 0.5 && integ4.tdir == 1.0)
  calc_dt_propose(integ4, 0.2)
  // (0.5 + 0.2) - 0.5 = 0.19999999999999996, not 0.2.
  assert_true(
    (integ4.dtpropose - 0.19999999999999996).abs() < 0.0000000000000001,
  )
  calc_dt_propose(integ4, 5.0)
  assert_true(integ4.dtpropose == 1.0)

  // Case E: backward integration keeps the sign (t = 1, tdir = -1).
  let prob_bwd = OdeProblem::iip(1.0, 0.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let integ5 = Integrator::new(prob_bwd, opts, 7, 5)
  assert_true(integ5.t == 1.0 && integ5.tdir == -1.0)
  calc_dt_propose(integ5, 0.2)
  assert_true(
    (integ5.dtpropose + 0.19999999999999996).abs() < 0.0000000000000001,
  )
}
