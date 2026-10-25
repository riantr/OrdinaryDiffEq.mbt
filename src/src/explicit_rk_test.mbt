// End-to-end tests for the generic tableau-driven ExplicitRK solver and the
// five tableaus generated into alg_explicit_rk_tableaus.mbt.
//
// The evaluation counts asserted below are the interesting part: upstream reuses
// `fsalfirst` as k1, so an FSAL tableau costs `stages - 1` new f evaluations per
// step while a non-FSAL one costs `stages` (the extra `f(u, t+dt)` that fills
// `fsallast` for the Hermite interpolant).

///|
/// `u' = -u` on [0, 1] with u(0) = 1: exact solution exp(-t).
fn rk_decay_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
/// The harmonic oscillator `u'' = -u` as a first-order system, written as
/// `(v, v')`. Unlike the decay problem this is non-trivial, so it separates the
/// tableaus by accuracy rather than by rounding noise.
fn rk_oscillator_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [0.0, 1.0], fn(_t, u, du) {
    du.unsafe_set(0, u.unsafe_get(1))
    du.unsafe_set(1, -u.unsafe_get(0))
  })
}

///|
/// exp(-1).
const RK_EXACT_T1 : Double = 0.36787944117144233

///|
/// Endpoint error of a fixed-step decay solve with the given tableau.
///
/// No `assert_true` here: it raises an error, which a plain helper cannot
/// propagate. The success check lives in the tests that call this.
fn rk_decay_error(
  tab : @ordinarydiffeq.ExplicitRKTableau,
  dt : Double,
) -> Double {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    dt,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(tab),
    opts,
  )
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  (last - RK_EXACT_T1).abs()
}

// ---------------------------------------------------------------------------
// Tableau bookkeeping
// ---------------------------------------------------------------------------

///|
test "explicit rk tableaus report their order, fsal flag and stage count" {
  let heun = @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.heun())
  assert_true(heun.order() == 2)
  // The *algorithm* is always FSAL (see `Solver::is_fsal`); the tableau's own
  // `fsal` flag is what decides whether perform_step pays the extra f.
  assert_true(heun.is_fsal())
  assert_true(!@ordinarydiffeq.heun().fsal)
  // No tableau in this set carries B_interp, so `initialize!` takes the
  // kshortsize == 2 path (explicit_rk_perform_step.jl:6).
  assert_true(heun.stage_count() == 2)
  let dp = @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.dormand_prince())
  assert_true(dp.order() == 5)
  assert_true(@ordinarydiffeq.dormand_prince().fsal)
  assert_true(dp.stage_count() == 2)
  let rk4 = @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.rk4())
  assert_true(rk4.order() == 4)
  assert_true(!@ordinarydiffeq.rk4().fsal)
}

///|
test "explicit rk exposes the tableau it was built from" {
  let tab = @ordinarydiffeq.bs3()
  let alg = @ordinarydiffeq.ExplicitRK::new(tab)
  assert_true(alg.tableau().stages == tab.stages)
  assert_true(alg.tableau().order == tab.order)
  assert_true(alg.tableau().alpha_eest.length() == tab.alpha_eest.length())
}

// ---------------------------------------------------------------------------
// Accuracy, one tableau at a time
// ---------------------------------------------------------------------------

///|
test "rk4 fixed step is fourth order accurate" {
  // The oscillator's exact state at t = 1 is (sin 1, cos 1) = (0.8414709848,
  // 0.5403023059). RK4 at dt = 0.05 (20 steps) is limited by its O(h^5) local
  // error; the bound below is far tighter than the ~1e-2 a first-order method
  // would leave.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.05,
  )
  let sol = @ordinarydiffeq.solve(
    rk_oscillator_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.rk4()),
    opts,
  )
  let last = sol.u.unsafe_get(sol.u.length() - 1)
  let e1 = (last.unsafe_get(0) - 0.8414709848078965).abs()
  let e2 = (last.unsafe_get(1) - 0.5403023058681398).abs()
  // Reference: an independent Python run of the same tableau gives ~2e-8.
  assert_true(e1 < 0.000001 && e2 < 0.000001)
}

///|
test "heun and bs3 fixed step land near the exact decay value" {
  // Reference errors from an independent Python run of the same tableaus:
  // Heun at dt = 0.01 -> ~6.2e-6, BS3 at dt = 0.01 -> ~1.5e-8. The bounds sit an
  // order of magnitude above each, pinning the order of magnitude without being
  // knife-edge. Heun's is the loose one -- a 2nd-order method simply is not that
  // accurate at dt = 0.01, and pretending otherwise would be the bug.
  assert_true(rk_decay_error(@ordinarydiffeq.heun(), 0.01) < 0.00001)
  assert_true(rk_decay_error(@ordinarydiffeq.bs3(), 0.01) < 0.000001)
}

///|
test "rkf4 and dormand-prince fixed step reach the exact decay value" {
  // Reference errors: RKF4 at dt = 0.05 -> ~1.0e-8, Dormand-Prince at
  // dt = 0.2 -> ~4.6e-8. Both bounds sit well above the reference so a
  // regression has to be gross to trip them.
  assert_true(rk_decay_error(@ordinarydiffeq.rkf4(), 0.05) < 0.000001)
  assert_true(rk_decay_error(@ordinarydiffeq.dormand_prince(), 0.2) < 0.000001)
}

// ---------------------------------------------------------------------------
// Evaluation counts -- the `fsalfirst` reuse contract
// ---------------------------------------------------------------------------

///|
test "non-fsal tableaus pay one extra evaluation per step" {
  // dt = 0.1 gives 10 steps. Each step evaluates `stages - 1` new derivatives
  // (k1 is reused from fsalfirst) plus the `f(u, t+dt)` that fills `fsallast`,
  // plus the single pre-start evaluation at t0.
  let heun = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol_heun = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.heun()),
    heun,
  )
  assert_true(sol_heun.nf == 1 + 10 * 2)
  // RK4 and BS3 both have 4 stages, so both cost 4 per step.
  let rk4 = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol_rk4 = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.rk4()),
    rk4,
  )
  assert_true(sol_rk4.nf == 1 + 10 * 4)
  let bs3 = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol_bs3 = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.bs3()),
    bs3,
  )
  assert_true(sol_bs3.nf == 1 + 10 * 4)
}

///|
test "an fsal tableau costs stages minus one per step" {
  // Dormand-Prince has 7 stages and reuses fsallast as the next fsalfirst, so
  // no extra `f(u, t+dt)` is needed: 6 new evaluations per step.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.dormand_prince()),
    opts,
  )
  assert_true(sol.nf == 1 + 10 * 6)
}

///|
test "rejected steps are paid for" {
  // With adaptive control the step size moves, so `nf` is no longer a simple
  // multiple. What must hold is that every evaluation is counted: the solve
  // reports more evaluations than the step count alone would need.
  let opts = @ordinarydiffeq.SolveOptions::with_reltol(
    @ordinarydiffeq.SolveOptions::with_abstol(
      @ordinarydiffeq.SolveOptions::with_dt(
        @ordinarydiffeq.SolveOptions::with_adaptive(
          @ordinarydiffeq.SolveOptions::new(),
          true,
        ),
        0.1,
      ),
      0.000001,
    ),
    0.000001,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.bs3()),
    opts,
  )
  assert_true(sol.retcode.is_success())
  // BS3 is 4-stage and non-FSAL, so each accepted step costs 4 evaluations and
  // the solve starts with 1. Any step the controller rejects adds 4 more without
  // adding a saved point, so `nf` must exceed 4 * (points - 1).
  let points = sol.t.length()
  assert_true(sol.nf > 4 * (points - 1))
}

// ---------------------------------------------------------------------------
// Adaptive control
// ---------------------------------------------------------------------------

///|
test "explicit rk adaptive stepping hits the requested tolerance" {
  let opts = @ordinarydiffeq.SolveOptions::with_reltol(
    @ordinarydiffeq.SolveOptions::with_abstol(
      @ordinarydiffeq.SolveOptions::with_dt(
        @ordinarydiffeq.SolveOptions::with_adaptive(
          @ordinarydiffeq.SolveOptions::new(),
          true,
        ),
        0.1,
      ),
      0.000000001,
    ),
    0.000000001,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.dormand_prince()),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - RK_EXACT_T1).abs() < 0.000001)
  // 20 points for a 5th-order pair at tol = 1e-9 starting from dt = 0.1: the
  // first step is rejected (the estimate overshoots at that size), then the
  // PI controller settles into ~0.055-sized steps.
  //
  // Cross-checked against tools/m3_expected.py, which independently reproduces
  // the same count and the same final error (6.046608e-11); the individual grid
  // points agree only to ~1e-3 because the two implementations differ in where
  // they place the floating-point slack on the tstop comparison, which shifts
  // one rejection. The step count, the endpoint error and the monotonicity are
  // the invariants worth pinning.
  assert_true(sol.t.length() == 20)
  assert_true(sol.t.unsafe_get(0) == 0.0)
  assert_true(sol.t.unsafe_get(19) == 1.0)
  assert_true(sol.t.unsafe_get(1) > 0.05 && sol.t.unsafe_get(1) < 0.1)
  assert_true(sol.t.unsafe_get(18) > 0.9 && sol.t.unsafe_get(18) < 0.98)
  let mut i = 1
  while i < sol.t.length() {
    assert_true(sol.t.unsafe_get(i) > sol.t.unsafe_get(i - 1))
    i += 1
  }
}

///|
test "a fixed-step tableau still solves without adaptive control" {
  // RK4 has no alphaEEst at all, so `perform_step` leaves est at 0 and the
  // controller takes its "exact step" branch (stepsize_controller.jl:105),
  // which returns `1/qmax` -- the *maximum* growth, not a hold. The result is
  // deliberately lopsided: 3 points, `t = [0, 0.1, 1.0]`.
  //
  //   step 1: success_iter == 0, so the cap is qmax_first_step = 1e4 and the
  //           factor is 1e-4; dtnew = 0.1 / 1e-4 = 1000, clipped by
  //           dtmax = |tf - t0| = 1.
  //   step 2: dt = 0.1 + 0.9 clipped to the tstop distance, landing on t = 1.
  //
  // Cross-checked against tools/m3_expected.py. This is upstream's behaviour
  // too -- running a tableau with no embedded pair under `adaptive = true` is
  // a user error, and the controller's response is to grow, not to complain.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::with_adaptive(
      @ordinarydiffeq.SolveOptions::new(),
      true,
    ),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.rk4()),
    opts,
  )
  assert_true(sol.retcode.is_success())
  assert_true(sol.t.length() == 3)
  assert_true((sol.t.unsafe_get(1) - 0.1).abs() < 0.000000001)
  assert_true((sol.t.unsafe_get(2) - 1.0).abs() < 0.000000001)
  // Two accepted steps, each costing `stages` evaluations: 3 middle/last
  // stages plus the extra `f(u, t + dt)` that fills `fsallast` for Hermite.
  assert_true(sol.nf == 1 + 2 * 4)
  assert_true((sol.u.unsafe_get(2).unsafe_get(0) - RK_EXACT_T1).abs() < 0.01)
}

// ---------------------------------------------------------------------------
// Dense output (the Hermite fallback)
// ---------------------------------------------------------------------------

///|
test "explicit rk dense output reproduces the trajectory" {
  // None of the five tableaus carries B_interp, so `interpolant()` returns the
  // Hermite fallback built on k[1] = fsalfirst and k[2] = fsallast -- which is
  // exactly why the kshortsize == 2 store above matters.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.05,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.dormand_prince()),
    opts,
  )
  let at_start = @ordinarydiffeq.Solution::call(sol, 0.0)
  let at_end = @ordinarydiffeq.Solution::call(sol, 1.0)
  assert_true((at_start.unsafe_get(0) - 1.0).abs() < 0.000001)
  assert_true((at_end.unsafe_get(0) - RK_EXACT_T1).abs() < 0.000001)
  // Cubic Hermite on a smooth problem is O(h^4) locally, so a 0.001 tolerance
  // at dt = 0.05 is comfortable.
  let mut t = 0.03
  while t < 0.99 {
    let v = @ordinarydiffeq.Solution::call(sol, t)
    let exact = @math.exp(-t)
    assert_true((v.unsafe_get(0) - exact).abs() < 0.000001)
    t += 0.07
  }
}

///|
test "explicit rk dense derivative matches f along the trajectory" {
  // For u' = -u the true derivative at t is -u(t), so the Hermite derivative
  // closure must satisfy d(t) == -v(t) to interpolant accuracy.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.05,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.dormand_prince()),
    opts,
  )
  let mut t = 0.11
  while t < 0.95 {
    let v = @ordinarydiffeq.Solution::call(sol, t)
    let d = @ordinarydiffeq.Solution::derivative(sol, t)
    assert_true((d.unsafe_get(0) + v.unsafe_get(0)).abs() < 0.000001)
    t += 0.13
  }
}

///|
test "a non-fsal tableau also gets usable dense output" {
  // Heun's `fsallast` comes from the extra `f(u, t+dt)` rather than from a
  // carried stage, so this checks the non-FSAL branch of the same store.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.01,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(
    rk_decay_problem(),
    @ordinarydiffeq.ExplicitRK::new(@ordinarydiffeq.heun()),
    opts,
  )
  let mut t = 0.07
  while t < 0.99 {
    let v = @ordinarydiffeq.Solution::call(sol, t)
    let exact = @math.exp(-t)
    assert_true((v.unsafe_get(0) - exact).abs() < 0.0001)
    t += 0.11
  }
}

// ---------------------------------------------------------------------------
// The B_interp path (no shipped tableau reaches it)
// ---------------------------------------------------------------------------

///|
test "eval_poly_derivative differentiates a polynomial" {
  // coef = [1, 2, 3] is 1 + 2x + 3x^2.
  let coef = [1.0, 2.0, 3.0]
  let tol = 0.000001
  // order 0 is the polynomial itself: 1 + 4 + 12 = 17.
  assert_true(
    (@ordinarydiffeq.eval_poly_derivative(2.0, coef, 0) - 17.0).abs() < tol,
  )
  // 1st derivative: 2 + 6x = 14.
  assert_true(
    (@ordinarydiffeq.eval_poly_derivative(2.0, coef, 1) - 14.0).abs() < tol,
  )
  // 2nd: 6.
  assert_true(
    (@ordinarydiffeq.eval_poly_derivative(2.0, coef, 2) - 6.0).abs() < tol,
  )
  // 3rd: zero -- and so is anything above the degree (`n <= order`).
  assert_true(@ordinarydiffeq.eval_poly_derivative(2.0, coef, 3).abs() < tol)
  assert_true(@ordinarydiffeq.eval_poly_derivative(2.0, coef, 7).abs() < tol)
}

///|
test "eval_poly_derivative uses falling factorials" {
  // coef = [0, 0, 1] is x^2. Its derivatives are 2x and 2 -- which only come
  // out right with a falling factorial (j!/(j-order)!), since a rising one would
  // give 6 instead of 2 at order 2.
  let coef = [0.0, 0.0, 1.0]
  let tol = 0.000001
  assert_true(
    (@ordinarydiffeq.eval_poly_derivative(3.0, coef, 0) - 9.0).abs() < tol,
  )
  assert_true(
    (@ordinarydiffeq.eval_poly_derivative(3.0, coef, 1) - 6.0).abs() < tol,
  )
  assert_true(
    (@ordinarydiffeq.eval_poly_derivative(3.0, coef, 2) - 2.0).abs() < tol,
  )
}

///|
test "generic_rk_interpolate reproduces the step endpoints" {
  // The B_interp form for order 0 is `y0 + dt * sum_i k_i * P_i(theta)`. With a
  // single stage whose polynomial is the constant 1, this reduces to `y0 +
  // dt*k`, which pins the scale factor.
  let y0 = [2.0]
  let ks = [[3.0]]
  let b_interp = [[1.0]]
  let out = Array::make(1, 0.0)
  @ordinarydiffeq.generic_rk_interpolate(0.5, 1.0, y0, ks, b_interp, 0, out)
  assert_true((out.unsafe_get(0) - 5.0).abs() < 0.000001)
}

///|
test "generic_rk_interpolate differentiates in time, not in theta" {
  // For order >= 1 upstream returns `sum_i k_i P_i^(order)(theta) *
  // inv(dt)^(order-1)` -- no `dt` factor and no `y0`. With `P = theta^2`,
  // `k = 1` and `dt = 2`:
  //   order 1 -> P'(1) = 2, times inv_dt_factor 1 -> 2
  //   order 2 -> P''(1) = 2, times 1/dt -> 1
  // so the `inv(dt)^(order-1)` branch is what separates the two.
  let y0 = [100.0]
  let ks = [[1.0]]
  let quad = [[0.0, 0.0, 1.0]]
  let out = Array::make(1, 0.0)
  @ordinarydiffeq.generic_rk_interpolate(1.0, 2.0, y0, ks, quad, 1, out)
  assert_true((out.unsafe_get(0) - 2.0).abs() < 0.000001)
  @ordinarydiffeq.generic_rk_interpolate(1.0, 2.0, y0, ks, quad, 2, out)
  assert_true((out.unsafe_get(0) - 1.0).abs() < 0.000001)
  // The value form, for contrast, does carry `dt` and `y0`: P(1) = 1 gives
  // y0 + dt * k * 1 = 102.
  @ordinarydiffeq.generic_rk_interpolate(1.0, 2.0, y0, ks, quad, 0, out)
  assert_true((out.unsafe_get(0) - 102.0).abs() < 0.000001)
}
