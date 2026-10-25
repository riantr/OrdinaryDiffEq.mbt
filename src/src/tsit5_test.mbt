// End-to-end tests for Tsit5: fixed-step accuracy, convergence order, dense
// output, saveat interpolation and backward integration.
//
// Test files see the package through its interface, so every reference is
// qualified with `@ordinarydiffeq.` and the algorithm comes from `new`.

///|
/// `u' = -u` on [0, 1] with u(0) = 1: the canonical accuracy problem, whose
/// exact solution is exp(-t). (Named apart from solve_test.mbt's helper —
/// the package shares one namespace across files.)
fn tsit5_decay_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
fn tsit5() -> @ordinarydiffeq.Tsit5 {
  @ordinarydiffeq.Tsit5::new()
}

///|
/// Solve the decay problem with fixed step `dt` and return the endpoint error
/// against exp(-1).
fn endpoint_error(dt : Double) -> Double {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    dt,
  )
  let sol = @ordinarydiffeq.solve(tsit5_decay_problem(), tsit5(), opts)
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  (last - @math.exp(-1.0)).abs()
}

///|
test "tsit5 solves u' = -u to high accuracy with fixed steps" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.05,
  )
  let sol = @ordinarydiffeq.solve(tsit5_decay_problem(), tsit5(), opts)
  assert_true(sol.retcode.is_success())
  // t0 plus 20 steps of 0.05; the last one is shortened to land exactly on tf.
  assert_eq(sol.t.length(), 21)
  assert_true((sol.t.unsafe_get(20) - 1.0).abs() < 0.000000000000001)
  // Milestone acceptance: error against exp(-1) below 1e-6. The method is
  // 5th order; the observed error is ~6.4e-12, so assert the tighter 1e-9.
  assert_true(
    (sol.u.unsafe_get(20).unsafe_get(0) - @math.exp(-1.0)).abs() < 0.000000001,
  )
  // FSAL accounting: one pre-start evaluation plus six per step.
  assert_eq(sol.nf, 121)
}

///|
test "tsit5 converges at fifth order" {
  let e1 = endpoint_error(0.2)
  let e2 = endpoint_error(0.1)
  let e3 = endpoint_error(0.05)
  let e4 = endpoint_error(0.025)
  // Halving dt cuts the error by 2^5 = 32 asymptotically. At these step
  // sizes the runs are still pre-asymptotic (the z^7 term of the local
  // error pushes the ratio above 32) and the smallest run approaches the
  // rounding floor, so accept a wide per-halving window and check the
  // aggregate slope over three halvings instead.
  let r1 = e1 / e2
  let r2 = e2 / e3
  let r3 = e3 / e4
  assert_true(r1 > 20.0 && r1 < 70.0)
  assert_true(r2 > 20.0 && r2 < 70.0)
  assert_true(r3 > 20.0 && r3 < 70.0)
  // log2(e(0.2)/e(0.025)) / 3 is the observed order: ~5.3 here, 5 in the
  // limit — accept anything strictly between 4th and 6th order.
  let slope = @math.ln(e1 / e4) / (@math.ln(2.0) * 3.0)
  assert_true(slope > 4.5 && slope < 6.0)
}

///|
test "tsit5 dense output interpolates accurately inside steps" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.05,
  )
  let sol = @ordinarydiffeq.solve(tsit5_decay_problem(), tsit5(), opts)
  // One 7-stage group per saved time point.
  assert_true(sol.k.length() == sol.t.length())
  assert_eq(sol.k.unsafe_get(1).length(), 7)
  // Endpoints are reproduced: theta = 1 evaluates to the step's own solution.
  let at_half = @ordinarydiffeq.Solution::call(sol, 0.5).unsafe_get(0)
  assert_true(
    (at_half - sol.u.unsafe_get(10).unsafe_get(0)).abs() < 0.000000000001,
  )
  // Mid-step: the free interpolant is 4th order, so the error is O(dt^5) —
  // observed ~5.5e-12 at dt = 0.05, asserted below 1e-9.
  let mid = @ordinarydiffeq.Solution::call(sol, 0.55).unsafe_get(0)
  assert_true((mid - @math.exp(-0.55)).abs() < 0.000000001)
  // The derivative at an endpoint is the FSAL stage k_7 = f(u, t).
  let d = @ordinarydiffeq.Solution::derivative(sol, 0.5).unsafe_get(0)
  assert_true((d + sol.u.unsafe_get(10).unsafe_get(0)).abs() < 0.000000001)
  // The derivative at the initial point is f(t0, u0).
  let d0 = @ordinarydiffeq.Solution::derivative(sol, 0.0).unsafe_get(0)
  assert_true((d0 + 1.0).abs() < 0.000000001)
}

///|
test "tsit5 uses its polynomial at saveat points" {
  // A non-empty saveat turns dense storage off (post-solve interpolation is
  // linear), but mid-step saveat points are still evaluated with the
  // solver's 4th-order polynomial, like Julia's `interp_at_saveat`.
  let opts = @ordinarydiffeq.SolveOptions::with_saveat(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.05,
    ),
    [0.13, 0.5, 0.77, 1.0],
  )
  let sol = @ordinarydiffeq.solve(tsit5_decay_problem(), tsit5(), opts)
  assert_eq(sol.t.length(), 4)
  assert_eq(sol.t.unsafe_get(0), 0.13)
  assert_eq(sol.t.unsafe_get(1), 0.5)
  assert_eq(sol.t.unsafe_get(2), 0.77)
  assert_eq(sol.t.unsafe_get(3), 1.0)
  // 0.13 and 0.77 fall inside steps: polynomial interpolation, ~1e-11 error.
  assert_true(
    (sol.u.unsafe_get(0).unsafe_get(0) - @math.exp(-0.13)).abs() < 0.000000001,
  )
  assert_true(
    (sol.u.unsafe_get(2).unsafe_get(0) - @math.exp(-0.77)).abs() < 0.000000001,
  )
  // 0.5 is a step endpoint: the solver's own state.
  assert_true(
    (sol.u.unsafe_get(1).unsafe_get(0) - @math.exp(-0.5)).abs() < 0.000001,
  )
  // No k stored, so post-solve calls interpolate linearly.
  assert_true(sol.k.length() == 0)
}

///|
test "tsit5 integrates backward" {
  // u' = -u from t = 1 down to t = 0 with u(1) = 1: u(0) = e.
  let prob = @ordinarydiffeq.OdeProblem::iip(1.0, 0.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.05,
  )
  let sol = @ordinarydiffeq.solve(prob, tsit5(), opts)
  assert_true(sol.retcode.is_success())
  assert_eq(sol.t.length(), 21)
  assert_true((sol.t.unsafe_get(0) - 1.0).abs() < 0.000000000000001)
  assert_true((sol.t.unsafe_get(20) - 0.0).abs() < 0.000000000000001)
  let last = sol.u.unsafe_get(20).unsafe_get(0)
  assert_true((last - @math.exp(1.0)).abs() < 0.000001)
}
