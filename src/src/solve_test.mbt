// End-to-end tests of the solve loop: tstops, saveat, backward integration,
// dense output and the failure paths. These exercise the machinery ported
// from OrdinaryDiffEqCore's solve.jl / integrator_utils.jl.
//
// Test files see the package through its interface, so every reference is
// qualified with `@ordinarydiffeq.` (this MoonBit version warns otherwise),
// and algorithm values come from their `new` constructors.

///|
/// `u' = -u` on [0, 1] with u(0) = 1: the canonical smoke problem.
fn decay_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
fn euler() -> @ordinarydiffeq.Euler {
  @ordinarydiffeq.Euler::new()
}

///|
test "fixed step hits every saved time exactly" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(decay_problem(), euler(), opts)
  assert_eq(sol.t.length(), 11)
  let mut i = 0
  while i < 11 {
    assert_true(
      (sol.t.unsafe_get(i) - i.to_double() * 0.1).abs() < 0.000000000000001,
    )
    i += 1
  }
  // Euler after 10 steps of 0.1 on u' = -u is exactly 0.9^10.
  let last = sol.u.unsafe_get(10).unsafe_get(0)
  assert_true((last - 0.3486784401).abs() < 0.0000001)
  // FSAL reuse: one derivative at t0 plus one per step.
  assert_eq(sol.nf, 11)
}

///|
test "tstops are hit exactly and dt is restored afterwards" {
  let opts = @ordinarydiffeq.SolveOptions::with_tstops(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.3,
    ),
    [0.5, 0.7],
  )
  let sol = @ordinarydiffeq.solve(decay_problem(), euler(), opts)
  // 0 -> 0.3 -> (clip) 0.5 -> (clip) 0.7 -> (clip) 1.0
  assert_eq(sol.t.length(), 5)
  assert_eq(sol.t.unsafe_get(0), 0.0)
  assert_eq(sol.t.unsafe_get(1), 0.3)
  assert_eq(sol.t.unsafe_get(2), 0.5)
  assert_eq(sol.t.unsafe_get(3), 0.7)
  assert_eq(sol.t.unsafe_get(4), 1.0)
}

///|
test "saveat switches to saveat-only output with interpolated values" {
  let opts = @ordinarydiffeq.SolveOptions::with_saveat(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    [0.25, 0.5, 0.75, 1.0],
  )
  let sol = @ordinarydiffeq.solve(decay_problem(), euler(), opts)
  assert_eq(sol.t.length(), 4)
  assert_eq(sol.t.unsafe_get(0), 0.25)
  assert_eq(sol.t.unsafe_get(1), 0.5)
  assert_eq(sol.t.unsafe_get(2), 0.75)
  assert_eq(sol.t.unsafe_get(3), 1.0)
  // 0.5 is a step endpoint: the value is the Euler state, exactly.
  let at_half = sol.u.unsafe_get(1).unsafe_get(0)
  assert_true(
    (at_half - @ordinarydiffeq.fastpower(0.9, 5.0)).abs() < 0.000000000000001,
  )
  // 0.25 falls inside the step [0.2, 0.3]: Euler's Hermite interpolant with
  // u(0.2)=0.81, u(0.3)=0.729, k1=-0.81, k2=-0.729, dt=0.1, theta=0.5 gives
  // 0.7684875 (both endpoints are themselves ~9e-3 off the exact solution,
  // so comparing against exp(-0.25) here would mostly measure Euler error).
  let at_quarter = sol.u.unsafe_get(0).unsafe_get(0)
  assert_true((at_quarter - 0.7684875).abs() < 0.000001)
  // Dense output is off in saveat mode, so post-solve calls interpolate
  // linearly between the saved points: 0.6 sits at theta = 0.4 of [0.5, 0.75].
  assert_true(sol.k.length() == 0)
  let between = @ordinarydiffeq.Solution::call(sol, 0.6).unsafe_get(0)
  let expect = 0.6 * at_half + 0.4 * sol.u.unsafe_get(2).unsafe_get(0)
  assert_true((between - expect).abs() < 0.000000000000001)
}

///|
test "backward integration walks t down to tf" {
  let prob = @ordinarydiffeq.OdeProblem::iip(1.0, 0.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(prob, euler(), opts)
  assert_eq(sol.t.length(), 11)
  assert_eq(sol.t.unsafe_get(0), 1.0)
  assert_eq(sol.t.unsafe_get(10), 0.0)
  // u' = -u integrated backwards from u(1)=1: Euler multiplies by 1.1 per step.
  let last = sol.u.unsafe_get(10).unsafe_get(0)
  assert_true((last - @ordinarydiffeq.fastpower(1.1, 10.0)).abs() < 0.0000001)
}

///|
test "dense output interpolates inside a step" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(decay_problem(), euler(), opts)
  assert_true(sol.k.length() == sol.t.length())
  // Endpoints are reproduced exactly.
  let at_half = @ordinarydiffeq.Solution::call(sol, 0.5).unsafe_get(0)
  assert_true(
    (at_half - @ordinarydiffeq.fastpower(0.9, 5.0)).abs() < 0.000000000000001,
  )
  // Mid-step values stay on the decaying trajectory within Euler's O(dt) error.
  let mid = @ordinarydiffeq.Solution::call(sol, 0.55).unsafe_get(0)
  assert_true(mid < at_half && mid > sol.u.unsafe_get(6).unsafe_get(0))
  assert_true((mid - @math.exp(-0.55)).abs() < 0.05)
  // The Hermite derivative at an endpoint is the stage derivative stored there.
  let d = @ordinarydiffeq.Solution::derivative(sol, 0.5).unsafe_get(0)
  assert_true(
    (d + @ordinarydiffeq.fastpower(0.9, 5.0)).abs() < 0.000000000000001,
  )
}

///|
test "maxiters stops the solve with MaxIters" {
  let opts = @ordinarydiffeq.SolveOptions::with_maxiters(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    5,
  )
  let sol = @ordinarydiffeq.solve(decay_problem(), euler(), opts)
  assert_true(sol.retcode.is_maxiters())
  // t0 plus the five accepted steps.
  assert_eq(sol.t.length(), 6)
  assert_true((sol.t.unsafe_get(5) - 0.5).abs() < 0.000000000000001)
}

///|
test "a blow-up problem reports Unstable" {
  // u' = u^2 blows up at t = 1 from u(0) = 1; stepping past it overflows.
  let prob = @ordinarydiffeq.OdeProblem::iip(0.0, 2.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, u.unsafe_get(0) * u.unsafe_get(0))
  })
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.05,
  )
  let sol = @ordinarydiffeq.solve(prob, euler(), opts)
  assert_true(sol.retcode.is_unstable())
}

///|
test "degenerate inputs fail with InitialFailure" {
  // No step size for a fixed-step solve.
  let sol = @ordinarydiffeq.solve(
    decay_problem(),
    euler(),
    @ordinarydiffeq.SolveOptions::new(),
  )
  assert_true(sol.retcode.is_initial_failure())
  assert_eq(sol.t.length(), 0)
  // Empty time span.
  let prob = @ordinarydiffeq.OdeProblem::iip(1.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let sol2 = @ordinarydiffeq.solve(
    prob,
    euler(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
  )
  assert_true(sol2.retcode.is_initial_failure())
  // Empty state.
  let prob2 = @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [], fn(_t, _u, _du) {
    ()
  })
  let sol3 = @ordinarydiffeq.solve(
    prob2,
    euler(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
  )
  assert_true(sol3.retcode.is_initial_failure())
}

///|
test "out-of-place right-hand sides work too" {
  let prob = @ordinarydiffeq.OdeProblem::oop(0.0, 1.0, [1.0], fn(t, u) {
    [-t * u.unsafe_get(0)]
  })
  // u' = -t u has u(t) = exp(-t^2/2); Euler with dt=0.1 tracks it to ~2e-2
  // (first-order error), so compare against Euler's exact recurrence value
  // u10 = 0.6281565095552947... rather than the analytic solution.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(prob, euler(), opts)
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - 0.6281565095552947).abs() < 0.0000000000001)
  assert_true((last - @math.exp(-0.5)).abs() < 0.03)
}
