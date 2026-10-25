// End-to-end tests for Vern7, Verner's 7(6) embedded Runge-Kutta pair.
//
// Fixed-step accuracy, convergence order, the evaluation count (the lazy
// interpolant stages must stay invisible to `sol.nf`), adaptive control,
// dense output, plus tableau and interpolant invariants.
//
// Test files see the package through its interface, so every reference is
// qualified with `@ordinarydiffeq.` and the algorithm comes from `new`.

///|
/// `u' = -u` on [0, 1] with u(0) = 1: the canonical accuracy problem, whose
/// exact solution is exp(-t). (Named apart from the other test files' helper —
/// the package shares one namespace across files.)
fn vern7_decay_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
fn vern7() -> @ordinarydiffeq.Vern7 {
  @ordinarydiffeq.Vern7::new()
}

///|
/// exp(-1).
const VERN7_EXACT_T1 : Double = 0.36787944117144233

///|
/// Endpoint error of a fixed-step Vern7 solve with step `dt`.
///
/// No `assert_true` here: it raises an error, which a plain helper cannot
/// propagate. The success check lives in the tests that call this.
fn vern7_endpoint_error(dt : Double) -> Double {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    dt,
  )
  let sol = @ordinarydiffeq.solve(vern7_decay_problem(), vern7(), opts)
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  (last - VERN7_EXACT_T1).abs()
}

///|
test "vern7 fixed step is far more accurate than Euler" {
  // 10 steps of dt = 0.1. A 7th-order method is limited by its local error,
  // so the global error is ~O(h^8); the bound below is generous enough to
  // absorb rounding yet far tighter than Euler's ~4e-3 on the same grid.
  assert_true(vern7_endpoint_error(0.1) < 0.00000001)
}

///|
test "vern7 converges at roughly its classical order" {
  let e_coarse = vern7_endpoint_error(0.2)
  let e_fine = vern7_endpoint_error(0.1)
  // Halving dt should cut the error by about 2^7 = 128; the factor 50 below
  // only asserts the order is "high", so a single-step anomaly cannot fail it.
  assert_true(e_fine < e_coarse / 50.0)
}

///|
test "vern7 costs ten evaluations per step plus the initial one" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(vern7_decay_problem(), vern7(), opts)
  // 10 steps of dt = 0.1 from t = 0 to t = 1. Each accepted step evaluates
  // k1..k10 (Vern7 is not FSAL, so k1 is recomputed every step), plus the one
  // pre-start evaluation of f(t0, u0). The six lazy interpolant stages go
  // through `eval_f_uncounted` and must NOT show up here.
  assert_true(sol.nf == 101)
}

///|
test "vern7 lazy stages stay invisible to the evaluation count" {
  // Dense output triggers `fill_dense_stages` for every saved step — six
  // extra f evaluations each — but upstream's `addsteps!` does not call
  // `increment_nf!`, so the reported count is unchanged. This is the single
  // most easily broken invariant of the lazy path.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(vern7_decay_problem(), vern7(), opts)
  assert_true(sol.nf == 101)
  assert_true(sol.k.length() > 0)
}

///|
test "vern7 adaptive stepping hits the requested tolerance" {
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
  let sol = @ordinarydiffeq.solve(vern7_decay_problem(), vern7(), opts)
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - VERN7_EXACT_T1).abs() < 0.00000001)
  // A 7th-order method at these tolerances needs only a handful of steps.
  assert_true(sol.t.length() < 20)
}

///|
test "vern7 reports its order, fsal flag and stage count" {
  assert_true(vern7().order() == 7)
  // alg_utils.jl: `isfsal(::Vern7) = false`.
  assert_true(!vern7().is_fsal())
  // 10 step stages plus 6 lazy interpolant stages.
  assert_true(vern7().stage_count() == 16)
}

///|
test "vern7 dense output reproduces the trajectory" {
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(vern7_decay_problem(), vern7(), opts)
  let at_start = @ordinarydiffeq.Solution::call(sol, 0.0)
  let at_end = @ordinarydiffeq.Solution::call(sol, 1.0)
  assert_true((at_start.unsafe_get(0) - 1.0).abs() < 0.000001)
  assert_true((at_end.unsafe_get(0) - VERN7_EXACT_T1).abs() < 0.000001)
  // Off-grid queries exercise the interpolant proper rather than a stored node.
  let mut t = 0.07
  while t < 0.99 {
    let v = @ordinarydiffeq.Solution::call(sol, t)
    let exact = @math.exp(-t)
    assert_true((v.unsafe_get(0) - exact).abs() < 0.000001)
    t += 0.11
  }
}

///|
test "vern7 dense derivative matches f along the trajectory" {
  // For u' = -u the true derivative at t is -u(t), so the interpolant's
  // derivative closure must satisfy d(t) == -v(t) to interpolant accuracy.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(vern7_decay_problem(), vern7(), opts)
  let mut t = 0.13
  while t < 0.95 {
    let v = @ordinarydiffeq.Solution::call(sol, t)
    let d = @ordinarydiffeq.Solution::derivative(sol, t)
    assert_true((d.unsafe_get(0) + v.unsafe_get(0)).abs() < 0.000001)
    t += 0.17
  }
}
