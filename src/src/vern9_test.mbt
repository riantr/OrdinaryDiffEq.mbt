// End-to-end tests for Vern9, Verner's 9(8) embedded Runge-Kutta pair — the
// highest-order method in this package.
//
// Fixed-step accuracy, convergence order, the evaluation count (the ten lazy
// interpolant stages must stay invisible to `sol.nf`), adaptive control, dense
// output and dense derivative. The tableau and interpolant invariants live
// inside alg_vern9.mbt, which can see the private coefficient consts.

///|
/// `u' = -u` on [0, 1] with u(0) = 1: exact solution exp(-t).
fn vern9_decay_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
fn vern9() -> @ordinarydiffeq.Vern9 {
  @ordinarydiffeq.Vern9::new()
}

///|
/// exp(-1).
const VERN9_EXACT_T1 : Double = 0.36787944117144233

///|
/// Endpoint error of a fixed-step Vern9 solve with step `dt`.
///
/// No `assert_true` here: it raises an error, which a plain helper cannot
/// propagate. The success check lives in the tests that call this.
fn vern9_endpoint_error(dt : Double) -> Double {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    dt,
  )
  let sol = @ordinarydiffeq.solve(vern9_decay_problem(), vern9(), opts)
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  (last - VERN9_EXACT_T1).abs()
}

///|
test "vern9 fixed step is far more accurate than Euler" {
  // 10 steps of dt = 0.1. A 9th-order method is limited by its local error,
  // so the global error is ~O(h^10); the bound below is generous enough to
  // absorb rounding yet far tighter than Euler's ~4e-3 on the same grid.
  assert_true(vern9_endpoint_error(0.1) < 0.00000001)
}

///|
test "vern9 converges at roughly its classical order" {
  // Deliberately coarse steps. Vern9 is so accurate that from dt = 0.2 down the
  // error is already at the rounding floor (~1e-16), so the asymptotic regime
  // is only observable on the coarse side. Reference values for u' = -u:
  // dt = 0.5 -> ~1.4e-12, dt = 0.25 -> ~6.2e-15.
  let e_coarse = vern9_endpoint_error(0.5)
  let e_fine = vern9_endpoint_error(0.25)
  // Guard against the comparison becoming vacuous if both land on zero.
  assert_true(e_coarse > 0.0000000000001)
  // Halving dt should cut the error by about 2^9 = 512; the factor 100 below
  // asserts the order is "high" without pinning it to an exact value.
  assert_true(e_fine < e_coarse / 100.0)
}

///|
test "vern9 costs sixteen evaluations per step plus the initial one" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(vern9_decay_problem(), vern9(), opts)
  // 10 steps of dt = 0.1 from t = 0 to t = 1. Each accepted step evaluates
  // k1..k16 (Vern9 is not FSAL, so k1 is recomputed every step), plus the one
  // pre-start evaluation of f(t0, u0). The ten lazy interpolant stages go
  // through `eval_f_uncounted` and must NOT show up here.
  assert_true(sol.nf == 161)
}

///|
test "vern9 lazy stages stay invisible to the evaluation count" {
  // Dense output triggers `fill_dense_stages` for every saved step — ten extra
  // f evaluations each — but upstream's `addsteps!` does not call
  // `increment_nf!`, so the reported count is unchanged. This is the single most
  // easily broken invariant of the lazy path.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(vern9_decay_problem(), vern9(), opts)
  assert_true(sol.nf == 161)
  assert_true(sol.k.length() > 0)
}

///|
test "vern9 adaptive stepping hits the requested tolerance" {
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
  let sol = @ordinarydiffeq.solve(vern9_decay_problem(), vern9(), opts)
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - VERN9_EXACT_T1).abs() < 0.00000001)
  // A 9th-order method at these tolerances needs only a handful of steps.
  assert_true(sol.t.length() < 20)
}

///|
test "vern9 reports its order, fsal flag and stage count" {
  assert_true(vern9().order() == 9)
  // alg_utils.jl: `isfsal(::Vern9) = false`.
  assert_true(!vern9().is_fsal())
  // 10 stored step slots plus 10 lazy interpolant slots.
  assert_true(vern9().stage_count() == 20)
}

///|
test "vern9 dense output reproduces the trajectory" {
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(vern9_decay_problem(), vern9(), opts)
  let at_start = @ordinarydiffeq.Solution::call(sol, 0.0)
  let at_end = @ordinarydiffeq.Solution::call(sol, 1.0)
  assert_true((at_start.unsafe_get(0) - 1.0).abs() < 0.000001)
  assert_true((at_end.unsafe_get(0) - VERN9_EXACT_T1).abs() < 0.000001)
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
test "vern9 dense derivative matches f along the trajectory" {
  // For u' = -u the true derivative at t is -u(t), so the interpolant's
  // derivative closure must satisfy d(t) == -v(t) to interpolant accuracy.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.1,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(vern9_decay_problem(), vern9(), opts)
  let mut t = 0.13
  while t < 0.95 {
    let v = @ordinarydiffeq.Solution::call(sol, t)
    let d = @ordinarydiffeq.Solution::derivative(sol, t)
    assert_true((d.unsafe_get(0) + v.unsafe_get(0)).abs() < 0.000001)
    t += 0.17
  }
}
