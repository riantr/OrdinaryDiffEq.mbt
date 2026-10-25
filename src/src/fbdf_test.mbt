// End-to-end tests for FBDF — the variable-order, variable-step multistep BDF.
//
// What these tests pin down that the in-file unit tests cannot:
//
//   * the full stepping loop (loopheader / perform_step / loopfooter) with
//     FBDF's own controllers selected through `Integrator.bdf_active`;
//   * the fixed-step collapse to the implicit Euler method, including the
//     `iters_from_event` invariant that keeps the history from ever forming;
//   * L-stability on a stiff decay, where an explicit method would need three
//     orders of magnitude more steps;
//   * the Chebyshev dense output, saveat interpolation and tstop landing;
//   * the 2x2 Jacobian path (the unit tests are all scalar).
//
// Numbers that depend on the exact controller trajectory (step counts, `nf`)
// are asserted as bounds or exact closed forms, never as bare equality with a
// recorded run — the closed forms are the specification.

///|
/// `u' = -u` on [0, 1] with u(0) = 1: exact solution exp(-t).
fn fbdf_decay() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
/// Adaptive options at the given tolerances, dt = 0 (automatic initial step).
fn fbdf_adaptive_opts(
  reltol : Double,
  abstol : Double,
) -> @ordinarydiffeq.SolveOptions {
  @ordinarydiffeq.SolveOptions::with_abstol(
    @ordinarydiffeq.SolveOptions::with_reltol(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        true,
      ),
      reltol,
    ),
    abstol,
  )
}

///|
/// A genuinely stiff linear problem: `u' = -1000u`, `u(0) = 1`, so
/// `u(t) = exp(-1000t)`. An explicit method needs `dt <~ 2.8e-3` merely to
/// stay stable, i.e. more than 350 steps over [0, 1]; an L-stable BDF takes
/// steps many orders of magnitude larger once the transient has passed.
fn fbdf_stiff_decay() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -1000.0 * u.unsafe_get(0))
  })
}

///|
/// A stiff linear system with an exactly known solution:
/// `u' = A u` where `A = [[-0.1, 0], [-99, -0.1]]`.
///
/// The eigenvalue -0.1 is defective (a 2x2 Jordan block), so
/// `A = -0.1 I + N` with `N = [[0, 0], [-99, 0]]` nilpotent (`N² = 0`), giving
/// `exp(At) = exp(-0.1t)(I + tN)`. The solution decays slowly while the
/// Jacobian carries the 99 coupling — this is the test that exercises the
/// dense `LU` and the finite-difference Jacobian at dimension 2.
fn fbdf_stiff_chain() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0, 0.0], fn(_t, u, du) {
    let a = u.unsafe_get(0)
    let b = u.unsafe_get(1)
    du.unsafe_set(0, -0.1 * a)
    du.unsafe_set(1, -99.0 * a - 0.1 * b)
  })
}

///|
/// The exact solution of `fbdf_stiff_chain`: `exp(-0.1t) * (1, -99t)`.
fn fbdf_stiff_chain_exact(t : Double) -> (Double, Double) {
  let e = @math.exp(-0.1 * t)
  (e, e * (-99.0 * t))
}

///|
test "fbdf reports order 1, fsal and max_order + 1 stages" {
  let alg = @ordinarydiffeq.FBDF::new()
  // `alg_order(FBDF)` is the dummy 1; `kshortsize = max_order + 1`.
  assert_eq(alg.order(), 1)
  assert_eq(alg.stage_count(), 6)
  assert_true(alg.is_fsal())
  // The BDF controllers are selected, not the PI ones.
  assert_true(alg.uses_bdf_controllers())
  let alg3 = @ordinarydiffeq.FBDF::with_max_order(3)
  assert_eq(alg3.stage_count(), 4)
}

///|
test "fbdf adaptive solves a linear decay accurately" {
  let sol = @ordinarydiffeq.solve(
    fbdf_decay(),
    @ordinarydiffeq.FBDF::new(),
    fbdf_adaptive_opts(0.000001, 0.000000001),
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  // A 5th-order method at reltol 1e-6 lands well inside 1e-4 at t = 1.
  assert_true((last - @math.exp(-1.0)).abs() < 0.0001)
  // Adaptive: far fewer than the 100 steps a fixed dt = 0.01 run would take,
  // and far more than the single step a pure order-1 method could not skip.
  assert_true(sol.t.length() > 3)
  assert_true(sol.t.length() < 100)
  // The trajectory is monotone decreasing and stays in (0, 1].
  let mut i = 0
  while i < sol.t.length() {
    let v = sol.u.unsafe_get(i).unsafe_get(0)
    assert_true(v > 0.0 && v <= 1.0)
    if i > 0 {
      assert_true(v < sol.u.unsafe_get(i - 1).unsafe_get(0))
    }
    i += 1
  }
  // Times advance monotonically to the end of the span.
  assert_true((sol.t.unsafe_get(sol.t.length() - 1) - 1.0).abs() < 0.000000001)
}

///|
/// The fixed-step invariant: with `time_filter` off and no adaptive
/// controller, `iters_from_event` never leaves 0, so the predictor is always
/// `u₀ = u`, no history accumulates, the order stays 1 — and FBDF *is* the
/// implicit Euler method, `u_{n+1} = u_n / (1 + dt)`.
///
/// The anchor is `(1/1.1)^10`, not `(0.9)^10`: the implicit relation
/// `(u_{n+1} - u_n)/dt = -u_{n+1}` solves to `u_{n+1} = u_n/(1 + dt)`.
test "fbdf fixed step is the implicit Euler method" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(
    fbdf_decay(),
    @ordinarydiffeq.FBDF::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  // 10 steps of exactly 0.1 over [0, 1]: the start, 9 interior saves and the
  // end point (t = 1.0 lands exactly on the final tstop).
  assert_eq(sol.t.length(), 11)
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true(
    (last - @ordinarydiffeq.fastpower(1.0 / 1.1, 10.0)).abs() < 0.000000001,
  )
  // Evaluation budget: the pre-start FSAL (1), then per step the finite-
  // difference Jacobian (dim = 1, so 1 — the base value is `fsalfirst`) plus
  // two Newton residuals (the linear solve needs one iteration to get close
  // and a second to certify convergence). The `fsallast` identity is free.
  assert_eq(sol.nf, 1 + 10 * (1 + 2))
}

///|
/// L-stability: the same problem that forces an explicit method into 350+
/// steps is taken by FBDF in a small multiple of ten, and the answer is
/// finite and tiny rather than an oscillation that grows without bound.
test "fbdf stays stable on a stiff decay" {
  let sol = @ordinarydiffeq.solve(
    fbdf_stiff_decay(),
    @ordinarydiffeq.FBDF::new(),
    fbdf_adaptive_opts(0.001, 0.000001),
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  // exp(-1000) is ~5e-435; anything below 1e-6 is "decayed", and the point
  // of the test is that it did not blow up or stall.
  assert_true(last.abs() < 0.000001)
  assert_true(@ordinarydiffeq.is_finite(last))
  // The step-count gap against an explicit method is the whole claim.
  assert_true(sol.t.length() < 100)
  assert_true(sol.t.length() > 2)
}

///|
/// The 2x2 Jordan-block problem: accuracy of the dense LU solve and the
/// finite-difference Jacobian at dimension 2, over a stiff coupling of 99.
test "fbdf solves a stiff 2d chain accurately" {
  let sol = @ordinarydiffeq.solve(
    fbdf_stiff_chain(),
    @ordinarydiffeq.FBDF::new(),
    fbdf_adaptive_opts(0.000001, 0.000000001),
  )
  assert_true(sol.retcode.is_success())
  let n = sol.u.length()
  let (u1, u2) = fbdf_stiff_chain_exact(1.0)
  assert_true((sol.u.unsafe_get(n - 1).unsafe_get(0) - u1).abs() < 0.0001)
  assert_true((sol.u.unsafe_get(n - 1).unsafe_get(1) - u2).abs() < 0.001)
  // Mid-span check through the dense output. It has to go through the
  // interpolant, not the saved values: adaptive steps cluster in the initial
  // transient (the coupling of 99 has a timescale of ~0.01), so the step at
  // index n/2 sits at t ~ 0.006, nowhere near the middle of the span.
  let (m1, m2) = fbdf_stiff_chain_exact(0.5)
  let mid = @ordinarydiffeq.Solution::call(sol, 0.5)
  assert_true((mid.unsafe_get(0) - m1).abs() < 0.001)
  assert_true((mid.unsafe_get(1) - m2).abs() < 0.01)
}

///|
/// Dense output: the Chebyshev-resampled stages are stored per saved point
/// (one group of `max_order + 1` vectors each), and both the value and the
/// derivative closure agree with the exact solution.
test "fbdf dense output reproduces the trajectory" {
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    fbdf_adaptive_opts(0.00000001, 0.0000000001),
    true,
  )
  let sol = @ordinarydiffeq.solve(
    fbdf_decay(),
    @ordinarydiffeq.FBDF::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  assert_true(sol.k.length() == sol.t.length())
  assert_eq(sol.k.unsafe_get(1).length(), 6)
  let mid = @ordinarydiffeq.Solution::call(sol, 0.3).unsafe_get(0)
  assert_true((mid - @math.exp(-0.3)).abs() < 0.0001)
  let d = @ordinarydiffeq.Solution::derivative(sol, 0.3).unsafe_get(0)
  assert_true((d + @math.exp(-0.3)).abs() < 0.001)
}

///|
/// saveat: a non-empty list switches off `save_everystep` (and dense storage),
/// and the mid-step points are evaluated with the solver's polynomial, the
/// way Julia's `interp_at_saveat` does.
test "fbdf saveat interpolates at the requested times" {
  let opts = @ordinarydiffeq.SolveOptions::with_saveat(
    fbdf_adaptive_opts(0.00000001, 0.0000000001),
    [0.0, 0.3, 1.0],
  )
  let sol = @ordinarydiffeq.solve(
    fbdf_decay(),
    @ordinarydiffeq.FBDF::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  assert_eq(sol.t.length(), 3)
  assert_eq(sol.t.unsafe_get(0), 0.0)
  assert_eq(sol.t.unsafe_get(1), 0.3)
  assert_eq(sol.t.unsafe_get(2), 1.0)
  assert_true(sol.k.length() == 0)
  let u03 = @ordinarydiffeq.Solution::call(sol, 0.3).unsafe_get(0)
  assert_true((u03 - @math.exp(-0.3)).abs() < 0.0001)
}

///|
/// tstops: the step aimed at a stop is snapped onto it exactly, and the run
/// continues past it to the end of the span.
test "fbdf tstops hits the stop exactly" {
  let opts = @ordinarydiffeq.SolveOptions::with_tstops(
    fbdf_adaptive_opts(0.00000001, 0.0000000001),
    [0.5],
  )
  let sol = @ordinarydiffeq.solve(
    fbdf_decay(),
    @ordinarydiffeq.FBDF::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let mut found = false
  for t in sol.t {
    if (t - 0.5).abs() < 0.000000001 {
      found = true
    }
  }
  assert_true(found)
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - @math.exp(-1.0)).abs() < 0.0001)
}

///|
/// A different `max_order`: the method must stay accurate with the order
/// capped at 3 (STALD's alpha-stability window), and the stage count follows.
test "fbdf with max_order 3 stays accurate" {
  let sol = @ordinarydiffeq.solve(
    fbdf_stiff_chain(),
    @ordinarydiffeq.FBDF::with_max_order(3),
    fbdf_adaptive_opts(0.000001, 0.000000001),
  )
  assert_true(sol.retcode.is_success())
  let n = sol.u.length()
  let (u1, u2) = fbdf_stiff_chain_exact(1.0)
  assert_true((sol.u.unsafe_get(n - 1).unsafe_get(0) - u1).abs() < 0.001)
  assert_true((sol.u.unsafe_get(n - 1).unsafe_get(1) - u2).abs() < 0.01)
}

///|
/// Backward integration: `tdir = -1` flows through the reject controller's
/// magnitude/direction split and the tstop/saveat bookkeeping. Exact solution
/// of `u' = -u` solved from t = 0 back to t = -1 is `exp(-t)` evaluated at
/// negative t, i.e. growing.
test "fbdf integrates backward in time" {
  let prob = @ordinarydiffeq.OdeProblem::iip(0.0, -1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let sol = @ordinarydiffeq.solve(
    prob,
    @ordinarydiffeq.FBDF::new(),
    fbdf_adaptive_opts(0.000001, 0.000000001),
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  // exp(1) at t = -1.
  assert_true((last - @math.exp(1.0)).abs() < 0.001)
  // Times descend.
  assert_true(sol.t.unsafe_get(sol.t.length() - 1) < sol.t.unsafe_get(0))
}
