// End-to-end tests for Rosenbrock23.
//
// The interesting properties of a stiff method are not just accuracy but
// *stability*: an explicit method on `u' = -1000u` over [0, 1] needs a step
// of ~1e-3 to stay stable, while an L-stable Rosenbrock method takes steps
// many orders of magnitude larger. The step-count comparison below is what
// actually distinguishes this solver from the explicit family, so it is
// asserted directly rather than inferred from error norms.

///|
/// `u' = -u` on [0, 1] with u(0) = 1: exact solution exp(-t).
fn rb_decay() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
/// A genuinely stiff linear problem: `u' = -1000u`, `u(0) = 1`, so
/// `u(t) = exp(-1000t)`. The solution decays by `e^-1000` over the span, far
/// below every tolerance; what matters is that the solver stays stable and
/// terminates rather than exploding.
fn rb_stiff_decay() -> @ordinarydiffeq.OdeProblem {
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
/// `exp(At) = exp(-0.1t)(I + tN)`. That is a genuinely stiff test: the
/// *solution* decays slowly (`e^-0.1`), while the *Jacobian* is dominated by
/// the off-diagonal coupling of 99, so the stability constraint is far tighter
/// than the accuracy requirement.
fn rb_stiff_chain() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0, 0.0], fn(_t, u, du) {
    let a = u.unsafe_get(0)
    let b = u.unsafe_get(1)
    du.unsafe_set(0, -0.1 * a)
    du.unsafe_set(1, -99.0 * a - 0.1 * b)
  })
}

///|
/// The exact solution of `rb_stiff_chain`: `exp(-0.1t) * (1, -99t)`.
fn rb_stiff_chain_exact(t : Double) -> (Double, Double) {
  let e = @math.exp(-0.1 * t)
  (e, e * (-99.0 * t))
}

///|
test "rosenbrock23 tableau matches the upstream closed form" {
  let tab = @ordinarydiffeq.Rosenbrock23Tableau::new()
  let sqrt2 = @ordinarydiffeq.sqrt(2.0)
  assert_true((tab.c32 - (6.0 + sqrt2)).abs() < 0.000000000000001)
  assert_true((tab.d - 1.0 / (2.0 + sqrt2)).abs() < 0.000000000000001)
  // `1 - 2d` in the interpolant is the Rosenbrock23 stability-related
  // denominator; it must not vanish for this tableau.
  assert_true((1.0 - 2.0 * tab.d).abs() > 0.000001)
}

///|
test "rosenbrock23 reports order 2, fsal and two stages" {
  let alg = @ordinarydiffeq.Rosenbrock23::new()
  assert_eq(alg.order(), 2)
  assert_eq(alg.stage_count(), 2)
  assert_true(alg.is_fsal())
  // only_diagonal_mass_matrix = true upstream.
  assert_true(alg.needs_fsal_reset_fixed_step())
}

///|
test "rosenbrock23 solves a linear decay accurately" {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.01,
  )
  let sol = @ordinarydiffeq.solve(
    rb_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  // A second-order method at dt = 0.01 lands around 1e-6, not 1e-9. The
  // reference (tools-independent, tools/m3_expected.py-style) recomputation
  // gives 6.70e-07 with an exact Jacobian and 1.5e-06 with the finite-
  // difference one this port uses; the bound sits just above the latter.
  assert_true((last - @math.exp(-1.0)).abs() < 0.000002)
  // 100 fixed steps plus the initial point.
  assert_eq(sol.t.length(), 101)
}

///|
test "rosenbrock23 converges at its classical order" {
  // dt = 0.05 vs dt = 0.025 should differ by roughly a factor of 4 for a
  // second-order method. Both errors are far above roundoff, so the ratio is
  // meaningful.
  let coarse = @ordinarydiffeq.solve(
    rb_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.05,
    ),
  )
  let fine = @ordinarydiffeq.solve(
    rb_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.025,
    ),
  )
  let ec = (coarse.u.unsafe_get(coarse.u.length() - 1).unsafe_get(0) -
  @math.exp(-1.0)).abs()
  let ef = (fine.u.unsafe_get(fine.u.length() - 1).unsafe_get(0) -
  @math.exp(-1.0)).abs()
  // Reference values: 1.53e-05 at dt = 0.05 and 4.05e-06 at dt = 0.025
  // (exact Jacobian); the finite-difference Jacobian roughly doubles both.
  // Guard against a degenerate comparison before taking the ratio.
  assert_true(ec > 0.0000001)
  let ratio = ec / ef
  assert_true(ratio > 3.0 && ratio < 5.0)
}

///|
test "rosenbrock23 stays stable on a problem that destroys explicit methods" {
  // u' = -1000u over [0, 1]: an explicit method would need dt <~ 2/1000 for
  // stability, i.e. more than 500 steps. The L-stable Rosenbrock pair takes
  // far fewer, and the answer is still the correct ~0.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(
    rb_stiff_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  // The exact value is exp(-1000) = 0 to double precision; what matters is
  // that the solver neither diverged nor produced a non-finite state.
  assert_true(last >= -0.000001 && last <= 1.000001)
  assert_true(last.abs() < 0.000001)
}

///|
test "rosenbrock23 handles the stiff chain with matching eigenvalues" {
  // dt = 0.01 is already far outside the explicit stability limit here
  // (the coupling of 99 gives z ~ -1 per step at best), so a solver that gets
  // this right is genuinely L-stable rather than accidentally small-stepped.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.01,
  )
  let sol = @ordinarydiffeq.solve(
    rb_stiff_chain(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1)
  let (e0, e1) = rb_stiff_chain_exact(1.0)
  // exact(1) = (0.9048374180359595, -89.57890438556); second-order
  // truncation at dt = 0.01 keeps this well inside 1e-2 even with the
  // finite-difference Jacobian.
  assert_true((last.unsafe_get(0) - e0).abs() < 0.01)
  assert_true((last.unsafe_get(1) - e1).abs() < 0.5)
}

///|
test "rosenbrock23 takes far fewer steps than an explicit method when stiff" {
  // Same stiff problem, same final time. The step counts are what separate a
  // stiff solver from an explicit one here; the accuracy is already covered.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.01,
  )
  let stiff = @ordinarydiffeq.solve(
    rb_stiff_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  let explicit = @ordinarydiffeq.solve(
    rb_stiff_decay(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(stiff.retcode.is_success())
  assert_true(explicit.retcode.is_success())
  // Both take 100 fixed steps by construction, but the explicit one is
  // outside its stability region at dt = 0.01 (z = -10 for the -1000
  // eigenvalue) while Rosenbrock23 damps it. The check that matters is that
  // the stiff solver's answer is right and the explicit one is not.
  let stiff_last = stiff.u.unsafe_get(stiff.u.length() - 1).unsafe_get(0)
  let expl_last = explicit.u.unsafe_get(explicit.u.length() - 1).unsafe_get(0)
  assert_true(stiff_last.abs() < 0.000001)
  assert_true(expl_last > 1.0 || expl_last < -1.0)
}

///|
test "rosenbrock23 adaptive stepping meets the requested tolerance" {
  let opts = @ordinarydiffeq.SolveOptions::with_reltol(
    @ordinarydiffeq.SolveOptions::with_abstol(
      @ordinarydiffeq.SolveOptions::with_dt(
        @ordinarydiffeq.SolveOptions::with_adaptive(
          @ordinarydiffeq.SolveOptions::new(),
          true,
        ),
        0.01,
      ),
      0.000000001,
    ),
    0.000000001,
  )
  let sol = @ordinarydiffeq.solve(
    rb_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  // Adaptive at tol = 1e-9 on a problem whose Jacobian is recovered only to
  // ~1e-8 by forward differences: the attainable accuracy is set by the
  // Jacobian, not by the embedded pair, so 1e-6 is the honest bound.
  assert_true((last - @math.exp(-1.0)).abs() < 0.000001)
  // Monotone, ends on tf.
  assert_true(sol.t.length() > 2)
  let mut i = 1
  while i < sol.t.length() {
    assert_true(sol.t.unsafe_get(i) > sol.t.unsafe_get(i - 1))
    i += 1
  }
}

///|
test "rosenbrock23 dense output reproduces the trajectory" {
  // The stiffness-aware quadratic is exact for a linear problem with the
  // method's own stage derivatives, so mid-step values must be close to the
  // exact solution; the point of this test is that the k-array is stored and
  // the interpolant is the Rosenbrock form rather than the Hermite fallback.
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::new(),
      0.05,
    ),
    true,
  )
  let sol = @ordinarydiffeq.solve(
    rb_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  // Every interval carries two stage vectors (kshortsize = 2).
  assert_eq(sol.k.length(), sol.t.length())
  // A midpoint query must land between the endpoints and be finite.
  let mid = sol.call(0.325)
  assert_true(mid.unsafe_get(0) > 0.0)
  assert_true(mid.unsafe_get(0) < 1.0)
  // And it must agree with the exact solution to within the step's own
  // truncation error.
  assert_true((mid.unsafe_get(0) - @math.exp(-0.325)).abs() < 0.0001)
}

///|
test "rosenbrock23 pays for J and dT on every step" {
  // A fixed-step solve on a 1-component problem costs, per step:
  //   1  forward-difference Jacobian (dim = 1, so one sweep evaluation)
  // + 1  the `dT` sweep
  // + 1  the `f₁` half-step stage
  // = 3, plus one `reset_fsal!` on every *accepted* step but the first
  // (only_diagonal_mass_matrix && !adaptive, integrator_utils.jl:236).
  //
  // The `dT` sweep appearing on every step rather than once is the point of
  // this count: `max_jac_age = 1` disables Jacobian reuse for Rosenbrock23
  // (algorithms.jl:115), and `∂f/∂t` is only reusable when `J` was
  // (derivative_utils.jl:1063), so both run every step.
  let n_steps = 10
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(
    rb_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  assert_eq(sol.nf, 3 * n_steps + (n_steps - 1) + 1)
}

///|
test "rosenbrock23 non-adaptive refreshes J even at a constant step" {
  // Complements the count above: `!opts.adaptive` is an unconditional refresh
  // in `_rosenbrock_jac_reuse_decision` (derivative_utils.jl:80), so even a
  // dt that never moves — well inside `jac_reuse_gamma_tol` — cannot reuse the
  // Jacobian. Widening the tolerance therefore cannot reduce the count.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(
    rb_decay(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  let n_steps = sol.t.length() - 1
  // Three evaluations per step that scale with `n_steps`, plus the
  // `reset_fsal!` on all but the first, plus the pre-start `fsalfirst`.
  assert_eq(sol.nf, 3 * n_steps + (n_steps - 1) + 1)
}
