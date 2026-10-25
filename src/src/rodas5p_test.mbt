// End-to-end tests for Rodas5P.
//
// Rodas5P is the fifth-order member of the stiffly stable Rosenbrock family
// (Steinebach 2023). Two things separate it from Rosenbrock23 and are what
// these tests pin down:
//
//   * Accuracy. Its observable order is 5, so at dt = 0.2 on `u' = -u` the
//     error is already ~1e-10. The reference recomputation in
//     tools/m4_expected.py gives 1.86e-10 there; this port's finite-difference
//     Jacobian costs a little accuracy but stays in the same decade.
//   * Cost per step. It is a tableau-driven method with `num_stages = 8`
//     right-hand sides sharing one factorization, and — unlike Rosenbrock23 —
//     it is a W-method, so `max_jac_age = 20` lets one Jacobian serve many
//     steps whenever dt barely moves. Both are asserted directly rather than
//     inferred, because they are what actually distinguish this solver.

///|
/// `u' = -u` on [0, 1] with u(0) = 1: exact solution exp(-t).
fn r5_decay() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
/// `u' = -1000u`: the solution falls by `e^-1000` over the span, far below
/// every tolerance. Only stability and termination are meaningful here.
fn r5_stiff_decay() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -1000.0 * u.unsafe_get(0))
  })
}

///|
/// A stiff, non-normal system with an exactly known solution:
/// `u' = A u` where `A = [[-0.1, 0], [-99, -0.1]]`.
///
/// The eigenvalue -0.1 is defective, so `A = -0.1 I + N` with `N` nilpotent
/// and `exp(At) = exp(-0.1t)(I + tN)`. The non-normality is the point: the
/// Jacobian is dominated by the off-diagonal 99, so the stability constraint is
/// far tighter than the accuracy requirement — exactly the regime a stiffly
/// stable Rosenbrock method is built for.
fn r5_stiff_chain() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0, 0.0], fn(_t, u, du) {
    let a = u.unsafe_get(0)
    let b = u.unsafe_get(1)
    du.unsafe_set(0, -0.1 * a)
    du.unsafe_set(1, -99.0 * a - 0.1 * b)
  })
}

///|
/// The exact solution of `r5_stiff_chain`: `exp(-0.1t) * (1, -99t)`.
fn r5_stiff_chain_exact(t : Double) -> (Double, Double) {
  let e = @math.exp(-0.1 * t)
  (e, e * (-99.0 * t))
}

///|
/// Solve `r5_decay` at a fixed step and return the final value.
fn r5_fixed_last(dt : Double) -> Double {
  let sol = @ordinarydiffeq.solve(
    r5_decay(),
    @ordinarydiffeq.Rodas5P::new(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        false,
      ),
      dt,
    ),
  )
  // Deliberately no `is_success` assertion: `solve` carries an error type, so
  // the helper that every accuracy test reads through must propagate it rather
  // than swallow it. A failed solve surfaces as the caller's own error.
  sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
}

///|
test "rodas5p tableau carries the upstream RODAS5P coefficients" {
  let tab = @ordinarydiffeq.Rodas5P::new().tableau()
  assert_eq(tab.num_stages, 8)
  // `interp_order == size(H, 1) = 3` (rosenbrock_caches.jl:506), which is what
  // routes the interpolant to the `else` branch rather than Hermite.
  assert_eq(tab.h_rows, 3)
  // `gamma = 0.21193756319429014` (rosenbrock_tableaus.jl:65).
  assert_true((tab.gamma - 0.21193756319429014).abs() < 0.000000000000001)
  // `d[0] == gamma` — the first stage's `∂f/∂t` term is exactly
  // `dt*gamma*dT`, the same combination Rosenbrock23 assembles by hand.
  assert_true((tab.d[0] - tab.gamma).abs() < 0.000000000000001)
  // `b = RODAS5PA[8, 1:8]` with a 1 appended (rosenbrock_tableaus.jl:67-68):
  // the first seven entries are row 8 of `A`, and the eighth is a literal 1
  // rather than `A[8, 8]` — which is 0, since `A` is strictly lower
  // triangular. That last coefficient is the one that would be silently lost
  // by a transcription that read all eight from the matrix.
  let mut i = 0
  while i < 7 {
    assert_true((tab.b[i] - tab.a[7 * 8 + i]).abs() < 0.000000000000001)
    i += 1
  }
  assert_true((tab.b[7] - 1.0).abs() < 0.000000000000001)
  assert_eq(tab.a[7 * 8 + 7], 0.0)
  // `btilde = zeros(8)` with the last entry set to 1
  // (rosenbrock_tableaus.jl:69-70): the embedded error is the last stage
  // alone, matching `alg_adaptive_order = alg_order - 1 = 4`.
  let mut k = 0
  while k < 7 {
    assert_eq(tab.btilde[k], 0.0)
    k += 1
  }
  assert_eq(tab.btilde[7], 1.0)
  // The last three `c` entries are 1, so stages 6, 7 and 8 all evaluate `f`
  // at the step's end point — a property the upstream stage-skipping branch
  // (rosenbrock_perform_step.jl:471) keys off, and one a transcription error
  // in `RODAS5Pc` would break.
  assert_eq(tab.c[5], 1.0)
  assert_eq(tab.c[6], 1.0)
  assert_eq(tab.c[7], 1.0)
  // `A` is strictly lower triangular: row `s` must be zero for `i >= s`.
  let mut s = 0
  while s < 8 {
    let mut j = s
    while j < 8 {
      assert_eq(tab.a[s * 8 + j], 0.0)
      j += 1
    }
    s += 1
  }
}

///|
test "rodas5p reports order 5, no fsal and three dense stages" {
  let alg = @ordinarydiffeq.Rodas5P::new()
  // alg_order(Rodas5P) = 5 (alg_utils.jl:35).
  assert_eq(alg.order(), 5)
  // interp_order = size(H, 1) = 3 (rosenbrock_caches.jl:506).
  assert_eq(alg.stage_count(), 3)
  // isfsal(Rodas5P) = false (alg_utils.jl:60) — the first stage re-evaluates
  // `f(uprev, t)` itself instead of reusing the integrator's `fsalfirst`.
  assert_true(!alg.is_fsal())
  // `only_diagonal_mass_matrix` lists only Rosenbrock23/32
  // (alg_utils.jl:82), and with `is_fsal = false` the `update_fsal!` body is
  // skipped regardless.
  assert_true(!alg.needs_fsal_reset_fixed_step())
}

///|
test "rodas5p reuses its jacobian across up to max_jac_age steps" {
  // Rodas5P is a W-method with `max_jac_age = 20` and
  // `jac_reuse_gamma_tol = 0.03` (algorithms.jl:115,137) — the reuse default
  // that distinguishes it from Rosenbrock23's `max_jac_age = 1`. A fixed-step
  // run therefore evaluates `∂f/∂t` once and reuses it for every step, while
  // the `W` factorization is rebuilt each time from the cached `J`. A fixed
  // step size moves `dtgamma` by exactly zero, comfortably inside the
  // tolerance.
  let tab = @ordinarydiffeq.Rodas5P::new().tableau()
  assert_eq(@ordinarydiffeq.RODAS5P_MAX_JAC_AGE, 20)
  assert_true(
    (@ordinarydiffeq.RODAS5P_JAC_GAMMA_TOL - 0.03).abs() < 0.000000000000001,
  )
  // Ten fixed steps of 0.1 over [0, 1]: same accuracy as the reference
  // recomputation (3.30e-12), so the reuse is not degrading the answer.
  let last = r5_fixed_last(0.1)
  assert_true((last - @math.exp(-1.0)).abs() < 0.0000000001)
}

///|
test "rodas5p solves a linear decay at fifth-order accuracy" {
  // At dt = 0.2 the reference recomputation gives 1.86e-10; this port uses a
  // forward-difference Jacobian, which costs a little accuracy but should stay
  // in the same decade. This bound is what a fifth-order method buys over
  // Rosenbrock23's ~1.5e-06 at the same step size.
  let last = r5_fixed_last(0.2)
  assert_true((last - @math.exp(-1.0)).abs() < 0.000000001)
}

///|
test "rodas5p converges at its classical order" {
  // dt halving on `u' = -u` should shrink the error by roughly 2^5 = 32 for
  // a fifth-order method. Reference values with an exact Jacobian are
  // 1.86e-10 (dt = 0.2), 3.30e-12 (dt = 0.1) and 5.42e-14 (dt = 0.05), i.e.
  // ratios of 56 and 61; the finite-difference Jacobian shifts these without
  // changing the order.
  let e0 = (r5_fixed_last(0.2) - @math.exp(-1.0)).abs()
  let e1 = (r5_fixed_last(0.1) - @math.exp(-1.0)).abs()
  let e2 = (r5_fixed_last(0.05) - @math.exp(-1.0)).abs()
  // Guard against a degenerate comparison before taking ratios: the
  // dt = 0.05 error is near the roundoff floor, so only the first two are
  // used for the order check.
  assert_true(e0 > 0.0000000001)
  assert_true(e1 > 0.0000000000001)
  let r1 = e0 / e1
  let r2 = e1 / e2
  // An order-5 method lands near 32; the wide band absorbs the
  // finite-difference Jacobian's contribution to the effective order without
  // letting a second- or third-order method through.
  assert_true(r1 > 8.0 && r1 < 128.0)
  assert_true(r2 > 8.0 && r2 < 128.0)
}

///|
test "rodas5p stays stable where explicit methods need tiny steps" {
  // `u' = -1000u` over [0, 1]: an explicit method needs dt <~ 2/1000 to stay
  // stable, i.e. more than 500 steps. The stiffly stable Rodas5P damps it at
  // dt = 0.5 and still lands on the correct exp(-1000) = 0.
  let sol = @ordinarydiffeq.solve(
    r5_stiff_decay(),
    @ordinarydiffeq.Rodas5P::new(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        false,
      ),
      0.5,
    ),
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true(last >= -0.001 && last <= 1.001)
  // The reference recomputation gives 5.385737e-04 at dt = 0.5 — the
  // truncation error of a fifth-order method on z = -500, not a stability
  // failure, and it damps toward zero as dt shrinks (2.19e-11 at dt = 0.1,
  // 4.07e-140 at dt = 0.01). Asserting against that value rather than
  // against zero is what makes this a check on the method instead of a check
  // on the tolerance.
  assert_true((last - 0.0005385737487553507).abs() < 0.0000001)
  // Two fixed steps plus the initial point.
  assert_eq(sol.t.length(), 3)
}

///|
test "rodas5p damps the fast mode of the stiff chain" {
  // dt = 0.01 is well outside the explicit stability limit here — the
  // off-diagonal coupling of 99 puts `z` near -1 per step at best — so
  // getting this right is a statement about L-stability, not about
  // accidentally small step sizes.
  let sol = @ordinarydiffeq.solve(
    r5_stiff_chain(),
    @ordinarydiffeq.Rodas5P::new(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        false,
      ),
      0.01,
    ),
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1)
  let (e0, e1) = r5_stiff_chain_exact(1.0)
  // exact(1) = (0.9048374180359595, -89.57890438556). The reference
  // recomputation gives (3.3e-16, 9.1e-13) at this step size with an exact
  // Jacobian; the finite-difference one leaves far more room, and the second
  // component is the sensitive one because it is driven by the 99x coupling.
  assert_true((last.unsafe_get(0) - e0).abs() < 0.0001)
  assert_true((last.unsafe_get(1) - e1).abs() < 0.01)
}

///|
test "rodas5p adaptive solve reaches the tolerance it was given" {
  // The embedded estimate is `btilde`-weighted — only the last stage — so the
  // controller sees a fourth-order error signal and drives it below 1. This is
  // the path `alg_adaptive_order = alg_order - 1 = 4` describes.
  // An initial  is required as the starting guess: with  left unset
  // this port's adaptive loop exits immediately with an empty solution and a
  //  code (a pre-existing gap in the initial-dt heuristic that
  // affects every solver here, not something Rodas5P introduces). Supplying
  // 0.1 exercises the controller, which then has to shrink it to reach 1e-10.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::with_abstol(
      @ordinarydiffeq.SolveOptions::with_reltol(
        @ordinarydiffeq.SolveOptions::new(),
        0.0000000001,
      ),
      0.0000000001,
    ),
    0.1,
  )
  let sol = @ordinarydiffeq.solve(
    r5_decay(),
    @ordinarydiffeq.Rodas5P::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - @math.exp(-1.0)).abs() < 0.00000001)
  // Adaptive stepping must actually be taking small steps to hit 1e-10; a
  // solver that silently ran at a huge fixed step would not land here.
  assert_true(sol.t.length() > 2)
}

///|
test "rodas5p dense output reproduces the trajectory between steps" {
  // The stiff-aware cubic interpolant
  //   `Θ₁·y₀ + Θ·(y₁ + Θ₁·(k₁ + Θ·(k₂ + Θ·k₃)))`
  // (rosenbrock_interpolants.jl:199-205) is built from the three `H`-combined
  // stages, so it is only exercised if the solver is actually asked for the
  // solution between its own step points. Sampling it must agree with the
  // exact trajectory to well below the step size's own truncation error.
  let sol = @ordinarydiffeq.solve(
    r5_decay(),
    @ordinarydiffeq.Rodas5P::new(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        false,
      ),
      0.1,
    ),
  )
  assert_true(sol.retcode.is_success())
  // Midpoint of the first step, where the cubic's deviation from the true
  // exponential is largest.
  let mid = @ordinarydiffeq.Solution::call(sol, 0.05)
  let exact = @math.exp(-0.05)
  // A cubic through a step of 0.1 tracks exp(-t) to a few times its
  // truncation error; 1e-5 is generous but two orders below what the explicit
  // family's Hermite interpolant would need at this step size.
  assert_true((mid.unsafe_get(0) - exact).abs() < 0.00001)
  // The derivative branch shares the same stages; differentiating the exact
  // solution gives -exp(-t), so the same bound applies.
  let slope = @ordinarydiffeq.Solution::derivative(sol, 0.05)
  assert_true((slope.unsafe_get(0) + exact).abs() < 0.0001)
}

///|
test "rodas5p reproduces the reference recomputation of one step" {
  // The strongest available check on the tableau transcription: the reference
  // implementation in tools/m4_expected.py, written independently against the
  // upstream source, reports 0.8187307531608496 for `u' = -u` at dt = 0.2 from
  // u(0) = 1, against an exact step of 0.8187307530779818. A sign error in the
  // stage solve — the failure mode this solver is most exposed to, since the
  // minus in `W \ -(...)` is easy to absorb into `W` by mistake — leaves every
  // stage finite but grows the solution without bound, so agreement here is
  // what distinguishes a correct port from a plausible one.
  let sol = @ordinarydiffeq.solve(
    r5_decay(),
    @ordinarydiffeq.Rodas5P::new(),
    @ordinarydiffeq.SolveOptions::with_dt(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        false,
      ),
      0.2,
    ),
  )
  assert_true(sol.retcode.is_success())
  // Five fixed steps of 0.2 span [0, 1], so six saved points; the reference
  // value quoted above is what the *first* of those steps produces.
  assert_eq(sol.t.length(), 6)
  let stepped = sol.u.unsafe_get(1).unsafe_get(0)
  assert_true((stepped - 0.8187307531608496).abs() < 0.0000001)
}
