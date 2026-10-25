// The solver trait and the reference Euler method.
//
// `Solver` is the single extension point: every method in this port is an
// in-place performer over the shared `Integrator` state, exactly mirroring
// OrdinaryDiffEqCore's `perform_step!(integrator, cache)` split.

///|
/// A one-step method for `u' = f(t, u)`.
///
/// Implementors are stateless descriptors (coefficients, order); all scratch
/// lives in the `Integrator`, which is passed by reference. `solve[A : Solver]`
/// monomorphizes on the algorithm, so there is no per-step dispatch cost.
pub trait Solver {
  /// Classical order of the method (drives the error controller, M2).
  fn order(Self) -> Int32

  /// Whether the method is FSAL: the last stage derivative equals `f` at the
  /// new point, so the integrator reuses it as the next step's first stage.
  fn is_fsal(Self) -> Bool

  /// Number of stage vectors `perform_step` fills in `integ.k`.
  fn stage_count(Self) -> Int32

  /// Advance one step. On entry `integ.t/tprev/uprev/dt/fsalfirst` hold the
  /// current step; on exit `integ.u` is the new state, `integ.k` the stages
  /// (in this solver's interpolant convention) and `integ.est` the error norm
  /// (0 for fixed-step methods).
  fn perform_step(Self, Integrator) -> Unit

  /// Dense-output interpolant used when dense output is enabled.
  fn interpolant(Self) -> SolutionInterp

  /// Fill in any stage derivatives that `perform_step` left out but the
  /// interpolant needs, writing them into the tail of `integ.k`.
  ///
  /// This is Julia's `_ode_addsteps!` (verner_addsteps.jl:655, and the
  /// `addsteps!` call in `interp_at_saveat`, integrator_utils.jl:423): Verner's
  /// 7(6) and 9(8) methods evaluate 10 resp. 16 stages per step but their
  /// interpolants need 16 resp. 20, so the remainder is computed on demand,
  /// just before the step's stages are stored.
  ///
  /// The extra evaluations must go through `Integrator::eval_f_uncounted` —
  /// upstream likewise omits `increment_nf!` there.
  ///
  /// Methods whose `perform_step` already fills every stage (Euler, Tsit5) and
  /// methods that do not store `k` at all implement this as a no-op; there is
  /// no trait default, so every impl says so explicitly.
  fn fill_dense_stages(Self, Integrator) -> Unit

  /// Whether a **fixed-step** solve must re-evaluate `f` at the new point
  /// rather than carrying `fsallast` over as the next `fsalfirst`.
  ///
  /// This is the `only_diagonal_mass_matrix(alg) && !opts.adaptive` branch of
  /// `update_fsal!` (integrator_utils.jl:236): a method with a diagonal mass
  /// matrix only reaches the `fsallast` assignment inside its
  /// `if integrator.opts.adaptive` block, so in a fixed-step solve `fsallast`
  /// is stale and copying it forward would corrupt the next step's first
  /// stage. `reset_fsal!` (integrator_utils.jl:1349) then pays one `f` to
  /// recompute it.
  ///
  /// Only the diagonal-mass-matrix methods (Rosenbrock23 / Rosenbrock32)
  /// answer true; everything else returns false, which reduces the branch to
  /// the plain `copy_into(fsalfirst, fsallast)`.
  fn needs_fsal_reset_fixed_step(Self) -> Bool

  /// Whether this method drives the stepping loop through the BDF
  /// controllers — `stepsize_controller!` (with `choose_order!` and STALD),
  /// `bdf_step_reject_controller!` and `post_newton_controller!` — instead of
  /// the PI controller of control.mbt and the generic step-reject path.
  ///
  /// Upstream dispatches on the cache type (`FBDFCache` under a
  /// `BDFControllerCache`); the port resolves the same decision once per
  /// solve into `Integrator.bdf_active`, which the loop reads. Only FBDF (and
  /// the other multistep BDFs) answer true.
  fn uses_bdf_controllers(Self) -> Bool
}

///|
/// The classic explicit Euler method: `u(t+dt) = u(t) + dt * f(t, u(t))`.
/// Order 1, no error estimate. Used as the reference solver and as the
/// end-to-end test vehicle for the stepping loop.
///
/// Euler carries no data, so it is a field-less struct. This MoonBit version
/// exports field-less structs to other packages as read-only values, which
/// rules out `Euler::{}` literals outside this package — hence the explicit
/// `new` constructor, which every caller (including the tests) should use.
pub struct Euler {}

///|
/// The Euler algorithm value.
pub fn Euler::new() -> Euler {
  Euler::{ }
}

///|
pub impl Solver for Euler with fn order(_self) -> Int32 {
  1
}

///|
pub impl Solver for Euler with fn is_fsal(_self) -> Bool {
  // Julia's `isfsal` defaults to true for OrdinaryDiffEqAlgorithm; Euler's
  // last stage *is* `f` at the new point, which the loop reuses.
  true
}

///|
pub impl Solver for Euler with fn stage_count(_self) -> Int32 {
  2
}

///|
pub impl Solver for Euler with fn perform_step(self, integ) -> Unit {
  euler_perform_step(integ)
}

///|
pub impl Solver for Euler with fn interpolant(_self) -> SolutionInterp {
  SolutionInterp::Hermite
}

///|
pub impl Solver for Euler with fn fill_dense_stages(_self, _integ) -> Unit {
  // Euler's Hermite interpolant only needs the two endpoint derivatives,
  // which `euler_perform_step` already stored in k[0] and k[1].
}

///|
pub impl Solver for Euler with fn needs_fsal_reset_fixed_step(_self) -> Bool {
  false
}

///|
pub impl Solver for Euler with fn uses_bdf_controllers(_self) -> Bool {
  // One-step method: the PI controller drives the step size.
  false
}

///|
/// Promote the trait methods onto `Euler` (this MoonBit version warns about
/// relying on implicit promotion).
pub extend Euler with Solver::{
  order,
  is_fsal,
  stage_count,
  perform_step,
  interpolant,
  fill_dense_stages,
  needs_fsal_reset_fixed_step,
  uses_bdf_controllers,
}

///|
/// Euler's step: `u = uprev + dt * fsalfirst`, then `fsallast = f(u, t+dt)`
/// which doubles as the next step's `fsalfirst` and as the second endpoint
/// derivative of the Hermite interpolant.
fn euler_perform_step(integ : Integrator) -> Unit {
  let dim = integ.u.length()
  copy_into(integ.k.unsafe_get(0), integ.fsalfirst)
  let dt = integ.dt
  let mut i = 0
  while i < dim {
    let v = integ.uprev.unsafe_get(i) + dt * integ.fsalfirst.unsafe_get(i)
    integ.u.unsafe_set(i, v)
    i += 1
  }
  Integrator::eval_f(integ, integ.t + dt, integ.u, integ.fsallast)
  copy_into(integ.k.unsafe_get(1), integ.fsallast)
  integ.est = 0.0
}

///|
test "euler takes one explicit step" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_dt(SolveOptions::new(), 0.1)
  let sol = solve(prob, Euler::new(), opts)
  assert_true(sol.retcode.equal(Retcode::Success))
  // 10 fixed steps of Euler on u' = -u from t=0 to t=1.
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - fastpower(0.9, 10.0)).abs() < 0.000001)
}
