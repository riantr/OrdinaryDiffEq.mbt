// The integrator: all mutable state shared by the solve loop and the solvers.
//
// Direct counterpart of OrdinaryDiffEqCore's `ODEIntegrator`, reduced to what
// a pure ODE with fixed or adaptive stepping needs (no noise, no mass matrix,
// no DAE, no callbacks).

///|
/// Integration state. One instance per solve; solvers receive it by reference
/// through `Solver::perform_step` and read/write its fields directly.
///
/// All fields are `mut`: this MoonBit version makes record fields immutable by
/// default, and the stepping loop (plus every solver) updates them in place.
pub struct Integrator {
  /// Current time: the end of the last accepted step.
  pub mut t : Double
  /// Time at the start of the current step.
  pub mut tprev : Double
  /// Integration direction: `+1` forward, `-1` backward.
  pub tdir : Double
  /// Final time of the time span.
  pub tf : Double
  /// Current (signed) step size.
  pub mut dt : Double
  /// Step size proposed for the next step (signed).
  pub mut dtpropose : Double
  /// Original unclipped step size; `modify_dt_for_tstops` restores from it
  /// after a step was shortened to land on a tstop.
  pub mut dtcache : Double
  /// State at the end of the current step (written by `perform_step`).
  pub mut u : Array[Double]
  /// State at the start of the current step.
  pub mut uprev : Array[Double]
  /// Right-hand side of the problem.
  pub f : OdeFunction
  /// Derivative at the start of the current step (`fsalfirst`).
  pub mut fsalfirst : Array[Double]
  /// Derivative at the end of the current step (`fsallast`); for FSAL
  /// methods this becomes the next step's `fsalfirst`.
  pub mut fsallast : Array[Double]
  /// Stage derivatives of the current step, in the convention the solver's
  /// interpolant expects. Length is the solver's `stage_count`.
  pub mut k : Array[Array[Double]]
  /// Scratch vector for stage states (mirrors the `tmp` field of Julia's
  /// per-algorithm caches); solvers may use it freely between `eval_f` calls.
  pub mut tmp : Array[Double]
  /// Embedded-error scratch: `utilde = dt * sum_i b~_i * k_i` (the `utilde`
  /// field of Julia's `Tsit5ConstantCache`), the difference between the two
  /// embedded solutions before tolerance scaling.
  pub mut utilde : Array[Double]
  /// Tolerance-scaled error residual (Julia's `atmp`); `est` is its norm.
  pub mut atmp : Array[Double]
  /// Stiff-method scratch: the dense Jacobian, the factorization of
  /// `W = I/(dt*gamma) - J`, and the cached `∂f/∂t` (Julia's per-algorithm
  /// `RosenbrockCache.J` / `.W` / `.dT`). Empty until a stiff method calls
  /// `StiffCache::ensure`, so explicit-only solves never allocate the `n*n`
  /// buffers.
  pub mut stiff : StiffCache
  /// Multistep (BDF) state: the history queue, the order bookkeeping and the
  /// nonlinear-solver state. Empty until a multistep method calls
  /// `FBDFCache::ensure`, and `bdf_active` is what routes the controller hooks
  /// in control.mbt to the BDF variants instead of the PI controller.
  pub mut bdf : FBDFCache
  /// Set by a multistep method's `perform_step` and read by the stepping loop
  /// to pick the BDF controllers (`stepsize_controller!` / `choose_order!`,
  /// `bdf_step_reject_controller!`, `post_newton_controller!`) over the PI
  /// ones. Upstream dispatches on the cache type instead; the flag is the same
  /// decision, resolved once per solve.
  pub mut bdf_active : Bool
  /// Set by an implicit method's `perform_step` when its Newton solve did not
  /// converge. `_loopfooter!` reads it to run `post_newton_controller!` and
  /// mark the step rejected (integrator_utils.jl:606-616); it is false for
  /// every explicit method, so the branch is dead code there.
  pub mut force_stepfail : Bool
  /// Per-step stage scratch for stiff methods: Rosenbrock stages need several
  /// simultaneous `n`-vectors beyond what `tmp`/`utilde`/`atmp` provide
  /// (Julia's cache `k₁`/`k₂`/`k₃`/`f₁`/`du1`/`du2`). Sized `dim` here and
  /// grown on demand by `Integrator::stiff_vecs`.
  pub mut stiff_bufs : Array[Array[Double]]
  /// Error-estimate norm of the current step (0 while fixed-step).
  pub mut est : Double
  /// PI step-size controller: knobs plus per-solve state (Julia's
  /// `controller_cache`; built from the algorithm's order).
  pub mut controller : PIController
  /// Consecutive accepted steps; `0` until the first step is accepted. The
  /// controller reads it to pick `qmax_first_step` over `qmax`
  /// (`get_current_qmax`, controllers.jl:288).
  pub mut success_iter : Int32
  /// Whether the method reuses `fsallast` as the next `fsalfirst`.
  pub mut is_fsal : Bool
  pub mut iter : Int32
  /// Number of right-hand-side evaluations so far.
  pub mut nf : Int32
  pub mut accept_step : Bool
  pub mut last_stepfail : Bool
  /// True right after a tstop was hit exactly.
  pub mut just_hit_tstop : Bool
  /// True when the current step was shortened to land on a tstop.
  pub mut next_step_tstop : Bool
  /// The tstop the current step is aiming at (physical time).
  pub mut tstop_target : Double
  pub mut opts : SolveOptions
  /// tdir-scaled stop times, ascending, including the final time; consumed
  /// through `tstop_idx`.
  pub mut tstops : Array[Double]
  pub mut tstop_idx : Int32
  /// tdir-scaled output times, ascending; consumed through `saveat_idx`.
  pub mut saveat : Array[Double]
  pub mut saveat_idx : Int32
  /// Number of points saved so far.
  pub mut saveiter : Int32
  pub mut sol : Solution
}

///|
/// Build the integrator for `prob` with `opts`, preallocating `stage_count`
/// stage vectors and the PI controller for an algorithm of classical order
/// `order`. The tstops/saveat queues are empty here; `solve` fills them once
/// the direction is known.
pub fn Integrator::new(
  prob : OdeProblem,
  opts : SolveOptions,
  stage_count : Int32,
  order : Int32,
) -> Integrator {
  let dim = prob.u0.length()
  let tdir = if prob.tf >= prob.t0 { 1.0 } else { -1.0 }
  let dt = tdir * opts.dt.abs()
  let k = Array::make(stage_count, Array::make(0, 0.0))
  let mut i = 0
  while i < stage_count {
    k.unsafe_set(i, Array::make(dim, 0.0))
    i += 1
  }
  {
    t: prob.t0,
    tprev: prob.t0,
    tdir,
    tf: prob.tf,
    dt,
    dtpropose: dt,
    dtcache: dt,
    u: copy_array(prob.u0),
    uprev: copy_array(prob.u0),
    f: prob.f,
    fsalfirst: Array::make(dim, 0.0),
    fsallast: Array::make(dim, 0.0),
    k,
    tmp: Array::make(dim, 0.0),
    utilde: Array::make(dim, 0.0),
    atmp: Array::make(dim, 0.0),
    est: 0.0,
    stiff: StiffCache::new(),
    bdf: FBDFCache::new(),
    bdf_active: false,
    force_stepfail: false,
    stiff_bufs: [],
    controller: PIController::for_order(order),
    success_iter: 0,
    is_fsal: false,
    iter: 0,
    nf: 0,
    accept_step: true,
    last_stepfail: false,
    just_hit_tstop: false,
    next_step_tstop: false,
    tstop_target: 0.0,
    opts,
    tstops: [],
    tstop_idx: 0,
    saveat: [],
    saveat_idx: 0,
    saveiter: 0,
    sol: {
      t: [],
      u: [],
      k: [],
      retcode: Retcode::Success,
      interp: SolutionInterp::Linear,
      nf: 0,
    },
  }
}

///|
/// Evaluate the right-hand side into `out` and count the evaluation.
pub fn Integrator::eval_f(
  integ : Integrator,
  t : Double,
  u : Array[Double],
  out : Array[Double],
) -> Unit {
  OdeFunction::eval_into(integ.f, t, u, out)
  integ.nf += 1
}

///|
/// Evaluate the right-hand side into `out` *without* counting the evaluation.
///
/// Julia's Verner-method `addsteps!` (verner_addsteps.jl:655) fills the extra
/// stages k11..k16 that the dense interpolant needs but the step itself does
/// not use, and deliberately does not call `increment_nf!`: those stages cost
/// nothing as far as the reported evaluation count (and the user's method
/// selection / `maxiters` budget) is concerned. This is the same function,
/// minus the counter.
pub fn Integrator::eval_f_uncounted(
  integ : Integrator,
  t : Double,
  u : Array[Double],
  out : Array[Double],
) -> Unit {
  OdeFunction::eval_into(integ.f, t, u, out)
}

///|
/// The time-dependent minimum step: `max(dtmin, eps * |t|)` (Julia's
/// `timedepentdtmin`).
pub fn Integrator::dtmin(integ : Integrator) -> Double {
  dmax(integ.opts.dtmin, eps * integ.t.abs())
}

///|
/// The `i`-th pooled stiff-method scratch vector, growing the pool on demand.
///
/// A Rosenbrock step needs several `n`-vectors live at once (the stage
/// derivatives `k₁`/`k₂`/`k₃`, the stage derivative `f₁`, the finite-
/// difference perturbation buffers). Upstream these are named fields of the
/// per-algorithm cache; here they are pooled on the integrator so that adding
/// a stiff method does not mean extending `Integrator` again. Explicit
/// methods never call this, so their solves allocate nothing.
pub fn Integrator::stiff_vec(self : Integrator, i : Int32) -> Array[Double] {
  let mut n = self.stiff_bufs.length()
  while n <= i {
    self.stiff_bufs.push(Array::make(self.u.length(), 0.0))
    n += 1
  }
  self.stiff_bufs.unsafe_get(i)
}

///|
/// The next pending tstop (tdir-scaled), if any.
pub fn Integrator::has_tstop(integ : Integrator) -> Bool {
  integ.tstop_idx < integ.tstops.length()
}

///|
/// The next pending tstop (tdir-scaled). Only valid when `has_tstop`.
pub fn Integrator::first_tstop(integ : Integrator) -> Double {
  integ.tstops.unsafe_get(integ.tstop_idx)
}

///|
/// Consume the next tstop.
pub fn Integrator::pop_tstop(integ : Integrator) -> Unit {
  integ.tstop_idx += 1
}
