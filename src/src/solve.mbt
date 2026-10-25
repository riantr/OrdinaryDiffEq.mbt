// The solve entry point and the integrator main loop.
//
// Structure mirrors OrdinaryDiffEqCore's `solve.jl` (`solve!` at 1062-1104)
// and `integrator_utils.jl` (`loopheader!` 84, `apply_step!` 175,
// `modify_dt_for_tstops!` 271, `savevalues!` 339, `_loopfooter!` 602,
// `handle_tstop!` 1312, `_postamble!` 496), reduced to pure ODEs:
// no callbacks, no noise, no mass matrix, no DAE, no step/stage limiters.

///|
/// Solve `prob` with algorithm `alg` and options `opts`.
///
/// Stepping is fixed-step when `opts.dt` is given and `opts.adaptive` is off.
/// With `opts.adaptive` on, the embedded error estimate drives `dt` through
/// the PI controller (control.mbt); a `dt` of 0 then selects the initial step
/// automatically (Hairer-Wanner).
pub fn[A : Solver] solve(
  prob : OdeProblem,
  alg : A,
  opts : SolveOptions,
) -> Solution {
  if prob.u0.length() == 0 ||
    prob.tf == prob.t0 ||
    (!opts.adaptive && opts.dt == 0.0) {
    return {
      t: [],
      u: [],
      k: [],
      retcode: Retcode::InitialFailure,
      interp: SolutionInterp::Linear,
      nf: 0,
    }
  }
  let integ = Integrator::new(prob, opts, alg.stage_count(), alg.order())
  // Upstream dispatches the stepping controllers on the cache type
  // (`FBDFCache` under a `BDFControllerCache`); the port resolves the same
  // decision once, here, and the loop reads `bdf_active`.
  integ.bdf_active = alg.uses_bdf_controllers()
  integ.is_fsal = alg.is_fsal()
  integ.sol.interp = if opts.dense {
    alg.interpolant()
  } else {
    SolutionInterp::Linear
  }
  resolve_options(integ, prob)
  integ.tstops = build_tstops(opts.tstops, prob.t0, prob.tf, integ.tdir)
  integ.saveat = build_saveat(opts.saveat, prob.t0, prob.tf, integ.tdir)

  // `handle_dt!` (solve.jl:1108): an adaptive solve with `dt == 0` picks the
  // initial step automatically. Upstream runs this inside `init!`, after the
  // algorithm's `initialize!`; this port runs it before the pre-start FSAL,
  // which keeps the evaluation count identical (2 here + 1 there).
  if opts.adaptive && integ.dt == 0.0 {
    integ.dt = determine_initdt(integ, alg.order())
    integ.dtpropose = integ.dt
  }

  // Pre-start FSAL: k[0] = f(t0, u0) (Julia's `initialize!`).
  Integrator::eval_f(integ, integ.t, integ.u, integ.fsalfirst)
  copy_into(integ.k.unsafe_get(0), integ.fsalfirst)

  // Initial save (integrator_interface.jl `reinit!` 608).
  if integ.opts.save_start {
    integ.saveiter += 1
    Solution::push(integ.sol, integ.t, copy_array(integ.u))
    if integ.opts.dense {
      Solution::push_ks(integ.sol, integ.k)
    }
  }

  // Main loop (solve.jl:1062).
  while Integrator::has_tstop(integ) {
    let first_tstop = Integrator::first_tstop(integ)
    while integ.tdir * integ.t < first_tstop {
      loopheader(integ, alg)
      if check_error(integ) != Retcode::Success {
        // `de_check_error!` (DiffEqBase check_error.jl:149) finalizes the
        // solution on failure: the code is already stored in `sol.retcode`,
        // and the postamble still saves the current point when `save_end`
        // asks for it (it is a no-op whenever the last accepted step was
        // saved, which is the common case).
        postamble(integ, alg)
        integ.sol.nf = integ.nf
        return integ.sol
      }
      if integ.next_step_tstop && integ.dt.abs() < eps * integ.t.abs() {
        // handle_tstop_step!: the step is too short to be worth taking.
        integ.accept_step = true
      } else {
        alg.perform_step(integ)
      }
      let should_exit = integ.next_step_tstop
      loopfooter(integ, alg)
      if !Integrator::has_tstop(integ) || should_exit {
        break
      }
    }
    handle_tstop(integ)
  }
  postamble(integ, alg)
  integ.sol.nf = integ.nf
  if integ.sol.retcode != Retcode::Success {
    return integ.sol
  }
  integ.sol.retcode = Retcode::Success
  integ.sol
}

// ---------------------------------------------------------------------------
// Option resolution
// ---------------------------------------------------------------------------

///|
/// Apply the span- and saveat-dependent option defaults.
///
/// Span part (DiffEqBase solve.jl:299-300): `dtmax` defaults to `|tf - t0|`
/// and `dtmin` to 0; a stored 0 means "unset".
///
/// Saveat part (solve.jl:285-295): a non-empty `saveat` switches off
/// `save_everystep`, derives `save_start`/`save_end` from membership, and
/// defaults `dense` off. The solver's interpolant deliberately stays in
/// `integ.sol.interp`: Julia's `interp_at_saveat` (integrator_utils.jl:423)
/// evaluates mid-step saveat points with the algorithm's polynomial no matter
/// what `dense` says — `dense` only controls k-array storage, and without
/// stored k's `Solution::call` falls back to linear through its `has_ks`
/// guard, exactly like Julia's post-solve interpolation.
fn resolve_options(integ : Integrator, prob : OdeProblem) -> Unit {
  let ropts = integ.opts
  if ropts.dtmax == 0.0 {
    ropts.dtmax = (prob.tf - prob.t0).abs()
  }
  if ropts.saveat.length() == 0 {
    integ.opts = ropts
    return
  }
  ropts.save_everystep = false
  ropts.save_start = contains_time(ropts.saveat, prob.t0)
  ropts.save_end = contains_time(ropts.saveat, prob.tf)
  ropts.dense = false
  integ.opts = ropts
}

///|
/// Exact membership test for a small list of times.
fn contains_time(times : Array[Double], t : Double) -> Bool {
  let mut i = 0
  while i < times.length() {
    if times.unsafe_get(i) == t {
      return true
    }
    i += 1
  }
  false
}

// ---------------------------------------------------------------------------
// tstop / saveat queues (solve.jl `initialize_tstops` 1179, `initialize_saveat` 1261)
// ---------------------------------------------------------------------------

///|
/// tdir-scaled stop times strictly inside `(t0, tf)`, ascending, with the
/// final time appended exactly once.
fn build_tstops(
  tstops : Array[Double],
  t0 : Double,
  tf : Double,
  tdir : Double,
) -> Array[Double] {
  let tdir_t0 = tdir * t0
  let tdir_tf = tdir * tf
  let out : Array[Double] = []
  for t in tstops {
    let tt = tdir * t
    if tdir_t0 < tt && tt < tdir_tf {
      out.push(tt)
    }
  }
  out.push(tdir_tf)
  out.sort()
  out
}

///|
/// tdir-scaled output times strictly after `t0` and at or before `tf`,
/// ascending.
fn build_saveat(
  saveat : Array[Double],
  t0 : Double,
  tf : Double,
  tdir : Double,
) -> Array[Double] {
  let tdir_t0 = tdir * t0
  let tdir_tf = tdir * tf
  let out : Array[Double] = []
  for t in saveat {
    let tt = tdir * t
    if tdir_t0 < tt && tt <= tdir_tf {
      out.push(tt)
    }
  }
  out.sort()
  out
}

// ---------------------------------------------------------------------------
// Loop stages
// ---------------------------------------------------------------------------

///|
/// `loopheader!` (integrator_utils.jl:84): accept or reject the previous
/// step, then fix `dt` at the bounds and shorten it for the next tstop.
///
/// The upstream accept test is `!force_stepfail && (!adaptive || accept_step
/// || isaposteriori)`; this port has no a posteriori methods, so that reduces
/// to `!force_stepfail && (!adaptive || accept_step)`. `force_stepfail` is
/// the post-Newton signal (only FBDF raises it): the step is treated as
/// rejected no matter what the error test said, and the flag is consumed by
/// exactly one decision — upstream clears it on the last line of
/// `loopheader!`, after `fix_dt_at_bounds!` and `modify_dt_for_tstops!`.
fn[A : Solver] loopheader(integ : Integrator, alg : A) -> Unit {
  if integ.iter > 0 {
    if !integ.force_stepfail && (!integ.opts.adaptive || integ.accept_step) {
      // ACCEPT
      integ.success_iter += 1
      apply_step(integ, alg)
    } else if integ.force_stepfail ||
      (integ.opts.adaptive && !integ.accept_step) {
      // REJECT
      handle_step_rejection(integ)
    }
  }
  integ.iter += 1
  fix_dt_at_bounds(integ)
  modify_dt_for_tstops(integ)
  integ.force_stepfail = false
}

///|
/// `handle_step_rejection!` (integrator_utils.jl:118): the pure-ODE path —
/// `isout` is always false (no `isoutofdomain` hook) and there is no noise
/// process to reject, so the step-size controller is the whole body. The
/// `dtmax`/`dtmin` clamp and tstop shortening happen right after, back in
/// `loopheader!`.
///
/// Upstream guards the controller with `elseif !integrator.force_stepfail`:
/// when the step failed inside the Newton solve, the post-Newton controller
/// (in the loopfooter) has already shrunk `dt`, so the error-driven reject
/// controller must not run a second time. Which controller runs is upstream's
/// `step_reject_controller!(integrator, alg)` dispatch — the BDF variant for
/// FBDF, the PI one for everything else.
fn handle_step_rejection(integ : Integrator) -> Unit {
  if !integ.force_stepfail {
    if integ.bdf_active {
      bdf_step_reject_controller(integ, integ.bdf)
    } else {
      step_reject_controller(integ)
    }
  }
}

///|
/// `apply_step!` (integrator_utils.jl:175): commit an accepted step.
fn[A : Solver] apply_step(integ : Integrator, alg : A) -> Unit {
  // update_uprev!
  copy_into(integ.uprev, integ.u)
  // dtchangeable is always true here, so dt follows the proposal.
  integ.dt = integ.dtpropose
  update_fsal(integ, alg)
  modify_dt_for_tstops(integ)
}

///|
/// `update_fsal!` (integrator_utils.jl:218), reduced to the two branches a pure
/// ODE with fixed or adaptive stepping can reach.
///
/// The port keeps two of upstream's four guards:
///
///   * `only_diagonal_mass_matrix(alg) && !opts.adaptive` forces
///     `reset_fsal!` — a diagonal-mass-matrix method only assigns `fsallast`
///     inside its `if opts.adaptive` block, so a fixed-step solve would
///     otherwise carry a stale value into the next step's first stage.
///   * otherwise a FSAL method copies `fsallast` over `fsalfirst`.
///
/// `has_discontinuity` is false and `isdp8` does not apply, so the
/// discontinuity and `!opts.calck` guards are dropped; `calck` is always true
/// here.
fn[A : Solver] update_fsal(integ : Integrator, alg : A) -> Unit {
  if !integ.is_fsal {
    return
  }
  if !integ.opts.adaptive && alg.needs_fsal_reset_fixed_step() {
    // `reset_fsal!` (integrator_utils.jl:1349): recompute rather than copy.
    Integrator::eval_f(integ, integ.t, integ.u, integ.fsalfirst)
  } else {
    copy_into(integ.fsalfirst, integ.fsallast)
  }
}

///|
/// `fix_dt_at_bounds!` (integrator_utils.jl:1265): clamp `dt` into
/// `[dtmin, dtmax]` in the integration direction.
fn fix_dt_at_bounds(integ : Integrator) -> Unit {
  let dtmax = integ.opts.dtmax.abs()
  let dtmin = Integrator::dtmin(integ)
  if integ.tdir > 0.0 {
    integ.dt = dmin(dtmax, integ.dt)
    integ.dt = dmax(dtmin, integ.dt)
  } else {
    integ.dt = dmax(-dtmax, integ.dt)
    integ.dt = dmin(-dtmin, integ.dt)
  }
}

///|
/// `modify_dt_for_tstops!` (integrator_utils.jl:271): shorten `dt` so the
/// step lands exactly on the next tstop, using `dtcache` (the unclipped step
/// size) to restore the original step afterwards.
fn modify_dt_for_tstops(integ : Integrator) -> Unit {
  if !Integrator::has_tstop(integ) {
    integ.next_step_tstop = false
    return
  }
  let tdir_t = integ.tdir * integ.t
  let tdir_tstop = Integrator::first_tstop(integ)
  let distance = (tdir_tstop - tdir_t).abs()
  // Floating-point slack so a dt that nominally matches the distance still
  // triggers the tstop branch.
  let tol = 100.0 * eps * dmax(integ.t.abs(), tdir_tstop.abs())
  if integ.opts.adaptive {
    // Adaptive stepping lands in M2; until then the fixed-step branches below
    // are the only reachable ones.
    let original_dt = integ.dt.abs()
    integ.dtpropose = integ.tdir * original_dt
    if original_dt + tol < distance {
      integ.next_step_tstop = false
    } else {
      integ.next_step_tstop = true
      integ.tstop_target = integ.tdir * tdir_tstop
    }
    integ.dt = integ.tdir * dmin(original_dt, distance)
  } else if integ.dtcache == 0.0 {
    // No fixed step given: step straight to the tstop.
    integ.dt = integ.tdir * distance
    integ.next_step_tstop = true
    integ.tstop_target = integ.tdir * tdir_tstop
  } else if integ.dtcache.abs() + tol < distance {
    integ.next_step_tstop = false
  } else {
    integ.next_step_tstop = true
    integ.tstop_target = integ.tdir * tdir_tstop
    integ.dt = integ.tdir * dmin(integ.dtcache.abs(), distance)
  }
}

///|
/// `fixed_t_for_tstop_error!` (integrator_utils.jl:1032): when the step was
/// aimed at a tstop, snap the new time to the tstop exactly instead of
/// accumulating rounding error.
fn fixed_t_for_tstop_error(integ : Integrator, ttmp : Double) -> Double {
  if integ.next_step_tstop {
    integ.next_step_tstop = false
    integ.tstop_target
  } else {
    ttmp
  }
}

///|
/// `_loopfooter!` (integrator_utils.jl:602): the adaptive branch runs the
/// step-size controller, decides acceptance, commits time and computes the
/// next proposal; the non-adaptive branch just advances time. Both then save
/// (upstream from `handle_callbacks!`); a rejected step saves nothing.
///
/// Three upstream branches, in its order:
///
///   1. `force_stepfail` — the step failed inside the nonlinear solve. `t` is
///      **not** advanced and nothing is saved. In adaptive mode
///      `handle_force_stepfail!` runs first (`post_newton_controller!` for a
///      BDF: halve `dt`, maybe drop the order; the generic fallback just
///      halves `dt`); in fixed-step mode a second consecutive failure returns
///      early, before any bookkeeping. Then `last_stepfail = true` and
///      `accept_step = false`, unconditionally.
///   2. adaptive — `stepsize_controller!` → `accept_step_controller!` →
///      `step_accept_controller!`. FBDF answers all three with its own
///      CVODE-style versions (`fbdf_stepsize_controller` with `choose_order!`
///      and STALD, and `fbdf_step_accept_controller`), selected by
///      `integ.bdf_active`.
///   3. fixed — advance time, accept.
fn[A : Solver] loopfooter(integ : Integrator, alg : A) -> Unit {
  let ttmp = integ.t + integ.dt
  let mut do_save = true
  if integ.force_stepfail {
    if integ.opts.adaptive {
      if integ.bdf_active {
        fbdf_post_newton_controller(integ, integ.bdf)
      } else {
        // The generic `post_newton_controller!`: halve `dt` and retry. No
        // solver other than FBDF raises `force_stepfail` in this port, so
        // this arm is the documented fallback, not a live path.
        integ.dt = integ.dt / 2.0
      }
    } else if integ.last_stepfail {
      return
    }
    integ.last_stepfail = true
    integ.accept_step = false
    do_save = false
  } else if integ.opts.adaptive {
    let q = if integ.bdf_active {
      fbdf_stepsize_controller(integ, integ.bdf)
    } else {
      stepsize_controller(integ)
    }
    integ.accept_step = accept_step_controller(integ)
    if integ.accept_step {
      // Accept
      integ.last_stepfail = false
      integ.tprev = integ.t
      if integ.next_step_tstop {
        // The step controller's `dt` is the (shortened) distance to the
        // tstop; restore the unclipped proposal so the next step is sized
        // from the controller's own scale (integrator_utils.jl:632).
        integ.dt = integ.dtpropose
      }
      integ.t = fixed_t_for_tstop_error(integ, ttmp)
      let dtnew = if integ.bdf_active {
        fbdf_step_accept_controller(integ, integ.bdf, q)
      } else {
        step_accept_controller(integ, q)
      }
      calc_dt_propose(integ, dtnew)
    } else {
      // Reject: `handle_step_rejection` runs at the next `loopheader!`.
      do_save = false
    }
  } else {
    integ.tprev = integ.t
    integ.t = fixed_t_for_tstop_error(integ, ttmp)
    integ.last_stepfail = false
    integ.accept_step = true
    integ.dtpropose = integ.dt
  }
  if do_save {
    savevalues(integ, alg)
  }
}

///|
/// `check_error!` (DiffEqBase check_error.jl:91, via `de_check_error!` 149):
/// the fixed-step subset — iteration cap and non-finite state — plus the
/// adaptive branches: the step size collapsing onto `dtmin`, and a rejected
/// step shrinking to the ULP of `t`. Upstream's `unstable_check` hook
/// defaults to false, and its `tstop` exception reads the *user*
/// `opts.tstops` list, which this port keeps as `opts.tstops` (the internal
/// queue lives on the integrator). A non-success code is stored in
/// `sol.retcode`; running the postamble on failure is the caller's job
/// (upstream's `de_check_error!` does it right after this returns).
fn check_error(integ : Integrator) -> Retcode {
  if integ.iter > integ.opts.maxiters {
    integ.sol.retcode = Retcode::MaxIters
    Retcode::MaxIters
  } else if !is_finite(integ.t) || !state_is_finite(integ.u) {
    integ.sol.retcode = Retcode::Unstable
    Retcode::Unstable
  } else if integ.opts.adaptive {
    let step_accepted = integ.accept_step
    // `t + dt` compared against the next pending tstop in physical time; a
    // step that lands exactly on a tstop is exempt from the dtmin abort.
    let next_tstop = if Integrator::has_tstop(integ) {
      integ.tdir * Integrator::first_tstop(integ)
    } else {
      integ.tf
    }
    if integ.dt.abs() <= integ.opts.dtmin.abs() &&
      integ.opts.dtmin > 0.0 &&
      (!step_accepted || integ.t + integ.dt < next_tstop) {
      integ.sol.retcode = Retcode::DtMinUnstable
      Retcode::DtMinUnstable
    } else if !step_accepted && integ.dt.abs() <= eps * integ.t.abs() {
      integ.sol.retcode = Retcode::Unstable
      Retcode::Unstable
    } else {
      Retcode::Success
    }
  } else {
    Retcode::Success
  }
}

///|
/// True when every component of `u` is finite.
fn state_is_finite(u : Array[Double]) -> Bool {
  let mut i = 0
  while i < u.length() {
    if !is_finite(u.unsafe_get(i)) {
      return false
    }
    i += 1
  }
  true
}

///|
/// `handle_tstop!` (integrator_utils.jl:1312): consume the tstop that was
/// just reached, including duplicates.
fn handle_tstop(integ : Integrator) -> Unit {
  if !Integrator::has_tstop(integ) {
    return
  }
  let tdir_t = integ.tdir * integ.t
  let mut tdir_tstop = Integrator::first_tstop(integ)
  if tdir_t == tdir_tstop {
    while tdir_t == tdir_tstop {
      Integrator::pop_tstop(integ)
      if !Integrator::has_tstop(integ) {
        break
      }
      tdir_tstop = Integrator::first_tstop(integ)
    }
    integ.just_hit_tstop = true
  } else if tdir_t > tdir_tstop {
    // `dtchangeable` is always true here, so upstream treats this as an
    // invariant violation; consume the stop defensively and keep going.
    Integrator::pop_tstop(integ)
    integ.just_hit_tstop = true
  }
}

///|
/// `_savevalues!` (integrator_utils.jl:343): consume due saveat points
/// (interpolating inside the current step) and, when saving every step, the
/// step endpoint together with its stage array.
fn[A : Solver] savevalues(integ : Integrator, alg : A) -> Unit {
  let tdir_t = integ.tdir * integ.t
  while integ.saveat_idx < integ.saveat.length() &&
        integ.saveat.unsafe_get(integ.saveat_idx) <= tdir_t {
    let curt = integ.saveat.unsafe_get(integ.saveat_idx)
    integ.saveat_idx += 1
    if curt == tdir_t && curt == integ.tdir * integ.tf && !integ.opts.save_end {
      // `skip_saveat_at_tspan_end` (integrator_utils.jl:451): a saveat point
      // that lands exactly on `tf` is skipped when `save_end` is off — the
      // endpoint is then nobody's responsibility, matching upstream.
      continue
    }
    integ.saveiter += 1
    if curt != tdir_t {
      let theta = (integ.tdir * curt - integ.tprev) / integ.dt
      let out = Array::make(integ.u.length(), 0.0)
      // `interp_at_saveat` (integrator_utils.jl:423) calls `addsteps!` before
      // evaluating, so a mid-step saveat point sees the complete stage array.
      alg.fill_dense_stages(integ)
      interp_current_step(integ, theta, out)
      Solution::push(integ.sol, integ.tdir * curt, out)
    } else {
      save_point(integ, alg)
    }
  }
  if integ.opts.save_everystep {
    let n = integ.sol.t.length()
    let last_is_here = n > 0 && integ.sol.t.unsafe_get(n - 1) == integ.t
    let end_guard = integ.opts.save_end || integ.t != integ.tf
    if n == 0 || ((!last_is_here || integ.dt == 0.0) && end_guard) {
      save_point(integ, alg)
    }
  }
}

///|
/// Save the current endpoint `(t, u)` plus, when dense output is on, the
/// stage array of the step that just ended.
///
/// `[A : Solver]` so the algorithm is available for the `addsteps!` hook;
/// `save_point` is only ever reached from `solve[A : Solver]`, so the
/// monomorphization is shared with the rest of the loop.
fn[A : Solver] save_point(integ : Integrator, alg : A) -> Unit {
  integ.saveiter += 1
  Solution::push(integ.sol, integ.t, copy_array(integ.u))
  if integ.opts.dense {
    // `addsteps!` (integrator_utils.jl:423 / generic_dense.jl:1430): complete
    // the stage array before it is stored, so a later `Solution::call`
    // between these two points sees every stage the interpolant needs.
    alg.fill_dense_stages(integ)
    Solution::push_ks(integ.sol, integ.k)
  }
}

///|
/// Evaluate the solver's dense interpolant inside the current step at
/// fraction `theta` (upstream `interp_at_saveat`, which first calls
/// `addsteps!` to guarantee the endpoint derivatives are present).
fn interp_current_step(
  integ : Integrator,
  theta : Double,
  out : Array[Double],
) -> Unit {
  match integ.sol.interp {
    Linear => linear_interp(integ.uprev, integ.u, theta, out)
    Hermite =>
      hermite_interpolate(
        integ.uprev,
        integ.u,
        integ.k.unsafe_get(0),
        integ.k.unsafe_get(1),
        theta,
        integ.dt,
        out,
      )
    Poly(f, _) => f(theta, integ.dt, integ.uprev, integ.u, integ.k, out)
  }
}

///|
/// `_postamble!` (integrator_utils.jl:496): save the final point if it was
/// not stepped on exactly.
fn[A : Solver] postamble(integ : Integrator, alg : A) -> Unit {
  if !integ.opts.save_end {
    return
  }
  let n = integ.sol.t.length()
  let last_is_here = n > 0 && integ.sol.t.unsafe_get(n - 1) == integ.t
  if n == 0 || !last_is_here {
    save_point(integ, alg)
  }
}
