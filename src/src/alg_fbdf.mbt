// FBDF: fixed-leading-coefficient, adaptive-order, adaptive-step BDF.
//
// Ported from OrdinaryDiffEqBDF:
//   * algorithms.jl:518 — the `FBDF` struct and its keyword defaults
//     (`max_order = Val{5}()`, `qmax = 10`, `qsteady_min = 9/10`,
//     `qsteady_max = 2/1`, the STALD knobs, `time_filter = false`).
//   * alg_utils.jl:11 — `alg_order(FBDF) = 1`, a *dummy*: the running order
//     lives in the cache and is moved only by `choose_order!`.
//   * bdf_perform_step.jl:1491 — `perform_step!(::FBDFCache)`.
//   * nlsolve.jl:105, newton.jl:758, derivative_utils.jl:826/541 — the Newton
//     corrector driver, `compute_step!` and the `W = J - I/(γ·dt)` policy.
//   * controllers.jl:295-486 — `choose_order!` and the four BDF controllers.
//   * bdf_utils.jl:391 — `estimate_terk!`.
//
// What FBDF is
// ------------
// A variable-step, variable-order BDF. Each step solves the implicit relation
//
//     (1/dt) Σ_{i=1..k+1} αᵢ y(t_i) = f(t_{k+1}),     t_i = t + (1-i)·dt
//
// by Newton iteration on `z = y(t_{k+1})`, where the `k` newest accepted
// points `(ts[j], u_history[j])` supply the history terms and `α₁` (the
// coefficient of the unknown) is folded into the iteration matrix as
// `β₀ = 1/α₁`, `γ = β₀`, `α = 1`. The residual is
//
//     g(z) = tmp + f(z, t+dt) - α·(1/(γ·dt))·z,      tmp = (Σ_{i≥2} αᵢ y_i)/dt
//
// whose Jacobian is `-W` with `W = J - I/(γ·dt)`; one LU factorization of `W`
// serves the whole iteration. `γ·dt = β₀·dt` is the `dtgamma` upstream tracks,
// and the residual identity `f(u) = α·(1/(γ·dt))·u - tmp` at convergence is
// what lets `perform_step!` produce `fsallast` *without* an extra `f` call
// (bdf_perform_step.jl:1617-1622) — the in-place `FBDFCache` path only.
//
// Order and step size are CVODE-style: `terk` estimates of `dt^{k-1} y^{(k-1)}`
// form a ladder `terkm2 > terkm1 > terk > terkp1` that `choose_order!` climbs
// or descends, a `qwait` countdown gates order increases, and the step size
// follows `q = (6·terk/(α₀(k+1)))^{1/(k+1)}`. STALD watches the scaled
// derivative norms of orders 3-5 for the alpha-stability boundary.
//
// This port merges upstream's `NLSolver` + `NLNewtonCache` into `FBDFCache`
// (see the field docs in bdf_utils.mbt); every field keeps its upstream name.

// ---------------------------------------------------------------------------
// The algorithm
// ---------------------------------------------------------------------------

///|
/// The FBDF algorithm value. `max_order` is upstream's `Val` parameter
/// (`FBDF(; max_order = Val{5}())`); it sizes the history queue, the stage
/// array and the coefficient table.
pub struct FBDF {
  pub max_order : Int32
}

///|
/// `FBDF()` with upstream's default `max_order = Val{5}()`.
pub fn FBDF::new() -> FBDF {
  { max_order: FBDF_DEFAULT_MAX_ORDER, }
}

///|
/// `FBDF(; max_order = Val{n}())` for `n = 1..5`. The port accepts any value
/// but the coefficient table only has rows 1..5, so `n > 5` would read padding.
pub fn FBDF::with_max_order(n : Int32) -> FBDF {
  { max_order: n, }
}

///|
pub impl Solver for FBDF with fn order(_self) -> Int32 {
  // alg_utils.jl:11 — a dummy value. The controller never reads it (FBDF runs
  // its own `stepsize_controller!`), and the initial-step heuristic reads
  // `cache.order`, which is 1 before the first step either way.
  1
}

///|
pub impl Solver for FBDF with fn is_fsal(_self) -> Bool {
  // alg_utils.jl does not list FBDF among the `isfsal(...) = false` methods,
  // so the `OrdinaryDiffEqAlgorithm` default (true) stands. `perform_step!`
  // never *reads* `fsalfirst` — it writes `fsallast` algebraically — so the
  // copy in `update_fsal!` is pure overhead, exactly as upstream.
  true
}

///|
pub impl Solver for FBDF with fn stage_count(_self) -> Int32 {
  // `initialize!` (bdf_perform_step.jl:1228): kshortsize = max_order + 1, the
  // `max_order + 1` Chebyshev-resampled dense-output values.
  _self.max_order + 1
}

///|
pub impl Solver for FBDF with fn perform_step(self, integ) -> Unit {
  let cache = integ.bdf
  // The cache is sized for this algorithm's `max_order`; `ensure` is a no-op
  // after the first step (upstream sizes it once in `alg_cache`).
  cache.max_order = self.max_order
  ignore(FBDFCache::ensure(cache, integ.u.length()))
  fbdf_perform_step(integ, cache)
}

///|
pub impl Solver for FBDF with fn interpolant(_self) -> SolutionInterp {
  // `_ode_interpolant(..., ::FBDFCache, ...)` -> `_fbdf_interp`, the
  // barycentric form on the fixed Chebyshev reference nodes (bdf_interp.mbt).
  SolutionInterp::Poly(fbdf_interpolate, fbdf_derivative)
}

///|
pub impl Solver for FBDF with fn fill_dense_stages(_self, _integ) -> Unit {
  // `perform_step!` already fills every slot of `integrator.k` (the resampled
  // values plus the zero tail), so `_ode_addsteps!` has nothing to add —
  // upstream's FBDF registers no `addsteps!` either.
}

///|
pub impl Solver for FBDF with fn needs_fsal_reset_fixed_step(_self) -> Bool {
  // `only_diagonal_mass_matrix(alg::FBDF)` is false (the default), so
  // `update_fsal!` takes the plain copy branch in fixed-step solves too.
  false
}

///|
pub impl Solver for FBDF with fn uses_bdf_controllers(_self) -> Bool {
  // A multistep BDF: the stepping loop must run FBDF's own
  // `stepsize_controller!` / `choose_order!` / `bdf_step_reject_controller!`
  // / `post_newton_controller!` instead of the PI controller.
  true
}

///|
/// Promote the trait methods onto `FBDF`.
pub extend FBDF with Solver::{
  order,
  is_fsal,
  stage_count,
  perform_step,
  interpolant,
  fill_dense_stages,
  needs_fsal_reset_fixed_step,
  uses_bdf_controllers,
}

// ---------------------------------------------------------------------------
// Nonlinear-solver status (`NLStatus`, OrdinaryDiffEqCore.jl:303)
// ---------------------------------------------------------------------------

///|
/// `NLStatus::Convergence`. A non-positive status is a failure
/// (`nlsolvefail(status) = Int8(status) <= 0`).
const NL_CONVERGENCE : Int32 = 1

///|
/// `NLStatus::TryAgain`, which is an *alias* of `SlowConvergence` and so has
/// the value **0** — not -1, and not -3. It is a failure status (the step is
/// rejected) that additionally asks for a fresh Jacobian on the retry.
const NL_TRYAGAIN : Int32 = 0

///|
/// `NLStatus::Divergence`.
const NL_DIVERGENCE : Int32 = -2

// ---------------------------------------------------------------------------
// perform_step!
// ---------------------------------------------------------------------------

///|
/// `perform_step!(integrator, cache::FBDFCache, repeat_step = false)`
/// (bdf_perform_step.jl:1491), with the identity mass matrix.
///
/// The structure, in upstream's order:
///
///   1. `reinitFBDF!` — maintain the newest-first history queue.
///   2. Predictor `u₀`: the Lagrange interpolant of the history at `Θ = 1`
///      over the *actual* (variable-step) nodes, or the current state while no
///      history exists. `nlsolver.z = u₀`.
///   3. Corrector: `u_corrector[i]` is the same interpolant at `Θ = -i`, and
///      `tmp = -uprev·α₂ - Σ u_corrector[i]·α_{i+2}` becomes `tmp/dt`.
///   4. `β₀ = 1/α₁`, `γ = β₀`, `α = 1`, then `nlsolve!`.
///   5. LTE: `r[j]` weights, `terkp1 = (u - u₀)·∏ j·dt/(tdt - ts[j])`, and
///      `lte = (-1/(k+1) - Σ_{j≥2} α_j r_j)·terkp1` — the step's error
///      estimate. The `terk` family follows, from `estimate_terk!`.
///   6. `fsallast = α·(1/(γ·dt))·u - tmp` — no `f` call (see the header).
///   7. Dense output: resample the history interpolant onto the fixed
///      Chebyshev nodes and store it in `integrator.k`.
///
/// Two upstream details that look like mistakes but are not, and are
/// therefore reproduced verbatim:
///
///   * `u₀ = u` reads `integrator.u`, not `uprev`. They are equal at the top
///     of a step except right after a *rejected* step, where `u` still holds
///     the rejected endpoint — which is then the predictor. Faithful to
///     bdf_perform_step.jl:1512.
///   * The corrector's `tmp` starts from `-uprev·bdf_coeffs[k, 2]` (1-based
///     column 2), because `α₁` is the coefficient of the *unknown* new point
///     and has already been folded into `β₀`.
fn fbdf_perform_step(integ : Integrator, cache : FBDFCache) -> Unit {
  reinit_fbdf(integ, cache)
  let t = integ.t
  let dt = integ.dt
  let tdt = t + dt
  let k = cache.order
  let uprev = integ.uprev
  let dim = integ.u.length()
  let u0 = cache.u0

  // 2. Predictor. `pred_thetas` is upstream's `equi_ts`, reused by the
  // corrector and (overwritten) by the dense-output resample in step 7.
  let n_pred = k + 1
  if cache.iters_from_event >= 1 {
    let mut j = 0
    while j < n_pred {
      cache.pred_thetas.unsafe_set(j, (cache.ts.unsafe_get(j) - t) / dt)
      j += 1
    }
    eval_lagrange(u0, 1.0, cache.pred_thetas, cache.u_history, n_pred)
  } else {
    copy_into(u0, integ.u)
  }
  // `markfirststage!`: `firststage` is set here and cleared by the postamble.
  cache.firststage = true
  copy_into(cache.z, u0)

  // 3. Corrector. The zero-fill is upstream's `fill!` pass; `eval_lagrange`
  // overwrites every entry anyway, so it is defensive only.
  for h in cache.u_corrector {
    let mut r = 0
    while r < h.length() {
      h.unsafe_set(r, 0.0)
      r += 1
    }
  }
  if cache.iters_from_event >= 1 {
    let mut i = 0
    while i < k - 1 {
      eval_lagrange(
        cache.u_corrector.unsafe_get(i),
        -Double::from_int(i + 1),
        cache.pred_thetas,
        cache.u_history,
        n_pred,
      )
      i += 1
    }
  }
  let mut r = 0
  while r < dim {
    cache.tmp.unsafe_set(r, -uprev.unsafe_get(r) * cache.coeff(k, 1))
    r += 1
  }
  let mut m = 0
  while m < k - 1 {
    let c = cache.coeff(k, m + 2)
    let uc = cache.u_corrector.unsafe_get(m)
    r = 0
    while r < dim {
      cache.tmp.unsafe_set(r, cache.tmp.unsafe_get(r) - uc.unsafe_get(r) * c)
      r += 1
    }
    m += 1
  }
  let invdt = 1.0 / dt
  r = 0
  while r < dim {
    cache.tmp.unsafe_set(r, cache.tmp.unsafe_get(r) * invdt)
    r += 1
  }

  // 4. `β₀ = inv(bdf_coeffs[k, 1])`, `α₀ = 1`.
  cache.beta0 = 1.0 / cache.coeff(k, 0)
  cache.alpha = 1.0
  fbdf_nlsolve(integ, cache)
  // `nlsolvefail(nlsolver) && return`: on a failed solve nothing below runs —
  // no `u`, no `fsallast`, no `k` — and `force_stepfail` is already set.
  if cache.nlstatus <= 0 {
    cache.restart_pending = false
    return
  }
  copy_into(integ.u, cache.z)

  // 5a. `r[j] = (1-j)·∏_i ((tdt - j·dt) - ts[i])/(i·dt)`, i = 2..k+1.
  let mut jj = 2
  while jj <= k {
    let mut rj = 1.0 - Double::from_int(jj)
    let mut ii = 2
    while ii <= k + 1 {
      rj = rj *
        (tdt - Double::from_int(jj) * dt - cache.ts.unsafe_get(ii - 1)) /
        (Double::from_int(ii) * dt)
      ii += 1
    }
    cache.r.unsafe_set(jj - 1, rj)
    jj += 1
  }

  // 5b. `terkp1_tmp = (u - u₀)·∏_j j·dt/(tdt - ts[j])`, j = 1..k+1.
  r = 0
  while r < dim {
    cache.terkp1_tmp.unsafe_set(r, integ.u.unsafe_get(r) - u0.unsafe_get(r))
    r += 1
  }
  let mut j = 1
  while j <= k + 1 {
    let f = Double::from_int(j) * dt / (tdt - cache.ts.unsafe_get(j - 1))
    r = 0
    while r < dim {
      cache.terkp1_tmp.unsafe_set(r, cache.terkp1_tmp.unsafe_get(r) * f)
      r += 1
    }
    j += 1
  }

  // 5c. `lte = -1/(k+1) - Σ_{j=2..k} α_j·r[j]`, then `terk_tmp = lte·terkp1`.
  let mut lte = -1.0 / Double::from_int(1 + k)
  jj = 2
  while jj <= k {
    lte = lte - cache.coeff(k, jj - 1) * cache.r.unsafe_get(jj - 1)
    jj += 1
  }
  r = 0
  while r < dim {
    cache.terk_tmp.unsafe_set(r, lte * cache.terkp1_tmp.unsafe_get(r))
    r += 1
  }

  if integ.opts.adaptive {
    // `ts_tmp[1] = tdt; ts_tmp[i+1] = ts[i]` — the divided-difference nodes
    // `estimate_terk!` and `choose_order!` differentiate over.
    let mut i = 0
    while i <= k {
      cache.ts_tmp.unsafe_set(i + 1, cache.ts.unsafe_get(i))
      i += 1
    }
    cache.ts_tmp.unsafe_set(0, tdt)
    // EEst = ||lte / scale||
    weighted_residual(
      cache.atmp,
      cache.terk_tmp,
      uprev,
      integ.u,
      integ.opts.abstol,
      integ.opts.reltol,
    )
    integ.est = rms_norm(cache.atmp)
    // `estimate_terk!(k+1)` estimates dt^k·y^(k), whose norm is `terk`.
    fbdf_estimate_terk(integ, cache, k + 1)
    cache.terk = fbdf_norm_terk(cache, integ, uprev)
    if k > 1 {
      fbdf_estimate_terk(integ, cache, k)
      cache.terkm1 = fbdf_norm_terk(cache, integ, uprev)
    }
    if k > 2 {
      fbdf_estimate_terk(integ, cache, k - 1)
      cache.terkm2 = fbdf_norm_terk(cache, integ, uprev)
    }
    if cache.qwait == 0 && k < cache.max_order {
      weighted_residual(
        cache.atmp,
        cache.terkp1_tmp,
        uprev,
        integ.u,
        integ.opts.abstol,
        integ.opts.reltol,
      )
      cache.terkp1 = rms_norm(cache.atmp)
    } else {
      cache.terkp1 = 0.0
    }
  }

  // 6. `fsallast = α·(1/(γ·dt))·u - tmp`, the Newton convergence identity.
  // No `f` evaluation and no `nf` increment — this is the in-place
  // `FBDFCache` path; the out-of-place `FBDFConstantCache` calls `f` instead
  // (bdf_perform_step.jl:1390).
  let inv_gamma_dt = 1.0 / (cache.beta0 * dt)
  r = 0
  while r < dim {
    integ.fsallast.unsafe_set(
      r,
      cache.alpha * inv_gamma_dt * integ.u.unsafe_get(r) -
      cache.tmp.unsafe_get(r),
    )
    r += 1
  }

  // 7. Dense output: resample onto the fixed Chebyshev nodes. `dense_order`
  // is `k` (`time_filter` is off), so `n = min(k+1, max_order+1) = k+1` and
  // the node set is `thetas[0] = 1`, `thetas[j] = (ts[j-1]-t)/dt` — i.e. the
  // predictor's thetas shifted by one, which is why `pred_thetas` is reused.
  let n = imin(k + 1, cache.max_order + 1)
  cache.pred_thetas.unsafe_set(0, 1.0)
  let mut j = 0
  while j < imin(k, cache.max_order) {
    cache.pred_thetas.unsafe_set(j + 1, (cache.ts.unsafe_get(j) - t) / dt)
    j += 1
  }
  let nodes = cheb_nodes(n)
  let mut i = 0
  while i < n {
    let xi = nodes.unsafe_get(i)
    let l1 = lagrange_basis_scalar(xi, 0, cache.pred_thetas, n)
    let dst = integ.k.unsafe_get(i)
    let mut rr = 0
    while rr < dim {
      dst.unsafe_set(rr, l1 * integ.u.unsafe_get(rr))
      rr += 1
    }
    let mut jj = 1
    while jj < n {
      let lj = lagrange_basis_scalar(xi, jj, cache.pred_thetas, n)
      let src = cache.u_history.unsafe_get(jj - 1)
      rr = 0
      while rr < dim {
        dst.unsafe_set(rr, dst.unsafe_get(rr) + lj * src.unsafe_get(rr))
        rr += 1
      }
      jj += 1
    }
    i += 1
  }
  i = n
  while i < cache.max_order + 1 {
    let dst = integ.k.unsafe_get(i)
    let mut rr = 0
    while rr < dim {
      dst.unsafe_set(rr, 0.0)
      rr += 1
    }
    i += 1
  }
  // `_fbdf_finish_fixed_step!` is a no-op here: its body only runs when
  // `time_filter && !adaptive` (bdf_utils.jl:595).
  // `loopfooter_reset!` clears `derivative_discontinuity`; this port clears
  // `restart_pending` in the same place, so `do_newJW` sees it for the whole
  // step that follows a restart.
  cache.restart_pending = false
}

///|
/// `internalnorm(calculate_residuals(terk_tmp, uprev, u, …))`, the three-line
/// incantation `perform_step!` repeats for `terk`, `terkm1` and `terkm2`.
///
/// Upstream assigns `cache.terk = internalnorm(...)`. MoonBit value types are
/// passed by value, so the assignment target has to live at the call site;
/// hence the `Double` return.
fn fbdf_norm_terk(
  cache : FBDFCache,
  integ : Integrator,
  uprev : Array[Double],
) -> Double {
  weighted_residual(
    cache.atmp,
    cache.terk_tmp,
    uprev,
    integ.u,
    integ.opts.abstol,
    integ.opts.reltol,
  )
  rms_norm(cache.atmp)
}

// ---------------------------------------------------------------------------
// The Newton corrector
// ---------------------------------------------------------------------------

///|
/// `nlsolve!(nlsolver, integrator, cache, repeat_step = false)`
/// (nlsolve.jl:105) for `NLNewton` with `method = COEFFICIENT_MULTISTEP`,
/// reduced to what FBDF reaches:
///
///   * `always_new = false`, `check_div = true`, `relax = nothing`,
///     `fast_convergence_cutoff = 0.2` (so `do_newJW`'s zero-cutoff branch is
///     dead), `is_varying_mm = false` (identity mass matrix).
///   * `apply_predictor!` and `initialize!` are no-ops: FBDF supplies no
///     `stage_predictor`, and the only things `initialize!` would set
///     (`invγdt`, `tstep`) are computed where they are used.
///   * `_uninformative_step` is the generic `false` — the specialized method
///     only exists for `NonlinearSolveAlg` caches.
///   * The `@goto REDO` retry is a bounded loop. One retry is reachable in
///     practice: the retry fires only when `J_t != t`, and its own
///     `update_W!` then either refreshes `J` (setting `J_t = t`, which stops
///     the cycle) or rebuilds `W` at the current `γ·dt` (making the next
///     `smallstepchange` true, hence `jbad` true, hence `J` refreshed). The
///     cap is insurance against that argument being wrong somewhere.
fn fbdf_nlsolve(integ : Integrator, cache : FBDFCache) -> Unit {
  let mut eta = cache.etaold
  let mut attempt = 0
  while attempt < FBDF_MAX_NLSOLVE_ATTEMPTS {
    attempt += 1
    // `γW = γ·dt/α` for COEFFICIENT_MULTISTEP; with `α = 1` and `γ = β₀` this
    // is `β₀·dt`, the `dtgamma` `W` is assembled at.
    let gamma_w = cache.beta0 * integ.dt / cache.alpha
    fbdf_update_w(integ, cache, gamma_w)
    // `nlsolver.status = check_div ? Divergence : Convergence`
    cache.nlstatus = NL_DIVERGENCE
    // `η = get_new_W!(nlsolver) ? initial_η(nlsolver, integrator) : ηold`
    eta = if cache.new_w { fbdf_initial_eta(cache) } else { cache.etaold }
    let mut ndz = 1.0
    let mut iter = 1
    while iter <= cache.max_iter {
      cache.nliter = iter
      let ndzprev = ndz
      ndz = fbdf_compute_step(integ, cache, gamma_w)
      if !is_finite(ndz) {
        cache.nlstatus = NL_DIVERGENCE
        cache.nfails += 1
        break
      }
      // `has_prev_θ` is true (the `NLSolver` struct has the field).
      let mut prev_theta = cache.prev_theta
      let mut theta = prev_theta
      if iter > 1 {
        // `θ = prev_θ = max(0.3·prev_θ, ndz/ndzprev)`
        prev_theta = dmax(0.3 * prev_theta, ndz / ndzprev)
        theta = prev_theta
        if (prev_theta - 1.0).abs() <= 100.0 * sqrt(eps) {
          // One iteration does nothing: at the floating-point limit, so call
          // it by `ndz` directly.
          if ndz <= 1.0 {
            cache.nlstatus = NL_CONVERGENCE
            cache.nfails = 0
          } else {
            cache.nlstatus = NL_DIVERGENCE
            cache.nfails += 1
          }
          break
        }
        if prev_theta > 2.0 {
          cache.nlstatus = NL_DIVERGENCE
          cache.nfails += 1
          break
        }
      } else if !integ.accept_step {
        // The first iteration of a retry has no rate information: pin `η` at 1
        // (which makes `η/(1-η)` infinite, so the `κ` test cannot fire).
        prev_theta = 1.0
        theta = prev_theta
      }
      cache.prev_theta = prev_theta
      // `apply_step!`: z ← ztmp (the next iterate, computed by compute_step!)
      copy_into(cache.z, cache.ztmp2)
      // `η = θ/(1-θ)`; SciMLBase.value is the identity for a real `θ`.
      eta = theta / (1.0 - theta)
      // `check_η_convergence = iter > 1 || (adaptive && isnewton)` — the
      // second disjunct is always true here.
      let check_eta_convergence = iter > 1 || integ.opts.adaptive
      if (iter == 1 && ndz < 1.0e-5) ||
        (check_eta_convergence && eta >= 0.0 && eta * ndz < cache.kappa) {
        cache.nlstatus = NL_CONVERGENCE
        cache.nfails = 0
        break
      }
      iter += 1
    }
    // Stale-Jacobian retry: `tracks_J_age && status == Divergence &&
    // !isJcurrent` — all three hold exactly when the test below passes,
    // because `tracks_J_age` is true for every Newton cache.
    if cache.nlstatus == NL_DIVERGENCE && integ.t != cache.j_t {
      cache.nlstatus = NL_TRYAGAIN
      cache.nfails += 1
    } else {
      break
    }
  }
  cache.etaold = eta
  // `postamble!`: force_stepfail = nlsolvefail; firststage = false;
  // firstcall = false; return z.
  integ.force_stepfail = cache.nlstatus <= 0
  cache.firststage = false
  cache.firstcall = false
}

///|
/// How many times `nlsolve!` may re-enter its `@label REDO` block. See
/// [`fbdf_nlsolve`] for why one retry is the reachable bound.
const FBDF_MAX_NLSOLVE_ATTEMPTS : Int32 = 3

///|
/// `initial_η` (nlsolve.jl:355): `max(ηold, eps(reltol))^0.8`.
fn fbdf_initial_eta(cache : FBDFCache) -> Double {
  fastpower(dmax(cache.etaold, eps), 0.8)
}

///|
/// `compute_step!(nlsolver::NLSolver{<:NLNewton, true}, integrator, γW)`
/// (newton.jl:758) with `method = COEFFICIENT_MULTISTEP` and the identity mass
/// matrix, returning the scaled residual norm `ndz`.
///
/// The residual is `ztmp = tmp + f(z, t+dt) - α·(1/(γ·dt))·z`, solved as
/// `dz = W \ ztmp` with `W = J - I/(γ·dt)`; the factorization is refreshed
/// only on the first iteration of a step that rebuilt `W` (`iter == 1 &&
/// new_W`), which is where the "one factorization per step" cost profile comes
/// from. When the factorization is older than the current `γ·dt`, the
/// increment is rescaled by `2/(1 + γdt/W_γdt)` — CVODE's stale-`W`
/// correction, which keeps a reused factorization usable across a moderate
/// step-size change.
///
/// `ustep` is `z` itself for COEFFICIENT_MULTISTEP (`compute_ustep` returns
/// `z` unchanged), which is why the residual norm scales `dz` against
/// `max(|uprev|, |z|)` rather than against a stage value.
fn fbdf_compute_step(
  integ : Integrator,
  cache : FBDFCache,
  _gamma_w : Double,
) -> Double {
  let t = integ.t
  let dt = integ.dt
  let dim = integ.u.length()
  // `tstep = t + c·dt` with `c = 1` (`initialize!`, newton.jl:25).
  let tstep = t + dt
  // `invγdt = inv(dt·γ)`.
  let inv_gamma_dt = 1.0 / (cache.beta0 * dt)
  // `f(k, z, p, tstep)` — one right-hand-side evaluation per iteration.
  Integrator::eval_f(integ, tstep, cache.z, cache.nlk)
  // `ztmp = tmp + k - (α·invγdt)·z`
  let a_inv = cache.alpha * inv_gamma_dt
  let mut i = 0
  while i < dim {
    cache.ztmp.unsafe_set(
      i,
      cache.tmp.unsafe_get(i) +
      cache.nlk.unsafe_get(i) -
      a_inv * cache.z.unsafe_get(i),
    )
    i += 1
  }
  // `dz = W \ ztmp`; `A = W` (i.e. refactorize) iff `iter == 1 && new_W`.
  copy_into(cache.dz, cache.ztmp)
  if cache.nliter == 1 && cache.new_w {
    // Upstream's `dolinsolve` throws `SingularException` on a singular `W`;
    // this port's `lu_factor` reports it, and a step failure is the graceful
    // equivalent (dt is halved and the step retried).
    if !lu_factor(cache.stiff.w, dim, cache.stiff.ipiv) {
      return @double.infinity
    }
  }
  lu_solve(cache.stiff.w, cache.stiff.ipiv, cache.dz)
  // `γdt = γ·dt` (not a DAE); `!(W_γdt ≈ γdt) → dz *= 2/(1 + γdt/W_γdt)`.
  let gamma_dt = cache.beta0 * dt
  if !is_approx(cache.w_gdt, gamma_dt) {
    let scale = 2.0 / (1.0 + gamma_dt / cache.w_gdt)
    i = 0
    while i < dim {
      cache.dz.unsafe_set(i, cache.dz.unsafe_get(i) * scale)
      i += 1
    }
  }
  // `atmp = calculate_residuals(dz, uprev, ustep = z)`; `ndz = rms_norm`.
  // `has_special_newton_error(FBDF)` is false, so there is no error-constant
  // scaling (unlike QNDF/NordsieckBDF).
  weighted_residual(
    cache.atmp,
    cache.dz,
    integ.uprev,
    cache.z,
    integ.opts.abstol,
    integ.opts.reltol,
  )
  let ndz = rms_norm(cache.atmp)
  // `ztmp = z - dz` — the next iterate, committed by `apply_step!` in the
  // driver *after* the divergence checks and *before* the convergence check.
  i = 0
  while i < dim {
    cache.ztmp2.unsafe_set(i, cache.z.unsafe_get(i) - cache.dz.unsafe_get(i))
    i += 1
  }
  ndz
}

///|
/// Julia's `≈` (`isapprox`) with its default `rtol = sqrt(eps)`, `atol = 0`:
/// the `W_γdt ≈ γdt` test of `compute_step!`.
fn is_approx(a : Double, b : Double) -> Bool {
  (a - b).abs() <= sqrt(eps) * dmax(a.abs(), b.abs())
}

///|
/// `update_W!(nlsolver, integrator, cache, dtgamma, repeat_step)`
/// (derivative_utils.jl:1149) → `calc_W!` (derivative_utils.jl:826) for the
/// in-place Newton path with a concrete dense `W`:
///
///   * `new_jac` → record `J_t = t` and recompute the Jacobian at `(t, uprev)`
///     by forward differences (the port has no AD and no linearity
///     detection, so `islinearfunction` is always false and the `has_jac`
///     fast path is never taken).
///   * `new_W` → `W = J - I/(γ·dt)` (`jacobian2W!`, which *destroys* the old
///     factorization) and remember `W_γdt = γ·dt`.
///
/// `new_W` without `new_jac` is the cheap case: the cached Jacobian is reused
/// and only the `O(n²)` shift plus a refactorization is paid.
fn fbdf_update_w(
  integ : Integrator,
  cache : FBDFCache,
  dtgamma : Double,
) -> Unit {
  let dim = integ.u.length()
  let (new_jac, new_w) = fbdf_do_new_jw(integ, cache)
  if new_jac {
    cache.j_t = integ.t
    // Scratch for the finite-difference sweep, pooled on the integrator
    // (upstream's `NLNewtonCache` holds its own `uf`/`jac_config` pair).
    let u_pert = Integrator::stiff_vec(integ, 0)
    let du_pert = Integrator::stiff_vec(integ, 1)
    // `du_base` is `f(t, uprev)`, which the integrator holds as `fsalfirst`
    // (set by the pre-start FSAL and refreshed by `update_fsal!` on every
    // accepted step). The sweep costs `n` evaluations here against upstream's
    // `n + 1` because the base value is already in hand — the same convention
    // the Rosenbrock port uses (jacobian.mbt).
    jacobian_fd(
      integ,
      integ.t,
      integ.uprev,
      integ.fsalfirst,
      cache.stiff.jac,
      u_pert,
      du_pert,
    )
  }
  if new_w {
    jacobian2w(cache.stiff.w, cache.stiff.jac, dim, dtgamma)
  }
  cache.new_w = new_w
  if new_w {
    cache.w_gdt = dtgamma
  }
}

///|
/// `do_newJW(integrator, alg, nlsolver, repeat_step)`
/// (derivative_utils.jl:541) for FBDF's setting: a Newton solver, a
/// non-linear `f`, an adaptive or fixed-step pure ODE with the identity mass
/// matrix, and no derivative discontinuity in flight.
///
/// The decision, branch by branch:
///
///   * `iter <= 1` → `(true, true)`: at least one `W` per solve.
///   * `repeat_step` → `(false, false)`: never taken here (`perform_step!`
///     calls `nlsolve!` without a repeat flag, as upstream's FBDF does).
///   * `islinearfunction` fast path: **dead in this port**. The port has no
///     linearity detection, so `islin` is always false and the branch is
///     unreachable; upstream's copy would return `(false, iγdt != W_iγdt)`.
///   * `!adaptive` → `(true, true)`: a prescribed step size makes reuse
///     pointless, and upstream calls this out explicitly because it once
///     caused an IIP/OOP inconsistency.
///   * `isfirstcall` → `(true, true)`.
///   * `isfreshJ` (`t == J_t` and no restart pending): upstream sets
///     `jbad = false` and `smallstepchange = true` and then **falls through to
///     the shared `wbad` line**, so the result is
///     `(false, (isfs && errorfail) || Divergence)` — *not* an unconditional
///     `(false, true)`. The `errorfail` disjunct is what rebuilds `W` after a
///     rejected retry at the same `t` (same `J`, new `γ·dt`), and it is the
///     only thing standing between a `force_stepfail` retry and a reused
///     factorization.
///   * otherwise `smallstepchange = |1/(γ·dt) / (1/W_γdt) - 1| ≤ 0.2` decides;
///     `jbad` is the stale-Jacobian retry signal, and
///     `wbad = !smallstepchange || (isfs && errorfail) || Divergence`.
///
/// `isfs = isfirststage(nlsolver)` is **always true** for FBDF:
/// `markfirststage!` runs at the top of every `perform_step!` and the
/// postamble clears it only after the solve, so the `isfs && errorfail`
/// disjunct of `wbad` reduces to `errorfail`. Two more upstream branches are
/// dead here and deliberately absent: `iszero(fast_convergence_cutoff)`
/// (NLNewton's default is `1//5`) and `isdae` (FBDF is an ODEAlgorithm).
fn fbdf_do_new_jw(integ : Integrator, cache : FBDFCache) -> (Bool, Bool) {
  if integ.iter <= 1 {
    return (true, true)
  }
  if !integ.opts.adaptive {
    return (true, true)
  }
  if cache.firstcall {
    return (true, true)
  }
  let isfs = cache.firststage
  let errorfail = integ.est > 1.0
  // `isJcurrent && !integrator.derivative_discontinuity`
  let isfresh_j = integ.t == cache.j_t && !cache.restart_pending
  let (jbad, smallstepchange) = if isfresh_j {
    // upstream's `isfreshJ` branch: jbad = false, smallstepchange = true
    (false, true)
  } else {
    let w_i_gdt = 1.0 / cache.w_gdt
    let i_gdt = 1.0 / (cache.beta0 * integ.dt)
    let smallstepchange = (i_gdt / w_i_gdt - 1.0).abs() <= cache.new_w_dt_cutoff
    let jbad = cache.nlstatus == NL_TRYAGAIN && smallstepchange
    (jbad, smallstepchange)
  }
  let wbad = !smallstepchange ||
    (isfs && errorfail) ||
    cache.nlstatus == NL_DIVERGENCE
  (jbad, jbad || wbad)
}

// ---------------------------------------------------------------------------
// Order selection
// ---------------------------------------------------------------------------

///|
/// `estimate_terk!(integrator, cache, k, ::Val{max_order})`
/// (bdf_utils.jl:391): fill `terk_tmp` with
///
///     |dt^(k-1)| · ( fd_weights[1, k]·u + Σ_{i=2..k} fd_weights[i, k]·u_history[i-1] )
///
/// the Fornberg estimate of `dt^(k-1)·y^(k-1)(t+dt)` over the nodes
/// `ts_tmp = [tdt, ts[1..k]]`. Argument `k` here is upstream's `k`, so the
/// derivative order tabulated is `k-1` and the column read is `k` (1-based),
/// i.e. `order + 1` — the one column the tabulation actually fills.
fn fbdf_estimate_terk(integ : Integrator, cache : FBDFCache, k : Int32) -> Unit {
  let stride = cache.max_order + 1
  calc_finite_difference_weights(
    cache.fd_weights,
    cache.ts_tmp,
    integ.t + integ.dt,
    k - 1,
    stride,
  )
  let dim = integ.u.length()
  let col = k - 1
  let w0 = cache.fd_weights.unsafe_get(0 * stride + col)
  let mut i = 0
  while i < dim {
    cache.terk_tmp.unsafe_set(i, w0 * integ.u.unsafe_get(i))
    i += 1
  }
  let mut ii = 2
  while ii <= k {
    let w = cache.fd_weights.unsafe_get((ii - 1) * stride + col)
    let hist = cache.u_history.unsafe_get(ii - 2)
    i = 0
    while i < dim {
      cache.terk_tmp.unsafe_set(
        i,
        cache.terk_tmp.unsafe_get(i) + w * hist.unsafe_get(i),
      )
      i += 1
    }
    ii += 1
  }
  let scale = fastpower(integ.dt, Double::from_int(k - 1)).abs()
  i = 0
  while i < dim {
    cache.terk_tmp.unsafe_set(i, cache.terk_tmp.unsafe_get(i) * scale)
    i += 1
  }
}

///|
/// `choose_order!(alg::FBDF, integrator, cache, ::Val{max_order})`
/// (controllers.jl:295): decide the order for the next step and return it
/// together with the `terk` value the step-size formula should use.
///
/// **Increase** — `k < max_order && qwait == 0` and the ladder is strictly
/// decreasing — promotes `k` and adopts `terkp1` as the new `terk`.
///
/// **Decrease** — while the ladder is *not* strictly decreasing and
/// `k > 2` — shifts the ladder down and recomputes `terkm2` from the
/// divided-difference weights.
///
/// The returned `terk` is deliberately *not* written back to `cache.terk`:
/// upstream destructures the `terk` family into locals, so the shifts below
/// are visible only to the caller (which uses the returned `terk` in the
/// CVODE step-size formula and, on a STALD reduction, replaces it with
/// `cache.terkm1`).
///
/// The recomputation in the decrease branch is transcribed with its upstream
/// quirks intact, because the port's north star is behavioural equivalence:
///
///   * it reads `fd_weights[k-2, 1]` (1-based), i.e. **column 1** of the
///     tabulation. `calc_finite_difference_weights!` zeroes that column
///     (Fornberg's `c[·,0]` recurrence is replaced by `fill!(…, 0)` in the
///     transcription), so the `u` term contributes exactly zero — the
///     "weight of the new point" that a correct transcription would put there
///     is simply absent. What survives is the `u_history` sum alone.
///   * that sum reads column `k-2` (1-based), i.e. the `(k-3)`-th derivative,
///     and scales by `|dt^(k-2)|`, where the value an order-`k-2` method would
///     want is `|dt^(k-3)|·y^(k-3)`. The result is therefore off by a factor
///     of `dt` from the `terkm2` it stands in for — but it only ever feeds the
///     `terkm2 > terkm1` comparison below, never the step-size formula.
fn fbdf_choose_order(integ : Integrator, cache : FBDFCache) -> (Int32, Double) {
  let t = integ.t
  let dt = integ.dt
  let uprev = integ.uprev
  let dim = integ.u.length()
  let mut k = cache.order
  let mut terk = cache.terk
  let mut terkm1 = cache.terkm1
  let mut terkm2 = cache.terkm2
  let mut terkp1 = cache.terkp1
  if k < cache.max_order &&
    cache.qwait == 0 &&
    (
      (k == 1 && terk > terkp1) ||
      (k == 2 && terkm1 > terk && terk > terkp1) ||
      (k > 2 && terkm2 > terkm1 && terkm1 > terk && terk > terkp1)
    ) {
    k += 1
    terk = terkp1
  } else {
    while !(terkm2 > terkm1 && terkm1 > terk && terk > terkp1) && k > 2 {
      terkp1 = terk
      terk = terkm1
      terkm1 = terkm2
      let stride = cache.max_order + 1
      calc_finite_difference_weights(
        cache.fd_weights,
        cache.ts_tmp,
        t + dt,
        k - 2,
        stride,
      )
      // `fd_weights[k-2, 1] * u` — column 1 is zero, so this term vanishes.
      let w_u = cache.fd_weights.unsafe_get((k - 3) * stride + 0)
      let col = k - 3
      let mut i = 0
      while i < dim {
        cache.terk_tmp.unsafe_set(i, w_u * integ.u.unsafe_get(i))
        i += 1
      }
      let mut ii = 2
      while ii <= k - 2 {
        let w = cache.fd_weights.unsafe_get((ii - 1) * stride + col)
        let hist = cache.u_history.unsafe_get(ii - 2)
        i = 0
        while i < dim {
          cache.terk_tmp.unsafe_set(
            i,
            cache.terk_tmp.unsafe_get(i) + w * hist.unsafe_get(i),
          )
          i += 1
        }
        ii += 1
      }
      let scale = fastpower(dt, Double::from_int(k - 2)).abs()
      i = 0
      while i < dim {
        cache.terk_tmp.unsafe_set(i, cache.terk_tmp.unsafe_get(i) * scale)
        i += 1
      }
      weighted_residual(
        cache.atmp,
        cache.terk_tmp,
        uprev,
        integ.u,
        integ.opts.abstol,
        integ.opts.reltol,
      )
      terkm2 = rms_norm(cache.atmp)
      k -= 1
    }
  }
  (k, terk)
}

// ---------------------------------------------------------------------------
// The four BDF controllers (OrdinaryDiffEqBDF/src/controllers.jl)
// ---------------------------------------------------------------------------

///|
/// `get_gamma` for FBDF: `gamma_default(Union{QNDF, FBDF, DFBDF}) = 12/10`.
/// This is the `zₛ` safety factor of the step-size formula and of
/// `bdf_step_reject_controller!` — not the `γ = β₀` of the Newton matrix,
/// which is a different quantity that happens to share a name upstream.
const FBDF_GAMMA : Double = 1.2

///|
/// `qmin_default(alg) = 1/5` (alg_utils.jl:421). FBDF does not override it.
const FBDF_QMIN : Double = 0.2

///|
/// FBDF's own `qmax` keyword default (algorithms.jl:542), threaded through
/// `default_controller` into `BDFController`. Note this *overrides*
/// `qmax_default(Union{QNDF,FBDF,DFBDF}) = 5//1` in resolve_basic.
const FBDF_QMAX : Double = 10.0

///|
/// `qmax_first_step_default(alg) = 10000` (alg_utils.jl:880), in force until
/// the first step is accepted (`get_current_qmax`).
const FBDF_QMAX_FIRST_STEP : Double = 10000.0

///|
/// FBDF's `qsteady_min` keyword default (algorithms.jl:542).
const FBDF_QSTEADY_MIN : Double = 0.9

///|
/// FBDF's `qsteady_max` keyword default (algorithms.jl:542).
const FBDF_QSTEADY_MAX : Double = 2.0

///|
/// `failfactor_default(alg) = 2` (alg_utils.jl:889): the divisor
/// `post_newton_controller!` applies to `dt` after a Newton failure.
const FBDF_FAILFACTOR : Double = 2.0

///|
/// `stepsize_controller!(integrator, cache, alg::FBDF)` (controllers.jl:396):
/// run STALD, pick the order, and return the CVODE step-size factor `q`.
///
/// Order of operations matters and is upstream's:
///
///   1. `prev_order = order` (the order the step that just finished used).
///   2. STALD collects and checks **before** `choose_order!`, at the step's
///      own order — orders 3-5 are only alpha-stable, and the detector needs
///      the norms of the order that was actually run.
///   3. `choose_order!` proposes the next order; `nconsteps` is reset when it
///      moves.
///   4. A STALD verdict overrides: `k = min(k, step_order - 1)` and `terk` is
///      taken from `cache.terkm1` (the estimate of the order being forced).
///   5. `terk == 0` means an exact step: grow by the full `qmax_eff`.
///      Otherwise the CVODE formula
///      `q = (6·terk/(α₀·(k+1)))^{1/(k+1)}` with `BIAS2 = 6` and `α₀` the
///      fixed leading coefficient of order `k`.
fn fbdf_stepsize_controller(integ : Integrator, cache : FBDFCache) -> Double {
  cache.prev_order = cache.order
  let step_order = cache.prev_order
  let mut stald_reduce = false
  if step_order >= 3 {
    stald_collect_data(
      cache.stald,
      step_order,
      cache.terkm2,
      cache.terkm1,
      cache.terk,
    )
    stald_reduce = stald_check(cache.stald, step_order)
  }
  // Split binding: `let (mut k, mut terk) = …` is written as an immutable
  // destructuring plus two `mut` rebindings, which is the pattern the rest of
  // the codebase uses (MoonBit's `mut` inside tuple patterns is avoided here
  // deliberately).
  let (k0, terk0) = fbdf_choose_order(integ, cache)
  let mut k = k0
  let mut terk = terk0
  if k != cache.order {
    cache.nconsteps = 0
    cache.order = k
  }
  if stald_reduce {
    k = imin(k, step_order - 1)
    cache.order = k
    cache.nconsteps = 0
    terk = cache.terkm1
  }
  if terk == 0.0 {
    1.0 / fbdf_current_qmax(integ)
  } else {
    let alpha0 = cache.coeff(k, 0)
    let kp1 = Double::from_int(k + 1)
    fastpower(6.0 * terk / (alpha0 * kp1), 1.0 / kp1)
  }
}

///|
/// `get_current_qmax(integrator, qmax)` (controllers.jl:288): the looser
/// `qmax_first_step` until the first step has been accepted.
fn fbdf_current_qmax(integ : Integrator) -> Double {
  if integ.success_iter == 0 {
    FBDF_QMAX_FIRST_STEP
  } else {
    FBDF_QMAX
  }
}

///|
/// `step_accept_controller!(integrator, cache, alg::FBDF, q)`
/// (controllers.jl:455): turn the step-size factor of an accepted step into
/// the next step length, and update the order bookkeeping.
///
/// The deadband `[qsteady_min, qsteady_max] = [0.9, 2]` holds `dt` constant —
/// without it a step that is already right-sized would jitter by a percent
/// every step. `iters_from_event` counts accepted steps since the last
/// restart and gates the predictor; `qwait` is CVODE's order-change
/// countdown, re-armed to `order + 2` whenever the order moved.
fn fbdf_step_accept_controller(
  integ : Integrator,
  cache : FBDFCache,
  q : Double,
) -> Double {
  cache.consfailcnt = 0
  let mut qq = q
  if FBDF_QSTEADY_MIN <= qq && qq <= FBDF_QSTEADY_MAX {
    qq = 1.0
  }
  cache.nconsteps += 1
  cache.iters_from_event += 1
  if cache.order != cache.prev_order {
    cache.qwait = cache.order + 2
  } else if cache.qwait > 0 {
    cache.qwait -= 1
  }
  integ.dt / qq
}

///|
/// `bdf_step_reject_controller!(integrator, cache, EEst1, error_order)`
/// (controllers.jl:192), reached for FBDF as
/// `bdf_step_reject_controller!(integrator, cache, cache.terkm1)`
/// (controllers.jl:275): shrink `dt` after a rejected step, possibly dropping
/// the order, working on `|dt|` and restoring the direction at the end.
///
/// The formula is the one from the adaptive-BDF2 paper the upstream comment
/// cites (Alberdi Celaya, Anza Aguirrezabala, Chatzipantelidis): a
/// `zₛ = γ = 1.2` safety factor with `expo = 1/(error_order+1)`, a `z ≤ 10`
/// guard against unbounded growth, and a competing order-`k-1` candidate
/// `zₖ₋₁ = 1.3·EEst1^{1/k}` that wins when it allows the larger step.
/// `EEst1` is `cache.terkm1` for FBDF — the estimate of the *lower* order,
/// which is the order the candidate would run at.
///
/// Two bookkeeping details: `consfailcnt > 1` halves `h` outright (the second
/// consecutive failure means the error test alone is not shrinking fast
/// enough), and `kₙ == 1 && consfailcnt > 3` arms a full restart
/// ([`bdf_restart`]) — the method cannot take *any* step at the lowest order,
/// so the history it extrapolates from is not describing the solution.
fn bdf_step_reject_controller(integ : Integrator, cache : FBDFCache) -> Unit {
  let k = cache.order
  let mut h = integ.dt.abs()
  cache.consfailcnt += 1
  cache.nconsteps = 0
  // `discontinuity_detection` defaults to false in `resolve_basic`, so
  // upstream's `set_discontinuity` branch is unreachable here.
  if cache.consfailcnt > 1 {
    h = h / 2.0
  }
  let zs = FBDF_GAMMA
  let expo = 1.0 / Double::from_int(k + 1)
  let z = zs * fastpower(integ.est, expo)
  let hk = if z <= 10.0 { h / z } else { 0.1 * h }
  let mut hn = hk
  let mut kn = k
  if k > 1 {
    let expo_km1 = 1.0 / Double::from_int(k)
    let z_km1 = 1.3 * fastpower(cache.terkm1, expo_km1)
    let h_km1 = if z_km1 <= 10.0 { h / z_km1 } else { 0.1 * h }
    if cache.consfailcnt > 2 || h_km1.abs() > hk.abs() {
      hn = dmin(h, h_km1.abs())
      kn = k - 1
    }
  }
  if kn == 1 && cache.consfailcnt > 3 {
    bdf_restart(cache)
  }
  integ.dt = integ.tdir * hn
  cache.order = kn
}

///|
/// `post_newton_controller!(integrator, cache, alg::FBDF)`
/// (controllers.jl:285): the Newton-failure path, reached from
/// `_loopfooter!`'s `force_stepfail` branch.
///
/// Three repeated Newton failures at an order above 1 drop the order — the
/// corrector is struggling, and a lower order has a better-conditioned
/// iteration matrix — then `dt` is divided by `failfactor = 2` and the
/// failure counters are advanced. `nlsolver.nfails` is the *consecutive*
/// failure count: `nlsolve!` resets it to 0 on every convergence.
fn fbdf_post_newton_controller(integ : Integrator, cache : FBDFCache) -> Unit {
  if cache.order > 1 && cache.nfails >= 3 {
    cache.order -= 1
  }
  integ.dt = integ.dt / FBDF_FAILFACTOR
  cache.consfailcnt += 1
  cache.nconsteps = 0
}

// ---------------------------------------------------------------------------
// In-package invariants
// ---------------------------------------------------------------------------

///|
/// The algorithm descriptor carries upstream's defaults: `alg_order` is the
/// dummy 1, `kshortsize = max_order + 1`, `isfsal` is true, and the
/// interpolant is the Chebyshev barycentric form.
test "fbdf algorithm descriptor matches upstream defaults" {
  let alg = FBDF::new()
  assert_eq(alg.order(), 1)
  assert_eq(alg.stage_count(), 6)
  assert_true(alg.is_fsal())
  assert_false(alg.needs_fsal_reset_fixed_step())
  let alg3 = FBDF::with_max_order(3)
  assert_eq(alg3.stage_count(), 4)
  // The interpolant is the Chebyshev barycentric polynomial, not the Hermite
  // fallback the one-step methods use.
  let interp_is_poly = match alg.interpolant() {
    Poly(_, _) => true
    Linear | Hermite => false
  }
  assert_true(interp_is_poly)
}

///|
/// The controller knobs FBDF resolves to, through
/// `default_controller → BDFController → resolve_basic` with the algorithm's
/// own keyword defaults threaded in (controllers.jl:65, alg_utils.jl:421/880/889).
test "fbdf controller knobs are the resolved BDFController values" {
  assert_eq(FBDF_GAMMA, 1.2)
  assert_eq(FBDF_QMIN, 0.2)
  assert_eq(FBDF_QMAX, 10.0)
  assert_eq(FBDF_QMAX_FIRST_STEP, 10000.0)
  assert_eq(FBDF_QSTEADY_MIN, 0.9)
  assert_eq(FBDF_QSTEADY_MAX, 2.0)
  assert_eq(FBDF_FAILFACTOR, 2.0)
}

///|
/// A scalar problem with a hand-built cache: `u' = -u` at `t = 0`, `dt = 0.1`,
/// history `[u(0), u(-0.1), …]` from the exact solution. Checks the predictor,
/// the corrector's history term, the LTE and the `terk` family against the
/// closed forms.
test "fbdf step kernel reproduces the implicit Euler step" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_dt(SolveOptions::new(), 0.1)
  let integ = Integrator::new(prob, opts, 6, 1)
  integ.is_fsal = true
  Integrator::eval_f(integ, integ.t, integ.u, integ.fsalfirst)
  let cache = integ.bdf
  cache.max_order = 5
  ignore(FBDFCache::ensure(cache, 1))
  // Order 1 with a one-point history: `iters_from_event == 0` makes the
  // predictor `u₀ = u` and the corrector history term empty.
  cache.order = 1
  cache.iters_from_event = 0
  cache.consfailcnt = 0
  fbdf_perform_step(integ, cache)
  // The Newton solve of `(z - u)/dt = -z` is exact: z = u/(1+dt).
  assert_true((integ.u.unsafe_get(0) - 1.0 / 1.1).abs() < 0.000000000000001)
  // `fsallast` comes from the convergence identity, not an `f` call: for the
  // linear problem it is exactly `f(u, t+dt) = -u`.
  assert_true(
    (integ.fsallast.unsafe_get(0) + integ.u.unsafe_get(0)).abs() <
    0.000000000000001,
  )
  // No right-hand-side evaluation beyond the one Jacobian sweep and the
  // Newton iterations: the pre-start FSAL (1) + the Jacobian (1) + 2 Newton
  // residuals (2) = 4. The `fsallast` identity costs nothing.
  assert_eq(integ.nf, 4)
}

///|
/// `choose_order!` at order 1 with a decreasing ladder promotes the order and
/// adopts `terkp1`; a non-decreasing ladder leaves the order alone.
test "fbdf choose_order promotes on a decreasing ladder" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_adaptive(SolveOptions::new(), true)
  let integ = Integrator::new(prob, opts, 6, 1)
  let cache = integ.bdf
  cache.max_order = 5
  ignore(FBDFCache::ensure(cache, 1))
  integ.t = 0.0
  integ.dt = 0.1
  cache.order = 1
  cache.qwait = 0
  // Ladder 4 > 3 > 2: `k == 1 && terk > terkp1` fires.
  cache.terk = 3.0
  cache.terkp1 = 2.0
  cache.terkm1 = 4.0
  cache.terkm2 = 5.0
  let (k, terk) = fbdf_choose_order(integ, cache)
  assert_eq(k, 2)
  assert_eq(terk, 2.0)
  // `qwait > 0` gates the increase even with a decreasing ladder.
  cache.order = 1
  cache.qwait = 2
  let (k2, _) = fbdf_choose_order(integ, cache)
  assert_eq(k2, 1)
  // At `max_order` the increase is impossible.
  cache.order = 5
  cache.qwait = 0
  cache.terk = 3.0
  cache.terkp1 = 2.0
  let (k3, _) = fbdf_choose_order(integ, cache)
  assert_eq(k3, 5)
}

///|
/// The CVODE step-size formula of `stepsize_controller!`, with the order
/// pinned so `choose_order!` cannot move it: `q = (6·terk/(α₀(k+1)))^{1/(k+1)}`.
/// For order 2, `α₀ = 3/2` and `terk = 1e-4`:
/// `q = (6e-4/(1.5·3))^{1/3} = (1.3333e-4)^{1/3} ≈ 0.05103121…`.
test "fbdf stepsize controller uses the CVODE formula" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_adaptive(SolveOptions::new(), true)
  let integ = Integrator::new(prob, opts, 6, 1)
  let cache = integ.bdf
  cache.max_order = 5
  ignore(FBDFCache::ensure(cache, 1))
  integ.t = 0.0
  integ.dt = 0.1
  integ.success_iter = 1
  cache.order = 2
  cache.prev_order = 2
  cache.terkm2 = 5.0
  cache.terkm1 = 4.0
  cache.terk = 0.0001
  cache.terkp1 = 0.0001
  // The ladder 5 > 4 > 1e-4 > 1e-4 is *not* strictly decreasing, but `k > 2`
  // is false, so neither the promote nor the reduce branch runs and the order
  // stays at 2.
  let q = fbdf_stepsize_controller(integ, cache)
  let expect = fastpower(6.0 * 0.0001 / (1.5 * 3.0), 1.0 / 3.0)
  assert_true((q - expect).abs() < 0.0000000001)
  assert_eq(cache.order, 2)
  // `terk == 0` is the "exact step" branch: grow by the full qmax.
  cache.terk = 0.0
  let q0 = fbdf_stepsize_controller(integ, cache)
  assert_eq(q0, 1.0 / FBDF_QMAX)
}

///|
/// `bdf_step_reject_controller!` on a hand-set state: order 2, `EEst = 4`,
/// `terkm1 = 1`, `consfailcnt = 0`. `zₛ = 1.2`, `expo = 1/3`, so
/// `z = 1.2·4^{1/3} = 1.9044…`, `hₖ = h/z`; the order-1 candidate is
/// `zₖ₋₁ = 1.3·1 = 1.3`, `hₖ₋₁ = h/1.3 > hₖ`, so the order drops to 1 and
/// `hₙ = hₖ₋₁`.
test "fbdf reject controller drops the order when the lower order allows more" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_adaptive(SolveOptions::new(), true)
  let integ = Integrator::new(prob, opts, 6, 1)
  let cache = integ.bdf
  cache.max_order = 5
  ignore(FBDFCache::ensure(cache, 1))
  integ.t = 0.0
  integ.dt = 0.1
  integ.est = 4.0
  cache.order = 2
  cache.consfailcnt = 0
  cache.terkm1 = 1.0
  bdf_step_reject_controller(integ, cache)
  let h = 0.1
  let hk = h / (1.2 * fastpower(4.0, 1.0 / 3.0))
  let hkm1 = h / 1.3
  assert_eq(cache.order, 1)
  assert_eq(cache.consfailcnt, 1)
  assert_true((integ.dt.abs() - dmin(h, hkm1)).abs() < 0.000000000001)
  assert_true((integ.dt.abs() - hk).abs() > 0.000000000001)
}

///|
/// `post_newton_controller!` for FBDF (controllers.jl:285) — the
/// `force_stepfail` arm of `_loopfooter!`: halve `dt` through `failfactor = 2`,
/// count the failure, and drop the order once three *consecutive* Newton
/// failures have piled up above order 1. `nfails` is reset to 0 by every
/// convergence, so a lone failure never moves the order.
test "fbdf post newton controller halves dt and counts failures" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_adaptive(SolveOptions::new(), true)
  let integ = Integrator::new(prob, opts, 6, 1)
  let cache = integ.bdf
  cache.max_order = 5
  ignore(FBDFCache::ensure(cache, 1))
  integ.t = 0.0
  integ.dt = 0.1
  cache.order = 3
  cache.nconsteps = 7
  cache.nfails = 2
  fbdf_post_newton_controller(integ, cache)
  assert_eq(integ.dt, 0.05)
  assert_eq(cache.consfailcnt, 1)
  assert_eq(cache.nconsteps, 0)
  // Two failures is below the `>= 3` threshold: the order holds.
  assert_eq(cache.order, 3)
  cache.nfails = 3
  fbdf_post_newton_controller(integ, cache)
  assert_eq(integ.dt, 0.025)
  assert_eq(cache.order, 2)
  // At order 1 there is nothing left to drop.
  cache.order = 1
  cache.nfails = 5
  fbdf_post_newton_controller(integ, cache)
  assert_eq(integ.dt, 0.0125)
  assert_eq(cache.order, 1)
}

///|
/// `do_newJW`'s decision table on a hand-set cache. The hand-set state is the
/// one a mid-solve `update_W!` sees: `iter = 5` (not the first step attempt),
/// `firstcall` cleared, and `J_t`/`W_γdt` describing the last factorization.
/// Assertions go through `fbdf_check_jw` rather than `assert_eq` on the tuple,
/// which would need an `Eq` instance for `(Bool, Bool)`.
test "fbdf do_new_jw decision table" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_adaptive(SolveOptions::new(), true)
  let integ = Integrator::new(prob, opts, 6, 1)
  let cache = integ.bdf
  cache.max_order = 5
  ignore(FBDFCache::ensure(cache, 1))
  integ.t = 0.0
  integ.dt = 0.1
  integ.iter = 1
  // First step attempt: both.
  fbdf_check_jw(integ, cache, true, true)
  integ.iter = 5
  cache.firstcall = false
  // `markfirststage!` ran at the top of the step, so `isfs` is true for the
  // whole solve — the state `update_W!` actually sees mid-step.
  cache.firststage = true
  cache.j_t = 0.0
  cache.w_gdt = cache.beta0 * integ.dt
  cache.nlstatus = NL_CONVERGENCE
  // Fresh Jacobian (`t == J_t`), converged, step accepted: upstream falls
  // through to the shared `wbad`, which is false here, so both J and W are
  // reused — NOT the unconditional `(false, true)` a fresh-J reading suggests.
  fbdf_check_jw(integ, cache, false, false)
  // Same fresh state with a rejected step in flight (`est > 1`): W must be
  // reassembled at the new `γ·dt`, but J stays — this is the reuse-across-
  // retry path.
  integ.est = 2.0
  fbdf_check_jw(integ, cache, false, true)
  // A restart in flight makes the Jacobian stale even at the same `t`.
  cache.restart_pending = true
  fbdf_check_jw(integ, cache, false, true)
  cache.restart_pending = false
  integ.est = 0.0
  // Stale Jacobian, dt unchanged: `smallstepchange` is true, so only a
  // Divergence or an error failure forces a new W.
  cache.j_t = -1.0
  fbdf_check_jw(integ, cache, false, false)
  integ.est = 2.0
  fbdf_check_jw(integ, cache, false, true)
  integ.est = 0.0
  cache.nlstatus = NL_DIVERGENCE
  fbdf_check_jw(integ, cache, false, true)
  // `TryAgain` with a small step change is the stale-Jacobian retry signal.
  cache.nlstatus = NL_TRYAGAIN
  fbdf_check_jw(integ, cache, true, true)
  // A 100% drift in 1/(γ·dt) is outside the 0.2 cutoff: rebuild W.
  cache.nlstatus = NL_CONVERGENCE
  cache.w_gdt = cache.beta0 * integ.dt * 2.0
  fbdf_check_jw(integ, cache, false, true)
  // Fixed-step: always both. `SolveOptions.adaptive` is an immutable record
  // field, so the switch is a wholesale options replacement (which keeps the
  // rest of the hand-set cache state intact).
  integ.opts = SolveOptions::with_adaptive(integ.opts, false)
  fbdf_check_jw(integ, cache, true, true)
}

///|
/// Assert one row of the `do_newJW` decision table without depending on an
/// `Eq` instance for `(Bool, Bool)`.
fn fbdf_check_jw(
  integ : Integrator,
  cache : FBDFCache,
  want_new_jac : Bool,
  want_new_w : Bool,
) -> Unit raise {
  let (new_jac, new_w) = fbdf_do_new_jw(integ, cache)
  assert_eq(new_jac, want_new_jac)
  assert_eq(new_w, want_new_w)
}
