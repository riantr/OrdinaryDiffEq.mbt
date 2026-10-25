// Dense differentiation and the W operator shared by every stiff method.
//
// Ported from OrdinaryDiffEqDifferentiation:
//   * derivative_wrappers.jl:85-113 — the `nf` accounting for a dense
//     differentiation pass (`num_of_chunks` for forward-mode AD,
//     `maximum(colorvec) + 1` for a forward finite-difference sweep).
//   * derivative_utils.jl:181 `calc_tderivative!` — `linsolve_tmp =
//     fsalfirst + dtd1 * dT`, where `dT` is `∂f/∂t` at `(t, uprev)`.
//   * derivative_utils.jl:681 `jacobian2W!` — for a scalar (identity) mass
//     matrix, `W = M/(dtgamma) - J` reduces to `W = I/(dtgamma) - J`, i.e.
//     the diagonal is shifted by `+1/dtgamma` on top of `copyto!(W, J)`.
//
// This port has no automatic differentiation, so the default path is a
// forward finite-difference Jacobian, which is the same `AutoFiniteDiff`
// branch upstream takes when the user asks for it: `nf += ncolors + 1` for
// `fdtype = :forward`. A user-supplied analytic Jacobian bypasses the
// differentiation entirely and costs nothing, matching upstream's
// `has_jac(f)` fast path.
//
// Zero register allocation was an M0 constraint, so the whole differentiation
// is driven by preallocated buffers held by `StiffCache`.

///|
/// A dense Jacobian plus the factorization of the W operator, reused across
/// steps (Julia's `RosenbrockCache.W` / `.J` / `.linsolve`).
///
/// `w` holds the LU factors of `W = I/(dt*gamma) - J` once `lu_factor` has
/// run; it is rebuilt only when the Jacobian is refreshed or `dt*gamma`
/// changes. `saved_dT` / `saved_dtgamma` implement the same reuse idea for
/// `∂f/∂t`, which only depends on `(t, uprev)` and the step size that scales
/// it.
///
/// The struct is filled by `StiffCache::ensure`, which knows `dim`; until
/// then every buffer is empty and `ready` is false.
pub(all) struct StiffCache {
  /// Dense Jacobian, row-major `n*n`, entry `(i, j)` at `i*n + j`.
  pub mut jac : Array[Double]
  /// Dense `W = I/(dt*gamma) - J`, overwritten by its LU factors.
  pub mut w : Array[Double]
  /// Pivot rows from `lu_factor`.
  pub mut ipiv : Array[Int32]
  /// `∂f/∂t` at the current `(t, uprev)`.
  pub mut dT : Array[Double]
  /// `∂f/∂t` as computed for `saved_dtgamma`, reused exactly when `J` itself
  /// is reused. Upstream stores the two together (`jac_reuse.cached_J` /
  /// `.cached_dT`, derivative_utils.jl:1098-1099) and hands both back from
  /// `_rosenbrock_jac_reuse_decision`'s reuse branch, so this port couples
  /// them the same way rather than keying on `t`/`u` alone.
  pub mut saved_dT : Array[Double]
  /// The `dt*gamma` that `saved_dT` belongs to.
  pub mut saved_dtgamma : Double
  /// `build_w` refreshed `J` on the call that last ran. `∂f/∂t` is only
  /// reusable when this is false, mirroring `calc_tderivative!`'s
  /// `repeat_step || !new_jac` skip condition (derivative_utils.jl:1063).
  pub mut jac_refreshed : Bool
  /// Value of `Integrator::success_iter` when `jac` was last recomputed.
  /// The age check `(naccept - last_naccept) >= max_jac_age`
  /// (derivative_utils.jl:161) compares against this.
  pub mut last_naccept : Int32
  /// `J` currently held in `jac` was computed for this `dt*gamma`.
  pub mut jac_dtgamma : Double
  /// `jac` is valid for the current `(t, uprev)`.
  pub mut jac_valid : Bool
  /// Buffers are allocated and sized for `dim`.
  pub mut ready : Bool
  /// Number of state components the buffers are sized for.
  pub mut dim : Int32
}

///|
/// An empty cache. `StiffCache::ensure` allocates it on first use, so a
/// solve that never runs a stiff method never pays for the buffers.
pub fn StiffCache::new() -> StiffCache {
  {
    jac: [],
    w: [],
    ipiv: [],
    dT: [],
    saved_dT: [],
    saved_dtgamma: 0.0,
    jac_refreshed: false,
    last_naccept: 0,
    jac_dtgamma: 0.0,
    jac_valid: false,
    ready: false,
    dim: 0,
  }
}

///|
/// Allocate (or resize) the cache for a problem of `dim` components.
///
/// Returns true when the buffers were (re)allocated, so the caller knows the
/// Jacobian is no longer valid. This mirrors Julia's cache constructor, which
/// sizes every scratch buffer from `size(u)` once per solve.
pub fn StiffCache::ensure(self : StiffCache, dim : Int32) -> Bool {
  if self.ready && self.dim == dim {
    return false
  }
  let n = dim
  self.jac = Array::make(n * n, 0.0)
  self.w = Array::make(n * n, 0.0)
  self.ipiv = Array::make(dim, 0)
  self.dT = Array::make(n, 0.0)
  self.saved_dT = Array::make(n, 0.0)
  self.saved_dtgamma = 0.0
  self.jac_refreshed = false
  self.last_naccept = 0
  self.jac_dtgamma = 0.0
  self.jac_valid = false
  self.ready = true
  self.dim = dim
  true
}

///|
/// Forget the cached Jacobian and `dT`, forcing a recompute on the next step.
///
/// This is `reset_fsal!`-adjacent housekeeping: Julia drops the factorization
/// whenever `dt` changes enough to matter or the state jumps (an accepted
/// discontinuity, a changed parameter set). Here the only trigger is an
/// explicit call, used by the error-retry path where a rejected step changes
/// `dt` and the old factorization no longer matches.
pub fn StiffCache::invalidate(self : StiffCache) -> Unit {
  self.jac_valid = false
  self.saved_dtgamma = 0.0
}

///|
/// The finite-difference step for a component of the state.
///
/// Forward differences use `sqrt(eps) * max(1, |u_i|)`, the standard
/// truncation-vs-cancellation balance: a step of `eps^(1/2)` makes the
/// truncation error `O(h)` comparable to the roundoff `O(eps/h)`, which is
/// the best a first-order difference can do. Upstream reaches the same
/// constant through `AutoFiniteDiff`'s own default.
pub fn fd_step(x : Double) -> Double {
  dmax(1.0, x.abs()) * 1.4901161193847656e-8 // sqrt(eps) for binary64
}

///|
/// The finite-difference step for time.
///
/// Time is a scalar with no natural magnitude, so it is scaled the same way
/// as a state component. `AutoFiniteDiff` additionally carries a `dir` field
/// that OrdinaryDiffEq sets to `sign(dt)` (stiff_addsteps.jl) so the sweep
/// runs along the direction of integration; forward differences are
/// direction-agnostic, but the caller still passes `dt` and this keeps the
/// two ports comparable.
pub fn fd_step_t(t : Double, dt : Double) -> Double {
  let scale = if dt == 0.0 { 1.0 } else { dt.abs() }
  dmax(dmax(1.0, t.abs()), scale) * 1.4901161193847656e-8
}

///|
/// Fill `jac` with the dense forward-difference Jacobian of `f` at
/// `(t, u)`, evaluating through `f_tmp` (the caller's scratch) and `du_base`
/// (the unperturbed derivative, which the caller already has as
/// `fsalfirst`).
///
/// Column `j` is `(f(u + h_j e_j) - f(u)) / h_j`, so the sweep costs exactly
/// `n` extra evaluations — upstream's `maximum(colorvec) + 1` with a dense
/// (one color per column) sparsity pattern, where the `+1` is the base
/// evaluation this port already has in hand.
///
/// `jacobian` counts those evaluations into `integ.nf`.
pub fn jacobian_fd(
  integ : Integrator,
  t : Double,
  u : Array[Double],
  du_base : Array[Double],
  jac : Array[Double],
  u_pert : Array[Double],
  du_pert : Array[Double],
) -> Unit {
  let n = u.length()
  copy_into(u_pert, u)
  let mut j = 0
  while j < n {
    let uj = u.unsafe_get(j)
    let h = fd_step(uj)
    u_pert.unsafe_set(j, uj + h)
    Integrator::eval_f(integ, t, u_pert, du_pert)
    u_pert.unsafe_set(j, uj)
    let inv_h = 1.0 / h
    let mut i = 0
    while i < n {
      jac.unsafe_set(
        i * n + j,
        (du_pert.unsafe_get(i) - du_base.unsafe_get(i)) * inv_h,
      )
      i += 1
    }
    j += 1
  }
}

///|
/// Fill `jac` from a user-supplied analytic Jacobian `dfdu(t, u, jac)`.
///
/// The buffer arrives zeroed and the callback is contractually required to
/// write every entry, so nothing is pre-filled here. No evaluations are
/// counted: an analytic Jacobian is not a right-hand-side call, which is
/// exactly why upstream's `has_jac(f)` branch skips `increment_nf!` too.
pub fn jacobian_analytic(
  t : Double,
  u : Array[Double],
  dfdu : (Double, Array[Double], Array[Double]) -> Unit,
  jac : Array[Double],
) -> Unit {
  dfdu(t, u, jac)
}

///|
/// Fill `dt_vec` with `∂f/∂t` at `(t, u)` by a forward difference in time:
/// `(f(t + h, u) - f(t, u)) / h`.
///
/// One extra evaluation, counted into `integ.nf` — upstream's
/// `calc_tderivative!` (derivative_utils.jl:221) does the same when it falls
/// back to finite differences.
pub fn tderivative_fd(
  integ : Integrator,
  t : Double,
  u : Array[Double],
  du_base : Array[Double],
  dt_vec : Array[Double],
  du_pert : Array[Double],
) -> Unit {
  let n = u.length()
  let h = fd_step_t(t, integ.dt)
  Integrator::eval_f(integ, t + h, u, du_pert)
  let inv_h = 1.0 / h
  let mut i = 0
  while i < n {
    dt_vec.unsafe_set(
      i,
      (du_pert.unsafe_get(i) - du_base.unsafe_get(i)) * inv_h,
    )
    i += 1
  }
}

///|
/// `w = J - I/dtgamma` for the identity mass matrix
/// (`jacobian2W!`, derivative_utils.jl:681, scalar-mass-matrix branch).
///
/// The whole matrix is a copy of `jac` minus `1/dtgamma` on the diagonal —
/// upstream's `copyto!(W, J)` followed by `W[i,i] = muladd(λ, invdtgamma,
/// J[i,i])`. The sign is easy to get wrong: `λ = -_scalar_massmatrix_λ(mm)`
/// (derivative_utils.jl:692) negates the mass matrix's own scalar, and for
/// `I` that scalar is `1`, so `λ = -1` and the diagonal shift is
/// **negative**. `calc_W` (derivative_utils.jl:1006) states the same thing
/// directly as `W = J - mass_matrix * inv(dtgamma)`.
///
/// The sign is what makes the method L-stable: `W ≈ -I/dtgamma` for a stiff
/// eigenvalue, and the `neginvdtγ` factors in `perform_step` restore the
/// correct overall scaling. Flipping this sign yields a method that looks
/// plausible and is silently wrong.
pub fn jacobian2w(
  w : Array[Double],
  jac : Array[Double],
  n : Int32,
  dtgamma : Double,
) -> Unit {
  let nn = n
  let inv_dtgamma = 1.0 / dtgamma
  let mut i = 0
  while i < nn {
    let row = i * nn
    let mut j = 0
    while j < nn {
      w.unsafe_set(row + j, jac.unsafe_get(row + j))
      j += 1
    }
    w.unsafe_set(row + i, w.unsafe_get(row + i) - inv_dtgamma)
    i += 1
  }
}

///|
/// Factor `cache.w` in place, refreshing `cache.jac` first when it is stale.
///
/// Returns false when the factorization failed (singular `W`), which is the
/// signal for a step that cannot be taken. `refresh_jac` is the closure that
/// computes the Jacobian — the caller owns how (finite differences vs. an
/// analytic callback), this only sequences it against the reuse flags.
///
/// Reuse rules, mirroring `_rosenbrock_jac_reuse_decision`
/// (derivative_utils.jl:46-171) for a W-method on a pure ODE:
///
///   * `!opts.adaptive` always refreshes (derivative_utils.jl:80) — a
///     prescribed timestep makes `J` reuse pointless and upstream calls this
///     out explicitly because it caused IIP/OOP inconsistency.
///   * `EEst > 1` on entry means the previous step was rejected; `dt` is about
///     to change, so refresh (derivative_utils.jl:144).
///   * a relative change in `dt*gamma` above `gamma_tol` (default
///     `jac_reuse_gamma_tol = 0.03`) refreshes (derivative_utils.jl:156).
///   * `max_jac_age` accepted steps without a refresh force one
///     (derivative_utils.jl:161). Rosenbrock23/32 default to `1`, which fires
///     every step and so disables reuse entirely; the higher-order W-methods
///     default to `20`.
///
/// Note the deliberate asymmetry with `∂f/∂t`: `W` is rebuilt from the cached
/// `J` on every step (cheap — an `O(n²)` diagonal shift plus an `O(n³)` LU),
/// while `J` itself is what the age/tolerance policy governs.
pub fn build_w(
  cache : StiffCache,
  integ : Integrator,
  dtgamma : Double,
  gamma_tol : Double,
  max_jac_age : Int32,
  refresh_jac : () -> Unit,
) -> Bool {
  let n = cache.dim
  let mut want_refresh = !cache.jac_valid
  if !want_refresh && !integ.opts.adaptive {
    want_refresh = true
  }
  if !want_refresh && integ.est > 1.0 {
    want_refresh = true
  }
  if !want_refresh {
    let prev = cache.jac_dtgamma
    let change = if prev == 0.0 {
      1.0
    } else {
      (dtgamma - prev).abs() / prev.abs()
    }
    if change > gamma_tol {
      want_refresh = true
    }
  }
  if !want_refresh && integ.success_iter - cache.last_naccept >= max_jac_age {
    want_refresh = true
  }
  if want_refresh {
    refresh_jac()
    cache.jac_valid = true
    cache.jac_dtgamma = dtgamma
    cache.jac_refreshed = true
    cache.last_naccept = integ.success_iter
  } else {
    cache.jac_refreshed = false
  }
  jacobian2w(cache.w, cache.jac, n, dtgamma)
  lu_factor(cache.w, n, cache.ipiv)
}

///|
/// Solve `W x = b` with the factorization already in `cache.w`/`cache.ipiv`.
///
/// `b` is overwritten with `x`. This is `dolinsolve(...; b = ...)` on the
/// second and third right-hand sides of a step: upstream refactors only when
/// `new_W` is true and reuses the factorization otherwise, which is where
/// Rosenbrock methods get their "two `f` calls and three back-substitutions
/// per step" cost profile.
pub fn w_solve(cache : StiffCache, b : Array[Double]) -> Unit {
  lu_solve(cache.w, cache.ipiv, b)
}

// ---------------------------------------------------------------------------
// In-package invariants
// ---------------------------------------------------------------------------

///|
/// The forward-difference Jacobian recovers a linear map exactly up to the
/// finite-difference truncation, which for an affine `f` is only the
/// roundoff of the subtraction.
test "jacobian_fd recovers the diagonal of a linear decay" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0, 2.0], fn(_t, u, du) {
    du.unsafe_set(0, -3.0 * u.unsafe_get(0))
    du.unsafe_set(1, -5.0 * u.unsafe_get(1))
  })
  let integ = Integrator::new(prob, SolveOptions::new(), 2, 2)
  let u = [1.0, 2.0]
  let du_base = Array::make(2, 0.0)
  Integrator::eval_f(integ, 0.0, u, du_base)
  let jac = Array::make(4, 0.0)
  jacobian_fd(
    integ,
    0.0,
    u,
    du_base,
    jac,
    Array::make(2, 0.0),
    Array::make(2, 0.0),
  )
  // The sweep costs exactly n evaluations (no +1: the base is already known).
  assert_eq(integ.nf, 1 + 2)
  assert_true((jac.unsafe_get(0) + 3.0).abs() < 0.000000001)
  assert_true(jac.unsafe_get(1).abs() < 0.000000001)
  assert_true(jac.unsafe_get(2).abs() < 0.000000001)
  assert_true((jac.unsafe_get(3) + 5.0).abs() < 0.000000001)
}

///|
/// The Jacobian of a genuinely coupled system: `u' = A u` with a non-diagonal
/// `A`, checked column by column.
test "jacobian_fd recovers off-diagonal coupling" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0, 0.0], fn(_t, u, du) {
    let a = u.unsafe_get(0)
    let b = u.unsafe_get(1)
    du.unsafe_set(0, -a + 2.0 * b)
    du.unsafe_set(1, 3.0 * a - 4.0 * b)
  })
  let integ = Integrator::new(prob, SolveOptions::new(), 2, 2)
  let u = [1.0, 0.5]
  let du_base = Array::make(2, 0.0)
  Integrator::eval_f(integ, 0.0, u, du_base)
  let jac = Array::make(4, 0.0)
  jacobian_fd(
    integ,
    0.0,
    u,
    du_base,
    jac,
    Array::make(2, 0.0),
    Array::make(2, 0.0),
  )
  assert_true((jac.unsafe_get(0) + 1.0).abs() < 0.000001)
  assert_true((jac.unsafe_get(1) - 2.0).abs() < 0.000001)
  assert_true((jac.unsafe_get(2) - 3.0).abs() < 0.000001)
  assert_true((jac.unsafe_get(3) + 4.0).abs() < 0.000001)
}

///|
/// `∂f/∂t` for `u' = t` is the constant 1, and the forward difference in time
/// must recover it to roundoff (the subtraction is exact here, since the
/// perturbed value is `t + h`).
test "tderivative_fd recovers d/dt of an explicit time dependence" {
  let prob = OdeProblem::iip(0.0, 1.0, [0.0], fn(t, _u, du) {
    du.unsafe_set(0, t)
  })
  let opts = SolveOptions::with_dt(SolveOptions::new(), 0.1)
  let integ = Integrator::new(prob, opts, 1, 1)
  let u = [0.0]
  let du_base = Array::make(1, 0.0)
  Integrator::eval_f(integ, 2.0, u, du_base)
  let dT = Array::make(1, 0.0)
  tderivative_fd(integ, 2.0, u, du_base, dT, Array::make(1, 0.0))
  // One evaluation for the sweep on top of the base one.
  assert_eq(integ.nf, 2)
  assert_true((dT.unsafe_get(0) - 1.0).abs() < 0.000001)
}

///|
/// `jacobian2w` is a copy of `J` with `1/dtgamma` added to the diagonal —
/// the identity-mass-matrix branch of `jacobian2W!`.
test "jacobian2w subtracts only on the diagonal" {
  let jac = [1.0, 2.0, 3.0, 4.0]
  let w = Array::make(4, 0.0)
  jacobian2w(w, jac, 2, 0.5)
  // 1/dtgamma = 2, and the shift is *negative*: W = J - I/dtgamma.
  assert_eq(w.unsafe_get(0), -1.0)
  assert_eq(w.unsafe_get(1), 2.0)
  assert_eq(w.unsafe_get(2), 3.0)
  assert_eq(w.unsafe_get(3), 2.0)
}

///|
/// `build_w` reuses the Jacobian while `dt*gamma` is stable and refreshes it
/// once the relative change passes the tolerance — the reuse policy that
/// `jac_reuse_gamma_tol` implements upstream (derivative_utils.jl:156).
test "build_w reuses the jacobian until dt moves too far" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  // Adaptive, because a fixed-step solve refreshes unconditionally
  // (derivative_utils.jl:80).
  let opts = SolveOptions::with_adaptive(
    SolveOptions::with_dt(SolveOptions::new(), 0.1),
    true,
  )
  let integ = Integrator::new(prob, opts, 1, 2)
  let cache = StiffCache::new()
  assert_true(cache.ready == false)
  assert_true(StiffCache::ensure(cache, 1))
  assert_true(cache.ready)
  let u = [1.0]
  let du_base = Array::make(1, 0.0)
  Integrator::eval_f(integ, 0.0, u, du_base)
  let u_pert = Array::make(1, 0.0)
  let du_pert = Array::make(1, 0.0)
  let refresh = fn() {
    jacobian_fd(
      integ,
      integ.t,
      integ.uprev,
      du_base,
      cache.jac,
      u_pert,
      du_pert,
    )
  }
  // max_jac_age = 20 here, so the age check stays out of the way for the
  // handful of calls below and the tolerance rule is what is under test.
  // First factorization: the Jacobian must be computed.
  assert_true(build_w(cache, integ, 0.1, 0.03, 20, refresh))
  assert_true(cache.jac_valid)
  assert_true(cache.jac_refreshed)
  let nf_after_first = integ.nf
  // Same dtgamma: reuse, so nf does not move.
  assert_true(build_w(cache, integ, 0.1, 0.03, 20, refresh))
  assert_eq(integ.nf, nf_after_first)
  assert_true(!cache.jac_refreshed)
  // A 1% change is inside the tolerance.
  assert_true(build_w(cache, integ, 0.101, 0.03, 20, refresh))
  assert_eq(integ.nf, nf_after_first)
  // A 10% change is outside it.
  assert_true(build_w(cache, integ, 0.111, 0.03, 20, refresh))
  assert_eq(integ.nf, nf_after_first + 1)
  // Invalidate forces a rebuild even at the same dtgamma.
  StiffCache::invalidate(cache)
  assert_true(build_w(cache, integ, 0.111, 0.03, 20, refresh))
  assert_eq(integ.nf, nf_after_first + 2)
}

///|
/// The two unconditional-refresh branches of `build_w`: a rejected previous
/// step (`EEst > 1`) and a fixed-step solve. Both correspond to explicit
/// early returns in `_rosenbrock_jac_reuse_decision`
/// (derivative_utils.jl:80 and :144).
test "build_w refreshes on rejection and on a fixed-step solve" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let mk = fn(adaptive : Bool) {
    let opts = SolveOptions::with_adaptive(
      SolveOptions::with_dt(SolveOptions::new(), 0.1),
      adaptive,
    )
    let integ = Integrator::new(prob, opts, 1, 2)
    let cache = StiffCache::new()
    ignore(StiffCache::ensure(cache, 1))
    let du_base = Array::make(1, 0.0)
    Integrator::eval_f(integ, 0.0, integ.uprev, du_base)
    let u_pert = Array::make(1, 0.0)
    let du_pert = Array::make(1, 0.0)
    let refresh = fn() {
      jacobian_fd(
        integ,
        integ.t,
        integ.uprev,
        du_base,
        cache.jac,
        u_pert,
        du_pert,
      )
    }
    (integ, cache, refresh)
  }
  // Adaptive, accepted so far: the first call refreshes, the second reuses.
  let (a_integ, a_cache, a_refresh) = mk(true)
  ignore(build_w(a_cache, a_integ, 0.1, 0.03, 20, a_refresh))
  let a_nf = a_integ.nf
  ignore(build_w(a_cache, a_integ, 0.1, 0.03, 20, a_refresh))
  assert_eq(a_integ.nf, a_nf)
  // Now the previous step failed: EEst = 2 > 1 forces a rebuild even at an
  // unchanged dtgamma.
  a_integ.est = 2.0
  ignore(build_w(a_cache, a_integ, 0.1, 0.03, 20, a_refresh))
  assert_eq(a_integ.nf, a_nf + 1)
  // Fixed-step: refreshes every time, twice over.
  let (f_integ, f_cache, f_refresh) = mk(false)
  ignore(build_w(f_cache, f_integ, 0.1, 0.03, 20, f_refresh))
  let f_nf = f_integ.nf
  ignore(build_w(f_cache, f_integ, 0.1, 0.03, 20, f_refresh))
  assert_eq(f_integ.nf, f_nf + 1)
}

///|
/// `max_jac_age = 1` (Rosenbrock23's upstream default, algorithms.jl:115)
/// fires the age check on every accepted step, which is what "effectively
/// disables reuse" means: two consecutive accepted steps, one `J` each.
test "build_w honours max_jac_age" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_adaptive(
    SolveOptions::with_dt(SolveOptions::new(), 0.1),
    true,
  )
  let integ = Integrator::new(prob, opts, 1, 2)
  let cache = StiffCache::new()
  ignore(StiffCache::ensure(cache, 1))
  let du_base = Array::make(1, 0.0)
  Integrator::eval_f(integ, 0.0, integ.uprev, du_base)
  let u_pert = Array::make(1, 0.0)
  let du_pert = Array::make(1, 0.0)
  let refresh = fn() {
    jacobian_fd(
      integ,
      integ.t,
      integ.uprev,
      du_base,
      cache.jac,
      u_pert,
      du_pert,
    )
  }
  ignore(build_w(cache, integ, 0.1, 0.03, 1, refresh))
  let nf = integ.nf
  // success_iter has not moved, so the age is 0 and J is still reused.
  ignore(build_w(cache, integ, 0.1, 0.03, 1, refresh))
  assert_eq(integ.nf, nf)
  // One accepted step later the age reaches 1 and J is rebuilt.
  integ.success_iter = 1
  ignore(build_w(cache, integ, 0.1, 0.03, 1, refresh))
  assert_eq(integ.nf, nf + 1)
}

///|
/// `w_solve` solves against the factorization `build_w` left behind: for
/// `u' = -u` at `dt = 0.1`, `W = -1 + 10 = 9`, so `W x = b` gives `x = b/9`.
test "w_solve uses the cached factorization" {
  let prob = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let opts = SolveOptions::with_dt(SolveOptions::new(), 0.1)
  let integ = Integrator::new(prob, opts, 1, 2)
  let cache = StiffCache::new()
  ignore(StiffCache::ensure(cache, 1))
  let du_base = Array::make(1, 0.0)
  Integrator::eval_f(integ, 0.0, integ.uprev, du_base)
  let refresh = fn() {
    jacobian_fd(
      integ,
      integ.t,
      integ.uprev,
      du_base,
      cache.jac,
      Array::make(1, 0.0),
      Array::make(1, 0.0),
    )
  }
  assert_true(build_w(cache, integ, 0.1, 0.03, 20, refresh))
  let b = [-11.0]
  w_solve(cache, b)
  assert_true((b.unsafe_get(0) - 1.0).abs() < 0.000000001)
}
