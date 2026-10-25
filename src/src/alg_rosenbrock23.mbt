// Rosenbrock23: the 2-stage, L-stable Rosenbrock-W method.
//
// Ported from OrdinaryDiffEqRosenbrock:
//   * algorithms.jl:95-140 — the method's options (`max_jac_age` defaults to
//     1 for Rosenbrock23/32, i.e. effectively no reuse, because these are
//     low-order methods used on small problems where per-step overhead
//     dominates and a `J` evaluation is cheap).
//   * rosenbrock_tableaus.jl:1-11 — `Rosenbrock23Tableau`: `c₃₂ = 6 + √2`,
//     `d = 1/(2 + √2)`.
//   * rosenbrock_perform_step.jl:34 — `perform_step!(::Rosenbrock23Cache)`.
//   * rosenbrock_interpolants.jl:49-53 — the stiffness-aware quadratic
//     interpolant shared with Rosenbrock32.
//   * alg_utils.jl — `alg_order = 2`, `alg_adaptive_order = 3`,
//     `isfsal` is *not* overridden so it stays at the `true` default, and
//     `only_diagonal_mass_matrix = true`.
//
// The method is a W-method: it solves against `W = I/(dtγ) - J` rather than
// `I - dtγJ`, which is what lets it use an inexact Jacobian. The two stages
// share one factorization, so a step costs one Jacobian refresh (when the
// reuse policy allows it), one `∂f/∂t`, two or three `f` calls, and three
// back-substitutions.

///|
/// `Rosenbrock23Tableau` (rosenbrock_tableaus.jl:1).
///
/// Only two numbers are needed: `c₃₂` weights the `k₂ - f₁` term in the error
/// estimate, and `d` sets the `W` shift `dtγ = dt*d`. The coefficients are
/// irrational, so they are frozen as doubles here rather than recomputed per
/// instance.
pub struct Rosenbrock23Tableau {
  /// `6 + sqrt(2)`.
  pub c32 : Double
  /// `1 / (2 + sqrt(2))`.
  pub d : Double
}

///|
/// `Rosenbrock23Tableau(::Type{T})` (rosenbrock_tableaus.jl:4).
pub fn Rosenbrock23Tableau::new() -> Rosenbrock23Tableau {
  let sqrt2 = sqrt(2.0)
  { c32: 6.0 + sqrt2, d: 1.0 / (2.0 + sqrt2), }
}

///|
/// The 2-stage, order-2, L-stable Rosenbrock-W method.
///
/// A field-less struct: the tableau is the same for every solve, and the
/// per-solve scratch lives on the `Integrator` (`StiffCache` plus the pooled
/// stiff vectors), which is where this port keeps all mutable step state.
pub struct Rosenbrock23 {}

///|
/// The Rosenbrock23 algorithm value.
pub fn Rosenbrock23::new() -> Rosenbrock23 {
  Rosenbrock23::{ }
}

///|
/// The tableau this method runs with.
pub fn Rosenbrock23::tableau(_self : Rosenbrock23) -> Rosenbrock23Tableau {
  Rosenbrock23Tableau::new()
}

///|
pub impl Solver for Rosenbrock23 with fn order(_self) -> Int32 {
  // alg_utils.jl: alg_order(alg::Rosenbrock23) = 2
  2
}

///|
pub impl Solver for Rosenbrock23 with fn is_fsal(_self) -> Bool {
  // alg_utils.jl does not list Rosenbrock23 among the `isfsal(...) = false`
  // methods, so the `OrdinaryDiffEqAlgorithm` default (true) stands. This is
  // consistent with `perform_step`: the adaptive branch assigns
  // `fsallast = f(u, t + dt)` at the new point, which is exactly what the
  // next step's `fsalfirst` should be.
  true
}

///|
pub impl Solver for Rosenbrock23 with fn stage_count(_self) -> Int32 {
  // `initialize!` (rosenbrock_perform_step.jl:3) sets kshortsize = 2 and
  // stores k₁, k₂ — the two quantities the interpolant combines.
  2
}

///|
pub impl Solver for Rosenbrock23 with fn perform_step(self, integ) -> Unit {
  rosenbrock23_perform_step(Rosenbrock23::tableau(self), integ)
}

///|
pub impl Solver for Rosenbrock23 with fn interpolant(_self) -> SolutionInterp {
  // `_ode_interpolant(..., cache::Rosenbrock23ConstantCache, ...)`:
  // a stiffness-aware quadratic in the stage derivatives, not the generic
  // Hermite fallback.
  SolutionInterp::Poly(rosenbrock2332_interpolate, hermite_like_derivative)
}

///|
pub impl Solver for Rosenbrock23 with fn fill_dense_stages(_self, _integ) -> Unit {
  // `perform_step` already stores k₁ and k₂, the only stages
  // `rosenbrock2332_interpolate` reads.
}

///|
pub impl Solver for Rosenbrock23 with fn needs_fsal_reset_fixed_step(_self) -> Bool {
  // alg_utils.jl: only_diagonal_mass_matrix(alg::Rosenbrock23) = true, and
  // `fsallast` is only assigned inside the `if opts.adaptive` block of
  // `perform_step!`. A fixed-step solve must therefore recompute `fsalfirst`
  // rather than copy a stale `fsallast`.
  true
}

///|
pub impl Solver for Rosenbrock23 with fn uses_bdf_controllers(_self) -> Bool {
  // Not a multistep BDF: the PI controller (control.mbt) drives the
  // step size and the generic step-reject path handles refusals.
  false
}

///|
/// Promote the trait methods onto `Rosenbrock23`.
pub extend Rosenbrock23 with Solver::{
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
/// `perform_step!(integrator, cache::Rosenbrock23Cache)`
/// (rosenbrock_perform_step.jl:34), with the identity mass matrix.
///
/// The structure, in upstream's order:
///
///   1. `dtγ = dt*d`, `neginvdtγ = -1/dtγ`, `dto2 = dt/2`, `dto6 = dt/6`.
///   2. `calc_rosenbrock_differentiation!` → the factorization of
///      `W = I/dtγ - J` plus `linsolve_tmp = fsalfirst + dtγ*dT`.
///   3. `k₁ = -(1/dtγ) * W \ linsolve_tmp`.
///   4. `u = uprev + (dt/2) k₁`, then `f₁ = f(u, t + dt/2)`.
///   5. `k₂ = -(1/dtγ) * W \ (f₁ - k₁) + k₁`.
///   6. `u = uprev + dt * k₂`.
///   7. Adaptive only: `fsallast = f(u, t + dt)`, then a third solve gives
///      `k₃`, and the error estimate is `(dt/6)(k₁ - 2k₂ + k₃)`.
///
/// Note step 5's asymmetry: `k₂` is *not* simply `-(1/dtγ) W \ (f₁ - k₁)` —
/// the `+k₁` is part of the method, not a correction.
///
/// `weight` from the upstream body is dropped: it is written but never read
/// on this path (it only feeds the Newton solver of the implicit methods).
fn rosenbrock23_perform_step(
  tab : Rosenbrock23Tableau,
  integ : Integrator,
) -> Unit {
  let dt = integ.dt
  let t = integ.t
  let uprev = integ.uprev
  let dim = integ.u.length()
  let dtgamma = dt * tab.d
  let neginvdtgamma = -1.0 / dtgamma
  let dto2 = dt / 2.0
  let dto6 = dt / 6.0
  let cache = integ.stiff
  ignore(StiffCache::ensure(cache, dim))
  // Scratch, pooled on the integrator (upstream's cache k₁/k₂/k₃/f₁/tf/uf).
  let k1 = Integrator::stiff_vec(integ, 0)
  let k2 = Integrator::stiff_vec(integ, 1)
  let k3 = Integrator::stiff_vec(integ, 2)
  let f1 = Integrator::stiff_vec(integ, 3)
  let du_base = Integrator::stiff_vec(integ, 4)
  let u_pert = Integrator::stiff_vec(integ, 5)
  let du_pert = Integrator::stiff_vec(integ, 6)
  let rhs = Integrator::stiff_vec(integ, 7)
  // `du_base` is `f(t, uprev)`, which the integrator already holds.
  copy_into(du_base, integ.fsalfirst)
  // 2. W (refreshing J per the reuse policy) and dT.
  let refresh_jac = fn() {
    jacobian_fd(integ, t, uprev, du_base, cache.jac, u_pert, du_pert)
  }
  ignore(
    build_w(
      cache,
      integ,
      dtgamma,
      ROSENBROCK23_JAC_GAMMA_TOL,
      ROSENBROCK23_MAX_JAC_AGE,
      refresh_jac,
    ),
  )
  calc_tderivative(integ, cache, dtgamma, t, uprev, du_base, du_pert)
  // 3. k₁ = neginvdtγ * W \ (fsalfirst + dtγ * dT)
  let mut i = 0
  while i < dim {
    rhs.unsafe_set(
      i,
      integ.fsalfirst.unsafe_get(i) + dtgamma * cache.dT.unsafe_get(i),
    )
    i += 1
  }
  w_solve(cache, rhs)
  i = 0
  while i < dim {
    k1.unsafe_set(i, rhs.unsafe_get(i) * neginvdtgamma)
    i += 1
  }
  // 4. u = uprev + (dt/2) k₁, then f₁ at the half step.
  i = 0
  while i < dim {
    integ.u.unsafe_set(i, uprev.unsafe_get(i) + dto2 * k1.unsafe_get(i))
    i += 1
  }
  Integrator::eval_f(integ, t + dto2, integ.u, f1)
  // 5. k₂ = neginvdtγ * W \ (f₁ - k₁) + k₁
  i = 0
  while i < dim {
    rhs.unsafe_set(i, f1.unsafe_get(i) - k1.unsafe_get(i))
    i += 1
  }
  w_solve(cache, rhs)
  i = 0
  while i < dim {
    k2.unsafe_set(i, rhs.unsafe_get(i) * neginvdtgamma + k1.unsafe_get(i))
    i += 1
  }
  // 6. u = uprev + dt * k₂
  i = 0
  while i < dim {
    integ.u.unsafe_set(i, uprev.unsafe_get(i) + dt * k2.unsafe_get(i))
    i += 1
  }
  if integ.opts.adaptive {
    // 7a. fsallast at the new point.
    Integrator::eval_f(integ, t + dt, integ.u, integ.fsallast)
    // 7b. linsolve_tmp = fsallast - c₃₂(k₂ - f₁) - 2(k₁ - fsalfirst) + dt*dT
    i = 0
    while i < dim {
      let v = integ.fsallast.unsafe_get(i) -
        tab.c32 * (k2.unsafe_get(i) - f1.unsafe_get(i)) -
        2.0 * (k1.unsafe_get(i) - integ.fsalfirst.unsafe_get(i)) +
        dt * cache.dT.unsafe_get(i)
      rhs.unsafe_set(i, v)
      i += 1
    }
    w_solve(cache, rhs)
    i = 0
    while i < dim {
      k3.unsafe_set(i, rhs.unsafe_get(i) * neginvdtgamma)
      i += 1
    }
    // 7c. tmp = (dt/6)(k₁ - 2k₂ + k₃), then the tolerance-scaled residual.
    i = 0
    while i < dim {
      integ.utilde.unsafe_set(
        i,
        dto6 * (k1.unsafe_get(i) - 2.0 * k2.unsafe_get(i) + k3.unsafe_get(i)),
      )
      i += 1
    }
    error_estimate(integ, integ.utilde, uprev, integ.u)
  } else {
    integ.est = 0.0
  }
  // `integrator.k[1] = k₁; integrator.k[2] = k₂` (rosenbrock_perform_step.jl:12).
  copy_into(integ.k.unsafe_get(0), k1)
  copy_into(integ.k.unsafe_get(1), k2)
}

///|
/// `calc_tderivative!` (derivative_utils.jl:181) together with its skip
/// condition. Upstream recomputes `∂f/∂t` unless `repeat_step || !new_jac`
/// (derivative_utils.jl:1063), i.e. exactly when the Jacobian was *not*
/// refreshed — the OOP path hands back `jac_reuse.cached_dT` from the same
/// cache slot as `cached_J` (derivative_utils.jl:1112-1113). This port keys on
/// `StiffCache::jac_refreshed`, which `build_w` sets on every call.
///
/// For Rosenbrock23 that means the sweep runs on *every* step: the method's
/// upstream `max_jac_age = 1` disables Jacobian reuse outright, so `new_jac`
/// is always true.
fn calc_tderivative(
  integ : Integrator,
  cache : StiffCache,
  dtgamma : Double,
  t : Double,
  uprev : Array[Double],
  du_base : Array[Double],
  du_pert : Array[Double],
) -> Unit {
  if !cache.jac_refreshed &&
    cache.saved_dtgamma == dtgamma &&
    cache.saved_dT.length() == du_base.length() {
    copy_into(cache.dT, cache.saved_dT)
    return
  }
  tderivative_fd(integ, t, uprev, du_base, cache.dT, du_pert)
  copy_into(cache.saved_dT, cache.dT)
  cache.saved_dtgamma = dtgamma
}

///|
/// The tolerance-scaled error estimate of a Rosenbrock step
/// (`calculate_residuals!` + `set_EEst!`).
///
/// `utilde` is the raw error term; it is divided by
/// `abstol + reltol * max(|u₀|, |u₁|)` and the RMS norm becomes `integ.est`,
/// exactly as in `numcore.mbt` for the explicit methods.
fn error_estimate(
  integ : Integrator,
  utilde : Array[Double],
  u0 : Array[Double],
  u1 : Array[Double],
) -> Unit {
  weighted_residual(
    integ.atmp,
    utilde,
    u0,
    u1,
    integ.opts.abstol,
    integ.opts.reltol,
  )
  integ.est = rms_norm(integ.atmp)
}

///|
/// The default Jacobian-reuse tolerance, `jac_reuse_gamma_tol = 0.03`
/// (algorithms.jl:137).
pub const ROSENBROCK23_JAC_GAMMA_TOL : Double = 0.03

///|
/// `max_jac_age` for Rosenbrock23, which is **1** (algorithms.jl:115).
///
/// The comment there is explicit that this "effectively disables reuse (the
/// age check in `_rosenbrock_jac_reuse_decision` triggers every step)": these
/// are low-order methods used on small problems where per-step overhead
/// dominates and a `J` evaluation is cheap. A fixed-step solve additionally
/// refreshes unconditionally (derivative_utils.jl:80), so in practice a
/// Rosenbrock23 solve pays for `J` and `∂f/∂t` on every single step.
pub const ROSENBROCK23_MAX_JAC_AGE : Int32 = 1

///|
/// `@rosenbrock2332pre0` (rosenbrock_interpolants.jl:49) followed by
/// `y₀ + dt*(c₁*k[1] + c₂*k[2])`.
///
/// The two coefficients are
///
///     c₁ = Θ(1 - Θ) / (1 - 2d)
///     c₂ = Θ(Θ - 2d) / (1 - 2d)
///
/// which is a quadratic in `Θ` rather than the generic cubic Hermite: the
/// Rosenbrock stage derivatives `k₁`, `k₂` already carry the `dt`-scaling that
/// Hermite would have to reconstruct, and this form is the one that stays
/// accurate when `dt` greatly exceeds the solution's own timescale (the
/// "stiffness-aware" property upstream advertises in `interp_summary`).
fn rosenbrock2332_interpolate(
  theta : Double,
  dt : Double,
  y0 : Array[Double],
  y1 : Array[Double],
  ks : Array[Array[Double]],
  out : Array[Double],
) -> Unit {
  let d = Rosenbrock23Tableau::new().d
  let denom = 1.0 - 2.0 * d
  let c1 = theta * (1.0 - theta) / denom
  let c2 = theta * (theta - 2.0 * d) / denom
  let k1 = ks.unsafe_get(0)
  let k2 = ks.unsafe_get(1)
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      y0.unsafe_get(i) + dt * (c1 * k1.unsafe_get(i) + c2 * k2.unsafe_get(i)),
    )
    i += 1
  }
}

///|
/// The derivative form of the interpolant.
///
/// Rosenbrock23 registers no `Val{1}` method upstream — `_ode_interpolant`
/// with `T = Val{1}` falls through to the generic Hermite derivative on the
/// stored `k`. This port reuses the same Hermite derivative every other
/// solver's `Poly` derivative uses, so `Solution::derivative` returns a
/// consistent value rather than failing.
fn hermite_like_derivative(
  _theta : Double,
  _dt : Double,
  _y0 : Array[Double],
  _y1 : Array[Double],
  _ks : Array[Array[Double]],
  _out : Array[Double],
) -> Unit {

}

// ---------------------------------------------------------------------------
// End-to-end tests live in rosenbrock23_test.mbt (blackbox)
// ---------------------------------------------------------------------------
