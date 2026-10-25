// Rodas5P: the 8-stage, 5th-order, stiffly stable Rosenbrock method of
// Steinebach (2023).
//
// Ported from OrdinaryDiffEqRosenbrock:
//   * rosenbrock_tableaus.jl:23-72 — `RODAS5PA` (8x8 `A`), `RODAS5PC` (8x7
//     `C`), `RODAS5Pc` (`c`, 8), `RODAS5Pd` (`d`, 8), `RODAS5PH` (`H`, 3x8)
//     and the derived weights `b` (row 8 of `A` with a `1` appended) and
//     `btilde` (all zero but the last entry).
//   * rosenbrock_perform_step.jl:431 — `perform_step!(::RosenbrockCombined-
//     ConstantCache)`, the generic tableau-driven Rosenbrock step that Rodas5P
//     shares with Rodas4/Rodas5/Rodas6P.
//   * stiff_addsteps.jl:53 — `_ode_addsteps!(::RosenbrockCombinedConstantCache)`,
//     the same tableau applied to the stored stages for dense output.
//   * rosenbrock_interpolants.jl:199-205 / :357-365 — the `interp_order == 3`
//     value and derivative branches.
//   * alg_utils.jl:35,60 — `alg_order(Rodas5P) = 5`, `isfsal(Rodas5P) = false`.
//     `alg_adaptive_order` is not overridden, so it is `alg_order - 1 = 4`
//     (OrdinaryDiffEqCore/alg_utils.jl:725), matching the embedded `btilde`
//     whose only nonzero entry is the last one.
//   * algorithms.jl:87,115,137 — `max_jac_age = 20` and
//     `jac_reuse_gamma_tol = 0.03`, i.e. the full reuse default that
//     distinguishes the higher-order W-methods from Rosenbrock23's `1`.
//
// Why the generic path differs from Rosenbrock23's hand-written one
// --------------------------------------------------------------------
// The Rosenbrock23 step multiplies each solved stage by `neginvdtγ = -1/dtγ`.
// The tableau-driven path does not: its stage solve is
// `W \ -(du + dtd*dT + dtC*ks)`, with the sign written out in front of the
// parenthesised right-hand side (rosenbrock_perform_step.jl:458,493).
//
// The sign is *not* absorbed into `W`. Two upstream paths build the same
// operator with opposite conventions, and mixing them up is the easy mistake:
//
//   * `stiff_addsteps.jl:83` constructs `W = mass_matrix/dtgamma - J` itself
//     and then solves `W \ _vec(linsolve_tmp)` — no minus (line 92).
//   * `perform_step!` gets its `W` from `calc_rosenbrock_differentiation`,
//     i.e. `calc_W` → `jacobian2W!`, which is `W = J - mass_matrix/dtgamma`
//     (derivative_utils.jl:673,1014) — and solves `W \ -_vec(linsolve_tmp)`.
//
// This port's `StiffCache` follows the *second* convention, the same one
// Rosenbrock23 already uses (`W = J - I/dtgamma`, jacobian.mbt:238), and
// `w_solve` consumes it as-is. So the negation stays where upstream put it, on
// the right-hand side:
//
//     k = W \ -(x)   with W = J - I/dtγ
//
// Dropping that minus sign silently flips every stage derivative and makes the
// method diverge — the stages stay finite but grow, so the error explodes
// instead of failing loudly. `w_solve` below is therefore fed `-(x)`.
//
// The `d` vector supplies the `∂f/∂t` scaling, and for Rodas5P
// `d[0] == gamma`, so the first stage's `dtd*dT` term is `dt*gamma*dT` — the
// same combination Rosenbrock23 builds by hand.
//
// A step costs one Jacobian refresh (subject to the reuse policy), one
// `∂f/∂t`, and `num_stages = 8` right-hand-side evaluations sharing a single
// factorization, plus a second one only when `W` has to be rebuilt.

///|
/// The frozen RODAS5P tableau.
///
/// Stored row-major with explicit extents so the flat arrays can be indexed
/// without a second indirection: `a` is `8*8`, `cmat` is `8*7`, `c`/`d`/`b`/
/// `btilde` are 8, and `h` is `h_rows * num_stages = 3*8`.
pub struct Rodas5PTableau {
  /// `RODAS5PA`, the strictly-lower-triangular `A`: entry `(s, i)` (0-based)
  /// at `a[s * 8 + i]`, meaningful only for `i < s`.
  pub a : Array[Double]
  /// `RODAS5PC`, the `C` matrix: entry `(s, i)` at `cmat[s * 7 + i]`,
  /// meaningful only for `i < s`.
  pub cmat : Array[Double]
  /// `RODAS5Pc`, the stage times as fractions of `dt`.
  pub c : Array[Double]
  /// `RODAS5Pd`, the `∂f/∂t` weights, scaled by `dt` per step.
  pub d : Array[Double]
  /// `RODAS5PH`, the dense-output coefficients: `h[j * 8 + i]` builds the
  /// `j`-th stored stage from stage `i`.
  pub h : Array[Double]
  /// `b`: the solution weights, `RODAS5PA[8, 1:8]` with a `1` appended
  /// (rosenbrock_tableaus.jl:67-68). These are the `A` entries, *not*
  /// multiplied by `dt` — the tableau is scaled so `k` already carries `dt`.
  pub b : Array[Double]
  /// `btilde`: the embedded weights, zero but for the last entry. The
  /// resulting embedded solution is `btilde' * k`, which is why the error
  /// estimate needs no `dt` factor.
  pub btilde : Array[Double]
  /// `gamma`, the `W`-shift: `W = I/(dt*gamma) - J`.
  pub gamma : Double
  /// `size(A, 1) = 8`.
  pub num_stages : Int32
  /// `size(H, 1) = 3`, the `kshortsize` and the dense-output order.
  pub h_rows : Int32
}

///|
/// `Rodas5PTableau(T, T2)` (rosenbrock_tableaus.jl:64).
///
/// The coefficients are frozen literals copied from upstream rather than
/// recomputed: they are the output of a numerical optimization run (they are
/// not expressible in closed form), so there is nothing to derive.
pub fn Rodas5PTableau::new() -> Rodas5PTableau {
  {
    a: [
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 2.849394379747939, 0.45842242204463923, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      -6.954028509809101, 2.489845061869568, -10.358996098473584, 0.0, 0.0, 0.0,
      0.0, 0.0, 2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.04721187230404641,
      0.0, 0.0, 0.0, 0.0, -7.502846399306121, 2.561846144803919, -11.627539656261098,
      -0.18268767659942256, 0.030198172008377946, 0.0, 0.0, 0.0, -7.502846399306121,
      2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946,
      1.0, 0.0, 0.0, -7.502846399306121, 2.561846144803919, -11.627539656261098,
      -0.18268767659942256, 0.030198172008377946, 1.0, 1.0, 0.0,
    ],
    cmat: [
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -14.155112264123755, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, -17.97296035885952, -2.859693295451294, 0.0, 0.0, 0.0, 0.0, 0.0,
      147.12150275711716, -1.41221402718213, 71.68940251302358, 0.0, 0.0, 0.0, 0.0,
      165.43517024871676, -0.4592823456491126, 42.90938336958603, -5.961986721573306,
      0.0, 0.0, 0.0, 24.854864614690072, -3.0009227002832186, 47.4931110020768, 5.5814197821558125,
      -0.6610691825249471, 0.0, 0.0, 30.91273214028599, -3.1208243349937974, 77.79954646070892,
      34.28646028294783, -19.097331116725623, -28.087943162872662, 0.0, 37.80277123390563,
      -3.2571969029072276, 112.26918849496327, 66.9347231244047, -40.06618937091002,
      -54.66780262877968, -9.48861652309627,
    ],
    c: [
      0.0, 0.6358126895828704, 0.4095798393397535, 0.9769306725060716, 0.4288403609558664,
      1.0, 1.0, 1.0,
    ],
    d: [
      0.21193756319429014, -0.42387512638858027, -0.3384627126235924, 1.8046452872882734,
      2.325825639765069, 0.0, 0.0, 0.0,
    ],
    h: [
      25.948786856663858, -2.5579724845846235, 10.433815404888879, -2.3679251022685204,
      0.524948541321073, 1.1241088310450404, 0.4272876194431874, -0.17202221070155493,
      -9.91568850695171, -0.9689944594115154, 3.0438037242978453, -24.495224566215796,
      20.176138334709044, 15.98066361424651, -6.789040303419874, -6.710236069923372,
      11.419903575922262, 2.8879645146136994, 72.92137995996029, 80.12511834622643,
      -52.072871366152654, -59.78993625266729, -0.15582684282751913, 4.883087185713722,
    ],
    b: [
      -7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256,
      0.030198172008377946, 1.0, 1.0, 1.0,
    ],
    btilde: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    gamma: 0.21193756319429014,
    num_stages: 8,
    h_rows: 3,
  }
}

///|
/// The 8-stage, order-5, stiffly stable Rosenbrock method with a stiff-aware
/// 4th-order dense interpolant.
pub struct Rodas5P {
  tableau : Rodas5PTableau
}

///|
/// The Rodas5P algorithm value.
pub fn Rodas5P::new() -> Rodas5P {
  { tableau: Rodas5PTableau::new(), }
}

///|
/// The tableau this method runs with.
pub fn Rodas5P::tableau(self : Rodas5P) -> Rodas5PTableau {
  self.tableau
}

///|
pub impl Solver for Rodas5P with fn order(_self) -> Int32 {
  // alg_utils.jl:35 — alg_order(alg::Rodas5P) = 5
  5
}

///|
pub impl Solver for Rodas5P with fn is_fsal(_self) -> Bool {
  // alg_utils.jl:60 — isfsal(alg::Rodas5P) = false. Consistent with the
  // generic `perform_step!`, which evaluates `du = f(uprev, p, t)` afresh on
  // every step rather than reading a carried-over `fsalfirst`.
  false
}

///|
pub impl Solver for Rodas5P with fn stage_count(_self) -> Int32 {
  // rosenbrock_perform_step.jl:421-422 — `initialize!` sets
  // `kshortsize = H_rows > 0 ? H_rows : 2`, and Rodas5P's H has 3 rows.
  3
}

///|
pub impl Solver for Rodas5P with fn perform_step(self, integ) -> Unit {
  rodas5p_perform_step(Rodas5P::tableau(self), integ)
}

///|
pub impl Solver for Rodas5P with fn interpolant(_self) -> SolutionInterp {
  // rosenbrock_caches.jl:506 sets `interp_order = H_rows = 3`, which selects
  // the `else` branch of `_ode_interpolant(..., Val{0})`
  // (rosenbrock_interpolants.jl:204) — the cubic in the stored stages.
  SolutionInterp::Poly(rodas5p_interpolate, rodas5p_derivative)
}

///|
pub impl Solver for Rodas5P with fn fill_dense_stages(_self, _integ) -> Unit {
  // `perform_step` already writes all three `H` rows into `integ.k`, which is
  // the complete stage set `rodas5p_interpolate` reads. No `addsteps!` pass
  // is needed — unlike Verner's methods, nothing has to be recomputed.
}

///|
pub impl Solver for Rodas5P with fn needs_fsal_reset_fixed_step(_self) -> Bool {
  // `only_diagonal_mass_matrix(alg)` lists only Rosenbrock23/32
  // (alg_utils.jl:82), and with `is_fsal = false` the whole `update_fsal!`
  // body is skipped anyway. Rodas5P re-evaluates `f(uprev, t)` itself.
  false
}

///|
pub impl Solver for Rodas5P with fn uses_bdf_controllers(_self) -> Bool {
  // Not a multistep BDF: the PI controller (control.mbt) drives the
  // step size and the generic step-reject path handles refusals.
  false
}

///|
/// Promote the trait methods onto `Rodas5P`.
pub extend Rodas5P with Solver::{
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
/// `jac_reuse_gamma_tol = 0.03`, Rodas5P's default (algorithms.jl:137).
pub const RODAS5P_JAC_GAMMA_TOL : Double = 0.03

///|
/// `max_jac_age = 20`, Rodas5P's default (algorithms.jl:115).
///
/// Unlike Rosenbrock23's `1`, twenty accepted steps may share one Jacobian —
/// the reuse default the higher-order W-methods inherit. It matters here
/// because `∂f/∂t` is only reused along with `J`
/// (derivative_utils.jl:1063), so this one number governs both.
pub const RODAS5P_MAX_JAC_AGE : Int32 = 20

///|
/// `perform_step!(integrator, cache::RosenbrockCombinedConstantCache)`
/// (rosenbrock_perform_step.jl:431) for the identity mass matrix.
///
/// The structure, in upstream's order:
///
///   1. `dtC = C ./ dt`, `dtd = dt .* d`, `dtgamma = dt * gamma`.
///   2. `calc_rosenbrock_differentiation` → `∂f/∂t` and the factorization of
///      `W`, subject to the Jacobian-reuse policy.
///   3. `du = f(uprev, t)`, then `k₁ = W \ -(du + dtd[1] * dT)`.
///   4. For stages `2..8`: build `u = uprev + Σ A[s,i] k_i`, evaluate
///      `du = f(u, t + c[s] * dt)`, then
///      `k_s = W \ -(du + dtd[s] * dT + Σ (C[s,i]/dt) k_i)`.
///   5. `u = uprev + Σ b_i k_i`.
///   6. Adaptive only: `utilde = Σ btilde_i k_i`, tolerance-scaled.
///   7. `k[j] = Σ_i H[j,i] k_i` for the dense interpolant.
///
/// Two details are easy to miss. `du` for the first stage is a *fresh*
/// evaluation, not the integrator's `fsalfirst`, because the method is not
/// FSAL; and the solution update uses `b` without a `dt` factor, because the
/// tableau's `k` already carry it.
///
/// The `W \ -x` form becomes `w_solve(x)` here because this port's `W` is
/// `J - I/dtgamma`, the negation of the combined path's `I/dtgamma - J`; see
/// the file header for the derivation.
fn rodas5p_perform_step(tab : Rodas5PTableau, integ : Integrator) -> Unit {
  let dt = integ.dt
  let t = integ.t
  let uprev = integ.uprev
  let dim = integ.u.length()
  let ns = tab.num_stages
  let dtgamma = dt * tab.gamma
  let cache = integ.stiff
  ignore(StiffCache::ensure(cache, dim))
  // Scratch, pooled on the integrator: eight stage derivatives plus the
  // stage state, the base derivative, and the finite-difference buffers.
  let k1 = Integrator::stiff_vec(integ, 0)
  let k2 = Integrator::stiff_vec(integ, 1)
  let k3 = Integrator::stiff_vec(integ, 2)
  let k4 = Integrator::stiff_vec(integ, 3)
  let k5 = Integrator::stiff_vec(integ, 4)
  let k6 = Integrator::stiff_vec(integ, 5)
  let k7 = Integrator::stiff_vec(integ, 6)
  let k8 = Integrator::stiff_vec(integ, 7)
  let u_stage = Integrator::stiff_vec(integ, 8)
  let du_base = Integrator::stiff_vec(integ, 9)
  let u_pert = Integrator::stiff_vec(integ, 10)
  let du_pert = Integrator::stiff_vec(integ, 11)
  // `du = f(uprev, t)` — the base for both the finite-difference Jacobian and
  // the `∂f/∂t` sweep, so those two each save the evaluation they'd otherwise
  // repeat (this is the `+1` in `maximum(colorvec) + 1`, derivative_wrappers).
  Integrator::eval_f(integ, t, uprev, du_base)
  // 2. W (refreshing J per the reuse policy) and ∂f/∂t.
  let refresh_jac = fn() {
    jacobian_fd(integ, t, uprev, du_base, cache.jac, u_pert, du_pert)
  }
  ignore(
    build_w(
      cache,
      integ,
      dtgamma,
      RODAS5P_JAC_GAMMA_TOL,
      RODAS5P_MAX_JAC_AGE,
      refresh_jac,
    ),
  )
  calc_tderivative(integ, cache, dtgamma, t, uprev, du_base, du_pert)
  let dT = cache.dT
  // 3. k₁ = W \ -(du + dt*d[0]*dT).  The minus sign is upstream's own
  //    (rosenbrock_perform_step.jl:458) and must survive: this port's `W` is
  //    `J - I/dtgamma`, so `w_solve` is a plain solve, not a negated one.
  let mut i = 0
  while i < dim {
    let x = du_base.unsafe_get(i) + dt * tab.d.unsafe_get(0) * dT.unsafe_get(i)
    u_stage.unsafe_set(i, -x)
    i += 1
  }
  w_solve(cache, u_stage)
  copy_into(k1, u_stage)
  // 4. Stages 2..8.
  let mut s = 1
  while s < ns {
    // u = uprev + Σ_{i<s} A[s,i] * k_i
    let arow = s * 8
    let crow = s * 7
    i = 0
    while i < dim {
      let mut v = uprev.unsafe_get(i)
      let mut j = 0
      while j < s {
        v = v + tab.a.unsafe_get(arow + j) * k_at(integ, j).unsafe_get(i)
        j += 1
      }
      u_stage.unsafe_set(i, v)
      i += 1
    }
    // du = f(u, t + c[s] * dt)
    Integrator::eval_f(integ, t + tab.c.unsafe_get(s) * dt, u_stage, du_pert)
    // k_s = W \ -(du + dt*d[s]*dT + Σ_{i<s} (C[s,i]/dt) * k_i)
    let dscale = dt * tab.d.unsafe_get(s)
    let inv_dt = 1.0 / dt
    i = 0
    while i < dim {
      let mut v = du_pert.unsafe_get(i) + dscale * dT.unsafe_get(i)
      let mut j = 0
      while j < s {
        v = v +
          tab.cmat.unsafe_get(crow + j) * inv_dt * k_at(integ, j).unsafe_get(i)
        j += 1
      }
      u_stage.unsafe_set(i, -v)
      i += 1
    }
    w_solve(cache, u_stage)
    copy_into(k_at(integ, s), u_stage)
    s += 1
  }
  // 5. u = uprev + Σ b_i k_i.  Upstream skips zero weights; every entry of
  //    Rodas5P's `b` is nonzero, but the guard is kept so a future tableau
  //    does not pay for a multiply by zero.
  i = 0
  while i < dim {
    let mut v = uprev.unsafe_get(i)
    let mut j = 0
    while j < ns {
      let bw = tab.b.unsafe_get(j)
      if bw != 0.0 {
        v = v + bw * k_at(integ, j).unsafe_get(i)
      }
      j += 1
    }
    integ.u.unsafe_set(i, v)
    i += 1
  }
  if integ.opts.adaptive {
    // 6. utilde = Σ btilde_i k_i — no `dt` factor, unlike Rosenbrock23's
    //    `dto6 * (k₁ - 2k₂ + k₃)`.
    i = 0
    while i < dim {
      let mut v = 0.0
      let mut j = 0
      while j < ns {
        let bw = tab.btilde.unsafe_get(j)
        if bw != 0.0 {
          v = v + bw * k_at(integ, j).unsafe_get(i)
        }
        j += 1
      }
      integ.utilde.unsafe_set(i, v)
      i += 1
    }
    error_estimate(integ, integ.utilde, uprev, integ.u)
  } else {
    integ.est = 0.0
  }
  // 7. k[j] = Σ_i H[j,i] k_i, for the cubic dense interpolant.
  let mut r = 0
  while r < tab.h_rows {
    let hrow = r * 8
    let dst = integ.k.unsafe_get(r)
    i = 0
    while i < dim {
      let mut v = 0.0
      let mut j = 0
      while j < ns {
        v = v + tab.h.unsafe_get(hrow + j) * k_at(integ, j).unsafe_get(i)
        j += 1
      }
      dst.unsafe_set(i, v)
      i += 1
    }
    r += 1
  }
}

///|
/// The `i`-th Rosenbrock stage derivative, addressed through the integrator's
/// scratch pool.
///
/// `perform_step` needs all eight simultaneously and in a nested loop, so the
/// stage vectors are not bound to locals — a closure over the pool keeps the
/// call sites readable without eight named bindings per access. Returns the
/// pooled vector; callers must not assume it survives another `stiff_vec`
/// call for a higher index, which is why every use here is immediately
/// followed by a read or a `copy_into`.
fn k_at(integ : Integrator, i : Int32) -> Array[Double] {
  Integrator::stiff_vec(integ, i)
}

///|
/// The `interp_order == 3` value branch of `_ode_interpolant`
/// (rosenbrock_interpolants.jl:204):
///
///     Θ₁·y₀ + Θ·(y₁ + Θ₁·(k₁ + Θ·(k₂ + Θ·k₃)))
///
/// This is the stiffness-aware cubic. It has the same cubic Hermite *shape*
/// as `SolutionInterp::Hermite`, but its `k₁..k₃` are the `H`-weighted stage
/// combinations rather than the two endpoint derivatives — which is what lets
/// it stay accurate when `dt` far exceeds the solution's own timescale. For a
/// method whose `H` happened to reproduce the endpoint derivatives, the two
/// would coincide.
fn rodas5p_interpolate(
  theta : Double,
  _dt : Double,
  y0 : Array[Double],
  y1 : Array[Double],
  ks : Array[Array[Double]],
  out : Array[Double],
) -> Unit {
  let onemt = 1.0 - theta
  let k1 = ks.unsafe_get(0)
  let k2 = ks.unsafe_get(1)
  let k3 = ks.unsafe_get(2)
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      onemt * y0.unsafe_get(i) +
      theta *
      (
        y1.unsafe_get(i) +
        onemt *
        (
          k1.unsafe_get(i) +
          theta * (k2.unsafe_get(i) + theta * k3.unsafe_get(i))
        )
      ),
    )
    i += 1
  }
}

///|
/// The `interp_order == 3` derivative branch of `_ode_interpolant`
/// (rosenbrock_interpolants.jl:357-365):
///
///     (k₁ + Θ·(-2k₁ + 2k₂ + Θ·(-3k₂ + 3k₃ - 4Θ·k₃)) - y₀ + y₁) / dt
///
/// The analytic derivative of `rodas5p_interpolate`, so `Solution::derivative`
/// is exact for the interpolant rather than a finite difference of it.
fn rodas5p_derivative(
  theta : Double,
  dt : Double,
  y0 : Array[Double],
  y1 : Array[Double],
  ks : Array[Array[Double]],
  out : Array[Double],
) -> Unit {
  if dt == 0.0 {
    return
  }
  let k1 = ks.unsafe_get(0)
  let k2 = ks.unsafe_get(1)
  let k3 = ks.unsafe_get(2)
  let n = out.length()
  let inv_dt = 1.0 / dt
  let mut i = 0
  while i < n {
    let a = k1.unsafe_get(i)
    let b = k2.unsafe_get(i)
    let c = k3.unsafe_get(i)
    let inner = -2.0 * a +
      2.0 * b +
      theta * (-3.0 * b + 3.0 * c - 4.0 * theta * c)
    out.unsafe_set(
      i,
      (a + theta * inner - y0.unsafe_get(i) + y1.unsafe_get(i)) * inv_dt,
    )
    i += 1
  }
}
