// The solution object: saved times, states, per-step stage derivatives and
// the terminal return code. Mirrors the subset of SciMLBase's ODESolution
// that a pure ODE solve produces (SciMLBase is an external package in the
// Julia monorepo, so the types are re-derived here).

///|
/// Why a solve stopped — the subset of SciMLBase's `ReturnCode` a pure ODE
/// solve can produce.
pub enum Retcode {
  /// The solve completed normally.
  Success
  /// More than `maxiters` steps were taken.
  MaxIters
  /// The step size fell below `dtmin` without producing an acceptable step.
  DtMinUnstable
  /// The state became non-finite (NaN or Inf).
  Unstable
  /// The problem/options combination cannot be solved (fixed step without
  /// `dt`, empty time span, empty state).
  InitialFailure
}

///|
/// Structural equality on return codes (needed by the loop's `!=` checks).
pub impl Eq for Retcode with fn equal(self, other) -> Bool {
  match (self, other) {
    (Success, Success) => true
    (MaxIters, MaxIters) => true
    (DtMinUnstable, DtMinUnstable) => true
    (Unstable, Unstable) => true
    (InitialFailure, InitialFailure) => true
    _ => false
  }
}

///|
/// Re-export `equal`/`not_equal` as methods on `Retcode`.
pub extend Retcode with Eq::{equal, not_equal}

///|
/// True when the solve completed normally.
pub fn Retcode::is_success(self : Retcode) -> Bool {
  self.equal(Retcode::Success)
}

///|
/// True when the `maxiters` step limit was reached.
pub fn Retcode::is_maxiters(self : Retcode) -> Bool {
  self.equal(Retcode::MaxIters)
}

///|
/// True when the step size fell below `dtmin`.
pub fn Retcode::is_dtmin_unstable(self : Retcode) -> Bool {
  self.equal(Retcode::DtMinUnstable)
}

///|
/// True when the state became non-finite (NaN or Inf).
pub fn Retcode::is_unstable(self : Retcode) -> Bool {
  self.equal(Retcode::Unstable)
}

///|
/// True when the problem/options combination cannot be solved.
pub fn Retcode::is_initial_failure(self : Retcode) -> Bool {
  self.equal(Retcode::InitialFailure)
}

///|
/// How a stored solution interpolates between saved time points.
///
/// * `Linear` — no dense output was stored.
/// * `Hermite` — cubic Hermite from the endpoint derivatives `k[0]`, `k[1]`.
/// * `Poly` — solver-specific polynomial. `value(theta, dt, y0, y1, ks, out)`
///   writes the state at fraction `theta` of the step that runs from `y0` to
///   `y1` (`ks` is the stage array stored for that interval); `derivative`
///   writes `dy/dt` at the same point. Mirrors Julia's
///   `ode_interpolant(Θ, dt, y₀, y₁, k, cache, idxs, Val{0}, ...)` and the
///   `Val{1}` dispatch for solvers with special interpolations.
///
/// `y1` is part of the signature even though the Verner-family closures ignore
/// it: upstream passes both endpoints to every `_ode_interpolant`, and the
/// Rosenbrock cubic genuinely needs `y₁` — it is the `Θ·(y₁ + ...)` term.
pub enum SolutionInterp {
  Linear
  Hermite
  Poly(
    (
      Double,
      Double,
      Array[Double],
      Array[Double],
      Array[Array[Double]],
      Array[Double],
    ) -> Unit,
    (
      Double,
      Double,
      Array[Double],
      Array[Double],
      Array[Array[Double]],
      Array[Double],
    ) -> Unit
  )
}

///|
/// A computed solution. `t` is chronological; `u[i]` is the state at `t[i]`;
/// `k[j]` (only when dense output is enabled) holds the stage derivatives of
/// the step that ended at `t[j]`, in the convention the interpolant expects.
pub struct Solution {
  pub mut t : Array[Double]
  pub mut u : Array[Array[Double]]
  pub mut k : Array[Array[Array[Double]]]
  pub mut retcode : Retcode
  pub mut interp : SolutionInterp
  /// Diagnostic: number of right-hand-side evaluations the solve performed.
  pub mut nf : Int32
}

///|
/// Append a saved time point (the state array is stored by reference; callers
/// pass a fresh copy or an array the integrator will not mutate).
pub fn Solution::push(sol : Solution, t : Double, u : Array[Double]) -> Unit {
  sol.t.push(t)
  sol.u.push(u)
}

///|
/// Append the stage arrays of the step that just ended (deep copy, so later
/// integrator steps cannot alias stored data).
pub fn Solution::push_ks(sol : Solution, ks : Array[Array[Double]]) -> Unit {
  let n = ks.length()
  let group = Array::make(n, Array::make(0, 0.0))
  let mut i = 0
  while i < n {
    let src = ks.unsafe_get(i)
    let len = src.length()
    let dst = Array::make(len, 0.0)
    let mut j = 0
    while j < len {
      dst.unsafe_set(j, src.unsafe_get(j))
      j += 1
    }
    group.unsafe_set(i, dst)
    i += 1
  }
  sol.k.push(group)
}

///|
/// Evaluate the solution at time `t` (dense output). Times outside the saved
/// range use the nearest interval (the same polynomial, extrapolated).
pub fn Solution::call(sol : Solution, t : Double) -> Array[Double] {
  let n = sol.t.length()
  if n == 0 {
    return Array::make(0, 0.0)
  }
  let dim = sol.u.unsafe_get(0).length()
  let out = Array::make(dim, 0.0)
  if n == 1 {
    copy_into(out, sol.u.unsafe_get(0))
    return out
  }
  let i = search_interval(sol.t, t)
  let dt = sol.t.unsafe_get(i + 1) - sol.t.unsafe_get(i)
  let theta = if dt == 0.0 { 1.0 } else { (t - sol.t.unsafe_get(i)) / dt }
  let y0 = sol.u.unsafe_get(i)
  let y1 = sol.u.unsafe_get(i + 1)
  let has_ks = sol.k.length() > i + 1
  match sol.interp {
    Linear => linear_interp(y0, y1, theta, out)
    Hermite =>
      if has_ks {
        let ks = sol.k.unsafe_get(i + 1)
        hermite_interpolate(
          y0,
          y1,
          ks.unsafe_get(0),
          ks.unsafe_get(1),
          theta,
          dt,
          out,
        )
      } else {
        linear_interp(y0, y1, theta, out)
      }
    Poly(f, _) =>
      if has_ks {
        f(theta, dt, y0, y1, sol.k.unsafe_get(i + 1), out)
      } else {
        linear_interp(y0, y1, theta, out)
      }
  }
  out
}

///|
/// First derivative of the solution at `t`. Exact for every interpolant:
/// linear and Hermite analytically, solver-specific polynomials through the
/// derivative closure stored alongside the value closure (Julia's
/// `ode_interpolant(..., Val{1})`).
pub fn Solution::derivative(sol : Solution, t : Double) -> Array[Double] {
  let n = sol.t.length()
  if n == 0 {
    return Array::make(0, 0.0)
  }
  let dim = sol.u.unsafe_get(0).length()
  let out = Array::make(dim, 0.0)
  if n == 1 {
    return out
  }
  match sol.interp {
    Hermite =>
      if sol.k.length() > 0 {
        let i = search_interval(sol.t, t)
        let dt = sol.t.unsafe_get(i + 1) - sol.t.unsafe_get(i)
        let theta = if dt == 0.0 { 1.0 } else { (t - sol.t.unsafe_get(i)) / dt }
        let y0 = sol.u.unsafe_get(i)
        let y1 = sol.u.unsafe_get(i + 1)
        let ks = sol.k.unsafe_get(i + 1)
        hermite_derivative(
          y0,
          y1,
          ks.unsafe_get(0),
          ks.unsafe_get(1),
          theta,
          dt,
          out,
        )
      } else {
        linear_derivative(sol, t, out)
      }
    Linear => linear_derivative(sol, t, out)
    Poly(_, df) =>
      if sol.k.length() > 0 {
        let i = search_interval(sol.t, t)
        let dt = sol.t.unsafe_get(i + 1) - sol.t.unsafe_get(i)
        let theta = if dt == 0.0 { 1.0 } else { (t - sol.t.unsafe_get(i)) / dt }
        df(
          theta,
          dt,
          sol.u.unsafe_get(i),
          sol.u.unsafe_get(i + 1),
          sol.k.unsafe_get(i + 1),
          out,
        )
      } else {
        linear_derivative(sol, t, out)
      }
  }
  out
}

///|
/// Small shared helpers (kept here so every module can use them).

///|
/// Element-wise copy `dst = src` (lengths must match).
pub fn copy_into(dst : Array[Double], src : Array[Double]) -> Unit {
  let n = src.length()
  let mut i = 0
  while i < n {
    dst.unsafe_set(i, src.unsafe_get(i))
    i += 1
  }
}

///|
/// Fresh copy of `a`.
pub fn copy_array(a : Array[Double]) -> Array[Double] {
  let n = a.length()
  let out = Array::make(n, 0.0)
  copy_into(out, a)
  out
}

///|
test "solution push and call round-trip" {
  let sol = Solution::{
    t: [],
    u: [],
    k: [],
    retcode: Retcode::Success,
    interp: SolutionInterp::Linear,
    nf: 0,
  }
  Solution::push(sol, 0.0, [1.0, 2.0])
  Solution::push(sol, 1.0, [3.0, 4.0])
  assert_eq(sol.t.length(), 2)
  let mid = Solution::call(sol, 0.5)
  assert_eq(mid.unsafe_get(0), 2.0)
  assert_eq(mid.unsafe_get(1), 3.0)
  // Outside the range: the nearest interval's polynomial, extrapolated.
  let after = Solution::call(sol, 2.0)
  assert_eq(after.unsafe_get(0), 5.0)
}
