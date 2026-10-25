// Problem and solve-option definitions.
//
// Mirrors the subset of SciMLBase's `ODEProblem` / `ODEFunction` / solve
// keyword options that a pure ODE solve needs. SciMLBase is an external
// package in the Julia monorepo, so the types are re-derived here.

///|
/// The right-hand side `f(t, u)` of an ODE.
///
/// The in-place variant is the primary path: it writes the derivative into a
/// caller-owned buffer, so a solve performs no per-step allocation. The
/// out-of-place variant is a convenience wrapper for small problems and
/// closures that naturally return a fresh vector.
pub enum OdeFunction {
  /// In-place: `du = f(t, u)`.
  Iip((Double, Array[Double], Array[Double]) -> Unit)
  /// Out-of-place: returns `f(t, u)`.
  Oop((Double, Array[Double]) -> Array[Double])
}

///|
/// Evaluate `f(t, u)` into `out`. For the OOP form the returned vector is
/// copied in (truncated to `out`'s length).
pub fn OdeFunction::eval_into(
  f : OdeFunction,
  t : Double,
  u : Array[Double],
  out : Array[Double],
) -> Unit {
  match f {
    Iip(g) => g(t, u, out)
    Oop(g) => {
      let v = g(t, u)
      let n = if out.length() < v.length() { out.length() } else { v.length() }
      let mut i = 0
      while i < n {
        out.unsafe_set(i, v.unsafe_get(i))
        i += 1
      }
    }
  }
}

///|
/// An ordinary differential equation `u' = f(t, u)` on `[t0, tf]` with initial
/// state `u0`.
pub struct OdeProblem {
  pub t0 : Double
  pub tf : Double
  pub u0 : Array[Double]
  pub f : OdeFunction
}

///|
/// Build an in-place problem.
pub fn OdeProblem::iip(
  t0 : Double,
  tf : Double,
  u0 : Array[Double],
  f : (Double, Array[Double], Array[Double]) -> Unit,
) -> OdeProblem {
  { t0, tf, u0, f: OdeFunction::Iip(f), }
}

///|
/// Build an out-of-place problem.
pub fn OdeProblem::oop(
  t0 : Double,
  tf : Double,
  u0 : Array[Double],
  f : (Double, Array[Double]) -> Array[Double],
) -> OdeProblem {
  { t0, tf, u0, f: OdeFunction::Oop(f), }
}

///|
/// Tunables for a solve. Defaults follow OrdinaryDiffEq/DiffEqBase:
/// `dtmin = 0`, `dtmax = tf - t0`, `maxiters = 100000`, `reltol = 1e-3`,
/// `abstol = 1e-6`. `abstol`/`reltol` scale the embedded error estimate that
/// drives adaptive stepping (and the automatic initial step).
pub struct SolveOptions {
  /// Step size. Required for fixed-step solves; `0` means "choose
  /// automatically" (adaptive only).
  pub dt : Double
  /// Enable adaptive step-size control (M2; fixed-step in M1).
  pub adaptive : Bool
  pub reltol : Double
  pub abstol : Double
  pub maxiters : Int32
  pub mut dtmax : Double
  pub mut dtmin : Double
  /// Store the solver's stage arrays for post-solve dense output.
  pub mut dense : Bool
  /// Save after every accepted step (default) or only at `saveat`/`save_end`.
  pub mut save_everystep : Bool
  /// Save the initial point `(t0, u0)`.
  pub mut save_start : Bool
  /// Save the final point `(tf, u(tf))` even if it was not stepped on.
  pub mut save_end : Bool
  /// Extra output times strictly inside `(t0, tf)`, physical units.
  pub saveat : Array[Double]
  /// Extra stopping times strictly inside `(t0, tf)`, physical units.
  pub tstops : Array[Double]
}

///|
/// Default options: fixed step, dense output on, save every step.
pub fn SolveOptions::new() -> SolveOptions {
  {
    dt: 0.0,
    adaptive: false,
    reltol: 0.001,
    abstol: 0.000001,
    maxiters: 100000,
    dtmax: 0.0,
    dtmin: 0.0,
    dense: true,
    save_everystep: true,
    save_start: true,
    save_end: true,
    saveat: [],
    tstops: [],
  }
}

///|
/// Copy `opts` with a different fixed step size.
pub fn SolveOptions::with_dt(opts : SolveOptions, dt : Double) -> SolveOptions {
  {
    dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with a different `dense` setting.
pub fn SolveOptions::with_dense(
  opts : SolveOptions,
  dense : Bool,
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with different extra output times.
pub fn SolveOptions::with_saveat(
  opts : SolveOptions,
  saveat : Array[Double],
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with different extra stopping times.
pub fn SolveOptions::with_tstops(
  opts : SolveOptions,
  tstops : Array[Double],
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops,
  }
}

///|
/// Copy `opts` with adaptive stepping switched on or off. Adaptive solves
/// require either a `dt` seed or `dt = 0` (automatic initial step); the
/// error estimate then drives `dt` through the PI controller.
pub fn SolveOptions::with_adaptive(
  opts : SolveOptions,
  adaptive : Bool,
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with a different relative tolerance.
pub fn SolveOptions::with_reltol(
  opts : SolveOptions,
  reltol : Double,
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with a different absolute tolerance.
pub fn SolveOptions::with_abstol(
  opts : SolveOptions,
  abstol : Double,
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with a different maximum step size. `0` means "unset" — the
/// solve replaces it with `|tf - t0|`.
pub fn SolveOptions::with_dtmax(
  opts : SolveOptions,
  dtmax : Double,
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with a different minimum step size. Adaptive solves abort with
/// `DtMinUnstable` once the controller shrinks `dt` to this bound without an
/// acceptable step (DiffEqBase check_error.jl:112).
pub fn SolveOptions::with_dtmin(
  opts : SolveOptions,
  dtmin : Double,
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters: opts.maxiters,
    dtmax: opts.dtmax,
    dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
/// Copy `opts` with a different `maxiters` step limit.
pub fn SolveOptions::with_maxiters(
  opts : SolveOptions,
  maxiters : Int32,
) -> SolveOptions {
  {
    dt: opts.dt,
    adaptive: opts.adaptive,
    reltol: opts.reltol,
    abstol: opts.abstol,
    maxiters,
    dtmax: opts.dtmax,
    dtmin: opts.dtmin,
    dense: opts.dense,
    save_everystep: opts.save_everystep,
    save_start: opts.save_start,
    save_end: opts.save_end,
    saveat: opts.saveat,
    tstops: opts.tstops,
  }
}

///|
test "problem constructors wire up the right-hand side" {
  let p1 = OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
  let out = Array::make(1, 0.0)
  OdeFunction::eval_into(p1.f, 0.5, [2.0], out)
  assert_eq(out.unsafe_get(0), -2.0)

  let p2 = OdeProblem::oop(0.0, 1.0, [1.0], fn(t, u) { [t * u.unsafe_get(0)] })
  OdeFunction::eval_into(p2.f, 3.0, [2.0], out)
  assert_eq(out.unsafe_get(0), 6.0)
}
