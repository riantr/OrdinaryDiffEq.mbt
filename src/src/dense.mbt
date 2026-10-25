// Dense-output interpolants shared by all solvers.
//
// These are direct ports of the Hermite and linear interpolants in
// OrdinaryDiffEqCore's dense/generic_dense.jl. Solver-specific dense output
// (Tsit5's quintic, Vern7's, ...) lives next to the solver itself and is
// installed into the solution as `SolutionInterp::Poly`.

///|
/// Binary search for the saved interval containing `t`: the largest `i` with
/// `ts[i] <= t`, clamped to `[0, n - 2]`. Assumes `n >= 2`.
pub fn search_interval(ts : Array[Double], t : Double) -> Int32 {
  let n = ts.length()
  if n <= 2 {
    return 0
  }
  let mut lo = 0
  let mut hi = n - 2
  while lo < hi {
    let mid = (lo + hi + 1) / 2
    if ts.unsafe_get(mid) <= t {
      lo = mid
    } else {
      hi = mid - 1
    }
  }
  lo
}

///|
/// `out = (1 - theta) * y0 + theta * y1` (generic_dense.jl linear_interpolant).
pub fn linear_interp(
  y0 : Array[Double],
  y1 : Array[Double],
  theta : Double,
  out : Array[Double],
) -> Unit {
  let n = y0.length()
  let mut i = 0
  while i < n {
    let v = (1.0 - theta) * y0.unsafe_get(i) + theta * y1.unsafe_get(i)
    out.unsafe_set(i, v)
    i += 1
  }
}

///|
/// Cubic Hermite interpolation between `y0` at the start of an interval of
/// length `dt` and `y1` at its end, using the endpoint derivatives `k1` and
/// `k2`, evaluated at `theta` in [0, 1].
///
/// Mirrors generic_dense.jl:1648:
/// `(1-Θ)y₀ + Θy₁ + Θ(Θ-1)((1-2Θ)(y₁-y₀) + (Θ-1)dt k1 + Θ dt k2)`
pub fn hermite_interpolate(
  y0 : Array[Double],
  y1 : Array[Double],
  k1 : Array[Double],
  k2 : Array[Double],
  theta : Double,
  dt : Double,
  out : Array[Double],
) -> Unit {
  let n = y0.length()
  let tm1 = theta - 1.0
  let onemt = 1.0 - theta
  let onem2t = 1.0 - 2.0 * theta
  let mut i = 0
  while i < n {
    let a = y0.unsafe_get(i)
    let b = y1.unsafe_get(i)
    let ka = k1.unsafe_get(i)
    let kb = k2.unsafe_get(i)
    let v = onemt * a +
      theta * b +
      theta * tm1 * (onem2t * (b - a) + tm1 * dt * ka + theta * dt * kb)
    out.unsafe_set(i, v)
    i += 1
  }
}

///|
/// Time derivative of the cubic Hermite interpolant (generic_dense.jl:1797):
/// `k1 + Θ(-4 dt k1 - 2 dt k2 - 6 y0 + Θ(3 dt k1 + 3 dt k2 + 6 y0 - 6 y1) + 6 y1) / dt`
pub fn hermite_derivative(
  y0 : Array[Double],
  y1 : Array[Double],
  k1 : Array[Double],
  k2 : Array[Double],
  theta : Double,
  dt : Double,
  out : Array[Double],
) -> Unit {
  if dt == 0.0 {
    return
  }
  let n = y0.length()
  let mut i = 0
  while i < n {
    let a = y0.unsafe_get(i)
    let b = y1.unsafe_get(i)
    let ka = k1.unsafe_get(i)
    let kb = k2.unsafe_get(i)
    let inner = -4.0 * dt * ka -
      2.0 * dt * kb -
      6.0 * a +
      theta * (3.0 * dt * ka + 3.0 * dt * kb + 6.0 * a - 6.0 * b) +
      6.0 * b
    out.unsafe_set(i, ka + theta * inner / dt)
    i += 1
  }
}

///|
/// `out = (y1 - y0) / dt` for the saved interval containing `t`; the exact
/// derivative of the linear interpolant (generic_dense.jl:2268).
pub fn linear_derivative(
  sol : Solution,
  t : Double,
  out : Array[Double],
) -> Unit {
  let n = sol.t.length()
  if n < 2 {
    return
  }
  let i = search_interval(sol.t, t)
  let dt = sol.t.unsafe_get(i + 1) - sol.t.unsafe_get(i)
  if dt == 0.0 {
    return
  }
  let y0 = sol.u.unsafe_get(i)
  let y1 = sol.u.unsafe_get(i + 1)
  let mut j = 0
  while j < out.length() {
    out.unsafe_set(j, (y1.unsafe_get(j) - y0.unsafe_get(j)) / dt)
    j += 1
  }
}

///|
test "search_interval brackets the query time" {
  let ts = [0.0, 1.0, 2.0, 3.0]
  assert_eq(search_interval(ts, -5.0), 0)
  assert_eq(search_interval(ts, 0.0), 0)
  assert_eq(search_interval(ts, 0.5), 0)
  assert_eq(search_interval(ts, 1.0), 1)
  assert_eq(search_interval(ts, 2.5), 2)
  assert_eq(search_interval(ts, 100.0), 2)
}

///|
test "hermite interpolation reproduces the endpoints" {
  let y0 = [1.0, 2.0]
  let y1 = [3.0, 4.0]
  let k1 = [0.5, -1.0]
  let k2 = [1.5, 2.0]
  let out = Array::make(2, 0.0)
  hermite_interpolate(y0, y1, k1, k2, 0.0, 0.25, out)
  assert_eq(out.unsafe_get(0), 1.0)
  assert_eq(out.unsafe_get(1), 2.0)
  hermite_interpolate(y0, y1, k1, k2, 1.0, 0.25, out)
  assert_eq(out.unsafe_get(0), 3.0)
  assert_eq(out.unsafe_get(1), 4.0)
  // Endpoint derivatives are exact: d/dt at theta=0 is k1, at theta=1 is k2.
  hermite_derivative(y0, y1, k1, k2, 0.0, 0.25, out)
  assert_eq(out.unsafe_get(0), 0.5)
  assert_eq(out.unsafe_get(1), -1.0)
  hermite_derivative(y0, y1, k1, k2, 1.0, 0.25, out)
  assert_eq(out.unsafe_get(0), 1.5)
  assert_eq(out.unsafe_get(1), 2.0)
}

///|
test "hermite on exp decay tracks the exact curve" {
  // Interpolate u' = -u between (0, 1) and (0.5, exp(-0.5)) with exact
  // endpoint derivatives; the cubic Hermite error should be tiny.
  let dt = 0.5
  let y0 = [1.0]
  let y1 = [@math.exp(-dt)]
  let k1 = [-1.0]
  let k2 = [-@math.exp(-dt)]
  let out = Array::make(1, 0.0)
  let mut worst = 0.0
  let mut i = 0
  while i <= 10 {
    let theta = i.to_double() / 10.0
    hermite_interpolate(y0, y1, k1, k2, theta, dt, out)
    let exact = @math.exp(-dt * theta)
    let err = (out.unsafe_get(0) - exact).abs()
    if err > worst {
      worst = err
    }
    i += 1
  }
  assert_true(worst < 0.0015)
}
