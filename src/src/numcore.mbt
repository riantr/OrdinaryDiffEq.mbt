// Norms and scaled residuals shared by every solver kernel.
//
// Ported from DiffEqBase:
//   * `ODE_DEFAULT_NORM`      (common_defaults.jl:63-87)   -> rms_norm
//   * `norm(u, Inf)`          (used by eigen estimates)    -> inf_norm
//   * `calculate_residuals!`  (calculate_residuals.jl)     -> weighted_residual
//
// The RMS norm scales with the magnitude of the state but not its
// dimensionality, which is what makes a single abstol/reltol pair meaningful
// across problems of different sizes.

///|
/// Weighted RMS norm — the integrators' default error norm
/// (`ODE_DEFAULT_NORM`, DiffEqBase/common_defaults.jl:63-87).
///
/// Vectors use `sqrt(sum |u_i|^2 / max(n, 1))`; the empty vector is 0.
pub fn rms_norm(u : Array[Double]) -> Double {
  let n = u.length()
  if n == 0 {
    return 0.0
  }
  let mut s = 0.0
  let mut i = 0
  while i < n {
    let ui = u.unsafe_get(i)
    s = s + ui * ui
    i += 1
  }
  sqrt(s / Double::from_int(n))
}

///|
/// Infinity norm `max |u_i|` (0 for the empty vector). Julia uses
/// `norm(_, Inf)` rather than `maximum(abs, _)` so that an empty state yields
/// 0 instead of reducing over an empty collection.
pub fn inf_norm(u : Array[Double]) -> Double {
  let n = u.length()
  if n == 0 {
    return 0.0
  }
  let mut m = 0.0
  let mut i = 0
  while i < n {
    let a = u.unsafe_get(i).abs()
    if a > m {
      m = a
    }
    i += 1
  }
  m
}

///|
/// Element-wise scaled residuals, written in place into `out`:
///
///     out_i = utilde_i / (abstol + max(|u0_i|, |u1_i|) * reltol)
///
/// (`calculate_residuals!`, DiffEqBase/calculate_residuals.jl). `out` must
/// have the same length as `utilde`. Each solver cache keeps one scratch
/// buffer and calls this every step without allocating.
pub fn weighted_residual(
  out : Array[Double],
  utilde : Array[Double],
  u0 : Array[Double],
  u1 : Array[Double],
  abstol : Double,
  reltol : Double,
) -> Unit {
  let n = utilde.length()
  let mut i = 0
  while i < n {
    let a0 = u0.unsafe_get(i).abs()
    let a1 = u1.unsafe_get(i).abs()
    let amax = if a0 > a1 { a0 } else { a1 }
    let scale = abstol + amax * reltol
    out.unsafe_set(i, utilde.unsafe_get(i) / scale)
    i += 1
  }
}

///|
test "rms_norm is sqrt of mean square" {
  assert_eq(rms_norm([3.0, 4.0]), 3.5355339059327378) // sqrt(12.5)
  assert_eq(rms_norm([1.0, 1.0, 1.0, 1.0]), 1.0)
  assert_eq(rms_norm([]), 0.0)
  assert_eq(rms_norm([-2.0, 2.0]), 2.0)
}

///|
test "inf_norm is the max absolute value" {
  assert_eq(inf_norm([3.0, -7.0, 2.0]), 7.0)
  assert_eq(inf_norm([]), 0.0)
  assert_eq(inf_norm([-1.5]), 1.5)
}

///|
test "weighted_residual scales by abstol + max*reltol" {
  let out = Array::make(2, 0.0)
  weighted_residual(out, [0.1, -0.2], [1.0, -2.0], [1.1, -2.2], 0.000001, 0.001)
  // |u0| = 1, |u1| = 1.1 -> scale = 1e-6 + 1.1e-3
  let s0 = 0.000001 + 0.0011
  // |u0| = 2, |u1| = 2.2 -> scale = 1e-6 + 2.2e-3
  let s1 = 0.000001 + 0.0022
  assert_true((out.unsafe_get(0) - 0.1 / s0).abs() < 0.000000000001)
  assert_true((out.unsafe_get(1) - -0.2 / s1).abs() < 0.000000000001)
}

///|
/// `out = base + a1 * x1`, element-wise. The first of the linear-combination
/// kernels every explicit Runge-Kutta stage is built from: Julia spells each
/// stage out as `uprev + dt * (a21 * k1 + a32 * k2 + ...)`, and the variants
/// below cover the arities real tableaus need (Tsit5 uses 1..6, Verner 7/9
/// use 1..7; the base-less `lincomb7` serves the embedded-error combination).
pub fn lincomb1(
  out : Array[Double],
  base : Array[Double],
  a1 : Double,
  x1 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(i, base.unsafe_get(i) + a1 * x1.unsafe_get(i))
    i += 1
  }
}

///|
/// `out = base + a1 * x1 + a2 * x2`.
pub fn lincomb2(
  out : Array[Double],
  base : Array[Double],
  a1 : Double,
  x1 : Array[Double],
  a2 : Double,
  x2 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      base.unsafe_get(i) + a1 * x1.unsafe_get(i) + a2 * x2.unsafe_get(i),
    )
    i += 1
  }
}

///|
/// `out = base + a1 * x1 + a2 * x2 + a3 * x3`.
pub fn lincomb3(
  out : Array[Double],
  base : Array[Double],
  a1 : Double,
  x1 : Array[Double],
  a2 : Double,
  x2 : Array[Double],
  a3 : Double,
  x3 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      base.unsafe_get(i) +
      a1 * x1.unsafe_get(i) +
      a2 * x2.unsafe_get(i) +
      a3 * x3.unsafe_get(i),
    )
    i += 1
  }
}

///|
/// `out = base + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4`.
pub fn lincomb4(
  out : Array[Double],
  base : Array[Double],
  a1 : Double,
  x1 : Array[Double],
  a2 : Double,
  x2 : Array[Double],
  a3 : Double,
  x3 : Array[Double],
  a4 : Double,
  x4 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      base.unsafe_get(i) +
      a1 * x1.unsafe_get(i) +
      a2 * x2.unsafe_get(i) +
      a3 * x3.unsafe_get(i) +
      a4 * x4.unsafe_get(i),
    )
    i += 1
  }
}

///|
/// `out = base + a1 * x1 + ... + a5 * x5`.
pub fn lincomb5(
  out : Array[Double],
  base : Array[Double],
  a1 : Double,
  x1 : Array[Double],
  a2 : Double,
  x2 : Array[Double],
  a3 : Double,
  x3 : Array[Double],
  a4 : Double,
  x4 : Array[Double],
  a5 : Double,
  x5 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      base.unsafe_get(i) +
      a1 * x1.unsafe_get(i) +
      a2 * x2.unsafe_get(i) +
      a3 * x3.unsafe_get(i) +
      a4 * x4.unsafe_get(i) +
      a5 * x5.unsafe_get(i),
    )
    i += 1
  }
}

///|
/// `out = base + a1 * x1 + ... + a6 * x6`.
pub fn lincomb6(
  out : Array[Double],
  base : Array[Double],
  a1 : Double,
  x1 : Array[Double],
  a2 : Double,
  x2 : Array[Double],
  a3 : Double,
  x3 : Array[Double],
  a4 : Double,
  x4 : Array[Double],
  a5 : Double,
  x5 : Array[Double],
  a6 : Double,
  x6 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      base.unsafe_get(i) +
      a1 * x1.unsafe_get(i) +
      a2 * x2.unsafe_get(i) +
      a3 * x3.unsafe_get(i) +
      a4 * x4.unsafe_get(i) +
      a5 * x5.unsafe_get(i) +
      a6 * x6.unsafe_get(i),
    )
    i += 1
  }
}

///|
/// `out = a1 * x1 + ... + a7 * x7` — the one linear-combination kernel with
/// no base term. The embedded error estimate needs exactly this shape:
/// `utilde = dt * (b̃1*k1 + ... + b̃7*k7)` (tsit_perform_step.jl:167), where
/// the b̃ row sums to zero so folding the combination into `uprev + ...`
/// would not be equivalent.
pub fn lincomb7(
  out : Array[Double],
  a1 : Double,
  x1 : Array[Double],
  a2 : Double,
  x2 : Array[Double],
  a3 : Double,
  x3 : Array[Double],
  a4 : Double,
  x4 : Array[Double],
  a5 : Double,
  x5 : Array[Double],
  a6 : Double,
  x6 : Array[Double],
  a7 : Double,
  x7 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      a1 * x1.unsafe_get(i) +
      a2 * x2.unsafe_get(i) +
      a3 * x3.unsafe_get(i) +
      a4 * x4.unsafe_get(i) +
      a5 * x5.unsafe_get(i) +
      a6 * x6.unsafe_get(i) +
      a7 * x7.unsafe_get(i),
    )
    i += 1
  }
}

///|
/// `out = a1 * x1 + ... + a8 * x8`.
///
/// Vern7's embedded-error row is eight terms wide (btilde1 and btilde4..10),
/// one past `lincomb7`; the Verner tableaus top out here, so this is the
/// widest kernel needed. Base-less like `lincomb7`, because that row sums to
/// zero and folding `uprev` in would not be equivalent.
pub fn lincomb8(
  out : Array[Double],
  a1 : Double,
  x1 : Array[Double],
  a2 : Double,
  x2 : Array[Double],
  a3 : Double,
  x3 : Array[Double],
  a4 : Double,
  x4 : Array[Double],
  a5 : Double,
  x5 : Array[Double],
  a6 : Double,
  x6 : Array[Double],
  a7 : Double,
  x7 : Array[Double],
  a8 : Double,
  x8 : Array[Double],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(
      i,
      a1 * x1.unsafe_get(i) +
      a2 * x2.unsafe_get(i) +
      a3 * x3.unsafe_get(i) +
      a4 * x4.unsafe_get(i) +
      a5 * x5.unsafe_get(i) +
      a6 * x6.unsafe_get(i) +
      a7 * x7.unsafe_get(i) +
      a8 * x8.unsafe_get(i),
    )
    i += 1
  }
}

///|
/// `out = base + sum_j a[j] * xs[j]` over a run-time number of terms.
///
/// The `lincomb1..lincomb8` kernels above unroll the term list, which the
/// hand-written tableaus can do at compile time. Verner's 9(8) method and the
/// generic tableau solver cannot: their term count is only known when the run
/// starts, so they walk two parallel arrays instead.
///
/// Entry `a[j]` is the coefficient of `xs[j]`, with **no skipped index** — a
/// caller expressing a Runge-Kutta row over stages k1..kN stores `a[0]` for
/// k1's coefficient. Preconditions (checked by the caller): `a.length() ==
/// xs.length()` and `a.length() >= 1`. Pass a zero vector as `base` for the
/// base-less (embedded-error) shape.
pub fn lincomb_range(
  out : Array[Double],
  base : Array[Double],
  a : Array[Double],
  xs : Array[Array[Double]],
) -> Unit {
  let n = out.length()
  let mut i = 0
  while i < n {
    out.unsafe_set(i, base.unsafe_get(i))
    i += 1
  }
  let mut j = 0
  while j < a.length() {
    let aj = a.unsafe_get(j)
    if aj != 0.0 {
      let xj = xs.unsafe_get(j)
      let mut i = 0
      while i < n {
        out.unsafe_set(i, out.unsafe_get(i) + aj * xj.unsafe_get(i))
        i += 1
      }
    }
    j += 1
  }
}

///|
test "lincomb kernels form base + sum a_i x_i" {
  let base = [1.0, 2.0, 3.0]
  let x1 = [1.0, 1.0, 1.0]
  let x2 = [0.5, -0.5, 2.0]
  let x3 = [2.0, 0.0, -1.0]
  let x4 = [0.25, 0.25, 0.25]
  let x5 = [-1.0, 3.0, 0.5]
  let x6 = [4.0, -2.0, 1.5]
  let o = Array::make(3, 0.0)
  let tol = 0.000000000000001
  lincomb1(o, base, 2.0, x1)
  assert_true(
    (o.unsafe_get(0) - 3.0).abs() < tol && (o.unsafe_get(2) - 5.0).abs() < tol,
  )
  lincomb2(o, base, 2.0, x1, -1.0, x2)
  assert_true(
    (o.unsafe_get(0) - 2.5).abs() < tol && (o.unsafe_get(1) - 4.5).abs() < tol,
  )
  lincomb3(o, base, 2.0, x1, -1.0, x2, 0.5, x3)
  assert_true(
    (o.unsafe_get(0) - 3.5).abs() < tol && (o.unsafe_get(2) - 2.5).abs() < tol,
  )
  lincomb4(o, base, 2.0, x1, -1.0, x2, 0.5, x3, 4.0, x4)
  assert_true(
    (o.unsafe_get(0) - 4.5).abs() < tol && (o.unsafe_get(1) - 5.5).abs() < tol,
  )
  lincomb5(o, base, 2.0, x1, -1.0, x2, 0.5, x3, 4.0, x4, -2.0, x5)
  assert_true(
    (o.unsafe_get(0) - 6.5).abs() < tol && (o.unsafe_get(1) - -0.5).abs() < tol,
  )
  lincomb6(o, base, 2.0, x1, -1.0, x2, 0.5, x3, 4.0, x4, -2.0, x5, 0.5, x6)
  assert_true(
    (o.unsafe_get(0) - 8.5).abs() < tol && (o.unsafe_get(2) - 3.25).abs() < tol,
  )
  // lincomb7 has no base term: out = sum a_i x_i.
  // x1+x2+x3+x4+x5+x6+x1 = [7.75, 2.75, 5.25]
  lincomb7(o, 1.0, x1, 1.0, x2, 1.0, x3, 1.0, x4, 1.0, x5, 1.0, x6, 1.0, x1)
  assert_true(
    (o.unsafe_get(0) - 7.75).abs() < tol &&
    (o.unsafe_get(1) - 2.75).abs() < tol &&
    (o.unsafe_get(2) - 5.25).abs() < tol,
  )
}

///|
test "lincomb_range walks a run-time number of terms" {
  let base = [1.0, 2.0, 3.0]
  let xs : Array[Array[Double]] = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.5, -0.5, 2.0],
    [2.0, 0.0, -1.0],
  ]
  let o = Array::make(3, 0.0)
  let tol = 0.000000000000001
  // Index 0 is the first term, not a skipped diagonal: a = 0.2*xs[1] ...
  let a = [0.0, 2.0, -1.0, 0.5]
  lincomb_range(o, base, a, xs)
  let o3 = Array::make(3, 0.0)
  lincomb3(
    o3,
    base,
    2.0,
    xs.unsafe_get(1),
    -1.0,
    xs.unsafe_get(2),
    0.5,
    xs.unsafe_get(3),
  )
  let mut i = 0
  while i < 3 {
    assert_true((o.unsafe_get(i) - o3.unsafe_get(i)).abs() < tol)
    i += 1
  }
  // A leading coefficient is honoured too — this is the Runge-Kutta row shape,
  // where a[0] carries the coefficient of k1.
  let a2 = [1.5, 1.0, 0.0, -2.0]
  lincomb_range(o, base, a2, xs)
  // 1.5*0 + 1.0*[1,1,1] + 0 - 2.0*[2,0,-1] + base = [-2, 3, 6]
  assert_true(
    (o.unsafe_get(0) - -2.0).abs() < tol &&
    (o.unsafe_get(1) - 3.0).abs() < tol &&
    (o.unsafe_get(2) - 6.0).abs() < tol,
  )
  // Zero coefficients contribute nothing, so a row can be padded with them.
  let a3 = [0.0, 1.0, 0.0, 0.0]
  lincomb_range(o, base, a3, xs)
  assert_true(
    (o.unsafe_get(0) - 2.0).abs() < tol && (o.unsafe_get(2) - 4.0).abs() < tol,
  )
}
