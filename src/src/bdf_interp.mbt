// FBDF dense output.
//
// Ported from OrdinaryDiffEqBDF/src/bdf_interpolants.jl:
//   * `_bdf_active_order` (bdf_interpolants.jl:221) — infer the active order
//     from the stored stage array.
//   * `_fbdf_interp(::Val{n}, …, ::Type{Val{0}})` (bdf_interpolants.jl:381) —
//     the barycentric value.
//   * `_fbdf_interp(::Val{n}, …, ::Type{Val{1}})` (bdf_interpolants.jl:501) —
//     the product-rule Lagrange-basis derivative.
//   * `_fbdf_lagrange_basis_deriv` (bdf_interpolants.jl:314).
//
// Why the two-step scheme
// -----------------------
// A multistep method's natural interpolant is a polynomial through the *actual*
// history nodes, which move every step. Evaluating it at an arbitrary `Θ` would
// then be `O(k²)` per point and would need the node set stored alongside the
// solution. FBDF instead does the expensive part once per step: it resamples the
// history interpolant onto the **fixed Chebyshev reference nodes**
// (`_resample_at_chebyshev!`, bdf_utils.jl:331) and stores *those* values in
// `k`. Dense output then only ever touches the fixed nodes, so:
//
//   * the value is the barycentric formula, `O(n)`;
//   * the derivative uses precomputed node geometry, `O(n²)` with `n ≤ 6`;
//   * nothing about the step's node set has to be remembered.
//
// The price is that `y₀` and `y₁` are unused — all the data lives in `k`, which
// is why the stored array is `max_order + 1` long and the tail beyond the active
// order is zero.

///|
/// `_bdf_active_order(k)` (bdf_interpolants.jl:221): the number of leading
/// non-zero entries of the stored stage array.
///
/// `perform_step!` writes `n = min(k + 1, max_order + 1)` resampled values and
/// zeroes the rest, so counting from the top recovers the order the step ran
/// at. Upstream's `_is_all_zero` is per-vector (`all(iszero, k[n])`), which is
/// what makes a state that happens to pass through zero still count as data.
fn bdf_active_order(ks : Array[Array[Double]]) -> Int32 {
  let n = ks.length()
  let mut m = n
  while m > 0 && vector_is_zero(ks.unsafe_get(m - 1)) {
    m -= 1
  }
  if m < 1 {
    1
  } else {
    m
  }
}

///|
/// `_is_all_zero(x::AbstractArray) = all(iszero, x)`.
fn vector_is_zero(v : Array[Double]) -> Bool {
  let mut i = 0
  while i < v.length() {
    if v.unsafe_get(i) != 0.0 {
      return false
    }
    i += 1
  }
  true
}

///|
/// `_fbdf_interp(::Val{n}, Θ, dt, y₀, y₁, k, idxs, ::Type{Val{0}})`
/// (bdf_interpolants.jl:381): the value of the dense-output polynomial at
/// fraction `Θ` of the step.
///
///     p(Θ) = [ Σ_j w_j / (Θ - ξ_j) · k[j] ] / [ Σ_j w_j / (Θ - ξ_j) ]
///
/// with `ξ` the fixed Chebyshev nodes and `w` their barycentric weights. The
/// exact-node short circuit is not an optimization: at `Θ == ξ_j` the formula
/// divides by zero, and the correct value is exactly `k[j]`.
///
/// `dt`, `y₀` and `y₁` are part of the `SolutionInterp::Poly` signature and are
/// deliberately unused — upstream's header says "All data lives in k".
pub fn fbdf_interpolate(
  theta : Double,
  _dt : Double,
  _y0 : Array[Double],
  _y1 : Array[Double],
  ks : Array[Array[Double]],
  out : Array[Double],
) -> Unit {
  let n = bdf_active_order(ks)
  let nodes = cheb_nodes(n)
  let weights = bary_weights(n)
  let mut j = 0
  while j < n {
    if theta == nodes.unsafe_get(j) {
      copy_into(out, ks.unsafe_get(j))
      return
    }
    j += 1
  }
  let mut den = 0.0
  j = 0
  while j < n {
    den = den + weights.unsafe_get(j) / (theta - nodes.unsafe_get(j))
    j += 1
  }
  let inv_den = 1.0 / den
  let dim = out.length()
  let mut i = 0
  while i < dim {
    out.unsafe_set(i, 0.0)
    i += 1
  }
  j = 0
  while j < n {
    let cj = weights.unsafe_get(j) / (theta - nodes.unsafe_get(j)) * inv_den
    let kj = ks.unsafe_get(j)
    i = 0
    while i < dim {
      out.unsafe_set(i, out.unsafe_get(i) + cj * kj.unsafe_get(i))
      i += 1
    }
    j += 1
  }
}

///|
/// `_fbdf_interp(::Val{n}, Θ, dt, y₀, y₁, k, idxs, ::Type{Val{1}})`
/// (bdf_interpolants.jl:501): the derivative of the same polynomial.
///
///     p'(Θ) = (1/dt) · Σ_j k[j] · L'_j(Θ)
///
/// where `L'_j` is the product-rule form on the fixed nodes. The `1/dt` is
/// because `Θ` is the *fraction* of the step, so `d/dt = (1/dt) d/dΘ`.
pub fn fbdf_derivative(
  theta : Double,
  dt : Double,
  _y0 : Array[Double],
  _y1 : Array[Double],
  ks : Array[Array[Double]],
  out : Array[Double],
) -> Unit {
  if dt == 0.0 {
    return
  }
  let n = bdf_active_order(ks)
  let inv_dt = 1.0 / dt
  let dim = out.length()
  let d0 = lagrange_basis_deriv(theta, 0, n)
  let k0 = ks.unsafe_get(0)
  let mut i = 0
  while i < dim {
    out.unsafe_set(i, d0 * inv_dt * k0.unsafe_get(i))
    i += 1
  }
  let mut j = 1
  while j < n {
    let dj = lagrange_basis_deriv(theta, j, n)
    let cj = dj * inv_dt
    let kj = ks.unsafe_get(j)
    i = 0
    while i < dim {
      out.unsafe_set(i, out.unsafe_get(i) + cj * kj.unsafe_get(i))
      i += 1
    }
    j += 1
  }
}

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

///|
/// Build a six-slot stage array holding the values of `p` at the first `n`
/// Chebyshev nodes, zeros elsewhere — exactly what `perform_step!`'s calck
/// block leaves behind for a step that ran at order `n - 1`.
fn fbdf_stages(n : Int32, p : (Double) -> Double) -> Array[Array[Double]] {
  let nodes = cheb_nodes(n)
  let ks : Array[Array[Double]] = []
  let mut j = 0
  while j < 6 {
    if j < n {
      ks.push([p(nodes.unsafe_get(j))])
    } else {
      ks.push([0.0])
    }
    j += 1
  }
  ks
}

///|
/// The barycentric formula reproduces the resampled polynomial exactly at the
/// nodes, and to roundoff between them: for a cubic through four Chebyshev
/// nodes the value at the midpoint of `[0, 1]` must be the cubic's own.
test "fbdf barycentric interpolant reproduces a cubic" {
  // p(Θ) = 1 + 2Θ - 3Θ² + Θ³, sampled at the four Chebyshev nodes.
  let p = fn(x : Double) -> Double { 1.0 + 2.0 * x - 3.0 * x * x + x * x * x }
  let ks = fbdf_stages(4, p)
  let out = Array::make(1, 0.0)
  fbdf_interpolate(0.5, 0.1, [0.0], [0.0], ks, out)
  assert_true((out.unsafe_get(0) - p(0.5)).abs() < 0.000000000000001)
  // And exactly at a node, where the formula would divide by zero.
  let nodes = cheb_nodes(4)
  fbdf_interpolate(nodes.unsafe_get(2), 0.1, [0.0], [0.0], ks, out)
  assert_eq(out.unsafe_get(0), p(nodes.unsafe_get(2)))
}

///|
/// `_bdf_active_order` reads the order back off the tail of zero slots, so a
/// step that ran at order 3 (four stored values) interpolates with `n = 4` and
/// ignores the two zero slots after it.
test "fbdf_active_order reads the zero tail" {
  let ks = fbdf_stages(4, fn(x) { x })
  assert_eq(bdf_active_order(ks), 4)
  // All-zero tail beyond index 4 does not change the answer ...
  let ks2 = fbdf_stages(6, fn(x) { x })
  assert_eq(bdf_active_order(ks2), 6)
  // ... and an all-zero leading block is clamped to 1, as upstream's
  // `max(n, 1)` does.
  let ks3 : Array[Array[Double]] = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
  assert_eq(bdf_active_order(ks3), 1)
}

///|
/// The derivative branch is the analytic derivative of the value branch, so it
/// must reproduce `p'` at the nodes (where both are evaluated without any
/// interpolation error) and to roundoff in between.
test "fbdf derivative matches the analytic derivative of the polynomial" {
  let p = fn(x : Double) -> Double { 1.0 + 2.0 * x - 3.0 * x * x + x * x * x }
  let dp = fn(x : Double) -> Double { 2.0 - 6.0 * x + 3.0 * x * x }
  let ks = fbdf_stages(4, p)
  let out = Array::make(1, 0.0)
  // dt = 0.25, so p'(Θ)/dt in physical units.
  fbdf_derivative(0.5, 0.25, [0.0], [0.0], ks, out)
  assert_true((out.unsafe_get(0) - dp(0.5) / 0.25).abs() < 0.000000000001)
  let nodes = cheb_nodes(4)
  fbdf_derivative(nodes.unsafe_get(1), 0.25, [0.0], [0.0], ks, out)
  assert_true(
    (out.unsafe_get(0) - dp(nodes.unsafe_get(1)) / 0.25).abs() < 0.000000000001,
  )
}

///|
/// A zero `dt` must not produce infinities: upstream's in-place kernels divide
/// by `dt` unconditionally, but a stored step of length zero carries no
/// derivative information, so the derivative is left alone.
test "fbdf derivative guards a zero step size" {
  let ks = fbdf_stages(3, fn(x) { x * x })
  let out = [42.0]
  fbdf_derivative(0.5, 0.0, [0.0], [0.0], ks, out)
  assert_eq(out.unsafe_get(0), 42.0)
}
