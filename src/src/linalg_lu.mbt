// Dense LU factorization with partial pivoting, in place on a row-major
// n x n matrix stored as a flat Array[Double] of length n*n.
//
// This is the only linear algebra the stiff solvers need:
//   * Rosenbrock methods factor W = I/(gamma*dt) - J once per step and reuse
//     the factorization while gamma*dt is unchanged (Julia's `dolinsolve`
//     branch, rosenbrock_perform_step.jl:63-67), then back-solve per stage.
//   * The BDF startup and implicit stages solve small systems through Newton,
//     which also factors once and re-solves.
//
// We deliberately do not depend on a registry linear-algebra package: the
// whole kernel is ~120 lines, keeps the package dependency-free (no version
// skew against the local toolchain), and lets the three targets share
// bit-identical numerics.
//
// The factorization returns false when a pivot column is exactly zero (or
// non-finite), i.e. the matrix is singular; callers turn that into a step
// failure rather than producing NaNs.

///|
/// Factor `a` (row-major, n x n, length n*n) in place with partial pivoting.
///
/// `ipiv` (length n) receives the pivot rows: at step `i`, row `i` was
/// swapped with row `ipiv[i]`. Returns false when a pivot column is exactly
/// zero or non-finite (singular input); the contents of `a` are then
/// unspecified.
pub fn lu_factor(a : Array[Double], n : Int32, ipiv : Array[Int32]) -> Bool {
  let mut i = 0
  while i < n {
    // Pivot search: the largest |a[j, i]| for j >= i.
    let mut p = i
    let mut pmax = a.unsafe_get(i * n + i).abs()
    let mut j = i + 1
    while j < n {
      let v = a.unsafe_get(j * n + i).abs()
      if v > pmax {
        pmax = v
        p = j
      }
      j += 1
    }
    ipiv.unsafe_set(i, p)
    if !(pmax == pmax) || pmax == 0.0 {
      // NaN pivot or exactly singular column.
      return false
    }
    if p != i {
      // Swap the full rows, L multipliers included (LAPACK convention).
      let mut k = 0
      while k < n {
        let tmp = a.unsafe_get(i * n + k)
        a.unsafe_set(i * n + k, a.unsafe_get(p * n + k))
        a.unsafe_set(p * n + k, tmp)
        k += 1
      }
    }
    let piv = a.unsafe_get(i * n + i)
    let mut r = i + 1
    while r < n {
      let l = a.unsafe_get(r * n + i) / piv
      a.unsafe_set(r * n + i, l)
      let mut c = i + 1
      while c < n {
        a.unsafe_set(
          r * n + c,
          a.unsafe_get(r * n + c) - l * a.unsafe_get(i * n + c),
        )
        c += 1
      }
      r += 1
    }
    i += 1
  }
  true
}

///|
/// Solve `A x = b` given the in-place factorization produced by `lu_factor`.
///
/// `b` (length n) is overwritten with the solution `x`. Applies the recorded
/// row swaps, then forward substitution against the unit-lower-triangular L,
/// then back substitution against U.
pub fn lu_solve(
  a : Array[Double],
  ipiv : Array[Int32],
  b : Array[Double],
) -> Unit {
  let n = ipiv.length()
  // Apply the row swaps in factorization order.
  let mut i = 0
  while i < n {
    let p = ipiv.unsafe_get(i)
    if p != i {
      let tmp = b.unsafe_get(i)
      b.unsafe_set(i, b.unsafe_get(p))
      b.unsafe_set(p, tmp)
    }
    i += 1
  }
  // Forward substitution: L has an implicit unit diagonal.
  let mut r = 1
  while r < n {
    let mut s = b.unsafe_get(r)
    let mut c = 0
    while c < r {
      s = s - a.unsafe_get(r * n + c) * b.unsafe_get(c)
      c += 1
    }
    b.unsafe_set(r, s)
    r += 1
  }
  // Back substitution.
  let mut rr = n - 1
  while rr >= 0 {
    let mut s = b.unsafe_get(rr)
    let mut c = rr + 1
    while c < n {
      s = s - a.unsafe_get(rr * n + c) * b.unsafe_get(c)
      c += 1
    }
    b.unsafe_set(rr, s / a.unsafe_get(rr * n + rr))
    rr -= 1
  }
}

///|
/// Solve `A x = b` for a fresh matrix (factor + solve, `a` is destroyed).
///
/// Convenience wrapper for one-off solves; returns false when `A` is singular.
/// Hot solver loops factor once and call `lu_solve` per right-hand side.
pub fn lu_solve_once(a : Array[Double], b : Array[Double]) -> Bool {
  let n = b.length()
  let ipiv = Array::make(n, 0)
  if lu_factor(a, n, ipiv) {
    lu_solve(a, ipiv, b)
    true
  } else {
    false
  }
}

///|
test "lu solves a diagonal system" {
  // A = diag(2, 3, 4), b = (2, 6, 8) -> x = (1, 2, 2)
  let a = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]
  let b = [2.0, 6.0, 8.0]
  assert_true(lu_solve_once(a, b))
  assert_true((b.unsafe_get(0) - 1.0).abs() < 0.00000000000001)
  assert_true((b.unsafe_get(1) - 2.0).abs() < 0.00000000000001)
  assert_true((b.unsafe_get(2) - 2.0).abs() < 0.00000000000001)
}

///|
test "lu handles a pivoting system" {
  // A = [[0, 1], [1, 0]] needs a row swap; b = (5, 7) -> x = (7, 5)
  let a = [0.0, 1.0, 1.0, 0.0]
  let b = [5.0, 7.0]
  assert_true(lu_solve_once(a, b))
  assert_true((b.unsafe_get(0) - 7.0).abs() < 0.00000000000001)
  assert_true((b.unsafe_get(1) - 5.0).abs() < 0.00000000000001)
}

///|
test "lu residual is tiny on a random-ish 4x4" {
  // A = [[4, 1, 0, 2], [1, 5, 1, 0], [0, 1, 6, 1], [2, 0, 1, 7]], x = (1, 2, 3, 4)
  let a = [
    4.0, 1.0, 0.0, 2.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 2.0, 0.0, 1.0, 7.0,
  ]
  let x = [1.0, 2.0, 3.0, 4.0]
  let b = [0.0, 0.0, 0.0, 0.0]
  let mut i = 0
  while i < 4 {
    let mut s = 0.0
    let mut j = 0
    while j < 4 {
      s = s + a.unsafe_get(i * 4 + j) * x.unsafe_get(j)
      j += 1
    }
    b.unsafe_set(i, s)
    i += 1
  }
  assert_true(lu_solve_once(a, b))
  assert_true((b.unsafe_get(0) - 1.0).abs() < 0.000000000001)
  assert_true((b.unsafe_get(1) - 2.0).abs() < 0.000000000001)
  assert_true((b.unsafe_get(2) - 3.0).abs() < 0.000000000001)
  assert_true((b.unsafe_get(3) - 4.0).abs() < 0.000000000001)
}

///|
test "lu reports singular matrices" {
  // [[1, 2], [2, 4]] is exactly singular.
  let a = [1.0, 2.0, 2.0, 4.0]
  let ipiv = Array::make(2, 0)
  assert_false(lu_factor(a, 2, ipiv))
}
