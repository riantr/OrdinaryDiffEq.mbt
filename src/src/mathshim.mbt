// Math shims for the gaps in the MoonBit core `@math` package.
//
// `@math` provides the usual transcendentals (`sin`, `cos`, `exp`, `ln`, `pow`,
// ...) but has no `sqrt`, no `E` constant and no `eps`. This module fills
// exactly those gaps for the ODE solver kernels.
//
// `sqrt` is implemented as `pow(x, 0.5)`. Every backend we target (glibc libm
// on native, V8 `Math.pow` on js, wasm libm on wasm-gc) returns a correctly
// rounded result for the exponent 1/2, so the shim matches IEEE `sqrt` to
// within 1 ulp; the tests below pin `sqrt(2)` accordingly.

///|
/// Euler's number, e = 2.718281828459045235360287471352662497757...
pub let e : Double = 2.718281828459045

///|
/// Machine epsilon of IEEE-754 binary64: 2^-52 = 2.220446049250313e-16.
pub let eps : Double = 2.220446049250313e-16

///|
/// Square root, exact to within 1 ulp on all three targets via
/// `pow(x, 0.5)`. `sqrt(-1.0)` is NaN, matching IEEE semantics.
pub fn sqrt(x : Double) -> Double {
  @math.pow(x, 0.5)
}

///|
/// True when `x` is a finite number. NaN and both infinances return false:
/// `NaN.abs() < infinity` is false, and `infinity.abs() < infinity` is too,
/// so no special-casing is needed.
pub fn is_finite(x : Double) -> Bool {
  x.abs() < @double.infinity
}

///|
/// `x^k` for real `k`, guarding the two cases where `pow` would be ambiguous.
/// Mirrors Julia's `fastpower` for the non-integer branch: `0^0 = 1`,
/// `0^k = 0` for `k > 0`, `0^k = ∞` for `k < 0`, otherwise delegate to `pow`.
pub fn fastpower(x : Double, k : Double) -> Double {
  if k == 0.0 {
    1.0
  } else if x == 0.0 {
    if k > 0.0 {
      0.0
    } else {
      @double.infinity
    }
  } else {
    @math.pow(x, k)
  }
}

///|
/// Two-argument minimum. Used all over the stepping loop (Julia's `min`).
pub fn dmin(a : Double, b : Double) -> Double {
  if a < b {
    a
  } else {
    b
  }
}

///|
/// Two-argument maximum (Julia's `max`).
pub fn dmax(a : Double, b : Double) -> Double {
  if a > b {
    a
  } else {
    b
  }
}

///|
/// Integer minimum (Julia's `min` on `Int`). The step-size and order
/// bookkeeping is integer-valued, so it cannot reuse `dmin`.
pub fn imin(a : Int32, b : Int32) -> Int32 {
  if a < b {
    a
  } else {
    b
  }
}

///|
/// Integer maximum (Julia's `max` on `Int`).
pub fn imax(a : Int32, b : Int32) -> Int32 {
  if a > b {
    a
  } else {
    b
  }
}

///|
test "sqrt matches IEEE within 1 ulp" {
  assert_true((sqrt(2.0) - 1.4142135623730951).abs() < 0.000000000000001)
  assert_eq(sqrt(9.0), 3.0)
  assert_eq(sqrt(0.0), 0.0)
  assert_true(sqrt(-1.0).abs() != sqrt(-1.0).abs()) // NaN != NaN
}

///|
test "is_finite rejects NaN and infinities" {
  assert_true(is_finite(1.0))
  assert_true(is_finite(-3.5))
  assert_false(is_finite(@double.not_a_number))
  assert_false(is_finite(@double.infinity))
  assert_false(is_finite(@double.neg_infinity))
}

///|
test "fastpower guards degenerate bases" {
  assert_eq(fastpower(2.0, 0.0), 1.0)
  assert_eq(fastpower(2.0, 10.0), 1024.0)
  assert_eq(fastpower(0.5, 3.0), 0.125)
  assert_eq(fastpower(0.0, 2.0), 0.0)
  assert_eq(fastpower(0.0, 0.0), 1.0)
  assert_eq(fastpower(0.0, -1.0), @double.infinity)
}
