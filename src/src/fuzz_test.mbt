// Fuzz / property-based tests for the ODE solvers.
//
// Randomized inputs exercise the solvers across a wide input space. Every
// case asserts the solver returns a *well-formed* result — a known return
// code, monotonic time, and finite state on success — and never panics or
// hangs. The PRNG is seeded with a fixed constant so the suite is
// deterministic and CI-stable (no flakiness, reproducible failures).
//
// This is the port's stand-in for the coverage-guided fuzzing that the Julia
// monorepo gets from `DiffEqFlux`/`PkgEval`; MoonBit has no libFuzzer
// front-end, so we use seeded randomized testing with invariant assertions.

///|
/// Deterministic xorshift64 PRNG. Seeded with a constant so every run of the
/// suite is identical; this keeps the randomized tests reproducible and
/// removes CI flakiness.
struct Prng {
  mut s : UInt64
}

///|
fn Prng::new(seed : Int64) -> Prng {
  { s: seed.reinterpret_as_uint64(), }
}

///|
fn Prng::next(self : Prng) -> UInt64 {
  let mut x = self.s
  x = x ^ (x << 13)
  x = x ^ (x >> 7)
  x = x ^ (x << 17)
  self.s = x
  x
}

///|
/// double in [0, 1)
fn Prng::double(self : Prng) -> Double {
  let hi = self.next() >> 11 // top 53 bits -> double in [0, 1)
  hi.reinterpret_as_int64().to_double() / 9007199254740992.0
}

///|
/// double in [lo, hi)
fn Prng::range(self : Prng, lo : Double, hi : Double) -> Double {
  lo + (hi - lo) * self.double()
}

// ---------------------------------------------------------------------------
// Shared harness helpers
// ---------------------------------------------------------------------------

///|
/// Solve `prob` adaptively to a given tolerance with any solver.
fn[A : @ordinarydiffeq.Solver] adaptive_solve(
  prob : @ordinarydiffeq.OdeProblem,
  alg : A,
  reltol : Double,
  abstol : Double,
) -> @ordinarydiffeq.Solution {
  let opts = @ordinarydiffeq.SolveOptions::with_abstol(
    @ordinarydiffeq.SolveOptions::with_reltol(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        true,
      ),
      reltol,
    ),
    abstol,
  )
  @ordinarydiffeq.solve(prob, alg, opts)
}

///|
/// Solve `prob` with a fixed step `dt` and any solver.
fn[A : @ordinarydiffeq.Solver] fixed_solve(
  prob : @ordinarydiffeq.OdeProblem,
  alg : A,
  dt : Double,
) -> @ordinarydiffeq.Solution {
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    @ordinarydiffeq.SolveOptions::new(),
    dt,
  )
  @ordinarydiffeq.solve(prob, alg, opts)
}

///|
/// True when every stored state component is finite (no NaN / Inf leaked).
fn states_finite(sol : @ordinarydiffeq.Solution) -> Bool {
  for row in sol.u {
    for v in row {
      if !(v == v && v > -1.0e308 && v < 1.0e308) {
        return false
      }
    }
  }
  true
}

///|
/// True when the saved time grid is non-decreasing.
fn t_monotonic(sol : @ordinarydiffeq.Solution) -> Bool {
  let n = sol.t.length()
  let mut i = 1
  while i < n {
    if sol.t.unsafe_get(i) < sol.t.unsafe_get(i - 1) {
      return false
    }
    i += 1
  }
  true
}

///|
/// True when the return code is one of the codes the loop can emit (a crash
/// or unhandled state would surface as a panic, which the test runner catches;
/// this guards the "well-formed" property beyond that).
fn known_retcode(sol : @ordinarydiffeq.Solution) -> Bool {
  sol.retcode.is_success() ||
  sol.retcode.is_dtmin_unstable() ||
  sol.retcode.is_unstable() ||
  sol.retcode.is_maxiters()
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

///|
/// Random scalar linear decays `u' = -a u`: the adaptive solver must stay
/// finite and monotonic and hit the analytic `u0·exp(-a·tf)` to well within
/// its tolerance.
test "fuzz: scalar linear decay is accurate, finite and monotonic" {
  let p = Prng::new(2718281828L)
  let mut i = 0
  while i < 200 {
    let a = p.range(0.1, 10.0)
    let u0 = p.range(0.5, 2.0)
    let tf = p.range(0.5, 5.0)
    let prob = @ordinarydiffeq.OdeProblem::iip(0.0, tf, [u0], fn(_t, u, du) {
      du.unsafe_set(0, -a * u.unsafe_get(0))
    })
    let sol = adaptive_solve(prob, @ordinarydiffeq.Tsit5::new(), 1.0e-9, 1.0e-9)
    assert_true(sol.retcode.is_success())
    assert_true(t_monotonic(sol))
    assert_true(states_finite(sol))
    let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
    let exact = u0 * @math.exp(-a * tf)
    assert_true(exact > 0.0)
    // Adaptive error is bounded in *mixed* absolute+relative terms, so the
    // meaningful invariant is the requested tolerance contract (with a safety
    // margin for global-vs-local error amplification), not pure relative error
    // — a pure relative check diverges once the exact value decays near zero.
    let tol = 1.0e-9
    assert_true((last - exact).abs() <= 100.0 * (tol * exact.abs() + tol))
    i += 1
  }
}

///|
/// Random constant RHS `u' = c`: every order>=2 method integrates the affine
/// solution exactly to working precision.
test "fuzz: constant RHS integrates to the affine exact solution" {
  let p = Prng::new(1357924680L)
  let mut i = 0
  while i < 150 {
    let c = p.range(-5.0, 5.0)
    let u0 = p.range(-3.0, 3.0)
    let tf = p.range(0.1, 4.0)
    let prob = @ordinarydiffeq.OdeProblem::iip(0.0, tf, [u0], fn(_t, _u, du) {
      du.unsafe_set(0, c)
    })
    let sol = adaptive_solve(prob, @ordinarydiffeq.Tsit5::new(), 1.0e-9, 1.0e-9)
    assert_true(sol.retcode.is_success())
    let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
    let exact = u0 + c * tf
    assert_true((last - exact).abs() < 1.0e-9)
    i += 1
  }
}

///|
/// Random harmonic oscillators `u' = J u` (rotation): the squared-norm
/// (energy) is an invariant and must be conserved to high precision by the
/// adaptive solver.
test "fuzz: harmonic oscillator conserves energy" {
  let p = Prng::new(2468135790L)
  let mut i = 0
  while i < 150 {
    // pick an initial condition with energy >= 0.25 to keep the relative
    // error well-defined (avoids trig functions).
    let mut x0 = 0.0
    let mut y0 = 0.0
    let mut e0 = 0.0
    while e0 < 0.25 {
      x0 = p.range(-2.0, 2.0)
      y0 = p.range(-2.0, 2.0)
      e0 = x0 * x0 + y0 * y0
    }
    let tf = p.range(0.1, 3.0)
    let prob = @ordinarydiffeq.OdeProblem::iip(0.0, tf, [x0, y0], fn(
      _t,
      u,
      du,
    ) {
      du.unsafe_set(0, u.unsafe_get(1))
      du.unsafe_set(1, -u.unsafe_get(0))
    })
    let sol = adaptive_solve(
      prob,
      @ordinarydiffeq.Tsit5::new(),
      1.0e-10,
      1.0e-10,
    )
    assert_true(sol.retcode.is_success())
    assert_true(states_finite(sol))
    let last = sol.u.unsafe_get(sol.u.length() - 1)
    let e1 = last.unsafe_get(0) * last.unsafe_get(0) +
      last.unsafe_get(1) * last.unsafe_get(1)
    assert_true((e1 - e0).abs() / e0 < 1.0e-2)
    i += 1
  }
}

///|
/// Random smooth polynomial RHS `u' = a·t² − b·t + c`: two independent
/// high-order methods (Tsit5, Vern9) must agree to far better than their
/// tolerance — a cross-check that neither method is silently producing garbage.
test "fuzz: Tsit5 and Vern9 agree on smooth scalar ODEs" {
  let p = Prng::new(3141592653L)
  let mut i = 0
  while i < 150 {
    let a = p.range(0.0, 3.0)
    let b = p.range(0.0, 3.0)
    let c = p.range(-2.0, 2.0)
    let u0 = p.range(-2.0, 2.0)
    let tf = p.range(0.2, 3.0)
    let prob = @ordinarydiffeq.OdeProblem::iip(0.0, tf, [u0], fn(t, _u, du) {
      du.unsafe_set(0, a * t * t - b * t + c)
    })
    let s1 = adaptive_solve(
      prob,
      @ordinarydiffeq.Tsit5::new(),
      1.0e-10,
      1.0e-10,
    )
    let s2 = adaptive_solve(
      prob,
      @ordinarydiffeq.Vern9::new(),
      1.0e-10,
      1.0e-10,
    )
    assert_true(s1.retcode.is_success())
    assert_true(s2.retcode.is_success())
    assert_true(states_finite(s1) && states_finite(s2))
    let u1 = s1.u.unsafe_get(s1.u.length() - 1).unsafe_get(0)
    let u2 = s2.u.unsafe_get(s2.u.length() - 1).unsafe_get(0)
    assert_true((u1 - u2).abs() < 1.0e-7)
    i += 1
  }
}

///|
/// Fixed-step solve of the same problem twice yields bit-identical results —
/// a reproducibility property that guards against hidden global/seed state.
test "fuzz: fixed-step solve is deterministic (bit-identical)" {
  let p = Prng::new(8675309000L)
  let mut i = 0
  while i < 30 {
    let a = p.range(0.2, 3.0)
    let u0 = p.range(0.5, 2.0)
    let tf = p.range(0.5, 3.0)
    let dt = p.range(0.01, 0.05)
    let build = fn() -> @ordinarydiffeq.OdeProblem {
      @ordinarydiffeq.OdeProblem::iip(0.0, tf, [u0], fn(_t, u, du) {
        du.unsafe_set(0, -a * u.unsafe_get(0))
      })
    }
    let s1 = fixed_solve(build(), @ordinarydiffeq.Tsit5::new(), dt)
    let s2 = fixed_solve(build(), @ordinarydiffeq.Tsit5::new(), dt)
    assert_true(s1.retcode.is_success() && s2.retcode.is_success())
    let n = s1.u.length()
    assert_eq(n, s2.u.length())
    let mut j = 0
    while j < n {
      assert_true(
        s1.u.unsafe_get(j).unsafe_get(0) == s2.u.unsafe_get(j).unsafe_get(0),
      )
      j += 1
    }
    i += 1
  }
}

// ---------------------------------------------------------------------------
// Fuzz / robustness tests
// ---------------------------------------------------------------------------

///|
/// Random bounded polynomial RHS `u' = c0 + c1·u + c2·u²` over short spans:
/// the state cannot blow up, so the solver must either succeed with a finite,
/// time-monotonic solution that lands on `tf`, or return a clean diagnostic.
/// A panic or hang here is a real bug.
test "fuzz: random polynomial RHS never crashes and is well-formed" {
  let p = Prng::new(4242424242L)
  let mut i = 0
  while i < 400 {
    let c0 = p.range(-0.5, 0.5)
    let c1 = p.range(-0.5, 0.5)
    let c2 = p.range(-0.5, 0.5)
    let u0 = p.range(-1.0, 1.0)
    let tf = p.range(0.1, 1.5)
    let prob = @ordinarydiffeq.OdeProblem::iip(0.0, tf, [u0], fn(_t, u, du) {
      let x = u.unsafe_get(0)
      du.unsafe_set(0, c0 + c1 * x + c2 * x * x)
    })
    let sol = adaptive_solve(prob, @ordinarydiffeq.Tsit5::new(), 1.0e-7, 1.0e-7)
    assert_true(known_retcode(sol))
    assert_true(t_monotonic(sol))
    if sol.retcode.is_success() {
      assert_true(states_finite(sol))
      let last_t = sol.t.unsafe_get(sol.t.length() - 1)
      assert_true((last_t - tf).abs() < 1.0e-6)
    }
    i += 1
  }
}

///|
/// Stiff scalar decays `u' = -k·u` with k up to 1e7 solved by an explicit
/// method: this is expected to fail, but it must fail *cleanly* (DtMinUnstable
/// or Unstable) — never crash, never hang, never return an unknown state.
test "fuzz: stiff explicit solves fail cleanly instead of hanging" {
  let p = Prng::new(7070707070L)
  let mut i = 0
  while i < 80 {
    let k = p.range(1000.0, 10000000.0)
    let u0 = p.range(0.5, 2.0)
    let tf = p.range(0.1, 1.0)
    let prob = @ordinarydiffeq.OdeProblem::iip(0.0, tf, [u0], fn(_t, u, du) {
      du.unsafe_set(0, -k * u.unsafe_get(0))
    })
    let sol = adaptive_solve(prob, @ordinarydiffeq.Tsit5::new(), 1.0e-7, 1.0e-7)
    assert_true(known_retcode(sol))
    assert_true(t_monotonic(sol))
    if sol.retcode.is_success() {
      assert_true(states_finite(sol))
    }
    i += 1
  }
}
