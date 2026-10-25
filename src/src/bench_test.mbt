// Micro-benchmarks for the solver families. Run with `moon bench` from `src/`.
//
// Each benchmark solves a canonical problem to a fixed tolerance and lets the
// measurement cover the whole solve (setup + stepping + interpolation), which
// is what an embedder pays. The problems are chosen so that the family
// trade-off is visible: non-stiff decay (all explicit methods), a mildly stiff
// van der Pol (Newton cost starts to matter) and a strongly stiff linear
// problem (only the implicit methods survive at a sane step count).

///|
/// `u' = -u` on [0, 1], u(0) = 1. The canonical non-stiff problem.
fn bench_decay_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
/// van der Pol, μ = 10, on [0, 20] from (2, 0): mildly stiff, the classic
/// stiff-detection stress case.
fn bench_vdp_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 20.0, [2.0, 0.0], fn(_t, u, du) {
    du.unsafe_set(0, u.unsafe_get(1))
    du.unsafe_set(
      1,
      10.0 * (1.0 - u.unsafe_get(0) * u.unsafe_get(0)) * u.unsafe_get(1) -
      u.unsafe_get(0),
    )
  })
}

///|
/// `u' = -1000(u - cos t)` on [0, 1], u(0) = 1: strongly stiff, the linear
/// scalar version of the Robertson-like test.
fn bench_stiff_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(t, u, du) {
    du.unsafe_set(0, -1000.0 * (u.unsafe_get(0) - @math.cos(t)))
  })
}

///|
/// Default options with a tight tolerance: `reltol = 1e-8`, `abstol = 1e-10`.
fn bench_opts() -> @ordinarydiffeq.SolveOptions {
  @ordinarydiffeq.SolveOptions::with_abstol(
    @ordinarydiffeq.SolveOptions::with_reltol(
      @ordinarydiffeq.SolveOptions::new(),
      1.0e-8,
    ),
    1.0e-10,
  )
}

///|
/// Non-stiff decay, explicit family at `reltol = 1e-8`.
test "bench: tsit5 decay" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_decay_problem(),
      @ordinarydiffeq.Tsit5::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}

///|
test "bench: vern7 decay" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_decay_problem(),
      @ordinarydiffeq.Vern7::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}

///|
/// Mildly stiff van der Pol: the explicit method pays for stability, the
/// implicit ones for the Newton solve.
test "bench: tsit5 van der pol" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_vdp_problem(),
      @ordinarydiffeq.Tsit5::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}

///|
test "bench: rosenbrock23 van der pol" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_vdp_problem(),
      @ordinarydiffeq.Rosenbrock23::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}

///|
test "bench: rodas5p van der pol" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_vdp_problem(),
      @ordinarydiffeq.Rodas5P::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}

///|
test "bench: fbdf van der pol" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_vdp_problem(),
      @ordinarydiffeq.FBDF::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}

///|
/// Strongly stiff linear problem: the two implicit families.
test "bench: rodas5p stiff" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_stiff_problem(),
      @ordinarydiffeq.Rodas5P::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}

///|
test "bench: fbdf stiff" (b : @bench.T) {
  b.bench(fn() {
    let sol = @ordinarydiffeq.solve(
      bench_stiff_problem(),
      @ordinarydiffeq.FBDF::new(),
      bench_opts(),
    )
    b.keep(sol.u.length())
  })
}
