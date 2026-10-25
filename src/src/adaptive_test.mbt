// M2 adaptive step-size control: the PI controller, the automatic initial
// step, and their interaction with tstops, saveat, dense output, backward
// integration and the dtmin abort.
//
// Every expected number below was derived independently (Python reference
// simulation of the ported algorithm, IEEE-754 binary64) rather than copied
// from a run of this package, so a green test is evidence about the port and
// not a snapshot of it.
//
// Test files see the package through its interface, so every reference is
// qualified with `@ordinarydiffeq.` (same convention as tsit5_test.mbt).

///|
/// `u' = -u` on [0, 1] with u(0) = 1 (exact solution exp(-t)), with the
/// tolerances left at their defaults so `determine_initdt` has a single
/// well-defined expected value. Named apart from the other decay helpers —
/// the package shares one namespace across files.
fn adaptive_decay_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
/// `u' = -u` integrated from t = 1 down to t = 0 with u(1) = 1: u(0) = e.
fn adaptive_backward_problem() -> @ordinarydiffeq.OdeProblem {
  @ordinarydiffeq.OdeProblem::iip(1.0, 0.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, -u.unsafe_get(0))
  })
}

///|
/// The adaptive options used by the end-to-end tests: tolerances that make
/// the controller take double-digit step counts over [0, 1].
fn tight_adaptive_opts(
  reltol : Double,
  abstol : Double,
) -> @ordinarydiffeq.SolveOptions {
  @ordinarydiffeq.SolveOptions::with_abstol(
    @ordinarydiffeq.SolveOptions::with_reltol(
      @ordinarydiffeq.SolveOptions::with_adaptive(
        @ordinarydiffeq.SolveOptions::new(),
        true,
      ),
      reltol,
    ),
    abstol,
  )
}

///|
test "automatic initial step matches the Hairer-Wanner heuristic" {
  // Default tolerances: sk = abstol + reltol = 1.001e-3, so
  // d0 = d1 = d2 = 1/1.001e-3 = 999.0009990010, dt0 = (d0/d1)/100 = 0.01 and
  // dt1 = 10^(-(2 + log10(999.0009990010))/5) = 0.10001999200479668.
  let prob = adaptive_decay_problem()
  let opts = @ordinarydiffeq.SolveOptions::with_dtmax(
    @ordinarydiffeq.SolveOptions::new(),
    1.0,
  )
  let integ = @ordinarydiffeq.Integrator::new(prob, opts, 7, 5)
  let dt = @ordinarydiffeq.determine_initdt(integ, 5)
  assert_true((dt - 0.10001999200479668).abs() < 0.000001)

  // Tight tolerances: d0 = d1 = d2 = 1/1.001e-6 = 999000.999..., dt0 = 0.01,
  // dt1 = 10^(-(2 + log10(999000.999...))/5) = 0.025123886079654534.
  let opts2 = @ordinarydiffeq.SolveOptions::with_dtmax(
    @ordinarydiffeq.SolveOptions::with_reltol(
      @ordinarydiffeq.SolveOptions::with_abstol(
        @ordinarydiffeq.SolveOptions::new(),
        0.000000001,
      ),
      0.000001,
    ),
    1.0,
  )
  let integ2 = @ordinarydiffeq.Integrator::new(prob, opts2, 7, 5)
  let dt2 = @ordinarydiffeq.determine_initdt(integ2, 5)
  assert_true((dt2 - 0.025123886079654534).abs() < 0.000001)

  // Backward: same magnitudes, sign flipped by tdir.
  let opts3 = @ordinarydiffeq.SolveOptions::with_dtmax(
    @ordinarydiffeq.SolveOptions::with_reltol(
      @ordinarydiffeq.SolveOptions::with_abstol(
        @ordinarydiffeq.SolveOptions::new(),
        0.000000001,
      ),
      0.000001,
    ),
    1.0,
  )
  let integ3 = @ordinarydiffeq.Integrator::new(
    adaptive_backward_problem(),
    opts3,
    7,
    5,
  )
  let dt3 = @ordinarydiffeq.determine_initdt(integ3, 5)
  assert_true((dt3 + 0.025123886079654534).abs() < 0.000001)
}

///|
test "adaptive Tsit5 auto-selects the step size and stays accurate" {
  let opts = tight_adaptive_opts(0.000001, 0.000000001)
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  // Global error tracks the tolerance: observed 1.4e-9 at reltol = 1e-6.
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - @math.exp(-1.0)).abs() < 0.00001)
  // The controller really stepped: 11 accepted steps here, i.e.
  // nf = 2 (initdt) + 1 (FSAL) + 6 per step = 69. Assert with headroom.
  assert_true(sol.t.length() >= 8)
  assert_true(sol.nf >= 50)
  // Time advances strictly.
  let mut i = 1
  while i < sol.t.length() {
    assert_true(sol.t.unsafe_get(i) > sol.t.unsafe_get(i - 1))
    i += 1
  }
}

///|
test "adaptive recovers from an over-large initial step" {
  // Seed dt = 1.0 (one step across the whole span): the first attempts are
  // rejected and the controller shrinks dt by the reject factor each time
  // until the embedded estimate fits the tolerance.
  let opts = @ordinarydiffeq.SolveOptions::with_dt(
    tight_adaptive_opts(0.0000000001, 0.000000000001),
    1.0,
  )
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - @math.exp(-1.0)).abs() < 0.000001)
  assert_true(sol.t.length() > 2)
  // Several rejections happened on the way (each costs a full 6-eval step).
  assert_true(sol.nf > 100)
}

///|
test "PI controller knobs default per the algorithm order" {
  // The stateful half of the controller (stepsize/accept/reject) is unit
  // tested inside control.mbt: it needs to write `est`/`dt`/`success_iter`,
  // and this MoonBit version exports struct fields as read-only to other
  // packages, so a black-box test cannot drive the controller's state.
  let prob = adaptive_decay_problem()
  let opts = @ordinarydiffeq.SolveOptions::with_dtmax(
    @ordinarydiffeq.SolveOptions::new(),
    1.0,
  )
  let integ = @ordinarydiffeq.Integrator::new(prob, opts, 7, 5)
  let c = integ.controller
  // Knob defaults for a 5th-order algorithm (alg_utils.jl:417-889).
  assert_true(c.beta1 == 0.14)
  assert_true(c.beta2 == 0.08)
  assert_true(c.qmin == 0.2)
  assert_true(c.qmax == 10.0)
  assert_true(c.qmax_first_step == 10000.0)
  assert_true(c.gamma == 0.9)
  assert_true(c.qsteady_min == 1.0 && c.qsteady_max == 1.0)
  assert_true(c.qoldinit == 0.0001)
  assert_true(c.errold == 0.0001)
  assert_true(c.q11 == 1.0)
  // A fresh integrator has EEst = 0, which the accept rule accepts.
  assert_true(@ordinarydiffeq.accept_step_controller(integ))
}

///|
test "adaptive with tstops hits the stop exactly" {
  let opts = @ordinarydiffeq.SolveOptions::with_tstops(
    tight_adaptive_opts(0.00000001, 0.0000000001),
    [0.5],
  )
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  // The step aimed at the tstop is snapped onto it exactly.
  let mut found = false
  for t in sol.t {
    if (t - 0.5).abs() < 0.000000001 {
      found = true
    }
  }
  assert_true(found)
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - @math.exp(-1.0)).abs() < 0.000001)
}

///|
test "adaptive with saveat interpolates at the requested times" {
  let opts = @ordinarydiffeq.SolveOptions::with_saveat(
    tight_adaptive_opts(0.00000001, 0.0000000001),
    [0.0, 0.3, 1.0],
  )
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  assert_eq(sol.t.length(), 3)
  assert_eq(sol.t.unsafe_get(0), 0.0)
  assert_eq(sol.t.unsafe_get(1), 0.3)
  assert_eq(sol.t.unsafe_get(2), 1.0)
  // A non-empty saveat turns dense storage off, but mid-step points are
  // still evaluated with the solver's polynomial (interp_at_saveat).
  assert_true(sol.k.length() == 0)
  let u03 = @ordinarydiffeq.Solution::call(sol, 0.3).unsafe_get(0)
  assert_true((u03 - @math.exp(-0.3)).abs() < 0.000000001)
}

///|
test "adaptive with dense output keeps the stage derivatives" {
  let opts = @ordinarydiffeq.SolveOptions::with_dense(
    tight_adaptive_opts(0.00000001, 0.0000000001),
    true,
  )
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  // One 7-stage group per saved time point, as in the fixed-step test.
  assert_true(sol.k.length() == sol.t.length())
  assert_eq(sol.k.unsafe_get(1).length(), 7)
  let mid = @ordinarydiffeq.Solution::call(sol, 0.3).unsafe_get(0)
  assert_true((mid - @math.exp(-0.3)).abs() < 0.000000001)
  // The derivative closure is exact for the polynomial: u' = -u.
  let d = @ordinarydiffeq.Solution::derivative(sol, 0.3).unsafe_get(0)
  assert_true((d + @math.exp(-0.3)).abs() < 0.000001)
}

///|
test "adaptive integrates backward in time" {
  let opts = tight_adaptive_opts(0.00000001, 0.0000000001)
  let sol = @ordinarydiffeq.solve(
    adaptive_backward_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  assert_true((last - @math.exp(1.0)).abs() < 0.000001)
  let mut i = 1
  while i < sol.t.length() {
    assert_true(sol.t.unsafe_get(i) < sol.t.unsafe_get(i - 1))
    i += 1
  }
}

///|
test "adaptive aborts with DtMinUnstable at dtmin" {
  // Tolerances so tight that no step can satisfy them: the reject path
  // shrinks dt until fix_dt_at_bounds pins it at dtmin, and check_error
  // aborts (DiffEqBase check_error.jl:112).
  let opts = @ordinarydiffeq.SolveOptions::with_dtmin(
    tight_adaptive_opts(0.00000000000001, 0.0000000000000001),
    0.05,
  )
  let opts2 = @ordinarydiffeq.SolveOptions::with_dt(opts, 0.5)
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts2,
  )
  assert_true(sol.retcode.is_dtmin_unstable())
  // Rejected steps save nothing, so only the initial point is stored.
  assert_eq(sol.t.length(), 1)
}

///|
test "a failed solve still runs the postamble" {
  // u' = u^2 on [0, 1] with u(0) = 1 blows up at t = 1: no finite step can
  // reach tf, so the solve aborts once the state overflows. Julia's
  // `de_check_error!` (check_error.jl:149) runs the postamble on failure, so
  // the abort point is saved when `save_end` asks for it — and in saveat
  // mode `save_end` is derived from tf's membership, hence 1.0 in the list.
  let prob = @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
    du.unsafe_set(0, u.unsafe_get(0) * u.unsafe_get(0))
  })
  let opts = @ordinarydiffeq.SolveOptions::with_saveat(
    tight_adaptive_opts(0.001, 0.000001),
    [0.3, 0.5, 1.0],
  )
  let sol = @ordinarydiffeq.solve(prob, @ordinarydiffeq.Tsit5::new(), opts)
  assert_true(sol.retcode.is_unstable())
  assert_eq(sol.t.length(), 3)
  assert_eq(sol.t.unsafe_get(0), 0.3)
  assert_eq(sol.t.unsafe_get(1), 0.5)
  // The third point is the abort time, not tf: the solve never got there.
  assert_true(sol.t.unsafe_get(2) > 0.5 && sol.t.unsafe_get(2) < 1.0)
  // The stored state is the blow-up one: u = 1/(1-t) has run away (the
  // controller collapses dt onto eps*|t| before u itself overflows, so the
  // value is finite but enormous).
  assert_true(sol.u.unsafe_get(2).unsafe_get(0) > 100.0)
}

///|
/// Regression guard for a *misdiagnosis*: the default `SolveOptions` has
/// `adaptive = false`, so a solve with no `dt` hits `solve`'s own guard and
/// returns an empty solution carrying `InitialFailure` — which is correct, not
/// a silent success. Asserting only `!is_success()` is not enough to tell the
/// two apart, because `InitialFailure`, `MaxIters`, `DtMinUnstable` and
/// `Unstable` all satisfy it; the code has to be pinned explicitly.
test "a fixed-step solve with no dt fails loudly instead of silently" {
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    @ordinarydiffeq.SolveOptions::new(),
  )
  assert_eq(sol.u.length(), 0)
  assert_eq(sol.nf, 0)
  assert_eq(sol.retcode.is_success(), false)
  // Rule out every other failure code, so this is pinned to `InitialFailure`
  // (an empty state with `Success` would be the actual bug).
  assert_eq(sol.retcode.is_maxiters(), false)
  assert_eq(sol.retcode.is_dtmin_unstable(), false)
  assert_eq(sol.retcode.is_unstable(), false)
  // The default options must really be non-adaptive for the above to be the
  // intended path — if that ever flips, this test is testing the wrong thing.
  assert_eq(@ordinarydiffeq.SolveOptions::new().adaptive, false)
}

///|
/// The companion: once `adaptive` is on and `dt` is left unset,
/// `handle_dt!` picks the initial step through the Hairer-Wanner heuristic
/// (solve.mbt:48) and the solve runs normally. The empty-solution symptom
/// above is often mistaken for a broken `determine_initdt`; this test is what
/// proves the heuristic is alive.
test "an adaptive solve with no dt picks its own initial step" {
  let opts = @ordinarydiffeq.SolveOptions::with_adaptive(
    @ordinarydiffeq.SolveOptions::new(),
    true,
  )
  let sol = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Tsit5::new(),
    opts,
  )
  assert_true(sol.retcode.is_success())
  // Not empty: the heuristic produced a usable first step.
  assert_true(sol.t.length() > 2)
  let last = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
  // Default reltol/abstol are 1e-3/1e-6, so 1e-5 is the right order of
  // accuracy for `u' = -u` over [0, 1].
  assert_true((last - @math.exp(-1.0)).abs() < 0.00001)
}

///|
/// Same auto-initial-step path for the stiff family, which reaches it through
/// `calc_rosenbrock_differentiation`'s own reuse policy rather than through an
/// FSAL first stage.
test "stiff solvers also pick their own initial step" {
  let opts = @ordinarydiffeq.SolveOptions::with_adaptive(
    @ordinarydiffeq.SolveOptions::new(),
    true,
  )
  let rodas = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Rodas5P::new(),
    opts,
  )
  assert_true(rodas.retcode.is_success())
  assert_true(rodas.t.length() > 2)
  let rb = @ordinarydiffeq.solve(
    adaptive_decay_problem(),
    @ordinarydiffeq.Rosenbrock23::new(),
    opts,
  )
  assert_true(rb.retcode.is_success())
  assert_true(rb.t.length() > 2)
}
