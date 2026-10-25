"""Independent reference for M3 acceptance.

Re-implements, in plain IEEE-754 Python, exactly the loop this MoonBit port
runs for a tableau-driven ExplicitRK method: the `perform_step` structure of
explicit_rk_perform_step.jl:20 (k1 reused from `fsalfirst`, last stage at
`c[end]`, extra `f(u, t+dt)` for a non-FSAL tableau, `alpha` accumulation,
difference-row error estimate) plus the M2 PI controller.

Nothing here imports or reads the MoonBit sources or the generated tableaus;
the tableau constants are transcribed from the generated
alg_explicit_rk_tableaus.mbt only so the two can be compared on identical
inputs. Every expected value printed below is produced by this file.
"""

import math

# --- tableaus (as generated) ------------------------------------------------

HEUN = dict(
    a=[[0.0, 0.0], [1.0, 0.0]],
    c=[0.0, 1.0],
    alpha=[0.5, 0.5],
    alpha_eest=[1.0, 0.0],
    stages=2, order=2, adaptive_order=1, fsal=False,
)
RK4 = dict(
    a=[[0.0] * 4, [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    c=[0.0, 0.5, 0.5, 1.0],
    alpha=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
    alpha_eest=[],
    stages=4, order=4, adaptive_order=0, fsal=False,
)
BS3 = dict(
    a=[[0.0] * 4, [0.5, 0.0, 0.0, 0.0], [0.0, 0.75, 0.0, 0.0],
       [2 / 9, 1 / 3, 4 / 9, 0.0]],
    c=[0.0, 0.5, 0.75, 1.0],
    alpha=[2 / 9, 1 / 3, 4 / 9, 0.0],
    alpha_eest=[7 / 24, 1 / 4, 1 / 3, 1 / 8],
    stages=4, order=3, adaptive_order=2, fsal=False,
)
RKF4 = dict(
    a=[[0.0] * 5, [0.25, 0.0, 0.0, 0.0, 0.0],
       [0.04938271604938271, 0.3950617283950617, 0.0, 0.0, 0.0],
       [0.5816326530612245, -1.2594752186588922, 1.5349854227405249, 0.0, 0.0],
       [1 / 6, 0.0, 0.5192307692307693, 0.3141025641025641, 0.0]],
    c=[0.0, 0.25, 4 / 9, 6 / 7, 1.0],
    alpha=[0.14930555555555555, 0.0, 0.5841346153846154,
           0.18322649572649571, 0.08333333333333333],
    alpha_eest=[1 / 6, 0.0, 0.5192307692307693, 0.3141025641025641, 0.0],
    stages=5, order=4, adaptive_order=3, fsal=False,
)
DP5 = dict(
    a=[[0.0] * 7,
       [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.075, 0.225, 0.0, 0.0, 0.0, 0.0, 0.0],
       [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0, 0.0, 0.0],
       [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0, 0.0, 0.0],
       [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0.0, 0.0],
       [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]],
    c=[0.0, 0.2, 0.3, 0.8, 8 / 9, 1.0, 1.0],
    alpha=[35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0],
    alpha_eest=[0.08991319444444444, 0.0, 0.4534890685834082, 0.6140625,
                -0.2715123820754717, 0.08904761904761904, 0.025],
    stages=7, order=5, adaptive_order=4, fsal=True,
)

# --- PI controller (control.mbt, from PIController defaults) ----------------


class PI:
    def __init__(self, order):
        p = float(order)
        self.q11 = 1.0
        self.errold = 1e-4
        self.beta1 = 7.0 / (10.0 * p)
        self.beta2 = 2.0 / (5.0 * p)
        self.qmin = 0.2
        self.qmax = 10.0
        self.qmax_first_step = 1e4
        self.gamma = 0.9
        self.qoldinit = 1e-4

    def stepsize(self, est, success_iter):
        qmax_eff = self.qmax_first_step if success_iter == 0 else self.qmax
        if est == 0.0:
            return 1.0 / qmax_eff
        q11 = est ** self.beta1
        q = q11 / (self.errold ** self.beta2)
        self.q11 = q11
        return min(max(q / self.gamma, 1.0 / qmax_eff), 1.0 / self.qmin)

    def accept(self, q, est, dt):
        # qsteady_min == qsteady_max == 1, so only q == 1 is held.
        qq = 1.0 if 1.0 <= q <= 1.0 else q
        self.errold = max(est, self.qoldinit)
        return dt / qq


def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0


# --- the loop ---------------------------------------------------------------


def solve(tab, f, u0, t0, tf, dt0, adaptive, abstol, reltol, dtmax=None,
          maxiters=100000):
    ctrl = PI(tab["order"])
    dtmax = abs(tf - t0) if dtmax is None else dtmax
    t, u, dt = t0, list(u0), dt0
    dtpropose = dt
    fsalfirst = f(t, u)
    fsallast = list(fsalfirst)
    nf = 1
    success_iter = 0
    ts = [t0]
    us = [list(u0)]
    it = 0
    while t < tf and it <= maxiters:
        it += 1
        dt = min(dt, dtmax)
        dtpropose = dt
        if abs(dtpropose) + 1e-14 < abs(tf - t):
            next_tstop = False
            dt = dt
        else:
            next_tstop = True
            dt = tf - t
        # perform_step
        stages = tab["stages"]
        kk = [None] * stages
        kk[0] = list(fsalfirst)
        for i in range(1, stages - 1):
            s = list(u)
            for j in range(i):
                for r in range(len(s)):
                    s[r] += dt * tab["a"][i][j] * kk[j][r]
            kk[i] = f(t + tab["c"][i] * dt, s)
            nf += 1
        last = stages - 1
        s = list(u)
        for j in range(last):
            for r in range(len(s)):
                s[r] += dt * tab["a"][last][j] * kk[j][r]
        kk[last] = f(t + tab["c"][last] * dt, s)
        nf += 1
        fsallast = list(kk[last])
        unew = list(u)
        for r in range(len(u)):
            acc = u[r]
            for j in range(stages):
                acc += dt * tab["alpha"][j] * kk[j][r]
            unew[r] = acc
        if not tab["fsal"]:
            fsallast = f(t + dt, unew)
            nf += 1
        est = 0.0
        if adaptive and tab["adaptive_order"] > 0:
            utilde = []
            for r in range(len(u)):
                acc = 0.0
                for j in range(stages):
                    acc += dt * (tab["alpha"][j] - tab["alpha_eest"][j]) * kk[j][r]
                utilde.append(acc)
            scale = [abstol + max(abs(u[r]), abs(unew[r])) * reltol
                     for r in range(len(u))]
            est = rms([utilde[r] / scale[r] for r in range(len(u))])
        accepted = (not adaptive) or est <= 1.0
        if accepted:
            success_iter += 1
            tprev = t
            u = unew
            if next_tstop:
                dt = dtpropose
            t = tf if next_tstop else t + dt
            fsalfirst = list(fsallast)
            if adaptive:
                # `loopfooter`'s adaptive branch runs the controller; the
                # fixed-step branch just carries dt over untouched.
                # `loopfooter` runs *after* `loopheader` already bumped
                # `success_iter`, so the controller sees the post-accept count.
                q = ctrl.stepsize(est, success_iter - 1)
                dtnew = ctrl.accept(q, est, dt)
                rounded = t + abs(dtnew) - t
                dtpropose = min(dtmax, abs(rounded))
                dt = dtpropose
            ts.append(t)
            us.append(list(u))
        else:
            dt = dt / min(1.0 / ctrl.qmin, ctrl.q11 / ctrl.gamma)
    return ts, us, nf


def decay(t, u):
    return [-u[0]]


def oscillator(t, u):
    # u = (v, v'), v'' = -v  ->  (v', v'') = (u[1], -u[0])
    return [u[1], -u[0]]


def fixed_step_error(tab, f, u0, t0, tf, dt, exact):
    ts, us, nf = solve(tab, f, u0, t0, tf, dt, False, 0.0, 0.0)
    return abs(us[-1][0] - exact), nf, len(ts)


def main():
    print("== fixed-step endpoint errors (abs) ==")
    exact_decay = math.exp(-1.0)
    exact_osc_t = 0.7
    exact_osc = (math.cos(exact_osc_t), -math.sin(exact_osc_t))
    for name, tab, f, dt, exact in [
        ("heun   @0.01", HEUN, decay, 0.01, exact_decay),
        ("bs3    @0.01", BS3, decay, 0.01, exact_decay),
        ("rkf4   @0.05", RKF4, decay, 0.05, exact_decay),
        ("dp5    @0.20", DP5, decay, 0.20, exact_decay),
        ("rk4    @0.05 (osc)", RK4, oscillator, 0.05, exact_osc[0]),
    ]:
        err, nf, n = fixed_step_error(tab, f, [1.0] if f is decay else [0.0, 1.0],
                                      0.0, 1.0, dt, exact)
        print(f"{name:22s} err={err:.6e}  nf={nf}  points={n}")

    print()
    print("== adaptive: DormandPrince, u'=-u, [0,1], dt0=0.1, tol=1e-9 ==")
    ts, us, nf = solve(DP5, decay, [1.0], 0.0, 1.0, 0.1, True, 1e-9, 1e-9)
    print(f"points={len(ts)}  nf={nf}  final_err={abs(us[-1][0] - exact_decay):.6e}")
    print(f"ts_sum={sum(ts):.17g}")
    print(f"ts={[round(x, 6) for x in ts]}")

    print()
    print("== adaptive: RK4 (no alphaEEst, est stays 0), dt0=0.1 ==")
    ts, us, nf = solve(RK4, decay, [1.0], 0.0, 1.0, 0.1, True, 1e-6, 1e-6)
    print(f"points={len(ts)}  nf={nf}  final_err={abs(us[-1][0] - exact_decay):.6e}")
    print(f"ts_sum={sum(ts):.17g}")
    print(f"ts={[round(x, 6) for x in ts]}")


if __name__ == "__main__":
    main()
