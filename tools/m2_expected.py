import math

EPS = 2.220446049250313e-16

def dmin(a, b): return a if a < b else b
def dmax(a, b): return a if a > b else b
def fastpower(x, k):
    if k == 0.0: return 1.0
    if x == 0.0: return 0.0 if k > 0.0 else math.inf
    return math.pow(x, k)
def sqrt(x): return math.sqrt(x)
def log10(x): return math.log(x) / math.log(10.0)

def rms_norm(u):
    n = len(u)
    if n == 0: return 0.0
    s = 0.0
    for x in u: s = s + x * x
    return sqrt(s / n)

# ---------------- PIController (control.mbt) ----------------
class Ctrl:
    def __init__(self, order):
        p = float(order)
        self.q11 = 1.0
        self.errold = 1e-4
        self.beta1 = 7.0 / (10.0 * p)
        self.beta2 = 2.0 / (5.0 * p)
        self.qoldinit = 1e-4
        self.qmin = 0.2
        self.qmax = 10.0
        self.qmax_first_step = 10000.0
        self.gamma = 0.9
        self.qsteady_min = 1.0
        self.qsteady_max = 1.0

def stepsize_controller(est, ctrl, success_iter):
    qmax_eff = ctrl.qmax_first_step if success_iter == 0 else ctrl.qmax
    if est == 0.0:
        return 1.0 / qmax_eff
    q11 = fastpower(est, ctrl.beta1)
    q = q11 / fastpower(ctrl.errold, ctrl.beta2)
    ctrl.q11 = q11
    return dmin(dmax(q / ctrl.gamma, 1.0 / qmax_eff), 1.0 / ctrl.qmin)

def step_accept_controller(est, dt, ctrl, q):
    qq = q
    if ctrl.qsteady_min <= qq and qq <= ctrl.qsteady_max:
        qq = 1.0
    ctrl.errold = dmax(est, ctrl.qoldinit)
    return dt / qq

def step_reject_controller(dt, ctrl):
    return dt / dmin(1.0 / ctrl.qmin, ctrl.q11 / ctrl.gamma)

def accept_step_controller(est):
    return est <= 1.0

def calc_dt_propose(t, tdir, dtnew, dtmax_opt, dtmin_opt):
    signed = tdir * abs(dtnew)
    rounded = (t + signed) - t
    dtpropose = tdir * dmin(abs(dtmax_opt), abs(rounded))
    return tdir * dmax(abs(dtpropose), dmax(dtmin_opt, EPS * abs(t)))

# ---------------- Tsit5 tableau ----------------
C = [0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0]
A = [
    None,
    [0.161],
    [-0.008480655492356989, 0.335480655492357],
    [2.8971530571054935, -6.359448489975075, 4.3622954328695815],
    [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525],
    [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383],
    [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774],
]
BT = [-0.001780011052225777, -0.0008164344596567469, 0.007880878010261995,
      -0.1447110071732629, 0.5823571654525552, -0.45808210592918697, 0.015151515151515152]
R = {
    'b1': [1.0, -2.763706197274826, 2.9132554618219126, -1.0530884977290216],
    'b2': [0.13169999999999998, -0.2234, 0.1017],
    'b3': [3.9302962368947516, -5.941033872131505, 2.490627285651253],
    'b4': [-12.411077166933676, 30.33818863028232, -16.548102889244902],
    'b5': [37.50931341651104, -88.1789048947664, 47.37952196281928],
    'b6': [-27.896526289197286, 65.09189467479366, -34.87065786149661],
    'b7': [1.5, -4.0, 2.5],
}

def tsit5_step(u, t, dt, k1):
    """One Tsit5 step; returns (u_new, ks[7], fsallast, est-raw utilde)."""
    ks = [None] * 7
    ks[0] = k1[:]
    def stage(idx, arow, cval):
        st = [u[i] + dt * sum(arow[j] * ks[j][i] for j in range(len(arow)))
              for i in range(len(u))]
        return st
    # stages 2..6 (ks indices 1..5); row for stage si+2 is A[si+1]
    for si in range(5):
        st = stage(si, A[si + 1], C[si])
        f = [st[0] * -1.0]  # u' = -u
        ks[si + 1] = f
    # solution: row 6 (a71..a76)
    unew = [u[i] + dt * sum(A[6][j] * ks[j][i] for j in range(6)) for i in range(len(u))]
    fsallast = [unew[0] * -1.0]
    ks[6] = fsallast[:]
    utilde = [dt * sum(BT[j] * ks[j][i] for j in range(7)) for i in range(len(u))]
    return unew, ks, fsallast, utilde

def weighted_residual(utilde, u0, u1, abstol, reltol):
    out = []
    for i in range(len(utilde)):
        amax = max(abs(u0[i]), abs(u1[i]))
        out.append(utilde[i] / (abstol + amax * reltol))
    return out

def tsit5_interp(theta, dt, y0, ks):
    th2 = theta * theta
    def poly2(x, c): return c[0] + x * (c[1] + x * c[2])
    def poly3(x, c): return c[0] + x * (c[1] + x * (c[2] + x * c[3]))
    b1 = theta * poly3(theta, R['b1'])
    b2 = th2 * poly2(theta, R['b2'])
    b3 = th2 * poly2(theta, R['b3'])
    b4 = th2 * poly2(theta, R['b4'])
    b5 = th2 * poly2(theta, R['b5'])
    b6 = th2 * poly2(theta, R['b6'])
    b7 = th2 * poly2(theta, R['b7'])
    bs = [b1, b2, b3, b4, b5, b6, b7]
    return [y0[i] + dt * sum(bs[j] * ks[j][i] for j in range(7)) for i in range(len(y0))]

# ---------------- determine_initdt (control.mbt) ----------------
def determine_initdt(t, u0, tdir, order, abstol, reltol, dtmax_opt, dtmin_opt, f):
    n = len(u0)
    dtmin = dmax(dtmin_opt, EPS * abs(t))
    smalldt = dmax(dtmin, 1e-6)
    if n == 0: return tdir * smalldt
    dtmax = abs(dtmax_opt)
    sk = [abstol + abs(u0[i]) * reltol for i in range(n)]
    f0 = f(t, u0)
    d0 = rms_norm([u0[i] / sk[i] for i in range(n)])
    d1 = rms_norm([f0[i] / sk[i] for i in range(n)])
    if d1 != d1: return tdir * dtmin
    dt0 = smalldt if (d0 < 1e-5 or d1 < 1e-5) else (d0 / d1) / 100.0
    dt0 = dmin(dt0, dtmax)
    tiny_first = dt0 < 10.0 * EPS
    dt0_tdir = tdir * dt0
    u1 = [u0[i] + dt0_tdir * f0[i] for i in range(n)]
    f1 = f(t + dt0_tdir, u1)
    if f0 == f1:
        return tdir * dmax(dtmin, 100.0 * dt0)
    d2 = rms_norm([(f1[i] - f0[i]) / sk[i] for i in range(n)]) / dt0
    maxd1d2 = dmax(d1, d2)
    if maxd1d2 <= 1e-15:
        dt1 = dmax(1e-6, dt0 * 0.001)
    else:
        dt1 = fastpower(10.0, -(2.0 + log10(maxd1d2)) / float(order))
    res = tdir * dmax(dtmin, dmin(dmin(100.0 * dt0, dt1), dtmax))
    if tiny_first and (not math.isfinite(res) or abs(res) < 10.0 * EPS):
        return tdir * dmax(smalldt, dtmin)
    return res

# ---------------- Full adaptive solve (solve.mbt + control.mbt) ----------------
def adaptive_solve(u0, t0, tf, dt_seed, reltol, abstol, dtmax_opt, dtmin_opt,
                   tstops=None, saveat=None, dense=False, order=5, verbose=False):
    tdir = 1.0 if tf >= t0 else -1.0
    ctrl = Ctrl(order)
    opts_dtmax = dtmax_opt if dtmax_opt != 0.0 else abs(tf - t0)
    t = t0
    dt = tdir * abs(dt_seed)
    dtpropose = dt
    u = u0[:]
    uprev = u0[:]
    nf = 0
    iter_ = 0
    accept_step = True
    est = 0.0
    success_iter = 0
    next_step_tstop = False
    tstop_target = 0.0
    sol_t, sol_u, sol_k = [], [], []
    save_everystep = True
    save_start = True
    save_end = True
    interp_poly = True  # sol.interp from opts.dense (default true)
    store_ks = dense
    f = lambda tt, uu: [uu[0] * -1.0]

    def eval_f(tt, uu):
        nonlocal nf
        nf += 1
        return f(tt, uu)

    # tstops queue: strictly-inside + tf, tdir-scaled ascending
    tstops_q = []
    if tstops:
        for ts in tstops:
            tt = tdir * ts
            if tdir * t0 < tt < tdir * tf:
                tstops_q.append(tt)
    tstops_q.append(tdir * tf)
    tstops_q.sort()
    tstop_idx = 0

    # saveat queue
    saveat_q = []
    if saveat is not None:
        save_everystep = False
        for ts in saveat:
            tt = tdir * ts
            if tdir * t0 < tt <= tdir * tf:
                saveat_q.append(tt)
        saveat_q.sort()
        save_start = any(tdir * t0 == x for x in saveat_q) or any(tdir * t0 == tdir * s for s in saveat)
        # resolve_options: contains_time on raw saveat vs t0
        save_start = any(t0 == s for s in saveat)
        save_end = any(tf == s for s in saveat)
        store_ks = False
    saveat_idx = 0

    # auto initdt
    if dt_seed == 0.0:
        dt = determine_initdt(t, u, tdir, order, abstol, reltol, opts_dtmax, dtmin_opt, f)
        dtpropose = dt
    # pre-start FSAL
    k1 = eval_f(t, u)
    if save_start:
        sol_t.append(t); sol_u.append(u[:])
        if store_ks: sol_k.append(k1[:])
    retcode = 'Success'

    def has_tstop(): return tstop_idx < len(tstops_q)
    def first_tstop(): return tstops_q[tstop_idx]

    while has_tstop():
        ft = first_tstop()
        while tdir * t < ft:
            # loopheader
            if iter_ > 0:
                if accept_step:
                    success_iter += 1
                    uprev = u[:]
                    dt = dtpropose
                # reject: handled in loopfooter of previous iteration
            iter_ += 1
            # fix_dt_at_bounds
            if tdir > 0:
                dt = dmin(opts_dtmax, dt); dt = dmax(dmax(dtmin_opt, EPS * abs(t)), dt)
            else:
                dt = dmax(-opts_dtmax, dt); dt = dmin(-dmax(dtmin_opt, EPS * abs(t)), dt)
            # modify_dt_for_tstops (adaptive branch)
            if has_tstop():
                original_dt = abs(dt)
                dtpropose = tdir * original_dt
                distance = abs(first_tstop() - tdir * t)
                tol = 100.0 * EPS * dmax(abs(t), abs(first_tstop()))
                if original_dt + tol < distance:
                    next_step_tstop = False
                else:
                    next_step_tstop = True
                    tstop_target = tdir * first_tstop()
                dt = tdir * dmin(original_dt, distance)
            # check_error
            if iter_ > 100000:
                retcode = 'MaxIters'; return dict(t=sol_t, u=sol_u, k=sol_k, retcode=retcode, nf=nf)
            next_tstop = tdir * first_tstop() if has_tstop() else tf
            if abs(dt) <= abs(dtmin_opt) and dtmin_opt > 0.0 and (not accept_step or t + dt < next_tstop):
                retcode = 'DtMinUnstable'
                return dict(t=sol_t, u=sol_u, k=sol_k, retcode=retcode, nf=nf)
            if not accept_step and abs(dt) <= EPS * abs(t):
                retcode = 'Unstable'
                return dict(t=sol_t, u=sol_u, k=sol_k, retcode=retcode, nf=nf)
            # perform_step: stages are built from `uprev` (the last accepted
            # state); the result lands in `u` and is only committed by
            # `apply_step` (uprev = u) on the next loopheader.
            unew, ks, fsallast, utilde = tsit5_step(uprev, t, dt, k1)
            atmp = weighted_residual(utilde, uprev, unew, abstol, reltol)
            est = rms_norm(atmp)
            u = unew
            should_exit = next_step_tstop
            # loopfooter
            ttmp = t + dt
            do_save = True
            q = stepsize_controller(est, ctrl, success_iter)
            accept_step = accept_step_controller(est)
            if accept_step:
                tprev = t
                if next_step_tstop:
                    dt = dtpropose
                t = tstop_target if next_step_tstop else ttmp
                next_step_tstop = False
                dtnew = step_accept_controller(est, dt, ctrl, q)
                dtpropose = calc_dt_propose(t, tdir, dtnew, opts_dtmax, dtmin_opt)
                k1 = fsallast[:]
            else:
                do_save = False
                dt = step_reject_controller(dt, ctrl)
            if do_save:
                # savevalues
                tdir_t = tdir * t
                while saveat_idx < len(saveat_q) and saveat_q[saveat_idx] <= tdir_t:
                    curt = saveat_q[saveat_idx]; saveat_idx += 1
                    if curt == tdir_t and curt == tdir * tf and not save_end:
                        continue
                    if curt != tdir_t:
                        theta = (tdir * curt - tdir * tprev) / dt
                        out = tsit5_interp(theta, dt, uprev, ks)
                        sol_t.append(tdir * curt); sol_u.append(out)
                    else:
                        sol_t.append(t); sol_u.append(u[:])
                        if store_ks: sol_k.append(ks)
                if save_everystep:
                    n = len(sol_t)
                    last_is_here = n > 0 and sol_t[n - 1] == t
                    end_guard = save_end or t != tf
                    if n == 0 or ((not last_is_here or dt == 0.0) and end_guard):
                        sol_t.append(t); sol_u.append(u[:])
                        if store_ks: sol_k.append(ks)
            if not has_tstop() or should_exit:
                break
        # handle_tstop
        if has_tstop():
            tdir_t = tdir * t
            if tdir_t == first_tstop():
                while tdir_t == first_tstop():
                    tstop_idx += 1
                    if not has_tstop(): break
            elif tdir_t > first_tstop():
                tstop_idx += 1
    # postamble
    if save_end:
        n = len(sol_t)
        if n == 0 or sol_t[n - 1] != t:
            sol_t.append(t); sol_u.append(u[:])
            if store_ks: sol_k.append(ks)
    return dict(t=sol_t, u=sol_u, k=sol_k, retcode=retcode, nf=nf)

# ============================================================
print("=== controller unit tests ===")
c = Ctrl(5)
print("beta1", repr(c.beta1), "beta2", repr(c.beta2))
# Case A
est, dt, errold, si = 4.0, 0.1, 1e-4, 0
c.q11, c.errold = 1.0, errold
q = stepsize_controller(est, c, si)
print("A q =", repr(q), " q11 =", repr(c.q11))
dt2 = step_reject_controller(dt, c)
print("A reject factor =", repr(dmin(1.0 / c.qmin, c.q11 / c.gamma)), " dt after =", repr(dt2))
# Case B (fresh)
c = Ctrl(5)
q2 = stepsize_controller(0.5, c, 1)
print("B q =", repr(q2))
dtnew = step_accept_controller(0.5, 0.1, c, q2)
print("B errold =", repr(c.errold), " dtnew =", repr(dtnew))
dtp = calc_dt_propose(0.0, 1.0, dtnew, 1.0, 0.0)
print("B dtpropose =", repr(dtp))
q3 = stepsize_controller(2.0, c, 1)
print("B reverse q =", repr(q3))
dtnew3 = step_accept_controller(2.0, 0.1, c, q3)
print("B reverse dtnew =", repr(dtnew3))
# Case C
c = Ctrl(5)
q4 = stepsize_controller(0.0, c, 1)
print("C q =", repr(q4), " q11 untouched =", repr(c.q11))
# Case D
print("D t=0.5 dtnew=0.2 ->", repr(calc_dt_propose(0.5, 1.0, 0.2, 1.0, 0.0)))
print("D t=0.5 dtnew=5.0 ->", repr(calc_dt_propose(0.5, 1.0, 5.0, 1.0, 0.0)))
# Case E
print("E t=1.0 tdir=-1 dtnew=0.2 ->", repr(calc_dt_propose(1.0, -1.0, 0.2, 1.0, 0.0)))

print()
print("=== determine_initdt ===")
f = lambda tt, uu: [uu[0] * -1.0]
r = determine_initdt(0.0, [1.0], 1.0, 5, 1e-6, 1e-3, 1.0, 0.0, f)
print("initdt default tol =", repr(r))
r2 = determine_initdt(0.0, [1.0], 1.0, 5, 1e-9, 1e-6, 1.0, 0.0, f)
print("initdt tight tol   =", repr(r2))
r3 = determine_initdt(0.0, [1.0], 1.0, 5, 1e-12, 1e-10, 1.0, 0.0, f)
print("initdt 1e-10 tol   =", repr(r3))
# backward initdt
r4 = determine_initdt(1.0, [math.e], -1.0, 5, 1e-9, 1e-6, 1.0, 0.0, f)
print("initdt backward    =", repr(r4))

print()
print("=== end-to-end adaptive solves ===")
for (rt, at, tag) in [(1e-6, 1e-9, "rtol=1e-6"), (1e-8, 1e-10, "rtol=1e-8"), (1e-10, 1e-12, "rtol=1e-10")]:
    res = adaptive_solve([1.0], 0.0, 1.0, 0.0, rt, at, 0.0, 0.0)
    err = abs(res['u'][-1][0] - math.exp(-1.0))
    print(f"{tag}: steps={len(res['t'])} nf={res['nf']} final_err={err:.3e} retcode={res['retcode']}")
    print("   ts:", [round(x, 6) for x in res['t']])

print()
print("=== saveat + adaptive (rtol=1e-8) ===")
res = adaptive_solve([1.0], 0.0, 1.0, 0.0, 1e-8, 1e-10, 0.0, 0.0, saveat=[0.0, 0.3, 1.0])
print("t:", res['t'], "k stored:", len(res['k']), "nf:", res['nf'])
print("u(0.3) err:", abs(res['u'][1][0] - math.exp(-0.3)))
print("u(1.0) err:", abs(res['u'][2][0] - math.exp(-1.0)))

print()
print("=== dense + adaptive (rtol=1e-8) ===")
res = adaptive_solve([1.0], 0.0, 1.0, 0.0, 1e-8, 1e-10, 0.0, 0.0, dense=True)
print("steps:", len(res['t']), "k groups:", len(res['k']), "nf:", res['nf'])
# find interval containing 0.3
i = 0
while i + 1 < len(res['t']) and res['t'][i + 1] <= 0.3: i += 1
dt = res['t'][i + 1] - res['t'][i]
theta = (0.3 - res['t'][i]) / dt
# NOTE: dense storage saves ks per step; our sim stores ks of the last step only
# for save_everystep endpoints, so mid-point evaluation needs per-step ks; approximate
# by re-deriving is skipped — instead evaluate at saved endpoints:
print("final err:", abs(res['u'][-1][0] - math.exp(-1.0)))

print()
print("=== tstops + adaptive (rtol=1e-8) ===")
res = adaptive_solve([1.0], 0.0, 1.0, 0.0, 1e-8, 1e-10, 0.0, 0.0, tstops=[0.5])
print("t:", res['t'])
print("hit 0.5 exactly:", 0.5 in res['t'])
print("final err:", abs(res['u'][-1][0] - math.exp(-1.0)), "nf:", res['nf'])

print()
print("=== backward adaptive (rtol=1e-8) ===")
res = adaptive_solve([math.e], 1.0, 0.0, 0.0, 1e-8, 1e-10, 0.0, 0.0)
print("steps:", len(res['t']), "nf:", res['nf'], "retcode:", res['retcode'])
print("t:", [round(x, 6) for x in res['t']])
print("u(0) err:", abs(res['u'][-1][0] - 1.0))

print()
print("=== over-large seed dt=1.0, rtol=1e-10 ===")
res = adaptive_solve([1.0], 0.0, 1.0, 1.0, 1e-10, 1e-12, 0.0, 0.0)
print("steps:", len(res['t']), "nf:", res['nf'], "retcode:", res['retcode'])
print("final err:", abs(res['u'][-1][0] - math.exp(-1.0)))

print()
print("=== dtmin abort (dt=0.5 seed, dtmin=0.05, rtol=1e-14) ===")
res = adaptive_solve([1.0], 0.0, 1.0, 0.5, 1e-14, 1e-16, 0.0, 0.05)
print("retcode:", res['retcode'], "t:", res['t'], "nf:", res['nf'])
