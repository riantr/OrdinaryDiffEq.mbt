#!/usr/bin/env python3
"""Independent reference for the Rodas5P port in OrdinaryDiffEq.mbt.

Re-derives the method from the upstream tableau (rosenbrock_tableaus.jl:23-72)
and the generic Rosenbrock step (rosenbrock_perform_step.jl:431-559) in plain
NumPy-free Python, then prints the quantities the MoonBit tests assert on.

Nothing here imports or reads the MoonBit implementation: the coefficients are
re-typed from the Julia source and the step loop is written from the upstream
pseudocode, so agreement between the two is real evidence rather than a
tautology.

Usage:  python tools/m4_expected.py
"""

import math

# --- RODAS5P tableau, transcribed from rosenbrock_tableaus.jl:23-55 ----------

A = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.849394379747939, 0.45842242204463923, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-6.954028509809101, 2.489845061869568, -10.358996098473584, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.04721187230404641,
     0.0, 0.0, 0.0, 0.0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256,
     0.030198172008377946, 0.0, 0.0, 0.0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256,
     0.030198172008377946, 1.0, 0.0, 0.0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256,
     0.030198172008377946, 1.0, 1.0, 0.0],
]

C = [
    [0.0] * 7,
    [-14.155112264123755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-17.97296035885952, -2.859693295451294, 0.0, 0.0, 0.0, 0.0, 0.0],
    [147.12150275711716, -1.41221402718213, 71.68940251302358, 0.0, 0.0, 0.0, 0.0],
    [165.43517024871676, -0.4592823456491126, 42.90938336958603, -5.961986721573306,
     0.0, 0.0, 0.0],
    [24.854864614690072, -3.0009227002832186, 47.4931110020768, 5.5814197821558125,
     -0.6610691825249471, 0.0, 0.0],
    [30.91273214028599, -3.1208243349937974, 77.79954646070892, 34.28646028294783,
     -19.097331116725623, -28.087943162872662, 0.0],
    [37.80277123390563, -3.2571969029072276, 112.26918849496327, 66.9347231244047,
     -40.06618937091002, -54.66780262877968, -9.48861652309627],
]

C_NODE = [0.0, 0.6358126895828704, 0.4095798393397535, 0.9769306725060716,
          0.4288403609558664, 1.0, 1.0, 1.0]

D = [0.21193756319429014, -0.42387512638858027, -0.3384627126235924,
     1.8046452872882734, 2.325825639765069, 0.0, 0.0, 0.0]

H = [
    [25.948786856663858, -2.5579724845846235, 10.433815404888879, -2.3679251022685204,
     0.524948541321073, 1.1241088310450404, 0.4272876194431874, -0.17202221070155493],
    [-9.91568850695171, -0.9689944594115154, 3.0438037242978453, -24.495224566215796,
     20.176138334709044, 15.98066361424651, -6.789040303419874, -6.710236069923372],
    [11.419903575922262, 2.8879645146136994, 72.92137995996029, 80.12511834622643,
     -52.072871366152654, -59.78993625266729, -0.15582684282751913, 4.883087185713722],
]

GAMMA = 0.21193756319429014
NS = 8
# b = RODAS5PA[8, 1:8] with a 1 appended (rosenbrock_tableaus.jl:67-68)
B = A[7][:7] + [1.0]
# btilde = zeros(8) with the last entry set to 1 (rosenbrock_tableaus.jl:69-70)
BTILDE = [0.0] * 7 + [1.0]


# --- tiny dense linear algebra (the port has its own LU in linalg_lu.mbt) --

def lu_factor(a, n):
    """In-place LU with partial pivoting. Returns (ok, ipiv)."""
    ipiv = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(a[i][k]))
        if abs(a[p][k]) < 1e-300:
            return False, ipiv
        if p != k:
            a[k], a[p] = a[p], a[k]
            ipiv[k], ipiv[p] = ipiv[p], ipiv[k]
        akk = a[k][k]
        for i in range(k + 1, n):
            m = a[i][k] / akk
            a[i][k] = m
            if m != 0.0:
                for j in range(k + 1, n):
                    a[i][j] -= m * a[k][j]
    return True, ipiv


def lu_solve(a, ipiv, b):
    n = len(b)
    x = list(b)
    for k in range(n):
        if ipiv[k] != k:
            x[k], x[ipiv[k]] = x[ipiv[k]], x[k]
    for i in range(n):
        xi = x[i]
        for j in range(i):
            xi -= a[i][j] * x[j]
        x[i] = xi
    for i in range(n - 1, -1, -1):
        xi = x[i]
        for j in range(i + 1, n):
            xi -= a[i][j] * x[j]
        x[i] = xi / a[i][i]
    return x


# --- forward-difference Jacobian, matching jacobian_fd's step ---------------

SQRT_EPS = 1.4901161193847656e-8


def fd_step(x):
    return max(1.0, abs(x)) * SQRT_EPS


def jacobian_fd(f, t, u, du_base):
    n = len(u)
    jac = [[0.0] * n for _ in range(n)]
    for j in range(n):
        up = list(u)
        h = fd_step(u[j])
        up[j] = u[j] + h
        dup = f(t, up)
        inv_h = 1.0 / h
        for i in range(n):
            jac[i][j] = (dup[i] - du_base[i]) * inv_h
    return jac


def fd_step_t(t, dt):
    scale = 1.0 if dt == 0.0 else abs(dt)
    return max(max(1.0, abs(t)), scale) * SQRT_EPS


def tderivative_fd(f, t, u, du_base, dt):
    n = len(u)
    h = fd_step_t(t, dt)
    dup = f(t + h, u)
    inv_h = 1.0 / h
    return [(dup[i] - du_base[i]) * inv_h for i in range(n)]


def jacobian2w(jac, n, dtgamma):
    """W = J - I/dtgamma (derivative_utils.jl:1014, jacobian2W!)."""
    inv = 1.0 / dtgamma
    return [[jac[i][j] - (inv if i == j else 0.0) for j in range(n)] for i in range(n)]


# --- the Rodas5P step ------------------------------------------------------

def rodas5p_step(f, t, uprev, dt, jac_cache):
    """One fixed-step Rodas5P step. Returns (u_new, ks, dense_ks).

    `jac_cache` is a dict holding `w`, `ipiv`, `jac`, `dT` so a fixed-step run
    can be compared with and without reuse. `refresh` forces recomputation.
    Mirrors rosenbrock_perform_step.jl:431-559 with the identity mass matrix.
    """
    n = len(uprev)
    dtgamma = dt * GAMMA

    du_base = f(t, uprev)                       # stage 1 `du = f(uprev, t)`
    dT = tderivative_fd(f, t, uprev, du_base, dt)

    # W is rebuilt from J on every step; only J itself is subject to reuse.
    if jac_cache.get("jac") is None:
        jac_cache["jac"] = jacobian_fd(f, t, uprev, du_base)
    w = jacobian2w(jac_cache["jac"], n, dtgamma)
    ok, ipiv = lu_factor(w, n)
    if not ok:
        raise RuntimeError("singular W")
    jac_cache["ipiv"] = ipiv

    ks = [None] * NS

    # k1 = W \ -(du + dt*d[0]*dT).  Upstream's minus sign survives: this port's
    # W is J - I/dtgamma (the calc_W convention), so w_solve is a plain solve.
    rhs = [-(du_base[i] + dt * D[0] * dT[i]) for i in range(n)]
    ks[0] = lu_solve(w, ipiv, rhs)

    for s in range(1, NS):
        u_stage = [uprev[i] + sum(A[s][j] * ks[j][i] for j in range(s))
                   for i in range(n)]
        du = f(t + C_NODE[s] * dt, u_stage)
        inv_dt = 1.0 / dt
        rhs = [-(du[i] + dt * D[s] * dT[i]
                 + sum(C[s][j] * inv_dt * ks[j][i] for j in range(s)))
               for i in range(n)]
        ks[s] = lu_solve(w, ipiv, rhs)

    u_new = [uprev[i] + sum(B[j] * ks[j][i] for j in range(NS)) for i in range(n)]

    # Embedded difference and the H-combined dense stages.
    utilde = [sum(BTILDE[j] * ks[j][i] for j in range(NS)) for i in range(n)]
    dense = [[sum(H[r][j] * ks[j][i] for j in range(NS)) for i in range(n)]
             for r in range(len(H))]
    return u_new, utilde, dense


def solve_fixed(f, t0, tf, u0, dt, refresh_always=True):
    """Fixed-step solve. Returns (t_list, u_list, step_count, jac_count)."""
    n = len(u0)
    steps = int(round((tf - t0) / dt))
    cache = {"jac": None}
    t = t0
    u = list(u0)
    jac_count = 0
    for _ in range(steps):
        if refresh_always:
            cache["jac"] = None
        before = cache["jac"] is None
        u, _, _ = rodas5p_step(f, t, u, dt, cache)
        if before:
            jac_count += 1
        t += dt
    return t, u, steps, jac_count


# --- test problems ---------------------------------------------------------

def decay(t, u):
    return [-u[0]]


def stiff_chain(t, u):
    # A = [[-0.1, 0], [-99, -0.1]]; exact solution exp(-0.1 t) * (1, -99 t)
    return [-0.1 * u[0], -99.0 * u[0] - 0.1 * u[1]]


def stiff_chain_exact(t):
    e = math.exp(-0.1 * t)
    return (e, e * (-99.0 * t))


def vanishing(t, u):
    # u' = -1000 u
    return [-1000.0 * u[0]]


# --- reports ---------------------------------------------------------------

def report_linear_decay():
    print("== u' = -u on [0, 1], u(0) = 1, exact exp(-1) = %.17g" % math.exp(-1.0))
    prev = None
    for dt in (0.2, 0.1, 0.05, 0.025):
        _, u, steps, njac = solve_fixed(decay, 0.0, 1.0, [1.0], dt)
        err = abs(u[0] - math.exp(-1.0))
        ratio = "" if prev is None else "  ratio vs coarser = %.4f" % (prev / err)
        print("   dt=%-6g steps=%-4d jac=%-4d |err|=%.6e%s" % (dt, steps, njac, err, ratio))
        prev = err


def report_order():
    print("\n== convergence order on u' = -u (expect ~32 for a 5th-order method)")
    errs = []
    for k in range(5):
        dt = 0.4 / (2 ** k)
        _, u, _, _ = solve_fixed(decay, 0.0, 1.0, [1.0], dt)
        errs.append(abs(u[0] - math.exp(-1.0)))
    for k in range(1, len(errs)):
        print("   dt=0.4/2^%d -> dt=0.4/2^%d : ratio %.4f  (order %.3f)"
              % (k - 1, k, errs[k - 1] / errs[k],
                 math.log(errs[k - 1] / errs[k]) / math.log(2.0)))


def report_stiff_chain():
    print("\n== stiff Jordan chain, exact(1) = (%.17g, %.17g)" % stiff_chain_exact(1.0))
    for dt in (0.05, 0.02, 0.01):
        _, u, steps, _ = solve_fixed(stiff_chain, 0.0, 1.0, [1.0, 0.0], dt)
        e0, e1 = stiff_chain_exact(1.0)
        print("   dt=%-6g steps=%-4d err=(%.6e, %.6e)"
              % (dt, steps, abs(u[0] - e0), abs(u[1] - e1)))


def report_stability():
    print("\n== u' = -1000 u on [0, 1]: exact value is exp(-1000) = 0")
    for dt in (0.5, 0.1, 0.01):
        _, u, steps, _ = solve_fixed(vanishing, 0.0, 1.0, [1.0], dt)
        print("   dt=%-6g steps=%-4d u(tf)=%.6e" % (dt, steps, u[0]))


def report_jac_reuse():
    print("\n== Jacobian reuse over a constant-dt run (fixed-step always refreshes)")
    _, u_ref, _, njac_ref = solve_fixed(decay, 0.0, 1.0, [1.0], 0.05,
                                       refresh_always=True)
    _, u_reuse, _, njac_reuse = solve_fixed(decay, 0.0, 1.0, [1.0], 0.05,
                                           refresh_always=False)
    print("   refresh every step: %d J evaluations, u(tf)=%.17g" % (njac_ref, u_ref[0]))
    print("   reuse J entirely : %d J evaluations, u(tf)=%.17g" % (njac_reuse, u_reuse[0]))
    print("   (the difference is the W-method's tolerance to a stale J)")


if __name__ == "__main__":
    report_linear_decay()
    report_order()
    report_stiff_chain()
    report_stability()
    report_jac_reuse()
