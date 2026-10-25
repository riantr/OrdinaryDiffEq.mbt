"""Independent Python reference for the FBDF multistep BDF step.

Written against OrdinaryDiffEqBDF/src (bdf_utils.jl, bdf_perform_step.jl,
controllers.jl) to cross-check the MoonBit port in src/src/alg_fbdf.mbt.
Pure Python, no numpy, matching the port's own arithmetic.
"""

import math

# ---------------------------------------------------------------------------
# Fixed BDF coefficients (`_make_bdf_coeffs_fbdf`, bdf_utils.jl:76)
# ---------------------------------------------------------------------------
# bdf_coeffs[k] is [a0, a1, ..., ak] with
#   (1/dt) * sum_i a_i * y(t_i) = f(t_{k+1}),   t_i = t + (1-i) dt
# so a0 is the leading coefficient and a0*(k+1) the local error constant.
BDF_COEFFS = [
    None,                                    # row 0 unused
    [1.0, -1.0],                             # k=1  implicit Euler
    [1.5, -2.0, 0.5],                        # k=2  BDF2
    [11.0 / 6.0, -3.0, 1.5, -1.0 / 3.0],     # k=3  BDF3
    [25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 0.25],   # k=4  BDF4
    [137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 1.25, -0.2],  # k=5  BDF5
]


def bdf_leading(k):
    return BDF_COEFFS[k][0]


# ---------------------------------------------------------------------------
# Fornberg finite-difference weights (`calc_finite_difference_weights!`,
# bdf_utils.jl:161).  Direct 1-based -> 0-based transcription.
# ---------------------------------------------------------------------------
def calc_fd_weights(ts, t, order):
    """c[i][k] is the weight of ts[i] in the (k+1)-th derivative at t.

    Direct 1-based -> 0-based transcription of `calc_finite_difference_weights!`
    (bdf_utils.jl:161).  Julia's `for k in i:-1:2` (k 1-based) maps to a
    0-based *column* index `kk = k-1` running `my_i .. 1`, i.e.
    `range(i, 0, -1)` — note the upper bound is `i`, not `i-1`.
    """
    n = order + 1
    c = [[0.0] * n for _ in range(n)]
    c[0][0] = 1.0
    c1 = 1.0
    c4 = ts[0] - t
    for i in range(1, n):                    # Julia i = 2..order+1
        c2 = 1.0
        c5 = c4
        c4 = ts[i] - t
        for j in range(0, i):                # Julia j = 1..i-1
            c3 = ts[i] - ts[j]
            c2 *= c3
            if j == i - 1:
                for kk in range(i, 0, -1):   # Julia k = i..2 -> col i..1
                    c[i][kk] = c1 * (kk * c[i - 1][kk - 1] - c5 * c[i - 1][kk]) / c2
                c[i][0] = 0.0
            for kk in range(i, 0, -1):
                c[j][kk] = (c4 * c[j][kk] - kk * c[j][kk - 1]) / c3
            c[j][0] = 0.0
        c1 = c2
    return c


def estimate_terk(ts, t, dt, k, u, u_history):
    """|dt^(k-1)| * d^{k-1}y/dt^{k-1} at t+dt  (bdf_perform_step.jl:403)."""
    c = calc_fd_weights(ts, t + dt, k - 1)
    terk = [c[0][k - 1] * u[i] for i in range(len(u))]
    for i in range(1, k):
        for r in range(len(u)):
            terk[r] += c[i][k - 1] * u_history[i - 1][r]
    scale = abs(dt ** (k - 1))
    return [v * scale for v in terk]


# ---------------------------------------------------------------------------
# Chebyshev reference nodes / barycentric weights (bdf_utils.jl:280)
# ---------------------------------------------------------------------------
def cheb_nodes(n):
    return [(1.0 + math.cos(math.pi * (2 * j - 1) / (2 * n))) / 2.0
            for j in range(1, n + 1)]


def bary_weights(n):
    return [((-1.0) ** (j - 1)) * math.sin(math.pi * (2 * j - 1) / (2 * n))
            for j in range(1, n + 1)]


def lagrange_basis_scalar(xi, j, thetas, n):
    tj = thetas[j]
    L = 1.0
    for m in range(n):
        if m == j:
            continue
        L *= (xi - thetas[m]) / (tj - thetas[m])
    return L


def eval_lagrange(xi, thetas, values, n):
    out = [0.0] * len(values[0])
    for j in range(n):
        Lj = lagrange_basis_scalar(xi, j, thetas, n)
        for i in range(len(out)):
            out[i] += Lj * values[j][i]
    return out


def lagrange_basis_deriv(theta, i, n):
    nodes = cheb_nodes(n)
    result = 0.0
    for m in range(n):
        if m == i:
            continue
        term = 1.0 / (nodes[i] - nodes[m])
        for l in range(n):
            if l == i or l == m:
                continue
            term *= (theta - nodes[l]) / (nodes[i] - nodes[l])
        result += term
    return result


def resample_at_chebyshev(k, values, thetas, n):
    """k[i] = p(xi_i) for the fixed nodes xi (bdf_utils.jl:331)."""
    nodes = cheb_nodes(n)
    for i in range(n):
        xi = nodes[i]
        k[i] = eval_lagrange(xi, thetas, values, n)


# ---------------------------------------------------------------------------
# Tiny dense linear algebra (mirrors linalg_lu.mbt)
# ---------------------------------------------------------------------------
def lu_factor(a, n):
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


# ---------------------------------------------------------------------------
# Newton solve of the corrector, COEFFICIENT_MULTISTEP form
# (`nlsolve!` + `compute_step!`, OrdinaryDiffEqNonlinearSolve)
# ---------------------------------------------------------------------------
KAPPA = 0.1          # NLNewton default
MAX_NEWTON_ITERS = 20
FAILFACTOR = 0.5     # get_failfactor(integrator) for a Newton method


def newton_solve(f, t, dt, u0_pred, tmp, w, ipiv, n, kappa=KAPPA,
                 maxiters=MAX_NEWTON_ITERS):
    """Solve `z = tmp + f(z, t+dt) - (alpha * inv_gamma_dt) * z` for z.

    Mirrors `nlsolve!` (nlsolve.jl:105) with `method = COEFFICIENT_MULTISTEP`:
    the residual's Jacobian is `-W` with `W = J - I/(dt*gamma)`, so one LU
    factorization serves the whole iteration.

    Returns (z, niter, nf, converged).
    """
    alpha = 1.0                       # nlsolver.alpha
    beta0 = 1.0 / bdf_leading(1)      # placeholder; real beta0 passed via tmp
    # `nlsolver.gamma * dt / alpha` is the gamma*dt W is assembled at; for the
    # BDF corrector beta0 = 1/a0, so gamma*dt = beta0*dt and inv_gamma_dt =
    # a0/dt.  The residual below is written with that scaling.
    inv_gamma_dt = 1.0 / (beta0 * dt)
    z = list(u0_pred)
    ztmp = [0.0] * n
    ndz_prev = None
    eta_old = 1e-16
    nf = 0
    niter = 0
    for it in range(1, maxiters + 1):
        niter = it
        # ustep = z (COEFFICIENT_MULTISTEP: compute_ustep returns z)
        fz = f(t + dt, z)
        nf += 1
        for i in range(n):
            ztmp[i] = tmp[i] + fz[i] - (alpha * inv_gamma_dt) * z[i]
        # Newton step: solve W dz = ztmp
        dz = lu_solve(w, ipiv, ztmp)
        for i in range(n):
            z[i] -= dz[i]
        # residual norm: the scaled norm of the *displacement*
        ndz = max(abs(v) for v in dz)
        if it > 1:
            theta = ndz / ndz_prev
            if abs(theta - 1.0) <= 1e-15:
                if ndz <= 1.0:
                    return z, niter, nf, True
                return z, niter, nf, False
            if theta > 2.0:
                return z, niter, nf, False
        else:
            if ndz < 1e-5:
                return z, niter, nf, True
        ndz_prev = ndz
        eta = theta if it > 1 else eta_old
        eta = eta / (1.0 - eta)
        if eta * ndz < kappa:
            return z, niter, nf, True
    return z, niter, nf, False


# ---------------------------------------------------------------------------
# The FBDF step (`perform_step!` for FBDFConstantCache, bdf_perform_step.jl:1245)
# ---------------------------------------------------------------------------
def fbdf_step(f, t, uprev, dt, cache, jac_cache, adaptive=True):
    """One FBDF step.

    `cache` holds the mutable method state: ts, u_history, order, consfailcnt,
    nconsteps, iters_from_event, prevorder, qwait, bdf_coeffs (fixed table),
    terkm2/terkm1/terk/terkp1 (order-selection estimates).
    `jac_cache` holds J and the factored W.
    Returns (u_new, fsallast, k_list).
    """
    n = len(uprev)
    k = cache["order"]
    tdt = t + dt
    ts = cache["ts"]
    hist = cache["u_history"]
    coeffs = BDF_COEFFS[k]

    # --- predictor: Lagrange interpolant through the actual history at theta=1
    if cache["iters_from_event"] >= 1:
        n_pred = k + 1
        pred_thetas = [(ts[j] - t) / dt for j in range(n_pred)]
        u0 = eval_lagrange(1.0, pred_thetas, hist[:n_pred], n_pred)
    else:
        u0 = list(uprev)

    # --- corrector: Lagrange interpolant at the equidistant past points
    u_corrector = [[0.0] * n for _ in range(k)]
    if cache["iters_from_event"] >= 1:
        for i in range(k - 1):
            u_corrector[i] = eval_lagrange(
                -float(i + 1), pred_thetas, hist[:n_pred], n_pred)

    # tmp = (sum a_i * y_i) / dt  with a_0 * u_new folded into the solve
    tmp = [-uprev[i] * coeffs[1] for i in range(n)]
    for i in range(k - 1):
        for r in range(n):
            tmp[r] -= u_corrector[i][r] * coeffs[i + 2]

    # --- Jacobian / W
    du_base = f(t, uprev)
    if jac_cache.get("J") is None:
        jac_cache["J"] = jacobian_fd(f, t, uprev, du_base)
    beta0 = 1.0 / coeffs[0]
    w = jacobian2w(jac_cache["J"], n, beta0 * dt)
    ok, ipiv = lu_factor(w, n)
    if not ok:
        return None, None, None
    jac_cache["ipiv"] = ipiv
    jac_cache["w"] = w

    # --- Newton
    z, niter, nf_newton, converged = newton_solve(
        f, t, dt, u0, tmp, w, ipiv, n)
    if not converged:
        return None, None, None
    u = z

    # --- local truncation error estimate (bdf_perform_step.jl:1320-1336)
    r = [0.0] * (k + 2)
    for j in range(2, k + 1):
        rj = 1.0 - j
        for i in range(2, k + 2):
            rj *= ((tdt - j * dt) - ts[i - 1]) / (i * dt)
        r[j - 1] = rj
    terkp1 = [u[i] - u0[i] for i in range(n)]
    for j in range(1, k + 2):
        for i in range(n):
            terkp1[i] *= j * dt / (tdt - ts[j - 1])
    lte = -1.0 / (1 + k)
    for j in range(2, k + 1):
        lte -= coeffs[j - 1] * r[j - 1]
    lte = [v * lte for v in terkp1]

    # --- terk family for order selection
    ts_tmp = [tdt] + list(ts[:k])
    terk = estimate_terk(ts_tmp, t, dt, k + 1, u, hist)
    terkm1 = estimate_terk(ts_tmp, t, dt, k, u, hist) if k > 1 else None
    terkm2 = estimate_terk(ts_tmp, t, dt, k - 1, u, hist) if k > 2 else None

    # --- store accepted-step bookkeeping
    cache["dtprev"] = dt
    cache["prevorder"] = k
    fsallast = f(tdt, u)

    # --- k for dense output: resample at the fixed Chebyshev nodes
    dense_order = k
    n_nodes = min(dense_order + 1, 6)
    calck_thetas = [1.0] + [(ts[j] - t) / dt
                            for j in range(min(dense_order, 6))]
    calck_values = [list(u)] + [list(hist[j])
                                for j in range(min(dense_order, 6))]
    k_out = [[0.0] * n for _ in range(7)]
    resample_at_chebyshev(k_out[:n_nodes], calck_values, calck_thetas, n_nodes)

    return u, fsallast, k_out


def jacobian_fd(f, t, u, du_base):
    n = len(u)
    jac = [[0.0] * n for _ in range(n)]
    for j in range(n):
        up = list(u)
        h = max(1.0, abs(u[j])) * 1.4901161193847656e-8
        up[j] = u[j] + h
        dup = f(t, up)
        inv_h = 1.0 / h
        for i in range(n):
            jac[i][j] = (dup[i] - du_base[i]) * inv_h
    return jac


def jacobian2w(jac, n, dtgamma):
    inv = 1.0 / dtgamma
    return [[jac[i][j] - (inv if i == j else 0.0) for j in range(n)]
            for i in range(n)]


# ---------------------------------------------------------------------------
# Stability Limit Detection (`_stald_detect`, stald.jl:186 = CVODE cvSLdet)
# ---------------------------------------------------------------------------
# Direct 1-based -> 0-based transcription.  `ssdat` is 5x3, row 0 newest.
# Returns the same integer status encoding upstream uses:
#   < 0  no characteristic root could be established
#   1..3 root found, magnitude <= rrcut (stable)
#   4..6 stability violation (rr > rrcut)
STALD_DEFAULTS = dict(
    rrcut=0.98, vrrtol=1.0e-4, vrrt2=5.0e-4,
    sqtol=1.0e-3, rrtol=1.0e-2, tiny=1.0e-90,
)


def stald_make(ssdat, **kw):
    st = dict(STALD_DEFAULTS)
    st.update(kw)
    st["ssdat"] = [list(r) for r in ssdat]
    for name in ("smax", "ssmax", "rav", "vrat", "drr", "rrc", "sigsq"):
        st[name] = [0.0] * 3
    st["rat"] = [[0.0] * 3 for _ in range(4)]
    st["qc"] = [[0.0] * 3 for _ in range(5)]
    st["qco"] = [[0.0] * 3 for _ in range(5)]
    return st


def stald_detect(st, q):
    ss = st["ssdat"]
    tiny = st["tiny"]
    # Phase 1: per-level statistics.
    for k in range(3):
        smink = ss[0][k]
        smaxk = 0.0
        for i in range(5):
            smink = min(smink, ss[i][k])
            smaxk = max(smaxk, ss[i][k])
        if smink < tiny * smaxk:
            return -1
        st["smax"][k] = smaxk
        st["ssmax"][k] = smaxk * smaxk
        sumrat = sumrsq = 0.0
        for i in range(4):
            st["rat"][i][k] = ss[i][k] / ss[i + 1][k]
            sumrat += st["rat"][i][k]
            sumrsq += st["rat"][i][k] ** 2
        st["rav"][k] = 0.25 * sumrat
        st["vrat"][k] = abs(0.25 * sumrsq - st["rav"][k] ** 2)
        st["qc"][4][k] = ss[0][k] * ss[2][k] - ss[1][k] * ss[1][k]
        st["qc"][3][k] = ss[1][k] * ss[2][k] - ss[0][k] * ss[3][k]
        st["qc"][2][k] = 0.0
        st["qc"][1][k] = ss[1][k] * ss[4][k] - ss[2][k] * ss[3][k]
        st["qc"][0][k] = ss[3][k] * ss[3][k] - ss[2][k] * ss[4][k]
    vmin = min(st["vrat"])
    vmax = max(st["vrat"])
    rr = 0.0
    kflag = 0
    if vmin < st["vrrtol"] ** 2:
        if vmax > st["vrrt2"] ** 2:
            return -2
        rr = sum(st["rav"]) / 3.0
        drrmax = 0.0
        for k in range(3):
            drrmax = max(drrmax, abs(st["rav"][k] - rr))
        if drrmax > st["vrrt2"]:
            return -3
        kflag = 1
    else:
        qc, qco = st["qc"], st["qco"]
        for j in range(5):
            for k in range(3):
                qco[j][k] = qc[j][k]
        if abs(qco[3][0]) > tiny * st["smax"][0] * st["smax"][0]:
            r1 = qco[3][1] / qco[3][0]
            r2 = qco[3][2] / qco[3][0]
            for j in range(4, -1, -1):
                if j != 3:
                    qco[j][1] -= r1 * qco[j][0]
                    qco[j][2] -= r2 * qco[j][0]
        else:
            return -4
        if abs(qco[4][1]) > tiny * st["smax"][1] * st["smax"][1]:
            r1 = qco[4][2] / qco[4][1]
            for j in range(4, -1, -1):
                if j != 3 and j != 4:
                    qco[j][2] -= r1 * qco[j][1]
        else:
            if abs(qco[4][2]) > tiny * st["smax"][2] * st["smax"][2]:
                return -4
        if abs(qco[3][2]) < tiny:
            return -4
        rr = -qco[4][2] / qco[3][2]
        if rr < tiny or rr > 100.0:
            return -5
        sqmax = 0.0
        for k in range(3):
            qkr = qc[4][k] + rr * (qc[3][k] + rr * rr * (qc[1][k] + rr * qc[0][k]))
            sqmax = max(sqmax, abs(qkr) / st["ssmax"][k])
        if sqmax < st["sqtol"]:
            kflag = 2
        else:
            kmin = 0
            for _ in range(3):
                st["drr"] = [0.0] * 3
                st["rrc"] = [0.0] * 3
                for k in range(3):
                    qkr = qc[4][k] + rr * (qc[3][k] + rr * rr * (qc[1][k] + rr * qc[0][k]))
                    qp = qc[3][k] + rr * rr * (3.0 * qc[1][k] + rr * 4.0 * qc[0][k])
                    if abs(qp) > tiny * st["ssmax"][k]:
                        st["drr"][k] = -qkr / qp
                    st["rrc"][k] = rr + st["drr"][k]
                sqmin = math.inf
                for k in range(3):
                    if st["rrc"][k] > 0:
                        rk = st["rrc"][k]
                        qkr = qc[4][k] + rk * (qc[3][k] + rk * rk * (qc[1][k] + rk * qc[0][k]))
                        saqk = abs(qkr) / st["ssmax"][k]
                        if saqk < sqmin:
                            sqmin = saqk
                            kmin = k
                if kmin == 0 and all(st["rrc"][k] <= 0 for k in range(3)):
                    return -6
                rr = st["rrc"][kmin]
                if sqmin < st["sqtol"]:
                    kflag = 3
                    break
            if kflag == 0:
                return -6
    # Phase 3: the sigsq sequence.
    for k in range(3):
        rsa = ss[0][k]
        rsb = ss[1][k] * rr
        rsc = ss[2][k] * rr * rr
        rsd = ss[3][k] * rr * rr * rr
        rd1a, rd1b, rd1c = rsa - rsb, rsb - rsc, rsc - rsd
        rd2a, rd2b = rd1a - rd1b, rd1b - rd1c
        rd3a = rd2a - rd2b
        if abs(rd1b) < tiny * st["smax"][k]:
            return -7
        cest1 = -rd3a / rd1b
        if cest1 < tiny or cest1 > 4.0:
            return -7
        corr1 = (rd2b / cest1) / (rr * rr)
        st["sigsq"][k] = ss[2][k] + corr1
    if st["sigsq"][1] < tiny:
        return -8
    ratp = st["sigsq"][2] / st["sigsq"][1]
    ratm = st["sigsq"][0] / st["sigsq"][1]
    qfac1 = 0.25 * (q * q - 1)
    qfac2 = 2.0 / (q - 1)
    bb = ratp * ratm - 1.0 - qfac1 * ratp
    tem = 1.0 - qfac2 * bb
    if abs(tem) < tiny:
        return -8
    rrb = 1.0 / tem
    if abs(rrb - rr) > st["rrtol"]:
        return -9
    if rr > st["rrcut"]:
        return kflag + 3
    return kflag


def twomode_ssdat(lam1, lam2, c1, c2, h, q, n0=6):
    """Squared scaled derivative norms for a two-mode solution.

    h^k y^(k)(t_n) = (h*lam1)^k c1 e^{lam1 t_n} + (h*lam2)^k c2 e^{lam2 t_n}
    A single geometric sequence always bails out at Phase 3 (the differences
    cancel exactly), so this sum-of-two-modes shape is what the detector is
    actually built for.
    """
    out = []
    for i in range(5):
        t = (n0 - i) * h
        row = []
        for k in range(3):
            lvl = q - 2 + k
            v = (h * lam1) ** lvl * c1 * math.exp(lam1 * t) + \
                (h * lam2) ** lvl * c2 * math.exp(lam2 * t)
            row.append(v * v)
        out.append(row)
    return out
