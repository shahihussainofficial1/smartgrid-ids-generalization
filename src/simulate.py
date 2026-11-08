"""IEEE 14-bus PMU simulation with false data injection (FDI) attacks.

1. Time-series AC power flow (pandapower) over DAYS days at STEP_MIN-minute
   resolution with daily and weekly load profiles and bus-level load noise.
2. PMUs at buses 2, 6, 7 and 9 (1-based) measure their bus voltage phasor and the
   current phasors of all incident branches. The measurement model z = H x is
   exactly linear in the complex bus voltages x, so H is built from Yf / Yt.
3. Gaussian PMU noise, linear weighted least-squares state estimation and the
   chi-square bad data detector (BDD).
4. Attack episodes: random FDI (bias on a few channels) and stealthy FDI
   a = H c, which leaves the WLS residual unchanged.
Output: data/sim/sim.npz
"""
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sim"
SEED = 7
DAYS = 42
STEP_MIN = 2
PMU_BUSES = [1, 5, 6, 8]       # 0-based: buses 2, 6, 7, 9
SIGMA = 0.005                   # relative complex noise std (well within 1% TVE)
ATTACK_FRACTION = 0.2


def network_model(net):
    """Return internal-index helpers and the complex measurement matrix H."""
    pp.runpp(net, numba=False)
    ppc = net._ppc
    Yf, Yt = ppc["internal"]["Yf"], ppc["internal"]["Yt"]
    br = ppc["internal"]["branch"]
    f_bus = br[:, 0].real.astype(int)
    t_bus = br[:, 1].real.astype(int)
    n = Yf.shape[1]
    lookup = net._pd2ppc_lookups["bus"]
    pmu_int = [int(lookup[b]) for b in PMU_BUSES]
    rows, names = [], []
    for b, bi in zip(PMU_BUSES, pmu_int):
        e = np.zeros(n, complex)
        e[bi] = 1
        rows.append(e); names.append(f"V{b + 1}")
        for k in range(Yf.shape[0]):
            if f_bus[k] == bi:
                rows.append(Yf[k].toarray().ravel()); names.append(f"I{b + 1}-br{k}f")
            if t_bus[k] == bi:
                rows.append(Yt[k].toarray().ravel()); names.append(f"I{b + 1}-br{k}t")
    H = np.array(rows)
    order = np.array([int(lookup[b]) for b in net.bus.index])  # pandapower bus -> internal index
    return H, names, order


def load_profile(T, n_load, rng):
    t = np.arange(T) * STEP_MIN / 60.0  # hours
    daily = 0.78 + 0.12 * np.sin(2 * np.pi * (t - 9) / 24) + 0.07 * np.sin(4 * np.pi * (t - 3) / 24)
    weekly = 1.0 - 0.06 * (((t // 24) % 7) >= 5)
    common = daily * weekly
    # slow per-load random walk (Ornstein-Uhlenbeck) and step-level noise
    walk = np.zeros((T, n_load))
    for i in range(1, T):
        walk[i] = 0.995 * walk[i - 1] + rng.normal(0, 0.004, n_load)
    return common[:, None] * (1 + walk + rng.normal(0, 0.005, (T, n_load)))


def wls(H, z, W):
    G = H.conj().T @ W @ H
    return np.linalg.solve(G, H.conj().T @ W @ z.T).T


def main():
    rng = np.random.default_rng(SEED)
    net = pn.case14()
    H, names, order = network_model(net)
    m, n = H.shape
    assert np.linalg.matrix_rank(H) == n, "PMU placement is not observable"
    T = DAYS * 24 * 60 // STEP_MIN
    p0, q0 = net.load.p_mw.values.copy(), net.load.q_mvar.values.copy()
    g0 = net.gen.p_mw.values.copy()
    prof = load_profile(T, len(p0), rng)

    X = np.zeros((T, n), complex)
    ok = np.ones(T, bool)
    for k in range(T):
        net.load.p_mw = p0 * prof[k]
        net.load.q_mvar = q0 * prof[k]
        net.gen.p_mw = g0 * prof[k].mean()
        try:
            pp.runpp(net, init="results", numba=False)
        except Exception:
            ok[k] = False
            continue
        vm = net.res_bus.vm_pu.values
        va = np.deg2rad(net.res_bus.va_degree.values)
        x = np.zeros(n, complex)
        x[order] = vm * np.exp(1j * va)
        X[k] = x
        if k % 2000 == 0:
            print("power flow", k, "/", T)

    Z_true = X @ H.T
    noise = SIGMA * np.abs(Z_true) * (rng.normal(size=Z_true.shape) + 1j * rng.normal(size=Z_true.shape)) / np.sqrt(2)
    Z = Z_true + noise

    # attack episodes: contiguous 10-60 min windows, attack type drawn at random
    label = np.zeros(T, int)        # 0 normal, 1 random FDI, 2 stealthy FDI
    target = np.full(T, -1)         # attacked bus (stealthy) or first channel (random)
    k, budget = 30, int(ATTACK_FRACTION * T)
    while label.astype(bool).sum() < budget and k < T - 60:
        k += int(rng.integers(15, 121))      # gap of 30-240 min
        L = int(rng.integers(5, 31))         # episode of 10-60 min
        typ = int(rng.integers(1, 3))
        sl = slice(k, min(k + L, T))
        if typ == 1:
            ch = rng.choice(m, size=int(rng.integers(2, 5)), replace=False)
            gain = rng.uniform(0.05, 0.15, len(ch)) * rng.choice([-1, 1], len(ch))
            phase = np.deg2rad(rng.uniform(-3, 3, len(ch)))
            Z[sl, ch] = Z[sl, ch] * (1 + gain) * np.exp(1j * phase)
            target[sl] = ch[0]
        else:
            b = int(rng.choice([i for i in range(n) if i != order[0]]))  # never the reference bus
            c = np.zeros(n, complex)
            c[b] = X[k, b] * (rng.uniform(0.01, 0.05) * rng.choice([-1, 1])
                              + 1j * np.deg2rad(rng.uniform(0.5, 3.0)) * rng.choice([-1, 1]))
            Z[sl] = Z[sl] + (H @ c)[None, :]
            target[sl] = b
        label[sl] = typ
        k = sl.stop

    # state estimation and BDD
    # each real component of the complex noise has variance (SIGMA |z|)^2 / 2,
    # so J = r^H W r follows a chi-square law with 2(m - n) degrees of freedom
    W = np.diag(2 / (SIGMA * np.abs(Z_true).mean(0)) ** 2)
    Xh = wls(H, Z, W)
    R = Z - Xh @ H.T
    J = np.real(np.einsum("ti,ii,ti->t", R.conj(), W, R))
    dof = 2 * (m - n)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "sim.npz", Z=Z, Xh=Xh, J=J, label=label, target=target, ok=ok,
                        H=H, names=np.array(names), dof=dof, sigma=SIGMA, step_min=STEP_MIN)
    print("measurements", m, "states", n, "dof", dof, "snapshots", T, "failed PF", (~ok).sum())
    print("label counts", np.bincount(label))


if __name__ == "__main__":
    main()
