"""Hand-rolled global sensitivity analysis on scipy.stats.qmc (no SALib available).

- morris_screen: elementary-effects screening -> mu_star (effect) and sigma (interaction/
  non-linearity) per factor. Cheap: (k+1)*r model evaluations.
- sobol_indices: Saltelli sampling + Jansen estimators -> first-order S_i and total S_Ti.
  Cost: N*(k+2) evaluations.
- gp_locate: GaussianProcess + expected-improvement active learning to find argmax of f.
Validated on the Ishigami test function in tests / run script before use on headroom.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def _scale(u, bounds):
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    return lo + u * (hi - lo)


def morris_screen(f, bounds, r=10, levels=8, seed=0):
    """Elementary effects (Morris). f takes a 1D array of length k. Returns per-factor
    mu (mean EE), mu_star (mean |EE|), sigma (std EE)."""
    k = len(bounds); rng = np.random.default_rng(seed)
    delta = levels / (2 * (levels - 1))
    ee = [[] for _ in range(k)]
    for _ in range(r):
        base = rng.integers(0, levels, size=k) / (levels - 1) * (1 - delta)
        perm = rng.permutation(k)
        x = base.copy(); fx = f(_scale(x, bounds))
        for j in perm:
            x2 = x.copy(); x2[j] = min(x2[j] + delta, 1.0)
            if x2[j] == x[j]:
                x2[j] = max(x[j] - delta, 0.0)
            fx2 = f(_scale(x2, bounds))
            d = (fx2 - fx) / (x2[j] - x[j]) if x2[j] != x[j] else 0.0
            ee[j].append(d); x, fx = x2, fx2
    out = {}
    for j, name in enumerate([b_[2] if len(b_) > 2 else f"x{j}" for j, b_ in enumerate(bounds)]):
        a = np.array(ee[j])
        out[name] = {"mu": float(a.mean()), "mu_star": float(np.abs(a).mean()), "sigma": float(a.std())}
    return out


def sobol_indices(f, bounds, N=256, seed=0):
    """Saltelli sampling + Jansen estimators for first-order S_i and total S_Ti."""
    k = len(bounds)
    m = 2 ** int(np.ceil(np.log2(N)))
    s = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    X = s.random(m)
    A = _scale(X[:, :k], bounds); B = _scale(X[:, k:], bounds)
    fA = np.array([f(x) for x in A]); fB = np.array([f(x) for x in B])
    varY = np.var(np.concatenate([fA, fB]))
    names = [b[2] if len(b) > 2 else f"x{i}" for i, b in enumerate(bounds)]
    out = {}
    for i in range(k):
        AB = A.copy(); AB[:, i] = B[:, i]
        fAB = np.array([f(x) for x in AB])
        # Jansen: S_i = 1 - mean((fB-fAB)^2)/(2 varY); S_Ti = mean((fA-fAB)^2)/(2 varY)
        Si = 1 - np.mean((fB - fAB) ** 2) / (2 * varY) if varY > 0 else 0.0
        STi = np.mean((fA - fAB) ** 2) / (2 * varY) if varY > 0 else 0.0
        out[names[i]] = {"S1": float(Si), "ST": float(STi), "interaction_gap": float(STi - Si)}
    out["_var"] = float(varY); out["_n_eval"] = int(m * (k + 2))
    return out


def gp_locate(f, bounds, n_init=16, n_iter=24, seed=0):
    """GP + expected-improvement active learning to maximize f. Returns best (x, y) + history."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from scipy.stats import norm
    k = len(bounds); rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    X = _scale(qmc.LatinHypercube(d=k, seed=seed).random(n_init), bounds)
    y = np.array([f(x) for x in X])
    kern = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(k)) + WhiteKernel(1e-4)
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=seed).fit(X, y)
        cand = lo + rng.random((2048, k)) * (hi - lo)
        mu, sd = gp.predict(cand, return_std=True)
        best = y.max(); imp = mu - best
        z = np.where(sd > 1e-9, imp / sd, 0.0)
        ei = np.where(sd > 1e-9, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)
        xn = cand[int(ei.argmax())]
        X = np.vstack([X, xn]); y = np.append(y, f(xn))
    j = int(y.argmax())
    return {"x_best": X[j].tolist(), "y_best": float(y[j]), "n_eval": len(y)}


# --- validation test function (known Sobol indices) ---
def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * (x[2] ** 4) * np.sin(x[0])


ISHIGAMI_BOUNDS = [(-np.pi, np.pi, "x1"), (-np.pi, np.pi, "x2"), (-np.pi, np.pi, "x3")]
# analytic (a=7,b=0.1): S1 ~ [0.314, 0.442, 0.0]; ST ~ [0.558, 0.442, 0.244]
