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


def gp_locate(f, bounds, n_init=16, n_iter=24, seed=0, grid=None):
    """GP + expected-improvement active learning to maximize ``f``.

    ``n_init`` is explicit because it consumes the evaluation budget before EI starts.  The
    returned ``history`` is the authoritative visit log.  When ``grid`` is supplied, continuous
    EI proposals are snapped to the nearest *unvisited* grid point in bounds-normalised distance;
    duplicate proposals are skipped and the next EI candidate is tried.  If all continuous
    proposals snap to visited points, the highest-EI unvisited grid point is used as a deterministic
    fallback.  This makes the continuous-to-discrete rule and duplicate policy part of the API
    rather than an undocumented comparator choice.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from scipy.stats import norm
    if int(n_init) < 1:
        raise ValueError("n_init must be at least 1")
    if int(n_iter) < 0:
        raise ValueError("n_iter must be non-negative")
    k = len(bounds); rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    span = np.maximum(hi - lo, 1e-12)

    grid_arr = None
    grid_seen: set[tuple[float, ...]] = set()
    available: set[int] | None = None
    if grid is not None:
        grid_arr = np.asarray(grid, dtype=float)
        if grid_arr.ndim != 2 or grid_arr.shape[1] != k or grid_arr.shape[0] == 0:
            raise ValueError("grid must be a non-empty 2D array with one column per bound")
        if np.any(grid_arr < lo) or np.any(grid_arr > hi):
            raise ValueError("grid contains a point outside bounds")
        # Duplicate grid rows are not separate configurations.  Keep the first occurrence so the
        # visit index remains deterministic and the no-replacement rule is explicit.
        unique_rows = []
        for row in grid_arr:
            key = tuple(float(value) for value in row)
            if key not in grid_seen:
                grid_seen.add(key)
                unique_rows.append(row)
        grid_arr = np.asarray(unique_rows, dtype=float)
        available = set(range(len(grid_arr)))

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    history: list[dict] = []

    def evaluate_point(x: np.ndarray, *, source: str, grid_index: int | None = None,
                       snap_distance: float | None = None) -> None:
        x = np.asarray(x, dtype=float)
        X_rows.append(x.copy())
        y_value = float(f(x))
        y_rows.append(y_value)
        row = {"x": x.tolist(), "y": y_value, "source": source}
        if grid_index is not None:
            row["grid_index"] = int(grid_index)
        if snap_distance is not None:
            row["snap_distance_normalised"] = float(snap_distance)
        history.append(row)

    def nearest_unvisited(x: np.ndarray) -> tuple[int, float] | None:
        if grid_arr is None or not available:
            return None
        candidates = np.array(sorted(available), dtype=int)
        distance = np.sum(((grid_arr[candidates] - x) / span) ** 2, axis=1)
        position = int(np.argmin(distance))
        return int(candidates[position]), float(np.sqrt(distance[position]))

    n_initial = int(n_init)
    initial_u = qmc.LatinHypercube(d=k, seed=seed).random(n_initial)
    initial_x = _scale(initial_u, bounds)
    if grid_arr is None:
        for x in initial_x:
            evaluate_point(x, source="lhs")
    else:
        for x in initial_x:
            nearest = nearest_unvisited(x)
            if nearest is None:
                break
            grid_index, distance = nearest
            available.remove(grid_index)
            evaluate_point(grid_arr[grid_index], source="lhs_snap",
                           grid_index=grid_index, snap_distance=distance)

    def fit_and_ei(candidates: np.ndarray) -> np.ndarray:
        kern = (ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(k))
                + WhiteKernel(1e-4))
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=seed).fit(np.asarray(X_rows), np.asarray(y_rows))
        mu, sd = gp.predict(candidates, return_std=True)
        best = max(y_rows)
        imp = mu - best
        z = np.where(sd > 1e-9, imp / sd, 0.0)
        return np.where(sd > 1e-9, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)

    for _ in range(int(n_iter)):
        if grid_arr is not None and not available:
            break
        cand = lo + rng.random((2048, k)) * (hi - lo)
        ei = fit_and_ei(cand)
        if grid_arr is None:
            evaluate_point(cand[int(np.argmax(ei))], source="ei")
            continue

        order = np.argsort(-ei, kind="stable")
        selected = None
        for position in order:
            nearest = nearest_unvisited(cand[int(position)])
            if nearest is not None:
                selected = nearest
                break
        if selected is None:
            break
        grid_index, distance = selected
        available.remove(grid_index)
        evaluate_point(grid_arr[grid_index], source="ei_snap",
                       grid_index=grid_index, snap_distance=distance)

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    j = int(y.argmax())
    return {
        "x_best": X[j].tolist(), "y_best": float(y[j]), "n_eval": len(y),
        "history": history, "visited": [row["x"] for row in history],
        "n_init": int(n_init), "n_iter": int(n_iter),
        "grid_anchored": grid_arr is not None,
        "grid_size": None if grid_arr is None else int(len(grid_arr)),
        "duplicate_policy": "skip_duplicate_grid_points_and_use_next_EI_candidate",
    }


# --- validation test function (known Sobol indices) ---
def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * (x[2] ** 4) * np.sin(x[0])


ISHIGAMI_BOUNDS = [(-np.pi, np.pi, "x1"), (-np.pi, np.pi, "x2"), (-np.pi, np.pi, "x3")]
# analytic (a=7,b=0.1): S1 ~ [0.314, 0.442, 0.0]; ST ~ [0.558, 0.442, 0.244]
