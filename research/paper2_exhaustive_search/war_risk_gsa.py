"""PROTOTYPE — NOT INTEGRABLE. DO NOT USE FOR SCIENCE.

Superseded by the SALib-based GSA layer. Audited defects (measured, see
GSA_LAYER_SPEC_AND_AUDIT_STANDARD_2026-07-15.md):
  * morris_screening: hand-rolled trajectories use clipping -> 208/800 steps NULL (26.0%) and
    186/800 off-delta (23.2%) = 49.2% invalid, silently skipped. USE SALib.
  * sobol_indices: assumes a DETERMINISTIC f. On a stochastic DES, internal noise deflates S1 and
    inflates ST -> MANUFACTURES FALSE INTERACTION (measured: ST[x3] 0.235 -> 0.546 at noise sd=3),
    biased toward the very hypothesis under test.
  * "interaction = ST - S1" is FALSE: it double-counts (measured 1.87x on Ishigami). Use S_ij/Shapley.
  * additive(): callable below the calibrated N; floor is ~0.038 at N=1024 and LARGER on a stochastic DES.
  * prim_box: greedy peel only; on pure noise it names irrelevant factors as restricting and
    min_support is inert. Needs pasting/CV/bagging/false-box control.

Retained ONLY as (a) the Ishigami validation bench in test_war_risk_gsa.py and (b) the provenance of
the measured constraints. Importing this for analysis is a governance violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Bounds = Sequence[tuple[float, float]]


def _scale(unit: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Map points from the unit hypercube to the factor bounds."""
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    return lo + unit * (hi - lo)


# --------------------------------------------------------------------------------------
# Morris screening (elementary effects)
# --------------------------------------------------------------------------------------

@dataclass
class MorrisResult:
    names: list[str]
    mu: np.ndarray          # mean elementary effect (signed)
    mu_star: np.ndarray     # mean |EE| -- overall influence ranking
    sigma: np.ndarray       # std of EE -- HIGH => non-linear and/or INTERACTING
    n_evaluations: int

    def interacting(self, ratio: float = 0.5) -> list[str]:
        """Factors whose sigma is large relative to mu_star => non-additive behaviour."""
        out = []
        for i, name in enumerate(self.names):
            if self.mu_star[i] > 0 and self.sigma[i] / max(self.mu_star[i], 1e-12) >= ratio:
                out.append(name)
        return out


def morris_trajectories(k: int, r: int, levels: int, rng: np.random.Generator) -> np.ndarray:
    """r Morris trajectories in the unit hypercube; shape (r, k+1, k).

    Each trajectory perturbs one factor at a time by delta, in random order and direction,
    so consecutive points differ in exactly one coordinate (the elementary-effect design).
    """
    delta = levels / (2.0 * (levels - 1))
    grid = np.linspace(0.0, 1.0 - delta, max(1, levels // 2))
    traj = np.empty((r, k + 1, k), dtype=float)
    for t in range(r):
        base = rng.choice(grid, size=k)
        order = rng.permutation(k)
        signs = rng.choice([-1.0, 1.0], size=k)
        point = base.copy()
        traj[t, 0] = point
        for step, j in enumerate(order):
            point = point.copy()
            point[j] = np.clip(point[j] + signs[step] * delta, 0.0, 1.0)
            traj[t, step + 1] = point
    return traj


def morris_screening(
    f: Callable[[np.ndarray], float],
    bounds: Bounds,
    *,
    names: Sequence[str] | None = None,
    r: int = 10,
    levels: int = 4,
    seed: int = 0,
) -> MorrisResult:
    """Elementary-effects screening. Cost: r*(k+1) evaluations."""
    k = len(bounds)
    names = list(names) if names is not None else [f"x{i}" for i in range(k)]
    rng = np.random.default_rng(seed)
    traj = morris_trajectories(k, r, levels, rng)

    ee: list[list[float]] = [[] for _ in range(k)]
    n_eval = 0
    for t in range(r):
        pts = traj[t]
        vals = []
        for p in pts:
            vals.append(float(f(_scale(p, bounds))))
            n_eval += 1
        for step in range(k):
            diff = pts[step + 1] - pts[step]
            j = int(np.argmax(np.abs(diff)))
            d = diff[j]
            if abs(d) < 1e-12:
                continue
            ee[j].append((vals[step + 1] - vals[step]) / d)

    mu = np.array([np.mean(e) if e else 0.0 for e in ee])
    mu_star = np.array([np.mean(np.abs(e)) if e else 0.0 for e in ee])
    sigma = np.array([np.std(e, ddof=1) if len(e) > 1 else 0.0 for e in ee])
    return MorrisResult(names=names, mu=mu, mu_star=mu_star, sigma=sigma, n_evaluations=n_eval)


# --------------------------------------------------------------------------------------
# Sobol / Saltelli variance-based indices
# --------------------------------------------------------------------------------------

@dataclass
class SobolResult:
    names: list[str]
    S1: np.ndarray          # first-order (main effect)
    ST: np.ndarray          # total effect (main + all interactions involving i)
    interaction: np.ndarray  # ST - S1 == the interaction mass OAT cannot see
    variance: float
    n_evaluations: int

    def additive(self, tol: float = 0.03) -> bool:
        """True when no factor carries meaningful interaction mass => OAT null generalises.

        CALIBRATION IS MANDATORY.  ``ST - S1`` is a difference of two noisy estimators, so it has
        a positive **noise floor** even when the true interaction is exactly zero.  Measured on a
        known-additive function (f = 3a + 2b - c), the floor is:

            n=1024 -> ~0.038 | n=4096 -> ~0.031 | n=16384 -> ~0.001-0.012

        Declaring "additive" with ``tol`` at or below the floor for the chosen ``n`` reads noise
        as a result.  Use n >= 16384 with tol ~0.03 (floor is then well under the tolerance), and
        verify the floor with ``test_additive_function_has_no_interaction_mass`` before trusting
        an additivity claim.  This calibration is pre-registration-relevant: the "additive =>
        the OAT null generalises to the interior" conclusion rests entirely on it.
        """
        return bool(np.all(self.interaction <= tol))


def sobol_indices(
    f: Callable[[np.ndarray], float],
    bounds: Bounds,
    *,
    names: Sequence[str] | None = None,
    n: int = 512,
    seed: int = 0,
) -> SobolResult:
    """Saltelli cross-sampling with Jansen estimators. Cost: n*(k+2) evaluations.

    S_i  (Saltelli 2010):  (1/n) sum f_B * (f_AB_i - f_A) / V
    S_Ti (Jansen 1999):    (1/2n) sum (f_A - f_AB_i)^2 / V
    """
    k = len(bounds)
    names = list(names) if names is not None else [f"x{i}" for i in range(k)]
    rng = np.random.default_rng(seed)

    A = rng.random((n, k))
    B = rng.random((n, k))

    def ev(mat: np.ndarray) -> np.ndarray:
        return np.array([float(f(_scale(row, bounds))) for row in mat])

    fA = ev(A)
    fB = ev(B)
    n_eval = 2 * n

    var = float(np.var(np.concatenate([fA, fB]), ddof=1))
    S1 = np.zeros(k)
    ST = np.zeros(k)
    if var <= 1e-15:  # degenerate/constant response
        return SobolResult(names, S1, ST, ST - S1, var, n_eval)

    for i in range(k):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB = ev(AB)
        n_eval += n
        S1[i] = float(np.mean(fB * (fAB - fA)) / var)
        ST[i] = float(np.mean((fA - fAB) ** 2) / (2.0 * var))

    S1 = np.clip(S1, 0.0, 1.0)
    ST = np.clip(ST, 0.0, 1.0)
    return SobolResult(names, S1, ST, np.clip(ST - S1, 0.0, 1.0), var, n_eval)


# --------------------------------------------------------------------------------------
# PRIM -- scenario discovery (bump hunting)
# --------------------------------------------------------------------------------------

@dataclass
class PrimBox:
    limits: list[tuple[float, float]]
    names: list[str]
    density: float      # mean target inside the box (for binary: fraction positive)
    coverage: float     # fraction of ALL positive cases captured by the box
    support: float      # fraction of all points inside the box
    n_inside: int
    restricted: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [
            f"{self.names[i]} in [{lo:.3f}, {hi:.3f}]"
            for i, (lo, hi) in enumerate(self.limits)
            if self.names[i] in self.restricted
        ]
        return " AND ".join(parts) if parts else "<no restriction — target not localisable>"


def prim_box(
    X: np.ndarray,
    y: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    peel_alpha: float = 0.05,
    min_support: float = 0.10,
    threshold: float | None = None,
) -> PrimBox:
    """Patient Rule Induction Method: find the box maximising mean(y) inside.

    ``y`` may be binary (reversal indicator) or continuous (tailoring gain). If ``threshold``
    is given, y is binarised as y >= threshold first (the scenario-discovery convention).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if threshold is not None:
        y = (y >= threshold).astype(float)
    n, k = X.shape
    names = list(names) if names is not None else [f"x{i}" for i in range(k)]

    limits = [(float(X[:, i].min()), float(X[:, i].max())) for i in range(k)]
    inside = np.ones(n, dtype=bool)
    total_positive = float(y.sum())
    restricted: set[str] = set()

    while inside.sum() / n > min_support:
        best = None
        for i in range(k):
            xi = X[inside, i]
            if xi.size == 0:
                continue
            for side in ("lo", "hi"):
                q = np.quantile(xi, peel_alpha if side == "lo" else 1.0 - peel_alpha)
                cand = inside & ((X[:, i] > q) if side == "lo" else (X[:, i] < q))
                if cand.sum() == 0 or cand.sum() / n <= min_support:
                    continue
                mean_in = float(y[cand].mean())
                if best is None or mean_in > best[0]:
                    best = (mean_in, i, side, q, cand)
        if best is None:
            break
        cur_mean = float(y[inside].mean())
        if best[0] <= cur_mean + 1e-12:  # no improvement available -> stop peeling
            break
        _, i, side, q, cand = best
        lo, hi = limits[i]
        limits[i] = (float(q), hi) if side == "lo" else (lo, float(q))
        inside = cand
        restricted.add(names[i])

    n_in = int(inside.sum())
    density = float(y[inside].mean()) if n_in else 0.0
    coverage = float(y[inside].sum() / total_positive) if total_positive > 0 else 0.0
    return PrimBox(
        limits=limits, names=names, density=density, coverage=coverage,
        support=n_in / n, n_inside=n_in, restricted=sorted(restricted),
    )
