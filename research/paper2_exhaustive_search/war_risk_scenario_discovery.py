"""Cross-validated PRIM scenario discovery for war-stress development data.

The implementation delegates peeling and pasting to EMA Workbench's
scenario-discovery PRIM.  This wrapper adds a frozen, non-interactive trajectory
selection rule and repeated configuration-level holdout validation.  A returned
box is still a development hypothesis and cannot promote without the parent
atlas's prospective validation gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*ipyparallel.*")
    from ema_workbench.analysis import prim


@dataclass(frozen=True)
class PrimSplitResult:
    split_id: int
    selected_step: int | None
    restricted_factors: tuple[str, ...]
    train_density: float
    train_coverage: float
    train_support: float
    holdout_density: float
    holdout_coverage: float
    holdout_support: float
    pass_holdout: bool


@dataclass(frozen=True)
class ValidatedPrimResult:
    status: str
    split_results: tuple[PrimSplitResult, ...]
    holdout_pass_fraction: float
    median_holdout_density: float
    median_holdout_coverage: float
    median_holdout_support: float
    holdout_density_ci95: tuple[float, float]
    holdout_coverage_ci95: tuple[float, float]
    holdout_support_ci95: tuple[float, float]
    factor_selection_frequency: dict[str, float]
    stable_factors: tuple[str, ...]
    permutation_repeats: int
    permutation_pvalue: float
    null_max_holdout_pass_fraction: float
    claim_limit: str = "development scenario hypothesis only; never a confirmatory result"


def _select_trajectory_step(
    trajectory: pd.DataFrame,
    *,
    minimum_density: float,
    minimum_coverage: float,
    minimum_support: float,
) -> int | None:
    """Select the earliest (least peeled) point satisfying all train gates."""
    eligible = trajectory[
        (trajectory["density"] >= minimum_density)
        & (trajectory["coverage"] >= minimum_coverage)
        & (trajectory["mass"] >= minimum_support)
    ]
    if eligible.empty:
        return None
    return int(eligible.index[0])


def _limits_and_restrictions(
    box: prim.PrimBox,
    step: int,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    limits = box.box_lims[int(step)].copy()
    restricted: list[str] = []
    for name in train.columns:
        lo, hi = float(limits.loc[0, name]), float(limits.loc[1, name])
        if lo > float(train[name].min()) + 1e-12 or hi < float(train[name].max()) - 1e-12:
            restricted.append(str(name))
    return limits, tuple(sorted(restricted))


def _inside(frame: pd.DataFrame, limits: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for name in frame.columns:
        mask &= frame[name].to_numpy(dtype=float) >= float(limits.loc[0, name]) - 1e-12
        mask &= frame[name].to_numpy(dtype=float) <= float(limits.loc[1, name]) + 1e-12
    return mask


def _box_metrics(y: np.ndarray, inside: np.ndarray) -> tuple[float, float, float]:
    support = float(np.mean(inside)) if inside.size else 0.0
    positives = float(np.sum(y))
    density = float(np.mean(y[inside])) if np.any(inside) else 0.0
    coverage = float(np.sum(y[inside]) / positives) if positives > 0.0 else 0.0
    return density, coverage, support


def _run_prim_splits(
    frame: pd.DataFrame,
    y: np.ndarray,
    split_indexes: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    peel_alpha: float,
    paste_alpha: float,
    minimum_train_density: float,
    minimum_train_coverage: float,
    minimum_support: float,
    minimum_holdout_density: float,
    minimum_holdout_coverage: float,
) -> list[PrimSplitResult]:
    results: list[PrimSplitResult] = []
    for split_id, (train_index, test_index) in enumerate(split_indexes):
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)
        y_train = y[train_index]
        y_test = y[test_index]
        if min(int(np.sum(y_train)), int(len(y_train) - np.sum(y_train))) < 2:
            step = None
            box = None
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*ipyparallel.*")
                algorithm = prim.Prim(
                    train,
                    y_train,
                    threshold=float(minimum_train_density),
                    peel_alpha=float(peel_alpha),
                    paste_alpha=float(paste_alpha),
                    mass_min=float(minimum_support),
                )
                box = algorithm.find_box()
            step = _select_trajectory_step(
                box.peeling_trajectory,
                minimum_density=minimum_train_density,
                minimum_coverage=minimum_train_coverage,
                minimum_support=minimum_support,
            )
        if step is None or box is None:
            results.append(
                PrimSplitResult(
                    split_id=split_id,
                    selected_step=None,
                    restricted_factors=(),
                    train_density=0.0,
                    train_coverage=0.0,
                    train_support=0.0,
                    holdout_density=0.0,
                    holdout_coverage=0.0,
                    holdout_support=0.0,
                    pass_holdout=False,
                )
            )
            continue
        trajectory_row = box.peeling_trajectory.loc[step]
        limits, restricted = _limits_and_restrictions(box, step, train)
        holdout_density, holdout_coverage, holdout_support = _box_metrics(
            y_test,
            _inside(test, limits),
        )
        passed = bool(
            holdout_density >= minimum_holdout_density
            and holdout_coverage >= minimum_holdout_coverage
            and holdout_support >= minimum_support
        )
        results.append(
            PrimSplitResult(
                split_id=split_id,
                selected_step=step,
                restricted_factors=restricted,
                train_density=float(trajectory_row["density"]),
                train_coverage=float(trajectory_row["coverage"]),
                train_support=float(trajectory_row["mass"]),
                holdout_density=holdout_density,
                holdout_coverage=holdout_coverage,
                holdout_support=holdout_support,
                pass_holdout=passed,
            )
        )
    return results


def validated_prim_discovery(
    X: np.ndarray,
    positive: Sequence[bool | int],
    *,
    names: Sequence[str],
    repeats: int = 20,
    test_fraction: float = 0.30,
    seed: int = 7470930,
    peel_alpha: float = 0.05,
    paste_alpha: float = 0.05,
    minimum_train_density: float = 0.80,
    minimum_train_coverage: float = 0.50,
    minimum_support: float = 0.05,
    minimum_holdout_density: float = 0.70,
    minimum_holdout_coverage: float = 0.40,
    minimum_holdout_pass_fraction: float = 0.70,
    stable_factor_frequency: float = 0.70,
    permutation_repeats: int = 99,
    maximum_permutation_pvalue: float = 0.05,
) -> ValidatedPrimResult:
    """Discover PRIM boxes with holdouts and an explicit false-box null."""
    points = np.asarray(X, dtype=float)
    y = np.asarray(positive, dtype=bool)
    names = tuple(map(str, names))
    if points.ndim != 2 or points.shape[1] != len(names):
        raise ValueError("scenario matrix does not match factor names")
    if y.shape != (points.shape[0],):
        raise ValueError("one binary outcome is required per scenario")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("scenario discovery requires unique configuration points")
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    if min(positives, negatives) < 10:
        return ValidatedPrimResult(
            status="INSUFFICIENT_POSITIVE_OR_NEGATIVE_CONFIGURATIONS",
            split_results=(),
            holdout_pass_fraction=0.0,
            median_holdout_density=0.0,
            median_holdout_coverage=0.0,
            median_holdout_support=0.0,
            holdout_density_ci95=(0.0, 0.0),
            holdout_coverage_ci95=(0.0, 0.0),
            holdout_support_ci95=(0.0, 0.0),
            factor_selection_frequency={name: 0.0 for name in names},
            stable_factors=(),
            permutation_repeats=int(permutation_repeats),
            permutation_pvalue=1.0,
            null_max_holdout_pass_fraction=0.0,
        )

    frame = pd.DataFrame(points, columns=names)
    splitter = StratifiedShuffleSplit(
        n_splits=int(repeats),
        test_size=float(test_fraction),
        random_state=int(seed),
    )
    split_indexes = list(splitter.split(frame, y))
    results = _run_prim_splits(
        frame,
        y,
        split_indexes,
        peel_alpha=peel_alpha,
        paste_alpha=paste_alpha,
        minimum_train_density=minimum_train_density,
        minimum_train_coverage=minimum_train_coverage,
        minimum_support=minimum_support,
        minimum_holdout_density=minimum_holdout_density,
        minimum_holdout_coverage=minimum_holdout_coverage,
    )
    factor_counts = {name: 0 for name in names}
    for row in results:
        for name in row.restricted_factors:
            factor_counts[name] += 1

    density = np.asarray([row.holdout_density for row in results], dtype=float)
    coverage = np.asarray([row.holdout_coverage for row in results], dtype=float)
    support = np.asarray([row.holdout_support for row in results], dtype=float)
    pass_fraction = float(np.mean([row.pass_holdout for row in results]))
    rng = np.random.default_rng(int(seed) + 1)
    null_pass_fractions: list[float] = []
    for _ in range(int(permutation_repeats)):
        permuted = rng.permutation(y)
        null_results = _run_prim_splits(
            frame,
            permuted,
            split_indexes,
            peel_alpha=peel_alpha,
            paste_alpha=paste_alpha,
            minimum_train_density=minimum_train_density,
            minimum_train_coverage=minimum_train_coverage,
            minimum_support=minimum_support,
            minimum_holdout_density=minimum_holdout_density,
            minimum_holdout_coverage=minimum_holdout_coverage,
        )
        null_pass_fractions.append(float(np.mean([row.pass_holdout for row in null_results])))
    permutation_pvalue = (
        (1.0 + sum(value >= pass_fraction for value in null_pass_fractions))
        / (1.0 + len(null_pass_fractions))
    )
    frequencies = {name: factor_counts[name] / len(results) for name in names}
    stable = tuple(sorted(name for name, value in frequencies.items() if value >= stable_factor_frequency))
    status = (
        "VALIDATED_DEVELOPMENT_BOX_HYPOTHESIS"
        if (
            pass_fraction >= minimum_holdout_pass_fraction
            and stable
            and permutation_pvalue <= maximum_permutation_pvalue
        )
        else "NO_STABLE_PRIM_BOX"
    )
    return ValidatedPrimResult(
        status=status,
        split_results=tuple(results),
        holdout_pass_fraction=pass_fraction,
        median_holdout_density=float(np.median(density)),
        median_holdout_coverage=float(np.median(coverage)),
        median_holdout_support=float(np.median(support)),
        holdout_density_ci95=tuple(np.quantile(density, [0.025, 0.975]).tolist()),
        holdout_coverage_ci95=tuple(np.quantile(coverage, [0.025, 0.975]).tolist()),
        holdout_support_ci95=tuple(np.quantile(support, [0.025, 0.975]).tolist()),
        factor_selection_frequency=frequencies,
        stable_factors=stable,
        permutation_repeats=int(permutation_repeats),
        permutation_pvalue=float(permutation_pvalue),
        null_max_holdout_pass_fraction=(
            max(null_pass_fractions) if null_pass_fractions else 0.0
        ),
    )
