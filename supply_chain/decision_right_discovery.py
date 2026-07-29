"""Program I utilities for global sensitivity and adaptive-headroom gates.

The module deliberately separates output sensitivity from decision eligibility.
It has no simulator dependency, so designs and gates can be unit-tested before
opening any scientific tape universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence
from itertools import product

import numpy as np


@dataclass(frozen=True)
class NumericFactor:
    name: str
    lower: float
    upper: float
    group: str
    factor_class: str = "decision_right"

    def scale(self, value: float) -> float:
        return self.lower + float(value) * (self.upper - self.lower)


def morris_trajectories(
    factors: Sequence[NumericFactor], trajectories: int, levels: int, seed: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate randomized one-at-a-time Morris trajectories.

    Returns scaled design rows and `(factor_index, direction)` for every edge.
    The implementation is dependency-free and deterministic. SALib is used by
    the production Sobol/refinement runner when installed.
    """
    if levels < 4 or levels % 2:
        raise ValueError("Morris levels must be an even integer >= 4")
    if trajectories < 1 or not factors:
        raise ValueError("Morris needs factors and at least one trajectory")
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1.0))
    grid = np.linspace(0.0, 1.0 - delta, levels // 2)
    rows: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    for _ in range(trajectories):
        x = rng.choice(grid, size=len(factors), replace=True).astype(float)
        rows.append(x.copy())
        for idx in rng.permutation(len(factors)):
            direction = 1 if x[idx] + delta <= 1.0 else -1
            x = x.copy()
            x[idx] += direction * delta
            rows.append(x.copy())
            edges.append((int(idx), direction))
    scaled = np.asarray(
        [[factor.scale(value) for factor, value in zip(factors, row)] for row in rows],
        dtype=float,
    )
    return scaled, edges


def morris_effects(
    design: np.ndarray,
    outputs: Sequence[float],
    factors: Sequence[NumericFactor],
    edges: Sequence[tuple[int, int]],
) -> dict[str, dict[str, float]]:
    """Calculate signed mean, mu-star and sigma from trajectory outputs."""
    y = np.asarray(outputs, dtype=float)
    k = len(factors)
    if (
        len(y) != len(design)
        or k == 0
        or len(design) % (k + 1)
        or len(edges) != len(design) - len(design) // (k + 1)
    ):
        raise ValueError("Design, edge, and output sizes do not form Morris trajectories")
    effects: dict[str, list[float]] = {factor.name: [] for factor in factors}
    edge_pos = 0
    row_pos = 0
    while row_pos < len(design):
        for step in range(k):
            idx, _direction = edges[edge_pos]
            dx = design[row_pos + step + 1, idx] - design[row_pos + step, idx]
            if abs(dx) <= 1e-15:
                raise ValueError("Zero Morris step")
            effects[factors[idx].name].append((y[row_pos + step + 1] - y[row_pos + step]) / dx)
            edge_pos += 1
        row_pos += k + 1
    result = {}
    for name, values in effects.items():
        arr = np.asarray(values, dtype=float)
        result[name] = {
            "mu": float(arr.mean()),
            "mu_star": float(np.abs(arr).mean()),
            "sigma": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "sign_stability": float(max(np.mean(arr >= 0), np.mean(arr <= 0))),
            "n_effects": int(len(arr)),
        }
    return result


def paired_bootstrap(values: Sequence[float], seed: int, n_boot: int = 4000) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    boot = np.mean(arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))], axis=1)
    return {"mean": float(arr.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def adaptive_headroom_verdict(
    state_action_values: Mapping[str, Mapping[str, float]],
    state_tapes: Mapping[str, str],
    observable_deltas: Sequence[float] | None = None,
    service_reduction: float = 0.0,
    resource_equivalent: bool = False,
    thresholds: Mapping[str, float] | None = None,
) -> dict:
    """Fail-closed conjunctive gate separating sensitivity from headroom."""
    t = {
        "action_support_min": .15,
        "single_action_support_max": .85,
        "oracle_ret_delta_min": .01,
        "service_loss_reduction_min": .05,
        "observable_capture_min": .30,
        "observable_favorable_tapes_min": .70,
        **(thresholds or {}),
    }
    states = sorted(state_action_values)
    if not states:
        raise ValueError("No branched states")
    actions = sorted(set().union(*(state_action_values[s] for s in states)))
    if len(actions) < 2:
        raise ValueError("At least two same-contract actions are required")
    constant_means = {a: float(np.mean([state_action_values[s][a] for s in states])) for a in actions}
    best_constant = max(actions, key=lambda a: (constant_means[a], a))
    support = {a: 0.0 for a in actions}
    tape_deltas: dict[str, list[float]] = {}
    for state in states:
        values = state_action_values[state]
        best = max(values.values())
        winners = [a for a in actions if np.isclose(values[a], best, atol=1e-12)]
        for action in winners:
            support[action] += 1.0 / len(winners)
        tape_deltas.setdefault(state_tapes[state], []).append(best - values[best_constant])
    support = {a: value / len(states) for a, value in support.items()}
    oracle_by_tape = [float(np.mean(v)) for v in tape_deltas.values()]
    oracle = paired_bootstrap(oracle_by_tape, seed=20260713)
    oracle_mean = float(oracle["mean"])
    obs = np.asarray(observable_deltas if observable_deltas is not None else [], dtype=float)
    observable_mean = float(obs.mean()) if len(obs) else float("nan")
    capture = observable_mean / oracle_mean if len(obs) and oracle_mean > 0 else float("nan")
    favorable = float(np.mean(obs > 0)) if len(obs) else 0.0
    material_support = [value for value in support.values() if value >= t["action_support_min"]]
    gates = {
        "ranking_diversity": len(material_support) >= 2 and max(support.values()) <= t["single_action_support_max"],
        "oracle_headroom": oracle_mean >= t["oracle_ret_delta_min"] and oracle["ci95"][0] > 0,
        "service_practical": service_reduction >= t["service_loss_reduction_min"],
        "resource_equivalence": bool(resource_equivalent),
        "observable_conversion": bool(len(obs)) and observable_mean > 0 and capture >= t["observable_capture_min"] and favorable >= t["observable_favorable_tapes_min"],
    }
    return {
        "best_constant": best_constant,
        "action_support": support,
        "oracle_delta": oracle,
        "observable_delta_mean": observable_mean,
        "observable_capture": capture,
        "observable_favorable_fraction": favorable,
        "gates": gates,
        "promote_to_rl": all(gates.values()),
    }


def evaluate_design(
    design: np.ndarray,
    factors: Sequence[NumericFactor],
    tapes: Iterable[int],
    simulator: Callable[[Mapping[str, float], int], Mapping[str, float]],
) -> dict[str, np.ndarray]:
    """Evaluate every configuration on identical tape ids (strict CRN API)."""
    tape_ids = tuple(int(tape) for tape in tapes)
    aggregate: dict[str, list[float]] = {}
    for row in design:
        params = {factor.name: float(value) for factor, value in zip(factors, row)}
        per_tape = [simulator(params, tape) for tape in tape_ids]
        keys = set(per_tape[0])
        if any(set(result) != keys for result in per_tape):
            raise ValueError("Simulator output schema changed between tapes")
        for key in keys:
            aggregate.setdefault(key, []).append(float(np.mean([result[key] for result in per_tape])))
    return {key: np.asarray(values, dtype=float) for key, values in aggregate.items()}


def d_optimal_categorical(
    levels: Mapping[str, Sequence[str]], n_rows: int, seed: int
) -> list[dict[str, str]]:
    """Select an approximate D-optimal main-effects design by greedy exchange."""
    names = list(levels)
    candidates = [dict(zip(names, values)) for values in product(*(levels[name] for name in names))]
    if n_rows >= len(candidates):
        return candidates
    columns = [(name, level) for name in names for level in levels[name][1:]]
    matrix = np.asarray([
        [1.0] + [float(row[name] == level) for name, level in columns]
        for row in candidates
    ])
    if n_rows < matrix.shape[1]:
        raise ValueError("D-optimal rows must cover the main-effects model rank")
    rng = np.random.default_rng(seed)
    selected = list(rng.choice(len(candidates), size=n_rows, replace=False))

    def score(indices: Sequence[int]) -> float:
        sign, logdet = np.linalg.slogdet(matrix[indices].T @ matrix[indices] + np.eye(matrix.shape[1]) * 1e-12)
        return float(logdet) if sign > 0 else -np.inf

    best = score(selected)
    improved = True
    while improved:
        improved = False
        unselected = [idx for idx in range(len(candidates)) if idx not in selected]
        for pos in rng.permutation(n_rows):
            for replacement in rng.permutation(unselected):
                trial = selected.copy()
                trial[int(pos)] = int(replacement)
                value = score(trial)
                if value > best + 1e-10:
                    selected, best, improved = trial, value, True
                    break
            if improved:
                break
    return [candidates[idx] for idx in sorted(selected)]


def select_candidate_families(
    sensitivity: Mapping[str, Mapping[str, float]],
    catalog_factors: Sequence[Mapping],
    minimum: int = 3,
    maximum: int = 5,
) -> list[dict]:
    """Select influential implemented decision families, never environment inputs."""
    catalog = {str(row["id"]): row for row in catalog_factors}
    eligible = []
    for factor_id, effects in sensitivity.items():
        row = catalog.get(factor_id)
        if not row or row.get("class") != "decision_right":
            continue
        if not str(row.get("status", "")).startswith("implemented"):
            continue
        eligible.append((float(effects["mu_star"]), float(effects["sigma"]), factor_id, row))
    eligible.sort(reverse=True)
    by_family: dict[str, dict] = {}
    for mu_star, sigma, factor_id, row in eligible:
        family = str(row["family"])
        if family not in by_family:
            by_family[family] = {
                "family": family, "lead_factor": factor_id,
                "mu_star": mu_star, "sigma": sigma,
                "status": "REQUIRES_ACTION_CONTRACT_AND_BRANCHING",
            }
    selected = list(by_family.values())[:maximum]
    if len(selected) < minimum:
        return []
    return selected
