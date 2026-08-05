#!/usr/bin/env python3
"""Run the preregistered DES-288 between-campaign Q2 experiment.

This runner is deliberately separate from the historical/replay runners.  It evaluates the
288-config extension once per context and seed, then lets five search arms consume only the
outcome of the configuration they selected.  The endpoint is a four-component lexicographic
service-first key; no component is collapsed into a weighted scalar.

The learner's retained state is a fixed-scale linear coefficient matrix ``rho``.  It is updated
only after a DES episode and is the only state allowed to cross a context boundary.  This is a
between-run test of Garrido's ③ -> ⑧ link, not an intra-episode controller and not RL.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import custody_falsifier  # noqa: E402
from supply_chain.service_first_metric import (  # noqa: E402
    SERVICE_FIRST_V2_COMPONENTS,
    claimant_fills,
    service_first_key_v2,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
}
FACTOR_NAMES = tuple(FACTORS)
CONFIGS = tuple(
    dict(zip(FACTOR_NAMES, combo))
    for combo in itertools.product(*FACTORS.values())
)
CONFIG_INDEX = {tuple(cfg[name] for name in FACTOR_NAMES): i for i, cfg in enumerate(CONFIGS)}
DEFAULT = {"buffer_hours": 0.0, "shifts": 1, "op9_rop": 24.0, "op12_rop": 24.0}
CONTEXTS = {
    "R1r": (R1R, {}),
    "R2r": (R2R, {}),
    "R1r+R2r": (R1R + R2R, {}),
    "R1r|esc": (R1R, {r: 3.0 for r in R1R}),
    "R2r|esc": (R2R, {r: 3.0 for r in R2R}),
    "R1r+R2r|esc": (R1R + R2R, {r: 3.0 for r in R1R + R2R}),
}
CONTEXT_ORDER = tuple(CONTEXTS)
METRIC = "service_first_resilience_v2"
SEED_BASE = 7_100_001

# Fixed a priori scales.  They are not estimated from the observed surface and are not
# retained as adaptive normalizers.  The third component is negative final backorder.
TARGET_SCALES = np.asarray((1.0, 1.0, 1_000_000.0, 1.0), dtype=float)


def _config_key(config: dict[str, float | int]) -> tuple[float | int, ...]:
    return tuple(config[name] for name in FACTOR_NAMES)


def selected_configs(max_configs: int | None) -> tuple[dict[str, float | int], ...]:
    """Return a smoke subset while retaining every proposal needed by the OFAT control."""
    if max_configs is None:
        return tuple(CONFIGS)
    chosen = list(CONFIGS[: max(1, int(max_configs))])
    required = []
    for name, levels in FACTORS.items():
        for level in levels:
            candidate = dict(DEFAULT)
            candidate[name] = level
            required.append(candidate)
    seen = {_config_key(cfg) for cfg in chosen}
    for candidate in required:
        if _config_key(candidate) not in seen:
            chosen.append(candidate)
            seen.add(_config_key(candidate))
    return tuple(chosen)


def design_features(config: dict[str, float | int]) -> np.ndarray:
    values = [
        float(FACTORS[name].index(config[name])) / float(len(FACTORS[name]) - 1)
        for name in FACTOR_NAMES
    ]
    return np.asarray(values + [1.0], dtype=float)


def evaluate(
    config: dict[str, float | int],
    context: str,
    seed: int,
    horizon: float,
) -> dict[str, Any]:
    risks, freq = CONTEXTS[context]
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={
            "op3_rm": 0.0,
            "op5_rm": 0.0,
            "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0,
        },
        inventory_replenishment_period=0.0,
        seed=int(seed),
        horizon=float(horizon),
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    sim.run()
    panel = compute_episode_metrics(sim)
    # The aggregate thesis lane has placeholder A/B dictionaries with zero mass.  Treating
    # those placeholders as claimants would manufacture a two-claimant partition, so the v2
    # endpoint correctly degenerates to aggregate fill unless split_v1 is explicitly active.
    fills = claimant_fills(sim) if sim.cssu_topology_mode == "split_v1" else {}
    key = service_first_key_v2(panel, fills)
    drivers = [
        float(panel["excel_case_pct_autotomy"]),
        float(panel["excel_case_pct_recovery"]),
        float(panel["excel_case_pct_risk_no_recovery"]),
        float(panel["excel_case_pct_fill_rate"]),
    ]
    demanded_by_claimant = {
        str(k): float(v) for k, v in getattr(sim, "cssu_demanded", {}).items()
    }
    delivered_by_claimant = {
        str(k): float(v) for k, v in getattr(sim, "cssu_delivered", {}).items()
    }
    cssu_total_demanded = float(
        getattr(sim, "total_demanded", sum(demanded_by_claimant.values()))
    )
    cssu_total_delivered = float(
        getattr(sim, "total_delivered", sum(delivered_by_claimant.values()))
    )
    return {
        "config": dict(config),
        "context": context,
        "seed": int(seed),
        "service_key": [float(x) for x in key],
        "claimant_fills": {str(k): float(v) for k, v in fills.items()},
        "demanded_by_claimant": demanded_by_claimant,
        "delivered_by_claimant": delivered_by_claimant,
        "cssu_total_demanded": cssu_total_demanded,
        "cssu_total_delivered": cssu_total_delivered,
        "drivers": drivers,
        "panel": {
            "n_orders": float(panel["n_orders"]),
            "n_served": float(panel["n_served"]),
            "n_lost": float(panel["n_lost"]),
            "flow_fill_rate": float(panel["flow_fill_rate"]),
            "fill_rate": float(panel["fill_rate"]),
            "backorder_qty_final": float(panel["backorder_qty_final"]),
            "service_loss_auc_ration_hours": float(
                panel["service_loss_auc_ration_hours"]
            ),
            "ret_excel_visible_clipped_0_1": float(
                panel["ret_excel_visible_clipped_0_1"]
            ),
            "ret_excel": float(panel["ret_excel"]),
            "delivered_rations": float(panel["delivered_rations"]),
            "demanded_rations": float(panel["demanded_rations"]),
        },
    }


class VectorLinearLearner:
    """Online linear vector predictor; ``rho`` is the only cross-campaign model state."""

    def __init__(self, *, retained: bool, update_enabled: bool, learning_rate: float = 0.25):
        self.retained = bool(retained)
        self.update_enabled = bool(update_enabled)
        self.learning_rate = float(learning_rate)
        self.rho = np.zeros((len(FACTOR_NAMES) + 1, len(SERVICE_FIRST_V2_COMPONENTS)))
        self.n_observations = 0

    def start_context(self) -> None:
        if not self.retained:
            self.rho.fill(0.0)
            self.n_observations = 0

    def observe(self, x: np.ndarray, key: tuple[float, ...]) -> None:
        if not self.update_enabled:
            return
        target = np.asarray(key, dtype=float) / TARGET_SCALES
        prediction = x @ self.rho
        self.rho += self.learning_rate * np.outer(x, target - prediction)
        self.n_observations += 1

    def predict(self, x: np.ndarray) -> tuple[float, ...]:
        return tuple((x @ self.rho * TARGET_SCALES).tolist())

    def select(
        self,
        available: list[int],
        configs: tuple[dict[str, float | int], ...],
    ) -> int:
        if not available:
            raise ValueError("no available configurations")
        if self.n_observations < 3:
            return int(min(available))
        scored = [
            (self.predict(design_features(configs[i])), int(i))
            for i in available
        ]
        return max(scored, key=lambda pair: (pair[0], -pair[1]))[1]

    def state_digest(self) -> str:
        return sha256(self.rho.tobytes()).hexdigest()


def _best_key(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    return max(tuple(row["service_key"]) for row in rows)


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: tuple(row["service_key"]))


def search(
    strategy: str,
    seed: int,
    rng: np.random.Generator,
    surface: dict[tuple[str, int], list[dict[str, Any]]],
    configs: tuple[dict[str, float | int], ...],
    contexts: tuple[str, ...],
    budget: int,
) -> dict[str, Any]:
    if strategy not in {"ofat", "random", "no_update", "retained", "reset"}:
        raise ValueError(strategy)
    learner = None
    if strategy in {"no_update", "retained", "reset"}:
        learner = VectorLinearLearner(
            retained=strategy == "retained", update_enabled=strategy != "no_update"
        )
    index_by_key = {_config_key(cfg): i for i, cfg in enumerate(configs)}
    per_context: dict[str, Any] = {}
    ofat_coordinate_changes: list[int] = []
    for context in contexts:
        if learner is not None:
            learner.start_context()
        table = surface[(context, int(seed))]
        oracle_key = _best_key(table)
        remaining = set(range(len(configs)))
        seen: list[int] = []
        observations: list[dict[str, Any]] = []
        current = dict(DEFAULT)
        factor_index = 0
        level_index = 0
        factor_best: tuple[tuple[float, ...], dict[str, float | int]] | None = None
        start_state_digest = learner.state_digest() if learner is not None else None
        for step in range(int(budget)):
            if not remaining:
                remaining = set(range(len(configs)))
            if strategy == "random":
                idx = int(rng.choice(sorted(remaining)))
            elif strategy == "ofat":
                if factor_index >= len(FACTOR_NAMES):
                    # Design exhausted: re-run the INCUMBENT. The previous guard was
                    # `"idx" not in locals()`, but `del idx` fires once per CONTEXT rather than
                    # once per step, so from step 1 onwards idx was already bound and this branch
                    # silently re-ran the arm's last PROPOSAL instead -- inside the comparator the
                    # headline contrast is measured against.
                    idx = index_by_key[_config_key(current)]
                else:
                    name = FACTOR_NAMES[factor_index]
                    candidate = dict(current, **{name: FACTORS[name][level_index]})
                    ofat_coordinate_changes.append(
                        sum(1 for n in FACTOR_NAMES if candidate[n] != current[n])
                    )
                    # A smoke subset must contain the OFAT proposals by construction.
                    if _config_key(candidate) not in index_by_key:
                        raise AssertionError("OFAT proposal missing from selected surface")
                    idx = index_by_key[_config_key(candidate)]
                    row = table[idx]
                    candidate_key = tuple(row["service_key"])
                    if factor_best is None or candidate_key > factor_best[0]:
                        factor_best = (candidate_key, candidate)
                    level_index += 1
                    if level_index >= len(FACTORS[name]):
                        current = factor_best[1]
                        factor_index += 1
                        level_index = 0
                        factor_best = None
            else:
                idx = learner.select(sorted(remaining), configs)  # type: ignore[union-attr]

            row = table[idx]
            remaining.discard(idx)
            seen.append(idx)
            observations.append(row)
            if learner is not None:
                learner.observe(design_features(configs[idx]), tuple(row["service_key"]))

        best_row = _best_row(observations) if observations else None
        best_key = tuple(best_row["service_key"]) if best_row else None
        runs_to_oracle = next(
            (i + 1 for i, row in enumerate(observations)
             if tuple(row["service_key"]) == oracle_key),
            int(budget) + 1,
        )
        per_context[context] = {
            "oracle_key": list(oracle_key),
            "best_so_far_key": list(best_key) if best_key is not None else None,
            "runs_to_oracle": int(runs_to_oracle),
            "visited_sequence": [int(i) for i in seen],
            "chosen_config": dict(best_row["config"]) if best_row else None,
            "chosen_service_key": list(best_key) if best_key is not None else None,
            "start_state_digest": start_state_digest,
            "end_state_digest": learner.state_digest() if learner is not None else None,
            "n_learner_observations": learner.n_observations if learner is not None else 0,
        }
        if "idx" in locals():
            del idx
    return {
        "strategy": strategy,
        "seed": int(seed),
        "per_context": per_context,
        "ofat_coordinate_changes": ofat_coordinate_changes,
        "final_state_digest": learner.state_digest() if learner is not None else None,
        "final_n_learner_observations": learner.n_observations if learner is not None else 0,
        "surface_sha256": _surface_digest(surface),
        "budget": int(budget),
        "context_order": list(contexts),
    }


def _bootstrap(values: np.ndarray, *, rng: np.random.Generator, n_boot: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "lcb95": float("nan"), "ucb95": float("nan"), "n_groups": 0}
    draws = values[rng.integers(0, values.size, size=(int(n_boot), values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lcb95": float(np.percentile(draws, 2.5)),
        "ucb95": float(np.percentile(draws, 97.5)),
        "n_groups": int(values.size),
    }


def _paired(
    results: dict[str, list[dict[str, Any]]],
    a: str,
    b: str,
    contexts: tuple[str, ...],
    field: str,
    *,
    rng: np.random.Generator,
    n_boot: int,
    sign: str = "a_minus_b",
) -> dict[str, float | int]:
    a_values = np.asarray([
        np.mean([run["per_context"][c][field] for c in contexts])
        for run in results[a]
    ], dtype=float)
    b_values = np.asarray([
        np.mean([run["per_context"][c][field] for c in contexts])
        for run in results[b]
    ], dtype=float)
    delta = a_values - b_values if sign == "a_minus_b" else b_values - a_values
    return _bootstrap(delta, rng=rng, n_boot=n_boot)


def _surface_digest(surface: dict[tuple[str, int], list[dict[str, Any]]]) -> str:
    canonical = {
        f"{context}|{seed}": rows
        for (context, seed), rows in sorted(surface.items())
    }
    return sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _falsifiers(
    *,
    surface: dict[tuple[str, int], list[dict[str, Any]]],
    results: dict[str, list[dict[str, Any]]],
    configs: tuple[dict[str, float | int], ...],
    contexts: tuple[str, ...],
    seeds: list[int],
    budget: int,
    rng: np.random.Generator,
    replay_of: str | None = None,
) -> dict[str, Any]:
    all_rows = [row for rows in surface.values() for row in rows]
    # The endpoint is a tuple, so variation is checked component-wise rather than by inventing
    # a scalar exchange rate.  At least one component must vary beyond its seed noise.
    keys = np.asarray([row["service_key"] for row in all_rows], dtype=float)
    component_spreads = np.ptp(keys, axis=0).tolist() if len(keys) else [0.0] * 4
    component_noise = [
        float(np.mean([
            np.std([
                surface[(context, seed)][i]["service_key"][component]
                for seed in seeds
            ])
            for context in contexts
            for i in range(len(configs))
        ]))
        if contexts and seeds else 0.0
        for component in range(len(SERVICE_FIRST_V2_COMPONENTS))
    ]
    component_variation = [
        spread > 2.0 * noise
        for spread, noise in zip(component_spreads, component_noise, strict=True)
    ]
    ofat_changes = [
        n for run in results["ofat"] for n in run["ofat_coordinate_changes"]
    ]

    common = {
        "surface_keys": sorted((context, int(seed)) for context in contexts for seed in seeds),
        "seed_order": [int(seed) for seed in seeds],
        "context_order": list(contexts),
        "budget": int(budget),
    }
    memory_contract = dict(common, rho_policy="carry", update="after_selected_outcome")
    reset_contract = dict(common, rho_policy="reset", update="after_selected_outcome")
    arm_surface_digests = {
        strategy: {
            run["surface_sha256"] for run in results[strategy]
        }
        for strategy in ("retained", "reset")
    }
    same_surface = (
        len(arm_surface_digests["retained"]) == 1
        and len(arm_surface_digests["reset"]) == 1
        and arm_surface_digests["retained"] == arm_surface_digests["reset"]
    )
    zero_surface = {
        key: [dict(row) for row in rows]
        for key, rows in surface.items()
    }
    zero_memory = search("retained", seeds[0], np.random.default_rng(901), zero_surface,
                         configs, contexts, 0)
    zero_reset = search("reset", seeds[0], np.random.default_rng(901), zero_surface,
                        configs, contexts, 0)
    zero_equal = zero_memory["per_context"] == zero_reset["per_context"]

    random_shadow = {}
    for key, rows in surface.items():
        random_shadow[key] = [
            dict(row, service_key=[float(10_000.0 - i), 0.0, -float(i), 0.0])
            for i, row in enumerate(rows)
        ]
    random_same = True
    random_compared = 0
    for repeat, seed in enumerate(seeds[: min(3, len(seeds))]):
        base = results["random"][repeat]
        shadow_run = search("random", seed, np.random.default_rng(90_000 + repeat), random_shadow,
                            configs, contexts, budget)
        for context in contexts:
            random_compared += 1
            random_same &= (
                base["per_context"][context]["visited_sequence"]
                == shadow_run["per_context"][context]["visited_sequence"]
            )

    driver_shadow = {}
    for key, rows in surface.items():
        order = np.random.default_rng(4242).permutation(len(rows))
        driver_shadow[key] = [
            dict(row, drivers=rows[int(order[i])]["drivers"])
            for i, row in enumerate(rows)
        ]
    driver_same = True
    driver_compared = 0
    for repeat, seed in enumerate(seeds[: min(3, len(seeds))]):
        for strategy in ("retained", "reset"):
            base = results[strategy][repeat]
            shadow_run = search(strategy, seed, np.random.default_rng(90_000 + repeat),
                                driver_shadow, configs, contexts, budget)
            for context in contexts:
                driver_compared += 1
                driver_same &= (
                    base["per_context"][context]["visited_sequence"]
                    == shadow_run["per_context"][context]["visited_sequence"]
                )

    endpoint_recomputed = True
    mass_consistent = True
    n_claimant_modes = set()
    for row in all_rows:
        panel = row["panel"]
        fills = row["claimant_fills"]
        expected = service_first_key_v2(panel, fills)
        endpoint_recomputed &= list(expected) == list(row["service_key"])
        n_claimant_modes.add(len(fills))
        if fills:
            mass_consistent &= (
                np.isclose(
                    sum(row["demanded_by_claimant"].values()),
                    row["cssu_total_demanded"],
                )
                and np.isclose(
                    sum(row["delivered_by_claimant"].values()),
                    row["cssu_total_delivered"],
                )
            )
        else:
            mass_consistent &= np.isclose(row["service_key"][0], panel["flow_fill_rate"])

    return {
        "f1_surface_has_real_variation": {
            "passed": bool(any(component_variation)),
            "evidence": {
                "why_it_can_fail": "if all configurations tie within seed noise, search has no target",
                "endpoint_components": list(SERVICE_FIRST_V2_COMPONENTS),
                "component_spreads": component_spreads,
                "component_seed_noise": component_noise,
                "component_variation": component_variation,
            },
        },
        "f2_ofat_moves_one_factor": {
            "passed": bool(ofat_changes) and max(ofat_changes) <= 1,
            "evidence": {
                "why_it_can_fail": "a multi-coordinate proposal is not the declared thesis-order control",
                "max_coordinates_changed": max(ofat_changes) if ofat_changes else None,
                "n_proposals": len(ofat_changes),
            },
        },
        "f3_memory_reset_share_contract": {
            "passed": memory_contract.copy() | {"rho_policy": "reset"} == reset_contract
            and same_surface
            and all(
                len(results["retained"][r]["per_context"])
                == len(results["reset"][r]["per_context"])
                == len(contexts)
                for r in range(len(seeds))
            ),
            "evidence": {
                "why_it_can_fail": "a seed, order, budget or trace-shape difference confounds retained-reset",
                "retained_contract": memory_contract,
                "reset_contract": reset_contract,
                "arm_surface_digests": {
                    key: sorted(value) for key, value in arm_surface_digests.items()
                },
                "same_surface": same_surface,
            },
        },
        "f4_zero_budget_is_identical": {
            "passed": zero_equal,
            "evidence": {
                "why_it_can_fail": "retained state must not change an arm when no update is allowed",
                "identical": zero_equal,
            },
        },
        "f5_random_does_not_read_outcomes_before_draw": {
            "passed": random_same,
            "evidence": {
                "why_it_can_fail": "changing values before the RNG draw would reveal outcome access",
                "sequences_compared": random_compared,
                "sequences_identical": random_same,
            },
        },
        "f6_drivers_are_post_episode_only": {
            "passed": driver_same,
            "evidence": {
                "why_it_can_fail": "a driver permutation must change no pre-run ranking sequence",
                "sequences_compared": driver_compared,
                "sequences_identical": driver_same,
            },
        },
        "f7_endpoint_key_recomputes_independently": {
            "passed": endpoint_recomputed,
            "evidence": {
                "why_it_can_fail": "a stored endpoint could be a stale or scalarized ranking",
                "all_rows_recomputed": endpoint_recomputed,
            },
        },
        "f8_service_mass_and_claimant_boundary": {
            "passed": mass_consistent and n_claimant_modes <= {0, 2},
            "evidence": {
                "why_it_can_fail": "the endpoint must not create mass or invent a claimant partition",
                "mass_consistent": mass_consistent,
                "claimant_modes": sorted(n_claimant_modes),
            },
        },
        # Custody goes through the central registry, which also scans sealed artifacts. The
        # hand-maintained PRIOR_SEEDS tuple is exactly what supply_chain.seed_custody exists to
        # abolish: it never learned that 7_100_001 had already been consumed by the smokes, so
        # every smoke artifact sealed `virgin_seed_block: true` for a burned seed.
        "f9_confirmation_seeds_are_virgin": custody_falsifier(
            [int(s) for s in seeds], replay_of=replay_of),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-of",
        default=None,
        help="registry block id this run deliberately re-executes; makes the custody falsifier "
             "NOT_APPLICABLE instead of a pass or a failure",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/garrido_q2_des288_v1/result.json")
    # --contract is REQUIRED: a default is how three artifacts got sealed against
    # the wrong document. Previous default was ROOT / "docs/PREREGISTRO_GARRIDO_Q2_DES288_V1_2026-08-01.md"
    parser.add_argument("--contract", type=Path,
                        required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/garrido_wrap_q1/result.json")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--horizon-weeks", type=int, default=52)
    parser.add_argument("--n-boot", type=int, default=5_000)
    parser.add_argument("--max-configs", type=int, default=None,
                        help="smoke-only subset; full confirmation uses all 288")
    parser.add_argument("--max-contexts", type=int, default=None,
                        help="smoke-only subset; full confirmation uses all six")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats <= 0 or args.budget <= 0:
        raise SystemExit("--repeats and --budget must be positive")
    configs = selected_configs(args.max_configs)
    contexts = CONTEXT_ORDER[: args.max_contexts] if args.max_contexts else CONTEXT_ORDER
    seeds = [int(args.seed_base) + i for i in range(int(args.repeats))]
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()
    surface: dict[tuple[str, int], list[dict[str, Any]]] = {}
    evaluations = 0
    for context in contexts:
        for seed in seeds:
            surface[(context, seed)] = [
                evaluate(config, context, seed, horizon) for config in configs
            ]
            evaluations += len(configs)
        print(
            f"  superficie {context} lista ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    results = {strategy: [] for strategy in ("ofat", "random", "no_update", "retained", "reset")}
    for repeat, seed in enumerate(seeds):
        for strategy in results:
            results[strategy].append(
                search(
                    strategy,
                    seed,
                    np.random.default_rng(90_000 + repeat),
                    surface,
                    configs,
                    tuple(contexts),
                    int(args.budget),
                )
            )
        print(f"  réplica {repeat + 1}/{len(seeds)}", flush=True)

    boot_rng = np.random.default_rng(20260801)
    alzheimer = _paired(
        results, "retained", "reset", tuple(contexts), "runs_to_oracle",
        rng=boot_rng, n_boot=args.n_boot, sign="b_minus_a",
    )
    retained_vs_ofat = _paired(
        results, "retained", "ofat", tuple(contexts), "runs_to_oracle",
        rng=boot_rng, n_boot=args.n_boot, sign="b_minus_a",
    )
    retained_vs_random = _paired(
        results, "retained", "random", tuple(contexts), "runs_to_oracle",
        rng=boot_rng, n_boot=args.n_boot, sign="b_minus_a",
    )

    service_deltas = {}
    for index, name in enumerate(SERVICE_FIRST_V2_COMPONENTS):
        service_deltas[name] = _paired(
            results, "retained", "reset", tuple(contexts), "chosen_service_key",
            rng=boot_rng, n_boot=args.n_boot, sign="a_minus_b",
        ) if index == 0 else None
        if index > 0:
            # Extract the selected tuple component without introducing a scalarized endpoint.
            a = np.asarray([
                np.mean([run["per_context"][c]["chosen_service_key"][index] for c in contexts])
                for run in results["retained"]
            ])
            b = np.asarray([
                np.mean([run["per_context"][c]["chosen_service_key"][index] for c in contexts])
                for run in results["reset"]
            ])
            service_deltas[name] = _bootstrap(a - b, rng=boot_rng, n_boot=args.n_boot)
    # The first component needs the same component-wise calculation; the helper above cannot
    # index a tuple field, so replace its placeholder with the independent calculation.
    for index, name in enumerate(SERVICE_FIRST_V2_COMPONENTS):
        a = np.asarray([
            np.mean([run["per_context"][c]["chosen_service_key"][index] for c in contexts])
            for run in results["retained"]
        ])
        b = np.asarray([
            np.mean([run["per_context"][c]["chosen_service_key"][index] for c in contexts])
            for run in results["reset"]
        ])
        service_deltas[name] = _bootstrap(a - b, rng=boot_rng, n_boot=args.n_boot)

    falsifiers = _falsifiers(
        surface=surface,
        results=results,
        configs=configs,
        contexts=tuple(contexts),
        seeds=seeds,
        budget=args.budget,
        rng=boot_rng,
        replay_of=getattr(args, "replay_of", None),
    )
    # A declared replay returns `not_applicable=True, passed=None`; counting it as a failure
    # would make every replay look broken, and counting it as a pass would let a falsifier that
    # cannot fail be reported as evidence. It belongs in neither column.
    falsifiers["all_passed"] = all(
        check["passed"] for key, check in falsifiers.items()
        if key != "all_passed" and isinstance(check, dict) and not check.get("not_applicable"))
    falsifiers["not_applicable"] = sorted(
        key for key, check in falsifiers.items()
        if key not in ("all_passed", "not_applicable")
        and isinstance(check, dict) and check.get("not_applicable"))
    service_guardrail = {
        name: service_deltas[name] for name in SERVICE_FIRST_V2_COMPONENTS[:3]
    }
    service_guardrail_passed = all(
        float(result["lcb95"]) >= 0.0 for result in service_guardrail.values()
    )
    primary = dict(alzheimer)
    primary_passed = bool(float(primary["lcb95"]) > 0.0)
    if not falsifiers["all_passed"]:
        claim_status = "HALTED_FALSIFIER_FAILED"
    elif primary_passed and service_guardrail_passed and args.max_configs is None and args.max_contexts is None:
        claim_status = "PASS_Q2_CLOSED_LOOP"
    else:
        claim_status = "Q2_EFFECT_NOT_ESTABLISHED"

    runner_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": "garrido_q2_des288_v1",
        "claim_status": claim_status,
        "contract_id": "garrido_q2_des288_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Does retained between-campaign state improve service-safe search over the 288-config DES surface?",
        "estimand": "retained_minus_reset_search_efficiency",
        "endpoint": METRIC,
        "endpoint_components": list(SERVICE_FIRST_V2_COMPONENTS),
        "n_configurations": len(configs),
        "declared_n_configurations": 288,
        "contexts": list(contexts),
        "declared_n_contexts": 6,
        "repeats": len(seeds),
        "budget": int(args.budget),
        "horizon_hours": horizon,
        "seeds": seeds,
        "seed_base": int(args.seed_base),
        "virgin_seed_block": bool(falsifiers["f9_confirmation_seeds_are_virgin"].get("passed") is True),
        "surface_evaluations": evaluations,
        "surface_sha256": _surface_digest(surface),
        "factors": {name: list(levels) for name, levels in FACTORS.items()},
        "arms": results,
        "comparisons": {
            "primary_retained_minus_reset": primary,
            "retained_vs_thesis_order": retained_vs_ofat,
            "retained_vs_random": retained_vs_random,
            "service_component_retained_minus_reset": service_deltas,
            "service_guardrail": service_guardrail,
            "service_guardrail_passed": service_guardrail_passed,
        },
        "falsifiers": falsifiers,
        "decision": {
            "primary_lcb95_positive": primary_passed,
            "service_guardrail_passed": service_guardrail_passed,
            "des_288_complete": len(configs) == 288 and len(contexts) == 6,
            "mlp_ppo_authorized": False,
        },
        "runner": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": runner_sha256,
        "physical_protocol": dict(P),
        "smoke": bool(args.max_configs is not None or args.max_contexts is not None),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=args.reference)
    print(f"Saved: {args.output}")
    print(f"claim_status: {claim_status}")
    print(f"primary retained-reset: {primary}")
    print(f"falsifiers: {'PASA' if falsifiers['all_passed'] else 'FALLA'}; seal: {digest[:16]}…")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
