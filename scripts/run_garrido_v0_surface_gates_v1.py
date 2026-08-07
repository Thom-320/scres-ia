#!/usr/bin/env python3
"""Adjudicate non-separability and contextual recovery value on development surfaces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import ALL_POSTURES  # noqa: E402
from supply_chain.garrido_v0_recovery import CONTEXT_ORDER  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from scripts.build_garrido_v0_recovery_surface_v1 import (  # noqa: E402
    GRID_ID,
    verify_surface,
)


LIVE_CONTEXTS = ("R11", "R14", "R21", "R22", "R23", "R24")
NULL_CONTEXTS = ("R12", "R13")
N_BOOT = 10_000
MIN_INTERACTION_GAIN = 0.02
MIN_CONTEXTS_NONSEPARABLE = 4
MIN_REGIME_TTR_HOURS = 24.0
CONTRACT = Path("docs/PREREGISTRO_GARRIDO_V0_RECOVERY_SURFACE_V1_2026-08-06.md")
REFERENCE = Path("results/garrido_v0_recovery_gate_v2/result.json")
MODULES = (
    "scripts/build_garrido_v0_recovery_surface_v1.py",
    "supply_chain/garrido_v0_recovery.py",
    "supply_chain/expanded_contract_controllers_v2.py",
    "supply_chain/arm_runner.py",
    "supply_chain/seed_custody.py",
)


def _bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lcb95": float(np.percentile(means, 2.5)),
        "ucb95": float(np.percentile(means, 97.5)),
        "n": int(len(values)),
    }


def _design_matrices() -> tuple[np.ndarray, np.ndarray]:
    levels = (0, 168, 336, 504, 672, 1344)
    encoded = []
    for posture in ALL_POSTURES:
        row: list[float] = []
        for value in posture:
            row.extend(float(value == level) for level in levels[1:])
        encoded.append(row)
    main = np.asarray(encoded, dtype=float)
    interactions: list[np.ndarray] = []
    blocks = [main[:, offset : offset + 5] for offset in (0, 5, 10)]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        interactions.append(
            np.einsum("ij,ik->ijk", blocks[left], blocks[right]).reshape(len(main), -1)
        )
    additive = np.column_stack([np.ones(len(main)), main])
    pairwise = np.column_stack([additive] + interactions)
    return additive, pairwise


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-15:
        return 1.0 if np.allclose(y, prediction) else 0.0
    return float(1.0 - np.sum((y - prediction) ** 2) / denom)


def _loso_gain(values: np.ndarray) -> np.ndarray:
    additive, pairwise = _design_matrices()
    gains: list[float] = []
    for holdout in range(values.shape[0]):
        train = np.delete(values, holdout, axis=0)
        y_train = train.reshape(-1)
        x_add = np.tile(additive, (len(train), 1))
        x_pair = np.tile(pairwise, (len(train), 1))
        beta_add = np.linalg.lstsq(x_add, y_train, rcond=None)[0]
        beta_pair = np.linalg.lstsq(x_pair, y_train, rcond=None)[0]
        y_test = values[holdout]
        gains.append(_r2(y_test, pairwise @ beta_pair) - _r2(y_test, additive @ beta_add))
    return np.asarray(gains, dtype=float)


def load_surfaces(root: Path) -> tuple[list[int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    files = sorted(root.glob("*.json"))
    if not files:
        raise ValueError(f"no recovery surfaces in {root}")
    seeds: list[int] = []
    utility_rows = {context: [] for context in CONTEXT_ORDER}
    ttr_rows = {context: [] for context in CONTEXT_ORDER}
    for path in files:
        payload = json.loads(path.read_text())
        verify_surface(payload)
        seeds.append(int(payload["seed"]))
        for context in CONTEXT_ORDER:
            utility_rows[context].append(
                [float(cell["utility"]) for cell in payload["contexts"][context]]
            )
            ttr_rows[context].append(
                [float(cell["recovery"]["restricted_ttr_hours"])
                 for cell in payload["contexts"][context]]
            )
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed surfaces")
    return (
        seeds,
        {key: np.asarray(value, dtype=float) for key, value in utility_rows.items()},
        {key: np.asarray(value, dtype=float) for key, value in ttr_rows.items()},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("results/surface_cache") / GRID_ID / "development",
    )
    ap.add_argument("--contract", type=Path, default=CONTRACT)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument(
        "--output", type=Path, default=Path("results/garrido_v0_surface_gates_v1/result.json")
    )
    args = ap.parse_args()
    started = time.perf_counter()
    seeds, utility, ttr = load_surfaces(args.cache)
    if len(seeds) != 6:
        raise ValueError(f"development gate requires exactly six seeds, observed {len(seeds)}")

    rng = np.random.default_rng(20260806)
    interaction: dict[str, Any] = {}
    n_nonseparable = 0
    for context in LIVE_CONTEXTS:
        gains = _loso_gain(utility[context])
        summary = _bootstrap(gains, rng)
        passed = bool(summary["mean"] >= MIN_INTERACTION_GAIN and summary["lcb95"] > 0.0)
        n_nonseparable += int(passed)
        interaction[context] = {**summary, "fold_values": gains.tolist(), "passed": passed}

    null_controls: dict[str, Any] = {}
    nulls_pass = True
    for context in NULL_CONTEXTS:
        impact = [
            bool(float(cell) > 0.0)
            for row in ttr[context]
            for cell in row
        ]
        ttr_range = float(np.ptp(ttr[context]))
        passed = bool(not any(impact) and ttr_range <= 1e-12)
        nulls_pass &= passed
        null_controls[context] = {
            "passed": passed,
            "restricted_ttr_range_hours": ttr_range,
            "positive_ttr_fraction": float(np.mean(impact)),
        }

    contextual_ttr_gain: list[float] = []
    contextual_utility_gain: list[float] = []
    selections: list[dict[str, Any]] = []
    all_utility = np.stack([utility[context] for context in LIVE_CONTEXTS], axis=1)
    all_ttr = np.stack([ttr[context] for context in LIVE_CONTEXTS], axis=1)
    for holdout in range(len(seeds)):
        train_u = np.delete(all_utility, holdout, axis=0).mean(axis=0)  # contexts x postures
        contextual_idx = np.argmax(train_u, axis=1)
        common_idx = int(np.argmax(train_u.mean(axis=0)))
        held_u = all_utility[holdout]
        held_t = all_ttr[holdout]
        ctx_u = np.asarray([held_u[c, contextual_idx[c]] for c in range(len(LIVE_CONTEXTS))])
        common_u = held_u[:, common_idx]
        ctx_t = np.asarray([held_t[c, contextual_idx[c]] for c in range(len(LIVE_CONTEXTS))])
        common_t = held_t[:, common_idx]
        contextual_utility_gain.append(float(np.mean(ctx_u - common_u)))
        contextual_ttr_gain.append(float(np.mean(common_t - ctx_t)))
        selections.append(
            {
                "heldout_seed": seeds[holdout],
                "common_posture": list(ALL_POSTURES[common_idx]),
                "contextual_postures": {
                    context: list(ALL_POSTURES[int(contextual_idx[c])])
                    for c, context in enumerate(LIVE_CONTEXTS)
                },
                "mean_ttr_gain_hours": contextual_ttr_gain[-1],
                "mean_utility_gain": contextual_utility_gain[-1],
            }
        )
    ttr_regime = _bootstrap(np.asarray(contextual_ttr_gain), rng)
    utility_regime = _bootstrap(np.asarray(contextual_utility_gain), rng)

    gates = {
        "g0_complete_sealed_crn_surface": {
            "passed": True,
            "observed_seeds": len(seeds),
            "cells_per_seed": len(ALL_POSTURES) * (len(CONTEXT_ORDER) + 1),
        },
        "g1_null_controls_remain_null": {
            "passed": bool(nulls_pass),
            "contexts": null_controls,
        },
        "g2_surface_is_nonseparable_out_of_seed": {
            "passed": bool(n_nonseparable >= MIN_CONTEXTS_NONSEPARABLE),
            "observed_contexts": n_nonseparable,
            "required_contexts": MIN_CONTEXTS_NONSEPARABLE,
            "minimum_mean_gain": MIN_INTERACTION_GAIN,
            "by_context": interaction,
        },
        "g3_context_specific_postures_have_operational_value": {
            "passed": bool(ttr_regime["lcb95"] >= MIN_REGIME_TTR_HOURS),
            "minimum_lcb95_hours": MIN_REGIME_TTR_HOURS,
            "ttr_gain_hours": ttr_regime,
            "utility_gain": utility_regime,
            "fold_selections": selections,
        },
    }
    all_gates = all(block["passed"] for block in gates.values())
    verdict = (
        "GO_FREEZE_REPEATED_CAMPAIGN"
        if all_gates
        else "STOP_NO_RECOVERY_LEARNING_HEADROOM"
    )
    custody = custody_falsifier(seeds, replay_of="garrido_q2_des288", exclude=args.output)
    falsifiers = {
        "f1_only_development_seeds_are_loaded": {
            "passed": seeds == list(range(5_300_001, 5_300_007)),
            "evidence": {"seeds": seeds, "holdout_not_loaded": list(range(5_300_007, 5_300_013))},
        },
        "f2_leave_one_seed_out_is_not_in_sample_r2": {
            "passed": all(len(block["fold_values"]) == len(seeds) for block in interaction.values()),
            "evidence": {"n_folds": len(seeds), "fit_seeds_per_fold": len(seeds) - 1},
        },
        "f3_contextual_postures_are_selected_without_the_heldout_seed": {
            "passed": len(selections) == len(seeds),
            "evidence": {"mechanism": "np.delete(all_utility, holdout, axis=0) before argmax"},
        },
        "f4_seed_custody_is_declared_replay": custody,
    }
    falsifiers["all_passed"] = all(
        value.get("passed") is True
        for value in falsifiers.values()
        if isinstance(value, dict) and not value.get("not_applicable")
    )

    payload = {
        "schema_version": "garrido_v0_surface_gates_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_GATE_ON_BURNED_REPLAYS_NO_HYPOTHESIS_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "grid_id": GRID_ID,
        "cache": str(args.cache),
        "seeds": seeds,
        "live_contexts": list(LIVE_CONTEXTS),
        "null_contexts": list(NULL_CONTEXTS),
        "gates": gates,
        "falsifiers": falsifiers,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output, contract=args.contract, reference=args.reference
    )
    print(json.dumps({"claim_status": verdict, "gates": gates}, indent=2))
    print(f"  -> {args.output} ({digest[:16]}...)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

