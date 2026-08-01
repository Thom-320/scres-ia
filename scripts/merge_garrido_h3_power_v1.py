#!/usr/bin/env python3
"""Merge the two preregistered H3' power slices without changing the estimand.

The active surface runner stores the per-context search costs, not a precomputed H3 variance.
This script derives the preregistered quantity independently from those sealed traces:
variance across the six contexts per replicate, followed by ``variance(reset) -
variance(retained)``. It refuses to interpret missing, resealed, overlapping, or
source-mismatched inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402

CONTRACT = ROOT / "docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md"
RUNNER = ROOT / "scripts/run_meta_learner_over_configs_v1.py"
DEFAULT_LOCAL = ROOT / "results/garrido_meta_learner_h3power_local/result.json"
DEFAULT_VPS = ROOT / "results/garrido_meta_learner_h3power_vps/result.json"
DEFAULT_OUTPUT = ROOT / "results/garrido_meta_learner_h3power_merge_v1/result.json"
EXPECTED_LOCAL_SEEDS = list(range(6_000_001, 6_000_091))
EXPECTED_VPS_SEEDS = list(range(6_000_091, 6_000_121))
EXPECTED_CONTEXTS = (
    "R1r", "R2r", "R1r+R2r", "R1r|esc", "R2r|esc", "R1r+R2r|esc"
)
SOURCE_FILES = (
    "scripts/run_meta_learner_over_configs_v1.py",
    "supply_chain/supply_chain.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/config.py",
    "supply_chain/arm_runner.py",
    "supply_chain/provenance.py",
    "supply_chain/fidelity_moments.py",
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def current_source_manifest() -> dict[str, str]:
    return {
        relative: file_sha256(ROOT / relative)
        for relative in SOURCE_FILES
    }


def recompute_seal(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "self_sha256"}
    return sha256(json.dumps(body, indent=1, sort_keys=True, default=str).encode()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("self_sha256") != recompute_seal(payload):
        raise ValueError(f"seal mismatch: {path}")
    return payload


def variance_trace(payload: dict[str, Any], strategy: str) -> np.ndarray:
    contexts = tuple(payload["contexts"])
    rows = payload["per_context"][strategy]
    values = []
    for repeat in rows:
        costs = [float(repeat[context]["runs_to_within_1pct"]) for context in contexts]
        values.append(float(np.var(costs, ddof=1)))
    return np.asarray(values, dtype=float)


def bootstrap(values: np.ndarray, *, n_boot: int) -> dict[str, Any]:
    rng = np.random.default_rng(20260801)
    draws = values[rng.integers(0, values.size, size=(int(n_boot), values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lcb95": float(np.percentile(draws, 2.5)),
        "ucb95": float(np.percentile(draws, 97.5)),
        "n_replicates": int(values.size),
        "inference_unit": "replicate; each replicate contains the six context costs",
    }


def checks(
    local: dict[str, Any],
    vps: dict[str, Any],
    *,
    remote_runner_sha256: str | None,
    remote_source_manifest: dict[str, str] | None,
) -> dict[str, Any]:
    local_seeds = [int(seed) for seed in local.get("seeds", [])]
    vps_seeds = [int(seed) for seed in vps.get("seeds", [])]
    local_contexts = tuple(local.get("contexts", ()))
    vps_contexts = tuple(vps.get("contexts", ()))
    local_falsifiers = local.get("falsifiers", {})
    vps_falsifiers = vps.get("falsifiers", {})
    input_seals_valid = (
        local.get("self_sha256") == recompute_seal(local)
        and vps.get("self_sha256") == recompute_seal(vps)
    )
    input_falsifiers_pass = all(
        bool(item.get("passed"))
        for item in list(local_falsifiers.values()) + list(vps_falsifiers.values())
        if isinstance(item, dict)
    ) and bool(local_falsifiers.get("all_passed")) and bool(vps_falsifiers.get("all_passed"))
    trace_present = all(
        strategy in local.get("per_context", {})
        and strategy in vps.get("per_context", {})
        and len(local["per_context"][strategy]) == len(local_seeds)
        and len(vps["per_context"][strategy]) == len(vps_seeds)
        and all(
            all(context in repeat and "runs_to_within_1pct" in repeat[context]
                for context in EXPECTED_CONTEXTS)
            for repeat in local["per_context"][strategy] + vps["per_context"][strategy]
        )
        for strategy in ("neuron_memory", "neuron_reset", "ofat", "random")
    )
    runner_sha = file_sha256(RUNNER)
    source_manifest = current_source_manifest()
    source_match = (
        remote_source_manifest is not None
        and remote_source_manifest == source_manifest
        and remote_source_manifest.get("scripts/run_meta_learner_over_configs_v1.py") == runner_sha
    )
    return {
        "f_merge_seeds_are_disjoint": {
            "passed": local_seeds == EXPECTED_LOCAL_SEEDS
            and vps_seeds == EXPECTED_VPS_SEEDS
            and not (set(local_seeds) & set(vps_seeds)),
            "evidence": {
                "why_it_can_fail": "overlapping or shifted slices are not independent replications",
                "local_seeds": local_seeds,
                "vps_seeds": vps_seeds,
                "overlap": sorted(set(local_seeds) & set(vps_seeds)),
            },
        },
        "f_merge_contexts_and_budget_match": {
            "passed": local_contexts == EXPECTED_CONTEXTS
            and vps_contexts == EXPECTED_CONTEXTS
            and local.get("budget") == 24
            and vps.get("budget") == 24
            and local.get("n_configurations") == 288
            and vps.get("n_configurations") == 288,
            "evidence": {
                "why_it_can_fail": "different contexts, budgets or surfaces would mix contracts",
                "local_contexts": list(local_contexts),
                "vps_contexts": list(vps_contexts),
                "local_budget": local.get("budget"),
                "vps_budget": vps.get("budget"),
                "local_n_configurations": local.get("n_configurations"),
                "vps_n_configurations": vps.get("n_configurations"),
            },
        },
        "f_merge_source_is_identical": {
            "passed": remote_runner_sha256 is not None
            and remote_runner_sha256 == runner_sha
            and source_match,
            "evidence": {
                "why_it_can_fail": "slices from different runner or DES module revisions are not one experiment",
                "local_runner_sha256": runner_sha,
                "remote_runner_sha256": remote_runner_sha256,
                "local_source_manifest": source_manifest,
                "remote_source_manifest": remote_source_manifest,
                "source_manifest_match": source_match,
            },
        },
        "f_input_seals_are_valid": {
            "passed": input_seals_valid,
            "evidence": {
                "why_it_can_fail": "editing a sealed input after completion invalidates the merge",
                "local_self_sha256": local.get("self_sha256"),
                "vps_self_sha256": vps.get("self_sha256"),
                "recomputed_match": input_seals_valid,
            },
        },
        "f_input_falsifiers_pass": {
            "passed": input_falsifiers_pass,
            "evidence": {
                "why_it_can_fail": "a failed source falsifier voids the powered merge",
                "local_all_passed": local_falsifiers.get("all_passed"),
                "vps_all_passed": vps_falsifiers.get("all_passed"),
            },
        },
        "f_h3_trace_estimand_is_available": {
            "passed": trace_present,
            "evidence": {
                "why_it_can_fail": "without six context costs per replicate H3' cannot be computed",
                "trace_present": trace_present,
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--vps", type=Path, default=DEFAULT_VPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-runner-sha256", required=True)
    parser.add_argument("--remote-source-manifest", type=Path, required=True,
                        help="JSON mapping of SOURCE_FILES to hashes captured on the VPS")
    parser.add_argument("--n-boot", type=int, default=10_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        local = load(args.local)
        vps = load(args.vps)
        remote_source_manifest = json.loads(args.remote_source_manifest.read_text())
    except FileNotFoundError as exc:
        print(f"H3 merge halted: {exc}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"H3 merge halted: invalid source manifest: {exc}")
        return 1
    except ValueError as exc:
        print(f"H3 merge halted: {exc}")
        return 1
    falsifiers = checks(
        local,
        vps,
        remote_runner_sha256=args.remote_runner_sha256,
        remote_source_manifest=remote_source_manifest,
    )
    falsifiers["all_passed"] = all(item["passed"] for item in falsifiers.values())
    if not falsifiers["all_passed"]:
        print("H3 merge halted: falsifier failed")
        for name, item in falsifiers.items():
            if name != "all_passed":
                print(f"  {name}: {'PASA' if item['passed'] else 'FALLA'}")
        return 1

    memory = np.concatenate([
        variance_trace(local, "neuron_memory"), variance_trace(vps, "neuron_memory")
    ])
    reset = np.concatenate([
        variance_trace(local, "neuron_reset"), variance_trace(vps, "neuron_reset")
    ])
    ofat = np.concatenate([variance_trace(local, "ofat"), variance_trace(vps, "ofat")])
    h3_memory_reset = bootstrap(reset - memory, n_boot=args.n_boot)
    h3_memory_ofat = bootstrap(ofat - memory, n_boot=args.n_boot)
    h3_supported = float(h3_memory_reset["lcb95"]) > 0.0
    payload: dict[str, Any] = {
        "schema_version": "garrido_h3_power_merge_v1",
        "claim_status": "H3_PRIME_SUPPORTED_POWERED" if h3_supported else "H3_PRIME_NOT_SUPPORTED_POWERED",
        "contract_path": str(CONTRACT),
        "estimand": "variance(reset) - variance(retained) across six contexts per replicate",
        "construct_change": "H3 prime is search-cost dispersion, not WRAP cost volatility",
        "h3": {
            "memory_vs_reset": h3_memory_reset,
            "memory_vs_ofat": h3_memory_ofat,
            "mean_variance": {
                "neuron_memory": float(memory.mean()),
                "neuron_reset": float(reset.mean()),
                "ofat": float(ofat.mean()),
            },
        },
        "inputs": {
            "local": {"path": str(args.local), "self_sha256": local["self_sha256"], "seeds": local["seeds"]},
            "vps": {"path": str(args.vps), "self_sha256": vps["self_sha256"], "seeds": vps["seeds"]},
            "runner_sha256": file_sha256(RUNNER),
            "remote_runner_sha256": args.remote_runner_sha256,
        },
        "repeats": int(memory.size),
        "contexts": list(EXPECTED_CONTEXTS),
        "budget": 24,
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = seal_and_write(payload, args.output, contract=CONTRACT, reference=args.local)
    print(f"Saved: {args.output}")
    print(f"claim_status: {payload['claim_status']}")
    print(f"memory vs reset variance: {h3_memory_reset}")
    print(f"self_sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
