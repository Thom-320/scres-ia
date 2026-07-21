#!/usr/bin/env python3
"""Prospective 24-history Q-R1 cold-start replication.

The contract is frozen in ``contracts/q_r1_cold_start_replication_v1.json``.
This runner opens only roots 7570801..7570824, reproduces the retained binary
context estimand, regenerates the frozen D1 candidate calendar family, and
adjudicates the persistent-only tested residual.  It never trains a learner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_q_r1_d0_cold_start_reanalysis import run as run_d0  # noqa: E402
from scripts.run_q_r1_d1_demand_memory import run as run_d1  # noqa: E402
from scripts.run_q_r1_d3_residual_bound import run as run_d3  # noqa: E402
from supply_chain.q_r1_retained_learning import RESOURCE_KEYS  # noqa: E402


CONTRACT = ROOT / "contracts/q_r1_cold_start_replication_v1.json"
OUTPUT_DIR = ROOT / "results/q_r1/cold_start_replication_v1"


def cluster_lcb(by_root: dict[int, list[float]], *, seed: int = 20260722) -> float:
    roots = sorted(by_root)
    means = np.asarray([np.mean(by_root[root]) for root in roots], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10_000, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), 0.025))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def retained_contrast(rows: list[dict], kappa: float, arm: str) -> dict:
    indexed = {
        (float(row["kappa"]), int(row["history_root"]), int(row["campaign_index"]), row["arm"]): row
        for row in rows
    }
    roots = sorted({int(row["history_root"]) for row in rows if float(row["kappa"]) == kappa})
    by_root: dict[int, list[float]] = defaultdict(list)
    favorable: list[bool] = []
    fill: list[float] = []
    unresolved: list[float] = []
    lost: list[float] = []
    resource_error = 0.0
    for root in roots:
        campaigns = sorted({
            int(row["campaign_index"])
            for row in rows
            if float(row["kappa"]) == kappa and int(row["history_root"]) == root
            and row["arm"] == arm and int(row["campaign_index"]) > 0
        })
        for campaign in campaigns:
            target = indexed[(kappa, root, campaign, arm)]
            reset = indexed[(kappa, root, campaign, "reset_posterior_0p5")]
            delta = float(target["early_ret_2w"] - reset["early_ret_2w"])
            by_root[root].append(delta)
            favorable.append(delta > 0.0)
            fill.append(float(target["worst_product_fill"] - reset["worst_product_fill"]))
            unresolved.append(float(target["unresolved_orders"] - reset["unresolved_orders"]))
            lost.append(float(target["lost_orders"] - reset["lost_orders"]))
            for key in RESOURCE_KEYS:
                resource_error = max(resource_error, abs(float(target[key] - reset[key])))
    values = [value for root in roots for value in by_root[root]]
    return {
        "mean_early_ret_delta": float(np.mean(values)),
        "history_clustered_lcb95": cluster_lcb(by_root),
        "favorable_fraction": float(np.mean(favorable)),
        "mean_worst_product_delta": float(np.mean(fill)),
        "max_unresolved_orders_delta": float(np.max(unresolved)),
        "max_lost_orders_delta": float(np.max(lost)),
        "max_resource_error": float(resource_error),
        "n_pairs": len(values),
    }


def adjudicate(d0: dict, d3: dict) -> dict:
    retained = {
        str(kappa): retained_contrast(d0["rows"], kappa, "retained_posterior")
        for kappa in (0.5, 0.75, 0.9)
    }
    shuffled_90 = retained_contrast(d0["rows"], 0.9, "shuffled_posterior")
    wrong_90 = retained_contrast(d0["rows"], 0.9, "wrong_posterior")
    primary = retained["0.9"]
    mechanism_pass = (
        primary["mean_early_ret_delta"] >= retained["0.75"]["mean_early_ret_delta"]
        and retained["0.75"]["mean_early_ret_delta"] > retained["0.5"]["mean_early_ret_delta"]
        and abs(retained["0.5"]["mean_early_ret_delta"]) <= 0.005
        and shuffled_90["mean_early_ret_delta"] <= 0.005
        and wrong_90["mean_early_ret_delta"] <= 0.0
    )
    guardrails_pass = (
        primary["mean_worst_product_delta"] >= -0.02
        and primary["max_unresolved_orders_delta"] <= 0.0
        and primary["max_lost_orders_delta"] <= 0.0
        and primary["max_resource_error"] == 0.0
    )
    retained_pass = (
        primary["mean_early_ret_delta"] >= 0.01
        and primary["history_clustered_lcb95"] > 0.0
        and mechanism_pass
        and guardrails_pass
    )
    residual = d3["primary_persistent"]
    residual_pass = (
        residual["history_clustered_lcb95"] >= 0.01
        and residual["action_divergence"] >= 0.10
    )
    if retained_pass and residual_pass:
        verdict = "PASS_Q_R1_REPLICATION_AUTHORIZE_D4_OBSERVABLE_CONVERSION"
    elif retained_pass:
        verdict = "PASS_RETAINED_BINARY_CONTEXT_COLD_START_ONLY"
    elif residual_pass:
        verdict = "STOP_RETAINED_EFFECT_NOT_REPLICATED_RESIDUAL_ONLY"
    else:
        verdict = "STOP_Q_R1_REPLICATION_NO_RETAINED_OR_RESIDUAL_PASS"
    return {
        "schema_version": "q_r1_cold_start_replication_adjudication_v1",
        "claim_status": "BURNED_PROSPECTIVE_REPLICATION_NO_CONFIRMATORY_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": str(CONTRACT),
        "retained_binary_context": {
            "by_persistence": retained,
            "shuffled_0p90": shuffled_90,
            "wrong_0p90": wrong_90,
            "mechanism_pass": mechanism_pass,
            "guardrails_pass": guardrails_pass,
            "pass": retained_pass,
        },
        "tested_persistent_residual": {
            **residual,
            "pass": residual_pass,
            "boundary": "privileged tested-calendar selector; a pass authorizes D4 only",
        },
        "verdict": verdict,
        "d4_authorized": retained_pass and residual_pass,
        "learner_training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--hard-cap-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    low, high = map(int, contract["history_roots"])
    histories = high - low + 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    d0_path = OUTPUT_DIR / "d0_retained_context.json"
    d0 = run_d0(SimpleNamespace(
        cell="rho90_share90", seed_start=low, histories=histories, campaigns=12,
        output=d0_path,
    ))
    d0_path.write_text(json.dumps(d0, indent=2, sort_keys=True) + "\n")

    d1_path = OUTPUT_DIR / "d1_candidate_calendars.json"
    d1 = run_d1(SimpleNamespace(
        seed_start=low, histories=histories, campaigns=12, horizon=3,
        mode="scenario", particles=args.particles,
        hard_cap_seconds=args.hard_cap_seconds, output=d1_path,
    ))
    d1_path.write_text(json.dumps(d1, indent=2, sort_keys=True) + "\n")

    d3_path = OUTPUT_DIR / "d3_tested_residual.json"
    d3 = run_d3(d1_path)
    d3_path.write_text(json.dumps(d3, indent=2, sort_keys=True) + "\n")

    adjudication = adjudicate(d0, d3)
    adjudication["history_roots_opened"] = [low, high]
    adjudication["histories"] = histories
    adjudication["elapsed_seconds"] = time.perf_counter() - started
    adjudication["artifact_sha256"] = {
        "contract": sha256(CONTRACT), "d0": sha256(d0_path),
        "d1_candidates": sha256(d1_path), "d3": sha256(d3_path),
    }
    output = OUTPUT_DIR / "adjudication.json"
    output.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps(adjudication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
