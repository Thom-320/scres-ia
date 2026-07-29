#!/usr/bin/env python3
"""Resource-adjusted terminal analysis for the preregistered DRA-2 gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_dra2_exact_branching import branch_actions, state_policy  # noqa: E402
from supply_chain.dra2_convoy import static_policies  # noqa: E402
from supply_chain.dra2_experiment import resource_dominance_static_comparator  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--branching-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    tapes = json.loads((args.frontier_dir / "tapes.json").read_text())
    tape_map = {row["tape_id"]: row for row in tapes}
    states = pd.read_csv(args.branching_dir / "states.csv")
    seq = pd.read_csv(args.branching_dir / "sequence_rows.csv")
    seq7 = seq[seq["sequence_days"] == 7].copy()
    sequence_state_ids = sorted(seq7["state_id"].unique())
    state_rows = {
        row["state_id"]: row
        for row in states.to_dict(orient="records")
        if row["state_id"] in sequence_state_ids
    }

    static_path = args.branching_dir / "resource_static_rows.csv"
    if not static_path.exists():
        static_rows: list[dict] = []
        for state_id in sequence_state_ids:
            state = state_rows[state_id]
            prefix = state_policy(state)
            for policy in static_policies():
                result = branch_actions(
                    tape_map[state["tape_id"]], state, prefix, (),
                    continuation_policy=policy,
                )
                static_rows.append({
                    "state_id": state_id,
                    "tape_id": state["tape_id"],
                    "family": state["family"],
                    "policy_id": policy.policy_id,
                    **result,
                })
            print(f"[dra2-resource] {state_id} static continuations complete", flush=True)
        write_csv(static_path, static_rows)
    static = pd.read_csv(static_path)

    oracle = (
        seq7.sort_values(["state_id", "long_ret", "long_service"], ascending=[True, False, True])
        .groupby("state_id", as_index=False).first()
    )
    candidate = {
        "mean_departures": float(oracle["op8_convoy_departures"].mean()),
        "mean_unavailable_hours": float(oracle["op8_convoy_unavailable_hours"].mean()),
    }
    static_summaries = []
    for policy_id, rows in static.groupby("policy_id"):
        static_summaries.append({
            "policy_id": policy_id,
            "mean_ret": float(rows["long_ret"].mean()),
            "mean_service": float(rows["long_service"].mean()),
            "mean_departures": float(rows["op8_convoy_departures"].mean()),
            "mean_unavailable_hours": float(rows["op8_convoy_unavailable_hours"].mean()),
        })
    comparator = resource_dominance_static_comparator(candidate, static_summaries)
    baseline = static[static["policy_id"] == comparator["policy_id"]].set_index("state_id")
    oracle_i = oracle.set_index("state_id")
    baseline = baseline.loc[oracle_i.index]
    delta_ret = (oracle_i["long_ret"] - baseline["long_ret"]).to_numpy()
    service_reduction = (
        (baseline["long_service"] - oracle_i["long_service"])
        / baseline["long_service"].abs().clip(lower=1.0)
    ).to_numpy()
    lost_delta = (oracle_i["long_lost"] - baseline["long_lost"]).to_numpy()
    ret_ci = bootstrap_mean(delta_ret, 20260712, args.n_boot)
    service_ci = bootstrap_mean(service_reduction, 20260713, args.n_boot)

    branch_verdict = json.loads((args.branching_dir / "verdict.json").read_text())
    counts = branch_verdict["one_action_diversity_supported_counts"]
    n_states = int(branch_verdict["n_states"])
    diversity_pass = all(counts[action] / n_states >= 0.15 for action in ("HOLD", "DISPATCH_NOW"))
    ret_pass = ret_ci[0] >= 0.01 and ret_ci[1] > 0.0
    service_pass = service_ci[0] >= 0.05 and service_ci[1] > 0.0
    sufficiency_pass = bool(branch_verdict["sequence_sufficiency"]["pass"])
    resource_pass = (
        comparator["mean_departures"] <= candidate["mean_departures"] + 1e-9
        and comparator["mean_unavailable_hours"] <= candidate["mean_unavailable_hours"] + 1e-9
    )
    all_pre_tree = all([
        branch_verdict["g_b_strong_liveness_pass"], diversity_pass,
        ret_pass, service_pass, sufficiency_pass, resource_pass,
    ])
    verdict = {
        "gate": "DRA2_RESOURCE_ADJUSTED_CALIBRATION_GATE",
        "n_tapes": len(sequence_state_ids),
        "resource_candidate": candidate,
        "resource_dominance_static_comparator": comparator,
        "ret_delta_mean_ci95": ret_ci,
        "service_loss_reduction_mean_ci95": service_ci,
        "lost_orders_delta_mean": float(lost_delta.mean()),
        "strong_liveness_pass": bool(branch_verdict["g_b_strong_liveness_pass"]),
        "diversity_pass": diversity_pass,
        "ret_practical_gate_pass": ret_pass,
        "service_practical_gate_pass": service_pass,
        "sequence_sufficiency_pass": sufficiency_pass,
        "resource_envelope_pass": resource_pass,
        "observable_tree_authorized": all_pre_tree,
        "holdout_opened": False,
        "ppo_authorized": False,
        "ppo_trained": False,
        "virgin_tapes_opened": 0,
        "interpretation": (
            "PROMOTE_TO_OBSERVABLE_TREE"
            if all_pre_tree
            else "STOP_DRA2_PRE_RL_GATE"
        ),
        "failed_gates": [
            name for name, passed in (
                ("strong_liveness", branch_verdict["g_b_strong_liveness_pass"]),
                ("action_diversity", diversity_pass),
                ("ret_practical", ret_pass),
                ("service_practical", service_pass),
                ("sequence_sufficiency_7d_10d", sufficiency_pass),
                ("resource_envelope", resource_pass),
            ) if not passed
        ],
    }
    (args.branching_dir / "resource_gate_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if all_pre_tree else 2


if __name__ == "__main__":
    raise SystemExit(main())
