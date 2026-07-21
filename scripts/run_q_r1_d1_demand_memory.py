#!/usr/bin/env python3
"""Burned Q-R1D demand-parameter retention gate; no learner is trained."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_t_full_des_mpc import FullDEST0Config  # noqa: E402
from supply_chain.program_t_joint_belief import ExactJointBelief  # noqa: E402
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    ARMS,
    PERSISTENCE_MODES,
    RESOURCE_KEYS,
    build_parameter_history,
    common_continuation_calendar,
    controller_calendar,
    controller_prefix,
    evaluate_calendar,
    retained_belief_path,
    wrong_belief,
)


def cluster_lcb(values: dict[int, list[float]], *, seed: int = 20260721) -> float:
    roots = sorted(values)
    means = np.asarray([np.mean(values[root]) for root in roots], dtype=float)
    if len(means) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(5000, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), 0.025))


def summarize(rows: list[dict]) -> dict:
    by_key = {
        (row["persistence_mode"], row["history_root"], row["campaign_index"], row["arm"]): row
        for row in rows
    }
    output = {}
    for mode in PERSISTENCE_MODES:
        mode_rows = [row for row in rows if row["persistence_mode"] == mode]
        roots = sorted({row["history_root"] for row in mode_rows})
        arms = {}
        for arm in ARMS:
            early: dict[int, list[float]] = defaultdict(list)
            total: dict[int, list[float]] = defaultdict(list)
            favorable = []
            divergence = []
            fill = []
            early_omitted_delta = []
            unresolved_delta = []
            lost_delta = []
            resource_error = 0.0
            theta_delta: dict[str, list[float]] = defaultdict(list)
            for root in roots:
                campaigns = sorted(
                    row["campaign_index"]
                    for row in mode_rows
                    if row["history_root"] == root
                    and row["arm"] == arm
                    and row["campaign_index"] > 0
                )
                for campaign in campaigns:
                    target = by_key[(mode, root, campaign, arm)]
                    reset = by_key[(mode, root, campaign, "reset_exact_bayes_mpc")]
                    delta = target["early_ret_2w"] - reset["early_ret_2w"]
                    early[root].append(delta)
                    total[root].append(target["ret_visible"] - reset["ret_visible"])
                    favorable.append(delta > 0.0)
                    divergence.append(target["calendar"][:2] != reset["calendar"][:2])
                    fill.append(target["worst_product_fill"] - reset["worst_product_fill"])
                    early_omitted_delta.append(target["early_omitted_rows"] - reset["early_omitted_rows"])
                    unresolved_delta.append(target["unresolved_orders"] - reset["unresolved_orders"])
                    lost_delta.append(target["lost_orders"] - reset["lost_orders"])
                    theta_delta[target["theta_id"]].append(delta)
                    for key in RESOURCE_KEYS:
                        resource_error = max(
                            resource_error, abs(target[key] - reset[key])
                        )
            early_values = [value for root in roots for value in early[root]]
            total_values = [value for root in roots for value in total[root]]
            arms[arm] = {
                "mean_early_ret_delta": float(np.mean(early_values)),
                "early_ret_lcb95_history_clustered": cluster_lcb(early),
                "mean_total_ret_delta": float(np.mean(total_values)),
                "favorable_fraction": float(np.mean(favorable)),
                "first_two_action_divergence": float(np.mean(divergence)),
                "mean_worst_product_delta": float(np.mean(fill)),
                "mean_early_omitted_rows_delta": float(np.mean(early_omitted_delta)),
                "max_unresolved_orders_delta": float(np.max(unresolved_delta)),
                "max_lost_orders_delta": float(np.max(lost_delta)),
                "anti_shedding_pass": (
                    float(np.mean(early_omitted_delta)) <= 0.0
                    and float(np.max(unresolved_delta)) <= 0.0
                    and float(np.max(lost_delta)) <= 0.0
                ),
                "max_resource_error": float(resource_error),
                "theta_mean_deltas": {
                    key: float(np.mean(value)) for key, value in theta_delta.items()
                },
                "n_pairs": len(early_values),
            }
        oracle = arms["oracle_parameters_mpc"]
        retained = arms["retained_exact_bayes_mpc"]
        no_theta_adverse = min(oracle["theta_mean_deltas"].values()) >= -0.005
        oracle_pass = (
            oracle["mean_early_ret_delta"] >= 0.02
            and oracle["early_ret_lcb95_history_clustered"] > 0.0
            and oracle["first_two_action_divergence"] >= 0.10
            and no_theta_adverse
            and oracle["max_resource_error"] == 0.0
            and oracle["anti_shedding_pass"]
        )
        retained_pass = (
            retained["mean_early_ret_delta"] >= 0.01
            and retained["early_ret_lcb95_history_clustered"] > 0.0
            and retained["max_resource_error"] == 0.0
            and retained["anti_shedding_pass"]
        )
        output[mode] = {
            "arms": arms,
            "oracle_headroom_pass": oracle_pass,
            "retained_information_pass": retained_pass,
        }
    iid = output["iid"]["arms"]["retained_exact_bayes_mpc"]["mean_early_ret_delta"]
    dose = (
        output["persistent_0p90"]["arms"]["retained_exact_bayes_mpc"]["mean_early_ret_delta"]
        >= output["persistent_0p75"]["arms"]["retained_exact_bayes_mpc"]["mean_early_ret_delta"]
    )
    oracle_any = any(output[mode]["oracle_headroom_pass"] for mode in PERSISTENCE_MODES)
    retained_pattern = (
        output["persistent_0p90"]["retained_information_pass"]
        and abs(iid) <= 0.005
        and dose
        and output["persistent_0p90"]["arms"]["shuffled_posterior_mpc"]["mean_early_ret_delta"] <= 0.005
        and output["persistent_0p90"]["arms"]["wrong_posterior_mpc"]["mean_early_ret_delta"] <= 0.005
    )
    if not oracle_any:
        verdict = "STOP_CARRIER_DEMAND_NO_COLD_START_HEADROOM"
    elif retained_pattern:
        verdict = "PASS_RETAINED_INFORMATION_VALUE_EXPLORATORY"
    else:
        verdict = "STOP_DEMAND_RETAINED_PATTERN_NOT_CONVERTED"
    return {
        "by_persistence": output,
        "iid_null_abs_le_0p005": abs(iid) <= 0.005,
        "dose_response_0p90_ge_0p75": dose,
        "verdict": verdict,
        "learner_authorized": verdict == "PASS_RETAINED_INFORMATION_VALUE_EXPLORATORY",
    }


def run(args: argparse.Namespace) -> dict:
    sched = scheduler()
    config = FullDEST0Config(
        horizon=args.horizon,
        mode=args.mode,
        particles=args.particles,
        worst_product_floor=0.70,
    )
    histories = {
        mode: [
            build_parameter_history(
                history_root=args.seed_start + index,
                campaigns=args.campaigns,
                persistence_mode=mode,
                scheduler=sched,
            )
            for index in range(args.histories)
        ]
        for mode in PERSISTENCE_MODES
    }
    retained_paths = {
        mode: [retained_belief_path(history) for history in mode_histories]
        for mode, mode_histories in histories.items()
    }
    rows: list[dict] = []
    started = time.perf_counter()
    for mode, mode_histories in histories.items():
        for history_index, history in enumerate(mode_histories):
            shuffled_index = (history_index + 1) % len(mode_histories)
            for campaign_index, campaign in enumerate(history):
                if time.perf_counter() - started > args.hard_cap_seconds:
                    raise TimeoutError("Q-R1 D1 hard cap exceeded")
                beliefs = {
                    "reset_exact_bayes_mpc": ExactJointBelief.uniform(),
                    "retained_exact_bayes_mpc": retained_paths[mode][history_index][campaign_index],
                    "oracle_parameters_mpc": ExactJointBelief.oracle_parameters(campaign.theta),
                    "shuffled_posterior_mpc": retained_paths[mode][shuffled_index][campaign_index],
                    "wrong_posterior_mpc": wrong_belief(retained_paths[mode][history_index][campaign_index]),
                }
                raw_calendars = {}
                diagnostics = {}
                continuation, reset_detail = controller_calendar(
                    campaign=campaign,
                    belief=beliefs["reset_exact_bayes_mpc"],
                    scheduler=sched,
                    config=config,
                )
                raw_calendars["reset_exact_bayes_mpc"] = continuation
                diagnostics["reset_exact_bayes_mpc"] = reset_detail
                for arm, belief in beliefs.items():
                    if arm == "reset_exact_bayes_mpc":
                        continue
                    prefix, detail = controller_prefix(
                        campaign=campaign,
                        belief=belief,
                        scheduler=sched,
                        config=config,
                        decisions=2,
                    )
                    raw_calendars[arm] = tuple(prefix) + tuple(continuation[2:])
                    diagnostics[arm] = detail
                for arm in ARMS:
                    calendar = common_continuation_calendar(
                        raw_calendars[arm], continuation
                    )
                    metrics = evaluate_calendar(
                        campaign=campaign, calendar=calendar, scheduler=sched
                    )
                    rows.append(
                        {
                            "persistence_mode": mode,
                            "persistence": PERSISTENCE_MODES[mode],
                            "history_root": campaign.history_root,
                            "campaign_index": campaign.campaign_index,
                            "tape_seed": campaign.tape_seed,
                            "theta": list(campaign.theta),
                            "theta_id": f"rho{int(campaign.theta[0]*100)}_share{int(campaign.theta[1]*100)}",
                            "initial_regime": campaign.initial_regime,
                            "arm": arm,
                            "initial_belief": beliefs[arm].as_dict(),
                            "calendar_before_common_continuation": list(raw_calendars[arm]),
                            "calendar": list(calendar),
                            "online_ms": float(diagnostics[arm].get("online_ms", 0.0)),
                            "skeleton_sha256": campaign.skeleton.skeleton_sha256,
                            "prefix_state_hash": campaign.skeleton.prefix_state_hash,
                            **metrics,
                        }
                    )
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "history": campaign.history_root,
                            "campaign": campaign.campaign_index,
                        }
                    ),
                    flush=True,
                )
    summary = summarize(rows)
    return {
        "schema_version": "q_r1_d1_demand_memory_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "histories_per_mode": args.histories,
        "campaigns_per_history": args.campaigns,
        "planner": config.config_id,
        "oracle_parameters_boundary": "knows theta but not Z_t or future demand; this is an information arm under the same stochastic MPC, not a mathematical value upper bound",
        "cold_start_decisions": 2,
        "common_continuation": "reset_exact_bayes_mpc weeks 3-8",
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=7_570_201)
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--mode", choices=("scenario", "robust", "constraint_aware"), default="scenario")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--hard-cap-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/q_r1/d1_demand_memory_v1/result.json",
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
