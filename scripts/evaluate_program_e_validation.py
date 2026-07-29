#!/usr/bin/env python3
"""Evaluate frozen Program E policies and emit the validation promotion gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sb3_contrib import MaskablePPO  # noqa: E402

from scripts.fit_program_e_policy_tree import TREE_FEATURES  # noqa: E402
from scripts.run_dra2b_long_horizon_gate import evaluate_horizon  # noqa: E402
from supply_chain.dra2_convoy import ConvoyThresholdPolicy, static_policies  # noqa: E402
from supply_chain.dra2_experiment import make_sim, state_hash  # noqa: E402
from supply_chain.dra2_policy_env import (  # noqa: E402
    ObservableHeuristic, ProgramEConvoyEnv,
)


def run_policy(tape: dict, normalizers: dict, kind: str, policy: Any) -> dict[str, Any]:
    env = ProgramEConvoyEnv([tape], normalizers, episode_days=56, random_tapes=False)
    obs, _ = env.reset(options={"tape": tape})
    action_counts = [0, 0]
    while True:
        mask = env.action_masks()
        raw = env.raw_observation()
        if kind == "static":
            action = int(policy.action(raw) == "DISPATCH_NOW")
        elif kind == "heuristic":
            action = policy.action(raw, mask)
        elif kind == "tree":
            x = np.asarray([[
                np.clip(raw[key] / max(float(policy["scales"][key]), 1e-9), -10, 10)
                for key in policy["features"]
            ]])
            action = int(policy["model"].predict(x)[0])
            if not mask[action]:
                action = 0
        elif kind == "ppo":
            action, _ = policy.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)
        else:
            raise ValueError(kind)
        action_counts[action] += 1
        obs, _, done, _, info = env.step(action)
        if done:
            break
    env.close()
    return {
        "ret": float(info["ret_excel_visible"]),
        "service": float(info["service_loss_auc_ration_hours"]),
        "lost": float(info["lost_orders"]),
        "backlog": float(info["backorder_qty_final"]),
        "departures": float(info["episode_departures"]),
        "unavailable_hours": float(info["episode_unavailable_hours"]),
        "hold_actions": action_counts[0], "dispatch_actions": action_counts[1],
    }


def convex_mixture(static_summary: list[dict], departures: float, unavailable: float) -> dict:
    ret = np.asarray([row["mean_ret"] for row in static_summary], dtype=float)
    dep = np.asarray([row["mean_departures"] for row in static_summary], dtype=float)
    unav = np.asarray([row["mean_unavailable_hours"] for row in static_summary], dtype=float)
    result = linprog(
        -ret, A_ub=np.vstack([dep, unav]), b_ub=np.asarray([departures, unavailable]),
        A_eq=np.ones((1, len(ret))), b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * len(ret), method="highs",
    )
    if not result.success:
        raise RuntimeError(f"NO_CONVEX_STATIC_COMPARATOR: {result.message}")
    return {
        "weights": result.x,
        "mean_ret": float(ret @ result.x),
        "mean_departures": float(dep @ result.x),
        "mean_unavailable_hours": float(unav @ result.x),
    }


def bootstrap_two_way(matrix: np.ndarray, seed: int, n_boot: int) -> tuple[float, float, float]:
    matrix = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        rows = rng.integers(0, matrix.shape[0], matrix.shape[0])
        cols = rng.integers(0, matrix.shape[1], matrix.shape[1])
        values.append(float(matrix[np.ix_(rows, cols)].mean()))
    return float(matrix.mean()), float(np.quantile(values, .025)), float(np.quantile(values, .975))


def oracle_worker(payload: tuple[dict, float, float]) -> dict:
    tape, threshold, wait = payload
    sim, start = make_sim(tape)
    obs = sim.get_op8_convoy_observation()
    state = {
        "state_id": f"{tape['tape_id']}|episode_start",
        "tape_id": tape["tape_id"], "family": tape["family"],
        "relative_time": 0.0, "prefix_hash": state_hash(sim),
        "prefix_policy_id": f"threshold_{int(threshold)}__wait_{int(wait)}h",
        "prefix_inventory_threshold": threshold,
        "prefix_maximum_wait_hours": wait,
        **obs,
    }
    result = evaluate_horizon(tape, state, 14)
    return {
        "tape_id": tape["tape_id"], "family": tape["family"],
        "ret": float(result["best"]["long_ret"]),
        "service": float(result["best"]["long_service"]),
        "lost": float(result["best"]["long_lost"]),
        "departures": float(result["best"]["op8_convoy_departures"]),
        "unavailable_hours": float(result["best"]["op8_convoy_unavailable_hours"]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("results/program_e/data"))
    parser.add_argument("--ppo-dir", type=Path, default=Path("results/program_e/ppo"))
    parser.add_argument("--tree-dir", type=Path, default=Path("results/program_e/tree"))
    parser.add_argument("--static-training", type=Path, default=Path("results/program_e/static_training/verdict.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/program_e/validation"))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--oracle-workers", type=int, default=8)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    tapes = json.loads((args.data_dir / "validation_tapes.json").read_text())
    normalizers = json.loads((args.data_dir / "normalizers.json").read_text())
    tree = joblib.load(args.tree_dir / "policy_tree.joblib")

    static_rows = []
    for tape in tapes:
        for policy in static_policies():
            static_rows.append({
                "tape_id": tape["tape_id"], "family": tape["family"],
                "policy_id": policy.policy_id,
                **run_policy(tape, normalizers, "static", policy),
            })
    write_csv(args.output_dir / "static_rows.csv", static_rows)
    static_summary = []
    for policy in static_policies():
        selected = [row for row in static_rows if row["policy_id"] == policy.policy_id]
        static_summary.append({
            "policy_id": policy.policy_id,
            "mean_ret": float(np.mean([row["ret"] for row in selected])),
            "mean_departures": float(np.mean([row["departures"] for row in selected])),
            "mean_unavailable_hours": float(np.mean([row["unavailable_hours"] for row in selected])),
        })

    auxiliary_rows = []
    for kind, policy in (("heuristic", ObservableHeuristic()), ("tree", tree)):
        for tape in tapes:
            auxiliary_rows.append({
                "policy": kind, "tape_id": tape["tape_id"], "family": tape["family"],
                **run_policy(tape, normalizers, kind, policy),
            })
    write_csv(args.output_dir / "auxiliary_rows.csv", auxiliary_rows)

    ppo_rows = []
    seeds = list(range(9301, 9311))
    for seed in seeds:
        model = MaskablePPO.load(args.ppo_dir / f"maskable_ppo_seed_{seed}.zip", device="cpu")
        for tape in tapes:
            ppo_rows.append({
                "learner_seed": seed, "tape_id": tape["tape_id"], "family": tape["family"],
                **run_policy(tape, normalizers, "ppo", model),
            })
        print(f"[program-e-validation] PPO seed={seed} complete", flush=True)
    write_csv(args.output_dir / "ppo_rows.csv", ppo_rows)

    training_best = json.loads(args.static_training.read_text())["best_admissible"]
    oracle_payloads = [
        (tape, float(training_best["inventory_threshold"]),
         float(training_best["maximum_wait_hours"])) for tape in tapes
    ]
    oracle_rows = []
    with ProcessPoolExecutor(max_workers=args.oracle_workers) as pool:
        futures = [pool.submit(oracle_worker, payload) for payload in oracle_payloads]
        for future in as_completed(futures):
            oracle_rows.append(future.result())
    oracle_rows.sort(key=lambda row: row["tape_id"])
    write_csv(args.output_dir / "oracle_rows.csv", oracle_rows)

    tape_ids = [tape["tape_id"] for tape in tapes]
    static_index = {(row["policy_id"], row["tape_id"]): row for row in static_rows}
    ppo_index = {(row["learner_seed"], row["tape_id"]): row for row in ppo_rows}
    mixtures = []
    delta_ret = np.zeros((len(seeds), len(tapes)))
    service_reduction = np.zeros_like(delta_ret)
    lost_delta = np.zeros_like(delta_ret)
    baseline_ret = np.zeros_like(delta_ret)
    candidate_ret = np.zeros_like(delta_ret)
    for i, seed in enumerate(seeds):
        selected = [row for row in ppo_rows if row["learner_seed"] == seed]
        dep = float(np.mean([row["departures"] for row in selected]))
        unav = float(np.mean([row["unavailable_hours"] for row in selected]))
        mix = convex_mixture(static_summary, dep, unav)
        mixtures.append({
            "learner_seed": seed,
            "weights": {static_summary[j]["policy_id"]: float(value)
                        for j, value in enumerate(mix["weights"]) if value > 1e-10},
            **{key: value for key, value in mix.items() if key != "weights"},
            "candidate_mean_departures": dep,
            "candidate_mean_unavailable_hours": unav,
        })
        for j, tape_id in enumerate(tape_ids):
            candidate = ppo_index[(seed, tape_id)]
            base = {
                metric: sum(
                    float(mix["weights"][k])
                    * float(static_index[(static_summary[k]["policy_id"], tape_id)][metric])
                    for k in range(len(static_summary))
                ) for metric in ("ret", "service", "lost")
            }
            candidate_ret[i, j] = candidate["ret"]; baseline_ret[i, j] = base["ret"]
            delta_ret[i, j] = candidate["ret"] - base["ret"]
            service_reduction[i, j] = (base["service"] - candidate["service"]) / max(abs(base["service"]), 1.0)
            lost_delta[i, j] = candidate["lost"] - base["lost"]
    (args.output_dir / "convex_mixtures.json").write_text(
        json.dumps(mixtures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    oracle_candidate = {
        "departures": float(np.mean([row["departures"] for row in oracle_rows])),
        "unavailable": float(np.mean([row["unavailable_hours"] for row in oracle_rows])),
    }
    oracle_mix = convex_mixture(static_summary, oracle_candidate["departures"], oracle_candidate["unavailable"])
    oracle_delta = []
    for row in oracle_rows:
        base = sum(
            float(oracle_mix["weights"][k])
            * float(static_index[(static_summary[k]["policy_id"], row["tape_id"])]["ret"])
            for k in range(len(static_summary))
        )
        oracle_delta.append(float(row["ret"]) - base)
    oracle_headroom = float(np.mean(oracle_delta))

    ret_ci = bootstrap_two_way(delta_ret, 20260730, args.n_boot)
    service_ci = bootstrap_two_way(service_reduction, 20260731, args.n_boot)
    lost_ci = bootstrap_two_way(lost_delta, 20260732, args.n_boot)
    eta = ret_ci[0] / max(oracle_headroom, 1e-12)
    positive_seeds = int(sum(delta_ret[i].mean() > 0 for i in range(len(seeds))))
    favorable_tapes = float(np.mean(delta_ret.mean(axis=0) > 0))
    candidate_tail = float(np.mean(np.sort(candidate_ret.ravel())[:max(1, int(.1 * candidate_ret.size))]))
    baseline_tail = float(np.mean(np.sort(baseline_ret.ravel())[:max(1, int(.1 * baseline_ret.size))]))
    tail_pass = candidate_tail >= baseline_tail - 1e-9
    gates = {
        "ret": ret_ci[0] >= .01 and ret_ci[1] > 0,
        "conversion": eta >= .50,
        "service_noninferiority": service_ci[1] >= 0,
        "lost_orders": lost_ci[2] <= 0,
        "resource_envelope": all(
            row["mean_departures"] <= row["candidate_mean_departures"] + 1e-9
            and row["mean_unavailable_hours"] <= row["candidate_mean_unavailable_hours"] + 1e-9
            for row in mixtures
        ),
        "positive_seeds": positive_seeds >= 8,
        "favorable_tapes": favorable_tapes >= .70,
        "tail": tail_pass,
    }
    promote = all(gates.values())
    verdict = {
        "gate": "PROGRAM_E_VALIDATION_GATE",
        "n_learner_seeds": len(seeds), "n_validation_tapes": len(tapes),
        "ret_delta_mean_ci95": ret_ci,
        "service_loss_reduction_mean_ci95": service_ci,
        "lost_orders_delta_mean_ci95": lost_ci,
        "restricted_oracle_headroom": oracle_headroom,
        "conversion_efficiency": eta,
        "positive_learner_seeds": positive_seeds,
        "favorable_tape_fraction": favorable_tapes,
        "candidate_cvar10_ret": candidate_tail,
        "baseline_cvar10_ret": baseline_tail,
        "gates": gates,
        "virgin_test_authorized": promote,
        "virgin_tapes_opened": 0,
        "interpretation": "PROMOTE_PROGRAM_E_TO_VIRGIN_TEST" if promote else "STOP_PROGRAM_E_VALIDATION",
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if promote else 2


if __name__ == "__main__":
    raise SystemExit(main())
