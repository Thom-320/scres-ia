#!/usr/bin/env python3
"""Trace saved Discrete(18) dynamic policies step-by-step.

This is a diagnostic companion to ``compare_garrido_dynamic_vs_static.py``.  It
loads saved PPO models from one or more comparison-run directories, evaluates
them on common seeds, and exports the weekly action trace.  The goal is to see
whether longer training destroys a promising signal by changing the timing or
mix of buffer/shift decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from stable_baselines3 import PPO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_garrido_dynamic_vs_static import (  # noqa: E402
    STATIC_BASELINES,
    build_env_kwargs,
    static_action,
)
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_discrete18_track_a_env,
)
from supply_chain.thesis_decision_env import Discrete18TrackAEnv  # noqa: E402


INV_PERIODS = [0, 168, 336, 504, 672, 1344]


def _label_action(action: int) -> str:
    decoded = Discrete18TrackAEnv.decode_discrete_action(int(action))
    inventory_level = int(decoded[0])
    shift_index = int(decoded[1])
    return f"S{shift_index + 1}_I{INV_PERIODS[inventory_level]}"


def _seed_from_model(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from model path: {path}")
    return int(match.group(1))


def _namespace_from_config(config: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "reward_mode": "ReT_garrido2024_raw",
        "observation_version": "v4",
        "stochastic_pt": False,
        "step_size_hours": 168.0,
        "max_steps": 52,
        "risk_occurrence_mode": "thesis_window",
        "risk_frequency_multiplier": 1.0,
        "risk_impact_multiplier": 1.0,
        "ret_g24_shift_cost": 1.0,
        "ret_g24_kappa_train_frac": 1.0,
        "w_bo": 4.0,
        "w_cost": 0.02,
        "w_disr": 0.0,
        "control_v2_w_fill": 1.0,
        "control_v2_w_service": 4.0,
        "control_v2_w_lost": 2.0,
        "control_v2_w_inventory": 0.05,
        "control_v2_w_shift": 0.08,
        "control_v2_w_switch": 0.02,
        "raw_material_flow_mode": "kit_equivalent_order_up_to",
        "raw_material_order_up_to_multiplier": 2.0,
        "demand_on_hand_fulfillment_delay": 54.0,
    }
    merged = {**defaults, **config}
    return argparse.Namespace(**merged)


def _env_kwargs(config: dict[str, Any], regime: str) -> dict[str, Any]:
    args = _namespace_from_config(config)
    kwargs = build_env_kwargs(args, regime)
    policy_name = str(config.get("ppo_initial_static_policy") or "")
    if policy_name:
        level, shift_index = STATIC_BASELINES[policy_name]
        kwargs["initial_action"] = static_action(level, shift_index)
    return kwargs


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def trace_run(run_dir: Path, eval_seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    config = dict(summary["config"])
    regimes = list(config.get("regimes") or ["severe"])
    model_paths = sorted((run_dir / "models").glob("ppo_*_seed*.zip"))
    traces: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for model_path in model_paths:
        train_seed = _seed_from_model(model_path)
        # Current runs are one-regime, but keep the parser robust.
        regime = next((r for r in regimes if f"ppo_{r}_" in model_path.name), regimes[0])
        model = PPO.load(model_path)
        for eval_seed in eval_seeds:
            env = make_discrete18_track_a_env(**_env_kwargs(config, regime))
            obs, info = env.reset(seed=int(eval_seed))
            done = False
            step = 0
            actions: list[int] = []
            switches = 0
            previous_action: int | None = None
            cd_steps: list[float] = []
            reward_total = 0.0
            while not done:
                predicted, _state = model.predict(obs, deterministic=True)
                action = int(np.asarray(predicted).item())
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if previous_action is not None and action != previous_action:
                    switches += 1
                previous_action = action
                actions.append(action)
                cd_step = _safe_float(info.get("ret_garrido2024_sigmoid_step"))
                cd_steps.append(cd_step)
                reward_total += float(reward)
                decision = info.get("thesis_decision", {})
                traces.append(
                    {
                        "run": run_dir.name,
                        "model_seed": train_seed,
                        "eval_seed": int(eval_seed),
                        "regime": regime,
                        "step": step,
                        "action": action,
                        "action_label": _label_action(action),
                        "inventory_level": int(
                            decision.get("common_inventory_level", -1)
                        ),
                        "shift_level": int(decision.get("assembly_shifts", -1)),
                        "reward": float(reward),
                        "cd_sigmoid_step": cd_step,
                        "new_demanded": _safe_float(info.get("new_demanded")),
                        "new_delivered": _safe_float(info.get("new_delivered")),
                        "new_backorder_qty": _safe_float(
                            info.get("new_backorder_qty")
                        ),
                        "pending_backorder_qty": _safe_float(
                            info.get("pending_backorder_qty")
                        ),
                        "service_loss_step": _safe_float(
                            info.get("service_loss_step")
                        ),
                        "disruption_fraction_step": _safe_float(
                            info.get("disruption_fraction_step")
                        ),
                    }
                )
                step += 1
            terminal = get_episode_terminal_metrics(env)
            counts: dict[str, int] = {}
            for action in actions:
                counts[_label_action(action)] = counts.get(_label_action(action), 0) + 1
            episodes.append(
                {
                    "run": run_dir.name,
                    "model_seed": train_seed,
                    "eval_seed": int(eval_seed),
                    "regime": regime,
                    "steps": len(actions),
                    "reward_total": reward_total,
                    "cd_sigmoid_mean_trace": (
                        sum(cd_steps) / max(len(cd_steps), 1)
                    ),
                    "mean_ret_excel_formula": float(
                        terminal["order_level_ret_excel_formula_mean"]
                    ),
                    "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
                    "action_switches": switches,
                    "first_actions": " ".join(_label_action(a) for a in actions[:8]),
                    "last_actions": " ".join(_label_action(a) for a in actions[-8:]),
                    "action_counts": json.dumps(counts, sort_keys=True),
                }
            )
            env.close()
    return traces, episodes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, episodes: list[dict[str, Any]]) -> None:
    lines = [
        "# Dynamic Action Trace Comparison",
        "",
        "| run | model seed | eval seed | Excel ReT | C-D trace | switches | first actions | last actions |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in episodes:
        lines.append(
            "| {run} | {model_seed} | {eval_seed} | "
            "{mean_ret_excel_formula:.6g} | {cd_sigmoid_mean_trace:.6g} | "
            "{action_switches} | {first_actions} | {last_actions} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--eval-seeds", default="12001,12002,12003")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/action_traces"),
    )
    args = parser.parse_args()
    eval_seeds = [int(x) for x in args.eval_seeds.split(",") if x.strip()]
    traces: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        run_traces, run_episodes = trace_run(run_dir, eval_seeds)
        traces.extend(run_traces)
        episodes.extend(run_episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "action_traces.csv", traces)
    write_csv(args.output_dir / "episode_trace_summary.csv", episodes)
    write_report(args.output_dir / "action_trace_report.md", episodes)
    print(f"Wrote {args.output_dir / 'action_traces.csv'}")
    print(f"Wrote {args.output_dir / 'episode_trace_summary.csv'}")
    print(f"Wrote {args.output_dir / 'action_trace_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
