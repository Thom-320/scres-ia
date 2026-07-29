#!/usr/bin/env python3
"""Reward gate on the static Track-A surface, before any retained/reset learning.

This script evaluates candidate training rewards by holding each of the 18
Track-A actions fixed.  It then checks whether total reward is monotone with
the paper-facing metrics on training tapes.  It is intentionally static-only:
no PPO/DQN outcome is used to select a reward.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics, merge_resource_metrics  # noqa: E402
from supply_chain.thesis_decision_env import Discrete18TrackAEnv  # noqa: E402
from supply_chain.external_env_interface import make_discrete18_track_a_env  # noqa: E402


PERIODS = [0, 168, 336, 504, 672, 1344]
SHIFT_INDICES = [0, 1, 2]


PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "faithful": {
        "risk_frequency_multiplier": 1.0,
        "risk_impact_multiplier": 1.0,
        "stochastic_pt": False,
        "risk_level": "severe",
    },
    "envb_aggr_g24_raw": {
        "risk_frequency_multiplier": 2.0,
        "risk_impact_multiplier": 1.5,
        "stochastic_pt": False,
        "risk_level": "severe",
    },
    "envb_frontier_v2": {
        "risk_frequency_multiplier": 1.0,
        "risk_impact_multiplier": 1.5,
        "stochastic_pt": False,
        "risk_level": "severe",
    },
    "envb_cons_control_v2": {
        "risk_frequency_multiplier": 1.0,
        "risk_impact_multiplier": 1.25,
        "stochastic_pt": False,
        "risk_level": "severe",
    },
}


CONTROL_V2_SERVICE_HEAVY = {
    "control_v2_w_fill": 1.2,
    "control_v2_w_service": 6.0,
    "control_v2_w_lost": 4.0,
    "control_v2_w_inventory": 0.04,
    "control_v2_w_shift": 0.06,
    "control_v2_w_switch": 0.01,
}


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _ints(value: str) -> list[int]:
    return [int(part) for part in _csv(value)]


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    rx = _rank([p[0] for p in pairs])
    ry = _rank([p[1] for p in pairs])
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry, strict=False))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    return float(num / (denx * deny)) if denx > 0 and deny > 0 else float("nan")


def action_label(action: int) -> str:
    factorized = Discrete18TrackAEnv.decode_discrete_action(action)
    level = int(factorized[0])
    shift = int(factorized[1]) + 1
    return f"S{shift}_I{PERIODS[level]}"


def env_kwargs(profile: str, reward_mode: str, max_steps: int) -> dict[str, Any]:
    profile_cfg = dict(PROFILE_CONFIGS[profile])
    kwargs = {
        "reward_mode": reward_mode,
        "observation_version": "v5",
        "risk_level": profile_cfg.pop("risk_level"),
        "downstream_q_source": "figure_6_2",
        "step_size_hours": 168.0,
        "max_steps": max_steps,
        "risk_occurrence_mode": "thesis_window",
        "year_basis": P["year_basis"],
        "warmup_trigger": P["warmup_trigger"],
        "r14_defect_mode": P["r14_defect_mode"],
        "raw_material_flow_mode": P["raw_material_flow_mode"],
        "raw_material_order_up_to_multiplier": P["raw_material_order_up_to_multiplier"],
        **profile_cfg,
    }
    if reward_mode == "control_v2":
        kwargs.update(CONTROL_V2_SERVICE_HEAVY)
    return kwargs


def run_policy(*, profile: str, reward_mode: str, action: int, seed: int, max_steps: int) -> dict[str, Any]:
    env = make_discrete18_track_a_env(**env_kwargs(profile, reward_mode, max_steps))
    obs, _info = env.reset(seed=seed, options={"initial_discrete_action": action})
    done = False
    reward_total = 0.0
    steps = 0
    while not done:
        obs, reward, terminated, truncated, _info = env.step(action)
        reward_total += float(reward)
        steps += 1
        done = bool(terminated or truncated)
    sim = env.unwrapped.sim
    metrics = compute_episode_metrics(sim)
    shift = (action % 3) + 1
    level = action // 3
    strategic_buffer_units = 0.0
    if level > 0:
        from supply_chain.config import INVENTORY_BUFFERS

        strategic_buffer_units = float(sum(INVENTORY_BUFFERS[PERIODS[level]].values()))
    merged = merge_resource_metrics(
        metrics,
        shift_hours=float(shift * max_steps * 168.0),
        extra_shift_hours=float((shift - 1) * max_steps * 168.0),
        strategic_buffer_units=strategic_buffer_units,
    )
    env.close()
    return {
        "profile": profile,
        "reward_mode": reward_mode,
        "seed": seed,
        "action": action,
        "policy": action_label(action),
        "reward_total": reward_total,
        "steps": steps,
        **merged,
    }


def mean_by_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({k for row in rows for k in row if k not in {"seed"}})
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["profile"], row["reward_mode"], row["policy"]), []).append(row)
    out = []
    for (_profile, _reward, _policy), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "profile": group[0]["profile"],
            "reward_mode": group[0]["reward_mode"],
            "policy": group[0]["policy"],
            "action": group[0]["action"],
            "n_seeds": len(group),
        }
        for key in keys:
            vals = [float(row[key]) for row in group if isinstance(row.get(key), (int, float))]
            if vals:
                agg[key] = statistics.mean(vals)
        out.append(agg)
    return out


def score_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reward = [float(row["reward_total"]) for row in rows]
    ret_excel = [float(row["ret_excel"]) for row in rows]
    cd_sigmoid = [float(row.get("cd_sigmoid_mean", row.get("ret_garrido2024_sigmoid_mean", 0.0))) for row in rows]
    flow_fill = [float(row["flow_fill_rate"]) for row in rows]
    service_loss_auc = [-float(row["service_loss_auc_per_order"]) for row in rows]
    lost_rate = [-float(row["lost_rate"]) for row in rows]
    resource = [
        -(float(row["shift_hours"]) + float(row["strategic_buffer_units"]))
        for row in rows
    ]
    best_reward = max(rows, key=lambda row: float(row["reward_total"]))
    best_excel = max(rows, key=lambda row: float(row["ret_excel"]))
    return {
        "profile": rows[0]["profile"],
        "reward_mode": rows[0]["reward_mode"],
        "n_policies": len(rows),
        "spearman_reward_ret_excel": spearman(reward, ret_excel),
        "spearman_reward_cd_sigmoid": spearman(reward, cd_sigmoid),
        "spearman_reward_flow_fill": spearman(reward, flow_fill),
        "spearman_reward_neg_service_loss_auc": spearman(reward, service_loss_auc),
        "spearman_reward_neg_lost_rate": spearman(reward, lost_rate),
        "spearman_reward_neg_resource": spearman(reward, resource),
        "best_reward_policy": best_reward["policy"],
        "best_excel_policy": best_excel["policy"],
        "best_reward_ret_excel": float(best_reward["ret_excel"]),
        "best_excel_ret_excel": float(best_excel["ret_excel"]),
        "best_reward_excel_gap": float(best_reward["ret_excel"]) - float(best_excel["ret_excel"]),
        "passes_excel_monotonicity": spearman(reward, ret_excel) > 0.0,
        "passes_service_monotonicity": spearman(reward, service_loss_auc) > 0.0,
        "passes_reward_gate": (
            spearman(reward, ret_excel) > 0.0
            and spearman(reward, service_loss_auc) > 0.0
            and float(best_reward["ret_excel"]) >= float(best_excel["ret_excel"]) - 0.002
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="faithful,envb_aggr_g24_raw,envb_cons_control_v2")
    parser.add_argument("--reward-modes", default="control_v1,control_v2")
    parser.add_argument("--seeds", default="8301,8302,8303")
    parser.add_argument("--max-steps", type=int, default=52)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/reward_alignment_static_surface"),
    )
    args = parser.parse_args()

    profiles = _csv(args.profiles)
    reward_modes = _csv(args.reward_modes)
    seeds = _ints(args.seeds)
    out = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(profiles) * len(reward_modes) * len(seeds) * 18
    done = 0
    for profile in profiles:
        if profile not in PROFILE_CONFIGS:
            raise ValueError(f"Unknown profile {profile!r}; expected {sorted(PROFILE_CONFIGS)}")
        for reward_mode in reward_modes:
            for seed in seeds:
                for action in range(18):
                    done += 1
                    print(f"[{done}/{total}] {profile} {reward_mode} seed={seed} {action_label(action)}", flush=True)
                    rows.append(
                        run_policy(
                            profile=profile,
                            reward_mode=reward_mode,
                            action=action,
                            seed=seed,
                            max_steps=args.max_steps,
                        )
                    )

    policy_rows = mean_by_policy(rows)
    score_rows = []
    for profile in profiles:
        for reward_mode in reward_modes:
            subset = [
                row for row in policy_rows
                if row["profile"] == profile and row["reward_mode"] == reward_mode
            ]
            score_rows.append(score_group(subset))

    write_csv(out / "episode_policy_seed_rows.csv", rows)
    write_csv(out / "policy_mean_rows.csv", policy_rows)
    write_csv(out / "reward_gate_summary.csv", score_rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
        "reward_modes": reward_modes,
        "seeds": seeds,
        "max_steps": args.max_steps,
        "summary": score_rows,
        "artifacts": {
            "episode_policy_seed_rows": str(out / "episode_policy_seed_rows.csv"),
            "policy_mean_rows": str(out / "policy_mean_rows.csv"),
            "reward_gate_summary": str(out / "reward_gate_summary.csv"),
        },
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
