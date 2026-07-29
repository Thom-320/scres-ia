#!/usr/bin/env python3
"""Run Program L Gate 1: 18 static policies plus the observable heuristic.

Only calibration tapes are accepted.  This runner never trains a neural policy
and never reads fixed/virgin probe universes.  Every buffer/shift combination is
evaluated on the same seed-indexed CampaignTape (CRN).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium  # noqa: E402
import stable_baselines3  # noqa: E402

from supply_chain.l_program_env import (  # noqa: E402
    BUFFER_LEVELS,
    OBSERVATION_FIELDS,
    CampaignTape,
    GarridoLearningEnv,
    fit_fixed_normalizer,
    materialize_campaign_tape,
)


DEFAULT_OUTPUT = Path("results/headroom")
FAMILIES = ("R1", "R2", "mixed")
RISK_LEVELS = ("current", "increased")


def build_calibration_tapes(
    *, count: int, seed_base: int, horizon_weeks: int
) -> list[CampaignTape]:
    if count <= 0:
        raise ValueError("count must be positive.")
    tapes = []
    for index in range(count):
        seed = int(seed_base + index)
        family = FAMILIES[index % len(FAMILIES)]
        risk_level = RISK_LEVELS[(index // len(FAMILIES)) % len(RISK_LEVELS)]
        tapes.append(
            CampaignTape(
                campaign_id=f"l-cal-{index:03d}-{seed}",
                family=family,
                risk_level=risk_level,
                base_seed=seed,
                horizon_weeks=horizon_weeks,
                split="calibration",
            )
        )
    return tapes


def static_policy(shift: int) -> Callable[[GarridoLearningEnv, dict[str, Any]], int]:
    action = int(shift) - 1

    def choose(_env: GarridoLearningEnv, _info: dict[str, Any]) -> int:
        return action

    return choose


def observable_heuristic(
    env: GarridoLearningEnv, _info: dict[str, Any]
) -> int:
    raw = env.raw_observation()
    values = dict(zip(OBSERVATION_FIELDS, raw, strict=True))
    fill = float(values["rolling_fill_rate_4w"])
    age = float(values["oldest_backorder_age_hours"])
    pending = float(values["pending_backorder_qty"])
    recent_demand = max(1.0, float(values["previous_week_demanded"]))
    utilization = float(values["previous_week_capacity_utilization"])
    if fill < 0.90 or age > 336.0:
        return 2  # S3
    if fill < 0.97 or pending > recent_demand or utilization > 0.90:
        return 1  # S2
    return 0  # S1


def run_episode(
    *,
    tape: CampaignTape,
    buffer_level: int,
    policy_name: str,
    initial_shift: int,
    policy_shift: int | None,
    policy: Callable[[GarridoLearningEnv, dict[str, Any]], int],
    lambda_shift: float,
    observation_sink: list[np.ndarray] | None = None,
    reward_component_sink: list[tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    env = GarridoLearningEnv(
        max_steps=tape.horizon_weeks,
        buffer_level=buffer_level,
        lambda_shift=lambda_shift,
    )
    try:
        _obs, info = env.reset(
            seed=tape.base_seed,
            options={
                "campaign_tape": tape,
                "buffer_level": buffer_level,
                "initial_state_seed": tape.base_seed,
                "initial_shift": initial_shift,
            },
        )
        terminated = truncated = False
        while not (terminated or truncated):
            if observation_sink is not None:
                observation_sink.append(env.raw_observation())
            action = int(policy(env, info))
            _obs, _reward, terminated, truncated, info = env.step(action)
            if reward_component_sink is not None:
                comp = info["l_program_reward_components"]
                reward_component_sink.append(
                    (
                        float(comp["late_backlog_hours"]),
                        float(comp["total_backlog_hours"]),
                        float(comp["extra_shift_hours"]),
                    )
                )
        metrics = env.terminal_metrics()
        if "ret_excel" not in metrics:
            raise RuntimeError("Gate 1 requires ret_excel; no fallback metric is allowed.")
        return {
            "contract_id": "garrido_learning_v1",
            "split": tape.split,
            "campaign_id": tape.campaign_id,
            "campaign_sha256": tape.digest(),
            "seed": tape.base_seed,
            "family": tape.family,
            "risk_level": tape.risk_level,
            "horizon_weeks": tape.horizon_weeks,
            "policy": policy_name,
            "buffer_level": buffer_level,
            "initial_shift": initial_shift,
            "policy_shift": int(policy_shift or initial_shift),
            **metrics,
        }
    finally:
        env.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), int(row["buffer_level"]), int(row["policy_shift"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (policy, buffer_level, shift), bucket in sorted(grouped.items()):
        output.append(
            {
                "policy": policy,
                "buffer_level": buffer_level,
                "initial_shift": 1,
                "policy_shift": shift,
                "n_tapes": len(bucket),
                "ret_excel_mean": float(np.mean([row["ret_excel"] for row in bucket])),
                "service_loss_auc_mean": float(
                    np.mean([row["service_loss_auc_ration_hours"] for row in bucket])
                ),
                "extra_shift_hours_mean": float(
                    np.mean([row["surge_hours"] for row in bucket])
                ),
                "switches_mean": float(np.mean([row["switches"] for row in bucket])),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=700_000)
    parser.add_argument("--horizon-weeks", type=int, default=104)
    parser.add_argument("--lambda-shift", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-heuristic", action="store_true", help="Diagnostic smoke only."
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tape_specs = build_calibration_tapes(
        count=args.tapes,
        seed_base=args.seed_base,
        horizon_weeks=args.horizon_weeks,
    )
    tapes: list[CampaignTape] = []
    for tape_index, tape in enumerate(tape_specs, start=1):
        print(
            f"[gate1] materialize risk tape {tape_index}/{len(tape_specs)} "
            f"{tape.family}/{tape.risk_level}",
            flush=True,
        )
        tapes.append(materialize_campaign_tape(tape))
    rows: list[dict[str, Any]] = []
    calibration_observations: list[np.ndarray] = []
    reward_components: list[tuple[float, float, float]] = []
    for tape_index, tape in enumerate(tapes, start=1):
        print(
            f"[gate1] tape {tape_index}/{len(tapes)} {tape.family}/{tape.risk_level}",
            flush=True,
        )
        for buffer_level in BUFFER_LEVELS:
            for shift in (1, 2, 3):
                rows.append(
                    run_episode(
                        tape=tape,
                        buffer_level=buffer_level,
                        policy_name=f"static_I{buffer_level}_S{shift}",
                        initial_shift=1,
                        policy_shift=shift,
                        policy=static_policy(shift),
                        lambda_shift=args.lambda_shift,
                        observation_sink=calibration_observations,
                        reward_component_sink=reward_components,
                    )
                )
            if not args.skip_heuristic:
                rows.append(
                    run_episode(
                        tape=tape,
                        buffer_level=buffer_level,
                        policy_name=f"heuristic_I{buffer_level}",
                        initial_shift=1,
                        policy_shift=None,
                        policy=observable_heuristic,
                        lambda_shift=args.lambda_shift,
                        observation_sink=calibration_observations,
                        reward_component_sink=reward_components,
                    )
                )

    panel_path = args.output_dir / "static_18_policy_panel.csv"
    write_csv(panel_path, rows)
    observations_path = args.output_dir / "calibration_observations.npz"
    observation_matrix = np.asarray(calibration_observations, dtype=np.float32)
    np.savez_compressed(
        observations_path,
        observations=observation_matrix,
        fields=np.asarray(OBSERVATION_FIELDS),
    )
    observations_sha = sha256(observations_path.read_bytes()).hexdigest()
    fixed_normalizer = fit_fixed_normalizer(
        observation_matrix,
        calibration_sha256=observations_sha,
    )
    (args.output_dir / "fixed_normalizer.json").write_text(
        json.dumps(fixed_normalizer.payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    reward_matrix = np.asarray(reward_components, dtype=np.float64)
    reward_scales = {
        "late_backlog_hours": float(max(1.0, np.quantile(reward_matrix[:, 0], 0.95))),
        "total_backlog_hours": float(max(1.0, np.quantile(reward_matrix[:, 1], 0.95))),
        "extra_shift_hours": float(max(1.0, np.quantile(reward_matrix[:, 2], 0.95))),
        "source": "Gate-1 calibration p95",
        "calibration_observations_sha256": observations_sha,
    }
    (args.output_dir / "reward_scales.json").write_text(
        json.dumps(reward_scales, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = aggregate(rows)
    write_csv(args.output_dir / "static_18_policy_summary.csv", summary)
    static_summary = [row for row in summary if str(row["policy"]).startswith("static_")]
    best_static = max(static_summary, key=lambda row: float(row["ret_excel_mean"]))
    best_any = max(summary, key=lambda row: float(row["ret_excel_mean"]))
    manifest = {
        "kind": "l_program_gate1_static_panel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": "garrido_learning_v1",
        "split": "calibration",
        "n_tapes": args.tapes,
        "seed_base": args.seed_base,
        "horizon_weeks": args.horizon_weeks,
        "lambda_shift": args.lambda_shift,
        "buffer_levels": list(BUFFER_LEVELS),
        "static_policy_count": 18,
        "includes_heuristic": not args.skip_heuristic,
        "best_static_by_unconstrained_mean_ret_excel": best_static,
        "best_any_by_unconstrained_mean_ret_excel": best_any,
        "tapes": [tape.payload(include_hash=True) for tape in tapes],
        "calibration_observations_sha256": observations_sha,
        "normalizer": fixed_normalizer.payload(),
        "reward_scales": reward_scales,
        "runtime": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": np.__version__,
        },
        "primary_metric": "ret_excel",
        "forbidden_metric_substitution": "order_level_ret_mean",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {"panel": str(panel_path), "best_static": best_static, "best_any": best_any},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
