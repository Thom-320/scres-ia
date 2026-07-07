#!/usr/bin/env python3
"""Build a (full v10 observation, future-risk label) dataset for pretraining a
belief encoder (Ruta A).

Unlike ``audit_track_b_risk_belief_predictor.py``'s dataset (a curated subset of
~15 operational/forecast/regime fields), this captures the FULL v10 observation
vector (101 dims) at every step, so the pretrained encoder's input exactly
matches what the real Track B env will feed it during RL training/eval --
required for the encoder's learned weights to transfer correctly when
transplanted into PPO's ``features_extractor``.

Rollouts use random actions under ``observation_version=v10`` (fixed-RNG env,
default now) for the widest, least policy-biased state coverage. Risk timing is
policy-independent under the fixed RNG for the 7 valid discrete risks, so
random-action episodes give valid (state, future-risk) pairs just like any
other policy's rollout would.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_prevention_mechanism import env_kwargs, save_csv  # noqa: E402
from supply_chain.external_env_interface import get_observation_fields, make_track_b_env  # noqa: E402

FORECAST_FIELD_NAMES = ("risk_forecast_48h_norm", "risk_forecast_168h_norm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/track_b_v10_belief_pretrain_dataset_2026-07-04"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument("--episodes-per-seed", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument(
        "--enabled-risks",
        default=None,
        help="Optional comma-separated risk IDs to enable, e.g. R22,R23,R24.",
    )
    parser.add_argument(
        "--risk-frequency-by-id",
        default=None,
        help="Optional per-risk frequency multipliers, e.g. R24=3.",
    )
    parser.add_argument(
        "--risk-impact-by-id",
        default=None,
        help="Optional per-risk impact multipliers, e.g. R22=1.5,R23=1.5.",
    )
    parser.add_argument(
        "--faithful",
        action="store_true",
        help="Use the Garrido-faithful Track B environment protocol.",
    )
    parser.add_argument("--observation-version", default="v10")
    parser.add_argument(
        "--mask-forecast",
        action="store_true",
        help="Zero explicit risk forecast fields in x_v10.npy while retaining v10 memory fields.",
    )
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--target-risk", default="R24")
    parser.add_argument("--rng-seed", type=int, default=7)
    return parser.parse_args()


def future_label(event_starts: list[int], step: int, horizon: int) -> int:
    return int(any(step < s <= step + horizon for s in event_starts))


def run_episode(args: argparse.Namespace, eval_seed: int, action_rng: np.random.Generator) -> dict[str, Any]:
    env = make_track_b_env(**env_kwargs(args))
    obs, _info = env.reset(seed=eval_seed)
    terminated = truncated = False
    obs_rows: list[np.ndarray] = []
    step = 0
    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        obs_rows.append(obs_before)
        action = action_rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, _reward, terminated, truncated, _info = env.step(action)
        step += 1
    event_starts = [
        int(float(ev.start_time) // float(args.step_size_hours))
        for ev in env.unwrapped.sim.risk_events
        if str(ev.risk_id) == args.target_risk
    ]
    env.close()
    return {"obs_rows": obs_rows, "event_starts": event_starts}


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fields = list(get_observation_fields(args.observation_version))
    forecast_indices = [
        fields.index(name) for name in FORECAST_FIELD_NAMES if name in fields
    ]
    action_rng = np.random.default_rng(args.rng_seed)

    x_rows: list[list[float]] = []
    label_rows: list[dict[str, int]] = []
    meta_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        for episode in range(1, int(args.episodes_per_seed) + 1):
            eval_seed = seed * 1000 + episode
            result = run_episode(args, eval_seed, action_rng)
            obs_rows = result["obs_rows"]
            event_starts = result["event_starts"]
            n_steps = len(obs_rows)
            for step in range(n_steps):
                row = np.asarray(obs_rows[step], dtype=np.float32).copy()
                if args.mask_forecast:
                    for idx in forecast_indices:
                        row[idx] = 0.0
                x_rows.append(list(row))
                label_rows.append(
                    {
                        f"y_{args.target_risk}_{h}w": future_label(event_starts, step, h)
                        for h in args.horizons
                    }
                )
                meta_rows.append({"seed": seed, "episode": episode, "step": step, "eval_seed": eval_seed})

    x = np.asarray(x_rows, dtype=np.float32)
    np.save(out / "x_v10.npy", x)

    combined_rows = []
    for meta, label in zip(meta_rows, label_rows):
        row = dict(meta)
        row.update(label)
        combined_rows.append(row)
    save_csv(out / "labels.csv", combined_rows)

    (out / "observation_fields.txt").write_text("\n".join(fields), encoding="utf-8")
    (out / "dataset_config.json").write_text(
        json.dumps(
            {
                "target_risk": args.target_risk,
                "horizons": [int(h) for h in args.horizons],
                "risk_level": args.risk_level,
                "enabled_risks": args.enabled_risks,
                "risk_frequency_by_id": args.risk_frequency_by_id,
                "risk_impact_by_id": args.risk_impact_by_id,
                "faithful": bool(args.faithful),
                "observation_version": args.observation_version,
                "mask_forecast": bool(args.mask_forecast),
                "forecast_indices_zeroed": forecast_indices if args.mask_forecast else [],
                "seeds": [int(seed) for seed in args.seeds],
                "episodes_per_seed": int(args.episodes_per_seed),
                "max_steps": int(args.max_steps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_pos = {
        f"y_{args.target_risk}_{h}w": sum(int(r[f"y_{args.target_risk}_{h}w"]) for r in label_rows)
        for h in args.horizons
    }
    print(f"Wrote {x.shape[0]} rows x {x.shape[1]} dims to {out}")
    for h in args.horizons:
        key = f"y_{args.target_risk}_{h}w"
        print(f"  {key}: base_rate={n_pos[key] / len(label_rows):.4f} (n_pos={n_pos[key]}/{len(label_rows)})")


if __name__ == "__main__":
    main()
