#!/usr/bin/env python3
"""Corrected causal gate: R_full - R_reset(pre-risk) with a proper calm action.

Motivation: the naive gate used by scripts/run_track_b_ruta_b_sidecar.py and
scripts/audit_ruta_b_gate_on_reactive_ppo.py defines "calm action" as the
median action over the WHOLE episode. Under Case C, R24 fires roughly every
~1.2 weeks (frequency x3), so almost every pre-event [-4,-1] window overlaps
with reactive carryover from a DIFFERENT, very recent R24 event. Swapping in
a contaminated "median" (not actually calm) removes that legitimate reactive
response and mechanically produces a positive delta for ANY sufficiently
reactive policy -- confirmed empirically: the reactive-only Case C PPO
baseline showed 79.7% positive rate under the naive gate, higher than Ruta B
itself (74.1%), which should be impossible if the gate were measuring
anticipation specifically.

This script ports the calm-action logic that scripts/audit_track_b_risk_event_
counterfactual.py already used (and that gave a clean null on the same
reactive baseline): calm action = mean action over steps OUTSIDE a (-4, +8)
exclusion halo around every target-risk event, pooled across all evaluation
episodes for that policy. If very frequent risks make that candidate set
empty, fall back to the policy's own lowest action-intensity quartile
(the policy's calmest observed moments, not its typical/median ones).

Runs the identical corrected gate on both a reactive PPO checkpoint (plain
PPO, no aux head) and a Ruta B checkpoint (RutaBAuxPPO + RutaBAuxFeaturesExtractor),
so the comparison is apples-to-apples on the corrected methodology too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import build_env_kwargs, extract_downstream_multipliers, save_csv  # noqa: E402
from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS  # noqa: E402
from scripts.run_track_b_ruta_b_sidecar import (  # noqa: E402
    EVAL_EPISODE_SEED_OFFSET,
    VecNormalizeKeepLastRaw,  # noqa: F401  (rebinds under __main__ so pickle.load can resolve it)
    build_args,
)
from scripts.ruta_b_aux_ppo import RutaBAuxPPO  # noqa: E402
from scripts.ruta_b_risk_label_wrapper import ConstantLabelPadWrapper  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

CALM_EXCLUSION_WINDOW = (-4, 8)
RESET_WINDOW = (-4, -1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument(
        "--policy-kind",
        choices=["reactive_ppo", "ruta_b"],
        required=True,
        help="reactive_ppo: plain PPO.load(). ruta_b: RutaBAuxPPO with the aux extractor + label pad.",
    )
    p.add_argument("--model-filename", default="ppo_model.zip")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--obs-config", default="v7_no_forecast", choices=list(OBS_ABLATION_CONFIGS.keys()))
    p.add_argument("--target-risks", nargs="+", default=["R22", "R24"])
    p.add_argument("--max-events-per-risk-episode", type=int, default=8)
    return p


def make_env(args: argparse.Namespace, obs_config: str, ruta_b: bool):
    env = make_track_b_env(**build_env_kwargs(args))
    wrapper = OBS_ABLATION_CONFIGS[str(obs_config)].wrapper
    if wrapper is not None:
        env = wrapper(env)
    if ruta_b:
        env = ConstantLabelPadWrapper(env)
    return env


def load_policy(cli: argparse.Namespace, seed: int, sample_env):
    if cli.policy_kind == "ruta_b":
        seed_dir = cli.checkpoint_root / f"seed{seed}"
    else:
        seed_dir = cli.checkpoint_root / "models" / f"seed{seed}"
    if cli.policy_kind == "ruta_b":
        model = RutaBAuxPPO.load(str(seed_dir / cli.model_filename), device="cpu")
    else:
        model = PPO.load(str(seed_dir / cli.model_filename), device="cpu")
    vec_norm = VecNormalize.load(str(seed_dir / "vec_normalize.pkl"), DummyVecEnv([lambda: sample_env]))
    vec_norm.training = False
    return model, vec_norm


def predict_action(model, vec_norm: VecNormalize, obs: np.ndarray) -> np.ndarray:
    obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = model.predict(obs_norm, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def action_intensity_from_info(info: dict[str, Any]) -> float:
    shift = int(info.get("shifts_active", 1))
    shift_norm = (float(shift) - 1.0) / 2.0
    op10_mult, op12_mult = extract_downstream_multipliers(info)
    op10_norm = np.clip((float(op10_mult) - 1.25) / 0.75, 0.0, 1.0)
    op12_norm = np.clip((float(op12_mult) - 1.25) / 0.75, 0.0, 1.0)
    return float(np.mean([shift_norm, op10_norm, op12_norm]))


def run_full_episode(model, vec_norm: VecNormalize, args, cli, eval_seed: int, ruta_b: bool):
    env = make_env(args, cli.obs_config, ruta_b)
    obs, _info = env.reset(seed=eval_seed)
    terminated = truncated = False
    step_actions: list[np.ndarray] = []
    step_intensity: list[float] = []
    while not (terminated or truncated):
        action = predict_action(model, vec_norm, obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        step_actions.append(action)
        step_intensity.append(action_intensity_from_info(info))
    metrics = compute_episode_metrics(env.unwrapped.sim)
    risk_events = list(env.unwrapped.sim.risk_events)
    env.close()
    return {
        "ret_excel": float(metrics["ret_excel"]),
        "step_actions": step_actions,
        "step_intensity": step_intensity,
        "risk_events": risk_events,
    }


def compute_calm_action(traces: list[dict[str, Any]], target_risks: set[str], step_size_hours: float, max_steps: int) -> np.ndarray:
    """Pooled across all episodes for one policy/seed, matching
    audit_track_b_risk_event_counterfactual.py's compute_calm_actions."""
    excluded_by_episode: list[set[int]] = []
    for trace in traces:
        excluded: set[int] = set()
        for ev in trace["risk_events"]:
            if str(ev.risk_id) not in target_risks:
                continue
            anchor = int(float(ev.start_time) // step_size_hours)
            lo, hi = CALM_EXCLUSION_WINDOW
            for step in range(anchor + lo, anchor + hi + 1):
                if 0 <= step < max_steps:
                    excluded.add(step)
        excluded_by_episode.append(excluded)

    candidate_actions: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_intensity: list[float] = []
    for trace, excluded in zip(traces, excluded_by_episode):
        for step, (action, intensity) in enumerate(zip(trace["step_actions"], trace["step_intensity"])):
            all_actions.append(action)
            all_intensity.append(intensity)
            if step not in excluded:
                candidate_actions.append(action)

    if candidate_actions:
        source = np.stack(candidate_actions)
        source_name = "outside_target_risk_neighborhood"
    else:
        ranked_idx = np.argsort(all_intensity)
        n_quartile = max(1, int(np.ceil(0.25 * len(ranked_idx))))
        source = np.stack([all_actions[i] for i in ranked_idx[:n_quartile]])
        source_name = "own_lowest_action_intensity_quartile"
    return np.mean(source, axis=0), source_name, len(candidate_actions), len(all_actions)


def run_counterfactual_for_seed(model, vec_norm, args, cli, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_risks = set(cli.target_risks)
    ruta_b = cli.policy_kind == "ruta_b"

    traces = []
    for episode_idx in range(cli.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        traces.append(run_full_episode(model, vec_norm, args, cli, eval_seed, ruta_b))

    calm_action, calm_source, n_candidate, n_total = compute_calm_action(
        traces, target_risks, float(args.step_size_hours), cli.max_steps
    )

    rows: list[dict[str, Any]] = []
    for episode_idx, trace in enumerate(traces):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        ret_full = trace["ret_excel"]
        anchors = [
            int(float(ev.start_time) // float(args.step_size_hours))
            for ev in trace["risk_events"]
            if str(ev.risk_id) in target_risks
        ]
        anchors = anchors[: cli.max_events_per_risk_episode]

        for anchor_step in anchors:
            lo, hi = RESET_WINDOW
            reset_steps = {s for s in range(anchor_step + lo, anchor_step + hi + 1) if 0 <= s < cli.max_steps}
            if not reset_steps:
                continue
            env = make_env(args, cli.obs_config, ruta_b)
            obs, _info = env.reset(seed=eval_seed)
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                if step in reset_steps:
                    action = calm_action
                else:
                    action = predict_action(model, vec_norm, obs)
                obs, _reward, terminated, truncated, _info = env.step(action)
                step += 1
            metrics_reset = compute_episode_metrics(env.unwrapped.sim)
            env.close()
            ret_reset = float(metrics_reset["ret_excel"])
            rows.append(
                {
                    "seed": seed,
                    "episode": episode_idx + 1,
                    "eval_seed": eval_seed,
                    "anchor_step": anchor_step,
                    "ret_full": ret_full,
                    "ret_reset": ret_reset,
                    "delta_ret_excel": ret_full - ret_reset,
                }
            )
    calm_meta = {
        "calm_source": calm_source,
        "n_calm_candidate_steps": n_candidate,
        "n_total_steps": n_total,
        "calm_action": calm_action.tolist(),
    }
    return rows, calm_meta


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)

    obs_config = OBS_ABLATION_CONFIGS[str(cli.obs_config)]
    args = build_args(
        {
            "max_steps": cli.max_steps,
            "eval_episodes": cli.eval_episodes,
            "observation_version": obs_config.observation_version,
        }
    )

    all_rows: list[dict[str, Any]] = []
    calm_meta_by_seed: dict[str, Any] = {}
    for seed in cli.seeds:
        print(f"=== seed {seed}: corrected counterfactual ({cli.policy_kind}) ===", flush=True)
        sample_env = make_env(args, cli.obs_config, cli.policy_kind == "ruta_b")
        model, vec_norm = load_policy(cli, seed, sample_env)
        rows, calm_meta = run_counterfactual_for_seed(model, vec_norm, args, cli, seed)
        all_rows.extend(rows)
        calm_meta_by_seed[str(seed)] = calm_meta

    save_csv(out / "counterfactual_rows.csv", all_rows)

    n_pairs = len(all_rows)
    n_positive = sum(1 for r in all_rows if r["delta_ret_excel"] > 0)
    mean_delta = float(np.mean([r["delta_ret_excel"] for r in all_rows])) if n_pairs else 0.0
    by_seed: dict[int, list[float]] = {}
    for r in all_rows:
        by_seed.setdefault(int(r["seed"]), []).append(float(r["delta_ret_excel"]))
    per_seed = {
        str(s): {
            "n_pairs": len(d),
            "n_positive": sum(1 for x in d if x > 0),
            "positive_rate": sum(1 for x in d if x > 0) / len(d),
            "mean_delta": float(np.mean(d)),
        }
        for s, d in sorted(by_seed.items())
    }
    summary = {
        "policy_kind": cli.policy_kind,
        "checkpoint_root": str(cli.checkpoint_root),
        "obs_config": str(cli.obs_config),
        "target_risks": list(cli.target_risks),
        "counterfactual_n_pairs": n_pairs,
        "counterfactual_n_positive": n_positive,
        "counterfactual_positive_rate": (n_positive / n_pairs) if n_pairs else 0.0,
        "counterfactual_mean_delta_ret_excel": mean_delta,
        "per_seed": per_seed,
        "calm_action_meta_by_seed": calm_meta_by_seed,
        "config": vars(cli),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "calm_action_meta_by_seed")}, indent=2))


if __name__ == "__main__":
    main()
