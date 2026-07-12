#!/usr/bin/env python3
"""Ruta B: PPO+MLP with a live joint auxiliary belief loss on Track B.

Trains ``RutaBAuxPPO`` (scripts/ruta_b_aux_ppo.py) with
``RutaBAuxFeaturesExtractor`` (scripts/ruta_b_aux_extractor.py) under
``RutaBRiskLabelWrapper`` (scripts/ruta_b_risk_label_wrapper.py), on the
selected Case C environment by default (the strongest headroom scenario this
session: R22/R23/R24 only, R24 frequency x3, R22/R23 impact x1.5). Unlike Ruta A
(pretrain-then-transplant, already shown to fail the causal test), the
auxiliary belief-prediction task stays alive through the entire PPO training
run, directly targeting the diagnosed Ruta-A failure mode (the representation
gets repurposed into a generic risk-averse posture, not anticipation).

This script both trains and evaluates:

1. Static/heuristic comparators (reused unmodified from run_track_b_smoke.py).
2. The trained Ruta B policy's Garrido Excel ReT.
3. The R_full - R_reset(pre-risk) causal counterfactual on R22/R23/R24,
   using the same replay logic as
   scripts/audit_track_b_risk_event_counterfactual.py, so the causal question
   this whole session has been asking gets a direct answer for this new
   training method too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import (  # noqa: E402
    STATIC_POLICY_SPECS,
    build_env_kwargs,
    build_parser as smoke_build_parser,
    evaluate_static_policy,
    save_csv,
)
from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS  # noqa: E402
from scripts.ruta_b_aux_extractor import (  # noqa: E402
    RutaBAuxFeaturesExtractor,
    RutaBRealKANAuxFeaturesExtractor,
)
from scripts.ruta_b_aux_ppo import RutaBAuxPPO  # noqa: E402
from scripts.ruta_b_risk_label_wrapper import (  # noqa: E402
    ConstantLabelPadWrapper,
    RutaBRiskLabelWrapper,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

BASE_OVERRIDES: dict[str, Any] = {
    "risk_level": "current",
    "faithful": True,
    "observation_version": "v10",
    "reward_mode": "control_v1",
    "max_steps": 104,
}

SCENARIO_OVERRIDES: dict[str, dict[str, Any]] = {
    "case_a_all_risks": {
        **BASE_OVERRIDES,
    },
    "case_b_downstream": {
        **BASE_OVERRIDES,
        "enabled_risks": "R22,R23,R24",
    },
    "case_c_selected": {
        **BASE_OVERRIDES,
        "enabled_risks": "R22,R23,R24",
        "risk_frequency_by_id": "R24=3",
        "risk_impact_by_id": "R22=1.5,R23=1.5",
    },
}

CASE_C_OVERRIDES: dict[str, Any] = {
    **BASE_OVERRIDES,
    "enabled_risks": "R22,R23,R24",
    "risk_frequency_by_id": "R24=3",
    "risk_impact_by_id": "R22=1.5,R23=1.5",
}

EVAL_EPISODE_SEED_OFFSET = 50_000


class VecNormalizeKeepLastRaw(VecNormalize):
    """VecNormalize variant for Ruta B's appended binary label.

    The policy should see normalized real observations, but the auxiliary BCE
    target appended by ``RutaBRiskLabelWrapper`` must remain a raw 0/1 label.
    Standard ``VecNormalize(norm_obs=True)`` would normalize that final column
    too, turning the BCE target into an invalid drifting z-score.
    """

    def normalize_obs(self, obs: Any) -> Any:
        normalized = super().normalize_obs(obs)
        if isinstance(obs, np.ndarray) and isinstance(normalized, np.ndarray):
            normalized[..., -1] = obs[..., -1]
        return normalized


def build_args(cli_overrides: dict[str, Any], scenario_overrides: dict[str, Any] | None = None) -> argparse.Namespace:
    args = smoke_build_parser().parse_args([])
    for key, value in (scenario_overrides or CASE_C_OVERRIDES).items():
        setattr(args, key, value)
    for key, value in cli_overrides.items():
        setattr(args, key, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--train-timesteps", type=int, default=30_000)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--aux-coef", type=float, default=0.5)
    p.add_argument("--aux-lead-weeks", type=int, default=2)
    p.add_argument("--aux-target-risks", nargs="+", default=["R22", "R24"])
    p.add_argument(
        "--aux-label-mode",
        choices=["true", "permuted", "constant"],
        default="true",
        help=(
            "'permuted' shuffles the true label over time (same base rate) as a negative "
            "control; 'constant' replaces every step with the episode base rate (trivial "
            "gradient) for the efficiency-attribution ladder."
        ),
    )
    p.add_argument(
        "--obs-config",
        choices=list(OBS_ABLATION_CONFIGS.keys()),
        default="v10_no_forecast",
        help="Observation contract for Ruta B. Default keeps v10 memory and masks explicit forecasts.",
    )
    p.add_argument("--features-dim", type=int, default=64)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--architecture", choices=["mlp", "real_kan"], default="mlp")
    p.add_argument(
        "--clairvoyant",
        action="store_true",
        help=(
            "Preventive-headroom ceiling test: plain PPO (default extractor, no aux head) "
            "that SEES the true future-risk label in train AND eval. An upper bound on the "
            "value of anticipatory information under this action contract."
        ),
    )
    p.add_argument("--kan-grid", type=int, default=3)
    p.add_argument("--kan-k", type=int, default=3)
    p.add_argument("--kan-head-width", type=int, default=0)
    p.add_argument("--max-events-per-risk-episode", type=int, default=8)
    p.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_OVERRIDES.keys()),
        default="case_c_selected",
        help="Risk scenario for Ruta B. Defaults to the selected Case C cell.",
    )
    return p


def make_training_env(args: argparse.Namespace, cli: argparse.Namespace, seed: int):
    def _init():
        env = make_track_b_env(**build_env_kwargs(args))
        obs_wrapper = OBS_ABLATION_CONFIGS[str(cli.obs_config)].wrapper
        if obs_wrapper is not None:
            env = obs_wrapper(env)
        env = RutaBRiskLabelWrapper(
            env,
            target_risks=tuple(cli.aux_target_risks),
            lead_weeks=cli.aux_lead_weeks,
            step_size_hours=float(args.step_size_hours),
            label_mode=getattr(cli, "aux_label_mode", "true"),
        )
        return env

    return _init


def make_eval_env(args: argparse.Namespace):
    env = make_track_b_env(**build_env_kwargs(args))
    obs_config = getattr(args, "_ruta_b_obs_config", "v10_no_forecast")
    obs_wrapper = OBS_ABLATION_CONFIGS[str(obs_config)].wrapper
    if obs_wrapper is not None:
        env = obs_wrapper(env)
    if getattr(args, "_ruta_b_clairvoyant", False):
        # Clairvoyant ceiling test: the policy consumes the TRUE future-risk
        # label at eval time too (discovery pass per episode), instead of the
        # constant pad that is correct only when the label feeds an aux head.
        return RutaBRiskLabelWrapper(
            env,
            target_risks=tuple(getattr(args, "_ruta_b_target_risks", ("R22", "R24"))),
            lead_weeks=int(getattr(args, "_ruta_b_lead_weeks", 2)),
            step_size_hours=float(args.step_size_hours),
            label_mode="true",
        )
    return ConstantLabelPadWrapper(env)


def predict_action(model: RutaBAuxPPO, vec_norm: VecNormalize, obs: np.ndarray) -> np.ndarray:
    obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = model.predict(obs_norm, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def evaluate_ruta_b_policy(
    model: RutaBAuxPPO, vec_norm: VecNormalize, args: argparse.Namespace, seed: int, eval_episodes: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_idx in range(eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_eval_env(args)
        obs, _info = env.reset(seed=eval_seed)
        terminated = truncated = False
        shifts: list[int] = []
        while not (terminated or truncated):
            action = predict_action(model, vec_norm, obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            shifts.append(int(info.get("shifts_active", 1)))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        cost_index = float(sum(shifts) / (3.0 * len(shifts))) if shifts else 0.0
        rows.append(
            {
                "policy": "ruta_b_ppo",
                "seed": seed,
                "episode": episode_idx + 1,
                "eval_seed": eval_seed,
                "order_ret_excel": metrics["ret_excel"],
                "assembly_cost_index": cost_index,
            }
        )
        env.close()
    return rows


def run_counterfactual(
    model: RutaBAuxPPO,
    vec_norm: VecNormalize,
    args: argparse.Namespace,
    seed: int,
    eval_episodes: int,
    target_risks: tuple[str, ...],
    max_events_per_risk_episode: int,
) -> list[dict[str, Any]]:
    """R_full - R_reset(pre-risk), same design as
    scripts/audit_track_b_risk_event_counterfactual.py, inlined so it can use
    the padded-observation wrapper this policy needs."""
    rows: list[dict[str, Any]] = []
    for episode_idx in range(eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx

        # Full pass.
        env = make_eval_env(args)
        obs, _info = env.reset(seed=eval_seed)
        terminated = truncated = False
        steps_full: list[np.ndarray] = []
        while not (terminated or truncated):
            action = predict_action(model, vec_norm, obs)
            steps_full.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
        metrics_full = compute_episode_metrics(env.unwrapped.sim)
        risk_events = list(env.unwrapped.sim.risk_events)
        env.close()
        ret_full = float(metrics_full["ret_excel"])

        calm_action = np.median(np.stack(steps_full), axis=0) if steps_full else None

        anchors = [
            int(float(ev.start_time) // float(args.step_size_hours))
            for ev in risk_events
            if str(ev.risk_id) in target_risks
        ]
        anchors = anchors[:max_events_per_risk_episode]

        for anchor_step in anchors:
            reset_steps = {s for s in range(anchor_step - 4, anchor_step) if 0 <= s < args.max_steps}
            if not reset_steps or calm_action is None:
                continue
            env = make_eval_env(args)
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
    return rows


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)

    obs_config = OBS_ABLATION_CONFIGS[str(cli.obs_config)]
    scenario_overrides = dict(SCENARIO_OVERRIDES[str(cli.scenario)])
    args = build_args(
        {
            "max_steps": cli.max_steps,
            "eval_episodes": cli.eval_episodes,
            "observation_version": obs_config.observation_version,
            "_ruta_b_obs_config": str(cli.obs_config),
            "_ruta_b_clairvoyant": bool(getattr(cli, "clairvoyant", False)),
            "_ruta_b_target_risks": tuple(cli.aux_target_risks),
            "_ruta_b_lead_weeks": int(cli.aux_lead_weeks),
        },
        scenario_overrides=scenario_overrides,
    )
    args._observation_wrapper = obs_config.wrapper

    static_rows: list[dict[str, Any]] = []
    ruta_b_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []

    for seed in cli.seeds:
        print(f"=== seed {seed}: training Ruta B ===", flush=True)
        vec_env = DummyVecEnv([make_training_env(args, cli, seed)])
        vec_norm = VecNormalizeKeepLastRaw(
            vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0, gamma=cli.gamma
        )
        if getattr(cli, "clairvoyant", False):
            # Ceiling test: default extractor so the policy CONSUMES the label
            # column (RutaBAuxFeaturesExtractor would strip it off), plain PPO
            # loss (aux head irrelevant when the label is a visible feature).
            extractor_class = None
            extractor_kwargs = None
            net_arch = {"pi": [64, 64], "vf": [64, 64]}
        elif cli.architecture == "real_kan":
            extractor_class = RutaBRealKANAuxFeaturesExtractor
            extractor_kwargs = {
                "features_dim": cli.features_dim,
                "hidden_width": cli.hidden_width,
                "grid": cli.kan_grid,
                "k": cli.kan_k,
                "seed": seed,
            }
            head_width = int(cli.kan_head_width)
            net_arch = {"pi": [head_width], "vf": [head_width]} if head_width > 0 else {"pi": [], "vf": []}
        else:
            extractor_class = RutaBAuxFeaturesExtractor
            extractor_kwargs = {
                "features_dim": cli.features_dim,
                "hidden_width": cli.hidden_width,
            }
            net_arch = {"pi": [64, 64], "vf": [64, 64]}
        model = RutaBAuxPPO(
            "MlpPolicy",
            vec_norm,
            learning_rate=cli.learning_rate,
            n_steps=cli.n_steps,
            batch_size=cli.batch_size,
            n_epochs=cli.n_epochs,
            gamma=cli.gamma,
            gae_lambda=cli.gae_lambda,
            clip_range=cli.clip_range,
            ent_coef=cli.ent_coef,
            aux_coef=0.0 if getattr(cli, "clairvoyant", False) else cli.aux_coef,
            policy_kwargs=(
                {"net_arch": net_arch}
                if extractor_class is None
                else {
                    "net_arch": net_arch,
                    "features_extractor_class": extractor_class,
                    "features_extractor_kwargs": extractor_kwargs,
                }
            ),
            seed=seed,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=cli.train_timesteps)

        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.save(seed_dir / "ruta_b_model.zip")
        vec_norm.save(str(seed_dir / "vec_normalize.pkl"))

        print(f"=== seed {seed}: evaluating Ruta B policy ===", flush=True)
        ruta_b_rows.extend(evaluate_ruta_b_policy(model, vec_norm, args, seed, cli.eval_episodes))

        if seed == cli.seeds[0]:
            print(f"=== seed {seed}: evaluating static comparators ===", flush=True)
            for policy in STATIC_POLICY_SPECS:
                static_rows.extend(evaluate_static_policy(policy, args=args, seed=seed))

        print(f"=== seed {seed}: causal counterfactual R22/R23/R24 ===", flush=True)
        counterfactual_rows.extend(
            run_counterfactual(
                model,
                vec_norm,
                args,
                seed,
                cli.eval_episodes,
                tuple(cli.aux_target_risks),
                cli.max_events_per_risk_episode,
            )
        )

    save_csv(out / "ruta_b_episode_metrics.csv", ruta_b_rows)
    save_csv(out / "static_episode_metrics.csv", static_rows)
    save_csv(out / "counterfactual_rows.csv", counterfactual_rows)

    ruta_b_ret = float(np.mean([r["order_ret_excel"] for r in ruta_b_rows]))
    ruta_b_cost = float(np.mean([r["assembly_cost_index"] for r in ruta_b_rows]))
    static_by_policy: dict[str, list[float]] = {}
    for row in static_rows:
        static_by_policy.setdefault(row["policy"], []).append(float(row["order_ret_excel"]))
    static_means = {p: float(np.mean(v)) for p, v in static_by_policy.items()}
    best_static_policy = max(static_means, key=lambda p: static_means[p])
    best_static_ret = static_means[best_static_policy]

    n_pairs = len(counterfactual_rows)
    n_positive = sum(1 for r in counterfactual_rows if r["delta_ret_excel"] > 0)
    mean_delta = float(np.mean([r["delta_ret_excel"] for r in counterfactual_rows])) if n_pairs else 0.0

    summary = {
        "ruta_b_ret_excel_mean": ruta_b_ret,
        "ruta_b_cost_index_mean": ruta_b_cost,
        "best_static_policy": best_static_policy,
        "best_static_ret_excel_mean": best_static_ret,
        "delta_vs_best_static": ruta_b_ret - best_static_ret,
        "relative_delta_vs_best_static_pct": 100.0 * (ruta_b_ret - best_static_ret) / best_static_ret,
        "counterfactual_n_pairs": n_pairs,
        "counterfactual_n_positive": n_positive,
        "counterfactual_positive_rate": (n_positive / n_pairs) if n_pairs else 0.0,
        "counterfactual_mean_delta_ret_excel": mean_delta,
        "config": {
            **scenario_overrides,
            **vars(cli),
            "observation_version": obs_config.observation_version,
            "obs_config": str(cli.obs_config),
            "label_channel_raw_under_vecnormalize": True,
            "output_dir": str(out),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
