#!/usr/bin/env python3
"""Track B risk-belief sidecar.

This is the first cheap test of the "preventive belief" idea:

1. Train a small supervised risk-belief head from observed historical risk
   memory features, using the dataset built by
   ``audit_track_b_risk_belief_predictor.py``.
2. Freeze that head.
3. Append its probabilities to the Track B v10 observation during PPO
   training/evaluation.

The PPO reward and final evaluation remain unchanged: Garrido Excel ReT is
still reported by the standard Track B smoke bundle. This runner tests whether
an explicit learned belief signal helps the policy use the already-observed
risk memory, without modifying SB3's PPO loss yet.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.real_kan_extractor import RealKANFeaturesExtractor
import scripts.run_track_b_smoke as smoke


DEFAULT_DATASET = Path(
    "outputs/experiments/track_b_risk_belief_predictor_2026-07-04/risk_belief_dataset.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/experiments/track_b_risk_belief_sidecar_2026-07-04/smoke"
)
MEMORY_FEATURES = (
    "mem_weeks_since_last_R11",
    "mem_count_R11_8w",
    "mem_count_R11_26w",
    "mem_ewma_R11_8w",
    "mem_weeks_since_last_R13",
    "mem_count_R13_8w",
    "mem_count_R13_26w",
    "mem_ewma_R13_8w",
    "mem_weeks_since_last_R24",
    "mem_count_R24_8w",
    "mem_count_R24_26w",
    "mem_ewma_R24_8w",
)
DEFAULT_BELIEF_TARGETS = ("R24:1", "R24:2")


def build_parser() -> argparse.ArgumentParser:
    parser = smoke.build_parser()
    parser.description = "Track B PPO sidecar with frozen supervised risk-belief features."
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT,
        seeds=[1],
        train_timesteps=10_000,
        eval_episodes=2,
        reward_mode="control_v1",
        risk_level="adaptive_benchmark_v2",
        observation_version="v10",
        max_steps=104,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        n_epochs=10,
        algo="ppo",
    )
    parser.add_argument(
        "--architecture",
        choices=("ppo_mlp", "real_kan"),
        default="ppo_mlp",
        help="Policy architecture to train with the same frozen belief inputs.",
    )
    parser.add_argument(
        "--belief-dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV produced by audit_track_b_risk_belief_predictor.py.",
    )
    parser.add_argument(
        "--belief-targets",
        nargs="+",
        default=list(DEFAULT_BELIEF_TARGETS),
        help="Belief targets to append, formatted as RISK:HORIZON_WEEKS, e.g. R24:1.",
    )
    parser.add_argument(
        "--belief-class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help=(
            "Class weighting for the frozen logistic belief heads. 'balanced' "
            "matches the Sprint 1 discrimination screen; 'none' preserves "
            "probability calibration better when base rates matter."
        ),
    )
    parser.add_argument("--kan-features-dim", type=int, default=32)
    parser.add_argument("--kan-hidden-width", type=int, default=32)
    parser.add_argument("--kan-grid", type=int, default=3)
    parser.add_argument("--kan-k", type=int, default=3)
    parser.add_argument(
        "--kan-head-width",
        type=int,
        default=0,
        help="Optional MLP head width after the KAN extractor. 0 uses linear heads.",
    )
    return parser


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_target(target: str) -> tuple[str, int]:
    try:
        risk, horizon = target.split(":", 1)
        horizon_i = int(horizon)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid belief target {target!r}; expected RISK:HORIZON_WEEKS."
        ) from exc
    return risk.strip(), horizon_i


def train_belief_models(
    dataset_path: Path,
    targets: list[tuple[str, int]],
    *,
    class_weight: str,
) -> tuple[list[Pipeline], dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Belief dataset not found: {dataset_path}")
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Belief dataset is empty: {dataset_path}")

    x = np.asarray(
        [[_f(row, feature) for feature in MEMORY_FEATURES] for row in rows],
        dtype=np.float32,
    )
    models: list[Pipeline] = []
    target_meta: list[dict[str, Any]] = []
    for risk, horizon in targets:
        label = f"y_{risk}_{horizon}w"
        if label not in rows[0]:
            raise KeyError(f"Belief label {label!r} not found in {dataset_path}")
        y = np.asarray([int(float(row[label])) for row in rows], dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError(f"Belief label {label!r} has a single class.")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight=None if class_weight == "none" else "balanced",
                solver="lbfgs",
            ),
        )
        model.fit(x, y)
        prob = model.predict_proba(x)[:, 1]
        target_meta.append(
            {
                "risk_id": risk,
                "horizon_weeks": horizon,
                "label": label,
                "train_rows": len(rows),
                "base_rate": float(np.mean(y)),
                "mean_predicted_probability": float(np.mean(prob)),
            }
        )
        models.append(model)
    return models, {
        "dataset_path": str(dataset_path),
        "class_weight": class_weight,
        "memory_features": list(MEMORY_FEATURES),
        "targets": target_meta,
    }


class RiskBeliefAppendWrapper(gym.ObservationWrapper):
    """Append frozen risk-belief probabilities to v10 observations."""

    def __init__(
        self,
        env: gym.Env[np.ndarray, np.ndarray],
        *,
        models: list[Pipeline],
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("RiskBeliefAppendWrapper requires Box observations.")
        if int(env.observation_space.shape[0]) < len(MEMORY_FEATURES):
            raise ValueError("Observation is too small to contain v10 risk-memory features.")
        self._models = models
        self._memory_slice = slice(-len(MEMORY_FEATURES), None)
        low = np.concatenate(
            [env.observation_space.low.reshape(-1), np.zeros(len(models), dtype=np.float32)]
        )
        high = np.concatenate(
            [env.observation_space.high.reshape(-1), np.ones(len(models), dtype=np.float32)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        memory = obs[self._memory_slice].reshape(1, -1)
        belief = np.asarray(
            [float(model.predict_proba(memory)[0, 1]) for model in self._models],
            dtype=np.float32,
        )
        return np.concatenate([obs, belief]).astype(np.float32)


def train_ppo_mlp_belief(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    n_envs = max(1, int(getattr(args, "n_envs", 1)))
    vec_env = DummyVecEnv(
        [smoke.make_monitored_training_env(args, seed + i) for i in range(n_envs)]
    )
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model: Any = PPO(
        "MlpPolicy",
        vec_norm,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=int(args.train_timesteps))
    model.save(run_dir / "ppo_mlp_belief_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def train_ppo_real_kan_belief(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    n_envs = max(1, int(getattr(args, "n_envs", 1)))
    vec_env = DummyVecEnv(
        [smoke.make_monitored_training_env(args, seed + i) for i in range(n_envs)]
    )
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    head_width = int(getattr(args, "kan_head_width", 0))
    net_arch: dict[str, list[int]] = (
        {"pi": [head_width], "vf": [head_width]} if head_width > 0 else {"pi": [], "vf": []}
    )
    policy_kwargs = {
        "features_extractor_class": RealKANFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(args.kan_features_dim),
            "hidden_width": int(args.kan_hidden_width),
            "grid": int(args.kan_grid),
            "k": int(args.kan_k),
            "seed": int(seed),
        },
        "net_arch": net_arch,
    }
    model: Any = PPO(
        "MlpPolicy",
        vec_norm,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=int(args.train_timesteps))
    model.save(run_dir / "real_kan_belief_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def _update_summary_with_belief_metadata(
    summary_path: Path,
    *,
    architecture: str,
    belief_meta: dict[str, Any],
) -> None:
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["belief_sidecar"] = {
        "architecture": architecture,
        "base_observation_version": "v10",
        "effective_observation_dim": 101 + len(belief_meta["targets"]),
        **belief_meta,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if str(args.observation_version) != "v10":
        raise ValueError("Risk-belief sidecar currently requires --observation-version v10.")
    targets = [_parse_target(str(target)) for target in args.belief_targets]
    models, belief_meta = train_belief_models(
        Path(args.belief_dataset),
        targets,
        class_weight=str(args.belief_class_weight),
    )

    def wrapper_factory(env: gym.Env[np.ndarray, np.ndarray]) -> RiskBeliefAppendWrapper:
        return RiskBeliefAppendWrapper(env, models=models)

    args._observation_wrapper = wrapper_factory  # type: ignore[attr-defined]
    args.invocation = "python scripts/run_track_b_risk_belief_sidecar.py " + " ".join(sys.argv[1:])

    original_train = smoke.train_ppo
    original_learned_policy_name = smoke.learned_policy_name
    original_model_filename = smoke.model_filename
    policy_name = "ppo_mlp_belief" if args.architecture == "ppo_mlp" else "real_kan_belief"
    try:
        if args.architecture == "ppo_mlp":
            smoke.train_ppo = train_ppo_mlp_belief  # type: ignore[assignment]
            smoke.model_filename = lambda _args=None: "ppo_mlp_belief_model.zip"  # type: ignore[assignment]
        else:
            smoke.train_ppo = train_ppo_real_kan_belief  # type: ignore[assignment]
            smoke.model_filename = lambda _args=None: "real_kan_belief_model.zip"  # type: ignore[assignment]
        smoke.learned_policy_name = lambda _args=None: policy_name  # type: ignore[assignment]
        summary = smoke.run_smoke(args)
    finally:
        smoke.train_ppo = original_train  # type: ignore[assignment]
        smoke.learned_policy_name = original_learned_policy_name  # type: ignore[assignment]
        smoke.model_filename = original_model_filename  # type: ignore[assignment]

    summary_path = Path(summary["artifacts"]["summary_json"])
    _update_summary_with_belief_metadata(
        summary_path,
        architecture=str(args.architecture),
        belief_meta=belief_meta,
    )
    print(f"Wrote Track B risk-belief sidecar bundle to {summary_path}")
    for target in belief_meta["targets"]:
        print(
            "Belief target {label}: base={base:.3f}, mean_pred={pred:.3f}".format(
                label=target["label"],
                base=float(target["base_rate"]),
                pred=float(target["mean_predicted_probability"]),
            )
        )
    for row in summary["policy_summary"]:
        if row["policy"] == policy_name:
            print(
                f"{policy_name}: order_ret_excel={float(row['order_ret_excel_mean']):.6f}, "
                f"cost={float(row['assembly_cost_index_mean']):.3f}"
            )


if __name__ == "__main__":
    main()
