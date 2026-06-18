#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - runtime guard for optional dependency.
    RecurrentPPO = None
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    RAW_MATERIAL_FLOW_MODE_OPTIONS,
    RISK_OCCURRENCE_MODE_OPTIONS,
)
from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import (  # noqa: E402
    ThesisDesignSpec,
    design_spec_for_cfi,
    parse_cf_range,
)
from scripts.audit_garrido_metric_saturation import (  # noqa: E402
    order_metric_distribution,
    pct,
    quantile,
)
from scripts.run_garrido_static_fidelity_stress import (  # noqa: E402
    RISK_PROFILES,
    risk_kwargs_for_profile,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_decision_ppo_smoke")
STATIC_POLICIES = ("static_s1", "static_s2", "static_s3")
INVENTORY_POLICIES = tuple(
    f"inventory_I{period}_S1" for period in THESIS_INVENTORY_PERIODS
)
ACTION_DIM = 18
FACTORIZED_ACTION_DIM = 4
THESIS_FACTORIZED_ACTION_DIM = 2
CONTINUOUS_IT_S_ACTION_DIM = 2
DEFAULT_DMLPA_HISTORY_WINDOW = 30
DEFAULT_DMLPA_FEATURES_DIM = 120


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO with Garrido thesis decision variables and a "
            "configurable observation surface."
        )
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--train-timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=None,
        help=(
            "Optional independent seed base for deterministic evaluation. "
            "Defaults to --seed for backwards compatibility; set this to a "
            "held-out value for paper-facing runs."
        ),
    )
    parser.add_argument(
        "--profile-eval-common-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When evaluating over --eval-risk-profile/--garrido-cfis, use the "
            "same held-out random seed for PPO and every static baseline within "
            "each Cf row. Default False preserves historical offset-per-policy "
            "behavior; enable for paper-facing paired comparisons."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument(
        "--observation-mode",
        choices=[
            "decision_reward",
            "env_reward",
            "env_state_reward",
            "env_sdm_history_reward",
        ],
        default="env_sdm_history_reward",
    )
    parser.add_argument(
        "--action-space-mode",
        choices=["onehot_18d", "factorized", "thesis_factorized", "continuous_it_s"],
        default="thesis_factorized",
    )
    parser.add_argument(
        "--inventory-period-mode",
        choices=["thesis_strict", "per_node"],
        default="thesis_strict",
        help=(
            "For action_space_mode=factorized, thesis_strict collapses Op3/Op5/Op9 "
            "to one common thesis period; per_node declares the inventory-period "
            "extension. Ignored by thesis_factorized."
        ),
    )
    parser.add_argument(
        "--algo",
        choices=["ppo_mlp", "recurrent_ppo", "dmlpa_ppo"],
        default="ppo_mlp",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_DMLPA_HISTORY_WINDOW,
        help=(
            "Frame history length for dmlpa_ppo. RecurrentPPO keeps memory in "
            "the LSTM and does not use this frame stack."
        ),
    )
    parser.add_argument(
        "--dmlpa-features-dim",
        type=int,
        default=DEFAULT_DMLPA_FEATURES_DIM,
        help="Transformer feature dimension for dmlpa_ppo.",
    )
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Run a compact observation/reward/architecture ablation suite.",
    )
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument(
        "--stochastic-pt-spread",
        type=float,
        default=1.0,
        help=(
            "Scales stochastic processing-time variability when --stochastic-pt "
            "is enabled. Historical default 1.0 is Tri(0.75*PT, PT, 1.5*PT); "
            "0.0 collapses to deterministic PT."
        ),
    )
    parser.add_argument(
        "--stochastic-pt-mean-preserving",
        action="store_true",
        help=(
            "Use a symmetric triangular PT envelope around the thesis PT, so "
            "changing --stochastic-pt-spread changes variance without changing "
            "the expected processing time."
        ),
    )
    parser.add_argument(
        "--raw-material-flow-mode",
        default="legacy_validated",
        choices=RAW_MATERIAL_FLOW_MODE_OPTIONS,
    )
    parser.add_argument(
        "--raw-material-order-up-to-multiplier", type=float, default=2.0
    )
    parser.add_argument(
        "--risk-occurrence-mode",
        default="legacy_renewal",
        choices=RISK_OCCURRENCE_MODE_OPTIONS,
    )
    parser.add_argument(
        "--garrido-cfis",
        default="31-90",
        help="Comma/range list of Garrido thesis static Cf rows to evaluate.",
    )
    parser.add_argument(
        "--learn-initial-decision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the first policy action as the thesis-comparable initial buffer/"
            "shift decision before warmup."
        ),
    )
    parser.add_argument(
        "--include-static-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate static I_t x S combinations as a stronger non-Garrido baseline.",
    )
    parser.add_argument(
        "--eval-ai-on-garrido-cfis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the trained AI policy under each selected Garrido Cf risk row.",
    )
    parser.add_argument(
        "--train-cfis",
        default=None,
        help=(
            "Optional comma/range list of Garrido Cf rows to sample at each "
            "training reset. Defaults to the fixed --risk-level env."
        ),
    )
    parser.add_argument(
        "--train-risk-profile",
        choices=RISK_PROFILES,
        default=None,
        help=(
            "When --train-cfis is set, override each sampled Cf row with this "
            "static-fidelity risk profile, e.g. war_stress_v1."
        ),
    )
    parser.add_argument(
        "--eval-risk-profile",
        choices=RISK_PROFILES,
        default=None,
        help=(
            "Evaluate PPO and static baselines over --garrido-cfis using this "
            "risk profile instead of the fixed all-risk --risk-level env."
        ),
    )
    parser.add_argument(
        "--policy-net-arch",
        choices=["small", "medium", "large"],
        default="medium",
        help="MLP policy/value size for PPO.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for learned policies. Use 'auto' on Kaggle GPU.",
    )
    parser.add_argument("--w-bo", type=float, default=1.0)
    parser.add_argument("--w-cost", type=float, default=0.06)
    parser.add_argument("--w-disr", type=float, default=0.0)
    parser.add_argument("--ret-seq-w-sc", type=float, default=0.60)
    parser.add_argument("--ret-seq-w-bc", type=float, default=0.25)
    parser.add_argument("--ret-seq-w-ae", type=float, default=0.15)
    parser.add_argument("--ret-seq-kappa", type=float, default=0.20)
    parser.add_argument("--ret-ladder-w-sc", type=float, default=0.65)
    parser.add_argument("--ret-ladder-w-rc", type=float, default=0.30)
    parser.add_argument("--ret-ladder-w-ef", type=float, default=0.05)
    parser.add_argument("--ret-ladder-cap-kappa", type=float, default=0.10)
    parser.add_argument("--ret-ladder-inv-kappa", type=float, default=0.05)
    parser.add_argument("--ret-ladder-gate-beta", type=float, default=12.0)
    parser.add_argument("--ret-ladder-gate-sc-threshold", type=float, default=0.95)
    parser.add_argument("--ret-ladder-gate-rc-threshold", type=float, default=0.70)
    # ReT_tail_v1 (tail/recovery-aligned reward, un-gated cost)
    parser.add_argument("--ret-tail-w-sc", type=float, default=0.30)
    parser.add_argument("--ret-tail-w-rc", type=float, default=0.60)
    parser.add_argument("--ret-tail-w-ce", type=float, default=0.10)
    parser.add_argument("--ret-tail-cap-kappa", type=float, default=0.40)
    parser.add_argument("--ret-tail-inv-kappa", type=float, default=0.25)
    parser.add_argument("--ret-tail-boost", type=float, default=0.0)
    parser.add_argument(
        "--ret-tail-transform",
        choices=["identity", "power", "exp_norm"],
        default="identity",
        help=(
            "Post-transform for ReT_tail_v1. identity keeps the tuned "
            "Cobb-Douglas reward; power uses R^gamma; exp_norm uses "
            "(exp(beta R)-1)/(exp(beta)-1)."
        ),
    )
    parser.add_argument("--ret-tail-gamma", type=float, default=1.0)
    parser.add_argument("--ret-tail-beta", type=float, default=2.0)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help=(
            "Number of parallel training envs (DummyVecEnv). Default 1 preserves "
            "prior behavior; 8 gives lower-variance gradients and matters more for "
            "recurrent_ppo. Each env gets a distinct base seed; per-episode "
            "disruptions still re-randomize (default_rng(None) on auto-reset)."
        ),
    )
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--vec-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize observations during training and deterministic evaluation.",
    )
    parser.add_argument(
        "--norm-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also normalize rewards by the running return std (VecNormalize "
            "norm_reward). Scale-only transform: helps value-function fit and "
            "convergence stability, does not change the optimal policy. Default "
            "False preserves prior reproducible results."
        ),
    )
    return parser


def env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "action_space_mode": args.action_space_mode,
        "inventory_period_mode": args.inventory_period_mode,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "stochastic_pt": args.stochastic_pt,
        "stochastic_pt_spread": args.stochastic_pt_spread,
        "stochastic_pt_mean_preserving": args.stochastic_pt_mean_preserving,
        "raw_material_flow_mode": args.raw_material_flow_mode,
        "raw_material_order_up_to_multiplier": args.raw_material_order_up_to_multiplier,
        "risk_occurrence_mode": args.risk_occurrence_mode,
        "learn_initial_decision": args.learn_initial_decision,
        "w_bo": args.w_bo,
        "w_cost": args.w_cost,
        "w_disr": args.w_disr,
        "ret_seq_w_sc": args.ret_seq_w_sc,
        "ret_seq_w_bc": args.ret_seq_w_bc,
        "ret_seq_w_ae": args.ret_seq_w_ae,
        "ret_seq_kappa": args.ret_seq_kappa,
        "ret_ladder_w_sc": args.ret_ladder_w_sc,
        "ret_ladder_w_rc": args.ret_ladder_w_rc,
        "ret_ladder_w_ef": args.ret_ladder_w_ef,
        "ret_ladder_cap_kappa": args.ret_ladder_cap_kappa,
        "ret_ladder_inv_kappa": args.ret_ladder_inv_kappa,
        "ret_ladder_gate_beta": args.ret_ladder_gate_beta,
        "ret_ladder_gate_sc_threshold": args.ret_ladder_gate_sc_threshold,
        "ret_ladder_gate_rc_threshold": args.ret_ladder_gate_rc_threshold,
        "ret_tail_w_sc": args.ret_tail_w_sc,
        "ret_tail_w_rc": args.ret_tail_w_rc,
        "ret_tail_w_ce": args.ret_tail_w_ce,
        "ret_tail_cap_kappa": args.ret_tail_cap_kappa,
        "ret_tail_inv_kappa": args.ret_tail_inv_kappa,
        "ret_tail_boost": args.ret_tail_boost,
        "ret_tail_transform": args.ret_tail_transform,
        "ret_tail_gamma": args.ret_tail_gamma,
        "ret_tail_beta": args.ret_tail_beta,
    }


def policy_net_arch(name: str, *, recurrent: bool = False) -> Any:
    if name == "small":
        return (
            {"pi": [64], "vf": [64]} if recurrent else {"pi": [64, 64], "vf": [64, 64]}
        )
    if name == "medium":
        return (
            {"pi": [128], "vf": [128]}
            if recurrent
            else {"pi": [128, 128], "vf": [128, 128]}
        )
    if name == "large":
        return (
            {"pi": [256], "vf": [256]}
            if recurrent
            else {"pi": [256, 256], "vf": [256, 256]}
        )
    raise ValueError(f"Unknown policy_net_arch={name!r}.")


class DMLPAPositionalExtractor(BaseFeaturesExtractor):
    """DMLPA-style Transformer over a fixed observation history window."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        history_window: int,
        features_dim: int = DEFAULT_DMLPA_FEATURES_DIM,
    ) -> None:
        super().__init__(observation_space, features_dim)
        flat_dim = int(np.prod(observation_space.shape))
        if history_window <= 0:
            raise ValueError("history_window must be positive.")
        if flat_dim % history_window != 0:
            raise ValueError(
                f"Observation dim {flat_dim} is not divisible by "
                f"history_window={history_window}."
            )
        self.history_window = int(history_window)
        self.obs_dimension = flat_dim // self.history_window
        self.latent_rw = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dimension, 100),
            torch.nn.GELU(),
            torch.nn.Linear(100, features_dim),
        )
        self.pre_norm = torch.nn.LayerNorm(features_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=12,
            batch_first=True,
        )
        self.accumulated = torch.nn.TransformerEncoder(layer, num_layers=4)
        self.register_buffer(
            "pos_encoding",
            self._build_sinusoidal_pe(self.history_window, features_dim),
        )

    @staticmethod
    def _build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.reshape(batch_size, self.history_window, self.obs_dimension)
        x = self.latent_rw(x)
        x = self.pre_norm(x + self.pos_encoding.to(dtype=x.dtype))
        x = self.accumulated(x)
        return x[:, -1, :]


class GarridoCfTrainingWrapper(gym.Wrapper):
    """Sample a Garrido Cf risk row at each training reset."""

    def __init__(
        self,
        env: gym.Env,
        specs: list[ThesisDesignSpec],
        *,
        seed: int,
        risk_profile: str | None = None,
        thesis_pattern_risk_level: str = "increased",
    ) -> None:
        super().__init__(env)
        if not specs:
            raise ValueError("At least one Garrido Cf spec is required.")
        self.specs = list(specs)
        self.rng = np.random.default_rng(seed)
        self.current_spec: ThesisDesignSpec | None = None
        self.risk_profile = risk_profile
        self.thesis_pattern_risk_level = thesis_pattern_risk_level

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        spec = self.specs[int(self.rng.integers(0, len(self.specs)))]
        self.current_spec = spec
        base_env = getattr(self.env, "unwrapped", self.env)
        if self.risk_profile:
            risk_kwargs = risk_kwargs_for_profile(
                spec=spec,
                profile=self.risk_profile,
                thesis_pattern_risk_level=self.thesis_pattern_risk_level,
            )
            base_env.risk_level = str(risk_kwargs["risk_level"])
            base_env.enabled_risks = set(risk_kwargs["enabled_risks"])
            base_env.risk_overrides = dict(risk_kwargs["risk_overrides"])
        else:
            base_env.enabled_risks = set(spec.enabled_risks)
            base_env.risk_overrides = dict(spec.risk_overrides)
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["training_cfi"] = spec.cfi
        info["training_cfi_family"] = spec.family
        info["training_source_cfi"] = spec.source_cfi
        if self.risk_profile:
            info["training_risk_profile"] = self.risk_profile
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.current_spec is not None:
            info = dict(info)
            info["training_cfi"] = self.current_spec.cfi
            info["training_cfi_family"] = self.current_spec.family
            info["training_source_cfi"] = self.current_spec.source_cfi
        return obs, reward, terminated, truncated, info


def make_env(args: argparse.Namespace, seed: int) -> Callable[[], Monitor]:
    kwargs = env_kwargs(args)
    train_specs = (
        [design_spec_for_cfi(cfi) for cfi in parse_cf_range(args.train_cfis)]
        if args.train_cfis
        else []
    )

    def _init() -> Monitor:
        env = make_dkana_thesis_faithful_env(**kwargs)
        if train_specs:
            env = GarridoCfTrainingWrapper(
                env,
                train_specs,
                seed=seed,
                risk_profile=args.train_risk_profile,
                thesis_pattern_risk_level=args.risk_level,
            )
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def static_action(shifts: int, *, action_space_mode: str) -> np.ndarray:
    if action_space_mode == "thesis_factorized":
        return np.array([0, shifts - 1], dtype=np.int64)
    if action_space_mode == "continuous_it_s":
        return np.array([0.0, shift_signal_for(shifts)], dtype=np.float32)
    if action_space_mode == "factorized":
        return np.array([0, 0, 0, shifts - 1], dtype=np.int64)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[15 + shifts - 1] = 1.0
    return action


def inventory_action(
    period: int, *, shifts: int = 1, action_space_mode: str
) -> np.ndarray:
    period_index = THESIS_INVENTORY_PERIODS.index(int(period))
    if action_space_mode == "thesis_factorized":
        return np.array([period_index + 1, shifts - 1], dtype=np.int64)
    if action_space_mode == "continuous_it_s":
        return np.array(
            [
                float(period) / float(max(THESIS_INVENTORY_PERIODS)),
                shift_signal_for(shifts),
            ],
            dtype=np.float32,
        )
    if action_space_mode == "factorized":
        level = period_index + 1
        return np.array([level, level, level, shifts - 1], dtype=np.int64)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    for node_index in range(3):
        action[node_index * 5 + period_index] = 1.0
    action[15 + shifts - 1] = 1.0
    return action


def shift_signal_for(shifts: int) -> float:
    if shifts == 1:
        return -1.0
    if shifts == 2:
        return 0.0
    if shifts == 3:
        return 1.0
    raise ValueError(f"Invalid shifts={shifts}; expected 1, 2, or 3.")


def thesis_design_action(
    spec: ThesisDesignSpec, *, action_space_mode: str
) -> np.ndarray:
    period = spec.inventory_replenishment_period
    if period is None:
        return static_action(spec.shifts, action_space_mode=action_space_mode)
    return inventory_action(
        int(period),
        shifts=spec.shifts,
        action_space_mode=action_space_mode,
    )


def thesis_design_env_kwargs(
    spec: ThesisDesignSpec, *, action_space_mode: str
) -> dict[str, Any]:
    return {
        "enabled_risks": set(spec.enabled_risks),
        "risk_overrides": dict(spec.risk_overrides),
        "initial_action": thesis_design_action(
            spec,
            action_space_mode=action_space_mode,
        ),
    }


def thesis_risk_env_kwargs(spec: ThesisDesignSpec) -> dict[str, Any]:
    return {
        "enabled_risks": set(spec.enabled_risks),
        "risk_overrides": dict(spec.risk_overrides),
    }


def profile_risk_env_kwargs(
    spec: ThesisDesignSpec, *, args: argparse.Namespace
) -> dict[str, Any]:
    if not args.eval_risk_profile:
        return thesis_risk_env_kwargs(spec)
    return risk_kwargs_for_profile(
        spec=spec,
        profile=args.eval_risk_profile,
        thesis_pattern_risk_level=args.risk_level,
    )


def evaluate_action_policy(
    *,
    args: argparse.Namespace,
    policy_name: str,
    action_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    seed: int,
    episodes: int | None = None,
    env_kwargs_override: dict[str, Any] | None = None,
    policy_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kwargs = env_kwargs(args)
    kwargs.update(env_kwargs_override or {})
    metadata = {"baseline_family": "", "cfi": "", "source_cfi": ""}
    metadata.update(policy_metadata or {})
    episode_count = args.eval_episodes if episodes is None else int(episodes)
    for episode in range(episode_count):
        env = make_dkana_thesis_faithful_env(**kwargs)
        obs, info = env.reset(seed=seed + 10_000 + episode)
        terminated = truncated = False
        reward_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        inventory_target_total_sum = 0.0
        assembly_shift_hours = 0.0
        weekly_flow_fills: list[float] = []
        weekly_stockout_flags: list[bool] = []
        while not (terminated or truncated):
            action = action_fn(np.asarray(obs, dtype=np.float32), info)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            if info.get("action_phase") == "initial_decision":
                continue
            steps += 1
            decision = info.get("thesis_decision", {})
            shift = int(decision.get("assembly_shifts", 1))
            shift_counts[shift] = shift_counts.get(shift, 0) + 1
            inventory_targets = decision.get("inventory_buffer_targets", {})
            if isinstance(inventory_targets, dict):
                inventory_target_total_sum += float(
                    sum(float(value) for value in inventory_targets.values())
                )
            assembly_shift_hours += float(shift) * float(args.step_size_hours)
            demanded = float(info.get("new_demanded", 0.0))
            backorder_qty = float(info.get("new_backorder_qty", 0.0))
            if demanded > 0.0:
                weekly_flow_fills.append(
                    max(0.0, min(1.0, 1.0 - backorder_qty / demanded))
                )
            base_env = getattr(env, "unwrapped", env)
            sim_now = getattr(base_env, "sim", None)
            pending_qty = 0.0
            if sim_now is not None:
                pending_qty = sum(
                    float(getattr(order, "remaining_qty", 0.0))
                    for order in getattr(sim_now, "pending_backorders", [])
                )
            weekly_stockout_flags.append(backorder_qty > 0.0 or pending_qty > 0.0)
        terminal = get_episode_terminal_metrics(env)
        sim = getattr(env.unwrapped, "sim", None)
        order_distribution = order_metric_distribution(sim) if sim is not None else {}
        total_steps = max(1, steps)
        row = {
            "policy": policy_name,
            "episode": episode,
            "seed": seed,
            "eval_seed": seed + 10_000 + episode,
            "steps": steps,
            "reward_total": reward_total,
            "fill_rate_order_level": terminal["fill_rate_order_level"],
            "backorder_rate_order_level": terminal["backorder_rate_order_level"],
            "order_level_ret_mean": terminal["order_level_ret_mean"],
            "ret_mean_all_orders_zero_unfulfilled": order_distribution.get(
                "ret_mean_all_orders_zero_unfulfilled",
                terminal["order_level_ret_mean"],
            ),
            "flow_fill_rate": (
                float(np.mean(weekly_flow_fills)) if weekly_flow_fills else 1.0
            ),
            "stockout_week_pct": pct(weekly_stockout_flags),
            "p10_step_flow_fill": quantile(weekly_flow_fills, 0.10),
            "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
            "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
            "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
            "assembly_shift_hours": assembly_shift_hours,
            "inventory_target_total_mean": inventory_target_total_sum / total_steps,
            "pending_backorders_count": float(
                len(getattr(sim, "pending_backorders", [])) if sim is not None else 0.0
            ),
            "pending_backorder_qty": float(
                sum(
                    float(getattr(order, "remaining_qty", 0.0))
                    for order in getattr(sim, "pending_backorders", [])
                )
                if sim is not None
                else 0.0
            ),
            "unattended_orders_total": float(
                getattr(sim, "total_unattended_orders", 0.0) if sim is not None else 0.0
            ),
        }
        for key in (
            "re_fr_contribution_all",
            "re_ap_contribution_all",
            "re_rp_contribution_all",
            "re_dp_rp_contribution_all",
            "dynamic_ret_contribution_all",
            "static_ret_contribution_all",
            "dynamic_case_pct",
            "pct_case_fill_rate",
            "pct_case_autotomy",
            "pct_case_recovery",
            "pct_case_non_recovery",
            "pct_case_unfulfilled",
            "pct_ret_eq_1",
            "pct_ret_lt_05",
            "ret_p10_all",
            "ret_p50_all",
            "ret_p90_all",
        ):
            row[key] = float(order_distribution.get(key, 0.0))
        row.update(metadata)
        rows.append(row)
        env.close()
    return rows


def eval_seed_base(args: argparse.Namespace) -> int:
    return int(args.eval_seed_base if args.eval_seed_base is not None else args.seed)


def profile_eval_seed(
    args: argparse.Namespace,
    spec: ThesisDesignSpec,
    *,
    seed_offset: int = 0,
) -> int:
    base = eval_seed_base(args) + spec.cfi * 1_000_000
    if bool(getattr(args, "profile_eval_common_seed", False)):
        return base
    return base + seed_offset


def evaluate_model_policy(
    *,
    args: argparse.Namespace,
    model: Any,
    policy_name: str,
    seed: int,
    vec_normalize: VecNormalize | None = None,
    episodes: int | None = None,
    env_kwargs_override: dict[str, Any] | None = None,
    policy_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    recurrent = args.algo == "recurrent_ppo"
    lstm_states: Any = None
    episode_starts: np.ndarray | None = None
    frame_buffer: list[np.ndarray] | None = None

    def _predict(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        nonlocal frame_buffer, lstm_states, episode_starts
        model_obs = obs
        if args.algo == "dmlpa_ppo":
            obs_frame = np.asarray(obs, dtype=np.float32)
            if frame_buffer is None:
                frame_buffer = [
                    np.zeros_like(obs_frame, dtype=np.float32)
                    for _ in range(args.history_window - 1)
                ]
            frame_buffer.append(obs_frame)
            frame_buffer = frame_buffer[-args.history_window :]
            model_obs = np.concatenate(frame_buffer, axis=0)
        if vec_normalize is not None:
            model_obs = vec_normalize.normalize_obs(model_obs.reshape(1, -1))[0]
        if recurrent:
            if episode_starts is None:
                episode_starts = np.ones((1,), dtype=bool)
            action, lstm_states = model.predict(
                model_obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            episode_starts = np.zeros((1,), dtype=bool)
            return np.asarray(action)
        action, _ = model.predict(model_obs, deterministic=True)
        return np.asarray(action)

    rows = []
    episode_count = args.eval_episodes if episodes is None else int(episodes)
    for episode in range(episode_count):
        frame_buffer = None
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        rows.extend(
            evaluate_action_policy(
                args=args,
                policy_name=policy_name,
                action_fn=_predict,
                seed=seed + episode * 100_000,
                episodes=1,
                env_kwargs_override=env_kwargs_override,
                policy_metadata=policy_metadata,
            )[:1]
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for policy in sorted({row["policy"] for row in rows}):
        bucket = [row for row in rows if row["policy"] == policy]
        out.append(
            {
                "policy": policy,
                "episode_count": len(bucket),
                "reward_total_mean": float(
                    np.mean([row["reward_total"] for row in bucket])
                ),
                "fill_rate_order_level_mean": float(
                    np.mean([row["fill_rate_order_level"] for row in bucket])
                ),
                "order_level_ret_mean": float(
                    np.mean([row["order_level_ret_mean"] for row in bucket])
                ),
                "ret_mean_all_orders_zero_unfulfilled_mean": float(
                    np.mean(
                        [row["ret_mean_all_orders_zero_unfulfilled"] for row in bucket]
                    )
                ),
                "flow_fill_rate_mean": float(
                    np.mean([row["flow_fill_rate"] for row in bucket])
                ),
                "stockout_week_pct_mean": float(
                    np.mean([row["stockout_week_pct"] for row in bucket])
                ),
                "p10_step_flow_fill_mean": float(
                    np.mean([row["p10_step_flow_fill"] for row in bucket])
                ),
                "re_fr_contribution_all_mean": float(
                    np.mean([row["re_fr_contribution_all"] for row in bucket])
                ),
                "re_ap_contribution_all_mean": float(
                    np.mean([row["re_ap_contribution_all"] for row in bucket])
                ),
                "re_rp_contribution_all_mean": float(
                    np.mean([row["re_rp_contribution_all"] for row in bucket])
                ),
                "re_dp_rp_contribution_all_mean": float(
                    np.mean([row["re_dp_rp_contribution_all"] for row in bucket])
                ),
                "dynamic_ret_contribution_all_mean": float(
                    np.mean([row["dynamic_ret_contribution_all"] for row in bucket])
                ),
                "dynamic_case_pct_mean": float(
                    np.mean([row["dynamic_case_pct"] for row in bucket])
                ),
                "pct_case_fill_rate_mean": float(
                    np.mean([row["pct_case_fill_rate"] for row in bucket])
                ),
                "pct_ret_eq_1_mean": float(
                    np.mean([row["pct_ret_eq_1"] for row in bucket])
                ),
                "pct_ret_lt_05_mean": float(
                    np.mean([row["pct_ret_lt_05"] for row in bucket])
                ),
                "ret_p10_all_mean": float(
                    np.mean([row["ret_p10_all"] for row in bucket])
                ),
                "ret_p50_all_mean": float(
                    np.mean([row["ret_p50_all"] for row in bucket])
                ),
                "ret_p90_all_mean": float(
                    np.mean([row["ret_p90_all"] for row in bucket])
                ),
                "pct_steps_S1_mean": float(
                    np.mean([row["pct_steps_S1"] for row in bucket])
                ),
                "pct_steps_S2_mean": float(
                    np.mean([row["pct_steps_S2"] for row in bucket])
                ),
                "pct_steps_S3_mean": float(
                    np.mean([row["pct_steps_S3"] for row in bucket])
                ),
                "assembly_shift_hours_mean": float(
                    np.mean([row["assembly_shift_hours"] for row in bucket])
                ),
                "inventory_target_total_mean": float(
                    np.mean([row["inventory_target_total_mean"] for row in bucket])
                ),
                "pending_backorders_count_mean": float(
                    np.mean([row["pending_backorders_count"] for row in bucket])
                ),
                "pending_backorder_qty_mean": float(
                    np.mean([row["pending_backorder_qty"] for row in bucket])
                ),
                "unattended_orders_total_mean": float(
                    np.mean([row["unattended_orders_total"] for row in bucket])
                ),
            }
        )
    return out


def best_garrido_by_family(
    aggregate_rows: list[dict[str, Any]], family: str
) -> dict[str, Any] | None:
    prefix = "garrido_Cf"
    family_rows = [
        row
        for row in aggregate_rows
        if str(row["policy"]).startswith(prefix)
        and str(row["policy"]).endswith(f"_{family}")
    ]
    return (
        max(family_rows, key=lambda row: row["fill_rate_order_level_mean"])
        if family_rows
        else None
    )


def best_static_grid(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    grid_rows = [
        row for row in aggregate_rows if str(row["policy"]).startswith("static_grid_")
    ]
    return (
        max(grid_rows, key=lambda row: row["fill_rate_order_level_mean"])
        if grid_rows
        else None
    )


def promotion_status(
    aggregate_rows: list[dict[str, Any]],
    *,
    algo_name: str,
) -> dict[str, Any]:
    ai_row = next((row for row in aggregate_rows if row["policy"] == algo_name), None)
    garrido_rows = [
        row for row in aggregate_rows if str(row["policy"]).startswith("garrido_")
    ]
    best_garrido = (
        max(garrido_rows, key=lambda row: row["fill_rate_order_level_mean"])
        if garrido_rows
        else None
    )
    beats_fill_rate = None
    if ai_row is not None and best_garrido is not None:
        beats_fill_rate = (
            ai_row["fill_rate_order_level_mean"]
            > best_garrido["fill_rate_order_level_mean"]
        )
    best_inventory = best_garrido_by_family(aggregate_rows, "inventory")
    best_capacity = best_garrido_by_family(aggregate_rows, "capacity")
    best_grid = best_static_grid(aggregate_rows)
    return {
        "gate_sequence": [
            "10k_x_3_seeds_implementation_sanity",
            "100k_x_5_seeds_publishability_check",
            "500k_x_5_seeds_only_if_100k_gate_passes",
        ],
        "promotion_rule": (
            "Promote to 500k x 5 only if 100k x 5 beats the best Garrido "
            "static Cf baseline on order-level fill rate, does not hide the win "
            "with worse order-level ReT, and reports S3/inventory usage."
        ),
        "ai_policy": ai_row,
        "best_static_garrido_by_fill_rate": best_garrido,
        "best_static_garrido_inventory_by_fill_rate": best_inventory,
        "best_static_garrido_capacity_by_fill_rate": best_capacity,
        "best_static_grid_by_fill_rate": best_grid,
        "beats_best_garrido_fill_rate": beats_fill_rate,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_garrido_comparisons(
    rows: list[dict[str, Any]], *, algo_name: str
) -> list[dict[str, Any]]:
    comparisons = []
    cfi_values = sorted(
        {
            int(row["cfi"])
            for row in rows
            if str(row.get("cfi", "")).strip()
            and (
                str(row["policy"]).startswith("garrido_")
                or str(row["policy"]).startswith(f"{algo_name}_on_")
            )
        }
    )
    for cfi in cfi_values:
        ai_bucket = [
            row
            for row in rows
            if row.get("cfi") == cfi
            and str(row["policy"]).startswith(f"{algo_name}_on_")
        ]
        static_bucket = [
            row
            for row in rows
            if row.get("cfi") == cfi and str(row["policy"]).startswith("garrido_")
        ]
        if not ai_bucket or not static_bucket:
            continue
        ai_fill = float(np.mean([row["fill_rate_order_level"] for row in ai_bucket]))
        static_fill = float(
            np.mean([row["fill_rate_order_level"] for row in static_bucket])
        )
        ai_ret = float(np.mean([row["order_level_ret_mean"] for row in ai_bucket]))
        static_ret = float(
            np.mean([row["order_level_ret_mean"] for row in static_bucket])
        )
        comparisons.append(
            {
                "cfi": cfi,
                "family": static_bucket[0].get("baseline_family", ""),
                "source_cfi": static_bucket[0].get("source_cfi", ""),
                "ai_fill_rate_order_level_mean": ai_fill,
                "garrido_fill_rate_order_level_mean": static_fill,
                "fill_rate_delta_ai_minus_garrido": ai_fill - static_fill,
                "ai_order_level_ret_mean": ai_ret,
                "garrido_order_level_ret_mean": static_ret,
                "ret_delta_ai_minus_garrido": ai_ret - static_ret,
                "ai_wins_fill_rate": ai_fill > static_fill,
            }
        )
    return comparisons


def evaluate_profile_panel(
    *,
    args: argparse.Namespace,
    model: Any,
    vec_normalize: VecNormalize | None,
    selected_specs: list[ThesisDesignSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def metadata(spec: ThesisDesignSpec) -> dict[str, Any]:
        return {
            "baseline_family": spec.family,
            "cfi": spec.cfi,
            "source_cfi": spec.source_cfi,
        }

    def add_model(spec: ThesisDesignSpec) -> None:
        rows.extend(
            evaluate_model_policy(
                args=args,
                model=model,
                policy_name=args.algo,
                seed=profile_eval_seed(args, spec),
                vec_normalize=vec_normalize,
                episodes=1,
                env_kwargs_override=profile_risk_env_kwargs(spec, args=args),
                policy_metadata=metadata(spec),
            )
        )

    def add_action(
        spec: ThesisDesignSpec,
        *,
        policy_name: str,
        action_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
        seed_offset: int,
    ) -> None:
        rows.extend(
            evaluate_action_policy(
                args=args,
                policy_name=policy_name,
                action_fn=action_fn,
                seed=profile_eval_seed(args, spec, seed_offset=seed_offset),
                episodes=1,
                env_kwargs_override=profile_risk_env_kwargs(spec, args=args),
                policy_metadata=metadata(spec),
            )
        )

    for spec in selected_specs:
        add_model(spec)
        for idx, policy in enumerate(STATIC_POLICIES, start=1):
            add_action(
                spec,
                policy_name=policy,
                action_fn=lambda obs, info, shifts=idx: static_action(
                    shifts, action_space_mode=args.action_space_mode
                ),
                seed_offset=10_000 + idx,
            )
        for period in THESIS_INVENTORY_PERIODS:
            add_action(
                spec,
                policy_name=f"inventory_I{period}_S1",
                action_fn=lambda obs, info, p=period: inventory_action(
                    p, action_space_mode=args.action_space_mode
                ),
                seed_offset=20_000 + int(period),
            )
        if args.include_static_grid:
            for period in THESIS_INVENTORY_PERIODS:
                for shifts in (1, 2, 3):
                    add_action(
                        spec,
                        policy_name=f"static_grid_I{period}_S{shifts}",
                        action_fn=lambda obs, info, p=period, s=shifts: inventory_action(
                            p,
                            shifts=s,
                            action_space_mode=args.action_space_mode,
                        ),
                        seed_offset=30_000 + int(period) + shifts,
                    )

        def garrido_action_fn(
            obs: np.ndarray,
            info: dict[str, Any],
            baseline_spec: ThesisDesignSpec = spec,
        ) -> np.ndarray:
            return thesis_design_action(
                baseline_spec,
                action_space_mode=args.action_space_mode,
            )

        add_action(
            spec,
            policy_name="garrido_matched_DOE_baseline",
            action_fn=garrido_action_fn,
            seed_offset=40_000,
        )
    return rows


def train_model(
    args: argparse.Namespace, run_dir: Path
) -> tuple[Any, VecNormalize | None]:
    if args.algo == "recurrent_ppo" and RecurrentPPO is None:
        raise RuntimeError("recurrent_ppo requested but sb3_contrib is not installed.")
    if args.algo == "recurrent_ppo" and args.action_space_mode == "onehot_18d":
        raise ValueError(
            "recurrent_ppo is supported only with categorical action spaces "
            "(thesis_factorized or factorized)."
        )
    if args.algo == "dmlpa_ppo" and args.history_window <= 1:
        raise ValueError("dmlpa_ppo requires --history-window greater than 1.")

    n_envs = max(1, int(args.n_envs))
    train_env = DummyVecEnv(
        [make_env(args, args.seed + i) for i in range(n_envs)]
    )
    vec_normalize: VecNormalize | None = None
    vec_env: Any = train_env
    if args.algo == "dmlpa_ppo":
        vec_env = VecFrameStack(vec_env, n_stack=args.history_window)
    if args.vec_normalize:
        vec_normalize = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=bool(args.norm_reward),
            clip_obs=10.0,
        )
        vec_env = vec_normalize
    if args.algo == "recurrent_ppo":
        model: Any = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs={
                "net_arch": policy_net_arch(args.policy_net_arch, recurrent=True),
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
            },
            seed=args.seed,
            verbose=0,
            device=args.device,
        )
    elif args.algo == "dmlpa_ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs={
                "features_extractor_class": DMLPAPositionalExtractor,
                "features_extractor_kwargs": {
                    "history_window": args.history_window,
                    "features_dim": args.dmlpa_features_dim,
                },
                "net_arch": policy_net_arch(args.policy_net_arch),
            },
            seed=args.seed,
            verbose=0,
            device=args.device,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs={"net_arch": policy_net_arch(args.policy_net_arch)},
            seed=args.seed,
            verbose=0,
            device=args.device,
        )
    model.learn(total_timesteps=args.train_timesteps)
    model.save(run_dir / f"{args.algo}_thesis_decision")
    if vec_normalize is not None:
        vec_normalize.training = False
        vec_normalize.save(run_dir / "vecnormalize.pkl")
    return model, vec_normalize


def run_single(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)

    model, vec_normalize = train_model(args, run_dir)
    selected_specs = [
        design_spec_for_cfi(cfi) for cfi in parse_cf_range(args.garrido_cfis)
    ]

    if args.eval_risk_profile:
        rows = evaluate_profile_panel(
            args=args,
            model=model,
            vec_normalize=vec_normalize,
            selected_specs=selected_specs,
        )
    else:
        rows = []
        rows.extend(
            evaluate_model_policy(
                args=args,
                model=model,
                policy_name=args.algo,
                seed=eval_seed_base(args),
                vec_normalize=vec_normalize,
            )
        )
        if args.eval_ai_on_garrido_cfis:
            for spec in selected_specs:
                rows.extend(
                    evaluate_model_policy(
                        args=args,
                        model=model,
                        policy_name=f"{args.algo}_on_{spec.label}_{spec.family}",
                        seed=eval_seed_base(args) + spec.cfi * 1_000_000,
                        vec_normalize=vec_normalize,
                        episodes=1,
                        env_kwargs_override=thesis_risk_env_kwargs(spec),
                        policy_metadata={
                            "baseline_family": spec.family,
                            "cfi": spec.cfi,
                            "source_cfi": spec.source_cfi,
                        },
                    )
                )
        for idx, policy in enumerate(STATIC_POLICIES, start=1):
            rows.extend(
                evaluate_action_policy(
                    args=args,
                    policy_name=policy,
                    action_fn=lambda obs, info, shifts=idx: static_action(
                        shifts, action_space_mode=args.action_space_mode
                    ),
                    seed=eval_seed_base(args),
                    policy_metadata={
                        "baseline_family": "capacity_reference",
                        "cfi": "",
                        "source_cfi": "",
                    },
                )
            )
        for policy_name, period in zip(
            INVENTORY_POLICIES, THESIS_INVENTORY_PERIODS, strict=True
        ):
            rows.extend(
                evaluate_action_policy(
                    args=args,
                    policy_name=policy_name,
                    action_fn=lambda obs, info, p=period: inventory_action(
                        p, action_space_mode=args.action_space_mode
                    ),
                    seed=eval_seed_base(args),
                    policy_metadata={
                        "baseline_family": "inventory_reference",
                        "cfi": "",
                        "source_cfi": "",
                    },
                )
            )
        if args.include_static_grid:
            for period in THESIS_INVENTORY_PERIODS:
                for shifts in (1, 2, 3):
                    rows.extend(
                        evaluate_action_policy(
                            args=args,
                            policy_name=f"static_grid_I{period}_S{shifts}",
                            action_fn=lambda obs, info, p=period, s=shifts: inventory_action(
                                p,
                                shifts=s,
                                action_space_mode=args.action_space_mode,
                            ),
                            seed=eval_seed_base(args),
                            policy_metadata={
                                "baseline_family": "static_inventory_capacity_grid",
                                "cfi": "",
                                "source_cfi": "",
                            },
                        )
                    )
        for spec in selected_specs:

            def garrido_action_fn(
                obs: np.ndarray,
                info: dict[str, Any],
                baseline_spec: ThesisDesignSpec = spec,
            ) -> np.ndarray:
                return thesis_design_action(
                    baseline_spec,
                    action_space_mode=args.action_space_mode,
                )

            rows.extend(
                evaluate_action_policy(
                    args=args,
                    policy_name=f"garrido_{spec.label}_{spec.family}",
                    action_fn=garrido_action_fn,
                    seed=eval_seed_base(args),
                    env_kwargs_override=thesis_design_env_kwargs(
                        spec,
                        action_space_mode=args.action_space_mode,
                    ),
                    policy_metadata={
                        "baseline_family": spec.family,
                        "cfi": spec.cfi,
                        "source_cfi": spec.source_cfi,
                    },
                )
            )
        rng = np.random.default_rng(args.seed)

        if args.action_space_mode == "thesis_factorized":

            def random_action_fn(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
                return np.array(
                    [
                        rng.integers(0, 6),
                        rng.integers(0, 3),
                    ],
                    dtype=np.int64,
                )

        elif args.action_space_mode == "continuous_it_s":

            def random_action_fn(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
                return np.array(
                    [
                        rng.uniform(0.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                    ],
                    dtype=np.float32,
                )

        elif args.action_space_mode == "factorized":

            def random_action_fn(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
                return np.array(
                    [
                        rng.integers(0, 6),
                        rng.integers(0, 6),
                        rng.integers(0, 6),
                        rng.integers(0, 3),
                    ],
                    dtype=np.int64,
                )

        else:

            def random_action_fn(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
                return rng.uniform(0.0, 1.0, ACTION_DIM).astype(np.float32)

        rows.extend(
            evaluate_action_policy(
                args=args,
                policy_name="random",
                action_fn=random_action_fn,
                seed=eval_seed_base(args),
                policy_metadata={
                    "baseline_family": "random",
                    "cfi": "",
                    "source_cfi": "",
                },
            )
        )

    aggregate_rows = aggregate(rows)
    gate_status = promotion_status(aggregate_rows, algo_name=args.algo)
    paired_comparisons = paired_garrido_comparisons(rows, algo_name=args.algo)
    summary = {
        "created_at": utc_now_iso(),
        "env_kwargs": env_kwargs(args),
        "garrido_cfis": args.garrido_cfis,
        "train_cfis": args.train_cfis,
        "train_risk_profile": args.train_risk_profile,
        "eval_risk_profile": args.eval_risk_profile,
        "include_static_grid": args.include_static_grid,
        "eval_ai_on_garrido_cfis": args.eval_ai_on_garrido_cfis,
        "train_timesteps": args.train_timesteps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "eval_seed_base": eval_seed_base(args),
        "profile_eval_common_seed": bool(args.profile_eval_common_seed),
        "n_envs": max(1, int(args.n_envs)),
        "action_contract": "thesis_faithful_dkana_v1",
        "action_space_mode": args.action_space_mode,
        "inventory_period_mode": args.inventory_period_mode,
        "action_dim": {
            "thesis_factorized": THESIS_FACTORIZED_ACTION_DIM,
            "continuous_it_s": CONTINUOUS_IT_S_ACTION_DIM,
            "factorized": FACTORIZED_ACTION_DIM,
            "onehot_18d": ACTION_DIM,
        }[args.action_space_mode],
        "algo": args.algo,
        "device": args.device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "history_window": args.history_window if args.algo == "dmlpa_ppo" else 1,
        "dmlpa_features_dim": (
            args.dmlpa_features_dim if args.algo == "dmlpa_ppo" else None
        ),
        "policy_net_arch": args.policy_net_arch,
        "best_static_garrido_by_fill_rate": gate_status[
            "best_static_garrido_by_fill_rate"
        ],
        "best_static_garrido_inventory_by_fill_rate": gate_status[
            "best_static_garrido_inventory_by_fill_rate"
        ],
        "best_static_garrido_capacity_by_fill_rate": gate_status[
            "best_static_garrido_capacity_by_fill_rate"
        ],
        "best_static_grid_by_fill_rate": gate_status["best_static_grid_by_fill_rate"],
        "promotion_gate": gate_status,
        "paired_garrido_win_count": int(
            sum(row["ai_wins_fill_rate"] for row in paired_comparisons)
        ),
        "paired_garrido_comparison_count": len(paired_comparisons),
        "paired_garrido_comparisons": paired_comparisons,
        "aggregate": aggregate_rows,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(run_dir / "episode_metrics.csv", rows)
    write_csv(run_dir / "policy_summary.csv", summary["aggregate"])
    write_csv(run_dir / "paired_garrido_comparisons.csv", paired_comparisons)
    if vec_normalize is not None:
        vec_normalize.close()

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved to: {run_dir}")
    return summary


def ablation_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for observation_mode in (
        "decision_reward",
        "env_reward",
        "env_state_reward",
        "env_sdm_history_reward",
    ):
        configs.append(
            {
                "name": f"{args.algo}_{observation_mode}_{args.reward_mode}",
                "observation_mode": observation_mode,
                "reward_mode": args.reward_mode,
                "algo": args.algo,
            }
        )
    for reward_mode in ("control_v1", "ReT_seq_v1"):
        configs.append(
            {
                "name": f"{args.algo}_{args.observation_mode}_{reward_mode}",
                "observation_mode": args.observation_mode,
                "reward_mode": reward_mode,
                "algo": args.algo,
            }
        )
    if RecurrentPPO is not None:
        configs.append(
            {
                "name": f"recurrent_ppo_{args.observation_mode}_{args.reward_mode}",
                "observation_mode": args.observation_mode,
                "reward_mode": args.reward_mode,
                "algo": "recurrent_ppo",
            }
        )

    deduped = []
    seen = set()
    for config in configs:
        key = (config["observation_mode"], config["reward_mode"], config["algo"])
        if key not in seen:
            deduped.append(config)
            seen.add(key)
    return deduped


def main() -> int:
    args = build_parser().parse_args()
    label = args.label or f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    run_dir = args.output_root / label
    if args.ablation_suite:
        run_dir.mkdir(parents=True, exist_ok=False)
        summaries = []
        for config in ablation_configs(args):
            child_args = argparse.Namespace(**vars(args))
            child_args.ablation_suite = False
            child_args.observation_mode = config["observation_mode"]
            child_args.reward_mode = config["reward_mode"]
            child_args.algo = config["algo"]
            summaries.append(run_single(child_args, run_dir / config["name"]))
        (run_dir / "ablation_summary.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        print(f"Saved ablation suite to: {run_dir}")
        return 0

    run_single(args, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
