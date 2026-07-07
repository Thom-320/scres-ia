#!/usr/bin/env python3
"""Audit whether frozen Track B policies behave preventively or reactively.

This is an evaluation-only mechanism audit.  It never retrains a model.  The
audit replays the same common-random-number episodes for frozen PPO+MLP and
Real-KAN policies, records weekly actions, applies windowed action resets, and
measures whether value is created before the risk event or after it.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_observation_ablation import (  # noqa: E402
    FORECAST_FIELD_NAMES,
    REGIME_FIELD_NAMES,
)
from scripts.run_track_b_smoke import (  # noqa: E402
    THESIS_FAITHFUL_PROTOCOL,
    StaticPolicySpec,
    build_static_policy_action,
    extract_downstream_multipliers,
    parse_risk_multiplier_map,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    get_observation_fields,
    make_track_b_env,
)


EVAL_EPISODE_SEED_OFFSET = 50_000
DEFAULT_OUTPUT_DIR = Path(
    "outputs/experiments/track_b_prevention_mechanism_audit_2026-07-03"
)
DEFAULT_PPO_BUNDLES = (
    Path("outputs/experiments/track_b_ablation_8d_final_2026-07-01/joint"),
    Path(
        "outputs/experiments/track_b_gain_2026-06-30/"
        "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
    ),
    Path(
        "outputs/experiments/track_b_seed_expansion_2026-07-02/"
        "track_b_seed_expansion_6_10_claude"
    ),
)
DEFAULT_REAL_KAN_BUNDLES = (
    Path("outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104"),
    Path(
        "outputs/experiments/track_b_real_kan_sidecar_2026-07-03/"
        "confirm_10seed_extension_6_10_60k_h104"
    ),
)
NEUTRAL_STATIC = StaticPolicySpec(
    label="s2_d1.50",
    assembly_shifts=2,
    downstream_multiplier=1.5,
)
ACTION_DIMS = (
    "op3_q",
    "op9_q",
    "op3_rop",
    "op9_rop",
    "op5_q",
    "shift",
    "op10_q",
    "op12_q",
)


@dataclass(frozen=True)
class ModelArtifact:
    policy: str
    seed: int
    model_path: Path
    vec_norm_path: Path


@dataclass
class PolicyRuntime:
    name: str
    seed: int
    model: Any
    vec_norm: VecNormalize


@dataclass
class EpisodeRun:
    policy: str
    seed: int
    episode: int
    eval_seed: int
    condition: str
    rows: list[dict[str, Any]]
    obs_samples: list[np.ndarray]
    ret_excel: float
    fill_rate: float
    service_loss_auc: float
    cost_index: float
    anchor_step: int


class ForecastZeroWrapper(gym.ObservationWrapper):
    """Zero explicit forecast channels without changing observation shape."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        fields = tuple(get_observation_fields("v7"))
        self._indices = tuple(fields.index(name) for name in FORECAST_FIELD_NAMES)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        masked = np.array(observation, dtype=np.float32, copy=True)
        for idx in self._indices:
            masked[idx] = 0.0
        return masked


class ForecastScrambleWrapper(gym.Wrapper):
    """Replace forecast channels with draws from an empirical forecast bank."""

    def __init__(self, env: gym.Env, *, forecast_bank: np.ndarray) -> None:
        super().__init__(env)
        if forecast_bank.ndim != 2 or forecast_bank.shape[1] != 2:
            raise ValueError("forecast_bank must have shape [N, 2].")
        if forecast_bank.shape[0] == 0:
            raise ValueError("forecast_bank must be non-empty.")
        fields = tuple(get_observation_fields("v7"))
        self._idx_48 = fields.index(FORECAST_FIELD_NAMES[0])
        self._idx_168 = fields.index(FORECAST_FIELD_NAMES[1])
        self._bank = np.asarray(forecast_bank, dtype=np.float32)
        self._rng = np.random.default_rng(0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._rng = np.random.default_rng(None if seed is None else int(seed) + 17_371)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._scramble(obs), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._scramble(obs), reward, terminated, truncated, info

    def _scramble(self, observation: np.ndarray) -> np.ndarray:
        scrambled = np.array(observation, dtype=np.float32, copy=True)
        draw = self._bank[int(self._rng.integers(0, self._bank.shape[0]))]
        scrambled[self._idx_48] = draw[0]
        scrambled[self._idx_168] = draw[1]
        return scrambled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit preventive vs reactive behavior for frozen Track B policies."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", nargs="+", default=["ppo_mlp", "real_kan"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--surge-inertia", action="store_true")
    parser.add_argument("--surge-ramp-per-step", type=int, default=1)
    parser.add_argument("--surge-budget-hours", type=float, default=float("inf"))
    parser.add_argument("--ppo-bundles", nargs="+", type=Path, default=list(DEFAULT_PPO_BUNDLES))
    parser.add_argument(
        "--real-kan-bundles",
        nargs="+",
        type=Path,
        default=list(DEFAULT_REAL_KAN_BUNDLES),
    )
    parser.add_argument("--attribution-samples", type=int, default=32)
    parser.add_argument("--max-granger-lag", type=int, default=4)
    parser.add_argument("--skip-forecast-ablation", action="store_true")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument(
        "--reset-policy",
        default=NEUTRAL_STATIC.label,
        choices=["s1_d1.00", "s1_d1.50", "s1_d2.00", "s2_d1.00", "s2_d1.50", "s2_d2.00", "s3_d1.00", "s3_d1.50", "s3_d2.00"],
    )
    return parser.parse_args()


def env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
        "action_contract": "track_b_v1",
    }
    if bool(getattr(args, "surge_inertia", False)):
        kwargs.update(
            {
                "surge_inertia": True,
                "surge_ramp_per_step": int(getattr(args, "surge_ramp_per_step", 1)),
                "surge_budget_hours": float(
                    getattr(args, "surge_budget_hours", float("inf"))
                ),
            }
        )
    enabled_risks = getattr(args, "enabled_risks", None)
    if enabled_risks:
        kwargs["enabled_risks"] = tuple(
            risk.strip() for risk in str(enabled_risks).split(",") if risk.strip()
        )
    risk_frequency_by_id = parse_risk_multiplier_map(getattr(args, "risk_frequency_by_id", None))
    if risk_frequency_by_id:
        kwargs["risk_frequency_multipliers_by_id"] = risk_frequency_by_id
    risk_impact_by_id = parse_risk_multiplier_map(getattr(args, "risk_impact_by_id", None))
    if risk_impact_by_id:
        kwargs["risk_impact_multipliers_by_id"] = risk_impact_by_id
    if bool(getattr(args, "faithful", False)):
        P = THESIS_FAITHFUL_PROTOCOL
        kwargs.update(
            {
                "year_basis": P["year_basis"],
                "warmup_trigger": P["warmup_trigger"],
                "r14_defect_mode": P["r14_defect_mode"],
                "downstream_q_source": "figure_6_2",
                "risk_occurrence_mode": "thesis_window",
                "raw_material_flow_mode": P["raw_material_flow_mode"],
                "raw_material_order_up_to_multiplier": P["raw_material_order_up_to_multiplier"],
                "demand_on_hand_fulfillment_delay": P["demand_on_hand_fulfillment_delay"],
                "stochastic_pt": False,
            }
        )
    return kwargs


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def static_spec_from_label(label: str) -> StaticPolicySpec:
    shift = int(label[1])
    downstream = float(label.split("_d", 1)[1])
    return StaticPolicySpec(label=label, assembly_shifts=shift, downstream_multiplier=downstream)


def vec_observation_shape(vec_path: Path) -> tuple[int, ...] | None:
    import pickle

    try:
        with vec_path.open("rb") as handle:
            vec_norm = pickle.load(handle)
        shape = getattr(getattr(vec_norm, "observation_space", None), "shape", None)
        return tuple(shape) if shape is not None else None
    except Exception:
        return None


def find_artifact(
    policy: str,
    seed: int,
    bundles: list[Path],
    *,
    expected_shape: tuple[int, ...] | None,
) -> ModelArtifact:
    filename = "ppo_real_kan_model.zip" if policy == "real_kan" else "ppo_model.zip"
    wrong_shape: list[str] = []
    for bundle in bundles:
        run_dir = bundle / "models" / f"seed{seed}"
        model_path = run_dir / filename
        vec_path = run_dir / "vec_normalize.pkl"
        if model_path.exists() and vec_path.exists():
            actual_shape = vec_observation_shape(vec_path)
            if expected_shape is not None and actual_shape != expected_shape:
                wrong_shape.append(f"{vec_path} shape={actual_shape}")
                continue
            return ModelArtifact(policy=policy, seed=seed, model_path=model_path, vec_norm_path=vec_path)
    searched = ", ".join(str(p) for p in bundles)
    suffix = f" Shape mismatches skipped: {wrong_shape}" if wrong_shape else ""
    raise FileNotFoundError(f"Missing {policy} seed {seed} artifact in: {searched}.{suffix}")


def load_runtime(policy: str, seed: int, args: argparse.Namespace) -> PolicyRuntime:
    bundles = list(args.real_kan_bundles if policy == "real_kan" else args.ppo_bundles)
    expected_shape = (len(get_observation_fields(args.observation_version)),)
    artifact = find_artifact(policy, seed, bundles, expected_shape=expected_shape)
    model = PPO.load(str(artifact.model_path), device="cpu")

    def _init() -> gym.Env:
        return make_track_b_env(**env_kwargs(args))

    vec_norm = VecNormalize.load(str(artifact.vec_norm_path), DummyVecEnv([_init]))
    vec_norm.training = False
    vec_norm.norm_reward = False
    return PolicyRuntime(name=policy, seed=seed, model=model, vec_norm=vec_norm)


def predict_action(runtime: PolicyRuntime, obs: np.ndarray) -> np.ndarray:
    obs_norm = runtime.vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = runtime.model.predict(obs_norm, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def policy_action(
    runtime: PolicyRuntime | None,
    obs: np.ndarray,
    neutral_action: dict[str, float | int],
) -> np.ndarray | dict[str, float | int]:
    if runtime is None:
        return neutral_action
    return predict_action(runtime, obs)


def make_env(
    args: argparse.Namespace,
    *,
    forecast_condition: str = "full",
    forecast_bank: np.ndarray | None = None,
) -> gym.Env:
    env = make_track_b_env(**env_kwargs(args))
    if forecast_condition == "zeroed":
        return ForecastZeroWrapper(env)
    if forecast_condition == "scrambled":
        if forecast_bank is None:
            raise ValueError("forecast_bank is required for scrambled condition.")
        return ForecastScrambleWrapper(env, forecast_bank=forecast_bank)
    if forecast_condition != "full":
        raise ValueError(f"Unknown forecast_condition={forecast_condition!r}.")
    return env


def observation_value(obs: np.ndarray, fields: tuple[str, ...], name: str) -> float:
    return float(obs[fields.index(name)]) if name in fields else 0.0


def row_from_step(
    *,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    condition: str,
    step: int,
    obs_before: np.ndarray,
    reward: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    fields = tuple(get_observation_fields("v7"))
    clipped = info.get("clipped_action")
    clipped_vec = list(clipped) if isinstance(clipped, (list, tuple, np.ndarray)) else []
    op10_mult, op12_mult = extract_downstream_multipliers(info)
    shift = int(info.get("shifts_active", 1))
    shift_norm = (float(shift) - 1.0) / 2.0
    op10_norm = (float(op10_mult) - 1.25) / 0.75
    op12_norm = (float(op12_mult) - 1.25) / 0.75
    action_intensity = float(np.mean([shift_norm, np.clip(op10_norm, 0.0, 1.0), np.clip(op12_norm, 0.0, 1.0)]))

    row: dict[str, Any] = {
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "condition": condition,
        "step": step,
        "reward": float(reward),
        "forecast_48h": observation_value(obs_before, fields, "risk_forecast_48h_norm"),
        "forecast_168h": observation_value(obs_before, fields, "risk_forecast_168h_norm"),
        "regime_nominal": observation_value(obs_before, fields, "regime_nominal"),
        "regime_strained": observation_value(obs_before, fields, "regime_strained"),
        "regime_pre_disruption": observation_value(obs_before, fields, "regime_pre_disruption"),
        "regime_disrupted": observation_value(obs_before, fields, "regime_disrupted"),
        "regime_recovery": observation_value(obs_before, fields, "regime_recovery"),
        "fill_rate_obs": observation_value(obs_before, fields, "fill_rate"),
        "rolling_fill_rate_4w": observation_value(obs_before, fields, "rolling_fill_rate_4w"),
        "backlog_age_norm": observation_value(obs_before, fields, "backlog_age_norm"),
        "op10_queue_pressure_norm": observation_value(obs_before, fields, "op10_queue_pressure_norm"),
        "op12_queue_pressure_norm": observation_value(obs_before, fields, "op12_queue_pressure_norm"),
        "new_demanded": float(info.get("new_demanded", 0.0)),
        "new_backorder_qty": float(info.get("new_backorder_qty", 0.0)),
        "pending_backorder_qty": float(info.get("pending_backorder_qty", 0.0)),
        "shift": shift,
        "shift_norm": shift_norm,
        "op10_multiplier": float(op10_mult),
        "op12_multiplier": float(op12_mult),
        "op10_norm": float(np.clip(op10_norm, 0.0, 1.0)),
        "op12_norm": float(np.clip(op12_norm, 0.0, 1.0)),
        "action_intensity": action_intensity,
    }
    for idx, name in enumerate(ACTION_DIMS):
        row[f"action_{name}"] = float(clipped_vec[idx]) if idx < len(clipped_vec) else ""
    return row


def infer_event_anchor(rows: list[dict[str, Any]]) -> int:
    for row in rows:
        if float(row.get("regime_pre_disruption", 0.0)) >= 0.5 or float(row.get("regime_disrupted", 0.0)) >= 0.5:
            return int(row["step"])
    for row in rows:
        if max(float(row.get("forecast_48h", 0.0)), float(row.get("forecast_168h", 0.0))) >= 0.20:
            return int(row["step"])
    if not rows:
        return 0
    return int(max(rows, key=lambda r: float(r.get("forecast_168h", 0.0)))["step"])


def finalize_episode(env: gym.Env, shift_values: list[int]) -> tuple[float, float, float, float]:
    metrics = compute_episode_metrics(env.unwrapped.sim)
    ret_excel = float(metrics.get("ret_excel", 0.0))
    fill_rate = float(metrics.get("fill_rate", 0.0))
    service_loss_auc = float(metrics.get("service_loss_auc_per_order", 0.0))
    cost_index = float(sum(shift_values) / (3.0 * len(shift_values))) if shift_values else 0.0
    return ret_excel, fill_rate, service_loss_auc, cost_index


def run_episode(
    *,
    runtime: PolicyRuntime | None,
    args: argparse.Namespace,
    seed: int,
    episode: int,
    eval_seed: int,
    policy_name: str,
    neutral_action: dict[str, float | int],
    condition: str = "full",
    forecast_bank: np.ndarray | None = None,
    reset_window: tuple[int, int] | None = None,
    anchor_step: int | None = None,
    keep_rows: bool = True,
    keep_obs: bool = True,
) -> EpisodeRun:
    env = make_env(args, forecast_condition=condition, forecast_bank=forecast_bank)
    obs, _info = env.reset(seed=eval_seed)
    terminated = False
    truncated = False
    step = 0
    rows: list[dict[str, Any]] = []
    obs_samples: list[np.ndarray] = []
    shifts: list[int] = []

    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action: Any = policy_action(runtime, obs_before, neutral_action)
        if reset_window is not None and anchor_step is not None:
            rel = step - int(anchor_step)
            if int(reset_window[0]) <= rel <= int(reset_window[1]):
                action = neutral_action
        obs, reward, terminated, truncated, info = env.step(action)
        shifts.append(int(info.get("shifts_active", 1)))
        if keep_rows:
            rows.append(
                row_from_step(
                    policy=policy_name,
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                    condition=condition,
                    step=step,
                    obs_before=obs_before,
                    reward=float(reward),
                    info=info,
                )
            )
        if keep_obs:
            obs_samples.append(obs_before)
        step += 1

    ret_excel, fill_rate, service_loss_auc, cost_index = finalize_episode(env, shifts)
    env.close()
    anchor = infer_event_anchor(rows) if anchor_step is None else int(anchor_step)
    for row in rows:
        row["event_anchor_step"] = anchor
        row["relative_week"] = int(row["step"]) - anchor
        row["event_window"] = window_label(int(row["relative_week"]))
    return EpisodeRun(
        policy=policy_name,
        seed=seed,
        episode=episode,
        eval_seed=eval_seed,
        condition=condition,
        rows=rows,
        obs_samples=obs_samples,
        ret_excel=ret_excel,
        fill_rate=fill_rate,
        service_loss_auc=service_loss_auc,
        cost_index=cost_index,
        anchor_step=anchor,
    )


def window_label(relative_week: int) -> str:
    if -4 <= relative_week <= -1:
        return "pre"
    if relative_week == 0:
        return "event"
    if 1 <= relative_week <= 8:
        return "post"
    return "baseline"


def collect_forecast_bank(args: argparse.Namespace, seeds: list[int], eval_episodes: int) -> np.ndarray:
    fields = tuple(get_observation_fields("v7"))
    idx_48 = fields.index(FORECAST_FIELD_NAMES[0])
    idx_168 = fields.index(FORECAST_FIELD_NAMES[1])
    neutral = build_static_policy_action(NEUTRAL_STATIC)
    rows: list[list[float]] = []
    for seed in seeds:
        for episode_idx in range(eval_episodes):
            eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
            env = make_track_b_env(**env_kwargs(args))
            obs, _ = env.reset(seed=eval_seed)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                obs_arr = np.asarray(obs, dtype=np.float32)
                rows.append([float(obs_arr[idx_48]), float(obs_arr[idx_168])])
                obs, _reward, terminated, truncated, _info = env.step(neutral)
            env.close()
    return np.asarray(rows, dtype=np.float32)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def group_event_study(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        rel = int(row.get("relative_week", 999))
        if -4 <= rel <= 8:
            groups.setdefault((str(row["policy"]), rel), []).append(row)
    out: list[dict[str, Any]] = []
    for (policy, rel), rows in sorted(groups.items()):
        out.append(
            {
                "policy": policy,
                "relative_week": rel,
                "n": len(rows),
                "action_intensity_mean": mean([float(r["action_intensity"]) for r in rows]),
                "shift_mean": mean([float(r["shift"]) for r in rows]),
                "op10_multiplier_mean": mean([float(r["op10_multiplier"]) for r in rows]),
                "op12_multiplier_mean": mean([float(r["op12_multiplier"]) for r in rows]),
                "forecast_168h_mean": mean([float(r["forecast_168h"]) for r in rows]),
                "pending_backorder_qty_mean": mean([float(r["pending_backorder_qty"]) for r in rows]),
                "rolling_fill_rate_4w_mean": mean([float(r["rolling_fill_rate_4w"]) for r in rows]),
            }
        )
    return out


def _ols_rss(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(np.sum(resid**2)), int(x.shape[1])


def granger_f_test(y: np.ndarray, x: np.ndarray, *, lag: int) -> dict[str, float | str]:
    """Small OLS/F-test implementation for Granger-style lead-lag screening.

    Full model: y_t ~ y_{t-1..lag} + x_{t-1..lag}
    Restricted: y_t ~ y_{t-1..lag}

    This is used as a directional lead-lag screen, not as proof of physical
    causality. Episodes are tested separately before aggregation.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n = len(y)
    if lag < 1 or n <= (2 * lag + 2):
        return {"f_stat": "", "p_value": "", "df_num": "", "df_den": "", "status": "insufficient_n"}
    if float(np.std(y)) <= 1e-12 or float(np.std(x)) <= 1e-12:
        return {"f_stat": "", "p_value": "", "df_num": "", "df_den": "", "status": "constant_series"}

    targets: list[float] = []
    restricted_rows: list[list[float]] = []
    full_rows: list[list[float]] = []
    for t in range(lag, n):
        y_lags = [float(y[t - k]) for k in range(1, lag + 1)]
        x_lags = [float(x[t - k]) for k in range(1, lag + 1)]
        targets.append(float(y[t]))
        restricted_rows.append([1.0, *y_lags])
        full_rows.append([1.0, *y_lags, *x_lags])

    y_target = np.asarray(targets, dtype=float)
    xr = np.asarray(restricted_rows, dtype=float)
    xf = np.asarray(full_rows, dtype=float)
    rss_r, _ = _ols_rss(y_target, xr)
    rss_f, p_full = _ols_rss(y_target, xf)
    df_num = int(lag)
    df_den = int(len(y_target) - p_full)
    if df_den <= 0 or rss_f <= 1e-15:
        return {"f_stat": "", "p_value": "", "df_num": df_num, "df_den": df_den, "status": "degenerate"}
    f_stat = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    f_stat = max(0.0, float(f_stat))
    p_value = float(stats.f.sf(f_stat, df_num, df_den))
    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "df_num": df_num,
        "df_den": df_den,
        "status": "ok",
    }


def lead_lag_signal_rows(step_rows: list[dict[str, Any]], *, max_lag: int) -> list[dict[str, Any]]:
    """Run directional lead-lag screens per policy/episode and signal pair."""
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        grouped.setdefault(
            (str(row["policy"]), int(row["seed"]), int(row["episode"])),
            [],
        ).append(row)

    pairs = (
        (
            "action_leads_risk_phase",
            "risk_phase",
            "action_intensity",
            "action lags help predict future risk phase; anticipatory alignment screen only",
        ),
        (
            "risk_phase_leads_action",
            "action_intensity",
            "risk_phase",
            "risk phase lags help predict future action; reactive screen",
        ),
        (
            "forecast_leads_action",
            "action_intensity",
            "forecast_168h",
            "forecast lags help predict future action; forecast-use screen",
        ),
        (
            "backlog_leads_action",
            "action_intensity",
            "backlog_signal",
            "backlog/service lags help predict future action; reactive recovery screen",
        ),
    )
    out: list[dict[str, Any]] = []
    for (policy, seed, episode), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: int(r["step"]))
        signals = {
            "action_intensity": np.asarray([float(r["action_intensity"]) for r in rows]),
            "risk_phase": np.asarray(
                [
                    max(
                        float(r.get("regime_pre_disruption", 0.0)),
                        float(r.get("regime_disrupted", 0.0)),
                        float(r.get("regime_recovery", 0.0)),
                    )
                    for r in rows
                ]
            ),
            "forecast_168h": np.asarray([float(r.get("forecast_168h", 0.0)) for r in rows]),
            "backlog_signal": np.asarray(
                [
                    max(
                        float(r.get("backlog_age_norm", 0.0)),
                        1.0 - float(r.get("rolling_fill_rate_4w", 1.0)),
                    )
                    for r in rows
                ]
            ),
        }
        for lag in range(1, max(1, int(max_lag)) + 1):
            for label, target, predictor, meaning in pairs:
                result = granger_f_test(signals[target], signals[predictor], lag=lag)
                out.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "episode": episode,
                        "lag": lag,
                        "test": label,
                        "target": target,
                        "predictor": predictor,
                        "meaning": meaning,
                        **result,
                    }
                )
    return out


def summarize_lead_lag(lead_lag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in lead_lag_rows:
        if row.get("status") == "ok" and row.get("p_value") != "":
            groups.setdefault((str(row["policy"]), str(row["test"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (policy, test), rows in sorted(groups.items()):
        p_values = [float(r["p_value"]) for r in rows]
        f_values = [float(r["f_stat"]) for r in rows]
        out.append(
            {
                "policy": policy,
                "test": test,
                "n_episode_lag_tests": len(rows),
                "min_p_value": min(p_values) if p_values else "",
                "median_p_value": statistics.median(p_values) if p_values else "",
                "mean_f_stat": mean(f_values),
                "share_p_lt_0p05": mean([1.0 if p < 0.05 else 0.0 for p in p_values]),
            }
        )
    return out


def cross_correlation_rows(step_rows: list[dict[str, Any]], *, max_lag: int = 8) -> list[dict[str, Any]]:
    """Lagged Pearson correlations for visual/mechanism screening.

    Positive lag means the predictor is observed before the target:
    corr(target[t], predictor[t-lag]).  This is descriptive only; the
    counterfactual reset remains the causal/value test.
    """
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        grouped.setdefault(
            (str(row["policy"]), int(row["seed"]), int(row["episode"])),
            [],
        ).append(row)
    pairs = (
        ("forecast_to_action", "action_intensity", "forecast_168h"),
        ("risk_phase_to_action", "action_intensity", "risk_phase"),
        ("backlog_to_action", "action_intensity", "backlog_signal"),
        ("action_to_future_risk_phase", "risk_phase", "action_intensity"),
    )
    out: list[dict[str, Any]] = []
    for (policy, seed, episode), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: int(r["step"]))
        signals = {
            "action_intensity": np.asarray([float(r["action_intensity"]) for r in rows]),
            "risk_phase": np.asarray(
                [
                    max(
                        float(r.get("regime_pre_disruption", 0.0)),
                        float(r.get("regime_disrupted", 0.0)),
                        float(r.get("regime_recovery", 0.0)),
                    )
                    for r in rows
                ]
            ),
            "forecast_168h": np.asarray([float(r.get("forecast_168h", 0.0)) for r in rows]),
            "backlog_signal": np.asarray(
                [
                    max(
                        float(r.get("backlog_age_norm", 0.0)),
                        1.0 - float(r.get("rolling_fill_rate_4w", 1.0)),
                    )
                    for r in rows
                ]
            ),
        }
        for label, target_name, predictor_name in pairs:
            target = signals[target_name]
            predictor = signals[predictor_name]
            for lag in range(-int(max_lag), int(max_lag) + 1):
                if lag > 0:
                    y = target[lag:]
                    x = predictor[:-lag]
                elif lag < 0:
                    y = target[:lag]
                    x = predictor[-lag:]
                else:
                    y = target
                    x = predictor
                if len(y) < 3 or float(np.std(y)) <= 1e-12 or float(np.std(x)) <= 1e-12:
                    corr: float | str = ""
                else:
                    corr = float(np.corrcoef(y, x)[0, 1])
                out.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "episode": episode,
                        "pair": label,
                        "target": target_name,
                        "predictor": predictor_name,
                        "lag": lag,
                        "corr": corr,
                    }
                )
    return out


def summarize_cross_correlation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        if row.get("corr") != "":
            groups.setdefault(
                (str(row["policy"]), str(row["pair"]), int(row["lag"])),
                [],
            ).append(float(row["corr"]))
    out: list[dict[str, Any]] = []
    for (policy, pair, lag), values in sorted(groups.items()):
        out.append(
            {
                "policy": policy,
                "pair": pair,
                "lag": lag,
                "corr_mean": mean(values),
                "corr_median": statistics.median(values),
                "n": len(values),
            }
        )
    return out


def aggregate_policy_windows(
    step_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    forecast_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    policies = sorted({str(row["policy"]) for row in step_rows})
    out: list[dict[str, Any]] = []
    for policy in policies:
        rows = [r for r in step_rows if r["policy"] == policy]
        base = [float(r["action_intensity"]) for r in rows if r.get("event_window") == "baseline"]
        pre = [float(r["action_intensity"]) for r in rows if r.get("event_window") == "pre"]
        post = [float(r["action_intensity"]) for r in rows if r.get("event_window") == "post"]
        cf = [r for r in counterfactual_rows if r["policy"] == policy]
        delta_by_window = {
            str(w): mean([float(r["delta_ret_excel"]) for r in cf if r["reset_window"] == w])
            for w in ("pre", "event", "post")
        }
        frows = [r for r in forecast_rows if r["policy"] == policy]
        full_ret = mean([float(r["ret_excel_mean"]) for r in frows if r["forecast_condition"] == "full"])
        zero_ret = mean([float(r["ret_excel_mean"]) for r in frows if r["forecast_condition"] == "zeroed"])
        scrambled_ret = mean([float(r["ret_excel_mean"]) for r in frows if r["forecast_condition"] == "scrambled"])
        pai = mean(pre) - mean(base)
        rri = mean(post) - mean(pre)
        rei = delta_by_window["post"]
        preventive = pai > 0.02 and delta_by_window["pre"] > 1e-7
        reactive = rri > 0.02 and delta_by_window["post"] > 1e-7
        if preventive and reactive:
            classification = "mixta"
        elif preventive:
            classification = "preventiva"
        elif reactive:
            classification = "reactiva"
        else:
            classification = "sin señal clara"
        out.append(
            {
                "policy": policy,
                "preventive_activation_index": pai,
                "reactive_response_index": rri,
                "recovery_effect_index": rei,
                "delta_ret_reset_pre": delta_by_window["pre"],
                "delta_ret_reset_event": delta_by_window["event"],
                "delta_ret_reset_post": delta_by_window["post"],
                "forecast_reliance_zeroed": full_ret - zero_ret if zero_ret else "",
                "forecast_reliance_scrambled": full_ret - scrambled_ret if scrambled_ret else "",
                "classification": classification,
            }
        )
    return out


def action_sensitivity(
    runtime: PolicyRuntime,
    obs_samples: list[np.ndarray],
    args: argparse.Namespace,
    *,
    max_samples: int,
) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return []
    fields = tuple(get_observation_fields("v7"))
    groups = {
        "forecast": [name for name in FORECAST_FIELD_NAMES if name in fields],
        "regime": [name for name in REGIME_FIELD_NAMES if name in fields],
        "downstream_pressure": [
            name for name in ("op10_queue_pressure_norm", "op12_queue_pressure_norm", "op10_down", "op12_down")
            if name in fields
        ],
        "backlog_service": [
            name for name in ("fill_rate", "rolling_fill_rate_4w", "backlog_age_norm")
            if name in fields
        ],
    }
    samples = obs_samples[:max_samples]
    rows: list[dict[str, Any]] = []
    for group, names in groups.items():
        if not names:
            continue
        idxs = [fields.index(name) for name in names]
        deltas: list[float] = []
        action_delta_by_dim = {name: [] for name in ACTION_DIMS}
        for obs in samples:
            original = predict_action(runtime, obs)
            occluded = np.array(obs, dtype=np.float32, copy=True)
            for idx in idxs:
                occluded[idx] = 0.0
            perturbed = predict_action(runtime, occluded)
            delta = np.asarray(original, dtype=float) - np.asarray(perturbed, dtype=float)
            deltas.append(float(np.linalg.norm(delta)))
            for i, dim in enumerate(ACTION_DIMS):
                if i < len(delta):
                    action_delta_by_dim[dim].append(abs(float(delta[i])))
        row = {
            "policy": runtime.name,
            "seed": runtime.seed,
            "feature_group": group,
            "n_samples": len(samples),
            "action_l2_change_mean": mean(deltas),
        }
        for dim, vals in action_delta_by_dim.items():
            row[f"{dim}_abs_change_mean"] = mean(vals)
        rows.append(row)
    return rows


def write_verdict(path: Path, summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Auditoria Track B: preventivo vs reactivo",
        "",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Esta auditoria es eval-only: no reentrena modelos. Repite episodios CRN, registra acciones semanales, reemplaza acciones por una politica neutral en ventanas pre/evento/post y mide `R_full - R_reset` con ReT Excel/Garrido.",
        "",
        "Regla de lectura: no llamamos preventivo a un modelo salvo que haya activacion antes del riesgo y contribucion positiva antes del riesgo bajo `R_full - R_reset`.",
        "",
        "Nota de validacion: antes de usar estas etiquetas como claim final, correr `scripts/audit_track_b_prevention_sanity_check.py`. Si las heuristicas de referencia no pasan, leer las etiquetas como descriptivas/provisionales y reportar solo patrones de accion, valor por ventana y dependencia del forecast.",
        "",
        "## Resultado",
        "",
        "| Politica | PAI | RRI | REI | Delta pre | Delta post | Forecast reliance scrambled | Clasificacion |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {pai:.6f} | {rri:.6f} | {rei:.6f} | {dpre:.6f} | {dpost:.6f} | {frs} | {cls} |".format(
                policy=row["policy"],
                pai=float(row["preventive_activation_index"]),
                rri=float(row["reactive_response_index"]),
                rei=float(row["recovery_effect_index"]),
                dpre=float(row["delta_ret_reset_pre"]),
                dpost=float(row["delta_ret_reset_post"]),
                frs=(
                    f"{float(row['forecast_reliance_scrambled']):.6f}"
                    if row.get("forecast_reliance_scrambled") != ""
                    else "n/a"
                ),
                cls=row["classification"],
            )
        )
    lines.extend(
        [
            "",
            "## Salidas",
            "",
            "- `step_ledger.csv`: una fila por decision semanal.",
            "- `event_study.csv`: acciones promedio alrededor del evento (`t=-4...+8`).",
            "- `counterfactual_reset.csv`: `R_full - R_reset` por ventana.",
            "- `forecast_ablation.csv`: full vs forecast zeroed/scrambled.",
            "- `feature_attribution.csv`: sensibilidad por oclusion de grupos de features.",
            "- `lead_lag_tests.csv`: tests direccionales tipo Granger por episodio y rezago.",
            "- `lead_lag_summary.csv`: resumen por politica y direccion de senal.",
            "- `cross_correlation.csv`: correlaciones con rezago por episodio.",
            "- `cross_correlation_summary.csv`: resumen de correlaciones por politica/par/rezago.",
            "- `summary.json`: configuracion y resumen mecanico.",
            "",
            "## Fuentes metodologicas",
            "",
            "- RUDDER/return decomposition: credito temporal para recompensas retrasadas.",
            "- Integrated Gradients y SHAP: inspiracion para atribucion; aqui usamos oclusion/sensibilidad como metodo primario robusto.",
            "- Cautela con atencion: no tratamos atencion como explicacion suficiente por si sola.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    neutral_action = build_static_policy_action(static_spec_from_label(args.reset_policy))
    seeds = list(args.seeds)
    episodes = list(range(1, int(args.eval_episodes) + 1))

    forecast_bank: np.ndarray | None = None
    if not args.skip_forecast_ablation:
        forecast_bank = collect_forecast_bank(args, seeds[: max(1, min(2, len(seeds)))], min(2, len(episodes)))

    step_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    forecast_eval_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    policy_episode_rows: list[dict[str, Any]] = []

    for policy in args.policies:
        runtimes = {seed: load_runtime(policy, seed, args) for seed in seeds}
        for seed, runtime in runtimes.items():
            obs_for_attr: list[np.ndarray] = []
            for episode in episodes:
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                full = run_episode(
                    runtime=runtime,
                    args=args,
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                    policy_name=policy,
                    neutral_action=neutral_action,
                    keep_rows=True,
                    keep_obs=True,
                )
                step_rows.extend(full.rows)
                obs_for_attr.extend(full.obs_samples)
                policy_episode_rows.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "episode": episode,
                        "eval_seed": eval_seed,
                        "ret_excel": full.ret_excel,
                        "fill_rate": full.fill_rate,
                        "service_loss_auc": full.service_loss_auc,
                        "assembly_cost_index": full.cost_index,
                        "anchor_step": full.anchor_step,
                    }
                )
                for label, window in {
                    "pre": (-4, -1),
                    "event": (0, 0),
                    "post": (1, 8),
                }.items():
                    reset = run_episode(
                        runtime=runtime,
                        args=args,
                        seed=seed,
                        episode=episode,
                        eval_seed=eval_seed,
                        policy_name=policy,
                        neutral_action=neutral_action,
                        reset_window=window,
                        anchor_step=full.anchor_step,
                        keep_rows=False,
                        keep_obs=False,
                    )
                    counterfactual_rows.append(
                        {
                            "policy": policy,
                            "seed": seed,
                            "episode": episode,
                            "eval_seed": eval_seed,
                            "reset_window": label,
                            "window_start": window[0],
                            "window_end": window[1],
                            "R_full": full.ret_excel,
                            "R_reset": reset.ret_excel,
                            "delta_ret_excel": full.ret_excel - reset.ret_excel,
                        }
                    )

            if not args.skip_attribution:
                attribution_rows.extend(
                    action_sensitivity(
                        runtime,
                        obs_for_attr,
                        args,
                        max_samples=int(args.attribution_samples),
                    )
                )

        if not args.skip_forecast_ablation:
            for condition in ("full", "zeroed", "scrambled"):
                rets: list[float] = []
                costs: list[float] = []
                for seed, runtime in runtimes.items():
                    for episode in episodes:
                        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                        run = run_episode(
                            runtime=runtime,
                            args=args,
                            seed=seed,
                            episode=episode,
                            eval_seed=eval_seed,
                            policy_name=policy,
                            neutral_action=neutral_action,
                            condition=condition,
                            forecast_bank=forecast_bank,
                            keep_rows=False,
                            keep_obs=False,
                        )
                        rets.append(run.ret_excel)
                        costs.append(run.cost_index)
                forecast_eval_rows.append(
                    {
                        "policy": policy,
                        "forecast_condition": condition,
                        "n": len(rets),
                        "ret_excel_mean": mean(rets),
                        "assembly_cost_index_mean": mean(costs),
                    }
                )

    event_study_rows = group_event_study(step_rows)
    lead_lag_rows = lead_lag_signal_rows(step_rows, max_lag=int(args.max_granger_lag))
    lead_lag_summary_rows = summarize_lead_lag(lead_lag_rows)
    cross_corr_rows = cross_correlation_rows(step_rows, max_lag=8)
    cross_corr_summary_rows = summarize_cross_correlation(cross_corr_rows)
    summary_rows = aggregate_policy_windows(step_rows, counterfactual_rows, forecast_eval_rows)

    save_csv(out / "step_ledger.csv", step_rows)
    save_csv(out / "event_study.csv", event_study_rows)
    save_csv(out / "counterfactual_reset.csv", counterfactual_rows)
    save_csv(out / "forecast_ablation.csv", forecast_eval_rows)
    save_csv(out / "feature_attribution.csv", attribution_rows)
    save_csv(out / "lead_lag_tests.csv", lead_lag_rows)
    save_csv(out / "lead_lag_summary.csv", lead_lag_summary_rows)
    save_csv(out / "cross_correlation.csv", cross_corr_rows)
    save_csv(out / "cross_correlation_summary.csv", cross_corr_summary_rows)
    save_csv(out / "policy_episode_metrics.csv", policy_episode_rows)
    save_csv(out / "policy_classification.csv", summary_rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "policies": args.policies,
            "seeds": seeds,
            "eval_episodes": int(args.eval_episodes),
            "eval_seed_offset": EVAL_EPISODE_SEED_OFFSET,
            "reward_mode": args.reward_mode,
            "risk_level": args.risk_level,
            "observation_version": args.observation_version,
            "action_contract": "track_b_v1",
            "reset_policy": args.reset_policy,
            "counterfactual_windows": {
                "pre": [-4, -1],
                "event": [0, 0],
                "post": [1, 8],
            },
        },
        "policy_classification": summary_rows,
        "row_counts": {
            "step_ledger": len(step_rows),
            "event_study": len(event_study_rows),
            "counterfactual_reset": len(counterfactual_rows),
            "forecast_ablation": len(forecast_eval_rows),
            "feature_attribution": len(attribution_rows),
            "lead_lag_tests": len(lead_lag_rows),
            "lead_lag_summary": len(lead_lag_summary_rows),
            "cross_correlation": len(cross_corr_rows),
            "cross_correlation_summary": len(cross_corr_summary_rows),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_verdict(out / "verdict.md", summary_rows, args)
    print(json.dumps(summary["row_counts"], indent=2))
    print(f"Wrote audit bundle: {out}")


if __name__ == "__main__":
    main()
