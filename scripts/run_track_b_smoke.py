#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mfsc_runtime_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np

try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - runtime guard for optional dependency.
    RecurrentPPO = None
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_control_reward import build_metric_contract_metadata
from scripts.track_b_heuristics import HEURISTIC_POLICY_NAMES, make_heuristic_defaults
from supply_chain.config import (
    OPERATIONS,
    THESIS_FAITHFUL_PROTOCOL,
    TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE,
    TRACK_A_TRAINING_RAW_MATERIAL_ORDER_UP_TO_MULTIPLIER,
    canonical_raw_material_flow_mode,
)
from supply_chain.dkana import DKANAOnlinePolicyAdapter, DKANAPolicy
from supply_chain.episode_metrics import METRIC_KEYS, compute_episode_metrics
from supply_chain.env_experimental_shifts import (
    OBSERVATION_VERSION_OPTIONS,
    REWARD_MODE_OPTIONS,
)
from supply_chain.ret_thesis import (
    compute_ret_per_order_excel_formula,
    order_counts_as_backorder_for_fill_rate,
)
from supply_chain.external_env_interface import (
    STATE_CONSTRAINT_FIELDS,
    get_episode_terminal_metrics,
    get_track_b_env_spec,
    make_track_b_env,
    spec_to_dict,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks")
DEFAULT_SEEDS = (11, 22, 33)
DEFAULT_TRAIN_TIMESTEPS = 100_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_MAX_STEPS = 260
DEFAULT_RET_SEQ_KAPPA = 0.20
EVAL_EPISODE_SEED_OFFSET = 50_000
DOWNSTREAM_NEAR_MAX_THRESHOLD = 1.90
RISK_COLS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24", "R3")

PRIMARY_METRICS = (
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
    "op10_multiplier_step_p95",
    "op12_multiplier_step_p95",
    "pct_steps_op10_multiplier_ge_190",
    "pct_steps_op12_multiplier_ge_190",
    "pct_steps_both_downstream_ge_190",
    "assembly_hours_total",
    "assembly_cost_index",
    "ret_garrido2024_raw_total",
    "ret_garrido2024_train_total",
    "ret_garrido2024_sigmoid_total",
    "ret_garrido2024_sigmoid_mean",
    "terminal_zeta_avg",
    "terminal_epsilon_avg",
    "terminal_phi_avg",
    "terminal_tau_avg",
    "terminal_kappa_dot",
    *tuple(f"order_{key}" for key in METRIC_KEYS),
)

HOURS_PER_SHIFT = 8.0

EPISODE_FIELDS = [
    "policy",
    "seed",
    "episode",
    "eval_seed",
    "steps",
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
    "op10_multiplier_step_p95",
    "op12_multiplier_step_p95",
    "pct_steps_op10_multiplier_ge_190",
    "pct_steps_op12_multiplier_ge_190",
    "pct_steps_both_downstream_ge_190",
    "assembly_hours_total",
    "assembly_cost_index",
]

COMPARISON_FIELDS = [
    "reward_mode",
    "reward_family",
    "action_contract",
    "observation_version",
    "risk_level",
    "learned_policy",
    "baseline_policy",
    "best_static_policy",
    "learned_reward_mean",
    "learned_fill_rate_mean",
    "learned_backorder_rate_mean",
    "learned_order_level_ret_mean",
    "baseline_reward_mean",
    "baseline_fill_rate_mean",
    "baseline_backorder_rate_mean",
    "baseline_order_level_ret_mean",
    "best_static_reward_mean",
    "best_static_fill_rate_mean",
    "best_static_backorder_rate_mean",
    "best_static_order_level_ret_mean",
    "learned_fill_gap_vs_baseline_pp",
    "learned_fill_gap_vs_best_static_pp",
    "learned_reward_gap_vs_best_static",
    "learned_order_level_ret_gap_vs_best_static",
    "learned_beats_s2_neutral_by_fill",
    "learned_matches_best_static_by_fill",
    "promote_to_long_run",
]


@dataclass(frozen=True)
class StaticPolicySpec:
    label: str
    assembly_shifts: int
    downstream_multiplier: float


STATIC_POLICY_SPECS: tuple[StaticPolicySpec, ...] = (
    StaticPolicySpec(label="s1_d1.00", assembly_shifts=1, downstream_multiplier=1.0),
    StaticPolicySpec(label="s1_d1.50", assembly_shifts=1, downstream_multiplier=1.5),
    StaticPolicySpec(label="s1_d2.00", assembly_shifts=1, downstream_multiplier=2.0),
    StaticPolicySpec(label="s2_d1.00", assembly_shifts=2, downstream_multiplier=1.0),
    StaticPolicySpec(label="s2_d1.50", assembly_shifts=2, downstream_multiplier=1.5),
    StaticPolicySpec(label="s2_d2.00", assembly_shifts=2, downstream_multiplier=2.0),
    StaticPolicySpec(label="s3_d1.00", assembly_shifts=3, downstream_multiplier=1.0),
    StaticPolicySpec(label="s3_d1.50", assembly_shifts=3, downstream_multiplier=1.5),
    StaticPolicySpec(label="s3_d2.00", assembly_shifts=3, downstream_multiplier=2.0),
)

STATIC_POLICY_ORDER = tuple(policy.label for policy in STATIC_POLICY_SPECS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a short PPO smoke test on the minimal Track B environment "
            "and compare it against the strongest static policies from the DOE."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the smoke benchmark bundle.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds. One PPO model is trained per seed.",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=DEFAULT_TRAIN_TIMESTEPS,
        help="Total PPO timesteps per seed.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Evaluation episodes per policy and seed.",
    )
    parser.add_argument(
        "--export-order-ledger",
        action="store_true",
        help=(
            "Write a Garrido-style per-order ledger CSV for evaluated policies. "
            "This can be large, so it is opt-in for confirmatory/audit runs."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        default="ReT_seq_v1",
        choices=list(REWARD_MODE_OPTIONS),
        help="Track B training reward.",
    )
    parser.add_argument(
        "--ret-seq-kappa",
        type=float,
        default=DEFAULT_RET_SEQ_KAPPA,
        help="ReT_seq_v1 kappa. Ignored by other reward modes.",
    )
    parser.add_argument(
        "--ret-excel-cvar-alpha",
        type=float,
        default=0.5,
        help="Penalty weight for ReT_excel_plus_cvar.",
    )
    parser.add_argument(
        "--ret-excel-cvar-tail-level",
        type=float,
        default=0.05,
        help="Tail quantile used by ReT_excel_plus_cvar.",
    )
    parser.add_argument(
        "--ret-excel-cvar-window",
        type=int,
        default=50,
        help="Rolling service-loss window for ReT_excel_plus_cvar.",
    )
    parser.add_argument(
        "--risk-level",
        default="adaptive_benchmark_v2",
        help="Track B risk profile.",
    )
    parser.add_argument(
        "--enabled-risks",
        default=None,
        help=(
            "Optional comma-separated risk IDs to enable, e.g. R21,R22,R23,R24. "
            "Leave unset to use the risk profile default."
        ),
    )
    parser.add_argument(
        "--risk-frequency-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for non-adaptive risk frequency in controlled Track B screens.",
    )
    parser.add_argument(
        "--risk-impact-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for non-adaptive risk impact/duration in controlled Track B screens.",
    )
    parser.add_argument(
        "--risk-frequency-by-id",
        default=None,
        help=(
            "Optional comma-separated per-risk frequency multipliers, e.g. "
            "R22=1.0,R23=1.0,R24=2.0. Overrides the global frequency "
            "multiplier for listed risk IDs."
        ),
    )
    parser.add_argument(
        "--risk-impact-by-id",
        default=None,
        help=(
            "Optional comma-separated per-risk impact multipliers, e.g. "
            "R22=1.5,R23=1.5,R24=1.25. Overrides the global impact "
            "multiplier for listed risk IDs."
        ),
    )
    parser.add_argument(
        "--demand-mean-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for mean daily demand; useful for downstream headroom screens.",
    )
    parser.add_argument(
        "--observation-version",
        choices=list(OBSERVATION_VERSION_OPTIONS),
        default="v7",
        help="Observation contract for Track B training/eval.",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "recurrent_ppo", "sac", "td3"],
        default="ppo",
        help="Learned-policy algorithm for the Track B adaptive lane.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=168.0,
        help="Decision cadence in hours.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode horizon in decision steps.",
    )
    parser.add_argument(
        "--eval-risk-levels",
        nargs="*",
        default=None,
        help=(
            "Additional risk levels for cross-scenario evaluation of the trained "
            "model. E.g. --eval-risk-levels current increased severe. "
            "The model is always trained on --risk-level; these are eval-only."
        ),
    )
    parser.add_argument(
        "--dkana-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a DKANA .pt checkpoint. If provided, a DKANA lane "
            "is evaluated alongside statics, heuristics, and PPO under the same "
            "Track B contract and CI95 aggregation."
        ),
    )
    parser.add_argument(
        "--faithful",
        action="store_true",
        help=(
            "Run Track B under the THESIS_FAITHFUL protocol (delay=54, thesis_window, "
            "figure_6_2, kit_equivalent m2.0, r14_strict, warmup op9_arrival, stochastic_pt off) "
            "instead of the legacy adaptive_benchmark Track B defaults. Garrido-comparable lane "
            "(Contract v2)."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="PPO entropy coefficient. Defaults to 0.0 to preserve historical Track B protocol.",
    )
    parser.add_argument(
        "--norm-reward",
        action="store_true",
        help=(
            "Enable VecNormalize reward normalization during training. "
            "Evaluation metrics are still reported on the original environment scale."
        ),
    )
    parser.add_argument(
        "--clip-reward",
        type=float,
        default=10.0,
        help="Reward clipping threshold used by VecNormalize when --norm-reward is enabled.",
    )
    parser.add_argument(
        "--surge-inertia",
        action="store_true",
        help=(
            "Enable capacity activation lag in the Track B environment. Requested "
            "shift increases ramp toward the target instead of applying instantly; "
            "used only for prevention diagnostics, not the canonical Track B lane."
        ),
    )
    parser.add_argument(
        "--surge-ramp-per-step",
        type=int,
        default=1,
        help="Maximum upward shift-level increase per decision step when --surge-inertia is enabled.",
    )
    parser.add_argument(
        "--surge-budget-hours",
        type=float,
        default=float("inf"),
        help=(
            "Episode surge-hour budget when --surge-inertia is enabled. "
            "S2 costs one step of surge-hours; S3 costs two. Use inf for lag-only."
        ),
    )
    return parser


def default_output_dir(train_timesteps: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"track_b_smoke_{train_timesteps}_{timestamp}"


def learned_policy_name(args: argparse.Namespace | None = None) -> str:
    return str(getattr(args, "algo", "ppo")) if args is not None else "ppo"


def model_filename(args: argparse.Namespace | None = None) -> str:
    policy = learned_policy_name(args)
    return "ppo_model.zip" if policy == "ppo" else f"{policy}_model.zip"


def ensure_algo_dependencies(args: argparse.Namespace) -> None:
    if learned_policy_name(args) == "recurrent_ppo" and RecurrentPPO is None:
        raise ImportError(
            "Track B recurrent_ppo requires sb3-contrib. Install requirements.txt."
        )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    mean = arr.mean()
    return float(mean - half), float(mean + half)


def parse_risk_multiplier_map(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    parsed: dict[str, float] = {}
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid risk multiplier token {token!r}; expected R22=1.5"
            )
        risk_id, value = token.split("=", 1)
        risk_id = risk_id.strip()
        if not risk_id:
            raise ValueError(f"Invalid empty risk id in {token!r}")
        parsed[risk_id] = max(1e-6, float(value))
    return parsed


def build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    risk_frequency_by_id = parse_risk_multiplier_map(
        getattr(args, "risk_frequency_by_id", None)
    )
    risk_impact_by_id = parse_risk_multiplier_map(
        getattr(args, "risk_impact_by_id", None)
    )
    kwargs: dict[str, Any] = {
        "reward_mode": args.reward_mode,
        "ret_seq_kappa": args.ret_seq_kappa,
        "ret_excel_cvar_alpha": args.ret_excel_cvar_alpha,
        "ret_excel_cvar_tail_level": args.ret_excel_cvar_tail_level,
        "ret_excel_cvar_window": args.ret_excel_cvar_window,
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "risk_frequency_multiplier": args.risk_frequency_multiplier,
        "risk_impact_multiplier": args.risk_impact_multiplier,
        "risk_frequency_multipliers_by_id": risk_frequency_by_id,
        "risk_impact_multipliers_by_id": risk_impact_by_id,
        "demand_mean_multiplier": args.demand_mean_multiplier,
        "raw_material_flow_mode": TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE,
        "raw_material_order_up_to_multiplier": (
            TRACK_A_TRAINING_RAW_MATERIAL_ORDER_UP_TO_MULTIPLIER
        ),
    }
    if getattr(args, "surge_inertia", False):
        kwargs.update(
            {
                "surge_inertia": True,
                "surge_ramp_per_step": int(args.surge_ramp_per_step),
                "surge_budget_hours": float(args.surge_budget_hours),
            }
        )
    if args.enabled_risks:
        kwargs["enabled_risks"] = tuple(
            risk.strip() for risk in str(args.enabled_risks).split(",") if risk.strip()
        )
    if getattr(args, "faithful", False):
        P = THESIS_FAITHFUL_PROTOCOL
        # Garrido-comparable Track B lane: overlay the faithful protocol on top of the
        # track_b_v1 downstream-control action space. Mirrors make_thesis_aligned_training_env.
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


def build_static_policy_action(policy: StaticPolicySpec) -> dict[str, float | int]:
    downstream_multiplier = float(policy.downstream_multiplier)
    return {
        "op3_q": float(OPERATIONS[3]["q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]),
        "op9_q_max": float(OPERATIONS[9]["q"][1]),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "op10_q_min": float(OPERATIONS[10]["q"][0]) * downstream_multiplier,
        "op10_q_max": float(OPERATIONS[10]["q"][1]) * downstream_multiplier,
        "op12_q_min": float(OPERATIONS[12]["q"][0]) * downstream_multiplier,
        "op12_q_max": float(OPERATIONS[12]["q"][1]) * downstream_multiplier,
        "assembly_shifts": int(policy.assembly_shifts),
    }


def extract_downstream_multipliers(final_info: dict[str, Any]) -> tuple[float, float]:
    clipped_action = final_info.get("clipped_action")
    if isinstance(clipped_action, (list, tuple)) and len(clipped_action) >= 8:
        return (
            float(1.25 + 0.75 * float(clipped_action[6])),
            float(1.25 + 0.75 * float(clipped_action[7])),
        )

    raw_action = final_info.get("raw_action")
    if isinstance(raw_action, dict):
        op10_base = float(OPERATIONS[10]["q"][0])
        op12_base = float(OPERATIONS[12]["q"][0])
        return (
            float(raw_action.get("op10_q_min", op10_base)) / op10_base,
            float(raw_action.get("op12_q_min", op12_base)) / op12_base,
        )

    return 1.0, 1.0


def init_cd_totals() -> dict[str, float]:
    return {
        "ret_garrido2024_raw_total": 0.0,
        "ret_garrido2024_train_total": 0.0,
        "ret_garrido2024_sigmoid_total": 0.0,
    }


def update_cd_totals(cd_totals: dict[str, float], info: dict[str, Any]) -> None:
    cd_totals["ret_garrido2024_raw_total"] += float(
        info.get("ret_garrido2024_raw_step", 0.0)
    )
    cd_totals["ret_garrido2024_train_total"] += float(
        info.get("ret_garrido2024_train_step", 0.0)
    )
    cd_totals["ret_garrido2024_sigmoid_total"] += float(
        info.get("ret_garrido2024_sigmoid_step", 0.0)
    )


def append_order_ledger_rows(
    ledger_rows: list[dict[str, Any]] | None,
    env: Any,
    *,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
) -> None:
    if ledger_rows is None:
        return
    sim = env.unwrapped.sim
    start = float(getattr(sim, "warmup_time", 0.0) or 0.0)
    orders = sorted(
        (
            o
            for o in sim.orders
            if not bool(getattr(o, "metrics_excluded", False))
            and float(getattr(o, "OPTj", 0.0) or 0.0) >= start
        ),
        key=lambda o: (
            int(getattr(o, "j", 0) or 0),
            float(getattr(o, "OPTj", 0.0) or 0.0),
        ),
    )
    cum_bt = 0
    cum_ut = 0
    horizon = float(sim.env.now)
    for idx, order in enumerate(orders, start=1):
        if bool(getattr(order, "lost", False)):
            cum_ut += 1
        elif order_counts_as_backorder_for_fill_rate(order, current_time=horizon):
            cum_bt += 1
        ret_j, case = compute_ret_per_order_excel_formula(
            order,
            j=idx,
            cumulative_backorders=cum_bt,
            cumulative_unattended=cum_ut,
        )
        opt = getattr(order, "OPTj", None)
        oat = getattr(order, "OATj", None)
        row = {
            "policy": policy,
            "seed": int(seed),
            "episode": int(episode),
            "eval_seed": int(eval_seed),
            "j": idx,
            "Q": float(getattr(order, "quantity", getattr(order, "Q", 0.0)) or 0.0),
            "OPTj": opt,
            "OATj": oat,
            "LT": getattr(order, "LTj", None),
            "CTj": getattr(order, "CTj", None),
            "APj": getattr(order, "APj", None),
            "RPj": getattr(order, "RPj", None),
            "DPj": getattr(order, "DPj", None),
            "sumBt": cum_bt,
            "sumUt": cum_ut,
            "lost": int(bool(getattr(order, "lost", False))),
            "backorder": int(bool(getattr(order, "backorder", False))),
            "ReTj": ret_j,
            "case": case,
        }
        risks = dict(getattr(order, "ret_risk_indicators", {}) or {})
        for risk_col in RISK_COLS:
            row[risk_col] = float(risks.get(risk_col, 0.0))
        ledger_rows.append(row)


def make_monitored_training_env(
    args: argparse.Namespace, seed: int
) -> callable[[], Monitor]:
    env_kwargs = build_env_kwargs(args)
    wrapper_cls = getattr(args, "_ablation_wrapper", None)
    observation_wrapper_cls = getattr(args, "_observation_wrapper", None)

    def _init() -> Monitor:
        env = make_track_b_env(**env_kwargs)
        if wrapper_cls is not None:
            env = wrapper_cls(env)
        if observation_wrapper_cls is not None:
            env = observation_wrapper_cls(env)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def apply_eval_wrappers(env: Any, args: argparse.Namespace) -> Any:
    wrapper_cls = getattr(args, "_ablation_wrapper", None)
    observation_wrapper_cls = getattr(args, "_observation_wrapper", None)
    if wrapper_cls is not None:
        env = wrapper_cls(env)
    if observation_wrapper_cls is not None:
        env = observation_wrapper_cls(env)
    return env


def train_ppo(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    ensure_algo_dependencies(args)
    n_envs = max(1, int(getattr(args, "n_envs", 1)))
    vec_env = DummyVecEnv(
        [make_monitored_training_env(args, seed + i) for i in range(n_envs)]
    )
    vec_norm = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=bool(getattr(args, "norm_reward", False)),
        clip_obs=10.0,
        clip_reward=float(getattr(args, "clip_reward", 10.0)),
        gamma=float(args.gamma),
    )
    algo = learned_policy_name(args)
    if algo == "ppo":
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
            ent_coef=args.ent_coef,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            seed=seed,
            verbose=0,
            device="cpu",
        )
    elif algo in ("sac", "td3"):
        # Off-policy robustness screen (reviewer objection closure): same env,
        # obs, eval protocol, and net width as the canonical PPO cell; SB3
        # defaults for the off-policy-specific hyperparameters.
        from stable_baselines3 import SAC, TD3

        off_policy_cls = SAC if algo == "sac" else TD3
        off_policy_kwargs: dict[str, Any] = {
            "policy": "MlpPolicy",
            "env": vec_norm,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "policy_kwargs": {"net_arch": [64, 64]},
            "seed": seed,
            "verbose": 0,
            "device": "cpu",
        }
        model = off_policy_cls(**off_policy_kwargs)
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_norm,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs={
                "net_arch": {"pi": [64], "vf": [64]},
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
            },
            seed=seed,
            verbose=0,
            device="cpu",
        )
    model.learn(total_timesteps=args.train_timesteps)
    model.save(run_dir / model_filename(args))
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def _finalize_episode_row(
    *,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    steps: int,
    reward_total: float,
    demanded_total: float,
    backorder_qty_total: float,
    shift_counts: dict[int, int],
    op10_multipliers: list[float],
    op12_multipliers: list[float],
    track_b_context: dict[str, Any],
    terminal_metrics: dict[str, float],
    final_info: dict[str, Any] | None = None,
    cd_totals: dict[str, float] | None = None,
    full_episode_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    total_steps = max(1, steps)
    if demanded_total > 0.0:
        flow_backorder_rate = backorder_qty_total / demanded_total
        flow_fill_rate = max(0.0, min(1.0, 1.0 - flow_backorder_rate))
    else:
        flow_backorder_rate = 0.0
        flow_fill_rate = 1.0
    op10_arr = np.asarray(op10_multipliers or [1.0], dtype=np.float64)
    op12_arr = np.asarray(op12_multipliers or [1.0], dtype=np.float64)
    row = {
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "steps": steps,
        "reward_total": reward_total,
        "fill_rate": float(terminal_metrics["fill_rate_order_level"]),
        "backorder_rate": float(terminal_metrics["backorder_rate_order_level"]),
        "order_level_ret_mean": float(terminal_metrics["order_level_ret_mean"]),
        "flow_fill_rate": flow_fill_rate,
        "flow_backorder_rate": flow_backorder_rate,
        "terminal_rolling_fill_rate_4w": float(track_b_context["rolling_fill_rate_4w"]),
        "terminal_rolling_backorder_rate_4w": float(
            track_b_context["rolling_backorder_rate_4w"]
        ),
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
        "op10_multiplier_step_mean": float(np.mean(op10_arr)),
        "op12_multiplier_step_mean": float(np.mean(op12_arr)),
        "op10_multiplier_step_p95": float(np.percentile(op10_arr, 95)),
        "op12_multiplier_step_p95": float(np.percentile(op12_arr, 95)),
        "pct_steps_op10_multiplier_ge_190": 100.0
        * float(np.mean(op10_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)),
        "pct_steps_op12_multiplier_ge_190": 100.0
        * float(np.mean(op12_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)),
        "pct_steps_both_downstream_ge_190": 100.0
        * float(
            np.mean(
                (op10_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)
                & (op12_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)
            )
        ),
        "assembly_hours_total": sum(
            shift_counts.get(s, 0) * s * HOURS_PER_SHIFT * 7.0 for s in (1, 2, 3)
        ),
        "assembly_cost_index": sum(shift_counts.get(s, 0) * s for s in (1, 2, 3))
        / (3.0 * total_steps),
    }
    cd_totals = cd_totals or init_cd_totals()
    final_info = final_info or {}
    row.update(
        {
            "ret_garrido2024_raw_total": float(cd_totals["ret_garrido2024_raw_total"]),
            "ret_garrido2024_train_total": float(cd_totals["ret_garrido2024_train_total"]),
            "ret_garrido2024_sigmoid_total": float(
                cd_totals["ret_garrido2024_sigmoid_total"]
            ),
            "ret_garrido2024_sigmoid_mean": float(
                cd_totals["ret_garrido2024_sigmoid_total"] / total_steps
            ),
            "terminal_zeta_avg": float(final_info.get("zeta_avg", 0.0)),
            "terminal_epsilon_avg": float(final_info.get("epsilon_avg", 0.0)),
            "terminal_phi_avg": float(final_info.get("phi_avg", 0.0)),
            "terminal_tau_avg": float(final_info.get("tau_avg", 0.0)),
            "terminal_kappa_dot": float(final_info.get("kappa_dot", 0.0)),
        }
    )
    for key in METRIC_KEYS:
        row[f"order_{key}"] = float((full_episode_metrics or {}).get(key, 0.0))
    return row


def evaluate_static_policy(
    policy: StaticPolicySpec,
    *,
    args: argparse.Namespace,
    seed: int,
    order_ledger_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)
    action_payload = build_static_policy_action(policy)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = apply_eval_wrappers(make_track_b_env(**env_kwargs), args)
        obs, info = env.reset(seed=eval_seed)
        del obs
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        cd_totals = init_cd_totals()
        final_info = info

        while not (terminated or truncated):
            _, reward, terminated, truncated, final_info = env.step(action_payload)
            reward_total += float(reward)
            update_cd_totals(cd_totals, final_info)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1

        rows.append(
            _finalize_episode_row(
                policy=policy.label,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
                final_info=final_info,
                cd_totals=cd_totals,
                full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
            )
        )
        append_order_ledger_rows(
            order_ledger_rows,
            env,
            policy=policy.label,
            seed=seed,
            episode=episode_idx + 1,
            eval_seed=eval_seed,
        )
        env.close()
    return rows


def evaluate_trained_policy(
    *,
    args: argparse.Namespace,
    seed: int,
    model: Any,
    vec_norm: VecNormalize,
    order_ledger_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)
    vec_norm.training = False
    algo = learned_policy_name(args)
    is_recurrent = algo == "recurrent_ppo"

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = apply_eval_wrappers(make_track_b_env(**env_kwargs), args)
        obs, info = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        cd_totals = init_cd_totals()
        final_info = info
        lstm_states: Any = None
        episode_start = np.ones((1,), dtype=bool)

        while not (terminated or truncated):
            obs_norm = vec_norm.normalize_obs(
                np.asarray(obs, dtype=np.float32)[None, :]
            )
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs_norm,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
            reward_total += float(reward)
            update_cd_totals(cd_totals, final_info)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1
            episode_start = np.array([terminated or truncated], dtype=bool)

        rows.append(
            _finalize_episode_row(
                policy=algo,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
                final_info=final_info,
                cd_totals=cd_totals,
                full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
            )
        )
        append_order_ledger_rows(
            order_ledger_rows,
            env,
            policy=algo,
            seed=seed,
            episode=episode_idx + 1,
            eval_seed=eval_seed,
        )
        env.close()
    return rows


def evaluate_heuristic_policy(
    label: str,
    heuristic: Any,
    *,
    args: argparse.Namespace,
    seed: int,
    order_ledger_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a Track B heuristic that takes (obs, info) -> 8D action array."""
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)
    heuristic.reset()

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = apply_eval_wrappers(make_track_b_env(**env_kwargs), args)
        obs, info = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        cd_totals = init_cd_totals()
        final_info = info
        heuristic.reset()

        while not (terminated or truncated):
            action = heuristic(obs, final_info)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            reward_total += float(reward)
            update_cd_totals(cd_totals, final_info)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1

        rows.append(
            _finalize_episode_row(
                policy=label,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
                final_info=final_info,
                cd_totals=cd_totals,
                full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
            )
        )
        append_order_ledger_rows(
            order_ledger_rows,
            env,
            policy=label,
            seed=seed,
            episode=episode_idx + 1,
            eval_seed=eval_seed,
        )
        env.close()
    return rows


def load_dkana_adapter(
    checkpoint_path: Path,
) -> tuple[DKANAOnlinePolicyAdapter, dict[str, Any]]:
    import torch

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    model_config = dict(checkpoint["model_config"])
    model = DKANAPolicy(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset_metadata = checkpoint["dataset_metadata"]
    observation_fields = tuple(dataset_metadata["env_spec"]["observation_fields"])
    relation_mode = str(dataset_metadata.get("relation_mode", "equality"))
    include_prev_reward = bool(dataset_metadata.get("include_prev_reward", False))
    window_size = int(dataset_metadata["window_size"])
    action_dim = int(model_config["action_dim"])
    adapter = DKANAOnlinePolicyAdapter(
        model,
        window_size=window_size,
        observation_fields=observation_fields,
        state_constraint_fields=STATE_CONSTRAINT_FIELDS,
        action_dim=action_dim,
        relation_mode=relation_mode,
        include_prev_reward=include_prev_reward,
    )
    return adapter, {
        "relation_mode": relation_mode,
        "include_prev_reward": include_prev_reward,
        "window_size": window_size,
        "action_dim": action_dim,
    }


def evaluate_dkana_policy(
    adapter: DKANAOnlinePolicyAdapter,
    *,
    args: argparse.Namespace,
    seed: int,
    label: str = "dkana",
    order_ledger_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = apply_eval_wrappers(make_track_b_env(**env_kwargs), args)
        obs, info = env.reset(seed=eval_seed)
        adapter.reset()
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        cd_totals = init_cd_totals()
        final_info = info

        while not (terminated or truncated):
            action = adapter(np.asarray(obs, dtype=np.float32), final_info)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            final_info["previous_reward"] = float(reward)
            reward_total += float(reward)
            update_cd_totals(cd_totals, final_info)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1

        rows.append(
            _finalize_episode_row(
                policy=label,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
                final_info=final_info,
                cd_totals=cd_totals,
                full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
            )
        )
        append_order_ledger_rows(
            order_ledger_rows,
            env,
            policy=label,
            seed=seed,
            episode=episode_idx + 1,
            eval_seed=eval_seed,
        )
        env.close()
    return rows


def aggregate_seed_metrics(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault((str(row["policy"]), int(row["seed"])), []).append(row)

    seed_rows: list[dict[str, Any]] = []
    for (policy, seed), rows in sorted(grouped.items()):
        out_row: dict[str, Any] = {
            "policy": policy,
            "seed": seed,
            "episodes": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            out_row[f"{metric}_mean"] = float(statistics.fmean(values))
            out_row[f"{metric}_std"] = (
                float(statistics.stdev(values)) if len(values) > 1 else 0.0
            )
        seed_rows.append(out_row)
    return seed_rows


def aggregate_policy_metrics(
    seed_rows: list[dict[str, Any]], *, learned_policy: str = "ppo"
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    policy_rows: list[dict[str, Any]] = []
    extra_policies = tuple(
        name
        for name in sorted(grouped.keys())
        if name not in STATIC_POLICY_ORDER
        and name not in HEURISTIC_POLICY_NAMES
        and name != learned_policy
    )
    for policy in (
        *STATIC_POLICY_ORDER,
        *HEURISTIC_POLICY_NAMES,
        learned_policy,
        *extra_policies,
    ):
        rows = grouped.get(policy, [])
        if not rows:
            continue
        out_row: dict[str, Any] = {
            "policy": policy,
            "seed_count": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[f"{metric}_mean"]) for row in rows]
            ci_low, ci_high = ci95(values)
            out_row[f"{metric}_mean"] = float(statistics.fmean(values))
            out_row[f"{metric}_std"] = (
                float(statistics.stdev(values)) if len(values) > 1 else 0.0
            )
            out_row[f"{metric}_ci95_low"] = ci_low
            out_row[f"{metric}_ci95_high"] = ci_high
        policy_rows.append(out_row)
    return policy_rows


def build_decision_summary(
    policy_rows: list[dict[str, Any]], *, learned_policy: str = "ppo"
) -> dict[str, Any]:
    def ret_metric(row: dict[str, Any]) -> float:
        return float(
            row.get("order_level_ret_mean_mean", row.get("order_level_ret_mean", 0.0))
        )

    by_policy = {str(row["policy"]): row for row in policy_rows}
    baseline = by_policy["s2_d1.00"]
    best_static = max(
        (by_policy[policy_name] for policy_name in STATIC_POLICY_ORDER),
        key=lambda row: (
            ret_metric(row),
            float(row["fill_rate_mean"]),
            -float(row["backorder_rate_mean"]),
        ),
    )
    learned_row = by_policy[learned_policy]
    fill_gap_vs_baseline_pp = 100.0 * (
        float(learned_row["fill_rate_mean"]) - float(baseline["fill_rate_mean"])
    )
    fill_gap_vs_best_static_pp = 100.0 * (
        float(learned_row["fill_rate_mean"]) - float(best_static["fill_rate_mean"])
    )
    reward_gap_vs_best_static = float(learned_row["reward_total_mean"]) - float(
        best_static["reward_total_mean"]
    )
    ret_gap_vs_best_static = ret_metric(learned_row) - ret_metric(best_static)
    raw_ret_win = ret_gap_vs_best_static > 0.0
    decision = {
        "learned_policy": learned_policy,
        "baseline_policy": "s2_d1.00",
        "best_static_policy": str(best_static["policy"]),
        "primary_metric": "order_level_ret_mean",
        "learned_fill_gap_vs_s2_neutral_pp": fill_gap_vs_baseline_pp,
        "learned_fill_gap_vs_best_static_pp": fill_gap_vs_best_static_pp,
        "learned_reward_gap_vs_best_static": reward_gap_vs_best_static,
        "learned_order_level_ret_gap_vs_best_static": ret_gap_vs_best_static,
        "learned_raw_ret_win_vs_best_static": raw_ret_win,
        "learned_beats_s2_neutral_by_fill": fill_gap_vs_baseline_pp > 0.0,
        "learned_matches_best_static_by_fill": fill_gap_vs_best_static_pp >= -0.5,
        "promote_to_long_run": raw_ret_win,
    }
    if learned_policy == "ppo":
        decision.update(
            {
                "ppo_fill_gap_vs_s2_neutral_pp": fill_gap_vs_baseline_pp,
                "ppo_fill_gap_vs_best_static_pp": fill_gap_vs_best_static_pp,
                "ppo_reward_gap_vs_best_static": reward_gap_vs_best_static,
                "ppo_order_level_ret_gap_vs_best_static": ret_gap_vs_best_static,
                "ppo_raw_ret_win_vs_best_static": raw_ret_win,
                "ppo_beats_s2_neutral_by_fill": fill_gap_vs_baseline_pp > 0.0,
                "ppo_matches_best_static_by_fill": fill_gap_vs_best_static_pp >= -0.5,
            }
        )
    return decision


def build_reward_contract(reward_mode: str) -> dict[str, Any]:
    reward_family = (
        "operational_penalty"
        if reward_mode in ("control_v1", "control_v1_pbrs")
        else "resilience_index"
    )
    return {
        "reward_mode": reward_mode,
        "reward_family": reward_family,
        "cross_mode_reward_comparison_allowed": False,
        "within_run_reward_comparison_allowed": True,
        "selection_metrics": [
            "fill_rate",
            "backorder_rate",
            "order_level_ret_mean",
            "reward_total_within_same_reward_mode_only",
        ],
    }


def build_comparison_rows(
    policy_rows: list[dict[str, Any]], *, args: argparse.Namespace
) -> list[dict[str, Any]]:
    learned_policy = learned_policy_name(args)
    by_policy = {str(row["policy"]): row for row in policy_rows}
    baseline = by_policy["s2_d1.00"]
    best_static_name = max(
        STATIC_POLICY_ORDER,
        key=lambda policy: (
            float(by_policy[policy]["fill_rate_mean"]),
            float(by_policy[policy]["order_level_ret_mean_mean"]),
            -float(by_policy[policy]["backorder_rate_mean"]),
        ),
    )
    best_static = by_policy[best_static_name]
    learned_row = by_policy[learned_policy]
    reward_contract = build_reward_contract(str(args.reward_mode))
    row = {
        "reward_mode": str(args.reward_mode),
        "reward_family": reward_contract["reward_family"],
        "action_contract": "track_b_v1",
        "observation_version": str(args.observation_version),
        "risk_level": str(args.risk_level),
        "learned_policy": learned_policy,
        "baseline_policy": "s2_d1.00",
        "best_static_policy": best_static_name,
        "learned_reward_mean": float(learned_row["reward_total_mean"]),
        "learned_fill_rate_mean": float(learned_row["fill_rate_mean"]),
        "learned_backorder_rate_mean": float(learned_row["backorder_rate_mean"]),
        "learned_order_level_ret_mean": float(learned_row["order_level_ret_mean_mean"]),
        "baseline_reward_mean": float(baseline["reward_total_mean"]),
        "baseline_fill_rate_mean": float(baseline["fill_rate_mean"]),
        "baseline_backorder_rate_mean": float(baseline["backorder_rate_mean"]),
        "baseline_order_level_ret_mean": float(baseline["order_level_ret_mean_mean"]),
        "best_static_reward_mean": float(best_static["reward_total_mean"]),
        "best_static_fill_rate_mean": float(best_static["fill_rate_mean"]),
        "best_static_backorder_rate_mean": float(best_static["backorder_rate_mean"]),
        "best_static_order_level_ret_mean": float(
            best_static["order_level_ret_mean_mean"]
        ),
        "learned_fill_gap_vs_baseline_pp": float(
            100.0
            * (float(learned_row["fill_rate_mean"]) - float(baseline["fill_rate_mean"]))
        ),
        "learned_fill_gap_vs_best_static_pp": float(
            100.0
            * (
                float(learned_row["fill_rate_mean"])
                - float(best_static["fill_rate_mean"])
            )
        ),
        "learned_reward_gap_vs_best_static": float(
            float(learned_row["reward_total_mean"])
            - float(best_static["reward_total_mean"])
        ),
        "learned_order_level_ret_gap_vs_best_static": float(
            float(learned_row["order_level_ret_mean_mean"])
            - float(best_static["order_level_ret_mean_mean"])
        ),
        "learned_beats_s2_neutral_by_fill": bool(
            float(learned_row["fill_rate_mean"]) > float(baseline["fill_rate_mean"])
        ),
        "learned_matches_best_static_by_fill": bool(
            (
                100.0
                * (
                    float(learned_row["fill_rate_mean"])
                    - float(best_static["fill_rate_mean"])
                )
            )
            >= -0.5
        ),
        "promote_to_long_run": bool(
            (
                100.0
                * (
                    float(learned_row["fill_rate_mean"])
                    - float(baseline["fill_rate_mean"])
                )
            )
            > 0.0
            and (
                100.0
                * (
                    float(learned_row["fill_rate_mean"])
                    - float(best_static["fill_rate_mean"])
                )
            )
            >= -1.0
        ),
    }
    if learned_policy == "ppo":
        row.update(
            {
                "ppo_reward_mean": row["learned_reward_mean"],
                "ppo_fill_rate_mean": row["learned_fill_rate_mean"],
                "ppo_backorder_rate_mean": row["learned_backorder_rate_mean"],
                "ppo_order_level_ret_mean": row["learned_order_level_ret_mean"],
                "ppo_fill_gap_vs_baseline_pp": row["learned_fill_gap_vs_baseline_pp"],
                "ppo_fill_gap_vs_best_static_pp": row[
                    "learned_fill_gap_vs_best_static_pp"
                ],
                "ppo_reward_gap_vs_best_static": row[
                    "learned_reward_gap_vs_best_static"
                ],
                "ppo_order_level_ret_gap_vs_best_static": row[
                    "learned_order_level_ret_gap_vs_best_static"
                ],
                "ppo_beats_s2_neutral_by_fill": row["learned_beats_s2_neutral_by_fill"],
                "ppo_matches_best_static_by_fill": row[
                    "learned_matches_best_static_by_fill"
                ],
            }
        )
    return [row]


def render_markdown(summary: dict[str, Any]) -> str:
    config = summary["config"]
    decision = summary["decision"]
    lines = [
        "# Track B Smoke Benchmark",
        "",
        "## Config",
        "",
        f"- Train timesteps: {config['train_timesteps']}",
        f"- Seeds: {config['seeds']}",
        f"- Eval episodes: {config['eval_episodes']}",
        f"- Reward mode: {config['reward_mode']}",
        f"- Risk level: {config['risk_level']}",
        "",
        "## Policy Summary",
        "",
        "| Policy | Reward | Fill | Backorder | Order-level ReT | Rolling fill 4w | Shift mix | Asm hrs | Cost idx |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary["policy_summary"]:
        shift_mix = (
            f"{float(row['pct_steps_S1_mean']):.1f}/"
            f"{float(row['pct_steps_S2_mean']):.1f}/"
            f"{float(row['pct_steps_S3_mean']):.1f}"
        )
        lines.append(
            "| {policy} | {reward:.2f} | {fill:.3f} | {backorder:.3f} | "
            "{ret:.3f} | {rolling_fill:.3f} | {shift_mix} | "
            "{asm_hrs:.0f} | {cost_idx:.3f} |".format(
                policy=row["policy"],
                reward=float(row["reward_total_mean"]),
                fill=float(row["fill_rate_mean"]),
                backorder=float(row["backorder_rate_mean"]),
                ret=float(row["order_level_ret_mean_mean"]),
                rolling_fill=float(row["terminal_rolling_fill_rate_4w_mean"]),
                shift_mix=shift_mix,
                asm_hrs=float(row.get("assembly_hours_total_mean", 0.0)),
                cost_idx=float(row.get("assembly_cost_index_mean", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Learned policy: `{decision['learned_policy']}`",
            f"- Primary decision metric: `{decision['primary_metric']}`",
            f"- Best static policy: `{decision['best_static_policy']}`",
            (
                f"- {decision['learned_policy']} fill gap vs `s2_d1.00`: "
                f"{float(decision['learned_fill_gap_vs_s2_neutral_pp']):+.2f} pp"
            ),
            (
                f"- {decision['learned_policy']} fill gap vs best static: "
                f"{float(decision['learned_fill_gap_vs_best_static_pp']):+.2f} pp"
            ),
            (
                f"- {decision['learned_policy']} reward gap vs best static: "
                f"{float(decision['learned_reward_gap_vs_best_static']):+.2f}"
            ),
            (
                f"- {decision['learned_policy']} order-level ReT gap vs best static: "
                f"{float(decision['learned_order_level_ret_gap_vs_best_static']):+.4f}"
            ),
            f"- Raw ReT win vs best static: `{decision['learned_raw_ret_win_vs_best_static']}`",
            f"- Promote to long run: `{decision['promote_to_long_run']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or default_output_dir(args.train_timesteps)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    order_ledger_rows: list[dict[str, Any]] | None = (
        [] if bool(getattr(args, "export_order_ledger", False)) else None
    )
    trained_models: list[dict[str, Any]] = []
    learned_policy = learned_policy_name(args)

    dkana_adapter: DKANAOnlinePolicyAdapter | None = None
    dkana_metadata: dict[str, Any] | None = None
    if args.dkana_checkpoint is not None:
        dkana_adapter, dkana_metadata = load_dkana_adapter(args.dkana_checkpoint)

    for seed in args.seeds:
        run_dir = models_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model, vec_norm = train_ppo(args, int(seed), run_dir)
        trained_models.append(
            {
                "seed": int(seed),
                "algo": learned_policy,
                "train_timesteps": int(args.train_timesteps),
                "model_path": str((run_dir / model_filename(args)).resolve()),
                "vec_normalize_path": str((run_dir / "vec_normalize.pkl").resolve()),
            }
        )

        for policy in STATIC_POLICY_SPECS:
            episode_rows.extend(
                evaluate_static_policy(
                    policy,
                    args=args,
                    seed=int(seed),
                    order_ledger_rows=order_ledger_rows,
                )
            )
        for h_label, h_policy in make_heuristic_defaults().items():
            try:
                episode_rows.extend(
                    evaluate_heuristic_policy(
                        h_label,
                        h_policy,
                        args=args,
                        seed=int(seed),
                        order_ledger_rows=order_ledger_rows,
                    )
                )
            except ValueError as exc:
                # Legacy heuristic defaults emit a 7-dim action; the track_b_v1 contract is 8-dim.
                # Heuristics are a secondary baseline — skip rather than abort the PPO-vs-static run.
                print(f"[warn] skipping heuristic {h_label}: {exc}", flush=True)
        episode_rows.extend(
            evaluate_trained_policy(
                args=args,
                seed=int(seed),
                model=model,
                vec_norm=vec_norm,
                order_ledger_rows=order_ledger_rows,
            )
        )
        if dkana_adapter is not None:
            episode_rows.extend(
                evaluate_dkana_policy(
                    dkana_adapter,
                    args=args,
                    seed=int(seed),
                    order_ledger_rows=order_ledger_rows,
                )
            )
        vec_norm.close()

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows, learned_policy=learned_policy)
    decision = build_decision_summary(policy_rows, learned_policy=learned_policy)
    comparison_rows = build_comparison_rows(policy_rows, args=args)

    episode_csv = output_dir / "episode_metrics.csv"
    seed_csv = output_dir / "seed_metrics.csv"
    policy_csv = output_dir / "policy_summary.csv"
    comparison_csv = output_dir / "comparison_table.csv"
    order_ledger_csv = output_dir / "order_ledger.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"

    save_csv(episode_csv, episode_rows)
    save_csv(seed_csv, seed_rows)
    save_csv(policy_csv, policy_rows)
    save_csv(comparison_csv, comparison_rows)
    if order_ledger_rows is not None:
        save_csv(order_ledger_csv, order_ledger_rows)
    reward_contract = build_reward_contract(str(args.reward_mode))
    env_kwargs_for_summary = build_env_kwargs(args)

    summary = {
        "config": {
            "seeds": [int(seed) for seed in args.seeds],
            "train_timesteps": int(args.train_timesteps),
            "eval_episodes": int(args.eval_episodes),
            "export_order_ledger": bool(getattr(args, "export_order_ledger", False)),
            "algo": learned_policy,
            "reward_mode": args.reward_mode,
            "ret_seq_kappa": float(args.ret_seq_kappa),
            "ret_excel_cvar_alpha": float(args.ret_excel_cvar_alpha),
            "ret_excel_cvar_tail_level": float(args.ret_excel_cvar_tail_level),
            "ret_excel_cvar_window": int(args.ret_excel_cvar_window),
            "risk_level": args.risk_level,
            "risk_frequency_multiplier": float(args.risk_frequency_multiplier),
            "risk_impact_multiplier": float(args.risk_impact_multiplier),
            "risk_frequency_by_id": dict(
                env_kwargs_for_summary.get("risk_frequency_multipliers_by_id", {})
            ),
            "risk_impact_by_id": dict(
                env_kwargs_for_summary.get("risk_impact_multipliers_by_id", {})
            ),
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
            "observation_version": str(args.observation_version),
            "action_contract": "track_b_v1",
            "year_basis": "thesis",
            "stochastic_pt": True,
            "raw_material_flow_mode_requested": str(
                env_kwargs_for_summary.get(
                    "raw_material_flow_mode", TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE
                )
            ),
            "raw_material_flow_mode_canonical": canonical_raw_material_flow_mode(
                str(
                    env_kwargs_for_summary.get(
                        "raw_material_flow_mode", TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE
                    )
                )
            ),
            "raw_material_order_up_to_multiplier": float(
                env_kwargs_for_summary.get(
                    "raw_material_order_up_to_multiplier",
                    TRACK_A_TRAINING_RAW_MATERIAL_ORDER_UP_TO_MULTIPLIER,
                )
            ),
            "learning_rate": float(args.learning_rate),
            "n_envs": int(getattr(args, "n_envs", 1)),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
            "ent_coef": float(args.ent_coef),
            "norm_reward": bool(getattr(args, "norm_reward", False)),
            "clip_reward": float(getattr(args, "clip_reward", 10.0)),
            "surge_inertia": bool(getattr(args, "surge_inertia", False)),
            "surge_ramp_per_step": int(getattr(args, "surge_ramp_per_step", 1)),
            "surge_budget_hours": float(
                getattr(args, "surge_budget_hours", float("inf"))
            ),
            "dkana_checkpoint": (
                str(args.dkana_checkpoint) if args.dkana_checkpoint else None
            ),
            "dkana_metadata": dkana_metadata,
        },
        "backbone": {
            "code_ref": "HEAD",
            "benchmark_protocol": "track_b_minimal_v1",
            "env_variant": "track_b_adaptive_control",
            "algo": learned_policy,
            "reward_mode": args.reward_mode,
            "observation_version": str(args.observation_version),
            "action_contract": "track_b_v1",
            "risk_level": args.risk_level,
            "risk_frequency_multiplier": float(args.risk_frequency_multiplier),
            "risk_impact_multiplier": float(args.risk_impact_multiplier),
            "risk_frequency_by_id": dict(
                env_kwargs_for_summary.get("risk_frequency_multipliers_by_id", {})
            ),
            "risk_impact_by_id": dict(
                env_kwargs_for_summary.get("risk_impact_multipliers_by_id", {})
            ),
            "year_basis": "thesis",
            "stochastic_pt": True,
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
        },
        "env_spec": spec_to_dict(
            get_track_b_env_spec(
                reward_mode=args.reward_mode,
                observation_version=str(args.observation_version),
                step_size_hours=args.step_size_hours,
            )
        ),
        "metric_contract": build_metric_contract_metadata(),
        "reward_contract": reward_contract,
        "benchmark_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": getattr(args, "invocation", None),
        },
        "trained_models": trained_models,
        "policies": [policy.label for policy in STATIC_POLICY_SPECS] + [learned_policy],
        "decision": decision,
        "seed_metrics": seed_rows,
        "policy_summary": policy_rows,
        "comparison_table": comparison_rows,
        "artifacts": {
            "episode_metrics_csv": str(episode_csv.resolve()),
            "seed_metrics_csv": str(seed_csv.resolve()),
            "policy_summary_csv": str(policy_csv.resolve()),
            "comparison_table_csv": str(comparison_csv.resolve()),
            "order_ledger_csv": (
                str(order_ledger_csv.resolve()) if order_ledger_rows is not None else None
            ),
            "summary_json": str(summary_json.resolve()),
            "summary_md": str(summary_md.resolve()),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    # Cross-scenario evaluation (if --eval-risk-levels provided)
    eval_risk_levels = getattr(args, "eval_risk_levels", None) or []
    if eval_risk_levels and trained_models:
        cross_eval_dir = output_dir / "cross_scenario"
        cross_eval_dir.mkdir(parents=True, exist_ok=True)
        for risk_level in eval_risk_levels:
            if risk_level == args.risk_level:
                continue
            cross_rows: list[dict[str, Any]] = []
            cross_args = argparse.Namespace(**vars(args))
            cross_args.risk_level = risk_level
            for model_info in trained_models:
                seed = int(model_info["seed"])
                model_path = Path(model_info["model_path"])
                vec_path = Path(model_info["vec_normalize_path"])
                if not model_path.exists() or not vec_path.exists():
                    continue
                algo_name = model_info["algo"]
                if algo_name == "recurrent_ppo" and RecurrentPPO is not None:
                    loaded_model = RecurrentPPO.load(str(model_path))
                else:
                    loaded_model = PPO.load(str(model_path))
                cross_env = DummyVecEnv([make_monitored_training_env(cross_args, seed)])
                loaded_vec = VecNormalize.load(str(vec_path), cross_env)
                loaded_vec.training = False
                for policy in STATIC_POLICY_SPECS:
                    cross_rows.extend(
                        evaluate_static_policy(policy, args=cross_args, seed=seed)
                    )
                for h_label, h_policy in make_heuristic_defaults().items():
                    cross_rows.extend(
                        evaluate_heuristic_policy(
                            h_label, h_policy, args=cross_args, seed=seed
                        )
                    )
                cross_rows.extend(
                    evaluate_trained_policy(
                        args=cross_args,
                        seed=seed,
                        model=loaded_model,
                        vec_norm=loaded_vec,
                    )
                )
                loaded_vec.close()
            if cross_rows:
                cross_seed = aggregate_seed_metrics(cross_rows)
                cross_policy = aggregate_policy_metrics(
                    cross_seed, learned_policy=learned_policy
                )
                save_csv(
                    cross_eval_dir / f"episode_metrics_{risk_level}.csv",
                    cross_rows,
                )
                save_csv(
                    cross_eval_dir / f"policy_summary_{risk_level}.csv",
                    cross_policy,
                )
                print(f"  Cross-eval ({risk_level}): {len(cross_rows)} episodes")

    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/run_track_b_smoke.py " + " ".join(sys.argv[1:])
    summary = run_smoke(args)
    print(f"Wrote Track B smoke bundle to {args.output_dir or 'auto output dir'}")
    learned_policy = summary["decision"]["learned_policy"]
    for row in summary["policy_summary"]:
        print(
            f"{row['policy']}: reward={float(row['reward_total_mean']):.2f}, "
            f"fill={float(row['fill_rate_mean']):.3f}, "
            f"backorder={float(row['backorder_rate_mean']):.3f}, "
            f"ret={float(row['order_level_ret_mean_mean']):.3f}"
        )
    print(
        "Decision: "
        f"best_static={summary['decision']['best_static_policy']}, "
        f"{learned_policy}_vs_best_ret={float(summary['decision']['learned_order_level_ret_gap_vs_best_static']):+.6f}, "
        f"raw_ret_win={summary['decision']['learned_raw_ret_win_vs_best_static']}, "
        f"promote={summary['decision']['promote_to_long_run']}"
    )


if __name__ == "__main__":
    main()
