#!/usr/bin/env python3
"""Risk-id anchored counterfactual prevention audit for frozen Track B policies.

Important RNG boundary (2026-07-03/04): splice-style counterfactuals were not
causal under the pre-fix Track B simulator because the adaptive benchmark
regime shared RNG state with action-dependent processes. They are only valid for
fresh checkpoints/evaluations produced after the fixed-RNG Track B environment
(`strict_exogenous_crn=True` plus a dedicated `regime_rng`) is active. See:

    docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md

This evaluation-only audit was originally designed to ask a narrower
causal-looking question than the descriptive risk-event ledger:

    Did the learned actions in the weeks before a specific risk type improve the
    final Garrido/Excel ReT of the same episode?

For each policy, seed, and evaluation episode, the script first records the full
rollout and the real DES ``sim.risk_events``. It then builds each policy's own
"calm" action from steps away from target risk events. Finally, it replays the
same eval seed while replacing actions in pre-risk windows for a given risk id
with that policy-specific calm action, and reports:

    delta_ret_excel = R_full - R_reset(pre-risk, risk_id)

Under the pre-fix Track B RNG, positive deltas are NOT valid causal evidence.
Under fixed-RNG Track B, the same statistic is the intended prevention audit.
This is not a training script; it never updates policies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_prevention_mechanism import (  # noqa: E402
    ACTION_DIMS,
    EVAL_EPISODE_SEED_OFFSET,
    PolicyRuntime,
    env_kwargs,
    finalize_episode,
    load_runtime,
    mean,
    predict_action,
    row_from_step,
    save_csv,
    vec_observation_shape,
)
from scripts.run_track_b_risk_belief_sidecar import (  # noqa: E402
    RiskBeliefAppendWrapper,
    _parse_target,
    train_belief_models,
)
from scripts.audit_track_b_risk_event_ledger import (  # noqa: E402
    RISK_CATEGORY,
    EXPECTED_EVENTS_PER_YEAR_CURRENT,
)
from scripts.ruta_b_aux_ppo import RutaBAuxPPO  # noqa: E402
from scripts.ruta_b_risk_label_wrapper import ConstantLabelPadWrapper  # noqa: E402
from scripts.run_track_b_ruta_b_sidecar import (  # noqa: E402
    VecNormalizeKeepLastRaw,  # noqa: F401  (rebinds under __main__ so pickle.load can resolve it)
)
from supply_chain.external_env_interface import get_observation_fields  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "outputs/experiments/track_b_risk_event_counterfactual_2026-07-03"
)
DEFAULT_TARGET_RISKS = ("R11", "R13", "R24", "R14")
CALM_EXCLUSION_WINDOW = (-4, 8)
RESET_WINDOWS = {
    "pre": (-4, -1),
}


@dataclass
class EpisodeTrace:
    policy: str
    seed: int
    episode: int
    eval_seed: int
    ret_excel: float
    fill_rate: float
    service_loss_auc: float
    cost_index: float
    steps: int
    step_rows: list[dict[str, Any]]
    risk_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", nargs="+", default=["ppo_mlp", "real_kan"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--enabled-risks", default=None)
    parser.add_argument("--risk-frequency-by-id", default=None)
    parser.add_argument("--risk-impact-by-id", default=None)
    parser.add_argument("--faithful", action="store_true")
    parser.add_argument("--obs-config", default=None)
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--surge-inertia", action="store_true")
    parser.add_argument("--surge-ramp-per-step", type=int, default=1)
    parser.add_argument("--surge-budget-hours", type=float, default=float("inf"))
    parser.add_argument("--target-risks", nargs="+", default=list(DEFAULT_TARGET_RISKS))
    parser.add_argument(
        "--max-events-per-risk-episode",
        type=int,
        default=8,
        help=(
            "Maximum event anchors sampled per risk/policy/seed/episode. "
            "Frequent risks overlap heavily, so resetting the union of all pre-windows "
            "would replace almost the whole episode."
        ),
    )
    parser.add_argument(
        "--reset-windows",
        nargs="+",
        default=["pre"],
        choices=sorted(RESET_WINDOWS),
        help="Counterfactual windows to reset. Default: pre-risk weeks -4..-1.",
    )
    parser.add_argument(
        "--ppo-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_ablation_8d_final_2026-07-01/joint"),
            Path(
                "outputs/experiments/track_b_gain_2026-06-30/"
                "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
            ),
            Path(
                "outputs/experiments/track_b_seed_expansion_2026-07-02/"
                "track_b_seed_expansion_6_10_claude"
            ),
        ],
    )
    parser.add_argument(
        "--real-kan-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104"),
            Path(
                "outputs/experiments/track_b_real_kan_sidecar_2026-07-03/"
                "confirm_10seed_extension_6_10_60k_h104"
            ),
        ],
    )
    parser.add_argument(
        "--ppo-belief-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_risk_belief_ppo_3seed_30k_2026-07-04"),
        ],
    )
    parser.add_argument(
        "--real-kan-belief-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_risk_belief_real_kan_3seed_30k_2026-07-04"),
        ],
    )
    parser.add_argument(
        "--ppo-belief-encoder-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_belief_encoder_ppo_3seed_30k_2026-07-04"),
        ],
    )
    parser.add_argument(
        "--real-kan-belief-encoder-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_belief_encoder_real_kan_3seed_30k_2026-07-04_v4"),
        ],
    )
    parser.add_argument(
        "--ruta-b-bundles",
        nargs="+",
        type=Path,
        default=[],
        help="Bundle roots for the 'ruta_b' policy (RutaBAuxPPO + label-pad wrapper).",
    )
    parser.add_argument(
        "--belief-dataset",
        type=Path,
        default=Path("outputs/experiments/track_b_risk_belief_predictor_2026-07-04/risk_belief_dataset.csv"),
    )
    parser.add_argument("--belief-targets", nargs="+", default=["R24:1", "R24:2"])
    parser.add_argument(
        "--belief-class-weight",
        choices=("balanced", "none"),
        default="balanced",
    )
    return parser.parse_args()


def _belief_models(args: argparse.Namespace) -> list[Any]:
    models = getattr(args, "_belief_models", None)
    if models is None:
        targets = [_parse_target(str(target)) for target in args.belief_targets]
        models, meta = train_belief_models(
            Path(args.belief_dataset),
            targets,
            class_weight=str(args.belief_class_weight),
        )
        args._belief_models = models
        args._belief_meta = meta
    return list(args._belief_models)


def _wrap_belief_env(env: gym.Env, args: argparse.Namespace) -> gym.Env:
    return RiskBeliefAppendWrapper(env, models=_belief_models(args))


def _model_filename_for_policy(policy: str) -> str:
    if policy == "real_kan":
        return "ppo_real_kan_model.zip"
    if policy == "ppo_mlp_belief":
        return "ppo_mlp_belief_model.zip"
    if policy == "real_kan_belief":
        return "real_kan_belief_model.zip"
    if policy == "ppo_mlp_belief_encoder":
        return "ppo_mlp_belief_encoder_model.zip"
    if policy == "real_kan_belief_encoder":
        return "real_kan_belief_encoder_model.zip"
    if policy == "ruta_b":
        return "ruta_b_model.zip"
    return "ppo_model.zip"


def _bundles_for_policy(policy: str, args: argparse.Namespace) -> list[Path]:
    if policy == "real_kan":
        return list(args.real_kan_bundles)
    if policy == "ppo_mlp_belief":
        return list(args.ppo_belief_bundles)
    if policy == "real_kan_belief":
        return list(args.real_kan_belief_bundles)
    if policy == "ppo_mlp_belief_encoder":
        return list(args.ppo_belief_encoder_bundles)
    if policy == "real_kan_belief_encoder":
        return list(args.real_kan_belief_encoder_bundles)
    if policy == "ruta_b":
        return list(getattr(args, "ruta_b_bundles", []))
    return list(args.ppo_bundles)


def _expected_shape_for_policy(policy: str, args: argparse.Namespace) -> tuple[int, ...]:
    base = len(get_observation_fields(args.observation_version))
    if policy in {"ppo_mlp_belief", "real_kan_belief"}:
        return (base + len(args.belief_targets),)
    if policy == "ruta_b":
        # RutaBRiskLabelWrapper/ConstantLabelPadWrapper append one label column.
        return (base + 1,)
    return (base,)


def _make_env_for_policy(policy: str, args: argparse.Namespace) -> gym.Env:
    env = make_track_b_env(**env_kwargs(args))
    obs_config = getattr(args, "obs_config", None)
    if obs_config:
        from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS

        wrapper_cls = OBS_ABLATION_CONFIGS[str(obs_config)].wrapper
        if wrapper_cls is not None:
            env = wrapper_cls(env)
    if policy in {"ppo_mlp_belief", "real_kan_belief"}:
        env = _wrap_belief_env(env, args)
    if policy == "ruta_b":
        env = ConstantLabelPadWrapper(env)
    return env


def load_runtime_for_counterfactual(policy: str, seed: int, args: argparse.Namespace) -> PolicyRuntime:
    bundles = _bundles_for_policy(policy, args)
    expected_shape = _expected_shape_for_policy(policy, args)
    filename = _model_filename_for_policy(policy)
    wrong_shape: list[str] = []
    for bundle in bundles:
        run_dir = bundle / "models" / f"seed{seed}"
        model_path = run_dir / filename
        vec_path = run_dir / "vec_normalize.pkl"
        if model_path.exists() and vec_path.exists():
            shape = vec_observation_shape(vec_path)
            if shape != expected_shape:
                wrong_shape.append(f"{vec_path} shape={shape}")
                continue
            model = (
                RutaBAuxPPO.load(str(model_path), device="cpu")
                if policy == "ruta_b"
                else PPO.load(str(model_path), device="cpu")
            )

            def _init() -> gym.Env:
                return _make_env_for_policy(policy, args)

            vec_norm = VecNormalize.load(str(vec_path), DummyVecEnv([_init]))
            vec_norm.training = False
            vec_norm.norm_reward = False
            return PolicyRuntime(name=policy, seed=seed, model=model, vec_norm=vec_norm)
    searched = ", ".join(str(p) for p in bundles)
    suffix = f" Shape mismatches skipped: {wrong_shape}" if wrong_shape else ""
    raise FileNotFoundError(f"Missing {policy} seed {seed} artifact in: {searched}.{suffix}")


def risk_event_rows(
    *,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    risk_events: list[Any],
    step_size_hours: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ev in risk_events:
        rows.append(
            {
                "policy": policy,
                "seed": seed,
                "episode": episode,
                "eval_seed": eval_seed,
                "risk_id": ev.risk_id,
                "category": RISK_CATEGORY.get(ev.risk_id, "sin_categoria"),
                "start_time_hours": float(ev.start_time),
                "end_time_hours": float(ev.end_time),
                "duration_hours": float(ev.duration),
                "start_step": int(ev.start_time // step_size_hours),
                "affected_ops": ",".join(str(op) for op in ev.affected_ops),
                "magnitude": float(ev.magnitude),
            }
        )
    return rows


def run_policy_episode(
    *,
    runtime: PolicyRuntime,
    args: argparse.Namespace,
    seed: int,
    episode: int,
    eval_seed: int,
    reset_steps: set[int] | None = None,
    reset_action: np.ndarray | None = None,
    condition: str = "full",
    keep_rows: bool = True,
) -> EpisodeTrace:
    env = _make_env_for_policy(runtime.name, args)
    obs, _info = env.reset(seed=eval_seed)
    terminated = False
    truncated = False
    step = 0
    rows: list[dict[str, Any]] = []
    shifts: list[int] = []

    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action = predict_action(runtime, obs_before)
        if reset_steps is not None and reset_action is not None and step in reset_steps:
            action = reset_action
        obs, reward, terminated, truncated, info = env.step(action)
        shifts.append(int(info.get("shifts_active", 1)))
        if keep_rows:
            rows.append(
                row_from_step(
                    policy=runtime.name,
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
        step += 1

    ret_excel, fill_rate, service_loss_auc, cost_index = finalize_episode(env, shifts)
    risk_rows = risk_event_rows(
        policy=runtime.name,
        seed=seed,
        episode=episode,
        eval_seed=eval_seed,
        risk_events=list(env.unwrapped.sim.risk_events),
        step_size_hours=float(args.step_size_hours),
    )
    env.close()
    return EpisodeTrace(
        policy=runtime.name,
        seed=seed,
        episode=episode,
        eval_seed=eval_seed,
        ret_excel=ret_excel,
        fill_rate=fill_rate,
        service_loss_auc=service_loss_auc,
        cost_index=cost_index,
        steps=step,
        step_rows=rows,
        risk_rows=risk_rows,
    )


def union_steps_for_events(
    risk_rows: list[dict[str, Any]],
    *,
    risk_id: str,
    window: tuple[int, int],
    max_steps: int,
) -> set[int]:
    steps: set[int] = set()
    lo, hi = window
    for ev in risk_rows:
        if str(ev.get("risk_id")) != risk_id:
            continue
        anchor = int(ev["start_step"])
        for step in range(anchor + lo, anchor + hi + 1):
            if 0 <= step < max_steps:
                steps.add(step)
    return steps


def target_event_neighborhood_steps(
    risk_rows: list[dict[str, Any]],
    *,
    target_risks: set[str],
    max_steps: int,
) -> set[int]:
    steps: set[int] = set()
    lo, hi = CALM_EXCLUSION_WINDOW
    for ev in risk_rows:
        if str(ev.get("risk_id")) not in target_risks:
            continue
        anchor = int(ev["start_step"])
        for step in range(anchor + lo, anchor + hi + 1):
            if 0 <= step < max_steps:
                steps.add(step)
    return steps


def numeric_action(row: dict[str, Any], dim: str) -> float | None:
    val = row.get(f"action_{dim}")
    if val in (None, ""):
        return None
    return float(val)


def compute_calm_actions(
    traces: list[EpisodeTrace],
    *,
    target_risks: set[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows_out: list[dict[str, Any]] = []
    calm_by_policy: dict[str, np.ndarray] = {}
    policies = sorted({trace.policy for trace in traces})
    for policy in policies:
        candidate_rows: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        excluded_steps_total = 0
        total_steps = 0
        for trace in traces:
            if trace.policy != policy:
                continue
            excluded = target_event_neighborhood_steps(
                trace.risk_rows,
                target_risks=target_risks,
                max_steps=trace.steps,
            )
            excluded_steps_total += len(excluded)
            total_steps += trace.steps
            for row in trace.step_rows:
                all_rows.append(row)
                if int(row["step"]) not in excluded:
                    candidate_rows.append(row)

        if candidate_rows:
            source_rows = candidate_rows
            source_name = "outside_target_risk_neighborhood"
        else:
            # Very frequent risks (especially R11/R14) can cover the whole
            # episode when using a -4..+8 exclusion halo. In that case, use the
            # policy's own lowest-intensity quartile as its empirical calm
            # posture, instead of falling back to a global static action.
            ranked = sorted(all_rows, key=lambda r: float(r.get("action_intensity", 0.0)))
            n_quartile = max(1, int(np.ceil(0.25 * len(ranked))))
            source_rows = ranked[:n_quartile]
            source_name = "own_lowest_action_intensity_quartile"
        action_values: list[float] = []
        for dim in ACTION_DIMS:
            vals = [numeric_action(row, dim) for row in source_rows]
            clean = [float(v) for v in vals if v is not None]
            action_values.append(mean(clean))
        vec = np.asarray(action_values, dtype=np.float32)
        calm_by_policy[policy] = vec
        rows_out.append(
            {
                "policy": policy,
                "calm_source": source_name,
                "n_calm_rows": len(source_rows),
                "n_total_rows": len(all_rows),
                "excluded_step_instances": excluded_steps_total,
                "excluded_step_fraction": excluded_steps_total / total_steps if total_steps else 0.0,
                **{f"calm_action_{dim}": float(vec[idx]) for idx, dim in enumerate(ACTION_DIMS)},
            }
        )
    return calm_by_policy, rows_out


def select_event_anchors(
    risk_rows: list[dict[str, Any]],
    *,
    risk_id: str,
    max_events: int,
) -> list[dict[str, Any]]:
    """Choose a bounded, approximately spread-out set of real risk events.

    R11 and R14 can occur many times per episode. Evaluating every event would
    be expensive and highly redundant; evaluating the union of their windows
    resets nearly the whole episode. This sampler uses unique start steps and
    spreads anchors across the episode.
    """
    matching = [ev for ev in risk_rows if str(ev.get("risk_id")) == risk_id]
    if not matching:
        return []

    by_step: dict[int, dict[str, Any]] = {}
    for ev in sorted(matching, key=lambda r: (int(r["start_step"]), float(r["start_time_hours"]))):
        by_step.setdefault(int(ev["start_step"]), ev)
    unique = list(by_step.values())
    if max_events <= 0 or len(unique) <= max_events:
        return unique

    indices = np.linspace(0, len(unique) - 1, num=max_events)
    selected_indices = sorted({int(round(i)) for i in indices})
    selected = [unique[i] for i in selected_indices]
    # Rounding can occasionally collapse indices; top up deterministically.
    if len(selected) < max_events:
        used = {int(ev["start_step"]) for ev in selected}
        for ev in unique:
            if int(ev["start_step"]) not in used:
                selected.append(ev)
                used.add(int(ev["start_step"]))
            if len(selected) >= max_events:
                break
    return sorted(selected, key=lambda r: int(r["start_step"]))


def steps_for_single_anchor(
    *,
    anchor_step: int,
    window: tuple[int, int],
    max_steps: int,
) -> set[int]:
    lo, hi = window
    return {step for step in range(anchor_step + lo, anchor_step + hi + 1) if 0 <= step < max_steps}


def aggregate_counterfactuals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["policy"]), str(row["risk_id"]), str(row["reset_window"])),
            [],
        ).append(row)

    out: list[dict[str, Any]] = []
    for (policy, risk_id, window), group in sorted(groups.items()):
        deltas = [float(r["delta_ret_excel"]) for r in group]
        covers = [float(r["reset_step_fraction"]) for r in group]
        event_counts = [float(r["n_events"]) for r in group]
        positive = sum(1 for d in deltas if d > 0)
        out.append(
            {
                "policy": policy,
                "risk_id": risk_id,
                "category": RISK_CATEGORY.get(risk_id, "sin_categoria"),
                "reset_window": window,
                "n_episode_pairs": len(group),
                "n_positive_pairs": positive,
                "positive_pair_rate": positive / len(group) if group else 0.0,
                "mean_R_full": mean([float(r["R_full"]) for r in group]),
                "mean_R_reset": mean([float(r["R_reset"]) for r in group]),
                "mean_delta_ret_excel": mean(deltas),
                "median_delta_ret_excel": float(np.median(deltas)) if deltas else 0.0,
                "mean_reset_step_fraction": mean(covers),
                "max_reset_step_fraction": max(covers) if covers else 0.0,
                "mean_events_per_episode": mean(event_counts),
                "interpretation": interpret_delta(deltas, covers),
            }
        )
    return out


def interpret_delta(deltas: list[float], covers: list[float]) -> str:
    if not deltas:
        return "sin_datos"
    mean_delta = mean(deltas)
    positive_rate = sum(1 for d in deltas if d > 0) / len(deltas)
    mean_cover = mean(covers)
    if mean_cover > 0.75:
        return "cobertura_muy_alta_interpretar_con_cautela"
    if mean_delta > 0.0 and positive_rate >= 0.67:
        return "pre_acciones_aportan_ReT"
    if mean_delta < 0.0 and positive_rate <= 0.33:
        return "pre_acciones_reducen_ReT_o_costo_de_oportunidad"
    return "sin_senal_causal_clara"


def write_verdict(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    calm_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    def label(policy: str) -> str:
        return {"ppo_mlp": "PPO+MLP", "real_kan": "Real-KAN"}.get(policy, policy)

    lines = [
        "# Track B risk-event counterfactual audit — pre-risk windows",
        "",
        "> **Estado:** auditor causal valido solo para checkpoints/evaluaciones fixed-RNG. Bajo el RNG pre-fix,",
        "> este mismo empalme no era causal porque sustituir acciones podia cambiar la trayectoria futura de riesgos.",
        "> Verificar siempre que el entorno use `strict_exogenous_crn=True` y `regime_rng` dedicado antes de citarlo.",
        "",
        f"Fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Pregunta",
        "",
        "Este auditor intentaba medir si las acciones aprendidas **antes** de riesgos reales de Garrido aportan valor en la metrica principal: ReT Excel.",
        "",
        "La cantidad reportada es:",
        "",
        "```text",
        "R_full - R_reset(pre-risk, risk_id)",
        "```",
        "",
        "donde `R_full` es el ReT Excel final del episodio con la politica congelada, y `R_reset` es una rama empalmada con la misma seed de evaluacion y las acciones de las semanas pre-riesgo reemplazadas por la calma empirica de esa misma politica. No se reentrena nada. Bajo el entorno fixed-RNG, esta comparacion preserva el calendario exogeno de riesgos discretos y demanda; bajo artefactos pre-fix no debe citarse como causal.",
        "",
        "## Configuracion",
        "",
        f"- Politicas: {', '.join(args.policies)}",
        f"- Seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- Episodios de evaluacion por seed: {args.eval_episodes}",
        f"- Riesgos auditados: {', '.join(args.target_risks)}",
        f"- Ventana pre-riesgo: semanas {RESET_WINDOWS['pre'][0]} a {RESET_WINDOWS['pre'][1]} relativas al inicio real del riesgo.",
        f"- Maximo de anclas por riesgo/episodio: {args.max_events_per_risk_episode}.",
        "- Resultado: `ret_excel` de `compute_episode_metrics`, es decir, la formula Excel/Garrido agregada al episodio.",
        "",
        "## Calma propia de cada politica",
        "",
        "| Politica | Fuente | Filas calma | Fraccion excluida por cercania a riesgos | shift | op10 | op12 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in calm_rows:
        lines.append(
            "| {policy} | {source} | {n} | {frac:.3f} | {shift:.3f} | {op10:.3f} | {op12:.3f} |".format(
                policy=label(str(row["policy"])),
                source=row["calm_source"],
                n=int(row["n_calm_rows"]),
                frac=float(row["excluded_step_fraction"]),
                shift=float(row["calm_action_shift"]),
                op10=float(row["calm_action_op10_q"]),
                op12=float(row["calm_action_op12_q"]),
            )
        )

    lines.extend(
        [
            "",
            "## Resultado por riesgo",
            "",
            "| Politica | Riesgo | Categoria | Pares | Positivos | Delta ReT Excel medio | Cobertura reset media | Lectura |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {policy} | {risk} | {cat} | {n} | {pos}/{n} | {delta:+.8f} | {cover:.3f} | {interp} |".format(
                policy=label(str(row["policy"])),
                risk=str(row["risk_id"]),
                cat=str(row["category"]),
                n=int(row["n_episode_pairs"]),
                pos=int(row["n_positive_pairs"]),
                delta=float(row["mean_delta_ret_excel"]),
                cover=float(row["mean_reset_step_fraction"]),
                interp=str(row["interpretation"]),
            )
        )

    lines.extend(
        [
            "",
            "## Lectura cautelosa",
            "",
            "Este intento usa eventos reales por `risk_id` y debe interpretarse solo dentro del protocolo fixed-RNG. Una senal positiva aun requiere estabilidad poblacional: delta ReT Excel positivo y una tasa de pares positivos alta, no solo una media pequena arrastrada por pocos episodios.",
            "",
            "No se debe llamar a una politica `preventiva` solo por un patron visual de acciones. Para usar esa palabra, necesitamos delta ReT Excel positivo y estable en ventanas pre-riesgo, idealmente separado por riesgo frecuente.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    target_risks = {str(r) for r in args.target_risks}

    traces: list[EpisodeTrace] = []
    full_metrics_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for policy in args.policies:
        for seed in args.seeds:
            runtime = load_runtime_for_counterfactual(policy, seed, args)
            for episode in range(1, int(args.eval_episodes) + 1):
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                trace = run_policy_episode(
                    runtime=runtime,
                    args=args,
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                    condition="full",
                    keep_rows=True,
                )
                traces.append(trace)
                step_rows.extend(trace.step_rows)
                risk_rows.extend(trace.risk_rows)
                full_metrics_rows.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "episode": episode,
                        "eval_seed": eval_seed,
                        "R_full": trace.ret_excel,
                        "fill_rate": trace.fill_rate,
                        "service_loss_auc": trace.service_loss_auc,
                        "cost_index": trace.cost_index,
                        "n_risk_events": len(trace.risk_rows),
                    }
                )

    calm_actions, calm_rows = compute_calm_actions(traces, target_risks=target_risks)

    counterfactual_rows: list[dict[str, Any]] = []
    for trace in traces:
        runtime = load_runtime_for_counterfactual(trace.policy, trace.seed, args)
        calm_action = calm_actions[trace.policy]
        for risk_id in args.target_risks:
            anchors = select_event_anchors(
                trace.risk_rows,
                risk_id=risk_id,
                max_events=int(args.max_events_per_risk_episode),
            )
            n_events = sum(1 for ev in trace.risk_rows if str(ev["risk_id"]) == risk_id)
            if not anchors:
                continue
            for anchor_idx, anchor in enumerate(anchors, start=1):
                anchor_step = int(anchor["start_step"])
                for window_name in args.reset_windows:
                    window = RESET_WINDOWS[window_name]
                    reset_steps = steps_for_single_anchor(
                        anchor_step=anchor_step,
                        window=window,
                        max_steps=trace.steps,
                    )
                    if not reset_steps:
                        continue
                    reset = run_policy_episode(
                        runtime=runtime,
                        args=args,
                        seed=trace.seed,
                        episode=trace.episode,
                        eval_seed=trace.eval_seed,
                        reset_steps=reset_steps,
                        reset_action=calm_action,
                        condition=f"reset_{risk_id}_{window_name}",
                        keep_rows=False,
                    )
                    counterfactual_rows.append(
                        {
                            "policy": trace.policy,
                            "seed": trace.seed,
                            "episode": trace.episode,
                            "eval_seed": trace.eval_seed,
                            "risk_id": risk_id,
                            "category": RISK_CATEGORY.get(risk_id, "sin_categoria"),
                            "reset_window": window_name,
                            "window_lo": window[0],
                            "window_hi": window[1],
                            "anchor_index": anchor_idx,
                            "anchor_step": anchor_step,
                            "anchor_start_time_hours": anchor["start_time_hours"],
                            "n_events": n_events,
                            "n_sampled_events": len(anchors),
                            "n_reset_steps": len(reset_steps),
                            "episode_steps": trace.steps,
                            "reset_step_fraction": len(reset_steps) / trace.steps if trace.steps else 0.0,
                            "R_full": trace.ret_excel,
                            "R_reset": reset.ret_excel,
                            "delta_ret_excel": trace.ret_excel - reset.ret_excel,
                            "full_fill_rate": trace.fill_rate,
                            "reset_fill_rate": reset.fill_rate,
                            "full_cost_index": trace.cost_index,
                            "reset_cost_index": reset.cost_index,
                        }
                    )

    summary_rows = aggregate_counterfactuals(counterfactual_rows)

    save_csv(out / "step_ledger_full.csv", step_rows)
    save_csv(out / "risk_event_ledger.csv", risk_rows)
    save_csv(out / "full_episode_metrics.csv", full_metrics_rows)
    save_csv(out / "calm_action_by_policy.csv", calm_rows)
    save_csv(out / "risk_event_counterfactual_pre.csv", counterfactual_rows)
    save_csv(out / "summary_by_policy_risk.csv", summary_rows)

    # Echo the RESOLVED environment kwargs, not just the raw args: on 2026-07-07
    # a stale copy of audit_track_b_prevention_mechanism.py on a remote host
    # parsed --enabled-risks/--faithful but silently ignored them in
    # env_kwargs(), producing a wrong-environment run that was only caught by
    # noticing mean_R_full was at the all-risks scale (~0.0056) instead of the
    # Case C scale (~0.48).
    resolved_env_kwargs = env_kwargs(args)
    mean_r_full_overall = (
        mean([float(r["R_full"]) for r in full_metrics_rows]) if full_metrics_rows else 0.0
    )
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policies": args.policies,
        "seeds": args.seeds,
        "eval_episodes": args.eval_episodes,
        "target_risks": args.target_risks,
        "reset_windows": args.reset_windows,
        "metric": "ret_excel from compute_episode_metrics (Garrido/Excel ReT)",
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "resolved_env_kwargs": {k: repr(v) for k, v in sorted(resolved_env_kwargs.items())},
        "obs_config": getattr(args, "obs_config", None),
        "mean_R_full_overall": mean_r_full_overall,
        "action_contract": "track_b_v1",
        "belief_sidecar": {
            "dataset": str(args.belief_dataset),
            "targets": args.belief_targets,
            "class_weight": args.belief_class_weight,
        },
        "expected_events_per_year_current_R1": {
            key: EXPECTED_EVENTS_PER_YEAR_CURRENT.get(key) for key in args.target_risks
        },
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_verdict(
        out / "verdict.md",
        summary_rows=summary_rows,
        calm_rows=calm_rows,
        args=args,
    )
    print(f"Wrote risk-event counterfactual bundle to {out}")
    for row in summary_rows:
        print(
            f"{row['policy']} {row['risk_id']} {row['reset_window']}: "
            f"delta={float(row['mean_delta_ret_excel']):+.8f}, "
            f"positive={row['n_positive_pairs']}/{row['n_episode_pairs']}, "
            f"coverage={float(row['mean_reset_step_fraction']):.3f}"
        )


if __name__ == "__main__":
    main()
