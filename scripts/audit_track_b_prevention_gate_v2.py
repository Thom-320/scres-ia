#!/usr/bin/env python3
"""Causal Prevention Gate v2 for Track B.

This auditor replaces the fragile post-hoc splice gate with complete reruns
under replayed/edited risk-event tapes.  It is an audit harness, not a training
script.  Defaults target the selected Case C stress assay:

  enabled R22/R23/R24, R24 frequency x3, R22/R23 impact x1.5, h104.

Modes
-----
forced_prep_sweep
    Discover a realized event tape, then rerun the whole episode with calm vs
    forced-prep actions during each anchor's pre-window.
event_on_off
    Rerun the whole episode with the anchor present vs removed.
oracle_warning
    Same as forced_prep_sweep, but reported as a perfect-warning/preaction
    assay: warning-prep allowed vs preaction blocked.

The primary outcomes are local to orders plausibly exposed to the anchor.  This
avoids using episode-level ReT as prevention evidence when frequent risks affect
only a slice of the order ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS  # noqa: E402
from scripts.run_track_b_ruta_b_sidecar import VecNormalizeKeepLastRaw  # noqa: F401,E402
from supply_chain.config import OPERATIONS  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_ret_per_order_excel_formula,
    order_counts_as_backorder_for_fill_rate,
)
from supply_chain.risk_event_tape import (  # noqa: E402
    event_to_record,
    load_risk_event_tape,
    remove_event_at_index,
    save_risk_event_tape,
)

HOURS_PER_WEEK = 168.0
EVAL_EPISODE_SEED_OFFSET = 50_000


@dataclass(frozen=True)
class ActionVariant:
    label: str
    action: np.ndarray


STATIC_VARIANTS: tuple[ActionVariant, ...] = (
    ActionVariant("calm", np.array([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)),
    ActionVariant("medium", np.zeros(8, dtype=np.float32)),
    ActionVariant("high_dispatch", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)),
    ActionVariant("combined_prep", np.ones(8, dtype=np.float32)),
)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_id_float_map(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    if raw.strip().startswith("{"):
        return {str(k): float(v) for k, v in json.loads(raw).items()}
    out: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        out[str(key).strip()] = float(value)
    return out


def load_case_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    out = dict(payload)
    if isinstance(out.get("enabled_risks"), str):
        out["enabled_risks"] = [
            item.strip() for item in str(out["enabled_risks"]).split(",") if item.strip()
        ]
    return out


def build_env_kwargs(args: argparse.Namespace, *, risk_event_tape: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cfg = load_case_config(args.case_config)
    obs_cfg = OBS_ABLATION_CONFIGS[str(args.obs_config)]
    enabled = cfg.get("enabled_risks", args.enabled_risks)
    if isinstance(enabled, str):
        enabled = [x.strip() for x in enabled.split(",") if x.strip()]
    return {
        "reward_mode": args.reward_mode,
        "observation_version": obs_cfg.observation_version,
        "action_contract": "track_b_v1",
        "risk_level": cfg.get("risk_level", args.risk_level),
        "step_size_hours": HOURS_PER_WEEK,
        "year_basis": "thesis",
        "stochastic_pt": True,
        "max_steps": args.max_steps,
        "enabled_risks": set(enabled) if enabled else None,
        "risk_frequency_multipliers_by_id": cfg.get(
            "risk_frequency_by_id", parse_id_float_map(args.risk_frequency_by_id)
        ),
        "risk_impact_multipliers_by_id": cfg.get(
            "risk_impact_by_id", parse_id_float_map(args.risk_impact_by_id)
        ),
        "risk_event_tape": risk_event_tape,
        "clear_backlog_after_priming": True,
    }


def make_env(args: argparse.Namespace, *, risk_event_tape: list[dict[str, Any]] | None = None):
    env = make_track_b_env(**build_env_kwargs(args, risk_event_tape=risk_event_tape))
    wrapper = OBS_ABLATION_CONFIGS[str(args.obs_config)].wrapper
    if wrapper is not None:
        env = wrapper(env)
    return env


def load_policy(args: argparse.Namespace, seed: int):
    if args.policy_bundle is None:
        return None, None
    seed_dir = Path(args.policy_bundle) / "models" / f"seed{seed}"
    if not seed_dir.exists():
        seed_dir = Path(args.policy_bundle) / f"seed{seed}"
    model_path = seed_dir / args.model_filename
    vec_path = seed_dir / "vec_normalize.pkl"
    sample_env = DummyVecEnv([lambda: make_env(args)])
    model = PPO.load(str(model_path), device="cpu")
    vec_norm = VecNormalize.load(str(vec_path), sample_env)
    vec_norm.training = False
    return model, vec_norm


def policy_action(model: Any, vec_norm: VecNormalize | None, obs: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if model is None or vec_norm is None:
        return np.asarray(fallback, dtype=np.float32)
    obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = model.predict(obs_norm, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def order_local_metrics(sim: Any, *, anchor_time: float, window_hours: float) -> dict[str, float]:
    horizon = float(sim.env.now)
    end_time = float(anchor_time) + float(window_hours)
    orders = []
    for order in sim.orders:
        if bool(getattr(order, "metrics_excluded", False)):
            continue
        opt = float(getattr(order, "OPTj", 0.0) or 0.0)
        oat_raw = getattr(order, "OATj", None)
        oat = float(oat_raw) if oat_raw is not None else horizon
        if opt <= end_time and oat >= anchor_time:
            orders.append(order)
    n = len(orders)
    if not orders:
        return {
            "event_n_orders": 0.0,
            "event_ret_excel_mean": 0.0,
            "event_ret_excel_cvar05": 0.0,
            "event_CTj_p95": 0.0,
            "event_CTj_p99": 0.0,
            "event_RPj_p95": 0.0,
            "event_RPj_p99": 0.0,
            "event_service_loss_auc": 0.0,
            "event_backlog_clearance_time": 0.0,
            "event_fill_branch_rate": 0.0,
        }

    ordered = sorted(orders, key=lambda o: (float(getattr(o, "OPTj", 0.0) or 0.0), int(getattr(o, "j", 0) or 0)))
    cumulative_backorders = 0
    cumulative_unattended = 0
    ret_values: list[float] = []
    fill_cases = 0
    service_loss_auc = 0.0
    ctj: list[float] = []
    rpj: list[float] = []
    clearance_times: list[float] = []

    for idx, order in enumerate(ordered, start=1):
        if bool(getattr(order, "lost", False)):
            cumulative_unattended += 1
        elif order_counts_as_backorder_for_fill_rate(order, current_time=horizon):
            cumulative_backorders += 1
        ret, case = compute_ret_per_order_excel_formula(
            order,
            j=idx,
            cumulative_backorders=cumulative_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        ret_values.append(float(ret))
        if str(case) == "excel_fill_rate":
            fill_cases += 1
        opt = float(getattr(order, "OPTj", 0.0) or 0.0)
        lt = float(getattr(order, "LTj", 0.0) or 0.0)
        oat_raw = getattr(order, "OATj", None)
        oat = float(oat_raw) if oat_raw is not None else horizon
        service_loss_auc += max(0.0, oat - (opt + lt)) * float(getattr(order, "quantity", 0.0) or 0.0)
        if getattr(order, "CTj", None) is not None:
            ctj.append(float(order.CTj))
        rp = float(getattr(order, "RPj", 0.0) or 0.0)
        if rp > 0.0:
            rpj.append(rp)
        if oat >= anchor_time:
            clearance_times.append(max(0.0, oat - anchor_time))

    xs = sorted(ret_values)
    tail_n = max(1, int(math.ceil(0.05 * len(xs))))
    return {
        "event_n_orders": float(n),
        "event_ret_excel_mean": float(sum(ret_values) / len(ret_values)),
        "event_ret_excel_cvar05": float(sum(xs[:tail_n]) / tail_n),
        "event_CTj_p95": percentile(ctj, 95),
        "event_CTj_p99": percentile(ctj, 99),
        "event_RPj_p95": percentile(rpj, 95),
        "event_RPj_p99": percentile(rpj, 99),
        "event_service_loss_auc": float(service_loss_auc),
        "event_backlog_clearance_time": max(clearance_times) if clearance_times else 0.0,
        "event_fill_branch_rate": float(fill_cases / n),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_episode(
    args: argparse.Namespace,
    *,
    seed: int,
    episode: int,
    risk_event_tape: list[dict[str, Any]] | None,
    model: Any,
    vec_norm: VecNormalize | None,
    base_action: np.ndarray,
    override_action: np.ndarray | None = None,
    override_window: tuple[float, float] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    env = make_env(args, risk_event_tape=risk_event_tape)
    eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode
    obs, _info = env.reset(seed=eval_seed)
    rows: list[dict[str, Any]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        now = float(env.unwrapped.sim.env.now)
        if override_action is not None and override_window is not None and override_window[0] <= now < override_window[1]:
            action = override_action
            source = "override"
        else:
            action = policy_action(model, vec_norm, obs, base_action)
            source = "policy" if model is not None else "static_base"
        obs, _reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "time": float(info.get("time", now)),
                "source": source,
                "action_mean": float(np.mean(np.asarray(action, dtype=np.float32))),
            }
        )
    sim = env.unwrapped.sim
    return sim, rows


def discover_tape(args: argparse.Namespace, *, seed: int, episode: int) -> list[dict[str, Any]]:
    if args.risk_event_tape:
        return load_risk_event_tape(args.risk_event_tape)
    sim, _ = run_episode(
        args,
        seed=seed,
        episode=episode,
        risk_event_tape=None,
        model=None,
        vec_norm=None,
        base_action=STATIC_VARIANTS[1].action,
    )
    return [event_to_record(event) for event in sim.risk_events]


def eligible_anchors(
    events: list[dict[str, Any]],
    *,
    target_risks: set[str],
    max_anchors: int,
    mode: str,
    min_anchor_time: float,
) -> list[tuple[int, dict[str, Any]]]:
    anchors = [
        (idx, event)
        for idx, event in enumerate(events)
        if str(event["risk_id"]) in target_risks
        and float(event["start_time"]) >= float(min_anchor_time)
    ]
    if mode == "strict_clean":
        clean: list[tuple[int, dict[str, Any]]] = []
        for idx, event in anchors:
            start = float(event["start_time"])
            lo = start - 6 * HOURS_PER_WEEK
            hi = start + 4 * HOURS_PER_WEEK
            overlaps = [
                other for j, other in enumerate(events)
                if j != idx
                and str(other["risk_id"]) in {"R22", "R23", "R24"}
                and lo <= float(other["start_time"]) <= hi
            ]
            if not overlaps:
                clean.append((idx, event))
        anchors = clean
    return anchors[:max_anchors]


def delta_row(prefix: str, a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    keys = [
        "event_ret_excel_mean",
        "event_ret_excel_cvar05",
        "event_CTj_p95",
        "event_CTj_p99",
        "event_RPj_p95",
        "event_RPj_p99",
        "event_service_loss_auc",
        "event_backlog_clearance_time",
        "event_fill_branch_rate",
    ]
    return {f"{prefix}_{key}": float(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["policy"]), str(row["comparison"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (mode, policy, comparison), group in sorted(groups.items()):
        deltas = [float(r.get("delta_event_ret_excel_mean", 0.0)) for r in group]
        positives = [x > 0.0 for x in deltas]
        by_seed: dict[int, list[float]] = {}
        for row, delta in zip(group, deltas):
            by_seed.setdefault(int(row["seed"]), []).append(delta)
        seed_means = [float(np.mean(v)) for v in by_seed.values()]
        ci_low, ci_high = ci95(seed_means)
        out.append(
            {
                "mode": mode,
                "policy": policy,
                "comparison": comparison,
                "n": len(group),
                "mean_delta_event_ret_excel": float(np.mean(deltas)) if deltas else 0.0,
                "positive_pair_rate": float(np.mean(positives)) if positives else 0.0,
                "seed_mean_ci95_low": ci_low,
                "seed_mean_ci95_high": ci_high,
                "positive_seed_count": int(sum(1 for x in seed_means if x > 0.0)),
                "seed_count": len(seed_means),
                "promotion_gate_pass": bool(
                    deltas
                    and np.mean(deltas) > 0.0
                    and np.mean(positives) > 0.60
                    and sum(1 for x in seed_means if x > 0.0) >= min(4, len(seed_means))
                    and ci_low > 0.0
                ),
            }
        )
    return out


def ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    mean = float(arr.mean())
    return mean - half, mean + half


def run_gate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_risks = {str(x) for x in args.target_risks}
    event_rows: list[dict[str, Any]] = []
    placebo_rows: list[dict[str, Any]] = []
    tape_rows: list[dict[str, Any]] = []
    base_variant = next(v for v in STATIC_VARIANTS if v.label == args.base_action)
    prep_variants = [v for v in STATIC_VARIANTS if v.label in set(args.prep_actions)]

    for seed in args.seeds:
        model, vec_norm = load_policy(args, seed)
        policy_name = args.policy_name or ("ppo_bundle" if model is not None else "static_medium")
        for episode in range(args.eval_episodes):
            tape = discover_tape(args, seed=seed, episode=episode)
            for event in tape:
                tape_rows.append({"seed": seed, "episode": episode + 1, **event})
            if args.write_tapes:
                save_risk_event_tape(tape, args.output_dir / "tapes" / f"seed{seed}_episode{episode + 1}.json")
            anchors = eligible_anchors(
                tape,
                target_risks=target_risks,
                max_anchors=args.max_anchors_per_episode,
                mode=args.anchor_filter,
                min_anchor_time=args.control_start_hours + args.lead_weeks * HOURS_PER_WEEK,
            )
            for anchor_idx, anchor in anchors:
                anchor_time = float(anchor["start_time"])
                pre_window = (
                    max(0.0, anchor_time - args.lead_weeks * HOURS_PER_WEEK),
                    anchor_time,
                )
                common = {
                    "seed": seed,
                    "episode": episode + 1,
                    "eval_seed": seed + EVAL_EPISODE_SEED_OFFSET + episode,
                    "policy": policy_name,
                    "mode": args.mode,
                    "anchor_index": anchor_idx,
                    "anchor_risk_id": str(anchor["risk_id"]),
                    "anchor_start_time": anchor_time,
                    "lead_weeks": args.lead_weeks,
                    "local_window_weeks": args.local_window_weeks,
                }

                if args.mode in {"forced_prep_sweep", "oracle_warning"}:
                    sim_calm, _ = run_episode(
                        args,
                        seed=seed,
                        episode=episode,
                        risk_event_tape=tape,
                        model=model,
                        vec_norm=vec_norm,
                        base_action=base_variant.action,
                        override_action=base_variant.action,
                        override_window=pre_window,
                    )
                    calm_metrics = order_local_metrics(
                        sim_calm,
                        anchor_time=anchor_time,
                        window_hours=args.local_window_weeks * HOURS_PER_WEEK,
                    )
                    for variant in prep_variants:
                        sim_prep, _ = run_episode(
                            args,
                            seed=seed,
                            episode=episode,
                            risk_event_tape=tape,
                            model=model,
                            vec_norm=vec_norm,
                            base_action=base_variant.action,
                            override_action=variant.action,
                            override_window=pre_window,
                        )
                        prep_metrics = order_local_metrics(
                            sim_prep,
                            anchor_time=anchor_time,
                            window_hours=args.local_window_weeks * HOURS_PER_WEEK,
                        )
                        event_rows.append(
                            {
                                **common,
                                "variant": variant.label,
                                "control_variant": base_variant.label,
                                "comparison": f"{variant.label}_minus_{base_variant.label}",
                                **{f"prep_{k}": v for k, v in prep_metrics.items()},
                                **{f"control_{k}": v for k, v in calm_metrics.items()},
                                **delta_row("delta", prep_metrics, calm_metrics),
                            }
                        )

                elif args.mode == "event_on_off":
                    sim_on, _ = run_episode(
                        args,
                        seed=seed,
                        episode=episode,
                        risk_event_tape=tape,
                        model=model,
                        vec_norm=vec_norm,
                        base_action=base_variant.action,
                    )
                    sim_off, _ = run_episode(
                        args,
                        seed=seed,
                        episode=episode,
                        risk_event_tape=remove_event_at_index(tape, anchor_idx),
                        model=model,
                        vec_norm=vec_norm,
                        base_action=base_variant.action,
                    )
                    on_metrics = order_local_metrics(
                        sim_on,
                        anchor_time=anchor_time,
                        window_hours=args.local_window_weeks * HOURS_PER_WEEK,
                    )
                    off_metrics = order_local_metrics(
                        sim_off,
                        anchor_time=anchor_time,
                        window_hours=args.local_window_weeks * HOURS_PER_WEEK,
                    )
                    event_rows.append(
                        {
                            **common,
                            "variant": "event_on",
                            "control_variant": "event_off",
                            "comparison": "event_on_minus_event_off",
                            **{f"on_{k}": v for k, v in on_metrics.items()},
                            **{f"off_{k}": v for k, v in off_metrics.items()},
                            **delta_row("delta", on_metrics, off_metrics),
                        }
                    )

            placebo_rows.extend(run_placebos(args, seed, episode, tape, model, vec_norm, base_variant, prep_variants, policy_name))
    return event_rows, summarize(event_rows), placebo_rows


def run_placebos(
    args: argparse.Namespace,
    seed: int,
    episode: int,
    tape: list[dict[str, Any]],
    model: Any,
    vec_norm: VecNormalize | None,
    base_variant: ActionVariant,
    prep_variants: list[ActionVariant],
    policy_name: str,
) -> list[dict[str, Any]]:
    if args.mode == "event_on_off":
        return []
    rng = np.random.default_rng(seed * 10_000 + episode)
    real_times = [float(e["start_time"]) for e in tape]
    rows: list[dict[str, Any]] = []
    for idx in range(args.placebo_anchors_per_episode):
        for _attempt in range(100):
            anchor_time = float(rng.uniform(args.lead_weeks * HOURS_PER_WEEK, args.max_steps * HOURS_PER_WEEK))
            if all(abs(anchor_time - t) > args.local_window_weeks * HOURS_PER_WEEK for t in real_times):
                break
        pre_window = (
            max(0.0, anchor_time - args.lead_weeks * HOURS_PER_WEEK),
            anchor_time,
        )
        sim_calm, _ = run_episode(
            args,
            seed=seed,
            episode=episode,
            risk_event_tape=tape,
            model=model,
            vec_norm=vec_norm,
            base_action=base_variant.action,
            override_action=base_variant.action,
            override_window=pre_window,
        )
        calm_metrics = order_local_metrics(
            sim_calm,
            anchor_time=anchor_time,
            window_hours=args.local_window_weeks * HOURS_PER_WEEK,
        )
        for variant in prep_variants:
            sim_prep, _ = run_episode(
                args,
                seed=seed,
                episode=episode,
                risk_event_tape=tape,
                model=model,
                vec_norm=vec_norm,
                base_action=base_variant.action,
                override_action=variant.action,
                override_window=pre_window,
            )
            prep_metrics = order_local_metrics(
                sim_prep,
                anchor_time=anchor_time,
                window_hours=args.local_window_weeks * HOURS_PER_WEEK,
            )
            rows.append(
                {
                    "seed": seed,
                    "episode": episode + 1,
                    "policy": policy_name,
                    "mode": f"{args.mode}_placebo",
                    "anchor_index": idx,
                    "anchor_risk_id": "FAKE",
                    "anchor_start_time": anchor_time,
                    "lead_weeks": args.lead_weeks,
                    "variant": variant.label,
                    "control_variant": base_variant.label,
                    "comparison": f"{variant.label}_minus_{base_variant.label}",
                    **{f"prep_{k}": v for k, v in prep_metrics.items()},
                    **{f"control_{k}": v for k, v in calm_metrics.items()},
                    **delta_row("delta", prep_metrics, calm_metrics),
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["forced_prep_sweep", "event_on_off", "oracle_warning"], required=True)
    parser.add_argument("--case-config", type=Path, default=Path("configs/track_b_case_c_r24_freq3_r22r23_impact1p5.json"))
    parser.add_argument("--risk-event-tape", type=Path, default=None)
    parser.add_argument("--write-tapes", action="store_true")
    parser.add_argument("--obs-config", default="v7_no_forecast", choices=list(OBS_ABLATION_CONFIGS))
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="current")
    parser.add_argument("--enabled-risks", nargs="+", default=["R22", "R23", "R24"])
    parser.add_argument("--risk-frequency-by-id", default="R24=3.0")
    parser.add_argument("--risk-impact-by-id", default="R22=1.5,R23=1.5")
    parser.add_argument("--target-risks", nargs="+", default=["R22", "R24"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--lead-weeks", type=int, default=4)
    parser.add_argument("--local-window-weeks", type=int, default=8)
    parser.add_argument("--control-start-hours", type=float, default=168.0)
    parser.add_argument("--max-anchors-per-episode", type=int, default=4)
    parser.add_argument("--anchor-filter", choices=["all", "strict_clean"], default="all")
    parser.add_argument("--base-action", choices=[v.label for v in STATIC_VARIANTS], default="medium")
    parser.add_argument("--prep-actions", nargs="+", choices=[v.label for v in STATIC_VARIANTS], default=["high_dispatch", "combined_prep"])
    parser.add_argument("--placebo-anchors-per-episode", type=int, default=1)
    parser.add_argument("--policy-bundle", type=Path, default=None)
    parser.add_argument("--policy-name", default=None)
    parser.add_argument("--model-filename", default="ppo_model.zip")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_rows, summary_rows, placebo_rows = run_gate(args)
    save_csv(args.output_dir / "gate_v2_event_rows.csv", event_rows)
    save_csv(args.output_dir / "gate_v2_summary.csv", summary_rows)
    save_csv(args.output_dir / "gate_v2_placebos.csv", placebo_rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "case_config": str(args.case_config),
        "obs_config": args.obs_config,
        "target_risks": list(args.target_risks),
        "seeds": list(args.seeds),
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "lead_weeks": args.lead_weeks,
        "local_window_weeks": args.local_window_weeks,
        "promotion_rule": {
            "mean_delta_event_ret_excel": "> 0",
            "positive_pair_rate": "> 0.60",
            "positive_seed_count": ">= 4 of 5 when five seeds are used",
            "seed_clustered_ci95_low": "> 0",
            "placebos": "approximately null",
        },
        "prevention_headroom_found": any(bool(row.get("promotion_gate_pass")) for row in summary_rows),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({"summary_rows": summary_rows, "metadata": metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
