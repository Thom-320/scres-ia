#!/usr/bin/env python3
"""Preventive-headroom ceiling test: forced-prep response surface on isolated anchors.

The go/no-go for all further prevention work (unanimous priority-1 across three
external reviews of the retracted Ruta B claim): before training or gating any
"preventive" policy, measure whether pre-event preparation has ANY causal value
in this DES's physics.

Design (no RL training, pure branch rollouts under fixed-RNG):

1. Roll the reference policy through each eval episode; record the exogenous
   risk-event calendar (action-independent under strict_exogenous_crn).
2. Select ISOLATED anchor events of the target risk (default R22): no other
   isolation-relevant event within [t_e - iso_before, t_e + iso_after] weeks.
   R24 is excluded from the isolation rule by default because at Case C
   frequency it blankets the calendar (that exclusion is itself reported).
3. Select PLACEBO anchors: event-free weeks satisfying the same isolation rule.
4. For each anchor and each forced posture in {calm=-1, medium=0, max_prep=+1}
   (applied to the whole 8D action vector during weeks [t_e-4, t_e-1]), rerun
   the episode from the same seed with the reference policy everywhere else.
5. Outcome is LOCAL, not episode ReT: per-order Garrido Excel ReT restricted to
   orders whose [OPTj, OATj-or-horizon] window overlaps
   [t_e, t_e + local_window] hours -- the orders causally exposed to the
   anchor. Episode ReT is recorded only as a reference column.

Estimand (difference-in-differences, per anchor pairing):

    headroom = (max_prep - calm | real anchors) - (max_prep - calm | placebo anchors)

A flat surface (headroom ~ 0, and max_prep ~ calm on real anchors) means the
environment does not reward preparation for this risk -- prevention is closed
as a boundary result and no gate redesign / architecture work is warranted.
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

from scripts.run_track_b_smoke import build_env_kwargs, build_parser as smoke_build_parser, save_csv  # noqa: E402
from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_ret_per_order_excel_formula,
    order_counts_as_backorder_for_fill_rate,
)
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402
from supply_chain.track_bp_env import make_track_bp_env  # noqa: E402

EVAL_EPISODE_SEED_OFFSET = 50_000
POSTURES = {"calm": -1.0, "medium": 0.0, "max_prep": 1.0}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, default=None,
                   help="Reference policy bundle: models/seed{N}/{ppo_model.zip,vec_normalize.pkl}. "
                        "Required for --reference-policy checkpoint; ignored for constant.")
    p.add_argument("--reference-policy", choices=["checkpoint", "constant"], default="checkpoint",
                   help="Policy that fills non-forced steps. 'constant' uses the neutral "
                        "medium action (all zeros) — no checkpoint needed, cleanest for a "
                        "pure-physics ceiling test on a new action contract.")
    p.add_argument("--env-factory", choices=["track_b", "track_bp"], default="track_b",
                   help="track_b = canonical 8D contract; track_bp = 11D preventive contract "
                        "(track_b_v1 + lagged buffer targets at Op3/Op5/Op9).")
    p.add_argument("--replenishment-lead-time", type=float, default=168.0,
                   help="Hours between a buffer target raise and stock arrival (track_bp only).")
    p.add_argument("--replenishment-period", type=float, default=None,
                   help="Optional sim-level buffer review cadence in hours (track_bp only).")
    p.add_argument("--initial-buffers", type=str, default=None,
                   help="Optional pre-positioned buffers at reset, e.g. 'op9_rations=15750'.")
    p.add_argument("--posture-dims", choices=["all", "buffers_only"], default="all",
                   help="Which action dims the forced posture controls. 'buffers_only' "
                        "forces only dims 8-10 (buffer fractions, track_bp) and leaves "
                        "dims 0-7 at the reference policy's action — isolates the buffer "
                        "channel from dispatch/shift preparation in the causal DiD.")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--obs-config", default="v7_no_forecast", choices=list(OBS_ABLATION_CONFIGS.keys()))
    # Environment scenario (defaults = Case C cell). For the Tier-1 "clean
    # physics" run, pass --enabled-risks R22 --risk-impact-by-id R22=1.5 and
    # drop the frequency multiplier.
    p.add_argument("--enabled-risks", default="R22,R23,R24")
    p.add_argument("--risk-frequency-by-id", default="R24=3")
    p.add_argument("--risk-impact-by-id", default="R22=1.5,R23=1.5")
    p.add_argument("--target-risk", default="R22")
    p.add_argument("--isolation-risks", nargs="+", default=["R22", "R23"],
                   help="Risks whose events invalidate an anchor's isolation window. R24 excluded "
                        "by default (blankets the calendar at Case C frequency).")
    p.add_argument("--iso-before-weeks", type=int, default=6)
    p.add_argument("--iso-after-weeks", type=int, default=4)
    p.add_argument("--prep-window-weeks", type=int, default=4)
    p.add_argument("--local-window-weeks", type=int, default=6,
                   help="Orders overlapping [t_e, t_e + this] hours count as exposed.")
    p.add_argument("--max-anchors-per-episode", type=int, default=4)
    p.add_argument("--max-placebos-per-episode", type=int, default=4)
    p.add_argument("--step-size-hours", type=float, default=168.0)
    # surge-inertia probe: makes shift changes lag + draw on a finite budget,
    # so pre-positioning capacity BEFORE a shock is rewarded (already wired in
    # run_track_b_smoke's parser + build_env_kwargs; just passed through here).
    p.add_argument("--surge-inertia", action="store_true")
    p.add_argument("--surge-ramp-per-step", type=int, default=1)
    p.add_argument("--surge-budget-hours", type=float, default=float("inf"))
    return p


def build_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = smoke_build_parser().parse_args([])
    args.risk_level = "current"
    args.faithful = True
    args.reward_mode = "control_v1"
    args.max_steps = cli.max_steps
    args.eval_episodes = cli.eval_episodes
    args.observation_version = OBS_ABLATION_CONFIGS[str(cli.obs_config)].observation_version
    args.enabled_risks = cli.enabled_risks
    args.risk_frequency_by_id = cli.risk_frequency_by_id or None
    args.risk_impact_by_id = cli.risk_impact_by_id or None
    if getattr(cli, "surge_inertia", False):
        args.surge_inertia = True
        args.surge_ramp_per_step = int(cli.surge_ramp_per_step)
        args.surge_budget_hours = float(cli.surge_budget_hours)
    if getattr(cli, "env_factory", "track_b") == "track_bp":
        args.inventory_replenishment_lead_time = float(cli.replenishment_lead_time)
        if cli.replenishment_period is not None:
            args.inventory_replenishment_period = float(cli.replenishment_period)
        if cli.initial_buffers:
            args.initial_buffers = cli.initial_buffers
    return args


def make_env(args: argparse.Namespace, obs_config: str, env_factory: str = "track_b"):
    kwargs = build_env_kwargs(args)
    if env_factory == "track_bp":
        lead = kwargs.pop("inventory_replenishment_lead_time", 168.0)
        env = make_track_bp_env(inventory_replenishment_lead_time=lead, **kwargs)
    else:
        env = make_track_b_env(**kwargs)
    wrapper = OBS_ABLATION_CONFIGS[str(obs_config)].wrapper
    if wrapper is not None:
        env = wrapper(env)
    return env


def load_policy(checkpoint_root: Path, seed: int, sample_env):
    seed_dir = checkpoint_root / "models" / f"seed{seed}"
    model = PPO.load(str(seed_dir / "ppo_model.zip"), device="cpu")
    vec_norm = VecNormalize.load(str(seed_dir / "vec_normalize.pkl"), DummyVecEnv([lambda: sample_env]))
    vec_norm.training = False
    return model, vec_norm


def predict_action(model, vec_norm, obs: np.ndarray) -> np.ndarray:
    obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = model.predict(obs_norm, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def local_order_outcomes(sim, *, window_start_hours: float, window_end_hours: float) -> dict[str, Any]:
    """Per-order Excel ReT restricted to orders exposed to [start, end].

    Reproduces the exact ordering + cumulative counters of
    episode_metrics._excel_ret_details so per-order values match the official
    metric, then filters to exposed orders.
    """
    orders = list(getattr(sim, "orders", []) or [])
    current_time = float(getattr(getattr(sim, "env", None), "now", 0.0) or 0.0)
    ordered = sorted(
        orders,
        key=lambda order: (int(getattr(order, "j", 0) or 0), float(getattr(order, "OPTj", 0.0) or 0.0)),
    )
    cumulative_backorders = 0
    cumulative_unattended = 0
    exposed_values: list[float] = []
    exposed_risk_touched: list[float] = []
    for idx, order in enumerate(ordered, start=1):
        if bool(getattr(order, "lost", False)):
            cumulative_unattended += 1
        elif order_counts_as_backorder_for_fill_rate(order, current_time=current_time):
            cumulative_backorders += 1
        ret, case = compute_ret_per_order_excel_formula(
            order,
            j=idx,
            cumulative_backorders=cumulative_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        opt = float(getattr(order, "OPTj", 0.0) or 0.0)
        oat = getattr(order, "OATj", None)
        end = float(oat) if oat is not None else current_time
        if end >= window_start_hours and opt <= window_end_hours:
            exposed_values.append(float(ret))
            if str(case) in ("excel_autotomy", "excel_recovery", "excel_risk_no_recovery"):
                exposed_risk_touched.append(float(ret))
    return {
        "local_ret_excel_mean": float(np.mean(exposed_values)) if exposed_values else 0.0,
        "local_n_orders": len(exposed_values),
        "local_ret_risk_conditional": float(np.mean(exposed_risk_touched)) if exposed_risk_touched else 0.0,
        "local_n_risk_touched": len(exposed_risk_touched),
    }


def run_episode(model, vec_norm, args, cli, eval_seed: int, *,
                forced_posture: float | None = None,
                forced_steps: set[int] | None = None) -> tuple[Any, list[int]]:
    env = make_env(args, cli.obs_config, getattr(cli, "env_factory", "track_b"))
    obs, _info = env.reset(seed=eval_seed)
    terminated = truncated = False
    step = 0
    shifts: list[int] = []
    action_dim = env.action_space.shape[0]
    buffers_only = getattr(cli, "posture_dims", "all") == "buffers_only"
    if buffers_only and action_dim <= 8:
        raise SystemExit("--posture-dims buffers_only requires the track_bp (11D) contract")
    while not (terminated or truncated):
        if model is None:
            # Constant-neutral reference: 'medium' posture on every dim.
            action = np.zeros(action_dim, dtype=np.float32)
        else:
            action = predict_action(model, vec_norm, obs)
        if forced_steps is not None and step in forced_steps:
            if buffers_only:
                # Force only the buffer fractions; dims 0-7 stay at reference.
                # Map the posture from [-1,1] to the buffer range [0,1] so the
                # calm/medium/max_prep surface stays 3-point (0 / 0.5 / 1.0).
                action = np.array(action, dtype=np.float32, copy=True)
                action[8:] = (float(forced_posture) + 1.0) / 2.0
            else:
                action = np.full(action_dim, float(forced_posture), dtype=np.float32)
        # Buffer fraction dims live in [0,1]; the env clips, so forced calm
        # (-1) decodes to zero buffers and max_prep (+1) to full buffers.
        obs, _reward, terminated, truncated, info = env.step(action)
        shifts.append(int(info.get("shifts_active", 1)))
        step += 1
    return env, shifts


def select_anchors(risk_events, cli) -> tuple[list[int], list[int]]:
    """Return (real anchor steps, placebo anchor steps) under the isolation rule."""
    step_h = float(cli.step_size_hours)
    events_by_risk: dict[str, list[int]] = {}
    for ev in risk_events:
        rid = str(ev.risk_id)
        events_by_risk.setdefault(rid, []).append(int(float(ev.start_time) // step_h))

    iso_steps: set[int] = set()
    for rid in cli.isolation_risks:
        iso_steps.update(events_by_risk.get(rid, []))

    def isolated(step: int, *, exclude_self_at: int | None = None) -> bool:
        for other in iso_steps:
            if exclude_self_at is not None and other == exclude_self_at:
                continue
            if step - cli.iso_before_weeks <= other <= step + cli.iso_after_weeks:
                return False
        return True

    target_steps = sorted(set(events_by_risk.get(str(cli.target_risk), [])))
    real = [
        t for t in target_steps
        if cli.prep_window_weeks <= t < cli.max_steps - cli.iso_after_weeks and isolated(t, exclude_self_at=t)
    ][: cli.max_anchors_per_episode]

    # Placebo: event-free weeks with the same isolation property, sampled
    # deterministically across the episode, at least iso window away from any
    # target event.
    target_set = set(target_steps)
    placebo_candidates = [
        t for t in range(cli.prep_window_weeks + cli.iso_before_weeks,
                         cli.max_steps - cli.iso_after_weeks)
        if isolated(t) and all(abs(t - ts) > cli.iso_after_weeks for ts in target_set)
    ]
    if placebo_candidates and cli.max_placebos_per_episode > 0:
        idx = np.linspace(0, len(placebo_candidates) - 1, num=min(cli.max_placebos_per_episode, len(placebo_candidates)))
        placebo = sorted({placebo_candidates[int(round(i))] for i in idx})
    else:
        placebo = []
    return real, placebo


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)
    step_h = float(cli.step_size_hours)
    local_window_h = cli.local_window_weeks * step_h

    rows: list[dict[str, Any]] = []
    anchor_counts = {"real": 0, "placebo": 0}

    if cli.reference_policy == "checkpoint" and cli.checkpoint_root is None:
        raise SystemExit("--checkpoint-root is required with --reference-policy checkpoint")

    for seed in cli.seeds:
        if cli.reference_policy == "constant":
            model, vec_norm = None, None
        else:
            sample_env = make_env(args, cli.obs_config, getattr(cli, "env_factory", "track_b"))
            model, vec_norm = load_policy(cli.checkpoint_root, seed, sample_env)
        for episode_idx in range(cli.eval_episodes):
            eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
            # Reference pass: discover calendar (exogenous under strict CRN).
            env, ref_shifts = run_episode(model, vec_norm, args, cli, eval_seed)
            risk_events = list(env.unwrapped.sim.risk_events)
            real, placebo = select_anchors(risk_events, cli)
            anchor_counts["real"] += len(real)
            anchor_counts["placebo"] += len(placebo)
            # Reference arm: the policy's own (unforced) rollout, evaluated
            # locally at every anchor, for context on whether the policy's
            # natural pre-window behavior differs from the forced postures.
            ref_metrics = compute_episode_metrics(env.unwrapped.sim)
            for anchor_kind, anchor_steps in (("real", real), ("placebo", placebo)):
                for anchor_step in anchor_steps:
                    local = local_order_outcomes(
                        env.unwrapped.sim,
                        window_start_hours=anchor_step * step_h,
                        window_end_hours=anchor_step * step_h + local_window_h,
                    )
                    rows.append({
                        "seed": seed,
                        "episode": episode_idx + 1,
                        "eval_seed": eval_seed,
                        "anchor_kind": anchor_kind,
                        "anchor_step": anchor_step,
                        "posture": "policy_unforced",
                        "episode_ret_excel": float(ref_metrics["ret_excel"]),
                        "cost_index": float(sum(ref_shifts) / (3.0 * len(ref_shifts))) if ref_shifts else 0.0,
                        **local,
                    })
            env.close()

            for anchor_kind, anchor_steps in (("real", real), ("placebo", placebo)):
                for anchor_step in anchor_steps:
                    forced_steps = {
                        s for s in range(anchor_step - cli.prep_window_weeks, anchor_step)
                        if 0 <= s < cli.max_steps
                    }
                    for posture_name, posture_value in POSTURES.items():
                        env, shifts = run_episode(
                            model, vec_norm, args, cli, eval_seed,
                            forced_posture=posture_value, forced_steps=forced_steps,
                        )
                        metrics = compute_episode_metrics(env.unwrapped.sim)
                        local = local_order_outcomes(
                            env.unwrapped.sim,
                            window_start_hours=anchor_step * step_h,
                            window_end_hours=anchor_step * step_h + local_window_h,
                        )
                        env.close()
                        rows.append({
                            "seed": seed,
                            "episode": episode_idx + 1,
                            "eval_seed": eval_seed,
                            "anchor_kind": anchor_kind,
                            "anchor_step": anchor_step,
                            "posture": posture_name,
                            "episode_ret_excel": float(metrics["ret_excel"]),
                            "cost_index": float(sum(shifts) / (3.0 * len(shifts))) if shifts else 0.0,
                            **local,
                        })
            print(f"seed {seed} ep {episode_idx+1}: {len(real)} real / {len(placebo)} placebo anchors", flush=True)

    save_csv(out / "headroom_rows.csv", rows)

    # Aggregate the response surface + DiD estimand.
    def agg(kind: str, posture: str, key: str) -> float:
        vals = [r[key] for r in rows if r["anchor_kind"] == kind and r["posture"] == posture]
        return float(np.mean(vals)) if vals else 0.0

    surface = {
        kind: {p: {k: agg(kind, p, k) for k in ("local_ret_excel_mean", "local_ret_risk_conditional", "episode_ret_excel")}
               for p in POSTURES}
        for kind in ("real", "placebo")
    }
    did = {
        k: (surface["real"]["max_prep"][k] - surface["real"]["calm"][k])
           - (surface["placebo"]["max_prep"][k] - surface["placebo"]["calm"][k])
        for k in ("local_ret_excel_mean", "local_ret_risk_conditional", "episode_ret_excel")
    }

    # Per-anchor paired deltas (max_prep - calm) for sign tests.
    paired: dict[str, list[float]] = {"real": [], "placebo": []}
    by_anchor: dict[tuple, dict[str, float]] = {}
    for r in rows:
        key = (r["seed"], r["eval_seed"], r["anchor_kind"], r["anchor_step"])
        by_anchor.setdefault(key, {})[r["posture"]] = r["local_ret_excel_mean"]
    for (seed, eval_seed, kind, anchor), postures in by_anchor.items():
        if "max_prep" in postures and "calm" in postures:
            paired[kind].append(postures["max_prep"] - postures["calm"])

    summary = {
        "config": vars(cli),
        "resolved_env_kwargs": {k: repr(v) for k, v in sorted(build_env_kwargs(args).items())},
        "anchor_counts": anchor_counts,
        "response_surface": surface,
        "did_max_prep_vs_calm": did,
        "paired_real_n": len(paired["real"]),
        "paired_real_positive": sum(1 for d in paired["real"] if d > 0),
        "paired_real_mean_delta": float(np.mean(paired["real"])) if paired["real"] else 0.0,
        "paired_placebo_n": len(paired["placebo"]),
        "paired_placebo_positive": sum(1 for d in paired["placebo"] if d > 0),
        "paired_placebo_mean_delta": float(np.mean(paired["placebo"])) if paired["placebo"] else 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "resolved_env_kwargs")}, indent=2))


if __name__ == "__main__":
    main()
