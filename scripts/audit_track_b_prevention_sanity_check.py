#!/usr/bin/env python3
"""Sanity-check the preventive/reactive classification rule against known references.

The main audit (``scripts/audit_track_b_prevention_mechanism.py``) classified PPO+MLP and
Real-KAN as reactive. Before trusting that verdict, this script validates the classification
RULE ITSELF against three references with known expected behavior:

- a pure static policy -> should show no adaptive signal ("sin senal clara"),
- ``heur_forecast_threshold`` (raises shift/dispatch when the forecast crosses a threshold,
  i.e. before the risk is realized) -> should classify as preventive,
- ``heur_downstream_reactive`` (raises shift/dispatch after observing queue pressure/backlog,
  i.e. after the risk already hit) -> should classify as reactive.

Reuses the pure helper functions from the main audit module (event anchor inference, window
labeling, episode-metric finalization, per-window aggregation) rather than reimplementing them,
so both scripts share exactly the same classification logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_prevention_mechanism import (  # noqa: E402
    ACTION_DIMS,
    EVAL_EPISODE_SEED_OFFSET,
    NEUTRAL_STATIC,
    aggregate_policy_windows,
    env_kwargs as build_env_kwargs,
    finalize_episode,
    group_event_study,
    infer_event_anchor,
    mean,
    row_from_step,
    save_csv,
    static_spec_from_label,
    window_label,
)
from scripts.run_track_b_smoke import build_static_policy_action  # noqa: E402
from scripts.track_b_heuristics import make_heuristic_defaults  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_ret_per_order_excel_formula,
    order_counts_as_backorder_for_fill_rate,
)


POST_WINDOW_END = 20  # widened from the original 8 weeks: lagging signals (queue
# pressure, backlog) can take longer than 8 weeks to build up after a leading
# regime-based anchor, in a 13-op chain with real transport lead times.


def wide_window_label(relative_week: int) -> str:
    """Same bands as ``window_label`` but with a wider post-event window."""
    if -4 <= relative_week <= -1:
        return "pre"
    if relative_week == 0:
        return "event"
    if 1 <= relative_week <= POST_WINDOW_END:
        return "post"
    return "baseline"


def windowed_ret_excel(
    orders: list[Any],
    *,
    window_start_hours: float,
    window_end_hours: float,
    current_time: float | None,
) -> float | None:
    """Mean Excel ReT restricted to orders placed inside [start, end) hours.

    Replicates ``compute_order_level_ret_excel_formula``'s cumulative-state loop
    exactly (so ``cumulative_backorders``/``cumulative_unattended`` reflect the
    TRUE full-episode history up to each order), but only *collects* ret values
    for orders whose placement time (OPTj) falls inside the window. This avoids
    diluting a window-local effect against ~90 untouched weeks of the episode.
    """
    order_list = sorted(
        list(orders),
        key=lambda order: (int(getattr(order, "j", 0) or 0), float(getattr(order, "OPTj", 0.0) or 0.0)),
    )
    cumulative_backorders = 0
    cumulative_unattended = 0
    windowed_values: list[float] = []
    for idx, order in enumerate(order_list, start=1):
        if bool(getattr(order, "lost", False)):
            cumulative_unattended += 1
        elif order_counts_as_backorder_for_fill_rate(order, current_time=current_time):
            cumulative_backorders += 1
        ret, _case = compute_ret_per_order_excel_formula(
            order,
            j=idx,
            cumulative_backorders=cumulative_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        optj = float(getattr(order, "OPTj", 0.0) or 0.0)
        if window_start_hours <= optj < window_end_hours:
            windowed_values.append(ret)
    if not windowed_values:
        return None
    return float(np.mean(windowed_values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/track_b_prevention_sanity_check_2026-07-03"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    return parser.parse_args()


def action_fn_for_static(label: str) -> Callable[[np.ndarray, dict[str, Any]], Any]:
    action = build_static_policy_action(static_spec_from_label(label))

    def _fn(_obs: np.ndarray, _info: dict[str, Any]) -> Any:
        return action

    return _fn


def action_fn_for_heuristic(name: str) -> Callable[[np.ndarray, dict[str, Any]], Any]:
    heuristic = make_heuristic_defaults()[name]

    def _fn(obs: np.ndarray, info: dict[str, Any]) -> Any:
        return np.asarray(heuristic(obs, info), dtype=np.float32)

    return _fn


def run_episode_for_reference(
    *,
    action_fn: Callable[[np.ndarray, dict[str, Any]], Any],
    heuristic_reset: Callable[[], None] | None,
    args: argparse.Namespace,
    policy_name: str,
    seed: int,
    episode: int,
    eval_seed: int,
    neutral_action: Any,
    reset_window: tuple[int, int] | None = None,
    anchor_step: int | None = None,
    keep_rows: bool = True,
) -> dict[str, Any]:
    env = make_track_b_env(**build_env_kwargs(args))
    if heuristic_reset is not None:
        heuristic_reset()
    obs, info = env.reset(seed=eval_seed)
    terminated = False
    truncated = False
    step = 0
    rows: list[dict[str, Any]] = []
    shifts: list[int] = []
    final_info = info

    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action = action_fn(obs_before, final_info)
        if reset_window is not None and anchor_step is not None:
            rel = step - int(anchor_step)
            if int(reset_window[0]) <= rel <= int(reset_window[1]):
                action = neutral_action
        obs, reward, terminated, truncated, final_info = env.step(action)
        shifts.append(int(final_info.get("shifts_active", 1)))
        if keep_rows:
            rows.append(
                row_from_step(
                    policy=policy_name,
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                    condition="full",
                    step=step,
                    obs_before=obs_before,
                    reward=float(reward),
                    info=final_info,
                )
            )
        step += 1

    ret_excel, fill_rate, service_loss_auc, cost_index = finalize_episode(env, shifts)
    orders_snapshot = list(env.unwrapped.sim.orders)
    horizon = float(env.unwrapped.sim.env.now)
    env.close()
    anchor = infer_event_anchor(rows) if anchor_step is None else int(anchor_step)
    for row in rows:
        row["event_anchor_step"] = anchor
        row["relative_week"] = int(row["step"]) - anchor
        row["event_window"] = wide_window_label(int(row["relative_week"]))
    return {
        "rows": rows,
        "ret_excel": ret_excel,
        "anchor_step": anchor,
        "orders": orders_snapshot,
        "horizon": horizon,
    }


def aggregate_policy_windows_windowed(
    step_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    expected_by_policy: dict[str, str],
) -> list[dict[str, Any]]:
    """Same PAI/RRI behavioral indices as the main audit, but the causal-value
    side (delta_pre/delta_post) uses the window-local ReT instead of the
    full-episode ReT, to avoid diluting a 13-week intervention against ~90
    untouched weeks."""
    policies = sorted({str(row["policy"]) for row in step_rows})
    out: list[dict[str, Any]] = []
    for policy in policies:
        rows = [r for r in step_rows if r["policy"] == policy]
        base = [float(r["action_intensity"]) for r in rows if r.get("event_window") == "baseline"]
        pre = [float(r["action_intensity"]) for r in rows if r.get("event_window") == "pre"]
        post = [float(r["action_intensity"]) for r in rows if r.get("event_window") == "post"]
        cf = [r for r in counterfactual_rows if r["policy"] == policy]
        delta_by_window = {}
        for w in ("pre", "event", "post"):
            vals = [
                float(r["delta_ret_excel_windowed"])
                for r in cf
                if r["reset_window"] == w and r.get("delta_ret_excel_windowed") is not None
            ]
            delta_by_window[w] = mean(vals) if vals else 0.0
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
                "expected": expected_by_policy.get(policy, ""),
                "preventive_activation_index": pai,
                "reactive_response_index": rri,
                "recovery_effect_index": rei,
                "delta_ret_reset_pre_windowed": delta_by_window["pre"],
                "delta_ret_reset_event_windowed": delta_by_window["event"],
                "delta_ret_reset_post_windowed": delta_by_window["post"],
                "classification": classification,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    neutral_action = build_static_policy_action(static_spec_from_label(NEUTRAL_STATIC.label))

    references = {
        "static_s2_d1.50": (action_fn_for_static("s2_d1.50"), None, "sin señal clara"),
        "heur_forecast_threshold": (
            action_fn_for_heuristic("heur_forecast_threshold"),
            make_heuristic_defaults()["heur_forecast_threshold"].reset,
            "preventiva",
        ),
        "heur_downstream_reactive": (
            action_fn_for_heuristic("heur_downstream_reactive"),
            make_heuristic_defaults()["heur_downstream_reactive"].reset,
            "reactiva",
        ),
    }

    step_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    expected_by_policy: dict[str, str] = {}
    full_cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    # Pass 1: run the real (frozen) policy on every episode, cache rows/orders/anchor.
    for policy_name, (action_fn, reset_fn, expected) in references.items():
        expected_by_policy[policy_name] = expected
        for seed in args.seeds:
            for episode in range(1, int(args.eval_episodes) + 1):
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                full = run_episode_for_reference(
                    action_fn=action_fn,
                    heuristic_reset=reset_fn,
                    args=args,
                    policy_name=policy_name,
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                    neutral_action=neutral_action,
                    keep_rows=True,
                )
                full_cache[(policy_name, seed, episode)] = full
                step_rows.extend(full["rows"])

    # Per-policy "own calm" reference: mean raw action vector over baseline-window
    # steps (far from any risk event), instead of one shared static for everyone.
    # Fixes a confound where resetting to a fixed reference (e.g. s2_d1.50) changes
    # dimensions (like shift) the policy under test never touches on its own.
    policy_baseline_action: dict[str, np.ndarray] = {}
    for policy_name in references:
        baseline_rows = [
            r for r in step_rows if r["policy"] == policy_name and r.get("event_window") == "baseline"
        ]
        def _numeric(rows: list[dict[str, Any]], name: str) -> list[float]:
            out = []
            for r in rows:
                val = r.get(f"action_{name}")
                if val not in (None, ""):
                    out.append(float(val))
            return out

        if baseline_rows and all(_numeric(baseline_rows, name) for name in ACTION_DIMS):
            vec = np.array([mean(_numeric(baseline_rows, name)) for name in ACTION_DIMS], dtype=np.float32)
        else:
            vec = np.zeros(len(ACTION_DIMS), dtype=np.float32)
        policy_baseline_action[policy_name] = vec

    # Pass 2: counterfactual reset windows, using each policy's OWN baseline action.
    for policy_name, (action_fn, reset_fn, expected) in references.items():
        own_neutral = policy_baseline_action[policy_name]
        for seed in args.seeds:
            for episode in range(1, int(args.eval_episodes) + 1):
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                full = full_cache[(policy_name, seed, episode)]
                for label, window in {"pre": (-4, -1), "event": (0, 0), "post": (1, POST_WINDOW_END)}.items():
                    reset = run_episode_for_reference(
                        action_fn=action_fn,
                        heuristic_reset=reset_fn,
                        args=args,
                        policy_name=policy_name,
                        seed=seed,
                        episode=episode,
                        eval_seed=eval_seed,
                        neutral_action=own_neutral,
                        reset_window=window,
                        anchor_step=full["anchor_step"],
                        keep_rows=False,
                    )
                    window_start_hours = (full["anchor_step"] + window[0]) * float(args.step_size_hours)
                    window_end_hours = (full["anchor_step"] + window[1] + 1) * float(args.step_size_hours)
                    horizon = max(full["horizon"], reset["horizon"])
                    full_windowed = windowed_ret_excel(
                        full["orders"],
                        window_start_hours=window_start_hours,
                        window_end_hours=window_end_hours,
                        current_time=horizon,
                    )
                    reset_windowed = windowed_ret_excel(
                        reset["orders"],
                        window_start_hours=window_start_hours,
                        window_end_hours=window_end_hours,
                        current_time=horizon,
                    )
                    delta_windowed = (
                        (full_windowed - reset_windowed)
                        if full_windowed is not None and reset_windowed is not None
                        else None
                    )
                    counterfactual_rows.append(
                        {
                            "policy": policy_name,
                            "seed": seed,
                            "episode": episode,
                            "eval_seed": eval_seed,
                            "reset_window": label,
                            "R_full": full["ret_excel"],
                            "R_reset": reset["ret_excel"],
                            "delta_ret_excel": full["ret_excel"] - reset["ret_excel"],
                            "R_full_windowed": full_windowed,
                            "R_reset_windowed": reset_windowed,
                            "delta_ret_excel_windowed": delta_windowed,
                        }
                    )

    event_study_rows = group_event_study(step_rows)
    summary_rows = aggregate_policy_windows(step_rows, counterfactual_rows, [])
    windowed_summary_rows = aggregate_policy_windows_windowed(
        step_rows, counterfactual_rows, expected_by_policy
    )

    save_csv(out / "step_ledger.csv", step_rows)
    save_csv(out / "event_study.csv", event_study_rows)
    save_csv(out / "counterfactual_reset.csv", counterfactual_rows)
    save_csv(out / "policy_classification.csv", summary_rows)
    save_csv(out / "policy_classification_windowed.csv", windowed_summary_rows)

    print("\n=== FULL-EPISODE delta (original method) ===")
    for row in summary_rows:
        print(row["policy"], "->", row["classification"], "delta_pre=", row["delta_ret_reset_pre"], "delta_post=", row["delta_ret_reset_post"])
    print("\n=== WINDOWED delta (fix) ===")
    for row in windowed_summary_rows:
        print(row["policy"], "->", row["classification"], "delta_pre=", row["delta_ret_reset_pre_windowed"], "delta_post=", row["delta_ret_reset_post_windowed"])

    verdict_lines = ["policy,expected,actual,match"]
    all_match = True
    for row in windowed_summary_rows:
        expected = expected_by_policy[row["policy"]]
        actual = row["classification"]
        match = expected == actual
        all_match = all_match and match
        verdict_lines.append(f"{row['policy']},{expected},{actual},{match}")
    (out / "sanity_verdict.csv").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    summary = {
        "config": {
            "seeds": args.seeds,
            "eval_episodes": args.eval_episodes,
            "policies": list(references.keys()),
        },
        "policy_classification": summary_rows,
        "expected": expected_by_policy,
        "all_references_match_expectation": all_match,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"all_references_match_expectation": all_match}, indent=2))
    for line in verdict_lines:
        print(line)
    print(f"Wrote sanity-check bundle: {out}")


if __name__ == "__main__":
    main()
