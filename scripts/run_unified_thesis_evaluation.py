#!/usr/bin/env python3
"""Unified thesis-decision evaluation protocol.

Every candidate policy (Garrido per-scenario oracle, best static uniform L1a,
best static per-node L1b, and trained PPO agents E2a/E2b) is scored on the SAME
panel of Garrido Cf scenarios, with the SAME stochastic seed per
(scenario, replication). Seeds depend only on (cfi, replication) -- never on the
policy -- so every policy sees an identical demand/disruption realization. This
is the paired design the ladder runs lacked: L0 was scored on each policy's own
matched Cf scenario (fill saturates at 1.0, "260 band") while L1a/L1b/PPO-direct
were scored on a single common scenario ("130 band"). Those bands are not
comparable; this script removes the confound.

Horizon caveat: --max-steps is a fixed abbreviated horizon applied identically
to all policies (default 260 weeks ~= 5.4 years), NOT the thesis 10/20-year
horizon. It is for cross-policy comparability, not a final replica-horizon claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import RISK_OCCURRENCE_MODE_OPTIONS  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import (  # noqa: E402
    design_spec_for_cfi,
    parse_cf_range,
)

# ----------------------------- action builders -----------------------------


def thesis_factorized_action(period: int | None, shifts: int) -> np.ndarray:
    if period is None:
        return np.array([0, shifts - 1], dtype=np.int64)
    return np.array(
        [THESIS_INVENTORY_PERIODS.index(int(period)) + 1, shifts - 1], dtype=np.int64
    )


def per_node_action(
    op3: int | None, op5: int | None, op9: int | None, shifts: int
) -> np.ndarray:
    levels = [
        0 if p is None else THESIS_INVENTORY_PERIODS.index(int(p)) + 1
        for p in (op3, op5, op9)
    ]
    return np.array([*levels, shifts - 1], dtype=np.int64)


def thesis_design_action(spec: Any) -> np.ndarray:
    period = spec.inventory_replenishment_period
    return thesis_factorized_action(
        None if period is None else int(period), spec.shifts
    )


def _period_from_label(label: str) -> int | None:
    """'I0' -> None, 'I672' -> 672."""
    n = int(label[1:])
    return None if n == 0 else n


# ----------------------------- env / rollout -------------------------------


def base_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "stochastic_pt": args.stochastic_pt,
        "learn_initial_decision": False,
        "raw_material_flow_mode": getattr(
            args, "raw_material_flow_mode", "legacy_validated"
        ),
        "raw_material_order_up_to_multiplier": getattr(
            args, "raw_material_order_up_to_multiplier", 2.0
        ),
        "risk_occurrence_mode": getattr(args, "risk_occurrence_mode", "legacy_renewal"),
    }


def rollout(
    *,
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    action_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    eval_seed: int,
) -> dict[str, float]:
    env = make_dkana_thesis_faithful_env(**env_kwargs)
    obs, info = env.reset(seed=eval_seed)
    terminated = truncated = False
    reward_total = 0.0
    steps = 0
    shift_counts = {1: 0, 2: 0, 3: 0}
    while not (terminated or truncated):
        action = action_fn(np.asarray(obs, dtype=np.float32), info)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        if info.get("action_phase") == "initial_decision":
            continue
        steps += 1
        shift = int(info.get("thesis_decision", {}).get("assembly_shifts", 1))
        shift_counts[shift] = shift_counts.get(shift, 0) + 1
    terminal = get_episode_terminal_metrics(env)
    total = max(1, steps)
    env.close()
    return {
        "reward_total": reward_total,
        "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
        "order_level_ret_mean": float(terminal["order_level_ret_mean"]),
        "backorder_rate_order_level": float(terminal["backorder_rate_order_level"]),
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total,
        "steps": steps,
    }


def fixed_action_fn(action: np.ndarray) -> Callable[..., np.ndarray]:
    arr = np.asarray(action)
    return lambda obs, info: arr


def model_action_fn(model: Any, vec: Any) -> Callable[..., np.ndarray]:
    def _fn(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        model_obs = obs
        if vec is not None:
            model_obs = vec.normalize_obs(obs.reshape(1, -1))[0]
        action, _ = model.predict(model_obs, deterministic=True)
        return np.asarray(action)

    return _fn


# ----------------------------- candidates ----------------------------------


def l1a_candidates() -> list[tuple[str, str, str, np.ndarray]]:
    """(name, action_space_mode, inventory_period_mode, action) for the 18 uniform configs."""
    out = []
    for period in [None, *THESIS_INVENTORY_PERIODS]:
        plabel = "I0" if period is None else f"I{period}"
        for shifts in (1, 2, 3):
            out.append(
                (
                    f"L1a_uniform_{plabel}_S{shifts}",
                    "thesis_factorized",
                    "thesis_strict",
                    thesis_factorized_action(period, shifts),
                )
            )
    return out


def _canonical_l1b_policy_name(name: str) -> str:
    return name[:-4] if name.endswith("_top") else name


def l1b_candidates_from_screening(
    csv_path: Path, top_k: int, *, selection: str
) -> list[tuple[str, str, str, np.ndarray]]:
    """Read prior L1b policy_summary and parse selected per-node actions.

    The screening winner was noisy in the overnight run; the best confirmed
    top-k policy ranked only 16th during screening. The unified panel should
    therefore include both screening leaders and confirmed top-k leaders unless
    the caller explicitly narrows the source.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[df["policy"].astype(str).str.startswith("L1b_per_node_")]
    frames = []
    if "stage" not in df.columns:
        frames.append(
            df.sort_values("fill_rate_order_level_mean", ascending=False).head(top_k)
        )
    else:
        if selection in {"screening", "combined"}:
            frames.append(
                df[df["stage"].eq("screening")]
                .sort_values("fill_rate_order_level_mean", ascending=False)
                .head(top_k)
            )
        if selection in {"top_k", "combined"}:
            frames.append(
                df[df["stage"].eq("top_k")]
                .sort_values("fill_rate_order_level_mean", ascending=False)
                .head(top_k)
            )
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    pat = re.compile(r"L1b_per_node_(I\d+)_(I\d+)_(I\d+)_S(\d)")
    out = []
    seen: set[str] = set()
    for raw_name in df["policy"]:
        name = _canonical_l1b_policy_name(str(raw_name))
        if name in seen:
            continue
        seen.add(name)
        m = pat.fullmatch(name)
        if not m:
            continue
        op3, op5, op9 = (_period_from_label(m.group(i)) for i in (1, 2, 3))
        shifts = int(m.group(4))
        out.append(
            (name, "factorized", "per_node", per_node_action(op3, op5, op9, shifts))
        )
    return out


def load_ppo(model_zip: Path, vecnorm_pkl: Path | None):
    from stable_baselines3 import PPO

    model = PPO.load(str(model_zip), device="cpu")
    vec = None
    if vecnorm_pkl and vecnorm_pkl.exists():
        with open(vecnorm_pkl, "rb") as f:
            vec = pickle.load(f)
        vec.training = False
        vec.norm_reward = False
    return model, vec


# ----------------------------- main ----------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default=None)
    p.add_argument("--output-root", default="outputs/benchmarks/unified_evaluation")
    p.add_argument(
        "--panel-cfis", default="31-90", help="Scenario panel (Garrido Cf range)."
    )
    p.add_argument(
        "--replications", type=int, default=3, help="Reps per (policy, scenario)."
    )
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=260)
    p.add_argument("--step-size-hours", type=float, default=168.0)
    p.add_argument("--reward-mode", default="ReT_cd_v1")
    p.add_argument("--risk-level", default="increased")
    p.add_argument("--observation-version", default="v5")
    p.add_argument("--observation-mode", default="env_sdm_history_reward")
    p.add_argument("--stochastic-pt", action="store_true")
    p.add_argument(
        "--raw-material-flow-mode",
        default="legacy_validated",
        help="Raw-material flow semantics for post-fix thesis-inventory reruns.",
    )
    p.add_argument("--raw-material-order-up-to-multiplier", type=float, default=2.0)
    p.add_argument(
        "--risk-occurrence-mode",
        choices=RISK_OCCURRENCE_MODE_OPTIONS,
        default="legacy_renewal",
    )
    p.add_argument(
        "--l1b-screening-csv",
        default=None,
        help="Prior L1b policy_summary.csv for candidate actions.",
    )
    p.add_argument("--l1b-top-k", type=int, default=10)
    p.add_argument(
        "--l1b-selection",
        choices=["screening", "top_k", "combined"],
        default="combined",
        help=(
            "Which L1b candidate source to evaluate. combined includes both "
            "screening leaders and confirmed top-k leaders, de-duplicated by action."
        ),
    )
    p.add_argument(
        "--ppo-run-root",
        default=None,
        help="Dir holding ppo_adaptive/<run>/ppo_mlp_thesis_decision.zip",
    )
    p.add_argument("--include-garrido-oracle", action="store_true", default=True)
    p.add_argument(
        "--no-garrido-oracle", dest="include_garrido_oracle", action="store_false"
    )
    p.add_argument("--include-l1a", action="store_true", default=True)
    p.add_argument("--no-l1a", dest="include_l1a", action="store_false")
    return p


def discover_ppo_models(run_root: Path) -> list[tuple[str, str, str, Path, Path, bool]]:
    """Return (name, action_space_mode, inventory_period_mode, zip, vecnorm, learn_initial_decision)."""
    out = []
    ppo_dir = run_root / "ppo_adaptive"
    if not ppo_dir.exists():
        return out
    for d in sorted(ppo_dir.glob("*")):
        zip_path = d / "ppo_mlp_thesis_decision.zip"
        if not zip_path.exists():
            continue
        vec = d / "vecnormalize.pkl"
        name = d.name
        if "E2a" in name or "thesis_factorized" in name:
            asm, ipm = "thesis_factorized", "thesis_strict"
        elif "E2b" in name or "per_node" in name:
            asm, ipm = "factorized", "per_node"
        else:
            # fall back to summary.json
            try:
                s = json.loads((d / "summary.json").read_text())
                asm = s.get("action_space_mode", "thesis_factorized")
                ipm = s.get("inventory_period_mode", "thesis_strict")
            except Exception:
                asm, ipm = "thesis_factorized", "thesis_strict"
        learn_initial_decision = False
        try:
            summary = json.loads((d / "summary.json").read_text())
            learn_initial_decision = bool(
                summary.get("env_kwargs", {}).get("learn_initial_decision", False)
            )
        except Exception:
            pass
        short = re.sub(r".*_(E2[ab]_[a-z_]*seed\d+)$", r"\1", name)
        out.append((f"PPO_{short}", asm, ipm, zip_path, vec, learn_initial_decision))
    return out


def main() -> int:
    args = build_parser().parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or f"unified_eval_{stamp}"
    out_dir = Path(args.output_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = parse_cf_range(args.panel_cfis)
    specs = {cfi: design_spec_for_cfi(cfi) for cfi in panel}
    print(
        f"panel: {len(panel)} scenarios (Cf{panel[0]}..Cf{panel[-1]}), reps={args.replications}, max_steps={args.max_steps}"
    )

    # assemble candidate list: (name, asm, ipm, action_or_None, model_or_None, vec_or_None, is_oracle)
    candidates: list[dict[str, Any]] = []
    if args.include_garrido_oracle:
        candidates.append({"name": "garrido_oracle", "kind": "oracle"})
    if args.include_l1a:
        for name, asm, ipm, act in l1a_candidates():
            candidates.append(
                {
                    "name": name,
                    "kind": "static",
                    "asm": asm,
                    "ipm": ipm,
                    "action": act,
                    "space": "uniform",
                }
            )
    if args.l1b_screening_csv:
        for name, asm, ipm, act in l1b_candidates_from_screening(
            Path(args.l1b_screening_csv),
            args.l1b_top_k,
            selection=args.l1b_selection,
        ):
            candidates.append(
                {
                    "name": name,
                    "kind": "static",
                    "asm": asm,
                    "ipm": ipm,
                    "action": act,
                    "space": "per_node",
                }
            )
    ppo_models = {}
    if args.ppo_run_root:
        for name, asm, ipm, zp, vp, learn_initial_decision in discover_ppo_models(
            Path(args.ppo_run_root)
        ):
            model, vec = load_ppo(zp, vp)
            ppo_models[name] = (model, vec)
            candidates.append(
                {
                    "name": name,
                    "kind": "ppo",
                    "asm": asm,
                    "ipm": ipm,
                    "model": model,
                    "vec": vec,
                    "space": "uniform" if "E2a" in name else "per_node",
                    "learn_initial_decision": learn_initial_decision,
                }
            )
    print(
        f"candidates: {len(candidates)}  ({sum(c['kind']=='static' for c in candidates)} static, "
        f"{sum(c['kind']=='ppo' for c in candidates)} ppo, oracle={args.include_garrido_oracle})"
    )

    per_scenario_rows: list[dict[str, Any]] = []
    ps_path = out_dir / "unified_per_scenario.csv"
    row_fieldnames = [
        "policy",
        "kind",
        "space",
        "cfi",
        "source_cfi",
        "family",
        "replication",
        "eval_seed",
        "reward_total",
        "fill_rate_order_level",
        "order_level_ret_mean",
        "backorder_rate_order_level",
        "pct_steps_S1",
        "pct_steps_S2",
        "pct_steps_S3",
        "steps",
    ]
    total = len(candidates) * len(panel) * args.replications
    done = 0
    with open(ps_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row_fieldnames)
        writer.writeheader()
        stream.flush()
        for cand in candidates:
            for cfi in panel:
                spec = specs[cfi]
                override = {
                    "enabled_risks": set(spec.enabled_risks),
                    "risk_overrides": dict(spec.risk_overrides),
                }
                if cand["kind"] == "oracle":
                    asm, ipm = "thesis_factorized", "thesis_strict"
                    initial_action = thesis_design_action(spec)
                    action_fn = fixed_action_fn(initial_action)
                    space = "garrido"
                else:
                    asm, ipm = cand["asm"], cand["ipm"]
                    space = cand["space"]
                    if cand["kind"] == "ppo":
                        initial_action = None
                        action_fn = model_action_fn(cand["model"], cand["vec"])
                    else:
                        initial_action = cand["action"]
                        action_fn = fixed_action_fn(cand["action"])
                ek = base_kwargs(args)
                ek.update({"action_space_mode": asm, "inventory_period_mode": ipm})
                if cand["kind"] in {"oracle", "static"}:
                    ek["initial_action"] = initial_action
                elif cand["kind"] == "ppo":
                    ek["learn_initial_decision"] = bool(
                        cand.get("learn_initial_decision", False)
                    )
                ek.update(override)
                for rep in range(args.replications):
                    eval_seed = (
                        args.base_seed + cfi * 1000 + rep
                    )  # depends ONLY on (cfi, rep)
                    m = rollout(
                        args=args,
                        env_kwargs=ek,
                        action_fn=action_fn,
                        eval_seed=eval_seed,
                    )
                    row = {
                        "policy": cand["name"],
                        "kind": cand["kind"],
                        "space": space,
                        "cfi": cfi,
                        "source_cfi": spec.source_cfi,
                        "family": spec.family,
                        "replication": rep,
                        "eval_seed": eval_seed,
                        **m,
                    }
                    per_scenario_rows.append(row)
                    writer.writerow({key: row[key] for key in row_fieldnames})
                    done += 1
                if done % 200 < args.replications:
                    stream.flush()
                    print(
                        f"  progress {done}/{total} ({100*done/total:.0f}%)", flush=True
                    )

    # aggregate per policy (panel mean)
    import pandas as pd

    df = pd.DataFrame(per_scenario_rows)
    agg = (
        df.groupby(["policy", "kind", "space"])
        .agg(
            fill_mean=("fill_rate_order_level", "mean"),
            fill_std=("fill_rate_order_level", "std"),
            ret_mean=("order_level_ret_mean", "mean"),
            reward_mean=("reward_total", "mean"),
            n=("fill_rate_order_level", "size"),
        )
        .reset_index()
        .sort_values(["fill_mean", "ret_mean", "reward_mean"], ascending=False)
    )
    agg.to_csv(out_dir / "unified_summary.csv", index=False)

    # best-in-space + paper table
    def best_in(space: str) -> dict | None:
        sub = agg[(agg.space == space) & (agg.kind == "static")]
        return sub.iloc[0].to_dict() if len(sub) else None

    best_uniform = best_in("uniform")
    best_per_node = best_in("per_node")
    oracle = agg[agg.kind == "oracle"]
    oracle_row = oracle.iloc[0].to_dict() if len(oracle) else None

    lines = [
        "# Unified thesis-decision evaluation",
        "",
        f"Panel: Cf{panel[0]}-Cf{panel[-1]} ({len(panel)} scenarios), {args.replications} reps each, "
        f"max_steps={args.max_steps} (~{args.max_steps*args.step_size_hours/8760:.1f} thesis-years). "
        "Every policy scored on the SAME scenarios with the SAME per-(scenario,rep) seeds.",
        "",
        "## Panel-mean by policy (sorted by fill)",
        "",
        "| policy | space | fill | ReT | reward | n |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r.policy} | {r.space} | {r.fill_mean:.4f} | {r.ret_mean:.4f} | {r.reward_mean:.2f} | {int(r.n)} |"
        )
    lines += ["", "## Headline comparison (best-in-space, same panel)", ""]
    if oracle_row:
        lines.append(
            f"- **Garrido oracle** (per-scenario matched): fill={oracle_row['fill_mean']:.4f}, ReT={oracle_row['ret_mean']:.4f}"
        )
    if best_uniform:
        lines.append(
            f"- **Best static uniform (L1a)**: {best_uniform['policy']} fill={best_uniform['fill_mean']:.4f}, ReT={best_uniform['ret_mean']:.4f}"
        )
    if best_per_node:
        lines.append(
            f"- **Best static per-node (L1b)**: {best_per_node['policy']} fill={best_per_node['fill_mean']:.4f}, ReT={best_per_node['ret_mean']:.4f}"
        )
        if best_uniform:
            d = best_per_node["fill_mean"] - best_uniform["fill_mean"]
            lines.append(
                f"  - per-node minus uniform (fill): {d:+.4f}  -> per-node {'ADDS value' if d > 0 else 'does NOT add value'} on this panel"
            )
    for sp, lbl in [("uniform", "E2a [6,3]"), ("per_node", "E2b [6,6,6,3]")]:
        ppo = agg[(agg.space == sp) & (agg.kind == "ppo")]
        best_static = best_uniform if sp == "uniform" else best_per_node
        if len(ppo) and best_static:
            best_ppo = ppo.iloc[0].to_dict()
            mean_fill = ppo["fill_mean"].mean()
            best_delta = best_ppo["fill_mean"] - best_static["fill_mean"]
            mean_delta = mean_fill - best_static["fill_mean"]
            lines.append(
                f"- **Best PPO {lbl}**: {best_ppo['policy']} fill={best_ppo['fill_mean']:.4f}, "
                f"ReT={best_ppo['ret_mean']:.4f} vs best static same space={best_static['fill_mean']:.4f} "
                f"-> best-seed adaptivity delta {best_delta:+.4f}"
            )
            lines.append(
                f"  - PPO {lbl} seed-mean fill={mean_fill:.4f} "
                f"-> mean adaptivity delta {mean_delta:+.4f}"
            )
    (out_dir / "unified_table.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote: {ps_path}")
    print(f"wrote: {out_dir/'unified_summary.csv'}")
    print(f"wrote: {out_dir/'unified_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
