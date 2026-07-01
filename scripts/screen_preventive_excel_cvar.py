#!/usr/bin/env python3
"""Small sequential screen for the strongest preventive Excel+CVaR lane.

This intentionally stays tiny: the objective is to choose one candidate for a
held-out dense-CRN confirmation, not to tune until something wins.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _slug_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _best_static_at_resource(summary: dict, *, key: str, higher_better: bool) -> dict | None:
    dyn_res = float(summary["dynamic"]["resource"])
    statics = [s for s in summary["statics"] if float(s["resource"]) <= dyn_res + 1e-9]
    if not statics:
        return None
    return max(statics, key=lambda s: float(s[key])) if higher_better else min(statics, key=lambda s: float(s[key]))


def _summarize_case(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    dyn = summary["dynamic"]
    best_excel = _best_static_at_resource(summary, key="excel", higher_better=True)
    best_cvar = _best_static_at_resource(summary, key="cvar", higher_better=False)
    excel_margin = float(dyn["excel"]) - float(best_excel["excel"]) if best_excel else float("-inf")
    cvar_margin = float(best_cvar["cvar"]) - float(dyn["cvar"]) if best_cvar else float("-inf")
    return {
        "path": str(path),
        "args": summary["args"],
        "dynamic": dyn,
        "excel_pareto": summary["excel_pareto"],
        "cvar_pareto": summary["cvar_pareto"],
        "best_static_at_resource_excel": best_excel,
        "best_static_at_resource_cvar": best_cvar,
        "excel_margin_at_resource": excel_margin,
        "cvar_margin_at_resource": cvar_margin,
        "adaptive": float(dyn.get("frac_std", 0.0)) > 0.02,
    }


def _rank_key(case: dict) -> tuple:
    """Higher is better; keep Excel primary, CVaR/resource/adaptivity secondary."""
    dyn = case["dynamic"]
    return (
        not case["excel_pareto"]["dominated_by_static"],
        not case["cvar_pareto"]["dominated_by_static"],
        case["excel_pareto"]["pareto_win"],
        case["excel_margin_at_resource"],
        case["cvar_margin_at_resource"],
        case["adaptive"],
        -float(dyn["resource"]),
    )


def _run_case(args: argparse.Namespace, alpha: float, holding_cost: float, root: Path) -> dict:
    case_dir = root / f"alpha{_slug_float(alpha)}_hc{_slug_float(holding_cost)}"
    cmd = [
        sys.executable,
        "scripts/run_preventive_pareto.py",
        "--reward-mode", "ReT_excel_plus_cvar",
        "--cvar-alpha", str(alpha),
        "--phi", str(args.phi),
        "--psi", str(args.psi),
        "--seeds", args.seeds,
        "--timesteps", str(args.timesteps),
        "--n-envs", str(args.n_envs),
        "--eval-episodes", str(args.eval_episodes),
        "--max-steps", str(args.max_steps),
        "--n-fracs", str(args.n_fracs),
        "--crn-eval",
        "--eval-seed0", str(args.eval_seed0),
        "--holding-cost", str(holding_cost),
        "--shift-cost", str(args.shift_cost),
        "--output", str(case_dir),
    ]
    if args.observation_version:
        cmd.extend(["--observation-version", args.observation_version])
    (case_dir).mkdir(parents=True, exist_ok=True)
    (case_dir / "command.json").write_text(json.dumps({"cmd": cmd}, indent=2))
    print("\nRUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return _summarize_case(case_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.1,0.2,0.4")
    ap.add_argument("--holding-costs", default="0.0")
    ap.add_argument("--extra-holding-cost", type=float, default=0.0003,
                    help="tested for the best alpha if the first pass is CVaR-dominated")
    ap.add_argument("--skip-extra-holding-cost", action="store_true")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--eval-seed0", type=int, default=7000)
    ap.add_argument("--crn-eval", action="store_true",
                    help="accepted for CLI symmetry; this screen always uses CRN evaluation")
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--n-fracs", type=int, default=21)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    root = Path(args.output or f"outputs/experiments/preventive_excel_cvar_minisweep_{stamp}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "screen_args.json").write_text(json.dumps(vars(args), indent=2))

    cases: list[dict] = []
    for holding_cost in _parse_floats(args.holding_costs):
        for alpha in _parse_floats(args.alphas):
            cases.append(_run_case(args, alpha, holding_cost, root))

    best = max(cases, key=_rank_key)
    if (
        not args.skip_extra_holding_cost
        and best["cvar_pareto"]["dominated_by_static"]
        and args.extra_holding_cost not in _parse_floats(args.holding_costs)
    ):
        alpha = float(best["args"]["cvar_alpha"])
        print(f"\nCVaR still dominated; trying extra holding_cost={args.extra_holding_cost} for alpha={alpha}", flush=True)
        cases.append(_run_case(args, alpha, args.extra_holding_cost, root))
        best = max(cases, key=_rank_key)

    decision = {
        "winner": best,
        "cases": sorted(cases, key=_rank_key, reverse=True),
        "promotion_rule": [
            "prefer not dominated by dense static on Excel and CVaR",
            "Excel ReT is primary",
            "CVaR margin is the tie-breaker",
            "adaptive frac_std > 0.02 is required for mechanism credibility",
            "confirmatory must use held-out eval_seed0=9000 and 5 seeds/60k",
        ],
        "recommended_confirmatory_command": [
            "scripts/run_preventive_pareto.py",
            "--reward-mode", "ReT_excel_plus_cvar",
            "--cvar-alpha", str(best["args"]["cvar_alpha"]),
            "--phi", str(args.phi),
            "--psi", str(args.psi),
            "--seeds", "1,2,3,4,5",
            "--timesteps", "60000",
            "--n-envs", str(args.n_envs),
            "--eval-episodes", "8",
            "--max-steps", str(args.max_steps),
            "--n-fracs", str(args.n_fracs),
            "--crn-eval",
            "--eval-seed0", "9000",
            "--holding-cost", str(best["args"]["holding_cost"]),
            "--shift-cost", str(args.shift_cost),
        ],
    }
    (root / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    print("\n=== MINI-SWEEP DECISION ===")
    print(json.dumps({
        "winner_alpha": best["args"]["cvar_alpha"],
        "winner_holding_cost": best["args"]["holding_cost"],
        "dynamic": best["dynamic"],
        "excel_margin_at_resource": best["excel_margin_at_resource"],
        "cvar_margin_at_resource": best["cvar_margin_at_resource"],
        "excel_pareto": best["excel_pareto"],
        "cvar_pareto": best["cvar_pareto"],
        "decision_path": str(root / "decision.json"),
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
