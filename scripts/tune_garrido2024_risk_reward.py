#!/usr/bin/env python3
"""Small risk/reward tuning harness for the Garrido-2024 direct RL lane.

The objective is explicitly paper-facing: rank configurations by learned-policy
``mean_ret_excel_formula`` against the best static/heuristic baseline, while also
reporting shift mix and Garrido-2024 reward totals.  This script is for screening;
promising cells still need a longer, frozen confirmatory run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_garrido2024_direct_rl import (  # noqa: E402
    build_parser as build_direct_parser,
    run_direct,
)


DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks/garrido2024_risk_reward_tuning")


def _float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--train-timesteps", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--skip-heuristics", action="store_true", default=True)
    parser.add_argument(
        "--risk-frequency-multipliers",
        nargs="+",
        default=["1.0", "1.5"],
        help="Comma/space separated frequency multipliers.",
    )
    parser.add_argument(
        "--risk-impact-multipliers",
        nargs="+",
        default=["1.0", "1.25"],
        help="Comma/space separated impact multipliers.",
    )
    parser.add_argument(
        "--ret-g24-kappa-train-fracs",
        nargs="+",
        default=["0.2", "0.4"],
        help="Comma/space separated fractions of n_kappa used for training.",
    )
    parser.add_argument(
        "--ret-g24-shift-costs",
        nargs="+",
        default=["0.5", "1.0"],
        help="Comma/space separated shift-hour cost coefficients.",
    )
    return parser


def _direct_args_for(args: argparse.Namespace, *, freq: float, impact: float, kappa_frac: float, shift_cost: float, cell_dir: Path) -> argparse.Namespace:
    argv = [
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--step-size-hours",
        "168",
        "--max-steps",
        str(args.max_steps),
        "--risk-level",
        str(args.risk_level),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--risk-occurrence-mode",
        "thesis_window",
        "--risk-frequency-multiplier",
        str(freq),
        "--risk-impact-multiplier",
        str(impact),
        "--ret-g24-kappa-train-frac",
        str(kappa_frac),
        "--ret-g24-shift-cost",
        str(shift_cost),
        "--output-dir",
        str(cell_dir),
    ]
    if args.stochastic_pt:
        argv.append("--stochastic-pt")
    if args.skip_heuristics:
        argv.append("--skip-heuristics")
    return build_direct_parser().parse_args(argv)


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    freqs = _float_list(args.risk_frequency_multipliers)
    impacts = _float_list(args.risk_impact_multipliers)
    kappa_fracs = _float_list(args.ret_g24_kappa_train_fracs)
    shift_costs = _float_list(args.ret_g24_shift_costs)

    for freq in freqs:
        for impact in impacts:
            for kappa_frac in kappa_fracs:
                for shift_cost in shift_costs:
                    cell_name = (
                        f"freq{freq:g}_impact{impact:g}_"
                        f"kappa{kappa_frac:g}_shift{shift_cost:g}"
                    ).replace(".", "p")
                    cell_dir = args.output_dir / cell_name
                    direct_args = _direct_args_for(
                        args,
                        freq=freq,
                        impact=impact,
                        kappa_frac=kappa_frac,
                        shift_cost=shift_cost,
                        cell_dir=cell_dir,
                    )
                    summary = run_direct(direct_args)
                    comparison = summary["comparison_table"][0]
                    learned = next(
                        row
                        for row in summary["policy_summary"]
                        if row["phase"] == "ppo_eval" and row["policy"] == "ppo"
                    )
                    rows.append(
                        {
                            "cell": cell_name,
                            "risk_frequency_multiplier": freq,
                            "risk_impact_multiplier": impact,
                            "ret_g24_kappa_train_frac": kappa_frac,
                            "ret_g24_shift_cost": shift_cost,
                            "learned_mean_ret_excel_formula": comparison[
                                "learned_mean_ret_excel_formula"
                            ],
                            "best_baseline_policy": comparison["best_baseline_policy"],
                            "best_baseline_mean_ret_excel_formula": comparison[
                                "best_baseline_mean_ret_excel_formula"
                            ],
                            "delta_vs_best_baseline": comparison[
                                "delta_vs_best_baseline"
                            ],
                            "delta_vs_garrido_cf_s2": comparison[
                                "delta_vs_garrido_cf_s2"
                            ],
                            "learned_beats_best_baseline": comparison[
                                "learned_beats_best_baseline"
                            ],
                            "learned_beats_garrido_cf_s2": comparison[
                                "learned_beats_garrido_cf_s2"
                            ],
                            "ppo_pct_steps_S1_mean": learned["pct_steps_S1_mean"],
                            "ppo_pct_steps_S2_mean": learned["pct_steps_S2_mean"],
                            "ppo_pct_steps_S3_mean": learned["pct_steps_S3_mean"],
                            "cell_summary_json": str(cell_dir / "summary.json"),
                        }
                    )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["delta_vs_best_baseline"]),
            float(row["learned_mean_ret_excel_formula"]),
        ),
        reverse=True,
    )
    output_csv = args.output_dir / "tuning_summary.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(ranked[0].keys()) if ranked else [])
        if ranked:
            writer.writeheader()
            writer.writerows(ranked)

    payload = {
        "description": "Screening grid; confirmatory runs must freeze the selected cell first.",
        "primary_metric": "mean_ret_excel_formula",
        "n_cells": len(ranked),
        "best_cell": ranked[0] if ranked else None,
        "rows": ranked,
        "artifacts": {"tuning_summary_csv": str(output_csv)},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    args = build_parser().parse_args()
    summary = run_grid(args)
    print(f"Wrote {args.output_dir / 'summary.json'}")
    if summary["best_cell"] is not None:
        best = summary["best_cell"]
        print(
            f"Best cell {best['cell']}: delta_vs_best="
            f"{float(best['delta_vs_best_baseline']):.6f}, "
            f"learned_ret={float(best['learned_mean_ret_excel_formula']):.6f}"
        )


if __name__ == "__main__":
    main()
