#!/usr/bin/env python3
"""Sensitivity sweep for the hand-chosen ``adaptive_benchmark_v2`` multipliers.

Unlike Garrido's own risk levels (``RISKS_CURRENT``/``RISKS_INCREASED``), which
are calibrated to match his thesis tables exactly (verified in
``docs/DES_MODEL_DOCUMENTATION.md``), the Track B ``adaptive_benchmark_v2``
regime-switching layer (Markov transitions + the downstream-risk uplift
multipliers ``ADAPTIVE_BENCHMARK_V2_RISK_MULTIPLIERS``/
``_RECOVERY_MULTIPLIERS``/``_SURGE_SCALE_MULTIPLIER``) has no thesis
equivalent -- these are hand-chosen engineering constants with no documented
calibration or sensitivity study behind them (confirmed via ``git log -S`` on
``supply_chain/config.py``: one commit, no rationale).

This script asks the direct robustness question a reviewer would ask: if
these specific numbers were meaningfully different (weaker or stronger
downstream stress), does PPO still beat the static/heuristic comparators
under the exact same training protocol? It monkey-patches the module-level
constants actually read by ``supply_chain/supply_chain.py`` at runtime (no
edits to any file on disk, so it cannot collide with concurrent activity on
``supply_chain/config.py``), then calls the existing
``scripts/run_track_b_smoke.py`` training+eval pipeline unmodified -- so the
same static grid (s1/s2/s3 x d1.00/1.50/2.00) and heuristics used everywhere
else in this project are evaluated fresh, under each perturbed setting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import supply_chain.supply_chain as sc_module  # noqa: E402

from scripts.run_track_b_smoke import build_parser as smoke_build_parser, run_smoke  # noqa: E402

SCENARIOS: dict[str, dict[str, Any]] = {
    "weak": {
        "risk_multipliers": {"R22": 1.15, "R23": 1.05, "R24": 1.10},
        "recovery_multipliers": {"R22": 1.05, "R23": 1.02},
        "surge_scale_multiplier": 1.05,
    },
    "current": {
        "risk_multipliers": {"R22": 1.35, "R23": 1.15, "R24": 1.25},
        "recovery_multipliers": {"R22": 1.20, "R23": 1.10},
        "surge_scale_multiplier": 1.20,
    },
    "strong": {
        "risk_multipliers": {"R22": 1.55, "R23": 1.35, "R24": 1.45},
        "recovery_multipliers": {"R22": 1.35, "R23": 1.25},
        "surge_scale_multiplier": 1.35,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = smoke_build_parser()
    parser.description = (
        "Track B PPO+MLP trained/evaluated under a perturbed adaptive_benchmark_v2 "
        "downstream-risk uplift scenario (sensitivity sweep, no file edits)."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    return parser


def apply_scenario(scenario: str) -> dict[str, Any]:
    cfg = SCENARIOS[scenario]
    original = {
        "risk_multipliers": dict(sc_module.ADAPTIVE_BENCHMARK_V2_RISK_MULTIPLIERS),
        "recovery_multipliers": dict(sc_module.ADAPTIVE_BENCHMARK_V2_RECOVERY_MULTIPLIERS),
        "surge_scale_multiplier": float(sc_module.ADAPTIVE_BENCHMARK_V2_SURGE_SCALE_MULTIPLIER),
    }
    sc_module.ADAPTIVE_BENCHMARK_V2_RISK_MULTIPLIERS = dict(cfg["risk_multipliers"])
    sc_module.ADAPTIVE_BENCHMARK_V2_RECOVERY_MULTIPLIERS = dict(cfg["recovery_multipliers"])
    sc_module.ADAPTIVE_BENCHMARK_V2_SURGE_SCALE_MULTIPLIER = float(cfg["surge_scale_multiplier"])
    return original


def main() -> None:
    args = build_parser().parse_args()
    applied = apply_scenario(args.scenario)
    args.invocation = "python scripts/run_track_b_adaptive_v2_sensitivity.py " + " ".join(sys.argv[1:])
    summary = run_smoke(args)
    summary["adaptive_benchmark_v2_scenario"] = {
        "name": args.scenario,
        "applied_values": SCENARIOS[args.scenario],
        "original_values": applied,
    }
    summary_path = Path(summary["artifacts"]["summary_json"])
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote sensitivity bundle ({args.scenario}) to {summary_path}")
    for row in summary["policy_summary"]:
        print(
            f"{row['policy']}: order_ret_excel={float(row['order_ret_excel_mean']):.6f}"
        )


if __name__ == "__main__":
    main()
