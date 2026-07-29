#!/usr/bin/env python3
"""Reprice a frozen Cobb-Douglas comparison set without replaying the DES.

This is a sensitivity analysis, not an economic calibration. It uses the seven
unpriced component means persisted by the corrected metric-panel fold, applies
the frozen relative-price grid, and recomputes set-relative kappa_dot and R_CD.
It cannot select a policy or replace the primary ReT/resource panel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    kappa_from_components,
    score_comparison_set,
    validate_costs,
)


def canonical_sha(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scenario_costs(contract: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = validate_costs(contract["baseline"]["costs"])
    out = {"garrido_unit_baseline": baseline}
    for name, spec in contract["one_factor_scenarios"].items():
        costs = dict(baseline)
        key = str(spec["coefficient"])
        costs[key] *= float(spec["multiplier"])
        out[name] = validate_costs(costs)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "results/metric_panel/panel_with_v2_arms_rpj_corrected_v2.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("contracts/cobb_douglas_economic_sensitivity_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/cobb_douglas/economic_sensitivity_v1/result.json"),
    )
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    contract = json.loads(args.contract.read_text())
    costs_by_scenario = scenario_costs(contract)
    exponents = panel["exponents"]

    families: dict[str, Any] = {}
    for family, family_result in panel["results"].items():
        cells = family_result["per_cell"]
        scenario_results: dict[str, Any] = {}
        for scenario, costs in costs_by_scenario.items():
            repriced = {
                name: {
                    **cell,
                    "kappa": kappa_from_components(cell, costs),
                }
                for name, cell in cells.items()
            }
            scores = score_comparison_set(repriced, exponents)
            ranking = sorted(
                scores,
                key=lambda name: (-scores[name]["R_cobb_douglas"], name),
            )
            scenario_results[scenario] = {
                "costs": costs,
                "winner": ranking[0],
                "ranking": ranking,
                "scores": {
                    name: {
                        "R_cobb_douglas": scores[name]["R_cobb_douglas"],
                        "kappa_raw": scores[name]["kappa_raw"],
                        "kappa_dot": scores[name]["component_kappa_dot"],
                    }
                    for name in ranking
                },
            }
        winners = {
            name: result["winner"] for name, result in scenario_results.items()
        }
        families[family] = {
            "scenarios": scenario_results,
            "winner_by_scenario": winners,
            "winner_stable_across_grid": len(set(winners.values())) == 1,
        }

    payload = {
        "schema_version": contract["schema_version"],
        "claim_status": contract["claim_status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(args.panel),
        "panel_self_sha256": panel["self_sha256"],
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract.read_bytes()).hexdigest(),
        "primary_endpoint_authorized": False,
        "policy_selection_authorized": False,
        "domain_calibration_status": contract["domain_calibration_status"],
        "strategic_injection_priced_in_kappa": False,
        "families": families,
        "interpretation_boundary": contract["interpretation_boundary"],
    }
    payload["self_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
