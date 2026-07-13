#!/usr/bin/env python3
"""Machine-check every Op3--Op13 decision-right catalog entry.

This is a routing/liveness audit, not a numerical positive screen.  It prevents
an adapter flag or configuration knob from being counted as a new executable
adaptive mechanism and maps each live entry to the already-audited family that
contains it.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "contracts" / "decision_right_catalog_v1.json"
OUTPUT = ROOT / "research" / "paper2_exhaustive_search" / "decision_right_catalog_coverage.json"


ROUTES = {
    "op3_inventory_target": ("closed_reduction", "A/D/K2/K3 inventory and replenishment", None, ["outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/summary.json"]),
    "op3_order_quantity": ("closed_reduction", "D/K2/K3 replenishment", None, ["results/k2/strong_comparators.json", "results/k3/open_loop_confound_audit.json"]),
    "op3_review_period": ("closed_reduction", "D/K2/K3 replenishment", None, ["results/k2/strong_comparators.json", "results/k3/open_loop_confound_audit.json"]),
    "op4_dispatch_quantity": ("action_absent_requires_adapter", "DRA2 finite transport", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op4_dispatch_period": ("action_absent_requires_adapter", "DRA2 finite transport", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op4_transport_capacity": ("action_absent_requires_adapter", "DRA2/reservation transport", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op5_inventory_target": ("closed_reduction", "A/D buffer control", None, ["outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/summary.json"]),
    "op5_capacity_posture": ("closed_reduction_resource_frontier", "A/B/C capacity posture", None, ["outputs/experiments/track_b_same_contract_challenge_2026-07-10/summary.json"]),
    "op5_maintenance_effort": ("action_absent_requires_adapter", "Program J maintenance", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op5_wip_limit": ("action_absent_requires_adapter", "buffer/WIP allocation", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op6_capacity_posture": ("closed_reduction_resource_frontier", "A/B/C shared shifts", None, ["outputs/experiments/track_b_same_contract_challenge_2026-07-10/summary.json"]),
    "op6_inspection_effort": ("action_absent_requires_adapter", "inspection effort versus throughput", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op6_rework_rule": ("fidelity_configuration_not_adaptive_action", "R14 thesis-strict rework", 0.0, ["supply_chain/supply_chain.py"]),
    "op6_wip_limit": ("action_absent_requires_adapter", "buffer/WIP allocation", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op7_batch_quantity": ("closed_reduction_resource_frontier", "A batching/DRA2 finite convoy", None, ["supply_chain/supply_chain.py", "results/program_d/dra2_preflight/verdict.json"]),
    "op7_release_period": ("transition_dead_configuration_field", "Op7 release timing", 0.0, ["supply_chain/config.py", "supply_chain/supply_chain.py"]),
    "op7_quality_release_rule": ("action_absent_requires_adapter", "quality priority/product mix", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op8_dispatch_threshold": ("closed_reduction", "DRA2/2b finite convoy", None, ["results/program_d/dra2_preflight/verdict.json"]),
    "op8_max_wait": ("closed_reduction", "DRA2/2b finite convoy", None, ["results/program_d/dra2_preflight/verdict.json"]),
    "op8_convoy_capacity": ("closed_reduction_resource_frontier", "DRA2/2b finite convoy", None, ["results/program_d/dra2_preflight/verdict.json"]),
    "op9_inventory_target": ("closed_reduction", "A/D buffer control", None, ["results/program_i/branching/verdict.json"]),
    "op9_release_period": ("closed_reduction_resource_frontier", "Program I dispatch cadence", None, ["results/program_i/branching/verdict.json"]),
    "op9_queue_rule": ("closed_reduction", "D1 priority control", None, ["results/program_d/d1_v3_visible_branching/verdict.json"]),
    "op10_dispatch_quantity": ("closed_reduction_resource_frontier", "B/DRA2/G transport", None, ["results/program_i/branching/verdict.json"]),
    "op10_dispatch_period": ("closed_reduction_resource_frontier", "B/DRA2/G transport", None, ["results/program_i/branching/verdict.json"]),
    "op10_route_priority": ("action_absent_requires_adapter", "DRA1/G/H spatial routing", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op11_allocation_rule": ("action_absent_requires_adapter", "DRA1/G/H CSSU allocation", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op11_forward_reserve": ("researcher_extension_reduction", "F/G mitigation and spatial reserve", None, ["contracts/decision_right_catalog_v1.json", "results/program_f/screen/verdict.json"]),
    "op12_dispatch_quantity": ("closed_reduction_resource_frontier", "B/DRA2/G transport", None, ["results/program_i/branching/verdict.json"]),
    "op12_dispatch_period": ("closed_reduction_resource_frontier", "B/DRA2/G transport", None, ["results/program_i/branching/verdict.json"]),
    "op12_routing_rule": ("action_absent_requires_adapter", "DRA1/G/H spatial routing", 0.0, ["contracts/decision_right_catalog_v1.json"]),
    "op13_rationing_rule": ("closed_reduction", "D1 priority control", None, ["results/program_d/d1_v3_visible_branching/verdict.json"]),
}


def main() -> int:
    catalog = json.loads(CATALOG.read_text())
    factors = [row for row in catalog["factors"] if row["class"] == "decision_right"]
    ids = {row["id"] for row in factors}
    if ids != set(ROUTES):
        raise AssertionError({"missing": sorted(ids - set(ROUTES)), "extra": sorted(set(ROUTES) - ids)})

    rows = []
    for factor in factors:
        disposition, route, ceiling, evidence = ROUTES[factor["id"]]
        rows.append({
            "factor_id": factor["id"],
            "operation": factor["operation"],
            "catalog_status": factor["status"],
            "disposition": disposition,
            "family_route": route,
            "current_kernel_h_pi_ceiling": ceiling,
            "current_kernel_h_obs_ceiling": ceiling,
            "evidence": evidence,
        })
    exact_zero = [row["factor_id"] for row in rows if row["current_kernel_h_pi_ceiling"] == 0.0]
    result = {
        "schema_version": "paper2_decision_right_catalog_coverage_v1",
        "generated_date": "2026-07-13",
        "catalog": str(CATALOG.relative_to(ROOT)),
        "catalog_contract_id": catalog["contract_id"],
        "decision_right_count": len(rows),
        "all_decision_rights_covered_once": len(rows) == len(ids),
        "new_executable_source_native_candidate_count": 0,
        "exact_current_kernel_zero_count": len(exact_zero),
        "exact_current_kernel_zero_factor_ids": exact_zero,
        "interpretation": "A zero applies only to the current transition kernel. It does not close a Garrido-approved researcher extension that adds a live action, conserved resource and operational parameters.",
        "rows": rows,
        "paper2_confirmed": False,
        "paper3_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("decision_right_count", "new_executable_source_native_candidate_count", "exact_current_kernel_zero_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
