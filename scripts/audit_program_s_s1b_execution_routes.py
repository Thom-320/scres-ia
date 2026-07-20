#!/usr/bin/env python3
"""Materialize S1b engine routes without opening any scientific seed."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_s_execution_router import route_risk_mask  # noqa: E402


def main() -> int:
    parent = json.loads(
        (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
    )
    certificate = json.loads(
        (ROOT / "results/program_s/s1_transducer_preflight_v1/result.json").read_text()
    )
    certified = {
        mask: bool(row.get("eligible")) for mask, row in certificate["masks"].items()
    }
    rows = [
        asdict(
            route_risk_mask(
                mask=mask,
                risks=risks,
                r14_probability_multiplier=1.0,
                certified_masks=certified,
                r14_action_dependence_certificate=False,
            )
        )
        for mask, risks in parent["physical_masks"].items()
    ]
    payload = {
        "schema_version": "program_s_s1b_execution_routes_v1",
        "status": "ROUTES_FROZEN_NO_SCIENTIFIC_SEEDS_AUTHORIZED",
        "scientific_seeds_opened": [],
        "routes": rows,
    }
    output = ROOT / "results/program_s/s1b_execution_route_audit_v1/result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

