#!/usr/bin/env python3
"""Audit the latest G3a-v2 result against its frozen falsifier contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/g3a_boundary_v2/result.full34.json"
PREDECESSOR = ROOT / "results/g3a_boundary_v2/result.json"
RUNNER = ROOT / "scripts/run_g3a_boundary_v2.py"
CONTRACT = ROOT / "docs/PREREGISTRO_G3A_V2_RECONSTRUCCION_2026-08-08.md"
DEFAULT_OUTPUT = ROOT / "results/g3a_boundary_v2/contract_audit_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict[str, object]:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    runner = RUNNER.read_text(encoding="utf-8")
    contract = CONTRACT.read_text(encoding="utf-8")
    result_falsifiers = result["falsifiers"]
    contract_ids = [f"f{i}" for i in range(1, 10)]
    missing = [fid for fid in contract_ids if not any(key.startswith(fid + "_")
                                                        for key in result_falsifiers)]
    same_seeds = result["seeds"] == predecessor["seeds"]
    rows = [cell["rows"] for cell in result["cells"].values()]
    held_complete = all(
        len(controller["held"]) == 30
        for cell_rows in rows for controller in cell_rows.values()
    )
    selection_rows_persisted = all(
        "select" in controller
        for cell_rows in rows for controller in cell_rows.values()
    )
    f9_source = next(line.strip() for line in runner.splitlines()
                     if 'float(best_hq["adaptive_forfeited"])' in line)
    payload = {
        "schema_version": "g3a_boundary_v2_contract_audit_v1",
        "audit_status": "HEADROOM_NEGATIVE_STANDS__CONTRACT_COMPLIANCE_INCOMPLETE",
        "scientific_reading": (
            "The full-34 development result is direct negative evidence for observable headroom "
            "in the current G3a-v2 implementation. It is not an independent replication of the "
            "14-controller run and it does not satisfy every preregistered integrity check."
        ),
        "facts": {
            "claim_status": result["claim_status"],
            "n_controllers": result["n_controllers"],
            "persistent_uniform_hard_quota_h_obs": result["cells"]
                ["persistent_uniform_hard_quota"]["h_obs"],
            "contract_declares_f1_through_f9": all(fid in contract for fid in contract_ids),
            "missing_preregistered_falsifier_ids": missing,
            "mass_falsifier_absent": "f3" in missing,
            "common_belief_model_falsifier_absent": "f5" in missing,
            "f9_uses_nonnegative_threshold": "F.ge(" in runner and ", 0.0," in runner,
            "f9_source_operand": f9_source,
            "held_evaluation_arrays_complete": held_complete,
            "per_seed_selection_rows_persisted": selection_rows_persisted,
            "full34_reuses_predecessor_seed_block": same_seeds,
            "full34_seed_count": len(result["seeds"]),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (RESULT, PREDECESSOR, RUNNER, CONTRACT)
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["facts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
