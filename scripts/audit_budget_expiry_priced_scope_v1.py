#!/usr/bin/env python3
"""Machine-readable scope audit of the priced strategic-buffer closure."""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_budget_expiry_priced_v2 as priced  # noqa: E402
import run_budget_expiry_boundary_v1 as boundary  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    result_path = ROOT / "results/budget_expiry_priced/result.json"
    producer_path = ROOT / "scripts/run_budget_expiry_priced_v2.py"
    play_path = ROOT / "scripts/run_budget_expiry_boundary_v1.py"
    result = json.loads(result_path.read_text())
    producer_source = producer_path.read_text()
    play_source = inspect.getsource(boundary.play)
    flat = [row for cell in result["cells"].values() for row in cell["per_lambda"].values()]
    raw_keys = {"L_matrix", "cost_matrix", "raw_rows", "raw_path", "raw_sha256"}
    constant_action = "while not (done or truncated)" in play_source and "env.step(action)" in play_source
    uses_all_data_scale = "norm = float(cost.max())" in producer_source
    return {
        "schema_version": "budget_expiry_priced_scope_audit_v1",
        "claim_status": "STATIC_BUFFER_POSTURE_CLASS_CLOSED__ORIGINAL_SEQUENTIAL_SCOPE_SUPERSEDED",
        "run_role": "STATIC_SOURCE_AND_ARTIFACT_AUDIT_NO_NEW_SIMULATION",
        "audited_result": {"path": str(result_path.relative_to(ROOT)), "sha256": digest(result_path),
                           "self_sha256": result.get("self_sha256")},
        "audited_producer": {"path": str(producer_path.relative_to(ROOT)), "sha256": digest(producer_path)},
        "imported_play": {"path": str(play_path.relative_to(ROOT)), "sha256": digest(play_path)},
        "facts": {
            "n_postures": len(priced.POSTURES),
            "posture_dimensions": 3,
            "horizon_steps": boundary.MAX_STEPS,
            "same_action_reused_at_every_step": constant_action,
            "enumerates_within_episode_schedules": False,
            "all_24_reported_distinct_static_optima_equal_one": all(
                int(row["distinct_optima_on_test"]) == 1 for row in flat),
            "raw_tape_by_posture_matrices_persisted": bool(raw_keys & set(result)),
            "cost_scale_uses_train_and_test_max": uses_all_data_scale,
        },
        "reading_rule": (
            "The numerical result closes the 27 time-invariant postures. It does not identify the "
            "optimum over schedules or feedback policies, and the endpoint scale uses test data."
        ),
        "supersedes_in_scope": {
            "path": "docs/CIERRE_FAMILIA_BUFFER_ESTRATEGICO_2026-08-08.md",
            "old": "STRATEGIC_BUFFER_FAMILY_CLOSED__NO_PRICED_SEQUENTIAL_HEADROOM",
            "new": "STATIC_BUFFER_POSTURE_CLASS_CLOSED__NO_TAPE_HETEROGENEITY_ON_27_CONSTANTS"
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/budget_expiry_priced/scope_audit_v1.json")
    args = parser.parse_args()
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload["facts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
