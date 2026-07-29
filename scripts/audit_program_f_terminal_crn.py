#!/usr/bin/env python3
"""Post-terminal audit of exogenous events actually consumed by Program F."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_program_f_phase_screen import action_id, state_weeks
from supply_chain.program_f import (
    ConstantPortfolio, CONTEXTS, actions_for_budget, branch_from_week,
    materialize_tape, profile_for_cell, run_policy,
)


def main() -> int:
    root = Path("results/program_f")
    design = json.loads((root / "screen/design.json").read_text())
    static_checks = 0
    branch_checks = 0
    mismatches: list[dict] = []
    for cell in design["cells"]:
        profile = profile_for_cell(cell)
        actions = actions_for_budget(profile["budget_tokens"])
        tapes = [
            materialize_tape(
                int(cell["screen_seed_start"]) + i, CONTEXTS[i % 3],
                "phase-screen", weeks=32, profile=profile,
            )
            for i in range(12)
        ]
        for tape_index, tape in enumerate(tapes):
            static_hashes = []
            for action in actions:
                result = run_policy(tape, ConstantPortfolio(action))
                static_hashes.append((
                    result["consumed_base_threat_sha256"],
                    result["realized_demand_sha256"],
                ))
            static_checks += 1
            if len(set(static_hashes)) != 1:
                mismatches.append({
                    "scope": "static", "cell_id": cell["cell_id"],
                    "tape_id": tape["tape_id"], "hashes": static_hashes,
                })
            for state_index, week in enumerate(state_weeks(tape)):
                prefix = actions[(tape_index * 4 + state_index) % len(actions)]
                branch_hashes = []
                for action in actions:
                    result = branch_from_week(
                        tape, prefix_action=prefix, state_week=week,
                        branch_action=action, horizon_weeks=4,
                    )
                    branch_hashes.append((
                        result["consumed_base_threat_sha256"],
                        result["realized_demand_sha256"],
                    ))
                branch_checks += 1
                if len(set(branch_hashes)) != 1:
                    mismatches.append({
                        "scope": "branch", "cell_id": cell["cell_id"],
                        "tape_id": tape["tape_id"], "state_week": week,
                        "prefix": action_id(prefix), "hashes": branch_hashes,
                    })
        print(f"[program-f-terminal-crn] {cell['cell_id']} complete", flush=True)
    verdict = {
        "gate": "PROGRAM_F_POST_TERMINAL_RUNTIME_CRN_AUDIT",
        "cells": len(design["cells"]), "static_tape_checks": static_checks,
        "prefix_balanced_branch_checks": branch_checks,
        "hashes": ["consumed_base_threat_sha256", "realized_demand_sha256"],
        "demand_fields": ["j", "time_hours", "quantity", "contingent", "destination"],
        "threat_fields": ["event_id", "risk_id", "onset_hours", "base_duration_hours", "affected_ops", "magnitude", "context_at_onset"],
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "calibration_tapes_opened": 0, "holdout_tapes_opened": 0,
        "virgin_tapes_opened": 0, "ppo_trained": False,
        "interpretation": "PASS_PROGRAM_F_RUNTIME_CRN_AUDIT" if not mismatches else "INVALIDATE_PROGRAM_F_RUNTIME_CRN",
    }
    output = root / "terminal_audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "crn_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: verdict[key] for key in (
        "interpretation", "static_tape_checks", "prefix_balanced_branch_checks", "mismatch_count"
    )}, indent=2))
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
