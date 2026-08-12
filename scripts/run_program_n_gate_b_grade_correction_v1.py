#!/usr/bin/env python3
"""Re-adjudicate the GRADE and the FALSIFIER SUMMARY of every Gate B artifact.

Retraction: `docs/RETRACTACION_CONTENTION_V1_Y_ENMIENDA_X_2026-08-12.md`

Two instrument defects, both mine, both found by an external review on 2026-08-12:

1. `run_role` and `scope` were frozen strings in `run_program_n_gate_b_v1.py`. `--seed-base` was
   added later and never touched them, so `gate_b_confirmation_v3` opened a fresh block and still
   sealed `run_role: DEVELOPMENT`, `scope: DEVELOPMENT_REANALYSIS_NO_NEW_SEEDS`. Every document
   that called it a confirmation was contradicted by the artifact itself.

2. `F.summarise` filtered on `computed is True` before scoring, and `custody_falsifier` carries no
   such key, so a RED custody check was invisible. `gate_b_cd_surface` seals `all_passed: true`
   beside `custody.passed: false`.

Nothing is edited in place and no seed is opened: the sealed artifacts are read, the corrected
grade and summary are computed from their own fields, and this artifact records both side by side.

THE CEILING ON WHAT MAY BE CLAIMED. The seed registry declares itself incomplete, and its own rule
says a missing result file is not virginity evidence. So the strongest honest grade a fresh block
can earn here is NO KNOWN COLLISION -- never "virgin".
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.arm_runner import seal_and_write                              # noqa: E402

CONTRACT = Path("docs/RETRACTACION_CONTENTION_V1_Y_ENMIENDA_X_2026-08-12.md")
OUT = Path("results/program_n/gate_b_grade_correction/result.json")

#: (artifact, endpoint, seed_base actually passed on the command line)
RUNS = [
    ("gate_b_cd_surface", "cobb_douglas", None),
    ("gate_b_confirmation", "cobb_douglas", 9400001),
    ("gate_b_confirmation_v2", "cobb_douglas", 9500001),
    ("gate_b_confirmation_v3", "cobb_douglas", 9600001),
    ("gate_b_sensitivity_ret_excel", "ret_excel", 9600001),
]


def derive_grade(endpoint: str, seed_base: int | None, custody_passed) -> tuple[str, str]:
    """The same rule now living in the runner, applied to what each artifact actually did."""
    opened = seed_base is not None and endpoint == "cobb_douglas"
    if endpoint == "ret_excel":
        return ("SENSITIVITY_REPLAY",
                "SENSITIVITY_REPLAY_SAME_TAPES_DIFFERENT_TARGET_NOT_A_CONFIRMATION")
    if opened and custody_passed is True:
        return ("PROSPECTIVE",
                "PROSPECTIVE_FRESH_BLOCK_NO_KNOWN_COLLISION_VIRGINITY_NOT_PROVEN")
    if opened:
        return ("PROSPECTIVE_WITH_CUSTODY_CONFLICT",
                "PROSPECTIVE_BLOCK_WITH_A_RECORDED_CUSTODY_CONFLICT")
    return "DEVELOPMENT", "DEVELOPMENT_REANALYSIS_NO_NEW_SEEDS"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/program_n"))
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    rows, grade_changed, summary_changed = {}, [], []
    for name, endpoint, seed_base in RUNS:
        path = args.root / name / "result.json"
        d = json.loads(path.read_text())
        custody = d["falsifiers"].get("custody", {})
        role, scope = derive_grade(endpoint, seed_base, custody.get("passed"))
        repaired = F.summarise(d["falsifiers"])
        sealed_summary = d.get("falsifier_summary", {})
        rows[name] = {
            "artifact": str(path), "endpoint_arg": endpoint, "seed_base_arg": seed_base,
            "sealed_run_role": d.get("run_role"), "sealed_scope": d.get("scope"),
            "corrected_run_role": role, "corrected_scope": scope,
            "custody_passed": custody.get("passed"),
            "custody_not_applicable": custody.get("not_applicable"),
            "sealed_all_passed": sealed_summary.get("all_passed"),
            "repaired_all_passed": repaired["all_passed"],
            "repaired_failed": repaired["failed"],
            "claim_status_unchanged": d.get("claim_status"),
        }
        if role != d.get("run_role"):
            grade_changed.append(name)
        if repaired["all_passed"] != sealed_summary.get("all_passed"):
            summary_changed.append(name)

    checks = {
        "g1_the_grade_rule_is_applied_to_every_gate_b_run": F.check(
            len(rows) == len(RUNS),
            "a Gate B run left out of this correction keeps a grade nobody re-derived, which is "
            "exactly how the defect survived three weeks",
            computed_from={"n_runs": len(RUNS), "n_corrected": len(rows)}),
        "g2_the_correction_changes_something": F.check(
            bool(grade_changed or summary_changed),
            "if no grade and no summary moves, the two defects were not defects and this artifact "
            "measures nothing",
            computed_from={"n_grade_changed": len(grade_changed),
                           "n_summary_changed": len(summary_changed)},
            grade_changed=grade_changed, summary_changed=summary_changed),
        "g3_no_run_is_promoted_to_virgin": F.check(
            all("VIRGINITY_NOT_PROVEN" in r["corrected_scope"]
                or "PROSPECTIVE" not in r["corrected_run_role"] for r in rows.values()),
            "the registry declares itself incomplete, so any scope asserting virginity would be "
            "claiming more than custody can support -- the failure this whole correction repairs",
            computed_from={"n_runs": len(rows), "n_prospective":
                           sum("PROSPECTIVE" == r["corrected_run_role"] for r in rows.values())}),
        "g4_no_claim_status_is_touched": F.check(
            all(r["claim_status_unchanged"] is not None for r in rows.values()),
            "this artifact corrects GRADE and SUMMARY only. Changing a claim_status here would be "
            "re-adjudicating science from a bookkeeping run",
            computed_from={"n_runs": len(rows)}),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: no seed is opened and no episode is run. Every "
                                        "field is read from artifacts already sealed.",
                     "seeds_opened": 0, "episodes_run": 0,
                     "artifacts_read": [r["artifact"] for r in rows.values()]}}
    summary = F.summarise(checks)

    status = ("GRADE_AND_SUMMARY_CORRECTED" if (grade_changed or summary_changed)
              else "NOTHING_TO_CORRECT")

    payload = {
        "schema_version": "program_n_gate_b_grade_correction_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "BOOKKEEPING_CORRECTION",
        "scope": "CORRECTS_GRADE_AND_FALSIFIER_SUMMARY_ONLY_NO_SEEDS_NO_EPISODES_NO_SCIENCE",
        "runs": rows, "grade_changed": grade_changed, "summary_changed": summary_changed,
        "falsifiers": checks, "falsifier_summary": summary,
        "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=args.root / "gate_b_confirmation_v3/result.json")

    print(f"\nveredicto: {status}\n")
    for name, r in rows.items():
        mark = "  <-- CAMBIA" if name in grade_changed or name in summary_changed else ""
        print(f"  {name}{mark}")
        print(f"    grado   {r['sealed_run_role']} -> {r['corrected_run_role']}")
        print(f"    resumen all_passed {r['sealed_all_passed']} -> {r['repaired_all_passed']}"
              f"  {r['repaired_failed'] or ''}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:48} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
