#!/usr/bin/env python3
"""Seal, from the outside, the bake-off artifacts whose own runner never sealed them.

WHY THIS EXISTS. scripts/run_architecture_bakeoff_v1.py does not import seal_and_write. Its
outputs carry no self_sha256, no contract, no calibration provenance -- and one of them,
results/architecture_bakeoff_200k/result.json, is what run_track_b_nonneural_v1.py reads to build
a field it calls `network_means_from_sealed_artifacts`. The name asserts something untrue, and the
number it feeds is the project's only neural positive.

WHAT THIS CANNOT DO. A retroactive seal fixes CONTENT, not PROVENANCE. This record says "this is
what the file contained on 2026-08-07"; it cannot say "this is what the run produced". Nobody can
seal backwards. The bake-off numbers stay development-grade with uncertified provenance and the
manuscript has to cite them that way.

The dated artifact is never edited: adding a digest to it would change its bytes and destroy the
one thing this record can contribute.

Contract: docs/ENMIENDA_SELLADO_RETROACTIVO_BAKEOFF_2026-08-07.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

RUNNER = Path("scripts/run_architecture_bakeoff_v1.py")
TARGETS = (Path("results/architecture_bakeoff_200k/result.json"),
           Path("results/architecture_bakeoff/result.json"))
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def _reject_constant(name: str):
    """json.loads accepts NaN/Infinity by default; a sealed record must not."""
    raise ValueError(f"non-finite JSON constant in the artifact: {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--targets", type=Path, nargs="+", default=list(TARGETS))
    args = ap.parse_args()

    runner_src = RUNNER.read_text()
    runner_seals = "seal_and_write" in runner_src
    rc = 0
    for target in args.targets:
        if not target.exists():
            print(f"  {target}: NO EXISTE, se omite")
            continue
        raw = target.read_bytes()
        digest_before = sha256(raw).hexdigest()
        payload_copy = json.loads(raw)

        # f1: the copy has to survive the EXACT serialization seal_and_write will apply. A
        # tautological version of this check (comparing the file to itself) would pass on an
        # artifact carrying NaN, Infinity, or non-string dict keys -- all of which json.dumps
        # emits happily and json.loads(strict) then rejects or silently retypes, so the record
        # would certify a digest for content it did not actually preserve.
        try:
            round_tripped = json.loads(
                json.dumps(payload_copy, indent=1, sort_keys=True, default=str),
                parse_constant=_reject_constant)
            faithful = round_tripped == payload_copy
        except ValueError as exc:
            round_tripped, faithful = None, False
            print(f"    f1: la serialización no sobrevive el viaje de ida y vuelta: {exc}")
        by_arch = payload_copy.get("by_arch") or {}
        consumer_ok = bool(by_arch) and all("mean" in v for v in by_arch.values())

        record = {
            "schema_version": "external_seal_v1",
            "claim_status": "CONTENT_SEALED_PROVENANCE_NOT_CERTIFIABLE",
            "scope": "CUSTODY_ONLY_NO_NUMBER_CHANGES_NO_ADJUDICATION",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "module_manifest": module_manifest(MODULES, script=__file__),
            "sealed_file": str(target),
            "sealed_file_sha256": digest_before,
            "sealed_file_bytes": len(raw),
            "producing_runner": str(RUNNER),
            "producing_runner_sha256": sha256(RUNNER.read_bytes()).hexdigest(),
            "what_this_certifies": ("the CONTENT of the named file as of this record's timestamp"),
            "what_this_does_not_certify": (
                "provenance. The runner did not seal at execution time, so nothing here attests "
                "that these bytes are what the run produced. The bake-off numbers remain "
                "development-grade with uncertified provenance."),
            "embedded_copy": payload_copy,
            "falsifiers": {},
        }
        record["falsifiers"] = {
            "f1_the_embedded_copy_is_byte_faithful": {
                "passed": bool(faithful),
                "evidence": {"why_it_can_fail": "a record that stores something other than what "
                                                "it hashed certifies nothing",
                             "sealed_file_sha256": digest_before}},
            "f3_the_runner_really_does_not_seal": {
                "passed": bool(not runner_seals),
                "evidence": {"why_it_can_fail": "if the runner does seal, this amendment has no "
                                                "reason to exist and the defect was misdiagnosed",
                             "runner": str(RUNNER), "found_seal_and_write": runner_seals}},
            "f4_the_downstream_consumer_rows_are_present": {
                "passed": bool(consumer_ok),
                "evidence": {"why_it_can_fail": "sealing an artifact that does not carry the rows "
                                                "run_track_b_nonneural_v1.py reads would seal the "
                                                "wrong file",
                             "architectures": sorted(by_arch)}},
        }
        out = target.parent / "sealed_record.json"
        seal = seal_and_write(record, out, contract=args.contract, reference=target)

        # f2 runs LAST and re-reads from disk: the point is to prove this script did not touch it.
        digest_after = sha256(target.read_bytes()).hexdigest()
        untouched = digest_after == digest_before
        rec = json.loads(out.read_text())
        rec["falsifiers"]["f2_the_original_is_not_modified"] = {
            "passed": bool(untouched),
            "evidence": {"why_it_can_fail": "never edit a dated artifact in place; if this script "
                                            "did, the digest it just published is worthless",
                         "sha256_before": digest_before, "sha256_after": digest_after}}
        rec["falsifiers"]["all_passed"] = all(
            v["passed"] for k, v in rec["falsifiers"].items()
            if k != "all_passed" and isinstance(v, dict))
        out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")

        ok = rec["falsifiers"]["all_passed"]
        rc |= 0 if ok else 1
        print(f"  {target}")
        print(f"    contenido {digest_before[:16]}…  registro {seal[:16]}…  -> {out}")
        for name, f in rec["falsifiers"].items():
            if name == "all_passed" or not isinstance(f, dict):
                continue
            print(f"      {name:<44} {'PASA' if f['passed'] else 'FALLA'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
