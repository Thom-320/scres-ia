#!/usr/bin/env python3
"""Seal the formal scope adjudication for the G3-obs source-contract mismatch.

This script is deliberately not a rerun and does not relabel the source artifact.  It
checks the source result and the supplemental f2 receipt, records which parts of the
legacy and v2 contracts match the execution envelope, and seals a separate adjudication
receipt.  The source result remains untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402


LEGACY_SEED_START = 5_200_001
LEGACY_SEED_END = 5_200_016
V2_SEED_START = 7_800_001
V2_SEED_END = 7_800_140
LEGACY_LOST_ORDERS_DELTA = 0.25
V2_LOST_ORDERS_DELTA = 0.50


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_self_sha256(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("self_sha256", None)
    encoded = json.dumps(body, indent=1, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _parse_declared_seed_range(text: str, start: int, end: int) -> bool:
    # The two declarations use localized thousands punctuation and a shortened end.
    if start == LEGACY_SEED_START:
        return "5.200.001–16" in text
    if start == V2_SEED_START:
        return bool(re.search(r"7\.800\.001\s*[–-]\s*7\.800\.140", text))
    return False


def _parse_lost_orders_delta(text: str) -> float:
    # The legacy contract has a table row; v2 states the amendment in prose
    # ("pasa de 0,25 a 0,50").  In both cases the last decimal on the line
    # containing the guardrail is the operative margin.
    for line in text.splitlines():
        if "lost_orders" not in line:
            continue
        matches = re.findall(r"[0-9]+[,.][0-9]+", line)
        if matches:
            return float(matches[-1].replace(",", "."))
    raise ValueError("contract does not declare a lost_orders margin")


def _range_facts(seeds: list[int], start: int, end: int) -> dict[str, Any]:
    return {
        "n": len(seeds),
        "min": min(seeds) if seeds else None,
        "max": max(seeds) if seeds else None,
        "declared_start": start,
        "declared_end": end,
        "matches_exactly": seeds == list(range(start, end + 1)),
    }


def build_adjudication(
    source: Path,
    f2_audit: Path,
    legacy_contract: Path,
    intended_v2_contract: Path,
    adjudication_contract: Path,
) -> dict[str, Any]:
    source_payload = load_json(source)
    f2_payload = load_json(f2_audit)
    legacy_text = legacy_contract.read_text(encoding="utf-8")
    v2_text = intended_v2_contract.read_text(encoding="utf-8")

    source_self = str(source_payload.get("self_sha256", ""))
    f2_self = str(f2_payload.get("self_sha256", ""))
    if not source_self or source_self != payload_self_sha256(source_payload):
        raise ValueError("source artifact self_sha256 is invalid")
    if not f2_self or f2_self != payload_self_sha256(f2_payload):
        raise ValueError("f2 audit self_sha256 is invalid")

    source_contract_path = ROOT / str(source_payload.get("contract_path", ""))
    if not source_contract_path.exists():
        raise ValueError(f"source contract is missing: {source_contract_path}")

    source_contract_sha = str(source_payload.get("contract_sha256", ""))
    source_contract_actual_sha = file_sha256(source_contract_path)
    legacy_contract_sha = file_sha256(legacy_contract)
    v2_contract_sha = file_sha256(intended_v2_contract)

    if source_contract_actual_sha != legacy_contract_sha:
        raise ValueError("source contract path is not the expected legacy contract")
    if source_contract_sha != source_contract_actual_sha:
        raise ValueError("source artifact does not carry a valid legacy contract hash")

    seeds = [int(seed) for seed in source_payload.get("seeds", [])]
    source_margins = source_payload.get("margins", {})
    source_lost_delta = float(source_margins.get("lost_orders"))
    source_schema = str(source_payload.get("schema_version", ""))

    source_seed_facts = _range_facts(seeds, LEGACY_SEED_START, LEGACY_SEED_END)
    v2_seed_facts = _range_facts(seeds, V2_SEED_START, V2_SEED_END)
    legacy_lost_delta = _parse_lost_orders_delta(legacy_text)
    v2_lost_delta = _parse_lost_orders_delta(v2_text)

    f2_source_match = f2_payload.get("source_artifact_self_sha256") == source_self
    f2_passed = bool(f2_payload.get("f2_all_cells_passed"))
    f2_intended_hash_valid = (
        f2_payload.get("intended_contract_sha256") == v2_contract_sha
    )

    source_contract_matches_v2 = source_contract_sha == v2_contract_sha
    source_matches_legacy_execution_scope = bool(
        source_seed_facts["matches_exactly"]
        and source_lost_delta == legacy_lost_delta
        and source_schema == "g3_obs_conversion_v1"
    )
    source_matches_v2_execution_fields = bool(
        v2_seed_facts["matches_exactly"]
        and source_lost_delta == v2_lost_delta
    )

    # This is intentionally conservative: the source is neither retroactively v2 nor
    # fully compliant with the older preregistration whose hash it carries.
    claim_status = "SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE"
    promotion_status = "BLOCKED_NO_RETROACTIVE_RESEAL_AND_NO_CONTRACT_CONFORMITY"

    return {
        "schema_version": "g3_obs_contract_scope_adjudication_v1",
        "adjudication_status": "FORMAL_SCOPE_ADJUDICATION_COMPLETE",
        "audit_status": "CONTRACT_SCOPE_ADJUDICATION_NO_NEW_SEEDS_NO_DES_RERUN",
        "claim_status": claim_status,
        "promotion_status": promotion_status,
        "decision": (
            "Preserve the source artifact exactly as sealed; do not classify it as a v2 run. "
            "Because its execution fields do not match the legacy seed block and margin, do "
            "not call it fully legacy-contract-confirmatory either. Treat it as development "
            "evidence with a supplemental f2 receipt."
        ),
        "source_artifact": {
            "path": str(source),
            "file_sha256": file_sha256(source),
            "self_sha256": source_self,
            "self_sha256_valid": True,
            "contract_path": str(source_contract_path.relative_to(ROOT)),
            "contract_sha256": source_contract_sha,
            "contract_hash_valid": source_contract_sha == source_contract_actual_sha,
            "schema_version": source_schema,
            "claim_status_at_source": source_payload.get("claim_status"),
            "seed_facts_against_legacy": source_seed_facts,
            "seed_facts_against_v2": v2_seed_facts,
            "margins_at_source": source_margins,
            "source_lost_orders_delta": source_lost_delta,
        },
        "contracts": {
            "legacy": {
                "path": str(legacy_contract.relative_to(ROOT)),
                "sha256": legacy_contract_sha,
                "declared_seed_range_present": _parse_declared_seed_range(
                    legacy_text, LEGACY_SEED_START, LEGACY_SEED_END
                ),
                "declared_seed_start": LEGACY_SEED_START,
                "declared_seed_end": LEGACY_SEED_END,
                "declared_lost_orders_delta": legacy_lost_delta,
            },
            "intended_v2": {
                "path": str(intended_v2_contract.relative_to(ROOT)),
                "sha256": v2_contract_sha,
                "declared_seed_range_present": _parse_declared_seed_range(
                    v2_text, V2_SEED_START, V2_SEED_END
                ),
                "declared_seed_start": V2_SEED_START,
                "declared_seed_end": V2_SEED_END,
                "declared_lost_orders_delta": v2_lost_delta,
            },
        },
        "scope_checks": {
            "source_seal_matches_legacy_contract": source_contract_sha == legacy_contract_sha,
            "source_seal_matches_intended_v2": source_contract_matches_v2,
            "source_execution_matches_legacy_declared_scope": source_matches_legacy_execution_scope,
            "source_execution_matches_v2_fields": source_matches_v2_execution_fields,
            "source_execution_is_fully_legacy_conformant": source_matches_legacy_execution_scope,
            "source_execution_is_v2_confirmatory": False,
            "reason_v2_confirmatory_is_false": (
                "The source seal and runner contract point to the legacy contract; the f2 "
                "receipt is an audit of stored summaries, not a retroactive v2 execution."
            ),
        },
        "supplemental_f2_audit": {
            "path": str(f2_audit),
            "file_sha256": file_sha256(f2_audit),
            "self_sha256": f2_self,
            "self_sha256_valid": True,
            "source_artifact_self_matches": f2_source_match,
            "intended_contract_hash_valid": f2_intended_hash_valid,
            "all_cells_passed": f2_passed,
            "claim_status_at_audit": f2_payload.get("claim_status"),
            "promotion_status_at_audit": f2_payload.get("promotion_status"),
            "interpretation": (
                "The complete f2 order is present in stored test summaries. This is "
                "supplemental evidence and does not alter the source contract seal."
            ),
        },
        "custody": {
            "source_artifact_modified": False,
            "retroactive_reseal": False,
            "new_seeds_opened": False,
            "des_rerun": False,
            "source_seed_block": seeds,
            "adjudication_reads_only": True,
        },
        "permitted_claims": [
            "The source artifact remains preserved with its original contract hash and self hash.",
            "The stored summaries satisfy the complete f2 ordering according to the supplemental audit.",
            "The source result may be cited as development evidence with this scope limitation.",
            "A future v2 execution would require its own execution-time v2 seal.",
        ],
        "prohibited_claims": [
            "The source run was executed or sealed under the v2 contract.",
            "The source run is fully confirmatory under the legacy contract.",
            "The source run is a virgin or independent v2 confirmation.",
            "The original runner executed the complete v2 f2 falsifier.",
            "A neural premium is confirmed by this artifact.",
        ],
        "reasons": [
            "The source self hash and source contract hash are internally valid.",
            "The source seal points to the legacy contract, not the intended v2 contract.",
            "The source seed block matches v2 fields but not the legacy contract's declared block.",
            "The source lost-orders margin matches v2 but not the legacy contract's declared margin.",
            "The supplemental f2 audit passes without DES rerun or new seeds.",
        ],
        "adjudication_contract_path": str(adjudication_contract.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/headroom/g3_obs_conversion_v2/result.json"))
    parser.add_argument("--f2-audit", type=Path,
                        default=Path("results/headroom/g3_obs_conversion_v2/f2_audit_result.json"))
    parser.add_argument("--legacy-contract", type=Path,
                        default=Path("docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md"))
    parser.add_argument("--intended-v2-contract", type=Path,
                        default=Path("docs/PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md"))
    parser.add_argument("--adjudication-contract", type=Path,
                        default=Path("docs/CONTRATO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/headroom/g3_obs_conversion_v2/contract_scope_adjudication.json"))
    args = parser.parse_args()

    paths = []
    for value in (args.source, args.f2_audit, args.legacy_contract,
                  args.intended_v2_contract, args.adjudication_contract):
        paths.append(value if value.is_absolute() else ROOT / value)
    source, f2_audit, legacy, intended_v2, adjudication_contract = paths
    output = args.output if args.output.is_absolute() else ROOT / args.output

    source_before = file_sha256(source)
    payload = build_adjudication(source, f2_audit, legacy, intended_v2, adjudication_contract)
    digest = seal_and_write(payload, output, contract=adjudication_contract, reference=source)
    source_after = file_sha256(source)
    if source_before != source_after:
        raise RuntimeError("source artifact changed during read-only adjudication")

    print(json.dumps({
        "claim_status": payload["claim_status"],
        "promotion_status": payload["promotion_status"],
        "source_execution_matches_legacy_declared_scope": payload["scope_checks"]["source_execution_matches_legacy_declared_scope"],
        "source_execution_matches_v2_fields": payload["scope_checks"]["source_execution_matches_v2_fields"],
        "f2_all_cells_passed": payload["supplemental_f2_audit"]["all_cells_passed"],
        "source_artifact_modified": payload["custody"]["source_artifact_modified"],
        "new_seeds_opened": payload["custody"]["new_seeds_opened"],
        "des_rerun": payload["custody"]["des_rerun"],
        "output": str(output),
        "self_sha256": digest,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
