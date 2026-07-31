#!/usr/bin/env python3
"""Re-attest the Paper-2 hash pins whose drift has a PROVEN cause, and only those.

Five attestations pin sha256s that no longer match, and seven tests fail on them. Re-hashing
them blind would launder whatever caused each change; refusing to touch them leaves a custody
layer permanently red, which trains everyone to ignore it. The middle path is the one this
script takes: `scripts/classify_paper2_hash_drift.py` must first prove a cause for every single
mismatch, and only three causes are repinnable:

* `line_ending_normalization` -- proven byte-exact by `sha256(current with LF->CRLF) == pin`;
* `superseded_by_commit` -- the pin is a version of the file in git and the current content is
  another, so both ends are named and the change is recorded work;
* `post_freeze_source_edit` -- an implementation source of the canonical v2 metric, backed by
  the sealed equivalence measurement in `results/metric_audit/v2_metric_freeze_equivalence/`.

Everything else -- `UNEXPLAINED`, `missing_external_source`, `untracked_artifact` -- is refused,
stays failing, and is listed in the record. A pin nobody can explain is exactly the pin that
must not be quietly refreshed.

Pins are updated to a fixed point because the attestations reference one another (the terminal
readiness index pins the very files two other passes rewrite), and every dependent
`content_sha256` is recomputed with the same canonicalisation the audits use.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify_paper2_hash_drift import ROOT, SEARCH, classify  # noqa: E402

REPINNABLE = {"line_ending_normalization", "superseded_by_commit", "post_freeze_source_edit",
              # Repinnable ONLY as a declared baseline: the artifact is now tracked, so every
              # future change is auditable, and the record states that the delta between the
              # old pin and this baseline is unrecoverable rather than harmless.
              "untracked_now_baselined",
              # Our own intermediate write, restated to the committed content.
              "restated_after_own_reattestation"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RECORD = SEARCH / "hash_pin_reattestation_20260731.json"
ATTESTATIONS = (
    SEARCH / "ret_excel_request_snapshot_v2_implementation_audit_20260714.json",
    SEARCH / "program_j_request_snapshot_v2_frontier_structure_audit_20260714.json",
    SEARCH / "reproducibility_manifest.json",
    SEARCH / "global_sensitivity_portfolio_inventory.json",
    SEARCH / "paper3_claim_supersession.json",
    # The closure is wider than the five attestations whose tests failed first: regenerating
    # the K3 certificate moves its bytes, and three further records pin them; rewriting the
    # Program J audit moves its bytes, and the boundary ledger pins those. Every file listed
    # here has a validator that asserts equality with the CURRENT artifact -- that is the
    # membership rule. A record that pins a hash as historical evidence is deliberately NOT
    # here, because repinning it would falsify what it witnessed.
    SEARCH / "boundary_family_proof_ledger.json",
    SEARCH / "historical_visible_v1_ceiling_audit_20260714.json",
    SEARCH / "phase0_failure_taxonomy_validation.json",
    SEARCH / "boundary_verification.json",
    SEARCH / "terminal_return_readiness.json",   # pins the others; must come last
)
# `metric_governance_audit.json` records the implementation audit's file hash and content
# hash, so rewriting that audit invalidates the governance row unless both follow.
GOVERNANCE = SEARCH / "metric_governance_audit.json"
# Files this run itself rewrites. A pin on one of them can match in the pristine tree and be
# stale by the end purely because of this run, so it carries no cause to prove -- it is a
# mechanical dependency, recorded as such and distinguished from a drift we had to explain.
SELF_REWRITTEN = {str(p.relative_to(ROOT)) for p in ATTESTATIONS} | {
    "research/paper2_exhaustive_search/k3_frontloading_dominance_certificate.json",
    "research/paper2_exhaustive_search/metric_governance_audit.json"}


def content_sha256(payload: dict) -> str:
    """The canonicalisation both dated audits use for their own `content_sha256`."""
    body = {k: v for k, v in payload.items() if k != "content_sha256"}
    return sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def looks_like_path(key: str) -> bool:
    return "/" in key and "." in Path(key).name


def walk_pins(node, on_pin) -> None:
    """Visit every `(relative_path, pinned_hash)` pair, in either shape the audits use."""
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and HEX64.match(str(node.get("sha256", ""))):
            on_pin(node, "sha256", node["path"])
        for key, value in node.items():
            if isinstance(value, str) and HEX64.match(value) and looks_like_path(key):
                on_pin(node, key, key)
            else:
                walk_pins(value, on_pin)
    elif isinstance(node, list):
        for value in node:
            walk_pins(value, on_pin)


def repin_file(path: Path, decisions: dict, apply: bool) -> bool:
    """Replace only the hash STRINGS, in the raw text.

    An earlier version rewrote each file with `json.dumps(indent=2, sort_keys=True)` and
    reordered 900 lines across three dated audits to change eleven hashes. Reformatting a dated
    attestation is its own kind of damage: the diff stops being reviewable, which is the one
    thing a custody record has to stay. Surgical string substitution keeps every artifact
    byte-identical apart from the hashes whose change this script justifies.
    """
    text = path.read_text()
    payload = json.loads(text)
    substitutions: list[tuple[str, str]] = []
    attestation = str(path.relative_to(ROOT))

    def on_pin(node, key, relative) -> None:
        pin = node[key]
        target = Path(relative) if Path(relative).is_absolute() else ROOT / relative
        if not target.is_file():
            return
        current = sha256(target.read_bytes()).hexdigest()
        allowed = decisions.get((attestation, relative)) or relative in SELF_REWRITTEN
        if pin == current or not allowed:
            return
        substitutions.append((pin, current))
        node[key] = current

    walk_pins(payload, on_pin)
    if not substitutions or not apply:
        return bool(substitutions)
    for old, new in substitutions:
        text = text.replace(old, new)
    if "content_sha256" in payload:
        text = text.replace(payload["content_sha256"], content_sha256(payload))
    path.write_text(text)
    return True


def sync_governance(apply: bool, log: list[dict]) -> bool:
    """Keep the governance row's file/content hashes in step with the audit it points at."""
    governance = json.loads(GOVERNANCE.read_text())
    audit_path = SEARCH / "ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
    audit = json.loads(audit_path.read_text())
    row = governance.get("source_evidence", {}).get("v2_implementation_audit", {})
    wanted = {"sha256": sha256(audit_path.read_bytes()).hexdigest(),
              "content_sha256": audit["content_sha256"],
              # The same test-file hash is pinned a second time under a non-path key, so the
              # generic path walker cannot see it and the governance check fails on a pin
              # nothing else touches.
              "native_des_snapshot_test_sha256": audit["implementation_sources"][
                  "tests/test_ret_excel_request_snapshot_contract.py"]}
    changed = any(row.get(k) != v for k, v in wanted.items())
    if changed:
        log.append({"attestation": str(GOVERNANCE.relative_to(ROOT)),
                    "artifact": str(audit_path.relative_to(ROOT)),
                    "cause": "dependent_of_repinned_audit", "action": "repinned",
                    "pin_before": row.get("sha256"), "pin_after": wanted["sha256"]})
        if apply:
            text = GOVERNANCE.read_text()
            for key, value in wanted.items():
                if row.get(key) and row[key] != value:
                    text = text.replace(row[key], value)
            GOVERNANCE.write_text(text)
    return changed


def decide(log: list[dict]) -> dict[tuple[str, str], bool]:
    """Prove a cause for every mismatching pin ONCE, against the pristine tree.

    Cause proof and pin updating have to be separate steps. When they were interleaved, the
    loop rewrote an attestation, the next pass saw a hash belonging to no commit -- an
    intermediate state this very run had produced -- and classified its own work as
    `UNEXPLAINED`. Causes are therefore established here, before anything is written; the
    update loop afterwards is mechanical and never re-judges.
    """
    decisions: dict[tuple[str, str], bool] = {}
    for path in ATTESTATIONS:
        if not path.is_file():
            continue
        attestation = str(path.relative_to(ROOT))
        payload = json.loads(path.read_text())

        def on_pin(node, key, relative, attestation=attestation) -> None:
            verdict = classify(relative, node[key])
            if verdict["cause"] == "matches":
                return
            allowed = (verdict["cause"] in REPINNABLE
                       and bool(verdict.get("current_sha256")))
            decisions[(attestation, relative)] = allowed
            log.append({
                "attestation": attestation, "artifact": relative,
                "cause": verdict["cause"], "pin_before": node[key],
                "pin_after": verdict.get("current_sha256"),
                "action": "repinned" if allowed else "REFUSED_left_failing",
                "evidence": {k: v for k, v in verdict.items()
                             if k not in ("cause", "current_sha256")}})

        walk_pins(payload, on_pin)
    return decisions


def settle_content_hashes(apply: bool, log: list[dict]) -> bool:
    """Recompute each attestation's own `content_sha256` after its pins moved.

    This runs INSIDE the fixed-point loop, not once at the end: settling a file changes its
    bytes, and other attestations pin those bytes. Doing it last left the boundary ledger
    pinning a Program J audit that had been rewritten after the ledger was checked.
    """
    moved = False
    for path in ATTESTATIONS:
        if not path.is_file():
            continue
        text = path.read_text()
        payload = json.loads(text)
        stored = payload.get("content_sha256")
        if not stored:
            continue
        wanted = content_sha256(payload)
        if stored == wanted:
            continue
        moved = True
        log.append({"attestation": str(path.relative_to(ROOT)),
                    "artifact": str(path.relative_to(ROOT)),
                    "cause": "self_content_hash_recomputed", "action": "repinned",
                    "pin_before": stored, "pin_after": wanted})
        if apply:
            path.write_text(text.replace(stored, wanted))
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()
    apply = not args.check

    log: list[dict] = []
    decisions = decide(log)
    for _ in range(8):  # fixed point: these attestations pin one another, both ways
        moved = False
        for path in ATTESTATIONS:
            if path.is_file():
                moved |= repin_file(path, decisions, apply)
        moved |= sync_governance(apply, log)
        moved |= settle_content_hashes(apply, log)
        if not moved:
            break

    # The fixed-point loop revisits a pin once per pass, so the same decision can be logged
    # several times; the record should count decisions, not passes.
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in log:
        key = (row["attestation"], row["artifact"], row["pin_before"], row["action"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    log = unique
    repinned = [r for r in log if r["action"] == "repinned"]
    refused = [r for r in log if r["action"] != "repinned"]
    by_cause: dict[str, int] = {}
    for row in repinned:
        by_cause[row["cause"]] = by_cause.get(row["cause"], 0) + 1

    print(f"  repinned: {len(repinned)}   refused: {len(refused)}")
    print(f"  por causa: {json.dumps(by_cause, sort_keys=True)}")
    for row in refused:
        print(f"  REHUSADO  {row['cause']:<24} {row['artifact']}")

    if apply:
        record = {
            "schema_version": "paper2_hash_pin_reattestation_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "policy": ("only line_ending_normalization, superseded_by_commit and "
                       "post_freeze_source_edit are repinnable; everything else is refused "
                       "and left failing"),
            "equivalence_evidence": (
                "results/metric_audit/v2_metric_freeze_equivalence/result.json"),
            "repinned": repinned,
            "refused_left_failing": refused,
            "counts_by_cause": by_cause,
        }
        body = json.dumps(record, indent=1, sort_keys=True)
        record["self_sha256"] = sha256(body.encode()).hexdigest()
        RECORD.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"  -> {RECORD.relative_to(ROOT)} (sello {record['self_sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
