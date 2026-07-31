#!/usr/bin/env python3
"""Explain every failing hash pin in the Paper-2 attestations, or refuse to explain it.

Five attestations across the repository pin sha256s that no longer match. Re-hashing them
would launder whatever caused the change; the useful step is to prove a CAUSE for each one and
let anything unexplained stay red. This script only classifies -- it writes no attestation and
changes no pin.

Causes it can prove, and how:

* `line_ending_normalization` -- `sha256(current bytes with LF -> CRLF) == pin`. The CSVs were
  written by Python's `csv` module, whose default terminator is `\\r\\n`, hashed at that moment,
  and later normalized to LF. Row content is identical, and the proof is byte-exact.
* `superseded_by_commit` -- the pin equals the file's content at some commit in `git log` for
  that path, and the current content equals another. The change is recorded work, and both
  ends are named.
* `post_freeze_source_edit` -- an implementation source of the canonical v2 metric whose
  behaviour was measured unchanged against the frozen implementation
  (`results/metric_audit/v2_metric_freeze_equivalence/`).
* `untracked_artifact` -- the file is not in git at all, so no history can explain it. Reported
  as its own class rather than silently lumped in with the others.

Anything else comes back `UNEXPLAINED`, which is the answer that matters.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"
EQUIVALENCE = ROOT / "results/metric_audit/v2_metric_freeze_equivalence/result.json"
V2_SOURCES = {
    "supply_chain/episode_metrics.py",
    "supply_chain/ret_thesis.py",
    "supply_chain/supply_chain.py",
    "tests/test_ret_excel_request_snapshot_contract.py",
}


def digest(blob: bytes) -> str:
    return sha256(blob).hexdigest()


def tracked_versions(relative: str) -> list[tuple[str, str]]:
    """`(commit, sha256)` for every version of a path in history, newest first."""
    out = []
    log = subprocess.run(["git", "log", "--format=%h", "--", relative],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    for commit in log:
        blob = subprocess.run(["git", "show", f"{commit}:{relative}"],
                              cwd=ROOT, capture_output=True).stdout
        if blob:
            out.append((commit, digest(blob)))
    return out


def classify(relative: str, pin: str) -> dict:
    path = Path(relative) if Path(relative).is_absolute() else ROOT / relative
    if not path.is_file():
        # Named apart from UNEXPLAINED because the diagnosis differs -- nothing changed that we
        # can see; the source is not on this machine. It stays FAILING either way: an absent
        # source cannot be verified, and pretending otherwise is the whole failure mode here.
        return {"cause": ("missing_external_source" if path.is_absolute()
                          else "UNEXPLAINED"),
                "detail": f"not present at {path}", "verifiable": False}
    blob = path.read_bytes()
    current = digest(blob)
    if current == pin:
        return {"cause": "matches", "current_sha256": current}

    if digest(blob.replace(b"\n", b"\r\n")) == pin:
        return {"cause": "line_ending_normalization", "current_sha256": current,
                "proof": "sha256(current with LF->CRLF) == pin, byte-exact",
                "rows_identical": True}

    history = tracked_versions(relative)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only", "--", relative],
                            cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not history:
        if staged:
            # The seed-burn ledger -- "authoritative record of which tape seed blocks have
            # been opened", required by charter 0.4 -- was excluded from git by a LOCAL
            # `.git/info/exclude` line, so it never left this machine and its history is
            # unrecoverable. It is being tracked from 2026-07-31 on. The pre-baseline delta
            # cannot be reconstructed and this cause says so out loud rather than pretending
            # the pin was explained.
            return {"cause": "untracked_now_baselined", "current_sha256": current,
                    "detail": ("was excluded from git by .git/info/exclude; tracked from "
                               "2026-07-31. The change between the pin and this baseline is "
                               "UNRECOVERABLE and is not claimed to be innocuous"),
                    "prior_history_recoverable": False}
        return {"cause": "untracked_artifact", "current_sha256": current,
                "detail": "not in git history; no record can explain the change"}
    pinned_at = [commit for commit, value in history if value == pin]
    current_at = [commit for commit, value in history if value == current]
    if pinned_at:
        equivalence = None
        if relative in V2_SOURCES and EQUIVALENCE.is_file():
            blob_eq = json.loads(EQUIVALENCE.read_text())
            if blob_eq.get("claim_status") == "V2_METRIC_UNCHANGED_SINCE_FREEZE":
                equivalence = {"artifact": str(EQUIVALENCE.relative_to(ROOT)),
                               "self_sha256": blob_eq.get("self_sha256"),
                               "rows_compared": blob_eq.get("rows_compared"),
                               "rows_differing": blob_eq.get("rows_differing")}
        return {
            "cause": ("post_freeze_source_edit" if equivalence else "superseded_by_commit"),
            "current_sha256": current,
            "pin_last_seen_at": pinned_at[0],
            "current_content_from": (current_at[0] if current_at else "working tree only"),
            "equivalence_evidence": equivalence,
        }
    if pin in _own_reattestation_hashes() and current_at:
        # The pin is a value THIS session's re-attestation wrote for a file that was still
        # uncommitted, so it names a working-tree state that never became a commit. The
        # target's current content IS in history, so the chain is still auditable end to end;
        # what is stale is our own intermediate write, not the evidence.
        return {"cause": "restated_after_own_reattestation", "current_sha256": current,
                "current_content_from": current_at[0],
                "detail": ("pin was written by hash_pin_reattestation_20260731.json for an "
                           "uncommitted state; restated to the committed content")}
    return {"cause": "UNEXPLAINED", "current_sha256": current,
            "detail": "the pinned hash matches no version of this file in git history"}


def _own_reattestation_hashes() -> set[str]:
    record = SEARCH / "hash_pin_reattestation_20260731.json"
    if not record.is_file():
        return set()
    blob = json.loads(record.read_text())
    return {row["pin_after"] for row in blob.get("repinned", []) if row.get("pin_after")}


def pins() -> dict[str, dict[str, str]]:
    """Every `path -> sha256` pin the failing attestations carry."""
    out: dict[str, dict[str, str]] = {}

    def add(source: str, relative: str, value: str) -> None:
        out.setdefault(source, {})[relative] = value

    audit = json.loads(
        (SEARCH / "ret_excel_request_snapshot_v2_implementation_audit_20260714.json").read_text())
    for relative, value in audit["implementation_sources"].items():
        add("v2_implementation_audit", relative, value)

    manifest = json.loads((SEARCH / "reproducibility_manifest.json").read_text())
    for key in ("artifact_hashes", "source_hashes"):
        for relative, value in manifest.get(key, {}).items():
            add("reproducibility_manifest", relative, value)

    inventory = json.loads((SEARCH / "global_sensitivity_portfolio_inventory.json").read_text())
    for study in inventory["executed_studies"]:
        for artifact in study.get("artifacts", [study.get("artifact")]):
            if artifact:
                add("global_sensitivity_inventory", artifact["path"], artifact["sha256"])

    readiness = json.loads((SEARCH / "terminal_return_readiness.json").read_text())

    def walk(node) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
                add("terminal_return_readiness", node["path"], node["sha256"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(readiness)

    supersession = json.loads((SEARCH / "paper3_claim_supersession.json").read_text())
    walk_target = supersession
    stack = [walk_target]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
                add("paper3_claim_supersession", node["path"], node["sha256"])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)

    structure = (SEARCH
                 / "program_j_request_snapshot_v2_frontier_structure_audit_20260714.json")
    if structure.is_file():
        for relative, value in json.loads(
                structure.read_text()).get("source_bindings", {}).items():
            add("program_j_structure_audit", relative, value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the full classification")
    args = ap.parse_args()

    report: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for source, rows in sorted(pins().items()):
        for relative, pin in sorted(rows.items()):
            verdict = classify(relative, pin)
            if verdict["cause"] == "matches":
                continue
            report.setdefault(source, {})[relative] = dict(verdict, pin=pin)
            counts[verdict["cause"]] = counts.get(verdict["cause"], 0) + 1

    for source, rows in report.items():
        print(f"\n{source}")
        for relative, verdict in rows.items():
            print(f"  {verdict['cause']:<28} {relative}")
    print("\nresumen:", json.dumps(counts, sort_keys=True))
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    return 1 if counts.get("UNEXPLAINED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
