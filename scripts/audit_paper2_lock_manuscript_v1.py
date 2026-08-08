#!/usr/bin/env python3
"""Audit Paper 2 manuscript text against the machine-readable claim lock.

This is a read-only scientific/editorial audit. It does not rewrite manuscript files, infer new
claims, or run simulations. It reports stale forbidden wording, lock/artifact status, lock age
relative to HEAD, and whether prose explicitly names claim IDs. The output is a diagnostic receipt,
not evidence for a scientific claim.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = ROOT / "papers/paper2/claim_lock.json"
DEFAULT_MANUSCRIPT = [
    ROOT / "papers/paper2/01_introduction.md",
    ROOT / "papers/paper2/02_methods.md",
    ROOT / "papers/paper2/03_results.md",
    ROOT / "papers/paper2/04_discussion.md",
]
DEFAULT_OUTPUT = ROOT / "results/paper2_claim_lock_manuscript_audit_v1/result.json"

# These are not all forbidden in every grammatical context: some may occur in a disclaimer. They
# are nevertheless useful as an explicit stale-vocabulary report because the current manuscript
# still uses the old comparator story in several paragraphs while the lock has already changed it.
LEGACY_VOCABULARY = {
    "state-blind": "historical comparator label; distinguish online cumulative from frozen prior",
    "mutually indistinguishable": "do not treat the post-hoc i.i.d. tie as valid under ordered accumulation",
    "transportable visit prior": "only the researcher-defined frozen factor prior is measured",
    "the transferable object": "mechanism remains qualified; do not promote a generic first-order object",
    "no corresponding neural advantage": "secondary trajectory wording is not an unqualified negative",
}


def git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def line_hits(path: Path, phrase: str) -> list[dict[str, object]]:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    hits: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if pattern.search(line):
            hits.append({"file": str(path.relative_to(ROOT)), "line": number, "text": line.strip()})
    return hits


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temp:
        temp.write(data)
        temp_path = Path(temp.name)
    try:
        # Refuse to overwrite a prior receipt. A new audit gets a new output path.
        if path.exists():
            raise FileExistsError(f"refusing to overwrite audit receipt: {path}")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manuscript", type=Path, nargs="*", default=DEFAULT_MANUSCRIPT)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    head = git_head()
    manuscript = [p for p in args.manuscript if p.exists()]
    missing_manuscript = [str(p.relative_to(ROOT)) for p in args.manuscript if not p.exists()]

    forbidden_hits: list[dict[str, object]] = []
    legacy_vocabulary_hits: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    missing_claim_ids: list[str] = []
    claim_id_hits: dict[str, list[dict[str, object]]] = {}
    all_text = "\n".join(p.read_text(encoding="utf-8") for p in manuscript)

    for phrase, reason in LEGACY_VOCABULARY.items():
        hits = []
        for path in manuscript:
            hits.extend(line_hits(path, phrase))
        if hits:
            legacy_vocabulary_hits.append({"phrase": phrase, "reason": reason, "hits": hits})

    for claim in lock.get("claims", []):
        claim_id = claim["claim_id"]
        hits_for_claim: list[dict[str, object]] = []
        for phrase in claim.get("forbidden", []):
            hits = []
            for path in manuscript:
                hits.extend(line_hits(path, phrase))
            if hits:
                hits_for_claim.append({"phrase": phrase, "hits": hits})
        if hits_for_claim:
            forbidden_hits.append({"claim_id": claim_id, "matches": hits_for_claim})

        if claim_id.lower() in all_text.lower():
            claim_id_hits[claim_id] = line_hits(
                next((p for p in manuscript if claim_id.lower() in p.read_text(encoding="utf-8").lower()), manuscript[0]),
                claim_id,
            )
        else:
            missing_claim_ids.append(claim_id)

        artifact = ROOT / claim["artifact"]
        successor = ROOT / claim["successor_when_sealed"] if claim.get("successor_when_sealed") else None
        artifact_rows.append({
            "claim_id": claim_id,
            "artifact": claim["artifact"],
            "exists": artifact.exists(),
            "file_sha256": sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None,
            "successor_when_sealed": str(successor.relative_to(ROOT)) if successor else None,
            "successor_exists": bool(successor and successor.exists()),
            "evidence_grade": claim.get("evidence_grade"),
            "inference_status": claim.get("inference_status"),
        })

    stale_lock = bool(head and lock.get("generated_at_commit") and lock["generated_at_commit"] != head)
    pending_successors = [r["claim_id"] for r in artifact_rows if r["successor_when_sealed"] and not r["successor_exists"]]
    status = "PASS" if not forbidden_hits and not missing_manuscript else "BLOCKED_MANUSCRIPT_LOCK_MISMATCH"
    if stale_lock and status == "PASS":
        status = "LOCK_REGENERATION_REQUIRED"

    payload = {
        "schema_version": "paper2_lock_manuscript_audit_v1",
        "status": status,
        "diagnostic_only": True,
        "git_head": head,
        "lock_path": str(args.lock.relative_to(ROOT)),
        "lock_generated_at_commit": lock.get("generated_at_commit"),
        "lock_is_stale_against_head": stale_lock,
        "manuscript_files": [str(p.relative_to(ROOT)) for p in manuscript],
        "missing_manuscript_files": missing_manuscript,
        "n_claims": len(lock.get("claims", [])),
        "n_claim_ids_absent_from_manuscript": len(missing_claim_ids),
        "claim_ids_absent_from_manuscript": missing_claim_ids,
        "forbidden_wording_hits": forbidden_hits,
        "legacy_vocabulary_hits": legacy_vocabulary_hits,
        "artifact_status": artifact_rows,
        "pending_successors": pending_successors,
        "lock_problems": lock.get("problems", []),
        "notes": [
            "Absence of a claim_id in prose is reported as a traceability warning, not a scientific failure.",
            "Forbidden wording hits require human review because a future manuscript may use a negated disclaimer.",
            "Legacy vocabulary hits are always reported for review, including when used in a disclaimer.",
            "A missing successor artifact is expected while comparator_repair_v2 is still running.",
        ],
    }
    write_exclusive(args.output, payload)
    print(json.dumps({
        "status": status,
        "forbidden_hits": len(forbidden_hits),
        "missing_claim_ids": len(missing_claim_ids),
        "pending_successors": pending_successors,
        "output": str(args.output),
    }, indent=2))
    return 1 if status == "BLOCKED_MANUSCRIPT_LOCK_MISMATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
