#!/usr/bin/env python3
"""One row per scientific experiment, not per file. Deduplicated by what makes an experiment.

WHY. A repo-wide census counted 439 positive intervals and called it evidence. Most of them are
two tape families re-analysed under ten directory names. Counting `result.json` files is
accounting, not replication; counting `claim_status` strings is worse, because that field is a
schema slot a June programmer could omit and a July one could over-claim into.

THE DEDUPLICATION KEY, which is the whole design:

    (contract_sha256, execution_fingerprint, seed_block_signature, endpoint, estimand_family)

Rows sharing a complete key are ONE experiment re-reported. Rows whose key is INCOMPLETE are never
merged with anything -- a missing field is not a match, and treating null == null would silently
fuse unrelated runs, which is the exact failure mode this file exists to end.

EVIDENCE GRADE IS DERIVED, NOT COPIED. `claim_status` is an author's label. The grade here comes
from checkable facts: does the run declare a confirmation role, did it consume a seed block the
custody registry records as virgin at opening, does it carry a contract hash at all. An artifact
that says CONFIRMED without a virgin block does not get to be confirmatory.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only over artifacts. No seeds, no simulation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
REGISTRY = Path("research/seed_custody_registry.json")
# Artifacts that live only on other branches and never landed here. Named explicitly because a
# glob over the worktree cannot see them, and leaving them out is how the census concluded the
# project had one confirmation.
OFF_HEAD = (
    ("codex/paper-b-retained-v5", "results/garrido_h2_h3_confirmation_v1/result.json"),
    ("codex/paper-b-cf1-cf20-replication",
     "results/q_r1/successor_confirmation_v1/merged/result.json"),
)


def sig(value) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def seed_signature(d: dict) -> str | None:
    """A block, not a list: two runs over the same range are the same custody unit."""
    seeds = d.get("seeds")
    if isinstance(seeds, list) and seeds:
        try:
            return f"{int(min(seeds))}-{int(max(seeds))}"
        except (TypeError, ValueError):
            return sig(seeds)
    blk = d.get("seed_block")
    if isinstance(blk, dict) and "start" in blk:
        return f"{blk['start']}-{blk['end']}"
    # Discrete tape roots are a custody unit too. Reading only `seeds` is why a valid confirmation
    # whose roots are recorded as a registered block still looked unregistered.
    roots = d.get("confirmation_tape_roots")
    if isinstance(roots, list) and roots:
        return f"{int(min(roots))}-{int(max(roots))}"
    for k in ("blocks", "source_slices", "sources"):
        if k in d:
            return sig(d[k])
    return None


def execution_fingerprint(d: dict) -> str | None:
    mm = d.get("module_manifest")
    if isinstance(mm, dict):
        return sig({"e": mm.get("entry_script_sha256"), "m": mm.get("modules")})
    return None


def endpoint_of(d: dict) -> str | None:
    for k in ("primary_metric", "metric", "endpoint"):
        if isinstance(d.get(k), str):
            return d[k]
    return None


def estimand_of(d: dict) -> str | None:
    for k in ("estimand", "schema_version"):
        if isinstance(d.get(k), str):
            return d[k]
    return None


def grade(d: dict, seed_sig: str | None, virgin_blocks: dict) -> tuple[str, str]:
    """Derived from checkable facts. Never from claim_status."""
    role = str(d.get("run_role") or "")
    scope = str(d.get("scope") or "")
    status = str(d.get("claim_status") or "")
    has_contract = bool(d.get("contract_sha256"))
    if not has_contract:
        return "UNCONTRACTED", "no contract hash: nothing fixes what this run was allowed to claim"
    virgin = seed_sig in virgin_blocks
    if "HALT" in status or "STOP" in status or "VOID" in status or "REFUT" in status:
        return "NEGATIVE_OR_HALTED", "the run's own terminal state is a stop or a refutation"
    # A confirmation is recognised from checkable structure, not only from a `run_role` string.
    # Evidence does not become development because the field did not exist when it was written:
    # an artifact that asserts its confirmation roots were opened and its development roots were
    # not is making the same claim the string makes, in a form that can be checked.
    declares_confirmation = bool(
        role == "CONFIRMATION" or "CONFIRMATION_ON" in scope
        or (d.get("confirmation_roots_opened") is True
            and d.get("development_roots_opened") is False))
    if declares_confirmation:
        if virgin:
            return "CONFIRMATORY", f"confirmation role over custody block {seed_sig}"
        return "CONFIRMATION_ROLE_WITHOUT_VIRGIN_BLOCK", (
            "declares a confirmation role but its seed block is not one the custody registry "
            "records as opened virgin for it")
    if "DIAGNOSTIC" in scope or "DIAGNOSTIC" in status or "INSTRUMENT" in scope:
        return "DIAGNOSTIC", "instrument property, no comparative claim"
    if "REPLAY" in scope or d.get("replay_of"):
        return "REPLAY", "declared re-execution of an already-open block"
    return "DEVELOPMENT", "contracted, but not a confirmation over a virgin block"


def load(path: Path, branch: str | None = None):
    if branch:
        out = subprocess.run(["git", "show", f"{branch}:{path}"], capture_output=True, text=True)
        if out.returncode:
            return None
        raw = out.stdout
    else:
        raw = path.read_text()
    try:
        d = json.loads(raw)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("research/evidence_registry.jsonl"))
    ap.add_argument("--receipt", type=Path,
                    default=Path("results/evidence_registry/result.json"))
    args = ap.parse_args()

    reg = json.loads(REGISTRY.read_text())
    virgin_blocks = {f"{b['start']}-{b['end']}": b["id"] for b in reg["blocks"]
                     if "CONFIRMATION_COMPLETE" in b.get("status", "")
                     or b.get("status") == "RESERVED_NOT_OPENED"}

    candidates = [(None, p) for p in sorted(Path("results").rglob("*.json"))
                  if p.name in {"result.json", "adjudication.json", "verdict.json"}]
    candidates += [(b, Path(p)) for b, p in OFF_HEAD]

    rows, skipped = [], 0
    for branch, path in candidates:
        d = load(path, branch)
        if d is None or not (d.get("claim_status") or d.get("status") or d.get("verdict")):
            skipped += 1
            continue
        ss = seed_signature(d)
        key_parts = {
            "contract_sha256": d.get("contract_sha256"),
            "execution": execution_fingerprint(d),
            "seed_block": ss,
            "endpoint": endpoint_of(d),
            "estimand": estimand_of(d),
        }
        complete = all(v is not None for v in key_parts.values())
        g, why = grade(d, ss, virgin_blocks)
        rows.append({
            "artifact_path": str(path), "branch": branch or "HEAD",
            "content_sha256": sig(d),
            "claim_status_as_authored": d.get("claim_status") or d.get("status")
            or d.get("verdict"),
            "evidence_grade": g, "grade_rationale": why,
            "contract_path": d.get("contract_path") or d.get("preregistration"),
            "self_sha256": d.get("self_sha256"),
            "created_at": d.get("created_at"),
            "supersedes": d.get("supersedes"), "replay_of": d.get("replay_of"),
            "dedup_key": key_parts, "dedup_key_complete": complete,
            "dedup_key_hash": sig(key_parts) if complete else None,
        })

    # Collapse only COMPLETE keys. An incomplete key never merges: a missing field is not a match.
    # Two merge bases, kept distinct on purpose.
    #  IDENTITY: byte-identical content is the same experiment no matter which branch it was read
    #            from. This is not a relaxation of the key -- it is an exact check on a different
    #            field, and it is what collapses an artifact rescued onto HEAD against the copy
    #            still living on its origin branch. Some artifacts carry no self_sha256, so the
    #            content hash is the fallback and is always available.
    #  KEY:      the five-part experiment key, and ONLY when complete.
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(f"identity:{r['content_sha256']}", []).append(r)
    for members in list(groups.values()):
        members.sort(key=lambda r: (str(r.get("created_at") or ""), r["branch"] != "HEAD"))
        for r in members[:-1]:
            r["duplicate_of"], r["merge_basis"] = members[-1]["artifact_path"], "identity"
        members[-1]["duplicate_of"], members[-1]["merge_basis"] = None, None

    survivors = [r for r in rows if r["duplicate_of"] is None]
    key_groups: dict[str, list[dict]] = {}
    for r in survivors:
        if r["dedup_key_complete"]:
            key_groups.setdefault(r["dedup_key_hash"], []).append(r)
    for members in key_groups.values():
        members.sort(key=lambda r: str(r.get("created_at") or ""))
        for r in members[:-1]:
            r["duplicate_of"], r["merge_basis"] = members[-1]["artifact_path"], "key"
    for r in rows:
        r.setdefault("duplicate_of", None)
        r.setdefault("merge_basis", None)
    n_experiments = sum(1 for r in rows if r["duplicate_of"] is None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    by_grade: dict[str, int] = {}
    for r in rows:
        if r["duplicate_of"] is None:
            by_grade[r["evidence_grade"]] = by_grade.get(r["evidence_grade"], 0) + 1
    confirmatory = [r["artifact_path"] for r in rows
                    if r["evidence_grade"] == "CONFIRMATORY" and r["duplicate_of"] is None]
    dup_collapsed = sum(1 for r in rows if r["duplicate_of"])

    # The family the audit named: ten directories over two tape sets. If the key does not collapse
    # them it is not doing its job.
    meta = [r for r in rows if "garrido_meta_learner" in r["artifact_path"]]
    meta_distinct = len({r["dedup_key_hash"] for r in meta if r["dedup_key_complete"]})
    # ...and the two H3 slices are DIFFERENT seed blocks and must NOT collapse.
    slices = [r for r in rows if "h3power_h3_contract" in r["artifact_path"]]
    slices_distinct = len({r["dedup_key_hash"] for r in slices if r["dedup_key_complete"]})

    falsifiers = {
        "f1_the_key_collapses_a_known_duplicate_family": {
            "passed": bool(len(meta) > meta_distinct >= 1),
            "evidence": {"why_it_can_fail": "the meta-learner family is many directories over two "
                                            "tape sets; a key that leaves them all distinct is "
                                            "counting files again",
                         "n_artifacts": len(meta), "n_distinct_keys": meta_distinct}},
        "f2_the_key_does_not_over_merge_different_blocks": {
            "passed": bool(slices_distinct == len([r for r in slices if r["dedup_key_complete"]])
                           and slices_distinct >= 2),
            "evidence": {"why_it_can_fail": "the local and vps H3 slices are DIFFERENT seed blocks "
                                            "(6000001-90 and 6000091-120); merging them would "
                                            "erase the independence the n=120 merge depends on",
                         "n_slices": len(slices), "n_distinct_keys": slices_distinct}},
        "f3_incomplete_keys_are_never_merged": {
            "passed": not any(r["merge_basis"] == "key" and not r["dedup_key_complete"]
                              for r in rows),
            "evidence": {"why_it_can_fail": "treating a missing field as equal to another missing "
                                            "field fuses unrelated runs, which is the exact "
                                            "failure this registry exists to end. Merging on an "
                                            "IDENTICAL content hash is a different, exact check "
                                            "and is permitted",
                         "n_incomplete": sum(1 for r in rows if not r["dedup_key_complete"]),
                         "merged_by_identity": sum(1 for r in rows
                                                   if r["merge_basis"] == "identity"),
                         "merged_by_key": sum(1 for r in rows if r["merge_basis"] == "key")}},
        "f4_the_grade_is_derived_not_copied": {
            "passed": any(r["evidence_grade"] != "CONFIRMATORY"
                          and "CONFIRM" in str(r["claim_status_as_authored"]).upper()
                          for r in rows),
            "evidence": {"why_it_can_fail": "if no artifact authored as CONFIRM is graded anything "
                                            "else, the grade is just echoing claim_status and "
                                            "certifies nothing",
                         "examples": [r["artifact_path"] for r in rows
                                      if r["evidence_grade"] != "CONFIRMATORY"
                                      and "CONFIRM" in str(r["claim_status_as_authored"]).upper()
                                      ][:4]}},
        "f5_off_head_artifacts_are_included": {
            "passed": bool(sum(1 for r in rows if r["branch"] != "HEAD") == len(OFF_HEAD)),
            "evidence": {"why_it_can_fail": "a glob over the worktree cannot see other branches, "
                                            "and leaving them out is how the census concluded the "
                                            "project had one confirmation",
                         "off_head": [r["artifact_path"] for r in rows if r["branch"] != "HEAD"]}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    print(f"  {len(rows)} artefactos leídos · {dup_collapsed} colapsados como re-reporte · "
          f"{n_experiments} experimentos distintos")
    print("  por grado (sin duplicados):")
    for g, n in sorted(by_grade.items(), key=lambda kv: -kv[1]):
        print(f"    {g:<42} {n}")
    print("\n  CONFIRMATORIOS:")
    for c in confirmatory:
        print(f"    {c}")
    print()
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<50} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "evidence_registry_v1",
        "claim_status": "EVIDENCE_REGISTRY_BUILT",
        "scope": "INDEX_ONLY_NO_SCIENTIFIC_CLAIM_NO_SEEDS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "registry_path": str(args.out),
        "dedup_key": ["contract_sha256", "execution_fingerprint", "seed_block_signature",
                      "endpoint", "estimand_family"],
        "n_artifacts": len(rows), "n_skipped_no_verdict": skipped,
        "n_collapsed_as_rereport": dup_collapsed, "n_distinct_experiments": n_experiments,
        "by_evidence_grade": by_grade, "confirmatory": confirmatory,
        "falsifiers": falsifiers,
    }
    digest = seal_and_write(payload, args.receipt, contract=args.contract,
                            reference=Path("docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md")
                            if False else Path("results/step3_split_pooled/result.json"))
    print(f"\n  -> {args.out}  ·  recibo {args.receipt} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
