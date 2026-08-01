#!/usr/bin/env python3
"""Chain-aware re-attestation of source pins across the content-addressed audit DAG.

When a pinned source file legitimately changes, several dated attestations under
`research/paper2_exhaustive_search/` stop matching the working tree. Substituting the new hash
by hand is not enough: some of those files carry a `content_sha256` of their own, and other
files pin THAT. Updating a leaf without propagating turns four failures into five -- which is
exactly what happened on 2026-07-31 before this script existed.

The DAG has TWO kinds of edge, and missing either one leaves the suite red:

  * `content_sha256` -- the canonical-JSON hash a file carries of ITSELF, which other files pin;
  * **whole-file sha256** -- the raw bytes of one attestation, pinned inside another's
    `source_bindings` exactly like a source file. Handling only the first is what left
    `program_j_*_structure_audit` failing on the second attempt.

Rewriting any file changes both of its hashes, which can invalidate a file already rewritten
earlier in the same pass, so a single topological sweep is not enough. This iterates to a FIXED
POINT over both hash kinds and fails loudly if it does not converge.

TWO THINGS THIS SCRIPT REFUSES TO DO, on purpose:

1. **It does not decide.** `--cause` is mandatory and must name the artifact that proves the
   change is behaviour-preserving. A pin is provenance; moving one without a proven cause is
   falsifying provenance, not maintenance.
2. **It does not reformat.** Substitution is on the file's bytes and the line count is asserted
   unchanged, because an earlier re-attestation rewrote 900 lines through `json.dumps` and
   destroyed the diff.

It also leaves the Program O execution freezes alone. Those pin an older hash by design -- they
record the code AT EXECUTION TIME of a sealed, burned validation, and are supposed to go stale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research/paper2_exhaustive_search"
# Execution freezes of sealed validations: historical records, never re-attested.
FROZEN_PREFIXES = ("program_o_",)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def content_hash_of(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload.get("content_sha256") if isinstance(payload, dict) else None


def candidates() -> list[Path]:
    return [p for p in sorted(SEARCH.glob("*.json"))
            if not p.name.startswith(FROZEN_PREFIXES)]


def topological(paths: list[Path]) -> list[Path]:
    """Order so a file is rewritten only after everything it references."""
    hashes = {content_hash_of(p): p.name for p in paths if content_hash_of(p)}
    deps: dict[str, set[str]] = {}
    for path in paths:
        text = path.read_text()
        deps[path.name] = {name for digest, name in hashes.items()
                           if digest in text and name != path.name}
    ordered, seen = [], set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dep in sorted(deps.get(name, ())):
            visit(dep)
        ordered.append(name)

    for path in paths:
        visit(path.name)
    by_name = {p.name: p for p in paths}
    return [by_name[n] for n in ordered if n in by_name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="repo-relative path of the changed source")
    ap.add_argument("--old", required=True, help="the sha256 currently pinned")
    ap.add_argument("--cause", required=True,
                    help="artifact or doc proving the change is behaviour-preserving")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source = ROOT / args.source
    if not source.is_file():
        raise SystemExit(f"source not found: {args.source}")
    new = sha256_bytes(source.read_bytes())
    if new == args.old:
        print("pin already current; nothing to do")
        return 0

    print(f"source   {args.source}")
    print(f"pin      {args.old[:16]} -> {new[:16]}")
    print(f"cause    {args.cause}\n")

    paths = topological(candidates())
    baseline = {p: p.read_text() for p in paths}
    rewrites: dict[str, str] = {args.old: new}   # any old digest -> its new value
    touched: set[str] = set()

    for sweep in range(1, 9):
        changed = False
        for path in paths:
            original = path.read_text()
            text = original
            for old_digest, new_digest in rewrites.items():
                text = text.replace(old_digest, new_digest)
            if text == original:
                continue
            file_before = sha256_bytes(original.encode())
            content_before = content_hash_of(path)
            if content_before is not None:
                payload = json.loads(text)
                payload.pop("content_sha256")
                rewrites[content_before] = json_sha256(payload)
                text = text.replace(content_before, rewrites[content_before])
            if text.count("\n") != baseline[path].count("\n"):
                raise SystemExit(f"line count moved in {path.name}: a reformat, not a pin")
            # The file's own bytes just changed, so anything pinning THEM must follow.
            rewrites[file_before] = sha256_bytes(text.encode())
            if not args.dry_run:
                path.write_text(text)
            touched.add(path.name)
            changed = True
            note = (f"  content {content_before[:10]} -> {rewrites[content_before][:10]}"
                    if content_before else "")
            print(f"  sweep {sweep}: {'would rewrite' if args.dry_run else 'rewrote'} "
                  f"{path.name}{note}")
        if not changed:
            break
        if args.dry_run:
            break          # a dry run cannot observe the next sweep's inputs
    else:
        raise SystemExit("re-attestation did not reach a fixed point in 8 sweeps")

    if not touched:
        print("  no file carried the old pin")
    print(f"\n{len(touched)} file(s); Program O execution freezes deliberately untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
