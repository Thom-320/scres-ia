#!/usr/bin/env python3
"""Static C&IE bundle gate for the canonical Paper 2 files.

The checker is intentionally conservative: it does not compile a manuscript or decide scientific
meaning. It catches stale wording, missing claim-lock rows, accidental identity leakage supplied by
the caller, and ZIP/PDF metadata that can defeat double-blind review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


FORBIDDEN = (
    "the chain learns",
    "organizational learning",
    "organisational learning",
    "neural premium",
    "factorized ucb policy",
    "excludes zero",
    "validation of the des",
    "the des is validated",
    "reproduces the simulink model",
)
RETRACTED = ("7.90", "7,90", "5.43", "5,43", "7.24", "7,24", "13.54", "13,54",
             "12.42", "12,42", "5.83", "5,83")


def _text_files(root: Path) -> list[Path]:
    names = {"01_introduction.md", "02_methods.md", "03_results.md", "04_discussion.md",
             "appendix_hypotheses.md", "submission_metadata.md"}
    return [root / name for name in sorted(names) if (root / name).exists()]


def _check_submission_metadata(path: Path) -> list[str]:
    if not path.exists():
        return ["submission metadata is missing"]
    text = path.read_text(errors="replace")
    if "## Abstract" not in text or "## Highlights" not in text:
        return ["submission metadata must contain Abstract and Highlights sections"]
    abstract = text.split("## Abstract", 1)[1].split("## Highlights", 1)[0]
    highlights = text.split("## Highlights", 1)[1].split("## Declarations", 1)[0]
    words = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", abstract)
    errors = []
    if len(words) > 250:
        errors.append(f"abstract exceeds 250 words ({len(words)})")
    lines = [line[2:].strip() for line in highlights.splitlines() if line.startswith("- ")]
    if not 3 <= len(lines) <= 5:
        errors.append(f"highlights count is {len(lines)}, expected 3-5")
    errors.extend(f"highlight exceeds 85 characters: {line}" for line in lines if len(line) > 85)
    return errors


def check_bundle(paper_root: Path, claim_lock: Path, pdf: Path | None = None,
                 bundle: Path | None = None, identity_tokens: tuple[str, ...] = ()) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_submission_metadata(paper_root / "submission_metadata.md"))
    try:
        lock = json.loads(claim_lock.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read claim lock: {exc}"]
    portfolio_lock_rel = lock.get("portfolio_lock")
    portfolio_digest = lock.get("portfolio_lock_sha256")
    if portfolio_lock_rel and portfolio_digest:
        portfolio_path = (claim_lock.parent.parent / Path(portfolio_lock_rel).name)
        if not portfolio_path.exists():
            errors.append(f"portfolio claim lock is missing: {portfolio_path}")
        elif hashlib.sha256(portfolio_path.read_bytes()).hexdigest() != portfolio_digest:
            errors.append("portfolio claim lock digest does not match the Paper 2 lock")
    else:
        errors.append("Paper 2 claim lock is not anchored to the portfolio claim lock")
    for problem in lock.get("problems", []):
        errors.append(f"claim lock problem: {problem}")
    for row in lock.get("claims", []):
        if row.get("paper_id") != "P2":
            errors.append(f"claim has no unique P2 owner: {row.get('claim_id')}")
        if not row.get("exists"):
            errors.append(f"missing claim artifact: {row.get('claim_id')}")
        if not row.get("artifact"):
            errors.append(f"claim has no artifact: {row.get('claim_id')}")

    text = "\n".join(p.read_text(errors="replace") for p in _text_files(paper_root))
    lowered = text.lower()
    for phrase in FORBIDDEN:
        if phrase in lowered:
            errors.append(f"forbidden wording: {phrase}")
    for number in RETRACTED:
        if re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])", text):
            errors.append(f"retracted figure: {number}")
    for token in identity_tokens:
        if token and token.lower() in lowered:
            errors.append(f"identity token in anonymous source: {token}")

    if pdf and pdf.exists():
        raw = pdf.read_bytes().lower()
        for token in identity_tokens:
            if token and token.lower().encode() in raw:
                errors.append(f"identity token in compiled PDF: {token}")
    if bundle and bundle.exists():
        try:
            with zipfile.ZipFile(bundle) as zf:
                names = zf.namelist()
                if any(Path(name).name in {".aux", ".log", ".bbl"} for name in names):
                    errors.append("compiled auxiliary files are present in the anonymous bundle")
                raw_names = "\n".join(names).lower()
                for token in identity_tokens:
                    if token and token.lower() in raw_names:
                        errors.append(f"identity token in bundle member name: {token}")
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"cannot inspect bundle: {exc}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-root", type=Path, default=Path("papers/paper2"))
    ap.add_argument("--claim-lock", type=Path, default=Path("papers/paper2/claim_lock.json"))
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--identity-token", action="append", default=[])
    args = ap.parse_args()
    errors = check_bundle(args.paper_root, args.claim_lock, args.pdf, args.bundle,
                          tuple(args.identity_token))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("OK: C&IE bundle static gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
