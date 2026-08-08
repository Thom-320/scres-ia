#!/usr/bin/env python3
"""Build a deterministic, identity-guarded C&IE review bundle for canonical Paper 2 files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
DEFAULT_TOKENS = (
    "thomas chisica", "thom chisica", "chisica", "urosario", "github.com/thom-320", "<USER>",
    "alexander.garrido@",
)
CANONICAL = (
    "01_introduction.md", "02_methods.md", "03_results.md", "04_discussion.md",
    "appendix_hypotheses.md", "submission_metadata.md",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_anonymous(path: Path, data: bytes, tokens: tuple[str, ...]) -> None:
    if path.suffix.lower() not in {".tex", ".bib", ".txt", ".md", ".json", ".pdf"}:
        return
    lowered = data.decode("utf-8", errors="ignore").lower()
    hits = [token for token in tokens if token.lower() in lowered]
    if hits:
        raise SystemExit(f"STOP_ANONYMITY_GUARD: {path} contains {hits}")


def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build(paper_root: Path, output: Path, *, pdf: Path | None,
          tokens: tuple[str, ...]) -> dict[str, object]:
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    sources = [(paper_root / name, f"manuscript/{name}") for name in CANONICAL]
    if pdf is not None:
        sources.append((pdf, "manuscript/compiled/main.pdf"))
    missing = [str(path) for path, _ in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    payload: list[tuple[str, bytes]] = []
    rows: list[dict[str, object]] = []
    for source, archive_name in sources:
        data = source.read_bytes()
        assert_anonymous(source, data, tokens)
        payload.append((archive_name, data))
        rows.append({"path": archive_name, "sha256": sha256_bytes(data),
                     "size_bytes": len(data)})

    readme = (
        "Anonymous review bundle for Computers & Industrial Engineering.\n"
        "TITLE_PAGE and author declarations are supplied separately.\n"
        "This bundle excludes custody, registry and public-URL governance files.\n"
    ).encode()
    payload.append(("SUBMISSION_README.txt", readme))
    rows.append({"path": "SUBMISSION_README.txt", "sha256": sha256_bytes(readme),
                 "size_bytes": len(readme)})
    manifest = {"schema_version": "paper2_cie_anonymous_bundle_v1",
                "anonymous_review": True, "title_page_included": False,
                "files": sorted(rows, key=lambda row: str(row["path"]))}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    payload.append(("MANIFEST.json", manifest_bytes))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in sorted(payload):
            add_bytes(archive, name, data)
    return {"output": str(output), "sha256": sha256_bytes(output.read_bytes()),
            "file_count": len(payload)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-root", type=Path, default=Path("papers/paper2"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--identity-token", action="append", default=[])
    args = ap.parse_args()
    tokens = tuple(DEFAULT_TOKENS) + tuple(args.identity_token)
    print(json.dumps(build(args.paper_root, args.output, pdf=args.pdf, tokens=tokens),
                    indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
