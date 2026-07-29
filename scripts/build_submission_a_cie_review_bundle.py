#!/usr/bin/env python3
"""Build a deterministic, identity-guarded C&IE anonymous-review bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "papers" / "submission_a_program_q"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
IDENTITY_TOKENS = (
    "thom",
    "chisica",
    "urosario",
    "github.com/thom-320",
    "chisicathomas",
    "alexander.garrido@",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def submission_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = [
        (PAPER_ROOT / "main.tex", "manuscript/main.tex"),
        (PAPER_ROOT / "references.bib", "manuscript/references.bib"),
        (PAPER_ROOT / "HIGHLIGHTS.txt", "HIGHLIGHTS.txt"),
    ]
    files.extend(
        (path, f"manuscript/generated/tables/{path.name}")
        for path in sorted((PAPER_ROOT / "generated" / "tables").glob("*.tex"))
    )
    files.extend(
        (path, f"manuscript/generated/figures/{path.name}")
        for path in sorted((PAPER_ROOT / "generated" / "figures").glob("figure*.png"))
    )
    return files


def assert_anonymous(path: Path, data: bytes) -> None:
    if path.suffix.lower() not in {".tex", ".bib", ".txt", ".md", ".json"}:
        return
    lowered = data.decode("utf-8").lower()
    hits = [token for token in IDENTITY_TOKENS if token in lowered]
    if hits:
        raise SystemExit(
            f"STOP_ANONYMITY_GUARD: {path.relative_to(REPO_ROOT)} contains {hits}"
        )


def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build(output: Path) -> dict[str, object]:
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    selected = submission_files()
    missing = [str(path) for path, _ in selected if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    payload: list[tuple[str, bytes]] = []
    manifest_rows: list[dict[str, object]] = []
    for source, archive_name in selected:
        data = source.read_bytes()
        assert_anonymous(source, data)
        payload.append((archive_name, data))
        manifest_rows.append(
            {
                "path": archive_name,
                "sha256": sha256_bytes(data),
                "size_bytes": len(data),
            }
        )

    readme = (
        "Anonymous review bundle for Computers & Industrial Engineering.\n"
        "Upload TITLE_PAGE.tex separately after replacing every PENDING field.\n"
        "This bundle intentionally excludes custody/admin files and public URLs.\n"
    ).encode()
    assert_anonymous(Path("SUBMISSION_README.txt"), readme)
    payload.append(("SUBMISSION_README.txt", readme))
    manifest_rows.append(
        {
            "path": "SUBMISSION_README.txt",
            "sha256": sha256_bytes(readme),
            "size_bytes": len(readme),
        }
    )

    manifest = {
        "schema_version": "submission_a_cie_anonymous_bundle_v1",
        "anonymous_review": True,
        "title_page_included": False,
        "scientific_execution_performed": False,
        "file_count_excluding_manifest": len(manifest_rows),
        "files": sorted(manifest_rows, key=lambda row: str(row["path"])),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for archive_name, data in sorted(payload):
            add_bytes(archive, archive_name, data)
        add_bytes(archive, "MANIFEST.json", manifest_bytes)

    return {
        "output": str(output),
        "sha256": sha256_bytes(output.read_bytes()),
        "file_count": len(manifest_rows) + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
