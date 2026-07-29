#!/usr/bin/env python3
"""Rebuild and fail-closed verify the Paper-2 source-text extractions.

The source PDFs, DOCX and workbooks are not redistributed by this repository.
This script verifies their identities, reconstructs every indexed searchable
text file with the recorded toolchain, and checks pages, byte counts, line/word
counts and SHA-256 digests.  Source locations may be overridden without
weakening identity checks, for example::

    python scripts/verify_source_extractions.py \
      --source garrido_thesis_2017=/secure/sources/thesis.pdf

The tracked root ``thesis.txt`` is also rebuilt and checked as an explicitly
non-canonical legacy extraction.  Its different hash is caused by omitting the
``-layout`` flag; it is not an unidentified second thesis source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = (
    ROOT / "research" / "paper2_exhaustive_search" / "source_extraction_index.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if not capture:
        return ""
    return (completed.stdout + completed.stderr).strip()


def tool_version(command: str) -> str:
    if command == "pandoc":
        text = run([command, "--version"], capture=True)
        match = re.search(r"^pandoc\s+(\S+)", text, flags=re.MULTILINE)
    else:
        text = run([command, "-v"], capture=True)
        match = re.search(rf"^{re.escape(command)} version\s+(\S+)", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not parse {command} version from: {text!r}")
    return match.group(1)


def pdf_pages(path: Path) -> int:
    text = run(["pdfinfo", str(path)], capture=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not parse page count for {path}")
    return int(match.group(1))


def wc_counts(path: Path, locale_name: str) -> dict[str, int]:
    env = os.environ.copy()
    env["LC_ALL"] = locale_name
    completed = subprocess.run(
        ["wc", "-l", "-w", "-c", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    fields = completed.stdout.split()
    if len(fields) < 4:
        raise RuntimeError(f"could not parse wc output for {path}: {completed.stdout!r}")
    return {"lines": int(fields[0]), "words": int(fields[1]), "bytes": int(fields[2])}


def source_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        if not separator or not source_id or not raw_path:
            raise ValueError(f"invalid --source value {value!r}; expected ID=/path")
        if source_id in overrides:
            raise ValueError(f"duplicate --source override for {source_id}")
        overrides[source_id] = Path(raw_path).expanduser().resolve()
    return overrides


def expected_counts(source: dict[str, Any]) -> dict[str, int]:
    extraction = source["extraction"]
    return {
        "lines": int(source["extracted_lines"]),
        "words": int(source["extracted_words"]),
        "bytes": int(extraction["output_bytes"]),
    }


def check_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def extract(source: dict[str, Any], source_path: Path, output_path: Path) -> None:
    extraction = source["extraction"]
    tool = extraction["tool"]
    arguments = [str(value) for value in extraction["arguments"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tool == "pdftotext":
        run([tool, *arguments, str(source_path), str(output_path)])
    elif tool == "pandoc":
        run([tool, str(source_path), *arguments, "-o", str(output_path)])
    else:
        raise RuntimeError(f"unsupported extraction tool {tool!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="retain reconstructed texts here; otherwise use a temporary directory",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ID=/PATH",
        help="override a source location while retaining the manifest hash check",
    )
    parser.add_argument("--report", type=Path, help="optional JSON verification report")
    parser.add_argument(
        "--allow-tool-version-drift",
        action="store_true",
        help="allow another tool version; all output hashes and counts still must match",
    )
    args = parser.parse_args()

    index_path = args.index.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    check_equal(
        "index schema",
        index.get("schema_version"),
        "paper2_source_extraction_index_v3",
    )
    reproducibility = index["extraction_reproducibility"]
    overrides = source_overrides(args.source)
    known_ids = {source["id"] for source in index["sources"]}
    unknown_overrides = sorted(set(overrides) - known_ids)
    if unknown_overrides:
        raise RuntimeError(f"unknown source override IDs: {unknown_overrides}")

    tools = reproducibility["toolchain"]
    actual_tools: dict[str, str] = {}
    for command, expected in tools.items():
        if shutil.which(command) is None:
            raise RuntimeError(f"required command is missing: {command}")
        actual = tool_version(command)
        actual_tools[command] = actual
        if not args.allow_tool_version_drift:
            check_equal(f"{command} version", actual, expected)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="paper2-source-extract-")
        output_dir = Path(temporary.name)
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    sources_by_id = {source["id"]: source for source in index["sources"]}
    try:
        for source in index["sources"]:
            source_id = source["id"]
            source_path = overrides.get(source_id, Path(source["source_path"]))
            if not source_path.is_file():
                raise RuntimeError(f"missing source {source_id}: {source_path}")
            actual_source_hash = sha256(source_path)
            check_equal(
                f"{source_id} source SHA-256",
                actual_source_hash,
                source["source_sha256"],
            )

            row: dict[str, Any] = {
                "id": source_id,
                "source_path": str(source_path),
                "source_sha256": actual_source_hash,
                "source_verified": True,
            }
            extraction = source.get("extraction")
            if extraction is not None:
                output_path = output_dir / extraction["output_filename"]
                extract(source, source_path, output_path)
                output_hash = sha256(output_path)
                counts = wc_counts(output_path, reproducibility["word_count_locale"])
                check_equal(
                    f"{source_id} extracted SHA-256",
                    output_hash,
                    source["extracted_text_sha256"],
                )
                check_equal(f"{source_id} extracted counts", counts, expected_counts(source))
                row.update(
                    {
                        "output_path": str(output_path),
                        "output_sha256": output_hash,
                        **counts,
                        "extraction_verified": True,
                    }
                )
                if extraction["tool"] == "pdftotext":
                    pages = pdf_pages(source_path)
                    check_equal(f"{source_id} PDF pages", pages, source["pages"])
                    row["pages"] = pages
            results.append(row)

        legacy_results: list[dict[str, Any]] = []
        for legacy in index["tracked_legacy_extracts"]:
            source = sources_by_id[legacy["source_id"]]
            source_path = overrides.get(source["id"], Path(source["source_path"]))
            regenerated_path = output_dir / legacy["regenerated_output_filename"]
            legacy_source = {
                "extraction": {
                    "tool": legacy["tool"],
                    "arguments": legacy["arguments"],
                }
            }
            extract(legacy_source, source_path, regenerated_path)
            regenerated_hash = sha256(regenerated_path)
            regenerated_counts = wc_counts(
                regenerated_path, reproducibility["word_count_locale"]
            )
            expected_legacy_counts = {
                "lines": int(legacy["lines"]),
                "words": int(legacy["words"]),
                "bytes": int(legacy["bytes"]),
            }
            check_equal(
                f"legacy {legacy['path']} regenerated SHA-256",
                regenerated_hash,
                legacy["sha256"],
            )
            check_equal(
                f"legacy {legacy['path']} regenerated counts",
                regenerated_counts,
                expected_legacy_counts,
            )
            tracked_path = ROOT / legacy["path"]
            if not tracked_path.is_file():
                raise RuntimeError(f"missing tracked legacy extract: {tracked_path}")
            tracked_hash = sha256(tracked_path)
            check_equal(
                f"legacy {legacy['path']} tracked SHA-256",
                tracked_hash,
                legacy["sha256"],
            )
            canonical_hash = sources_by_id[legacy["source_id"]]["extracted_text_sha256"]
            if tracked_hash == canonical_hash:
                raise RuntimeError(
                    f"legacy {legacy['path']} unexpectedly equals the canonical layout extract"
                )
            legacy_results.append(
                {
                    "path": legacy["path"],
                    "source_id": legacy["source_id"],
                    "tracked_sha256": tracked_hash,
                    "regenerated_sha256": regenerated_hash,
                    "canonical_sha256": canonical_hash,
                    "relationship_verified": "same_source_different_recorded_extraction_mode",
                    **regenerated_counts,
                }
            )

        report = {
            "schema_version": "paper2_source_extraction_verification_v1",
            "index_path": str(index_path),
            "index_sha256": sha256(index_path),
            "toolchain": actual_tools,
            "tool_version_drift_allowed": bool(args.allow_tool_version_drift),
            "word_count_locale": reproducibility["word_count_locale"],
            "sources_verified": len(results),
            "extractions_verified": sum("extraction_verified" in row for row in results),
            "results": results,
            "legacy_extracts": legacy_results,
            "passed": True,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
