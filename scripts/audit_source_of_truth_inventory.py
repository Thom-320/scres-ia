#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_benchmark_bundles import (
    audit_bundle,
    compare_to_reference,
    normalize_run_ref,
)

DEFAULT_DOCS: tuple[str, ...] = (
    "docs/REPOSITORY_SOURCE_OF_TRUTH.md",
    "docs/REPRODUCIBILITY.md",
    "docs/PAPER_EXPERIMENTAL_CHECKLIST.md",
    "docs/briefs/preliminary_results_synthesis.md",
    "docs/manuscript_notes/control_reward_500k_source_of_truth.md",
    "docs/manuscript_notes/section_4_3_source_of_truth.md",
)
PATH_PATTERN = re.compile(
    r"(?P<path>"
    r"/[^\s`)\]]*(?:docs/artifacts/control_reward|outputs/benchmarks|outputs/paper_benchmarks)[^\s`)\]]*"
    r"|(?:docs/artifacts/control_reward|outputs/benchmarks|outputs/paper_benchmarks)[^\s`)\]]*"
    r")"
)
REFERENCE_PRIORITY: tuple[str, ...] = (
    "paper_ret_seq_k020_500k",
    "paper_control_v1_500k",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the benchmark bundles cited by repository source-of-truth "
            "documents and classify each citation as auditable, historical, or invalid."
        )
    )
    parser.add_argument(
        "--docs",
        nargs="*",
        default=list(DEFAULT_DOCS),
        help="Source-of-truth documents to scan for benchmark references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/source_of_truth_inventory"),
    )
    return parser


def resolve_repo_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def normalize_reference_path(raw_path: str, repo_root: Path) -> Path:
    if raw_path.startswith(str(repo_root)):
        return Path(raw_path)
    return repo_root / raw_path


def classify_reference_target(path: Path) -> tuple[str, Path | None]:
    if path.suffix in {".md", ".csv", ".png"}:
        if path.name in {"summary.json", "manifest.json"}:
            return "bundle_file", path.parent
        return "non_bundle_reference", None
    if path.suffix == ".json":
        if path.name in {"summary.json", "manifest.json", "status.json"}:
            return "bundle_file", path.parent
        return "json_reference", None
    if path.is_dir():
        return "bundle_dir", path
    if path.suffix == "":
        return "bundle_dir", path
    return "unknown_reference", None


def extract_doc_references(doc_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    text = doc_path.read_text(encoding="utf-8")
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in PATH_PATTERN.finditer(text):
        raw = match.group("path").rstrip(".,:;")
        if raw in seen:
            continue
        seen.add(raw)
        abs_path = normalize_reference_path(raw, repo_root)
        ref_type, bundle_root = classify_reference_target(abs_path)
        refs.append(
            {
                "doc_path": str(doc_path.resolve()),
                "raw_reference": raw,
                "resolved_path": str(abs_path.resolve()),
                "reference_type": ref_type,
                "bundle_root": (
                    str(bundle_root.resolve()) if bundle_root is not None else None
                ),
                "exists": abs_path.exists(),
            }
        )
    return refs


def choose_reference_label(audited_rows: list[dict[str, Any]]) -> str | None:
    available = {row["label"]: row for row in audited_rows}
    for label in REFERENCE_PRIORITY:
        if label in available and available[label]["audit_status"] == "auditable":
            return label
    for row in audited_rows:
        if row["audit_status"] == "auditable":
            return row["label"]
    return None


def recommended_action(
    audited_row: dict[str, Any],
    compatibility_row: dict[str, Any] | None,
) -> str:
    status = audited_row["audit_status"]
    if status == "invalid":
        return "remove_from_source_of_truth"
    if status == "historical_artifact":
        return "demote_to_historical"
    if compatibility_row is None:
        return "keep_with_manual_review"
    if compatibility_row["service_metrics_comparable"]:
        return "keep"
    return "manual_review"


def build_markdown_report(
    *,
    docs: list[Path],
    doc_refs: list[dict[str, Any]],
    audited_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    reference_label: str | None,
    bundle_citations: dict[str, list[str]],
) -> str:
    compatibility_by_label = {row["label"]: row for row in compatibility_rows}
    lines = [
        "# Source-of-Truth Inventory Audit",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Documents scanned: {len(docs)}",
        (
            f"- Reference bundle: `{reference_label}`"
            if reference_label
            else "- Reference bundle: none"
        ),
        "",
        "## Documents scanned",
        "",
    ]
    for doc in docs:
        lines.append(f"- `{doc}`")

    lines.extend(
        [
            "",
            "## Bundle inventory",
            "",
            "| Bundle | Audit status | Recommended action | Cited by | Reasons |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in audited_rows:
        citations = ", ".join(
            f"`{Path(path).name}`"
            for path in sorted(bundle_citations.get(row["label"], []))
        )
        compatibility_row = compatibility_by_label.get(row["label"])
        lines.append(
            f"| `{row['label']}` | `{row['audit_status']}` | "
            f"`{recommended_action(row, compatibility_row)}` | "
            f"{citations or 'n/a'} | "
            f"{', '.join(row.get('reasons', [])) or 'ok'} |"
        )

    non_bundle_refs = [
        ref
        for ref in doc_refs
        if ref["reference_type"] != "bundle_dir"
        and ref["reference_type"] != "bundle_file"
    ]
    if non_bundle_refs:
        lines.extend(["", "## Non-bundle references", ""])
        for ref in non_bundle_refs:
            lines.append(
                f"- `{Path(ref['doc_path']).name}` -> `{ref['raw_reference']}` "
                f"({ref['reference_type']})"
            )

    lines.extend(
        [
            "",
            "## Compatibility vs reference",
            "",
            "| Bundle | Service-metric comparable | Raw-reward comparable | Reasons |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in compatibility_rows:
        lines.append(
            f"| `{row['label']}` | `{row['service_metrics_comparable']}` | "
            f"`{row['raw_reward_comparable']}` | "
            f"{', '.join(row['reasons']) or 'ok'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = resolve_repo_root()
    docs = [repo_root / doc for doc in args.docs]

    doc_refs: list[dict[str, Any]] = []
    bundle_roots: dict[str, Path] = {}
    bundle_citations: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        refs = extract_doc_references(doc, repo_root)
        doc_refs.extend(refs)
        for ref in refs:
            bundle_root = ref.get("bundle_root")
            if bundle_root is None:
                continue
            bundle_path = Path(bundle_root)
            bundle_roots[str(bundle_path.resolve())] = bundle_path
            bundle_citations[bundle_path.name].append(ref["doc_path"])

    audited_rows = [
        audit_bundle(normalize_run_ref(str(bundle_path)))
        for bundle_path in sorted(bundle_roots.values())
    ]

    reference_label = choose_reference_label(audited_rows)
    reference_row = None
    if reference_label is not None:
        for row in audited_rows:
            if row["label"] == reference_label:
                reference_row = row
                break

    compatibility_rows: list[dict[str, Any]] = []
    if reference_row is not None:
        compatibility_rows = [
            compare_to_reference(row, reference_row) for row in audited_rows
        ]

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "docs_scanned": [str(doc.resolve()) for doc in docs],
        "doc_references": doc_refs,
        "reference_label": reference_label,
        "audited_rows": audited_rows,
        "compatibility_rows": compatibility_rows,
    }

    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "bundle_inventory.csv"
    md_path = args.output_dir / "report.md"

    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "label",
            "audit_status",
            "recommended_action",
            "cited_by",
            "reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        compatibility_by_label = {row["label"]: row for row in compatibility_rows}
        for row in audited_rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "audit_status": row["audit_status"],
                    "recommended_action": recommended_action(
                        row, compatibility_by_label.get(row["label"])
                    ),
                    "cited_by": ";".join(
                        sorted(bundle_citations.get(row["label"], []))
                    ),
                    "reasons": ";".join(row.get("reasons", [])),
                }
            )
    md_path.write_text(
        build_markdown_report(
            docs=docs,
            doc_refs=doc_refs,
            audited_rows=audited_rows,
            compatibility_rows=compatibility_rows,
            reference_label=reference_label,
            bundle_citations=bundle_citations,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
