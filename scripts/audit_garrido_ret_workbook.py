#!/usr/bin/env python3
"""Read-only audit of Garrido-Rios order-level ReT workbook populations.

This utility uses only the Python standard library. It does not recalculate
ReT, modify the workbook, or authorize any scientific execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _column_number(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    value = 0
    for char in letters.upper():
        value = value * 26 + ord(char) - ord("A") + 1
    return value


class WorkbookReader:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.shared_strings = self._load_shared_strings()
        self.sheet_paths = self._load_sheet_paths()

    def close(self) -> None:
        self.archive.close()

    def _load_shared_strings(self) -> list[str]:
        try:
            root = ET.fromstring(self.archive.read("xl/sharedStrings.xml"))
        except KeyError:
            return []
        values = []
        for item in root.findall(f"{{{MAIN_NS}}}si"):
            values.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
        return values

    def _load_sheet_paths(self) -> dict[str, str]:
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        rels = ET.fromstring(self.archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{REL_NS}}}Relationship")
        }
        paths: dict[str, str] = {}
        sheets = workbook.find(f"{{{MAIN_NS}}}sheets")
        if sheets is None:
            raise ValueError("Workbook has no sheets collection")
        for sheet in sheets:
            rel_id = sheet.attrib[f"{{{DOC_REL_NS}}}id"]
            target = targets[rel_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            paths[sheet.attrib["name"]] = target
        return paths

    def numeric_cells(
        self,
        sheet_name: str,
        *,
        min_row: int,
        max_row: int | None,
        min_column: int,
        max_column: int,
    ) -> dict[int, list[float]]:
        path = self.sheet_paths[sheet_name]
        root = ET.fromstring(self.archive.read(path))
        result = {column: [] for column in range(min_column, max_column + 1)}
        sheet_data = root.find(f"{{{MAIN_NS}}}sheetData")
        if sheet_data is None:
            return result
        for row in sheet_data.findall(f"{{{MAIN_NS}}}row"):
            row_number = int(row.attrib["r"])
            if row_number < min_row or (max_row is not None and row_number > max_row):
                continue
            for cell in row.findall(f"{{{MAIN_NS}}}c"):
                column = _column_number(cell.attrib["r"])
                if column not in result:
                    continue
                value_node = cell.find(f"{{{MAIN_NS}}}v")
                if value_node is None or value_node.text is None:
                    continue
                if cell.attrib.get("t") in {"s", "str", "inlineStr"}:
                    continue
                try:
                    value = float(value_node.text)
                except ValueError:
                    continue
                if math.isfinite(value):
                    result[column].append(value)
        return result


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("At least one numeric ReT value is required.")
    tail = [value for value in rows if value >= 0.5]
    above_one = [value for value in rows if value > 1.0]
    without_tail = [value for value in rows if value < 0.5]
    without_above_one = [value for value in rows if value <= 1.0]
    mean = statistics.fmean(rows)
    return {
        "n": len(rows),
        "mean": mean,
        "sample_sd": statistics.stdev(rows) if len(rows) > 1 else 0.0,
        "minimum": min(rows),
        "maximum": max(rows),
        "zero_count": sum(value == 0.0 for value in rows),
        "unique_count": len(set(rows)),
        "tail_ge_0_5_count": len(tail),
        "above_1_count": len(above_one),
        "tail_mean_share": (sum(tail) / len(rows)) / mean if mean else 0.0,
        "mean_clipped_0_1": statistics.fmean(
            min(1.0, max(0.0, value)) for value in rows
        ),
        "mean_excluding_above_1_diagnostic_only": statistics.fmean(
            without_above_one
        ),
        "mean_excluding_tail_diagnostic_only": statistics.fmean(without_tail),
    }


def rank(summaries: dict[str, dict[str, float | int]], field: str) -> list[str]:
    return sorted(summaries, key=lambda name: float(summaries[name][field]), reverse=True)


def audit(path: Path, *, re_data_end_row: int = 2520) -> dict[str, object]:
    reader = WorkbookReader(path)
    try:
        per_cf: dict[str, dict[str, float | int]] = {}
        cf_values: dict[str, list[float]] = {}
        for index in range(1, 13):
            name = f"Cf{index}"
            values = reader.numeric_cells(
                name, min_row=12, max_row=None, min_column=9, max_column=9
            )[9]
            cf_values[name] = values
            per_cf[name] = summarize(values)

        re_columns = reader.numeric_cells(
            "Re",
            min_row=3,
            max_row=re_data_end_row,
            min_column=1,
            max_column=12,
        )
        aggregate_re = {
            f"Cf{index}": summarize(re_columns[index]) for index in range(1, 13)
        }
        population_differences = {}
        for index in range(1, 13):
            name = f"Cf{index}"
            per_counter = Counter(round(value, 12) for value in cf_values[name])
            agg_counter = Counter(round(value, 12) for value in re_columns[index])
            population_differences[name] = {
                "cf_n": len(cf_values[name]),
                "aggregate_re_n": len(re_columns[index]),
                "aggregate_extra_count": sum((agg_counter - per_counter).values()),
                "cf_missing_from_aggregate_count": sum(
                    (per_counter - agg_counter).values()
                ),
            }

        return {
            "schema_version": "garrido_ret_workbook_audit_v1",
            "workbook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scientific_claim_status": "SOURCE_DIAGNOSTIC_NO_POLICY_CLAIM",
            "correction_semantics": {
                "clip_0_1": "range enforcement diagnostic",
                "exclude_above_1": "deletion diagnostic only; not an authorized correction",
                "exclude_tail_ge_0_5": "tail sensitivity only; not an authorized correction",
            },
            "per_cf_sheet": per_cf,
            "aggregate_re_sheet": aggregate_re,
            "population_differences": population_differences,
            "rankings": {
                "observed_mean": rank(per_cf, "mean"),
                "clipped_mean": rank(per_cf, "mean_clipped_0_1"),
                "excluding_above_1_diagnostic_only": rank(
                    per_cf, "mean_excluding_above_1_diagnostic_only"
                ),
                "excluding_tail_diagnostic_only": rank(
                    per_cf, "mean_excluding_tail_diagnostic_only"
                ),
            },
        }
    finally:
        reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--re-data-end-row", type=int, default=2520)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = audit(args.workbook, re_data_end_row=args.re_data_end_row)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
