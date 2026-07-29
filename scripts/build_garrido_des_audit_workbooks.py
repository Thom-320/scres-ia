#!/usr/bin/env python3
"""Package Garrido DES-vs-Excel audit outputs into review workbooks.

This script does not rerun the DES.  It consumes a finished
``replicate_garrido_excel.py`` output directory, especially
``replication_audit.json`` and ``des_order_exports/*.csv``, then writes a
manifest/README and delegates XLSX authoring to the bundled artifact-tool
builder.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_WORKBOOKS = [
    Path.home() / "Downloads" / "Raw_data1+Re.xlsx",
    Path.home() / "Downloads" / "Raw_data2+Re.xlsx",
]
DEFAULT_RSULT_WORKBOOK = Path.home() / "Downloads" / "Rsult_1.xlsx"
DEFAULT_BUNDLED_NODE = (
    Path.home()
    / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
)
DEFAULT_BUNDLED_NODE_MODULES = (
    Path.home()
    / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
)
AUDIT_FORMULA = (
    "IF(AVERAGE(risk_flags)>0, IF(APj>0, APj/LT, 0.5/RPj), "
    "1-((sumBt+sumUt)/j))"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _latest_replication_dir() -> Path:
    candidates = [
        path.parent
        for path in (REPO_ROOT / "outputs" / "audits").glob("*/replication_audit.json")
        if (path.parent / "des_order_exports").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No replication audit directory with replication_audit.json and "
            "des_order_exports was found under outputs/audits."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _git_info() -> dict[str, Any]:
    def run_git(*parts: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *parts],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status_short": run_git("status", "--short"),
    }


def _family_for_cfi(cfi: int) -> str:
    return "R1 / Raw_data1" if cfi <= 10 else "R2 / Raw_data2"


def _cf_label(cfi: int) -> str:
    return f"CF{cfi:02d}"


def _best_rows(replication: dict[str, Any]) -> list[dict[str, Any]]:
    best = replication.get("best_config", {})
    rows = []
    for row in replication.get("rows", []):
        if all(
            str(row.get(key)) == str(best.get(key))
            for key in (
                "demand_source",
                "risk_occurrence_mode",
                "risk_attribution_source",
                "seed_stream_mode",
            )
        ):
            rows.append(row)
    if not rows:
        rows = list(replication.get("rows", []))
    return sorted(rows, key=lambda row: int(row["cfi"]))


def _cf_summary_rows(replication: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _best_rows(replication):
        cfi = int(row["cfi"])
        out.append(
            {
                "CF": _cf_label(cfi),
                "family": row.get("family") or _family_for_cfi(cfi),
                "target_ret": _to_float(row.get("target_ret_excel_mean")),
                "des_ret": _to_float(row.get("sim_ret_excel_formula_mean")),
                "signed_gap": _to_float(row.get("signed_ret_gap")),
                "abs_gap": _to_float(row.get("abs_ret_gap")),
                "target_orders": _to_int(row.get("target_n_orders")),
                "des_orders": _to_int(row.get("sim_n_orders")),
                "order_gap": _to_int(row.get("n_order_gap")),
                "q_max_abs_gap": _to_float(row.get("q_max_abs_gap")),
                "optj_max_abs_gap": _to_float(row.get("optj_max_abs_gap")),
                "branch_gap_pct": _to_float(row.get("max_branch_share_gap_pct")),
                "horizon_hours": _to_float(row.get("horizon_hours")),
                "target_horizon_hours": _to_float(row.get("target_horizon_hours")),
                "workbook_seed": _to_int(row.get("workbook_seed")),
            }
        )
    return out


def _export_path_by_cf(replication_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted((replication_dir / "des_order_exports").glob("CF*.csv")):
        try:
            cfi = int(path.name[2:4])
        except ValueError:
            continue
        out[cfi] = path
    return out


def _risk_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name.startswith("R") and name[1:3].isdigit()]


def _risk_attribution_rows(export_paths: dict[int, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfi, path in sorted(export_paths.items()):
        with path.open(newline="", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            fields = list(reader.fieldnames or [])
            risk_cols = _risk_columns(fields)
            total = 0
            risk_active = 0
            case_counts: dict[str, int] = {}
            risk_nonzero = {risk: 0 for risk in risk_cols}
            for row in reader:
                total += 1
                active = any(_to_float(row.get(risk)) > 0.0 for risk in risk_cols)
                risk_active += int(active)
                case = str(row.get("excel_case", ""))
                case_counts[case] = case_counts.get(case, 0) + 1
                for risk in risk_cols:
                    if _to_float(row.get(risk)) > 0.0:
                        risk_nonzero[risk] += 1
        rows.append(
            {
                "CF": _cf_label(cfi),
                "family": _family_for_cfi(cfi),
                "n_orders": total,
                "risk_columns": ", ".join(risk_cols),
                "risk_active_share": risk_active / total if total else 0.0,
                "fill_rate_branch_share": case_counts.get("excel_fill_rate", 0)
                / total
                if total
                else 0.0,
                "autotomy_branch_share": case_counts.get("excel_autotomy", 0) / total
                if total
                else 0.0,
                "recovery_branch_share": case_counts.get("excel_recovery", 0) / total
                if total
                else 0.0,
                "unfulfilled_branch_share": case_counts.get("excel_unfulfilled", 0)
                / total
                if total
                else 0.0,
                "top_risk_columns": ", ".join(
                    risk
                    for risk, _count in sorted(
                        risk_nonzero.items(), key=lambda item: item[1], reverse=True
                    )[:4]
                ),
            }
        )
    return rows


def _delta_rows(cf_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in cf_rows:
        flags: list[str] = []
        if abs(float(row["signed_gap"])) > 0.01:
            flags.append("ReT gap > 0.01")
        if int(row["order_gap"]) != 0:
            flags.append("order count mismatch")
        if float(row["branch_gap_pct"]) > 5.0:
            flags.append("branch share gap > 5pp")
        if float(row["q_max_abs_gap"]) > 1e-9:
            flags.append("Q mismatch")
        if float(row["optj_max_abs_gap"]) > 1e-9:
            flags.append("OPTj mismatch")
        out.append({**row, "audit_flags": "; ".join(flags) if flags else "OK"})
    return out


def _ledger_sample_rows(
    export_paths: dict[int, Path], selected_cfs: list[int], sample_rows_per_cf: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cfi in selected_cfs:
        path = export_paths.get(cfi)
        if path is None:
            continue
        for row in _read_csv(path, limit=sample_rows_per_cf):
            out.append({"CF": _cf_label(cfi), **row})
    return out


def _rows_for_ledgers(export_paths: dict[int, Path]) -> dict[str, list[dict[str, Any]]]:
    return {_cf_label(cfi): _read_csv(path) for cfi, path in sorted(export_paths.items())}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_readme(path: Path, payload: dict[str, Any]) -> None:
    manifest = payload["manifest"]
    best = payload["replication"]["best_config"]
    text = f"""# Garrido DES-vs-Excel Audit Package

Generated: {manifest["generated_at_utc"]}

This package makes the DES-vs-Excel replication auditable without reading loose
logs.  It uses the already-generated DES order exports from:

`{manifest["replication_dir"]}`

## Inputs

- Raw Excel workbooks: {", ".join(manifest["raw_workbooks"])}
- Rsult workbook: {manifest["rsult_workbook"]}
- Replication audit JSON: {manifest["replication_audit_json"]}

## Formula

`{AUDIT_FORMULA}`

## Best replay configuration

- demand source: `{best.get("demand_source")}`
- risk occurrence: `{best.get("risk_occurrence_mode")}`
- risk attribution: `{best.get("risk_attribution_source")}`
- seed stream: `{best.get("seed_stream_mode")}`

## Artifacts

- `garrido_des_audit_summary.xlsx`: workbook summary and selected ledgers.
- `garrido_des_ledgers.xlsx`: Garrido-style DES ledger sheets for CF01-CF20.
- `audit_manifest.json`: machine-readable manifest and compact data.

## Git

- branch: `{manifest["git"].get("branch")}`
- commit: `{manifest["git"].get("commit")}`

"""
    path.write_text(text, encoding="utf-8")


def _invoke_node_builder(
    *,
    node_bin: Path,
    node_modules: Path,
    builder: Path,
    data_json: Path,
    output_dir: Path,
) -> None:
    if not node_bin.exists():
        raise FileNotFoundError(f"Node executable not found: {node_bin}")
    if not node_modules.exists():
        raise FileNotFoundError(f"Node modules directory not found: {node_modules}")
    scratch = output_dir / "_node_runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    symlink = scratch / "node_modules"
    if symlink.exists() or symlink.is_symlink():
        if symlink.resolve() != node_modules.resolve():
            symlink.unlink()
    if not symlink.exists():
        symlink.symlink_to(node_modules, target_is_directory=True)
    scratch_builder = scratch / builder.name
    shutil.copy2(builder, scratch_builder)
    env = dict(os.environ)
    env["NODE_PATH"] = str(node_modules)
    subprocess.run(
        [
            str(node_bin),
            str(scratch_builder),
            "--data-json",
            str(data_json),
            "--output-dir",
            str(output_dir),
        ],
        cwd=scratch,
        env=env,
        check=True,
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    replication_dir = args.replication_dir or _latest_replication_dir()
    replication_dir = replication_dir.resolve()
    replication_json = replication_dir / "replication_audit.json"
    if not replication_json.exists():
        raise FileNotFoundError(f"Missing replication audit JSON: {replication_json}")
    export_paths = _export_path_by_cf(replication_dir)
    if not export_paths:
        raise FileNotFoundError(f"No CF CSV exports found under {replication_dir}")
    replication = _read_json(replication_json)
    cf_rows = _cf_summary_rows(replication)
    risk_rows = _risk_attribution_rows(export_paths)
    delta_rows = _delta_rows(cf_rows)
    selected_cfs = [
        int(part.strip())
        for part in args.selected_cfs.split(",")
        if part.strip()
    ]
    selected_ledgers = _ledger_sample_rows(
        export_paths, selected_cfs, int(args.sample_rows_per_cf)
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "replication_dir": str(replication_dir),
        "replication_audit_json": str(replication_json),
        "raw_workbooks": [str(path) for path in args.raw_workbooks],
        "rsult_workbook": str(args.rsult_workbook),
        "git": _git_info(),
        "selected_cfs": selected_cfs,
        "sample_rows_per_cf": int(args.sample_rows_per_cf),
        "workbook_builder": str(Path(__file__).with_suffix(".mjs")),
    }
    compact_replication = {
        "formula": replication.get("formula", AUDIT_FORMULA),
        "formula_audit": replication.get("formula_audit", {}),
        "gates": replication.get("gates", {}),
        "replication_status": replication.get("replication_status"),
        "best_summary": replication.get("best_summary", {}),
        "best_config": replication.get("best_config", {}),
        "blockers": replication.get("blockers", []),
    }
    return {
        "manifest": manifest,
        "replication": compact_replication,
        "cf_summary": cf_rows,
        "risk_attribution": risk_rows,
        "deltas": delta_rows,
        "selected_ledgers": selected_ledgers,
        "ledger_rows_by_cf": _rows_for_ledgers(export_paths),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-dir", type=Path, default=None)
    parser.add_argument("--raw-workbooks", nargs="*", type=Path, default=DEFAULT_RAW_WORKBOOKS)
    parser.add_argument("--rsult-workbook", type=Path, default=DEFAULT_RSULT_WORKBOOK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_des_excel_audit_package"),
    )
    parser.add_argument("--selected-cfs", default="1,11,20")
    parser.add_argument("--sample-rows-per-cf", type=int, default=250)
    parser.add_argument("--node-bin", type=Path, default=DEFAULT_BUNDLED_NODE)
    parser.add_argument("--node-modules", type=Path, default=DEFAULT_BUNDLED_NODE_MODULES)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    manifest_path = output_dir / "audit_manifest.json"
    data_path = output_dir / "_audit_workbook_data.json"
    readme_path = output_dir / "README_AUDIT.md"
    _write_json(data_path, payload)
    _write_json(manifest_path, {k: v for k, v in payload.items() if k != "ledger_rows_by_cf"})
    _write_readme(readme_path, payload)
    if not bool(args.prepare_only):
        _invoke_node_builder(
            node_bin=args.node_bin,
            node_modules=args.node_modules,
            builder=Path(__file__).with_suffix(".mjs"),
            data_json=data_path,
            output_dir=output_dir,
        )
    print(json.dumps({
        "output_dir": str(output_dir),
        "summary_workbook": str(output_dir / "garrido_des_audit_summary.xlsx"),
        "ledger_workbook": str(output_dir / "garrido_des_ledgers.xlsx"),
        "manifest": str(manifest_path),
        "readme": str(readme_path),
        "replication_status": payload["replication"].get("replication_status"),
        "n_cf": len(payload["cf_summary"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
