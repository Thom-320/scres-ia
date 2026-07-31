#!/usr/bin/env python3
"""Rebuild the canonical fidelity reference as v4.

Two corrections over v3, both found on 2026-07-31 and both material:

**CF2 was silently dropped.** v3 covers 19 of 20 sheets. `CF2` exists with 4,420 complete
rows and its header sits on the SECOND row, so a fixed `header=0` read returns nothing but
`Unnamed:` columns and the sheet fell out. That is about 20% of the R1r evidence, and every
R1r figure quoted from v3 was computed over 9 of 10 sheets without saying so. v3's own
`change_from_v2` note even mentions CF2 by name, which is how invisible the loss was.

**The population rate used an inconsistent window.** v3 divides the row count by
`max(OPTj)/8064`, but Garrido's sheets already exclude the warm-up -- `min(OPTj)` runs
823-1,225 h -- so the denominator counted a stretch the numerator does not. v4 uses the
SCORED WINDOW, `(max(OPTj) - min(OPTj))/8064`, which is the same convention our own runs
must use. See `contracts/paper_b_v2_amendment_2026-07-31.json` section 2.

Nothing else changes: same six moments, same definitions, same two workbooks. `Rsult_1.xlsx`
stays excluded -- its twelve configurations differ from the raw workbooks by -1,949 to +735
rows and store `Re` as a pasted constant rather than the live formula.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.fidelity_moments import (  # noqa: E402
    FAMILY_SHEETS, HOURS_PER_THESIS_YEAR, MOMENT_NAMES,
)

WORKBOOKS = ("Raw_data1+Re.xlsx", "Raw_data2+Re.xlsx")
NEEDED = ("OPTj", "CTj", "APj", "RPj", "ReT")


def read_sheet(path: Path, sheet: str) -> pd.DataFrame | None:
    """Read a canonical sheet, finding the header row instead of assuming row 0.

    CF2's header is on row 1. Assuming row 0 is what lost it from v3.
    """
    for header in (0, 1, 2):
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=header)
        except Exception:
            continue
        cols = {str(c).strip(): c for c in df.columns}
        if all(k in cols for k in NEEDED):
            out = df[[cols[k] for k in NEEDED]].apply(pd.to_numeric, errors="coerce")
            out.columns = list(NEEDED)
            out.attrs["header_row"] = header
            return out.dropna(subset=["OPTj", "CTj"])
    return None


def moments_for_sheet(d: pd.DataFrame) -> dict[str, float]:
    ret = d["ReT"].dropna().to_numpy(dtype=float)
    apj = d["APj"].fillna(0.0).to_numpy(dtype=float)
    rpj = d["RPj"].fillna(0.0).to_numpy(dtype=float)
    opt = d["OPTj"].to_numpy(dtype=float)
    pos = np.sort(rpj[rpj > 0.0])
    n = len(ret)
    # The SCORED window: his rows already exclude the warm-up, so the denominator must
    # start where the numerator does.
    window_hours = max(float(opt.max() - opt.min()), 1e-9)
    return {
        "autotomy_share": float((apj > 0.0).sum() / n),
        "ret_mean": float(ret.mean()),
        "ret_above_one_share": float((ret > 1.0).sum() / n),
        "rpj_mean": float(pos.mean()) if pos.size else 0.0,
        "rpj_p95": float(np.percentile(pos, 95)) if pos.size else 0.0,
        "scored_orders_per_year": float(n / (window_hours / HOURS_PER_THESIS_YEAR)),
        "_n_rows": float(n),
        "_optj_min": float(opt.min()),
        "_optj_max": float(opt.max()),
        "_window_years": float(window_hours / HOURS_PER_THESIS_YEAR),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook-dir", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v4/result.json"))
    args = ap.parse_args()

    per_sheet: dict[str, dict] = {}
    headers: dict[str, int] = {}
    missing: list[str] = []
    for sheet in [f"CF{i}" for i in range(1, 21)]:
        found = None
        for wb in WORKBOOKS:
            path = args.workbook_dir / wb
            if not path.exists():
                continue
            d = read_sheet(path, sheet)
            if d is not None and len(d):
                found = (d, wb)
                break
        if found is None:
            missing.append(sheet)
            continue
        d, wb = found
        headers[sheet] = int(d.attrs.get("header_row", 0))
        m = moments_for_sheet(d)
        m["_workbook"] = wb
        per_sheet[sheet] = m
        print(f"  {sheet:<6} header={headers[sheet]}  n={int(m['_n_rows']):>5}  "
              f"window={m['_window_years']:.2f}y  ret_mean={m['ret_mean']:.4f}", flush=True)

    reference: dict[str, dict] = {}
    for family, sheets in FAMILY_SHEETS.items():
        present = [s for s in sheets if s in per_sheet]
        if len(present) < 2:
            raise SystemExit(f"{family}: need >= 2 sheets, got {present}")
        reference[family] = {}
        for m in MOMENT_NAMES:
            vals = [per_sheet[s][m] for s in present]
            mean = float(np.mean(vals))
            spread = float(np.std(vals, ddof=1))
            reference[family][m] = {"mean": mean, "spread": spread,
                                    "n_sheets": len(present)}
        reference[family]["_sheets"] = present

    payload = {
        "schema_version": "fidelity_reference_v4",
        "claim_status": "REFERENCE_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": list(WORKBOOKS),
        "excluded_source": (
            "Rsult_1.xlsx -- its twelve configurations differ from the raw workbooks by "
            "-1,949 to +735 rows and store Re as a pasted constant, not the live formula"),
        "ret_column": "ReT",
        "change_from_v3": {
            "cf2_recovered": (
                "v3 covered 19 of 20 sheets. CF2's header is on the SECOND row, so a fixed "
                "header=0 read returned only 'Unnamed:' columns and the sheet fell out "
                "silently -- 4,420 rows, about 20% of the R1r evidence. Every R1r figure "
                "quoted from v3 was computed over 9 of 10 sheets without saying so."),
            "scored_window_denominator": (
                "v3 divided the row count by max(OPTj)/8064, but his sheets already exclude "
                "the warm-up (min(OPTj) is 823-1,225 h), so the denominator counted a "
                "stretch the numerator does not. v4 uses (max - min)/8064, the same "
                "convention our own runs must use."),
            "nothing_else": "same six moments, same definitions, same two workbooks",
        },
        "header_row_by_sheet": headers,
        "sheets_missing": missing,
        "per_sheet": per_sheet,
        "reference_by_family": reference,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print(f"\n  hojas: {len(per_sheet)}/20   faltantes: {missing or 'ninguna'}")
    for family in FAMILY_SHEETS:
        print(f"  {family}: {len(reference[family]['_sheets'])} hojas")
    print(f"\n-> {args.output}  (sello {payload['self_sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
