#!/usr/bin/env python3
"""Audit whether the cheap continuous sweet spot contradicts Garrido.

This is a read-only synthesis gate.  It consumes the thesis-faithful static
panel (18 Garrido policies) and the workbook-fidelity extraction artifacts, and
answers a narrow question:

    Does the Python DES already contradict Garrido's "more inventory/capacity
    helps" result under the thesis-like discrete policy set, or is the cheap
    f0.10_S1 finding specific to the later continuous/war extension?

The output is intentionally prose-heavy because it is meant to be sent to the
paper/advisor thread as a claim-boundary note.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]
SHIFT_LEVELS = [1, 2, 3]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def policy_rows(rows: list[dict[str, str]], regime: str, shifts: int) -> list[dict[str, str]]:
    return sorted(
        [r for r in rows if r["regime"] == regime and int(r["shifts"]) == shifts],
        key=lambda r: int(r["inventory"]),
    )


def inventory_profile(rows: list[dict[str, str]], regime: str, shifts: int) -> dict[str, Any]:
    rr = policy_rows(rows, regime, shifts)
    by_inv = {int(r["inventory"]): r for r in rr}
    ret_by_inv = {inv: as_float(by_inv[inv], "ret_excel") for inv in INVENTORY_LEVELS}
    fill_by_inv = {inv: as_float(by_inv[inv], "flow_fill_rate") for inv in INVENTORY_LEVELS}
    lost_by_inv = {inv: as_float(by_inv[inv], "lost_rate") for inv in INVENTORY_LEVELS}
    best_inv = max(INVENTORY_LEVELS, key=lambda inv: ret_by_inv[inv])
    gain_i168 = ret_by_inv[168] - ret_by_inv[0]
    best_gain = ret_by_inv[best_inv] - ret_by_inv[0]
    capture = (gain_i168 / best_gain) if best_gain > 0 else 1.0
    post_buffer_values = [ret_by_inv[inv] for inv in INVENTORY_LEVELS if inv >= 168]
    return {
        "regime": regime,
        "shifts": shifts,
        "ret_by_inventory": ret_by_inv,
        "fill_by_inventory": fill_by_inv,
        "lost_by_inventory": lost_by_inv,
        "best_inventory": best_inv,
        "gain_i168_vs_i0": gain_i168,
        "best_gain_vs_i0": best_gain,
        "i168_gain_capture": capture,
        "post_i168_ret_range": max(post_buffer_values) - min(post_buffer_values),
        "strictly_monotone_ret": all(
            ret_by_inv[a] <= ret_by_inv[b] + 1e-12
            for a, b in zip(INVENTORY_LEVELS, INVENTORY_LEVELS[1:])
        ),
    }


def shift_profile(rows: list[dict[str, str]], regime: str, inventory: int) -> dict[str, Any]:
    rr = sorted(
        [r for r in rows if r["regime"] == regime and int(r["inventory"]) == inventory],
        key=lambda r: int(r["shifts"]),
    )
    ret_by_shift = {int(r["shifts"]): as_float(r, "ret_excel") for r in rr}
    flow_by_shift = {int(r["shifts"]): as_float(r, "flow_fill_rate") for r in rr}
    lost_by_shift = {int(r["shifts"]): as_float(r, "lost_rate") for r in rr}
    best_shift = max(SHIFT_LEVELS, key=lambda s: ret_by_shift[s])
    return {
        "regime": regime,
        "inventory": inventory,
        "ret_by_shift": ret_by_shift,
        "flow_by_shift": flow_by_shift,
        "lost_by_shift": lost_by_shift,
        "best_shift": best_shift,
        "s3_minus_s1_ret": ret_by_shift[3] - ret_by_shift[1],
        "strictly_monotone_ret": all(
            ret_by_shift[a] <= ret_by_shift[b] + 1e-12
            for a, b in zip(SHIFT_LEVELS, SHIFT_LEVELS[1:])
        ),
    }


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for regime in sorted({r["regime"] for r in rows}):
        rr = [r for r in rows if r["regime"] == regime]
        best = max(rr, key=lambda r: as_float(r, "ret_excel"))
        out.append(
            {
                "regime": regime,
                "policy": best["policy"],
                "ret_excel": as_float(best, "ret_excel"),
                "flow_fill_rate": as_float(best, "flow_fill_rate"),
                "lost_rate": as_float(best, "lost_rate"),
                "strategic_buffer_units": as_float(best, "strategic_buffer_units"),
            }
        )
    return out


def load_workbook_roles(audit_dir: Path) -> dict[str, Any]:
    raw_path = audit_dir / "raw_cf_summary.csv"
    rsult_path = audit_dir / "rsult_summary.csv"
    inventory_path = audit_dir / "workbook_inventory.csv"
    roles: dict[str, Any] = {}
    if raw_path.exists():
        raw = read_csv(raw_path)
        roles["raw_workbooks"] = {
            "rows": len(raw),
            "cf_range": [min(int(r["cfi"]) for r in raw), max(int(r["cfi"]) for r in raw)],
            "recomputed_mismatches": sum(int(float(r["recomputed_mismatches"])) for r in raw),
            "max_recomputed_gap": max(float(r["recomputed_max_abs_gap"]) for r in raw),
        }
    if rsult_path.exists():
        rs = read_csv(rsult_path)
        roles["rsult_1"] = {
            "rows": len(rs),
            "sheets": sorted({r["sheet"] for r in rs}),
            "role": "secondary aggregate/distribution workbook",
        }
    if inventory_path.exists():
        inv = read_csv(inventory_path)
        roles["formula_examples"] = [
            {
                "workbook": r["workbook"],
                "sheet": r["sheet"],
                "first_formula_cell": r["first_formula_cell"],
                "first_formula": r["first_formula"],
            }
            for r in inv
            if r.get("workbook", "").startswith("Raw_data")
        ][:2]
    return roles


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Garrido Sweet-Spot Fidelity Audit",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Question",
        "",
        (
            "Does the cheap continuous `f0.10_S1` sweet spot mean Garrido missed an "
            "obvious optimum, or does it signal that our later continuous/war lane "
            "is outside the thesis-faithful regime?"
        ),
        "",
        "## Verdict",
        "",
        (
            "**It is a red flag worth auditing, but it is not yet evidence that "
            "Garrido missed a simple optimum.** In the thesis-faithful 18-policy "
            "panel, inventory still helps strongly from `I0` to the first thesis "
            "buffer level, and high-buffer policies remain among the best under "
            "`increased` and `severe` regimes. The surprising `f0.10_S1` result "
            "belongs to the later continuous war-extension/dense-frontier lane, "
            "not to Garrido's original discrete design."
        ),
        "",
        "The genuine discrepancy is subtler: after `I168`, the DES often saturates "
        "or becomes nearly flat. That means the Python DES reproduces a **buffer "
        "benefit**, but not a clean strict-monotone 'more is always better' curve "
        "at every level. This is exactly where the next fidelity check should focus.",
        "",
        "## Thesis-Faithful 18-Policy Panel",
        "",
        "| regime | best ReT policy | ReT | flow | lost | buffer units |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["best_by_regime"]:
        lines.append(
            f"| {row['regime']} | `{row['policy']}` | {row['ret_excel']:.6f} | "
            f"{row['flow_fill_rate']:.3f} | {row['lost_rate']:.3f} | "
            f"{row['strategic_buffer_units']:.0f} |"
        )
    lines += [
        "",
        "### Inventory Effect At S1",
        "",
        "| regime | I0 ReT | I168 ReT | best inventory | best ReT | I168 captures best gain | post-I168 range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for prof in [p for p in payload["inventory_profiles"] if p["shifts"] == 1]:
        ret = prof["ret_by_inventory"]
        lines.append(
            f"| {prof['regime']} | {ret[0]:.6f} | {ret[168]:.6f} | "
            f"I{prof['best_inventory']} | {ret[prof['best_inventory']]:.6f} | "
            f"{pct(prof['i168_gain_capture'])} | {prof['post_i168_ret_range']:.6f} |"
        )
    lines += [
        "",
        "Interpretation: the first thesis buffer level removes most of the damage. "
        "That makes a low continuous buffer plausible, but Garrido did not test "
        "`f0.05/f0.10/f0.15`; his inventory levels start at `I168` and then jump "
        "through larger discrete stocks.",
        "",
        "### Shift Effect At I0 And I1344",
        "",
        "| regime | inventory | best shift | S3-S1 ReT | monotone? |",
        "|---|---:|---:|---:|---:|",
    ]
    for prof in payload["shift_profiles"]:
        lines.append(
            f"| {prof['regime']} | I{prof['inventory']} | S{prof['best_shift']} | "
            f"{prof['s3_minus_s1_ret']:.6f} | {str(prof['strictly_monotone_ret']).lower()} |"
        )
    lines += [
        "",
        "Shift is not a universally clean lever in this DES: it helps in some "
        "regime/inventory combinations, but can flatten or hurt once the downstream "
        "distribution cap is binding. That is compatible with our bottleneck audit, "
        "and it is also the part that deserves a Garrido-facing question.",
        "",
        "## Workbook Scope",
        "",
        "- `Raw_data1+Re.xlsx` is the order-level source for `CF1`-`CF10`.",
        "- `Raw_data2+Re.xlsx` is the order-level source for `CF11`-`CF20`.",
        "- `Rsult_1.xlsx` is a secondary aggregate/distribution workbook, not the same kind of replay ledger.",
    ]
    wb = payload.get("workbook_roles", {})
    if wb.get("raw_workbooks"):
        raw = wb["raw_workbooks"]
        lines.append(
            f"- Prior formula gate audited raw CF{raw['cf_range'][0]}-CF{raw['cf_range'][1]} "
            f"with `{raw['recomputed_mismatches']}` formula mismatches and max gap "
            f"`{raw['max_recomputed_gap']}`."
        )
    lines += [
        "",
        "## What This Means",
        "",
        "1. Garrido probably did **not** miss `f0.10_S1` because his design was not a dense continuous resource optimization; it used discrete thesis inventory levels and separate hypothesis tests.",
        "2. Our cheap continuous sweet spot is still interesting, but it must be presented as a continuous-extension/frontier finding, not as a thesis-reproduction fact.",
        "3. If the same low-buffer dominance appears under Garrido's exact `CF31`-`CF90` inventory/capacity design rows, then we have a fidelity bug or a metric mismatch. That is the next hard gate.",
        "",
        "## Recommended Next Gate",
        "",
        "Run the actual thesis factorial rows `CF31`-`CF90` against the DES and compare them to `Raw_data1+Re`, `Raw_data2+Re`, and `Rsult_1` aggregates by family. The pass criterion is not that every level is strictly monotone, but that the broad H2/H3 direction from the thesis is reproduced under the same scenario rows.",
    ]
    if payload.get("cf31_90_gate"):
        gate = payload["cf31_90_gate"]
        lines += [
            "",
            "## CF31-CF90 Smoke Gate",
            "",
            (
                f"Ran `{gate['run_count']}` thesis factorial rows from "
                f"`{gate['summary_csv']}`. A naive Spearman trend over these rows is "
                f"`{gate['inventory_spearman']:.3f}` for inventory and "
                f"`{gate['capacity_spearman']:.3f}` for shifts."
            ),
            "",
            (
                "Interpretation: this smoke is **not** a valid monotonicity verdict by itself. "
                "CF31-CF90 are sampled design rows tied to different source risk scenarios, not "
                "paired same-risk comparisons across all inventory/shift levels. The near-zero "
                "trend mainly says that risk family/source dominates a naive pooled correlation. "
                "The proper fidelity gate is to reproduce Garrido's H2/H3 statistical contrast, "
                "or to build paired CRN counterfactuals per source risk row."
            ),
        ]
    (out_dir / "GARRIDO_SWEET_SPOT_FIDELITY_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    import math

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den_x * den_y) if den_x and den_y else float("nan")


def load_cf31_90_gate(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    rows = read_csv(path)
    inventory_rows = [r for r in rows if r["family"] == "inventory"]
    capacity_rows = [r for r in rows if r["family"] == "capacity"]
    inventory_x = [float(r["inventory_replenishment_period"]) for r in inventory_rows]
    inventory_y = [float(r["mean_ret"]) for r in inventory_rows]
    capacity_x = [float(r["shifts"]) for r in capacity_rows]
    capacity_y = [float(r["mean_ret"]) for r in capacity_rows]
    return {
        "summary_csv": str(path),
        "run_count": len(rows),
        "inventory_spearman": spearman(inventory_x, inventory_y),
        "capacity_spearman": spearman(capacity_x, capacity_y),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-panel",
        type=Path,
        default=Path("outputs/experiments/garrido_static_full_panel_2026-06-26/panel.csv"),
    )
    parser.add_argument(
        "--workbook-audit-dir",
        type=Path,
        default=Path("outputs/audits/garrido_workbook_fidelity_2026-06-26"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/audits/garrido_sweet_spot_fidelity_2026-06-28"),
    )
    parser.add_argument(
        "--cf31-90-summary",
        type=Path,
        default=Path(
            "outputs/thesis_faithful/factorial/"
            "cf31_90_monotonicity_gate_2026_06_28/summary.csv"
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.static_panel)
    regimes = sorted({r["regime"] for r in rows})
    inventory_profiles = [
        inventory_profile(rows, regime, shifts)
        for regime in regimes
        for shifts in SHIFT_LEVELS
    ]
    shift_profiles = [
        shift_profile(rows, regime, inventory)
        for regime in regimes
        for inventory in (0, 1344)
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "static_panel": str(args.static_panel),
        "workbook_audit_dir": str(args.workbook_audit_dir),
        "best_by_regime": best_rows(rows),
        "inventory_profiles": inventory_profiles,
        "shift_profiles": shift_profiles,
        "workbook_roles": load_workbook_roles(args.workbook_audit_dir),
        "cf31_90_gate": load_cf31_90_gate(args.cf31_90_summary),
        "claim_boundary": {
            "f010_s1_is_thesis_reproduction": False,
            "cheap_sweet_spot_requires_continuous_extension": True,
            "needs_cf31_cf90_factorial_gate": True,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "best_by_regime": payload["best_by_regime"],
        "needs_cf31_cf90_factorial_gate": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
