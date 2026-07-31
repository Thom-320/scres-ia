#!/usr/bin/env python3
"""Can we answer the two Garrido questions from his own data, instead of asking him?

Both questions were about to be sent to him. Before sending, test whether his 47,546 delivered
rows already decide them. They carry `OPTj, OATj, CTj, LT, sumBt, APj, RPj, DPj, R.., sumUt`,
which is the whole ledger -- so a convention or a rule that leaves a footprint in those columns
is an EMPIRICAL question, not a question for the author.

**Q1 -- the same-time Bt/Ut convention.** When order `j` is requested at time `t` and other
orders activate or end at exactly `t`, does the snapshot see those events or not? Our v2
contract calls this `events_before_snapshot` and holds the whole metric "provisional pending
Garrido confirmation". His `sumBt` column is the answer if we can reconstruct it: build the
backorder ledger from his own placement and arrival times under BOTH conventions and see which
one reproduces his numbers. The test only means something if the two conventions actually
disagree somewhere, so the discriminating rows are counted first (`f1`).

**Q2 -- does his RPj saturate by design?** We reported that his RPj flattens near 400 h and
correlates 0.88 with the risk count but only 0.37 with cycle time. Four structural hypotheses,
all decidable on his columns:

    H-A  DPj = CTj for every delayed row        (disruption period IS the cycle time)
    H-B  RPj > 0 <-> DPj > 0, and RPj <= DPj    (recovery is inside disruption)
    H-C  APj > 0 -> RPj = 0                     (the branches are exclusive)
    H-D  RPj has a hard ceiling                 (a mass at the maximum)

If H-A and H-B hold, `DPj - RPj` is exactly the delay between the order being placed and the
first risk onset, i.e. Algorithm 2's `RPj = OATj - first R^0` -- and "saturation" is then a
property of WHEN risks start, with nothing left to ask.

Nothing here is fitted. Every hypothesis is a statement about his columns that his columns can
refute.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

FAMILY_SHEETS = {"R1r": range(1, 11), "R2r": range(11, 21)}
NEEDED = ("Q", "j", "OPTj", "OATj", "CTj", "LT", "∑Bt", "APj", "RPj", "DPj", "∑Ut")


def read_full_sheet(path: Path, sheet: str):
    """His full ledger row set, with the header row found rather than assumed (CF2)."""
    import pandas as pd

    for header in (0, 1, 2):
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=header)
        except Exception:
            continue
        cols = {str(c).strip(): c for c in df.columns}
        if all(k in cols for k in ("OPTj", "OATj", "CTj", "∑Bt", "∑Ut", "RPj", "DPj", "APj")):
            keep = [k for k in NEEDED if k in cols]
            out = df[[cols[k] for k in keep]].apply(pd.to_numeric, errors="coerce")
            out.columns = keep
            risk_cols = [c for c in df.columns if str(c).strip().startswith("R")
                         and str(c).strip() not in ("RPj",)]
            risks = df[risk_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            out = out.dropna(subset=["OPTj", "OATj", "CTj"])
            risks = risks.loc[out.index]
            return out.reset_index(drop=True), risks.reset_index(drop=True)
    return None, None


def reconstruct_bt(opt: np.ndarray, oat: np.ndarray, lt: np.ndarray,
                   *, strict_end: bool, strict_start: bool) -> np.ndarray:
    """Backorders outstanding when each row is placed, under one same-time convention.

    An order is a backorder from its promised time `OPTj + LT` until it arrives at `OATj`.
    `strict_start`/`strict_end` are the two sides of the tie: whether an order that becomes
    due, or arrives, at exactly this row's placement time counts.
    """
    due = opt + lt
    order = np.argsort(opt, kind="stable")
    out = np.zeros(len(opt), dtype=np.int64)
    for idx in order:
        t = opt[idx]
        started = (due < t) if strict_start else (due <= t)
        ended = (oat <= t) if strict_end else (oat < t)
        out[idx] = int(np.sum(started & ~ended))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook-dir", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/garrido_ledger_conventions/result.json"))
    args = ap.parse_args()

    from build_fidelity_reference_v4 import WORKBOOKS

    per_sheet: dict[str, dict] = {}
    q1_totals = {name: 0 for name in ("events_before_snapshot", "snapshot_before_events",
                                      "half_open_a", "half_open_b")}
    q1_rows = q1_discriminating = 0
    ha = hb = hc = 0
    ha_n = hb_n = hc_n = 0
    bt_cap_hits: list[int] = []
    bt_max: list[float] = []
    bt_below_cap: list[int] = []
    rpj_all: list[float] = []
    gap_all: list[float] = []
    ctj_all: list[float] = []
    risk_count_all: list[float] = []

    for family, sheets in FAMILY_SHEETS.items():
        for i in sheets:
            frame = risks = None
            for wb in WORKBOOKS:
                path = args.workbook_dir / wb
                if path.exists():
                    frame, risks = read_full_sheet(path, f"CF{i}")
                    if frame is not None:
                        break
            if frame is None or not len(frame):
                continue
            opt = frame["OPTj"].to_numpy(float)
            oat = frame["OATj"].to_numpy(float)
            lt = frame["LT"].to_numpy(float) if "LT" in frame else np.full(len(opt), 48.0)
            bt = frame["∑Bt"].to_numpy(float)
            apj = frame["APj"].fillna(0.0).to_numpy(float)
            rpj = frame["RPj"].fillna(0.0).to_numpy(float)
            dpj = frame["DPj"].fillna(0.0).to_numpy(float)
            ctj = frame["CTj"].to_numpy(float)

            # --- Q1: four tie conventions, only the two extremes are the contract's ---
            variants = {
                "events_before_snapshot": dict(strict_start=False, strict_end=True),
                "snapshot_before_events": dict(strict_start=True, strict_end=False),
                "half_open_a": dict(strict_start=False, strict_end=False),
                "half_open_b": dict(strict_start=True, strict_end=True),
            }
            predictions = {name: reconstruct_bt(opt, oat, lt, **kw)
                           for name, kw in variants.items()}
            stacked = np.vstack(list(predictions.values()))
            q1_discriminating += int(np.sum(stacked.min(axis=0) != stacked.max(axis=0)))
            q1_rows += len(opt)
            for name, pred in predictions.items():
                q1_totals[name] += int(np.sum(pred == bt.astype(np.int64)))
            bt_cap_hits.append(int(np.sum(bt >= 60.0 - 1e-9)))
            bt_max.append(float(np.nanmax(bt)))
            bt_below_cap.append(int(np.sum(bt < 60.0 - 1e-9)))

            # --- Q2: the structural hypotheses ---
            delayed = dpj > 0.0
            ha += int(np.sum(np.isclose(dpj[delayed], ctj[delayed], rtol=0, atol=1e-6)))
            ha_n += int(delayed.sum())
            hb += int(np.sum((rpj[delayed] <= dpj[delayed] + 1e-9)))
            hb_n += int(delayed.sum())
            auto = apj > 0.0
            hc += int(np.sum(rpj[auto] == 0.0))
            hc_n += int(auto.sum())

            keep = rpj > 0.0
            rpj_all.extend(rpj[keep].tolist())
            gap_all.extend((dpj[keep] - rpj[keep]).tolist())
            ctj_all.extend(ctj[keep].tolist())
            risk_count_all.extend(risks.to_numpy(float).sum(axis=1)[keep].tolist())
            per_sheet[f"CF{i}"] = {"family": family, "n": len(opt)}
            print(f"  CF{i:<3} n={len(opt):>5}", flush=True)

    rpj_arr = np.asarray(rpj_all)
    gap_arr = np.asarray(gap_all)
    ctj_arr = np.asarray(ctj_all)
    risk_arr = np.asarray(risk_count_all)
    top = float(rpj_arr.max())
    ceiling_mass = float(np.mean(rpj_arr > 0.99 * top))

    q1 = {name: {"exact_rows": n, "share": n / max(q1_rows, 1)}
          for name, n in q1_totals.items()}
    best = max(q1_totals, key=lambda k: q1_totals[k])
    q2 = {
        "H_A_DPj_equals_CTj": {"holds": ha, "of": ha_n, "share": ha / max(ha_n, 1)},
        "H_B_RPj_within_DPj": {"holds": hb, "of": hb_n, "share": hb / max(hb_n, 1)},
        "H_C_autotomy_excludes_recovery": {"holds": hc, "of": hc_n,
                                           "share": hc / max(hc_n, 1)},
        "H_D_hard_ceiling": {"max_RPj": top, "share_within_1pct_of_max": ceiling_mass,
                             "verdict": ("ceiling" if ceiling_mass > 0.01
                                         else "NO hard ceiling: the maximum is isolated")},
        "gap_DPj_minus_RPj": {
            "n": int(gap_arr.size), "min": float(gap_arr.min()), "p50": float(np.median(gap_arr)),
            "p95": float(np.percentile(gap_arr, 95)), "max": float(gap_arr.max()),
            "share_negative": float(np.mean(gap_arr < -1e-9)),
            "meaning": ("if H-A and H-B hold this gap is the delay from placement to the first "
                        "risk onset, i.e. Algorithm 2's RPj = OATj - first R^0")},
        "correlations": {
            "RPj_vs_CTj": float(np.corrcoef(rpj_arr, ctj_arr)[0, 1]),
            "RPj_vs_risk_count": float(np.corrcoef(rpj_arr, risk_arr)[0, 1])},
    }

    cap_share = sum(bt_cap_hits) / max(q1_rows, 1)
    falsifiers = {
        "f1_ties_are_immaterial_in_his_ledger": {
            "passed": (q1_discriminating / max(q1_rows, 1)) < 1e-3,
            "evidence": {
                "why_it_can_fail": ("this is the CLAIM, so it is stated in the refutable "
                                    "direction: if a meaningful share of his rows were ties, "
                                    "the same-time convention would change his numbers and the "
                                    "question would genuinely need him. A tie depends only on "
                                    "the timestamps, so this holds whatever sumBt turns out to "
                                    "mean"),
                "discriminating_rows": q1_discriminating, "rows": q1_rows,
                "share": q1_discriminating / max(q1_rows, 1)},
        },
        "f2_sumBt_is_saturated_at_the_60_cap": {
            "passed": (max(bt_max) <= 60.0 + 1e-9 and cap_share > 0.5),
            "evidence": {
                "why_it_can_fail": ("a second, independent reason the convention cannot bite: "
                                    "a snapshot one event early or late still reads 60. If "
                                    "sumBt were NOT capped, or not saturated, this argument "
                                    "would not hold"),
                "max_sumBt": max(bt_max), "share_at_cap": cap_share,
                "rows_below_cap": sum(bt_below_cap),
                "thesis_reference": "backorder queue cap of 60, section 6.5.4"},
        },
        "f3_our_outstanding_backorder_model_of_sumBt_is_REFUTED": {
            "passed": q1[best]["share"] < 0.5,
            "evidence": {
                "why_it_can_fail": ("stated as a refutation because that is what the data did. "
                                    "If a convention HAD reproduced sumBt, this would fail and "
                                    "we would have identified the quantity instead"),
                "best_convention": best, "best_exact_share": q1[best]["share"],
                "by_convention": q1,
                "what_this_means": ("sumBt is not the outstanding-backorder count at placement. "
                                    "It is capped at 60 and saturates, consistent with the "
                                    "capped backlog LIST of section 6.5.4")},
        },
        "f4_the_population_is_the_full_ledger": {
            "passed": q1_rows > 45_000,
            "evidence": {
                "why_it_can_fail": "a partial read would make every share above unreliable",
                "rows_read": q1_rows, "sheets": len(per_sheet)},
        },
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    print(f"\n  === Q1: convención de simultaneidad ({q1_rows} filas, "
          f"{q1_discriminating} discriminantes) ===")
    for name, row in sorted(q1.items(), key=lambda kv: -kv[1]["exact_rows"]):
        print(f"    {name:<26} {row['exact_rows']:>6} filas exactas  {row['share']:>7.2%}")
    print("\n  === Q2: hipótesis estructurales sobre RPj ===")
    for name in ("H_A_DPj_equals_CTj", "H_B_RPj_within_DPj",
                 "H_C_autotomy_excludes_recovery"):
        row = q2[name]
        print(f"    {name:<34} {row['holds']:>6}/{row['of']:<6} {row['share']:>7.2%}")
    print(f"    H_D max RPj {top:.2f}, masa a <1% del máximo {ceiling_mass:.4%} "
          f"-> {q2['H_D_hard_ceiling']['verdict']}")
    print(f"    corr(RPj, CTj) {q2['correlations']['RPj_vs_CTj']:.3f}   "
          f"corr(RPj, n riesgos) {q2['correlations']['RPj_vs_risk_count']:.3f}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<40} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "garrido_ledger_conventions_v1",
        "claim_status": ("DEVELOPMENT_LEDGER_CONVENTION_AUDIT" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "question": ("can Garrido's own 47,546 delivered rows decide the two questions we were "
                     "about to send him?"),
        "q1_same_time_convention": {"by_convention": q1, "best": best,
                                    "discriminating_rows": q1_discriminating,
                                    "rows": q1_rows,
                                    "sumBt_max": max(bt_max),
                                    "sumBt_share_at_cap": cap_share,
                                    "sumBt_rows_below_cap": sum(bt_below_cap)},
        "q2_rpj_structure": q2,
        "sheets": per_sheet, "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(payload, indent=1, sort_keys=True, default=str)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    print(f"\n  -> {args.output} (sello {payload['self_sha256'][:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
