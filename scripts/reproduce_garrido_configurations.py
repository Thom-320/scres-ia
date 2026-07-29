#!/usr/bin/env python3
"""Reproduce Garrido's Cf1-Cf90 design in our DES, and validate Cf1-Cf20 against his data.

The thesis publishes the complete 90-configuration design (Tables 6.11-6.23) but only two
of the three delivered workbooks are the thesis data, and they cover Cf1-Cf20 only. So:

  * Cf1-Cf20  -- we have his per-order outputs, so these are a VALIDATION target.
  * Cf21-Cf90 -- never delivered. Regenerated here from the published design. Cf31-Cf90 are
                 the buffer and shift scenarios, i.e. hypotheses H2 and H3.

Mapping of the design onto `MFSCSimulation`:

  risk family   only that family's risks are enabled. Confirmed by the workbooks themselves:
                `Raw_data1` CF1 carries columns R11_1, R11_2, R12, R13, R14 and nothing else;
                `Raw_data2` CF11 carries R21_1..R21_5, R22_1..R22_4, R23, R24.
  '+' / '-'     `risk_overrides[risk_id] = "increased"` for '+', base level otherwise.
                `config.py` reproduces Table 6.12 exactly for all nine risks at both levels.
  buffers       Table 6.16 levels via `initial_buffers`; zero in Scenarios I and III.
  shifts        Table 6.20 via `shifts`; S = 1 outside Scenario III.
  horizon       10 or 20 years at 8,064 h/year, per section 6.7.

What a mismatch would mean. Our ReT implementation was previously verified against these
workbooks at the formula level (0/47,546 cell mismatches). A divergence here is therefore
about the DES trajectory, not the metric -- which is exactly the useful thing to learn.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics as st
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.config import INVENTORY_BUFFERS  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.garrido_thesis_design import (  # noqa: E402
    DESIGN, VALIDATABLE, Configuration,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILY_RISKS = {
    "R1r": ("R11", "R12", "R13", "R14"),
    "R2r": ("R21", "R22", "R23", "R24"),
    "R3": ("R3",),
}
# Excel column holding ReT: U (21) in Raw_data1, AA (27) in Raw_data2.
EXCEL = {
    "Raw_data1+Re.xlsx": (range(1, 11), 21),
    "Raw_data2+Re.xlsx": (range(11, 21), 27),
}


def run_configuration(cfg: Configuration, seed: int) -> dict:
    """One full-DES episode for a published configuration."""
    # Scenario II is a REPLENISHMENT POLICY, not an initial endowment. Section 6.7.3:
    # "independently of the occurrence of the above risks, every t = 168, 336, 504, 672,
    # or 1,344 hours, the level of I_tS is replenished in the quantities of raw material
    # and rations indicated in Table 6.16." `_inventory_buffer_replenishment` returns
    # immediately unless `inventory_replenishment_period` is set, so passing only
    # `initial_buffers` measures a one-off injection and not the thesis policy.
    buffers = None
    period = None
    if cfg.buffer_hours:
        lvl = INVENTORY_BUFFERS[cfg.buffer_hours]
        buffers = {"op3_rm": float(lvl["op3_rm"]),
                   "op5_rm": float(lvl["op5_rm"]),
                   "op9_rations": float(lvl["op9_rations"])}
        period = float(cfg.buffer_hours)
    sim = MFSCSimulation(
        shifts=cfg.shifts,
        initial_buffers=buffers,
        inventory_replenishment_period=period,
        seed=seed,
        horizon=cfg.horizon_hours,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILY_RISKS[cfg.risk_family]),
        risk_overrides={r: "increased" for r in cfg.increased_risks},
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )
    sim.step(action=None, step_hours=cfg.horizon_hours)
    m = compute_episode_metrics(sim)
    return {
        "ret_excel": float(m["ret_excel"]),
        "ret_excel_full_ledger": float(m["ret_excel_full_ledger"]),
        "n_orders": float(m["n_orders"]),
        "visible_n": float(m["ret_excel_visible_n"]),
        "omitted_n": float(m["ret_excel_omitted_n"]),
        "flow_fill_rate": float(m["flow_fill_rate"]),
        "lost_orders": float(m["lost_orders"]),
        "delivered_rations": float(m["delivered_rations"]),
    }


def load_thesis_outputs(downloads: Path) -> dict[int, dict]:
    """Per-configuration ReT summary from the two genuine workbooks."""
    import openpyxl
    out: dict[int, dict] = {}
    for fname, (rng, col) in EXCEL.items():
        path = downloads / fname
        if not path.exists():
            print(f"  (missing {fname}; skipping its configurations)")
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for i in rng:
            ws = wb[f"CF{i}"]
            vals = [r[0] for r in ws.iter_rows(min_row=2, min_col=col, max_col=col,
                                               values_only=True)
                    if isinstance(r[0], (int, float))]
            if vals:
                out[i] = {"n": len(vals), "mean": st.mean(vals), "max": max(vals),
                          "n_gt1": sum(1 for v in vals if v > 1.0)}
        wb.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", type=int, default=None,
                    help="subset to run; default is all 90")
    ap.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--fallback-seed-base", type=int, default=900_000,
                    help="seed for configurations whose thesis seed is unknown")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/garrido_reproduction"))
    args = ap.parse_args()

    indices = args.configs or list(range(1, 91))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("loading thesis outputs for the validation set...")
    thesis = load_thesis_outputs(args.downloads)
    print(f"  loaded {len(thesis)} configurations from the genuine workbooks\n")

    started = time.perf_counter()
    rows = []
    print(f"{'Cf':>4} {'hyp':>4} {'fam':>4} {'pat':>5} {'buf':>6} {'S':>2} {'yr':>3} "
          f"{'seed':>7} | {'ours_ReT':>10} {'ours_n':>7} | {'thesis_ReT':>11} "
          f"{'thesis_n':>9} | {'ReT ratio':>10} {'n ratio':>8}")
    for i in indices:
        cfg = DESIGN[i]
        # The fallback MUST key on base_index, not on i: the thesis reuses one seed
        # across each (Cf_b, Cf_b+30, Cf_b+60) triple, and the paired contrasts that
        # identify the buffer and shift effects depend on that reuse.
        seed = (cfg.seed if cfg.seed is not None
                else args.fallback_seed_base + cfg.base_index)
        seed_known = cfg.seed is not None
        r = run_configuration(cfg, seed)
        t = thesis.get(i)
        rr = (r["ret_excel"] / t["mean"]) if t and t["mean"] else None
        nr = (r["n_orders"] / t["n"]) if t and t["n"] else None
        rows.append({"cf": i, "hypothesis": cfg.hypothesis, "family": cfg.risk_family,
                     "pattern": cfg.risk_pattern, "buffer_hours": cfg.buffer_hours,
                     "shifts": cfg.shifts, "horizon_years": cfg.horizon_years,
                     "seed": seed, "seed_is_thesis_seed": seed_known,
                     "ours": r, "thesis": t,
                     "ret_ratio": rr, "n_ratio": nr})
        t_mean = f"{t['mean']:11.6f}" if t else f"{'-':>11}"
        t_n = f"{t['n']:9d}" if t else f"{'-':>9}"
        s_rr = f"{rr:10.3f}" if rr is not None else f"{'-':>10}"
        s_nr = f"{nr:8.3f}" if nr is not None else f"{'-':>8}"
        mark = "" if seed_known else "*"
        print(f"{i:4d} {cfg.hypothesis:>4} {cfg.risk_family:>4} {cfg.risk_pattern:>5} "
              f"{cfg.buffer_hours:6d} {cfg.shifts:2d} {cfg.horizon_years:3d} "
              f"{seed:6d}{mark:1s} | {r['ret_excel']:10.6f} {r['n_orders']:7.0f} | "
              f"{t_mean} {t_n} | {s_rr} {s_nr}", flush=True)

    payload = {
        "schema_version": "garrido_reproduction_v1",
        "claim_status": "DEVELOPMENT_REPRODUCTION_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_source": "WRAP_Theses_Garrido_Rios_2017.pdf Tables 6.11-6.23",
        "validation_source": ["Raw_data1+Re.xlsx", "Raw_data2+Re.xlsx"],
        "validatable_configurations": list(VALIDATABLE),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = args.output_dir / "reproduction.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n* = thesis seed unknown, deterministic fallback used")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
