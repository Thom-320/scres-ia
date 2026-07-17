#!/usr/bin/env python3
"""Paper 2 master results table — generated from custodied artifacts, never hand-typed.

Every cell of the table carries its source file and the SHA-256 prefix of that file, so a
reviewer (or a later build) can verify each number mechanically. Missing sources fail loudly
with the available keys (A11: no silent defaults). Program Q is emitted as an explicit
placeholder until its terminal artifact exists.

Usage:
  build_results_table.py [--learner-result PATH] [--program-q PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = ["rho75_share90", "rho90_share75", "rho90_share90"]

HPI = ROOT / "results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json"
CORRECTIVE = ROOT / "results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json"
DEFAULT_LEARNER = ROOT / "results/program_o/ret_only_learner_v1/calibration_run/result.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def need(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise SystemExit(f"MISSING FIELD {'/'.join(map(str, keys))}; available: "
                             f"{sorted(cur)[:20] if isinstance(cur, dict) else type(cur)}")
        cur = cur[k]
    return cur


def deep_find(d, want):
    """Find a scalar by key-substring match anywhere (used only for L1 whose schema is verbose)."""
    hits = {}
    def walk(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}/{k}")
        elif isinstance(o, (int, float)) and any(w in p.lower() for w in want):
            hits[p] = o
    walk(d)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-result", type=Path, default=DEFAULT_LEARNER)
    ap.add_argument("--program-q", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()

    rows, sources = [], {}

    # ---- Level 1: physical opportunity -------------------------------------------------------
    hpi = json.loads(HPI.read_text())
    sources["L1"] = f"{HPI.relative_to(ROOT)}@{sha(HPI)}"
    l1 = deep_find(hpi, ("safe_h_pi", "lcb"))
    h_pi = next((v for p, v in l1.items() if "safe_h_pi" in p.lower() and "lcb" not in p.lower()), None)
    h_pi_lcb = next((v for p, v in l1.items() if "lcb" in p.lower()), None)
    if h_pi is None or h_pi_lcb is None:
        raise SystemExit(f"L1 fields not found; candidates: {l1}")
    rows.append(("L1 physical opportunity", "all (safe oracle)",
                 f"H_PI = {h_pi:.5f}", f"LCB95 = {h_pi_lcb:.5f}",
                 "fungible null = 0 (exact)", sources["L1"]))

    # ---- Level 2: observable classical conversion --------------------------------------------
    corr = json.loads(CORRECTIVE.read_text())
    sources["L2"] = f"{CORRECTIVE.relative_to(ROOT)}@{sha(CORRECTIVE)}"
    placebos = need(corr, "simultaneous_inference", "all_27_placebos_pass")
    failed = need(corr, "simultaneous_inference", "failed_guardrails")
    for cell in CELLS:
        lcb = need(corr, "primary", cell, "simultaneous_lcb95")
        fav = need(corr, "primary", cell, "favorable_tapes")
        cvar = next((f"CVaR10 LCB {v['simultaneous_lcb95']:+.4f} (pt {v['estimate']:+.4f})"
                     for k, v in failed.items() if k.startswith(cell)), "CVaR10 gate met")
        rows.append(("L2 classical H_obs", cell, f"LCB95 = {lcb:+.5f}", f"{fav}/48 favorable",
                     f"placebos 27/27={placebos}; {cvar}", sources["L2"]))

    # ---- Levels 3-4: learned adaptation and neural premium -----------------------------------
    lr = json.loads(Path(a.learner_result).read_text())
    sources["L3L4"] = f"{Path(a.learner_result).name}@{sha(Path(a.learner_result))}"
    est = need(lr, "inference", "estimates")
    for cell in CELLS:
        s = need(lr, "cell_summaries", cell)
        hol = est[f"{cell}::H_learned"]
        rows.append(("L3 learned H_OL", cell,
                     f"est {hol['estimate']:+.5f}", f"LCB95 = {hol['lcb95']:+.5f}",
                     f"{s['favorable_tapes_vs_open_loop']}/48 vs 65,536 frontier",
                     sources["L3L4"]))
    for cell in CELLS:
        s = need(lr, "cell_summaries", cell)
        dn = est[f"{cell}::H_neural"]
        rows.append(("L4 neural premium Δ_N", cell,
                     f"est {dn['estimate']:+.5f}", f"LCB95 = {dn['lcb95']:+.5f}",
                     f"{s['positive_learner_seeds_vs_both']}/10 seeds beat both comparators",
                     sources["L3L4"]))

    # ---- Program Q ---------------------------------------------------------------------------
    if a.program_q and Path(a.program_q).exists():
        q = json.loads(Path(a.program_q).read_text())
        rows.append(("Q prospective replication", "all",
                     str(q.get("terminal_outcome") or q.get("status")), "", "",
                     f"{Path(a.program_q).name}@{sha(Path(a.program_q))}"))
    else:
        rows.append(("Q prospective replication", "all", "PENDING",
                     "contract frozen: N=128/cell, block 7490001+",
                     "outcomes: PASS_PREMIUM / PASS_EQUIVALENT / BOUND / STOP", "contract@frozen"))

    # ---- emit --------------------------------------------------------------------------------
    md = ["| Level | Cell | Point | Bound | Integrity / notes | Source |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | `{r[5]}` |" for r in rows]
    out_md = a.out / "results_table.md"
    out_md.write_text("# Paper 2 — master results table (machine-generated; do not edit)\n\n"
                      + "\n".join(md) + "\n")
    (a.out / "results_table.json").write_text(json.dumps(
        {"rows": rows, "sources": sources, "generator": Path(__file__).name}, indent=1))
    print(f"wrote {out_md} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
