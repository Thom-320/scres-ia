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
DEFAULT_Q_DIR = ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation"


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
    # EXACT fields only. A fuzzy 'first lcb match' previously grabbed simultaneous_RAW_lcb95
    # (0.11652) instead of the safe companion (0.11562) -- fuzzy key matching IS inferring
    # from labels, the exact failure class A11 forbids.
    h_pi = need(hpi, "primary", "safe_h_pi")
    h_pi_lcb = need(hpi, "primary", "simultaneous_safe_lcb95")
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
    qdir = Path(a.program_q) if a.program_q else DEFAULT_Q_DIR
    if qdir.exists():
        q_res_p = qdir / "evaluation/result.json"
        q_adj_p = qdir / "adjudication.json"
        q_res = json.loads(q_res_p.read_text())
        q_adj = json.loads(q_adj_p.read_text())
        src_res = f"{q_res_p.relative_to(ROOT)}@{sha(q_res_p)}"
        src_adj = f"{q_adj_p.relative_to(ROOT)}@{sha(q_adj_p)}"
        sources["Q"] = src_res
        q_est = need(q_res, "inference", "estimates")
        q_guard = need(q_res, "guardrail_inference", "estimates")
        q_n = need(q_res, "N")
        for cell in CELLS:
            e = need(q_est, f"{cell}::H_OL")
            s_ = need(q_res, "cell_summaries", cell)
            fav = need(s_, "favorable_tapes_fraction_vs_open_loop")
            seeds = need(s_, "positive_learner_seeds_H_OL")
            rows.append(("Q replicated H_OL", cell,
                         f"est {e['point']:+.5f}", f"LCB95 = {e['lcb95']:+.5f}",
                         f"{fav:.1%} of {q_n} tapes favorable; {seeds}/10 seeds positive",
                         src_res))
        for cell in CELLS:
            e = need(q_est, f"{cell}::Delta_N")
            rows.append(("Q neural relation Δ_N", cell,
                         f"est {e['point']:+.5f}",
                         f"CI95 [{e['lcb95']:+.5f}, {e['ucb95']:+.5f}]",
                         "TOST equivalence bar: CI ⊂ [−0.01, +0.01]", src_res))
        for cell in CELLS:
            g = need(q_guard, f"{cell}::worst_product_fill::vs_classical")
            rows.append(("Q guardrail worst-product fill (vs classical)", cell,
                         f"est {g['point']:+.5f}", f"LCB95 = {g['lcb95']:+.5f}",
                         "frozen Class-B margin −0.02 (binding)", src_res))
        rows.append(("Q terminal adjudication", "all",
                     str(need(q_adj, "verdict")),
                     f"N={q_n}/cell, seeds {need(q_res, 'seed_range')[0]}–{need(q_res, 'seed_range')[1]}",
                     "compound label; components above reported separately as preregistered",
                     src_adj))
    else:
        rows.append(("Q prospective replication", "all", "PENDING",
                     "contract frozen: N=256/cell, block 7490001–7490256",
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
