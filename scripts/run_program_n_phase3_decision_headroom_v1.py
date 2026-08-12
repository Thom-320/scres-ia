#!/usr/bin/env python3
"""How much is the configuration DECISION worth, before asking who takes it better?

Companion to `docs/PREREGISTRO_FASE_3_SURROGATE_ORIENTADO_A_DECISION_2026-08-12.md`.
Written and frozen BEFORE reading Phase 3's verdict.

CLAUDE.md's third rule is measure the headroom before spending on a learner, and Phase 3 skipped
it: it compares surrogates at choosing a buffer without ever asking what choosing is worth. The
smoke run put decision regret at ~1e-4, which is the scale at which this project has repeatedly
found nothing, so the comparison may be a race for a prize that does not exist.

THE ESTIMAND is what CONTEXT-CONDITIONING buys over one fixed buffer:

    H_decision = mean_c [ max_b R(c,b) ]  -  max_b [ mean_c R(c,b) ]
                 \\____ best buffer per cell ____/   \\__ best single buffer for all cells __/

An oracle upper bound: no surrogate can beat it, because it is handed the answer. If H_decision is
immaterial, every Phase 3 arm is competing for nothing and that is the finding, not who won.

THE NULL IS THE ONE THIS REPOSITORY ALREADY USES. H_decision is a mean-of-maxima minus a
max-of-means, so it is positive under pure noise by Jensen's inequality alone. The cell labels are
permuted and the statistic recomputed, which prices exactly that bias. A clairvoyant ceiling in this
tree already died to this null once.

No seed is opened: declared replay of the tapes gate_b_confirmation_v3 consumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from run_cd_surface_prediction_premium import (                                  # noqa: E402
    BUFFER_HOURS, ESCALATIONS, FAMILIES, FAMILY_RISKS, episode)
from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402
from supply_chain.cobb_douglas_resilience import derive_exponents, resilience_index  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK                                   # noqa: E402
from supply_chain.seed_custody import custody_falsifier                          # noqa: E402

CONTRACT = Path("docs/PREREGISTRO_FASE_3_SURROGATE_ORIENTADO_A_DECISION_2026-08-12.md")
OUT = Path("results/program_n/phase3_decision_headroom/result.json")
SEED_BASE, N_SEEDS = 9600001, 8
REPLAY_OF = "program_n_gate_b_confirmation_v3"
#: The bar. 0.01 is the SESOI this project uses for a material effect on a [0,1] endpoint.
MATERIAL_BAR = 0.01
N_PERM = 20_000
PERM_SEED = 20260812


def h_decision(mat: np.ndarray) -> float:
    """mean over cells of the per-cell best, minus the best single buffer shared by all cells."""
    return float(np.nanmean(np.nanmax(mat, axis=1)) - np.nanmax(np.nanmean(mat, axis=0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()
    started = time.perf_counter()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)

    aggs, index = [], []
    for family in FAMILIES:
        for escalation, mult in ESCALATIONS.items():
            for buf in BUFFER_HOURS:
                for seed in seeds:
                    agg, _ = episode(FAMILY_RISKS[family], mult, buf, seed, horizon)
                    aggs.append(agg)
                    index.append((f"{family}|{escalation}", buf, seed))
        print(f"  {family} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    maxima = {v: max(max(a[v] for a in aggs), 1.0 + 1e-9)
              for v in ("zeta", "epsilon", "phi", "tau")}
    maxima["kappa_dot"] = float(len(aggs))
    exps = derive_exponents(maxima)
    total = float(sum(a["kappa"] for a in aggs))
    scale = len(aggs) / total if total > 0 else 1.0
    y = np.array([resilience_index(
        {"zeta": a["zeta"], "epsilon": a["epsilon"], "phi": a["phi"], "tau": a["tau"],
         "kappa_dot": max(a["kappa"] * scale, 1e-9)}, exps)["R_cobb_douglas"] for a in aggs])

    cell_names = sorted({c for c, _, _ in index})
    buf_levels = list(BUFFER_HOURS)
    per_seed = {}
    for s in seeds:
        m = np.full((len(cell_names), len(buf_levels)), np.nan)
        for i, (c, b, sd) in enumerate(index):
            if sd == s:
                m[cell_names.index(c), buf_levels.index(b)] = y[i]
        per_seed[s] = m
    stacked = np.stack([per_seed[s] for s in seeds])             # (seeds, cells, buffers)
    mean_mat = np.nanmean(stacked, axis=0)

    observed = h_decision(mean_mat)
    per_seed_h = [h_decision(per_seed[s]) for s in seeds]
    se = float(np.std(per_seed_h, ddof=1) / np.sqrt(len(per_seed_h)))
    lcb = float(np.mean(per_seed_h) - 2.365 * se)                # t(7), one-sided 95%

    # The Jensen null: permute the CELL labels within each buffer column, which destroys any
    # cell-specific optimum while keeping every marginal intact.
    rng = np.random.default_rng(PERM_SEED)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        shuffled = np.array([mean_mat[rng.permutation(len(cell_names)), j]
                             for j in range(len(buf_levels))]).T
        null[k] = h_decision(shuffled)
    null_mean, null_p95 = float(null.mean()), float(np.quantile(null, 0.95))
    p_value = float((null >= observed).mean())

    best_per_cell = {cell_names[c]: buf_levels[int(np.nanargmax(mean_mat[c]))]
                     for c in range(len(cell_names))}
    global_best = buf_levels[int(np.nanargmax(np.nanmean(mean_mat, axis=0)))]
    within_cell_spread = {cell_names[c]: float(np.nanmax(mean_mat[c]) - np.nanmin(mean_mat[c]))
                          for c in range(len(cell_names))}

    checks = {
        "j1_the_headroom_is_material": F.check(
            observed >= MATERIAL_BAR,
            "an oracle that is HANDED the best buffer for every context must still buy something "
            "worth having; below the bar, no surrogate can matter and Phase 3 is a race for a "
            "prize that does not exist",
            computed_from={"observed": observed, "bar": MATERIAL_BAR}),
        "j2_it_survives_its_own_jensen_null": F.check(
            observed > null_p95,
            "mean-of-maxima minus max-of-means is positive under pure noise by Jensen alone. A "
            "clairvoyant ceiling in this tree already died to exactly this null, so it can fail",
            computed_from={"observed": observed, "null_p95": null_p95, "null_mean": null_mean,
                           "p_value": p_value}),
        "j3_the_optimum_actually_moves": F.check(
            len(set(best_per_cell.values())) > 1,
            "if one buffer is optimal in every cell there is no context-conditioning to buy, "
            "whatever the arithmetic says",
            computed_from={"n_distinct_optima": len(set(best_per_cell.values())),
                           "n_cells": len(cell_names)},
            best_per_cell={k: v for k, v in best_per_cell.items()},
            best_single_buffer=global_best),
        "j4_the_surface_is_not_flat": F.check(
            max(within_cell_spread.values()) >= MATERIAL_BAR,
            "a cell whose seventeen buffers all score the same offers no decision at all; this "
            "separates 'the optimum does not move' from 'nothing moves'",
            computed_from={"max_within_cell_spread": max(within_cell_spread.values()),
                           "bar": MATERIAL_BAR},
            spread_by_cell=within_cell_spread),
    }
    checks["custody"] = custody_falsifier(seeds, replay_of=REPLAY_OF)
    summary = F.summarise(checks)

    if not checks["j4_the_surface_is_not_flat"]["passed"]:
        status = "THE_SURFACE_IS_FLAT_IN_BUFFER_NO_DECISION_EXISTS"
    elif not checks["j3_the_optimum_actually_moves"]["passed"]:
        status = "ONE_BUFFER_IS_OPTIMAL_EVERYWHERE_NO_CONTEXT_CONDITIONING_TO_BUY"
    elif not checks["j2_it_survives_its_own_jensen_null"]["passed"]:
        status = "DECISION_HEADROOM_IS_JENSEN_BIAS"
    elif not checks["j1_the_headroom_is_material"]["passed"]:
        status = "DECISION_HEADROOM_REAL_BUT_IMMATERIAL"
    else:
        status = "DECISION_HEADROOM_IS_MATERIAL"

    payload = {
        "schema_version": "program_n_phase3_decision_headroom_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DECLARED_REPLAY",
        "scope": "ORACLE_UPPER_BOUND_ON_THE_CONFIGURATION_DECISION_NO_SEEDS_NO_LEARNER",
        "endpoint": "H_decision_on_R_cobb_douglas__context_conditioning_over_one_fixed_buffer",
        "seeds": seeds, "material_bar": MATERIAL_BAR,
        "h_decision": observed, "h_decision_per_seed": per_seed_h, "h_decision_lcb95": lcb,
        "jensen_null": {"mean": null_mean, "p95": null_p95, "p_value": p_value, "n_draws": N_PERM},
        "best_buffer_per_cell": best_per_cell, "best_single_buffer": global_best,
        "within_cell_spread": within_cell_spread,
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started, "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/program_n/gate_b_confirmation_v3/result.json"))

    print(f"\nveredicto: {status}\n")
    print(f"  H_decision              {observed:+.6f}   LCB95 {lcb:+.6f}   barra {MATERIAL_BAR}")
    print(f"  nulo de Jensen          media {null_mean:+.6f}  p95 {null_p95:+.6f}  p={p_value:.4f}")
    print(f"  mejor buffer unico      {global_best:.0f} h")
    print(f"  optimos por celda       {sorted(set(best_per_cell.values()))}")
    print(f"  rango dentro de celda   min {min(within_cell_spread.values()):.6f}  "
          f"max {max(within_cell_spread.values()):.6f}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:44} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
