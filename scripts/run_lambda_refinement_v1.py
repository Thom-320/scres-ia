#!/usr/bin/env python3
"""Where exactly does the timing band open and close? A dense lambda sweep on the sealed matrices.

Contract: `docs/ENMIENDA_COSTE_BUFFER_DECLARADO_2026-08-08.md`. Reads
`results/priced_clairvoyant_ceiling/result.json` and re-prices it; NO new episodes are simulated,
because `J(lambda) = L* + lambda * inventory_hours / max_inventory_hours` is a pure function of the
two matrices that artifact already sealed. Re-running the DES would produce the identical numbers
and burn compute to look diligent.

WHAT THE COARSE SWEEP FOUND. The gap was exactly zero at lambda 0, 1, 2 and 4 and strictly positive
only at 0.5, where LCB95 was +0.000724 with UCB95 0.011908 -- straddling the 0.01 bar, hence
INCONCLUSIVE. Free holding makes the longest schedule win everywhere; expensive holding makes never
holding win everywhere; only between them does the tape decide.

WHAT THIS CAN AND CANNOT ADD. It can locate the band's edges and its peak precisely, because those
are properties of the same six test tapes read at more prices. It CANNOT narrow the intervals: the
precision limit is six test tapes, and no amount of lambda resolution changes that. Stating this
before the numbers is the point -- a denser grid looks like more evidence and is not.

MULTIPLICITY IS PAID. Sweeping many prices and reporting the best one is the metric shopping this
project has already measured; the Holm-adjusted result is what decides, and the full profile is
reported either way.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.falsifiers import (  # noqa: E402
    disclosure, ge, gt, not_applicable, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

DELTA = 0.01
N_BOOT = 4_000
N_PLACEBO = 400
SOURCE = Path("results/priced_clairvoyant_ceiling/result.json")
MODULES = ("supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=31)
    ap.add_argument("--output", type=Path,
                    default=Path("results/lambda_refinement/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)

    src = json.loads(args.source.read_text())
    L = np.asarray(src["L_matrix"], dtype=float)
    IH = np.asarray(src["inventory_hours_matrix"], dtype=float)
    max_ih = float(src["max_inventory_hours"])
    seeds = list(src["design"]["seeds"]) if "design" in src else list(src["splits"]["train"]) + \
        list(src["splits"]["test"])
    train_seeds = list(src["splits"]["train"])
    test_seeds = list(src["splits"]["test"])
    all_seeds = train_seeds + test_seeds
    tr = [all_seeds.index(s) for s in train_seeds]
    te = [all_seeds.index(s) for s in test_seeds]
    opts = [tuple(o) for o in src["options"]]
    grid = [round(float(x), 6) for x in np.linspace(args.lo, args.hi, args.steps)]
    print(f"  {len(grid)} precios entre {args.lo} y {args.hi} sobre la matriz sellada "
          f"{L.shape[0]}x{L.shape[1]} · sin episodios nuevos")

    rows = {}
    for lam in grid:
        J = L + lam * (IH / max_ih)
        fixed = int(np.argmin(J[tr].mean(axis=0)))          # selected on TRAIN only
        open_loop = J[te, fixed]
        clair = J[te].min(axis=1)
        d = open_loop - clair
        boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                         for _ in range(N_BOOT)])
        plac = np.array([float(np.mean(J[te, rng.integers(0, len(opts), len(te))]))
                         for _ in range(N_PLACEBO)])
        rows[str(lam)] = {
            "lambda": lam, "fixed_option": list(opts[fixed]),
            "open_loop_J": float(open_loop.mean()), "clairvoyant_J": float(clair.mean()),
            "gap_mean": float(d.mean()),
            "lcb95": float(np.percentile(boot, 2.5)),
            "ucb95": float(np.percentile(boot, 97.5)),
            "p_not_above_delta": float(np.mean(boot <= DELTA)),
            "clairvoyant_beats_placebo": bool(float(clair.mean()) < float(plac.mean())),
            "unique_per_tape_optima": int(len(set(J[te].argmin(axis=1).tolist()))),
        }

    # HOLM OVER THE WHOLE SWEEP. Reporting the best price out of 31 without paying multiplicity is
    # the metric shopping this project already measured and priced.
    ks = list(rows)
    order = sorted(range(len(ks)), key=lambda i: rows[ks[i]]["p_not_above_delta"])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(ks) - rank) * rows[ks[idx]]["p_not_above_delta"]))
        rows[ks[idx]]["holm_adjusted_p"] = running

    detectable = [k for k, v in rows.items() if v["lcb95"] > 0.0]
    established = [k for k, v in rows.items()
                   if v["lcb95"] >= DELTA and v["holm_adjusted_p"] < 0.05]
    absent = [k for k, v in rows.items() if v["ucb95"] < DELTA]
    peak = max(rows, key=lambda k: rows[k]["gap_mean"])
    band = ([float(rows[k]["lambda"]) for k in detectable] or [])

    falsifiers = {
        "f1_no_new_episodes_were_simulated": ge(
            float(L.shape[0] * L.shape[1]), float(len(all_seeds) * len(opts)),
            "if this had re-simulated instead of re-pricing, the numbers would be a second sample "
            "presented as a refinement of the first; J is a pure function of the sealed matrices",
            matrix_shape=list(L.shape), source=str(args.source),
            source_sha=src.get("self_sha256")),
        "f2_open_loop_still_selected_on_train_only": ge(
            1 - len(set(train_seeds) & set(test_seeds)), 1,
            "selecting the comparator on the tapes it is scored against shrinks the gap it is "
            "meant to bound -- the defect the benchmark shipped",
            n_overlap=len(set(train_seeds) & set(test_seeds)),
            train=train_seeds, test=test_seeds),
        "f3_multiplicity_is_paid": ge(
            len(ks), args.steps,
            "reporting the best of 31 prices without correction is metric shopping; Holm runs over "
            "the whole sweep and the full profile is reported either way",
            n_tests=len(ks)),
        "f4_clairvoyant_weakly_dominates": ge(
            min(v["gap_mean"] for v in rows.values()), -1e-12,
            "a per-tape minimum cannot exceed a fixed column; a negative gap means a misindexed "
            "estimator",
            min_gap=min(v["gap_mean"] for v in rows.values())),
        "f5_clairvoyant_beats_the_uninformed_placebo_where_detectable": ge(
            sum(1 for k in detectable if rows[k]["clairvoyant_beats_placebo"]),
            max(len(detectable), 1) if detectable else 1,
            "at op12 an uninformed placebo matched the state-conditioned rule; if it matches here, "
            "a positive gap is the freedom to vary rather than information",
            n_detectable=len(detectable), placebo_draws=N_PLACEBO),
        "d1_resolution_does_not_buy_precision": disclosure(
            "a denser grid locates the band's edges; it CANNOT narrow the intervals, because the "
            "precision limit is six test tapes and no lambda resolution changes that",
            n_test_tapes=len(test_seeds), n_prices=len(grid)),
        "d2_no_stop_branch": disclosure(
            "absence needs UCB95 < delta over the enumerated class; failing to clear the bar from "
            "below is not absence",
            delta=DELTA, absent_at=absent),
        "d3_no_fresh_seeds": not_applicable(
            "re-pricing a sealed matrix consumes no tape at all",
            custody=custody_falsifier(all_seeds, replay_of=args.replay_of, exclude=args.output)),
    }
    summary = summarise(falsifiers)
    if not summary["all_passed"]:
        verdict = "BLOCKED_INSTRUMENT"
    elif established:
        verdict = "HEADROOM_ESTABLISHED_IN_A_PRICE_BAND"
    elif detectable:
        verdict = "TIMING_VALUE_DETECTABLE_BUT_BELOW_THE_BAR"
    elif len(absent) == len(grid):
        verdict = "NO_MATERIAL_HEADROOM_ACROSS_THE_REFINED_BAND"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  {'lambda':>7} {'hueco':>10} {'lcb95':>10} {'ucb95':>10} {'holm':>7} {'opt':>4}")
    for k in ks:
        v = rows[k]
        mark = "  <-- detectable" if v["lcb95"] > 0 else ""
        if v["lambda"] in (float(rows[peak]["lambda"]),):
            mark += "  PICO"
        print(f"  {v['lambda']:7.4f} {v['gap_mean']:10.6f} {v['lcb95']:10.6f} "
              f"{v['ucb95']:10.6f} {v['holm_adjusted_p']:7.3f} {v['unique_per_tape_optima']:4d}"
              f"{mark}")
    print(f"\n  banda detectable (LCB > 0): "
          f"{f'lambda {min(band):.4f} a {max(band):.4f}' if band else 'vacia'}")
    print(f"  pico: lambda {rows[peak]['lambda']:.4f}  hueco {rows[peak]['gap_mean']:.6f}")
    print(f"\n  veredicto: {verdict}   (delta = {DELTA})")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones, {summary['n_not_applicable']} no aplicables")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:56s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "lambda_refinement_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_REPRICING_OF_A_SEALED_MATRIX_NO_NEW_EPISODES",
        "run_role": "PRICE_BAND_REFINEMENT", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "source": {"path": str(args.source), "self_sha256": src.get("self_sha256"),
                   "claim_status": src.get("claim_status")},
        "grid": grid, "delta": DELTA,
        "rows": rows, "detectable_at": detectable, "established_at": established,
        "absent_at": absent, "peak_lambda": rows[peak]["lambda"],
        "detectable_band": [min(band), max(band)] if band else None,
        "precision_limit": {"n_test_tapes": len(test_seeds),
                            "note": "resolution in lambda does not narrow the intervals"},
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=args.source)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
