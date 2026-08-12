#!/usr/bin/env python3
"""Exact studentized max-t over the pooled 288 tapes, straight from the calendar matrices.

Preregistration: `docs/EXCEPCION_PI_Y_PREREGISTRO_REPLICA_CON_POTENCIA_2026-08-12.md`
Supersedes the pooling approximation in `results/program_o/powered_replication_v1/result.json`.

The pooled run took ONE declared shortcut: it combined the six sub-blocks' point estimates and
standard errors analytically and reused the largest of their six simultaneous critical values,
arguing that pooling independent replicates leaves the correlation structure of the studentized
estimands unchanged. That argument is probably right and it is still an argument. This removes it.

WHAT THIS DOES INSTEAD. It rebuilds the 69 estimands tape by tape from the raw calendar matrices of
all six sub-blocks, concatenates them into a single 288-tape panel per cell, and runs the SAME
studentized max-t bootstrap the frozen runner runs -- the only change being that `n_tapes` is a
parameter rather than the literal 48, which is the one line that made the frozen function unusable
here.

MEMORY. A 288 x 65,536 x 34 panel is 5 GB per cell. It is never built: the bootstrap only ever
touches each metric at the policy column, the frozen static column and the placebo columns, so each
tape is opened once and reduced to a handful of scalars.

THE CONTROL IS THE POINT. Run at n=48 on sub-block 1 with the same seed, this must reproduce that
sub-block's sealed critical value and its sealed bounds to floating-point identity. If it does not,
the reimplementation is a different estimator and its 288-tape answer means nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from screen_program_o_fixed_clock_hobs_validation import (                       # noqa: E402
    HIGHER_KEYS, LOWER_KEYS)
from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402

CONTRACT = Path("docs/EXCEPCION_PI_Y_PREREGISTRO_REPLICA_CON_POTENCIA_2026-08-12.md")
OUT = Path("results/program_o/powered_exact_maxt_v1/result.json")
POOLED = Path("results/program_o/powered_replication_v1/result.json")
CELLS = ("rho75_share90", "rho90_share75", "rho90_share90")
TAIL = "ret_visible_cvar10"
RESAMPLES = 10_000
#: Byte-identical to the frozen runner's seed derivation.
SEED_STRING = b"program-o-fixed-clock-hobs-validation-v1"


def columns_for(runs: list[Path], cell: str) -> dict:
    """Per-tape scalars at the policy, static and placebo columns. Never the whole matrix."""
    keys = ("ret_visible", *HIGHER_KEYS, *LOWER_KEYS)
    policy, static, placebo, statics_seen = {k: [] for k in keys}, {k: [] for k in keys}, {}, set()
    for run in runs:
        res = json.loads((run / "result.json").read_text())
        row = res["cells"][cell]
        statics_seen.add(int(row["static_index"]))
        idx = [int(i) for i in row["calendar_indices"]]
        static_index = int(row["static_index"])
        pl_rows = res["placebos"][cell]
        pl_idx = {f"{fam}::{mode}": [int(i) for i in body["calendar_indices"]]
                  for fam, fr in pl_rows.items() for mode, body in fr["placebos"].items()}
        for name in pl_idx:
            placebo.setdefault(name, [])
        matrices = sorted((run / "raw_calendar_matrix" / cell).glob("tape_*.npz"),
                          key=lambda p: int(p.stem.split("_")[1]))
        if len(matrices) != len(idx):
            raise AssertionError(f"{run}/{cell}: {len(matrices)} matrices vs {len(idx)} indices")
        for position, path in enumerate(matrices):
            with np.load(path) as shard:
                for key in keys:
                    col = shard[key]
                    policy[key].append(float(col[idx[position]]))
                    static[key].append(float(col[static_index]))
                rv = shard["ret_visible"]
                for name, seq in pl_idx.items():
                    placebo[name].append(float(rv[seq[position]]))
    if len(statics_seen) != 1:
        raise AssertionError(f"{cell}: static index drifted across sub-blocks: {statics_seen}")
    return {"policy": {k: np.asarray(v) for k, v in policy.items()},
            "static": {k: np.asarray(v) for k, v in static.items()},
            "placebo": {k: np.asarray(v) for k, v in placebo.items()},
            "static_index": statics_seen.pop()}


def max_t(per_cell: dict, n_tapes: int, resamples: int) -> dict:
    """The frozen procedure with `n_tapes` as a parameter instead of the literal 48."""
    seed = int.from_bytes(hashlib.sha256(SEED_STRING).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, n_tapes, size=(resamples, n_tapes))
    counts = np.zeros((resamples, n_tapes), dtype=np.float64)
    for position, sample in enumerate(bootstrap_indices):
        counts[position] = np.bincount(sample, minlength=n_tapes)
    counts /= float(n_tapes)

    definitions, points, boot_columns = [], {}, []
    for cell in CELLS:
        cc = per_cell[cell]
        signed = [("ret_visible", 1.0, "primary")]
        signed += [(k, 1.0, "guardrail") for k in HIGHER_KEYS]
        signed += [(k, -1.0, "guardrail") for k in LOWER_KEYS]
        for key, sign, kind in signed:
            name = f"{cell}::{kind}::{key}"
            p, s = cc["policy"][key], cc["static"][key]
            definitions.append((name, kind, key))
            points[name] = float(sign * (p - s).mean())
            boot_columns.append(sign * (counts @ p - counts @ s))
        real = cc["policy"]["ret_visible"]
        for name, values in cc["placebo"].items():
            fam, mode = name.split("::")
            full = f"{cell}::placebo::{fam}::{mode}"
            contrast = real - values
            definitions.append((full, "placebo", mode))
            points[full] = float(contrast.mean())
            boot_columns.append(counts @ contrast)

    boot = np.column_stack(boot_columns)
    point_vector = np.asarray([points[n] for n, _k, _m in definitions])
    se = boot.std(axis=0, ddof=1)
    active = se > 1e-15
    maxima = np.zeros(resamples)
    if np.any(active):
        maxima = np.max((point_vector[None, active] - boot[:, active]) / se[active], axis=1)
    critical = float(np.quantile(maxima, 0.95))
    lower = point_vector.copy()
    lower[active] = point_vector[active] - critical * se[active]
    return {"simultaneous_critical": critical, "estimand_count": len(definitions),
            "resamples": resamples,
            "estimates": {n: {"kind": k, "metric_or_mode": m, "estimate": float(point_vector[i]),
                              "bootstrap_se": float(se[i]),
                              "simultaneous_lcb95": float(lower[i])}
                          for i, (n, k, m) in enumerate(definitions)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("/tmp/o_powered"))
    ap.add_argument("--output", type=Path, default=ROOT / OUT)
    args = ap.parse_args()
    all_runs = [args.runs / f"block{k}" for k in range(1, 7)]

    print("  control de reproduccion sobre el sub-bloque 1 (n=48)...", flush=True)
    solo = {c: columns_for([all_runs[0]], c) for c in CELLS}
    recomputed = max_t(solo, 48, RESAMPLES)
    sealed = json.loads((all_runs[0] / "result.json").read_text())["inference"]
    crit_gap = abs(recomputed["simultaneous_critical"] - float(sealed["simultaneous_critical"]))
    shared = sorted(set(recomputed["estimates"]) & set(sealed["estimates"]))
    lcb_gap = max((abs(recomputed["estimates"][n]["simultaneous_lcb95"]
                       - sealed["estimates"][n]["simultaneous_lcb95"]) for n in shared),
                  default=float("inf"))
    est_gap = max((abs(recomputed["estimates"][n]["estimate"]
                       - sealed["estimates"][n]["estimate"]) for n in shared), default=float("inf"))
    reproduces = crit_gap < 1e-12 and lcb_gap < 1e-12 and est_gap < 1e-12
    print(f"    critico {recomputed['simultaneous_critical']:.12f} vs sellado "
          f"{float(sealed['simultaneous_critical']):.12f}  gap {crit_gap:.3e}")
    print(f"    estimandos comunes {len(shared)}/{len(sealed['estimates'])}  "
          f"max gap punto {est_gap:.3e}  max gap LCB {lcb_gap:.3e}")

    print("  agrupando las 288 tapas...", flush=True)
    pooled_cols = {c: columns_for(all_runs, c) for c in CELLS}
    n_tapes = len(pooled_cols[CELLS[0]]["policy"]["ret_visible"])
    exact = max_t(pooled_cols, n_tapes, RESAMPLES)

    approx = json.loads(POOLED.read_text())
    tail_exact = {c: exact["estimates"][f"{c}::guardrail::{TAIL}"] for c in CELLS}
    tail_approx = {c: approx["pooled_tail"][c] for c in CELLS}
    prim_exact = {c: exact["estimates"][f"{c}::primary::ret_visible"] for c in CELLS}
    other = {n: v for n, v in exact["estimates"].items()
             if v["kind"] == "guardrail" and TAIL not in n}
    guardrails_ok = all(v["simultaneous_lcb95"] >= -1e-9 or v["estimate"] >= 0.0
                        for v in other.values())

    checks = {
        "y1_the_reimplementation_reproduces_the_frozen_one": F.check(
            reproduces,
            "at n=48 on sub-block 1 with the same seed this must return the sealed critical value "
            "and every sealed bound to floating-point identity; if it does not, it is a different "
            "estimator and its 288-tape answer is worthless",
            computed_from={"critical_gap": crit_gap, "max_lcb_gap": lcb_gap,
                           "max_estimate_gap": est_gap, "n_shared_estimands": len(shared)}),
        "y2_the_estimand_family_is_the_same_size": F.check(
            exact["estimand_count"] == int(sealed["estimand_count"]),
            "a max-t over a different number of estimands is a different critical value; the "
            "family must be the same 69",
            computed_from={"exact": exact["estimand_count"],
                           "sealed": int(sealed["estimand_count"])}),
        "y3_all_288_tapes_are_present": F.check(
            n_tapes == 288,
            "a sub-block silently missing would make this a smaller replication reported as a "
            "bigger one",
            computed_from={"n_tapes": n_tapes, "expected": 288}),
        "y4_the_tail_clears_zero_in_every_cell": F.check(
            all(v["simultaneous_lcb95"] > 0.0 for v in tail_exact.values()),
            "THE HEADLINE, now with no approximation anywhere in the chain. It can fail, and if it "
            "does the pooled verdict was an artefact of reusing the sub-block critical values",
            computed_from={"critical": exact["simultaneous_critical"],
                           "n_clearing": sum(v["simultaneous_lcb95"] > 0.0
                                             for v in tail_exact.values())},
            exact_lcb={c: v["simultaneous_lcb95"] for c, v in tail_exact.items()},
            approximate_lcb={c: v["simultaneous_lcb95"] for c, v in tail_approx.items()}),
        "y5_the_other_guardrails_stay_noninferior": F.check(
            guardrails_ok,
            "clearing the tail does not license breaking any other guardrail in the vector",
            computed_from={"n_other_guardrails": len(other),
                           "n_violating": sum(not (v["simultaneous_lcb95"] >= -1e-9
                                                   or v["estimate"] >= 0.0)
                                              for v in other.values())}),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: this reads calendar matrices written by the "
                                        "already-consumed 7500001-7500288 block. No seed is opened "
                                        "and no episode is run.",
                     "seeds_opened": 0, "episodes_run": 0, "n_tapes_per_cell": n_tapes}}
    summary = F.summarise(checks)

    if not checks["y1_the_reimplementation_reproduces_the_frozen_one"]["passed"] \
            or not checks["y2_the_estimand_family_is_the_same_size"]["passed"] \
            or not checks["y3_all_288_tapes_are_present"]["passed"]:
        status = "BLOCKED_REIMPLEMENTATION_IS_NOT_THE_FROZEN_ESTIMATOR"
    elif not checks["y4_the_tail_clears_zero_in_every_cell"]["passed"]:
        status = "EXACT_MAXT_WITHDRAWS_THE_POOLED_VERDICT"
    elif not checks["y5_the_other_guardrails_stay_noninferior"]["passed"]:
        status = "TAIL_CLEARED_BUT_ANOTHER_GUARDRAIL_BROKE"
    else:
        status = "OBSERVABLE_CONVERSION_SURVIVES_UNDER_THE_EXACT_MAXT"

    payload = {
        "schema_version": "program_o_powered_exact_maxt_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "EXACT_REANALYSIS_OF_THE_POWERED_BLOCK",
        "scope": "REMOVES_THE_POOLING_APPROXIMATION_NO_SEEDS_NO_EPISODES",
        "endpoint": "studentized_simultaneous_max_t_over_69_estimands_on_288_tapes",
        "n_tapes_per_cell": n_tapes, "resamples": RESAMPLES,
        "simultaneous_critical_exact": exact["simultaneous_critical"],
        "simultaneous_critical_approximate": approx["simultaneous_critical_used"],
        "reproduction_control": {"critical_gap": crit_gap, "max_lcb_gap": lcb_gap,
                                 "max_estimate_gap": est_gap,
                                 "recomputed_critical": recomputed["simultaneous_critical"],
                                 "sealed_critical": float(sealed["simultaneous_critical"])},
        "tail_exact": tail_exact, "tail_approximate": tail_approx, "primary_exact": prim_exact,
        "estimates": exact["estimates"],
        "falsifiers": checks, "falsifier_summary": summary,
        "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT, reference=POOLED)

    print(f"\nveredicto: {status}\n")
    print(f"  critico exacto {exact['simultaneous_critical']:.6f}   "
          f"aproximado {approx['simultaneous_critical_used']:.6f}   "
          f"estimandos {exact['estimand_count']}   tapas {n_tapes}\n")
    print(f"  {'celda':16}{'cola punto':>13}{'LCB exacto':>13}{'LCB aprox':>13}{'primario LCB':>14}")
    for c in CELLS:
        print(f"  {c:16}{tail_exact[c]['estimate']:+13.6f}"
              f"{tail_exact[c]['simultaneous_lcb95']:+13.6f}"
              f"{tail_approx[c]['simultaneous_lcb95']:+13.6f}"
              f"{prim_exact[c]['simultaneous_lcb95']:+14.6f}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:52} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
