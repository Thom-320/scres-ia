#!/usr/bin/env python3
"""R1: does the declared policy class contain a member that satisfies the TAIL constraint?

Preregistration: `docs/PREREGISTRO_R1_RESTRICCION_DENTRO_DEL_OBJETIVO_2026-08-12.md`

Program O won on the mean in all three cells and died on `ret_visible_cvar10` under simultaneous
inference, with the point estimates POSITIVE. Reading the fit code shows why: admissibility is
checked on MEANS of the guardrail deltas and the objective then maximises the MEAN of ret_visible,
with the guardrails only as tie-breakers. The selection never checked the quantity that killed it.

And the oracle CAN satisfy it -- imposing the whole per-tape guardrail vector costs 0.81% of raw
headroom. The policy was simply never asked to.

THIS IS EXHAUSTIVE, NOT A TRAINING. The declared class is finite -- 4 policy ids x 4 initial actions
-- and every configuration's outcome already sits in the calendar matrices. So the question
"does a tail-feasible member exist" is answered by enumeration, not by optimisation, and there is no
hyperparameter anyone could tune after seeing the result.

Development grade on the FIT block. No seed is opened. Evaluating on the validation block would be a
second rescue, which Program O's own audit forbids.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from screen_program_o_hobs_fit import (                                          # noqa: E402
    HIGHER_KEYS, LOWER_KEYS, configurations, load_cell_panel, policy_calendar_for_skeleton)
from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402
from supply_chain.program_o_hobs import calendar_index                           # noqa: E402

CONTRACT = Path("docs/PREREGISTRO_R1_RESTRICCION_DENTRO_DEL_OBJETIVO_2026-08-12.md")
OUT = Path("results/program_o/r1_tail_in_the_objective_v1/result.json")
FIT_ROOT = ROOT / "outputs/program_o_runs/program-o-hobs-fit-v1-20260715/artifacts/fit"
SEALED_FIT = FIT_ROOT / "result.json"
CELL_CONTRACT = ROOT / "contracts/program_o_state_rich_hobs_prelearner_v1.json"
TAIL_KEY = "ret_visible_cvar10"
MEAN_KEY = "ret_visible"
#: t(47) one-sided 95%. 48 tapes per cell.
T_ONE_SIDED_47 = 1.6779


def lcb95(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return float(v.mean() - T_ONE_SIDED_47 * se)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / OUT)
    args = ap.parse_args()
    started = time.perf_counter()

    sealed = json.loads(SEALED_FIT.read_text())
    shipped = sealed["selected_config"]
    cells = json.loads(CELL_CONTRACT.read_text())["cells"]
    seeds = sorted(int(p.stem.split("_")[1])
                   for p in (FIT_ROOT / "skeletons" / cells[0]["id"]).glob("tape_*.json"))
    configs = configurations()

    per_cell, s_mean_choice, s_cvar_choice = {}, {}, {}
    for cell in cells:
        cid = cell["id"]
        panel = load_cell_panel(FIT_ROOT, cid, seeds)
        skeletons = [json.loads((FIT_ROOT / "skeletons" / cid / f"tape_{s}.json").read_text())
                     for s in seeds]
        static_index = int(np.argmax(panel[MEAN_KEY].mean(axis=0)))
        tape_rows = np.arange(len(skeletons), dtype=np.int64)
        model = {"regime_persistence": float(cell["regime_persistence"]),
                 "dominant_product_share": float(cell["dominant_product_share"])}

        rows = []
        for cfg in configs:
            idx = np.asarray([calendar_index(policy_calendar_for_skeleton(sk, cfg, model)[0])
                              for sk in skeletons], dtype=np.int64)
            deltas = {k: panel[k][tape_rows, idx] - panel[k][:, static_index]
                      for k in (MEAN_KEY, *HIGHER_KEYS, *LOWER_KEYS)}
            admissible = (all(float(deltas[k].mean()) >= -1e-12 for k in HIGHER_KEYS)
                          and all(float(deltas[k].mean()) <= 1e-12 for k in LOWER_KEYS))
            rows.append({
                **cfg, "admissible": bool(admissible),
                "mean_ret_visible": float(panel[MEAN_KEY][tape_rows, idx].mean()),
                "mean_delta": float(deltas[MEAN_KEY].mean()),
                "mean_delta_lcb95": lcb95(deltas[MEAN_KEY]),
                "tail_delta": float(deltas[TAIL_KEY].mean()),
                "tail_delta_lcb95": lcb95(deltas[TAIL_KEY]),
                "tail_feasible": bool(lcb95(deltas[TAIL_KEY]) >= 0.0),
                "worst_product_fill_delta": float(deltas["worst_product_fill"].mean()),
                "worst_product_fill_lcb95": lcb95(deltas["worst_product_fill"]),
            })

        eligible = [i for i, r in enumerate(rows) if r["admissible"]]
        # S_mean: exactly the shipped rule.
        s_mean = min(eligible, key=lambda i: (-rows[i]["mean_ret_visible"],
                                              -rows[i]["worst_product_fill_delta"],
                                              i)) if eligible else -1
        # S_cvar: lexicographic on the constraint that killed it, then the mean.
        tail_ok = [i for i in eligible if rows[i]["tail_feasible"]]
        s_cvar = min(tail_ok, key=lambda i: (-rows[i]["mean_ret_visible"], i)) if tail_ok else -1

        per_cell[cid] = {"rows": rows, "n_admissible": len(eligible), "n_tail_feasible": len(tail_ok),
                         "static_index": static_index,
                         "s_mean_index": s_mean, "s_cvar_index": s_cvar,
                         "s_mean": rows[s_mean] if s_mean >= 0 else None,
                         "s_cvar": rows[s_cvar] if s_cvar >= 0 else None}
        s_mean_choice[cid] = rows[s_mean] if s_mean >= 0 else None
        s_cvar_choice[cid] = rows[s_cvar] if s_cvar >= 0 else None
        print(f"  {cid}: {len(eligible)}/16 admisibles, {len(tail_ok)} factibles en cola "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    primary = "rho75_share90"
    reproduced = (s_mean_choice[primary] is not None
                  and s_mean_choice[primary]["policy_id"] == shipped["policy_id"]
                  and s_mean_choice[primary]["initial_action"] == shipped["initial_action"])
    any_tail_feasible = any(c["n_tail_feasible"] > 0 for c in per_cell.values())
    cvar_keeps_mean = all(c["s_cvar"] is not None and c["s_cvar"]["mean_delta_lcb95"] > 0.0
                          for c in per_cell.values() if c["s_cvar"] is not None) and any_tail_feasible
    rules_differ = [cid for cid, c in per_cell.items()
                    if c["s_mean_index"] != c["s_cvar_index"]]

    checks = {
        "p1_i_reproduce_the_shipped_selection": F.check(
            reproduced,
            "S_mean must reselect exactly what the sealed fit froze; if it does not I am not "
            "reproducing the pipeline and nothing downstream means anything",
            computed_from={"n_cells": len(per_cell), "n_configs": len(configs)},
            shipped=shipped, reselected=s_mean_choice[primary]),
        "p2_the_class_contains_a_tail_feasible_member": F.check(
            any_tail_feasible,
            "with sixteen configurations enumerated exhaustively this can fail outright, and that "
            "failure is the informative one: the tail would not be controllable INSIDE the declared "
            "policy class, closing the family by policy class rather than by a badly placed gate",
            computed_from={"n_tail_feasible_total":
                           sum(c["n_tail_feasible"] for c in per_cell.values()),
                           "n_cells": len(per_cell)},
            per_cell={cid: c["n_tail_feasible"] for cid, c in per_cell.items()}),
        "p3_the_tail_feasible_policy_keeps_the_mean": F.check(
            cvar_keeps_mean,
            "if the mean advantage was bought in the lower tail it disappears here, and the "
            "conclusion is that the adaptation WAS the concentration",
            computed_from={"n_cells_with_cvar_choice":
                           sum(c["s_cvar"] is not None for c in per_cell.values()),
                           "n_cells": len(per_cell)},
            mean_lcb_by_cell={cid: (c["s_cvar"]["mean_delta_lcb95"] if c["s_cvar"] else None)
                              for cid, c in per_cell.items()}),
        "p4_the_two_rules_can_differ": F.check(
            len(rules_differ) > 0,
            "if the mean-only rule and the tail-aware rule pick the same configuration in every "
            "cell, this experiment measures nothing and must say so",
            computed_from={"n_cells_where_they_differ": len(rules_differ),
                           "n_cells": len(per_cell)},
            cells_where_they_differ=rules_differ),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: declared replay of the already-consumed fit "
                                        "block 7420001-7420048. No seed opened, no episode run, and "
                                        "the validation block is never touched.",
                     "seeds": seeds, "seeds_opened": 0, "episodes_run": 0,
                     "validation_block_touched": False}}
    summary = F.summarise(checks)

    if not checks["p1_i_reproduce_the_shipped_selection"]["passed"]:
        status = "BLOCKED_CANNOT_REPRODUCE_THE_SHIPPED_SELECTION"
    elif not checks["p2_the_class_contains_a_tail_feasible_member"]["passed"]:
        status = "THE_POLICY_CLASS_CONTAINS_NO_TAIL_FEASIBLE_MEMBER"
    elif not checks["p3_the_tail_feasible_policy_keeps_the_mean"]["passed"]:
        status = "TAIL_FEASIBLE_BUT_THE_MEAN_ADVANTAGE_WAS_THE_TAIL"
    else:
        status = "A_TAIL_FEASIBLE_POLICY_EXISTS_IN_THE_DECLARED_CLASS"

    payload = {
        "schema_version": "program_o_r1_tail_in_the_objective_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DECLARED_REPLAY_OF_THE_FIT_BLOCK",
        "scope": "DEVELOPMENT_ONLY_EXHAUSTIVE_RESELECTION_NO_SEEDS_VALIDATION_BLOCK_UNTOUCHED",
        "endpoint": "paired_per_tape_deltas_vs_best_static_on_ret_visible_and_ret_visible_cvar10",
        "seeds": seeds, "n_configurations": len(configs), "tail_key": TAIL_KEY,
        "shipped_selection": shipped, "reselected_by_s_mean": s_mean_choice,
        "reselected_by_s_cvar": s_cvar_choice, "cells_where_the_rules_differ": rules_differ,
        "per_cell": per_cell,
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started, "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT, reference=SEALED_FIT)

    print(f"\nveredicto: {status}\n")
    for cid, c in per_cell.items():
        print(f"  {cid}  admisibles {c['n_admissible']}/16  factibles en cola {c['n_tail_feasible']}")
        for tag, r in (("S_mean", c["s_mean"]), ("S_cvar", c["s_cvar"])):
            if r is None:
                print(f"    {tag:7} ninguna")
                continue
            print(f"    {tag:7} {r['policy_id']}/a{r['initial_action']}  "
                  f"media {r['mean_delta']:+.6f} [LCB {r['mean_delta_lcb95']:+.6f}]  "
                  f"cola {r['tail_delta']:+.6f} [LCB {r['tail_delta_lcb95']:+.6f}]")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:48} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
