"""Recompute the full 65,536-calendar Program O-R full-DES frontier for one burned
calibration tape (cell, seed) and the 10 classical calendars. Read-only on repo.
Usage: python frontier_tape.py REPO OUTDIR CELL_INDEX SEED [--check-determinism]
"""
from __future__ import annotations
import json, sys, time, hashlib
from pathlib import Path
import numpy as np

repo = Path(sys.argv[1]); out = Path(sys.argv[2]); ci = int(sys.argv[3]); seed = int(sys.argv[4])
check = "--check-determinism" in sys.argv
sys.path.insert(0, str(repo))
from supply_chain.program_o_full_des_transducer import (extract_full_des_skeleton,
    simulate_full_des_frontier, full_action_calendars)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS
from supply_chain.program_o_state_rich import finite_state_rich_configurations, state_rich_calendar

def scheduler():  # copied verbatim from scripts/evaluate_program_o_ret_learner.py (avoids sb3 import)
    parent = json.loads((repo / "contracts/program_o_full_des_hpi_translation_v1.json").read_text())
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]

def encode(cal):
    i = 0
    for a in cal: i = i * 4 + int(a)
    return i

cell = CONFIRMED_RET_CELLS[ci]
t0 = time.time()
def run():
    sk, _ = extract_full_des_skeleton(seed=seed, scheduler=scheduler(),
        regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share,
        downstream_freight_physics_mode="fixed_clock_physical_v1")
    cals = full_action_calendars()
    panel = simulate_full_des_frontier(skeleton=sk, scheduler=scheduler(), calendars=cals)
    return sk, cals, panel
sk, cals, panel = run()
t1 = time.time()
assert all(encode(cals[i]) == i for i in (0, 1, 5, 777, 65535)), "calendar ordering != encode_calendar"
idx = []
for config in finite_state_rich_configurations():
    cal, _ = state_rich_calendar(skeleton=sk.as_dict(), scheduler=scheduler(), config=config,
                                 regime_persistence=0.75, dominant_share=0.90)
    idx.append(encode(tuple(cal)))
rv = np.asarray(panel["ret_visible"], dtype=np.float64)
res = {"cell": cell.cell_id, "seed": seed, "frontier_seconds": t1 - t0,
       "classical_indices": idx, "max": float(rv.max()), "argmax": int(rv.argmax()),
       "sha256_ret_visible": hashlib.sha256(rv.tobytes()).hexdigest()}
if check:
    _, _, panel2 = run()
    rv2 = np.asarray(panel2["ret_visible"], dtype=np.float64)
    res["determinism_bitwise_equal"] = bool(np.array_equal(rv, rv2))
    # one classical calendar evaluated alone must equal the frontier entry
    single = simulate_full_des_frontier(skeleton=sk, scheduler=scheduler(),
                                        calendars=np.asarray([cals[idx[0]]], dtype=np.uint8))
    res["single_vs_frontier_absdiff"] = float(abs(single["ret_visible"][0] - rv[idx[0]]))
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / f"{cell.cell_id}__tape_{seed}.npz", ret_visible=rv,
                    classical_indices=np.asarray(idx, dtype=np.int64))
(out / f"{cell.cell_id}__tape_{seed}.json").write_text(json.dumps(res))
print(json.dumps(res))
