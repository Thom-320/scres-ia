"""Paso 0: (a) ret_visible (transductor) == ret_excel_request_snapshot_v2 (DES directo, compute_episode_metrics);
(b) ~2016 calendarios al azar NO usados por el aprendiz (ni clasicos ni mejor open-loop): transductor vs DES directo, atol 1e-8.
Solo semillas de calibracion quemadas 7480001-7480048.
Usage: python step0_direct_audit.py REPO OUT_JSON [WORKERS]"""
import json, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sys.path.insert(0, str(REPO))
from supply_chain.program_o_full_des import run_program_o_full_des_episode
from supply_chain.program_o_full_des_transducer import (MATRIX_KEYS, direct_full_des_vector,
    extract_full_des_skeleton, simulate_full_des_frontier)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.program_o_state_rich import finite_state_rich_configurations, state_rich_calendar
def scheduler():
    p = json.loads((REPO / "contracts/program_o_full_des_hpi_translation_v1.json").read_text())
    return p["action"]["within_week_schedulers"][p["action"]["primary_scheduler"]]
def enc(c):
    i = 0
    for a in c: i = i * 4 + int(a)
    return i
def dec(i):
    return tuple((i >> (2 * (7 - k))) & 3 for k in range(8))
def work(task):
    ci, seed, idxs = task
    cell = CONFIRMED_RET_CELLS[ci]; sch = scheduler()
    sk, _ = extract_full_des_skeleton(seed=seed, scheduler=sch, regime_persistence=cell.regime_persistence,
        dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
    cals = np.asarray([dec(i) for i in idxs], dtype=np.uint8)
    panel = simulate_full_des_frontier(skeleton=sk, scheduler=sch, calendars=cals)
    rows = []
    for j, i in enumerate(idxs):
        sim, dp = run_program_o_full_des_episode(seed=seed, calendar=dec(i), scheduler=sch,
            regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share,
            downstream_freight_physics_mode="fixed_clock_physical_v1")
        direct = direct_full_des_vector(sim, dp)
        # (a) independent recomputation of the canonical endpoint on the same direct episode
        m2 = compute_episode_metrics(sim, treatment_start=float(sim.program_o_decision_start or 0.0),
                                     ret_excel_contract_version="ret_excel_request_snapshot_v2")
        err = {k: abs(float(direct[k]) - float(panel[k][j])) for k in MATRIX_KEYS}
        rows.append({"cell": cell.cell_id, "seed": seed, "idx": int(i),
                     "ret_visible_transducer": float(panel["ret_visible"][j]),
                     "ret_excel_v2_direct": float(m2["ret_excel"]),
                     "abs_err_endpoint": abs(float(m2["ret_excel"]) - float(panel["ret_visible"][j])),
                     "max_abs_err_all_keys": max(err.values()), "worst_key": max(err, key=err.get)})
    return rows
if __name__ == "__main__":
    cal = json.loads((REPO / "results/program_o/ret_only_learner_v1/calibration_run/result.json").read_text())
    rng = np.random.default_rng(20261126)
    tasks = []
    for ci, cell in enumerate(CONFIRMED_RET_CELLS):
        cs = cal["cell_summaries"][cell.cell_id]; aud = cal["trajectory_audits"][cell.cell_id]
        tapes = sorted(rng.choice(np.arange(7480001, 7480049), size=16, replace=False).tolist())
        for seed in tapes:
            t = seed - 7480001
            used = {enc(aud[s]["calendars"][t]) for s in aud} | {cs["best_open_loop_index"],
                    enc(cs["best_classical_calendars"][t])}
            picks = []
            while len(picks) < 42:
                i = int(rng.integers(0, 65536))
                if i not in used and i not in picks: picks.append(i)
            tasks.append((ci, int(seed), picks))
    t0 = time.time()
    with Pool(W) as pool:
        rows = [r for chunk in pool.imap_unordered(work, tasks) for r in chunk]
    worst = max(rows, key=lambda r: r["max_abs_err_all_keys"])
    res = {"n_calendars": len(rows), "n_cell_tapes": len(tasks), "atol": 1e-8, "seconds": time.time() - t0,
           "max_abs_err_endpoint_v2_vs_transducer": max(r["abs_err_endpoint"] for r in rows),
           "max_abs_err_all_matrix_keys": worst["max_abs_err_all_keys"], "worst": worst,
           "failures": [r for r in rows if r["max_abs_err_all_keys"] > 1e-8 or r["abs_err_endpoint"] > 1e-8],
           "per_cell_n": {c.cell_id: sum(r["cell"] == c.cell_id for r in rows) for c in CONFIRMED_RET_CELLS},
           "rows": rows}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1)[:3000])
