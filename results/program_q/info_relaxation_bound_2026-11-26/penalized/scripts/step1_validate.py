"""Paso 1 validaciones ejecutables: modelo exogeno, filtro/sampler condicional, controlador clasico vectorizado.
Usage: python step1_validate.py OUT_JSON"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import bss_lib as B
B.setup()
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton
from supply_chain.program_o_full_des import product_demand_tape
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS
OUT = Path(sys.argv[1]); sched = B.scheduler(); res = {}
BEST = {"rho75_share90": "max_pressure__0", "rho90_share75": "min_cost_flow__2", "rho90_share90": "max_pressure__0"}
cache = np.load(B.REPO / "results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz")
KIDX = {"max_pressure__0": 2, "min_cost_flow__2": 5}
sks = {}
for ci, cell in enumerate(CONFIRMED_RET_CELLS):
    for s in range(7480001, 7480049):
        sks[(cell.cell_id, s)] = extract_full_des_skeleton(seed=s, scheduler=sched,
            regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share,
            downstream_freight_physics_mode="fixed_clock_physical_v1")[0]
# (1) exogenous model
ref = sks[("rho75_share90", 7480001)].as_dict()
varying = set()
for sk in sks.values():
    d = sk.as_dict()
    for k in d:
        if d[k] != ref[k]: varying.add(k)
res["fields_varying_across_144_tapes"] = sorted(varying)
lab_ok = 0; q_ok = 0; offsets = set(); qs = []
for (cid, s), sk in sks.items():
    cell = next(c for c in CONFIRMED_RET_CELLS if c.cell_id == cid)
    tape = product_demand_tape(s, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share)
    lab_ok += tape["order_products"] == list(sk.order_products)
    stream = np.random.default_rng(np.random.SeedSequence(s).spawn(4)[1]).integers(B.QMIN, B.QMAX + 1, size=400)
    qq = np.asarray(sk.order_quantities)
    qs.append(qq)
    for off in range(400 - 48):
        if np.array_equal(stream[off:off + 48], qq):
            q_ok += 1; offsets.add(off); break
res["labels_equal_product_demand_tape"] = f"{lab_ok}/144"
res["quantities_found_contiguous_in_demand_stream_U2400_2600"] = f"{q_ok}/144"
res["demand_stream_offsets"] = sorted(offsets)
qa = np.concatenate(qs); res["quantity_min_max_mean"] = [float(qa.min()), float(qa.max()), float(qa.mean())]
# (2) conditional sampler vs rejection from the unconditional sampler (own RNG, no contract seeds)
rng = np.random.default_rng(1)
chk = []
for (rho, share) in [(0.75, 0.9), (0.9, 0.75), (0.9, 0.9)]:
    for t, hist in [(1, [1, 1, 1, 1, 0, 1]), (2, [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]), (1, [1, 0, 1, 0, 0, 1])]:
        hb = np.asarray(hist + [0] * (48 - len(hist)), np.int8)
        Lu, _ = B.sample_future(rng, hb, np.full(48, 2500.0), 0, rho, share, 2_000_000)
        acc = Lu[np.all(Lu[:, :6 * t] == hb[:6 * t], 1)]
        Lc, _ = B.sample_future(rng, hb, np.full(48, 2500.0), t, rho, share, 200_000)
        f_rej = np.bincount(acc[:, 6 * t:6 * t + 6].sum(1), minlength=7) / len(acc)
        f_con = np.bincount(Lc[:, 6 * t:6 * t + 6].sum(1), minlength=7) / len(Lc)
        f2r = acc[:, 6 * t + 6:6 * t + 12].mean(); f2c = Lc[:, 6 * t + 6:6 * t + 12].mean()
        se = float(np.sqrt(0.25 / len(acc)))
        chk.append({"rho": rho, "share": share, "t": t, "n_accepted": int(len(acc)),
                    "max_abs_diff_next_week_countC_dist": float(np.abs(f_rej - f_con).max()),
                    "week_t+1_shareC_rej_vs_cond": [float(f2r), float(f2c)], "se_approx": se})
res["sampler_vs_rejection"] = chk
# (3) vectorized classical controller vs repo state_rich_calendar
rng = np.random.default_rng(2); cc = {}
for ci, cell in enumerate(CONFIRMED_RET_CELLS):
    cfg = BEST[cell.cell_id]; mism = 0; n = 0; cache_m = 0
    for ti, s in enumerate(range(7480001, 7480049)):
        sk = sks[(cell.cell_id, s)]
        v = B.classical_continuations(sk, np.zeros((1, 0), np.uint8), cfg, sched)[0]
        cache_m += B.enc(v) != int(cache[f"{cell.cell_id}__calendar_indices"][KIDX[cfg], ti])
        # random prefixes on the real tape and on an inner (conditionally resampled) skeleton
        Lb = B.labels_to_bits(sk.order_products)
        for which in ("outer", "inner"):
            if which == "inner":
                t = int(rng.integers(0, 8))
                l1, q1 = B.sample_future(rng, Lb, np.asarray(sk.order_quantities), t, cell.regime_persistence, cell.dominant_share, 1)
                sk2 = B.make_skeleton(sk, l1[0], q1[0])
            else:
                sk2 = sk
            Ls = rng.integers(0, 8, size=4)
            for Lp in Ls:
                pre = rng.integers(0, 4, size=(3, int(Lp))).astype(np.uint8)
                vec = B.classical_continuations(sk2, pre, cfg, sched)
                for r in range(3):
                    n += 1
                    mism += tuple(vec[r].tolist()) != B.reference_continuation(sk2, pre[r].tolist(), cfg, sched)
    cc[cell.cell_id] = {"config": cfg, "full_calendar_mismatch_vs_cache": int(cache_m), "prefix_checks": n, "prefix_mismatches": int(mism)}
res["classical_vectorized_vs_reference"] = cc
sk = sks[("rho90_share75", 7480001)]
for cfg in ("max_pressure__0", "min_cost_flow__2"):
    t0 = time.time(); B.classical_continuations(sk, B.all_prefixes(7), cfg, sched)
    res[f"seconds_classical_16384_rows_{cfg}"] = time.time() - t0
OUT.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
