"""Inner-sampling convergence: U_pen - V_CL for nested replicate sets (M = 8*len) and split halves.
Usage: python bss_convergence.py INNER_DIR NREPS OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import bss_bound as BB
inner, nrep, outp = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
cache = np.load(BB.B.REPO / "results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz")
sets = {}
k = 1
while k <= nrep:
    sets[f"M{8 * k}"] = list(range(k)); k *= 2
h = nrep // 2
sets[f"halfA_M{8 * h}"] = list(range(h)); sets[f"halfB_M{8 * h}"] = list(range(h, nrep))
out = {}
for ci, cell in enumerate(BB.CELLS):
    seeds = list(range(7480001, 7480049)); all_seeds = cache[f"{cell}__tape_seeds"].tolist()
    CL = np.asarray([float(np.load(BB.COTA / "frontier" / f"{cell}__tape_{s}.npz")["ret_visible"][
        int(cache[f"{cell}__calendar_indices"][BB.BEST_K[cell], all_seeds.index(s)])]) for s in seeds])
    row = {}
    for name, reps in sets.items():
        U = np.asarray([BB.dp_max(BB.advantages(*BB.load_avg(inner, cell, s, 8, reps))) for s in seeds])
        row[name] = {"U_pen": float(U.mean()), "gap": float((U - CL).mean()), "per_tape_U": U.tolist()}
    a, b = np.asarray(row[f"halfA_M{8*h}"]["per_tape_U"]), np.asarray(row[f"halfB_M{8*h}"]["per_tape_U"])
    row["split_half_inner_se_of_mean_at_M%d" % (8 * h)] = float((a - b).std(ddof=1) / np.sqrt(len(a)) / np.sqrt(2))
    for v in row.values():
        if isinstance(v, dict): v.pop("per_tape_U")
    out[cell] = row
    print(cell, json.dumps(row))
outp.write_text(json.dumps(out, indent=1))
