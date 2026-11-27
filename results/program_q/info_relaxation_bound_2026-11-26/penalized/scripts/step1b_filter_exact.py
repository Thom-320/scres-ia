"""Exact check of the regime filter by brute-force enumeration of regime paths (independent of the forward recursion),
on the realized label histories of the 144 burned tapes, all t = 0..8. Usage: python step1b_filter_exact.py OUT_JSON"""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import bss_lib as B; B.setup()
from supply_chain.program_o_full_des import product_demand_tape
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS
worst = 0.0; n = 0
for cell in CONFIRMED_RET_CELLS:
    rho, sh = cell.regime_persistence, cell.dominant_share
    for s in range(7480001, 7480049):
        lb = B.labels_to_bits(product_demand_tape(s, regime_persistence=rho, dominant_share=sh)["order_products"])
        for t in range(9):
            num = den = 0.0
            for path in itertools.product((0, 1), repeat=t + 1):  # regimes of weeks 0..t, 1 = P_C
                p = 0.5
                for k in range(t + 1):
                    if k: p *= rho if path[k] == path[k - 1] else 1 - rho
                    if k < t:
                        w = lb[6 * k:6 * k + 6]; nd = int((w == path[k]).sum())
                        p *= sh ** nd * (1 - sh) ** (6 - nd)
                den += p; num += p * (path[t] == 1)
            if t < 8:
                worst = max(worst, abs(num / den - B.regime_posterior_c(lb, t, rho, sh))); n += 1
res = {"checks": n, "max_abs_diff_filter_vs_enumeration": worst}
Path(sys.argv[1]).write_text(json.dumps(res)); print(res)
