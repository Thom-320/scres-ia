"""Inner (nested) simulation for the BSS penalized bound, one replicate of M inner futures per stage.

For outer tape w (cell, seed), stage t = 0..7 and replicate r, draws M futures from P(. | H_t) (H_t = labels and
quantities of the orders of weeks < t, all revealed before decision t) and stores
    Fbar_t[p] = mean_j f(p (+) kappa, w'_j),  p in {0..3}^(t+1)  (base-4 index order),
where f = transducer ret_visible and kappa = classical continuation of the cell's best classical policy.
Also stores kappa_t[p_t] = classical action at week t after prefix p_t (depends only on H_t).
Replicates are independent, so Fbar of R replicates can be averaged into an M*R estimator.

Usage: python bss_inner.py OUTDIR M REP_FROM REP_TO WORKERS CELL_INDEX:SEED_FROM-SEED_TO [...]
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bss_lib as B  # noqa: E402

B.setup()
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton, simulate_full_des_frontier)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402

BEST = {"rho75_share90": "max_pressure__0", "rho90_share75": "min_cost_flow__2",
        "rho90_share90": "max_pressure__0"}


def run(task):
    outdir, M, rep, ci, seed = task
    out = Path(outdir) / f"{CONFIRMED_RET_CELLS[ci].cell_id}__tape_{seed}__M{M}__rep{rep}.npz"
    if out.exists():
        return str(out), 0.0
    t0 = time.time()
    cell = CONFIRMED_RET_CELLS[ci]
    cfg = BEST[cell.cell_id]
    sched = B.scheduler()
    sk, _ = extract_full_des_skeleton(seed=seed, scheduler=sched, regime_persistence=cell.regime_persistence,
                                      dominant_share=cell.dominant_share,
                                      downstream_freight_physics_mode="fixed_clock_physical_v1")
    lb = B.labels_to_bits(sk.order_products)
    qo = np.asarray(sk.order_quantities, float)
    # inner RNG: own SeedSequence, never a contract seed; independent across (cell, tape, M, replicate)
    rng = np.random.default_rng(np.random.SeedSequence([20261126, ci, seed, M, rep]))
    arrays = {}
    for t in range(8):
        pre_t = B.all_prefixes(t)
        kap = B.classical_continuations(sk, pre_t, cfg, sched)[:, t] if t < 8 else None
        arrays[f"kappa_{t}"] = kap.astype(np.uint8)
        pre = B.all_prefixes(t + 1)
        Ls, Qs = B.sample_future(rng, lb, qo, t, cell.regime_persistence, cell.dominant_share, M)
        acc = np.zeros(4 ** (t + 1))
        for j in range(M):
            skj = B.make_skeleton(sk, Ls[j], Qs[j])
            C = B.classical_continuations(skj, pre, cfg, sched)
            if j == 0:  # the week-t classical action is H_t-measurable: identical on inner and outer skeletons
                kj = B.classical_continuations(skj, pre_t, cfg, sched)[:, t]
                assert np.array_equal(kj, kap), "classical week-t action depends on the future"
                assert np.array_equal(C[:, :t + 1], pre), "prefix not respected"
            acc += np.asarray(simulate_full_des_frontier(skeleton=skj, scheduler=sched, calendars=C)["ret_visible"],
                              dtype=np.float64)
        arrays[f"Fbar_{t}"] = acc / M
    arrays["seconds"] = np.asarray(time.time() - t0)
    np.savez_compressed(out, **arrays)
    return str(out), time.time() - t0


if __name__ == "__main__":
    outdir, M, r0, r1, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    tasks = []
    for spec in sys.argv[6:]:
        ci, rng_s = spec.split(":")
        a, b = map(int, rng_s.split("-"))
        assert 7480001 <= a <= b <= 7480048, "only burned calibration tapes 7480001-7480048"
        for rep in range(r0, r1 + 1):
            tasks += [(outdir, M, rep, int(ci), s) for s in range(a, b + 1)]
    t0 = time.time()
    with Pool(W) as pool:
        secs = [s for _p, s in pool.imap_unordered(run, tasks)]
    print(f"tasks={len(tasks)} wall={time.time() - t0:.1f}s mean_task={np.mean(secs):.1f}s")
