#!/usr/bin/env python3
"""The expanded design on virgin block 8700001-8700048: ceiling first, then the signal search.

Contract: `docs/AUTORIZACION_PI_BLOQUE_8700001_2026-08-08.md`, frozen at commit b9115292 BEFORE the
block was touched. Falsifiers and the permutation null from `supply_chain.falsifiers`.

A VIRGIN SEED IS A ONE-WAY DOOR, so everything happens in ONE pass: the J matrix, the ceiling, its
interaction null, and the signal search. A defect found at run time burns the block; it does not
license a rerun. The pre-flight therefore runs on already-burned seeds first.

THE ORDER IS FIXED AND IT MATTERS. The ceiling is measured and null-tested on the new block FIRST.
If it does not replicate, the twelve-tape ceiling was a twelve-tape artifact and nothing about
signals is read at all -- the four failed conversion attempts would then have been failing to catch
something that was never there.

WHAT THE EXPANSION BUYS. The previous search had six training tapes, which support a
nearest-neighbour map on one feature and nothing richer; its own d2 said a negative could not be
told from no power. Twenty-four training tapes support k = 3 as well, so both map families run,
26 tests under Holm.
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

from scripts.run_priced_buffer_gate_v1 import (  # noqa: E402
    MAX_STEPS, SCENARIO, STEP_HOURS, exposure, make_env, options, play,
)
from scripts.run_signal_search_v1 import prefix_features  # noqa: E402
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.falsifiers import (  # noqa: E402
    disclosure, ge, gt, lt, not_applicable, permutation_null, preflight, selection_gap, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

LAMBDA = 0.35
BLOCK = tuple(range(8700001, 8700049))
TRAIN, TEST = BLOCK[:24], BLOCK[24:]
KS = (1, 3)
N_NULL = 20_000
N_BOOT = 4_000
N_PLACEBO = 300
DESIGN_FROZEN_AT = "b9115292"
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/expanded_signal_search/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)
    opts = options()
    seeds = list(BLOCK)
    tr = [seeds.index(s) for s in TRAIN]
    te = [seeds.index(s) for s in TEST]

    # ---- PRE-FLIGHT ON A BURNED SEED, before the virgin block is touched -------------------
    env = make_env()
    env.reset(seed=8600001)
    sim = env.unwrapped.sim
    live = {"demand_process": getattr(sim, "demand_process", None),
            "strategic_buffer_release_mode": getattr(sim, "strategic_buffer_release_mode", None),
            "inventory_replenishment_lead_time":
                float(getattr(sim, "inventory_replenishment_lead_time", 0.0))}
    reset_now = float(sim.env.now)
    env.close()
    pre = preflight(probe=lambda o: play(o, 8600001)["L"], options=opts,
                    reset_now=reset_now, horizon=MAX_STEPS * STEP_HOURS,
                    scenario=live, expected_scenario=SCENARIO)
    pre_ok = summarise(pre)["all_passed"]
    print(f"  pre-vuelo sobre semilla quemada: {'OK' if pre_ok else 'FALLA'}")
    if not pre_ok:
        print("  no se abre el bloque")
        return 1

    print(f"  abriendo {len(seeds)} semillas virgenes x {len(opts)} calendarios = "
          f"{len(seeds) * len(opts)} episodios")
    L = np.zeros((len(seeds), len(opts)))
    IH = np.zeros_like(L)
    for i, s in enumerate(seeds):
        for j, o in enumerate(opts):
            r = play(o, s)
            L[i, j], IH[i, j] = r["L"], r["inventory_hours"]
        if (i + 1) % 12 == 0:
            print(f"    {i + 1}/{len(seeds)} tapes")
    max_ih = float(IH.max()) or 1.0
    J = L + LAMBDA * (IH / max_ih)

    # ---- STEP 1: does the ceiling replicate? -----------------------------------------------
    fixed = int(np.argmin(J[tr].mean(axis=0)))
    open_loop = J[te, fixed]
    clair = J[te].min(axis=1)
    gap = float((open_loop - clair).mean())
    nul = permutation_null(J, tr, te, n_draws=N_NULL, rng=rng, statistic=selection_gap)
    print(f"\n  techo en test: {gap:+.6f} · nulo media {nul['null_mean']:+.6f} "
          f"p95 {nul['null_p95']:+.6f} · p = {nul['p_value']:.4f}")
    replicated = nul["p_value"] < 0.05

    # ---- STEP 2: the signal search, only meaningful if it replicated -----------------------
    feats = {s: prefix_features(s) for s in seeds}
    names = sorted(next(iter(feats.values())))
    X = np.array([[feats[s][n] for n in names] for s in seeds], dtype=float)

    def apply_map(col: np.ndarray, k: int):
        picks = []
        for i in te:
            near = sorted(tr, key=lambda t: abs(col[t] - col[i]))[:k]
            votes = [int(J[t].argmin()) for t in near]
            picks.append(max(set(votes), key=votes.count))
        return np.array([J[i, p] for i, p in zip(te, picks)]), picks

    rows = {}
    for k in KS:
        for c, name in enumerate(names):
            col = X[:, c]
            key = f"{name}|k{k}"
            if np.std(col) < 1e-12:
                rows[key] = {"usable": False, "sd": 0.0}
                continue
            arm, picks = apply_map(col, k)
            d = open_loop - arm
            boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                             for _ in range(N_BOOT)])
            plac = []
            for _ in range(N_PLACEBO):
                sh = col.copy()
                rng.shuffle(sh)
                a, _ = apply_map(sh, k)
                plac.append(float((open_loop - a).mean()))
            plac = np.asarray(plac)
            rows[key] = {
                "usable": True, "feature": name, "k": k, "sd": float(np.std(col)),
                "gain_vs_open_loop": float(d.mean()),
                "lcb95": float(np.percentile(boot, 2.5)),
                "ucb95": float(np.percentile(boot, 97.5)),
                "placebo_mean_gain": float(plac.mean()),
                "p_vs_placebo": float((plac >= float(d.mean())).mean()),
                "share_of_ceiling": float(d.mean() / gap) if gap > 0 else 0.0,
            }

    usable = {k: v for k, v in rows.items() if v.get("usable")}
    order = sorted(usable, key=lambda k: usable[k]["p_vs_placebo"])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(usable) - rank) * usable[k]["p_vs_placebo"]))
        usable[k]["holm_adjusted_p"] = running
    winners = [k for k, v in usable.items() if v["lcb95"] > 0 and v["holm_adjusted_p"] < 0.05]
    best = max(usable, key=lambda k: usable[k]["gain_vs_open_loop"]) if usable else None

    falsifiers = {
        **pre,
        "f5_ceiling_replicates_on_the_new_block": lt(
            nul["p_value"], 0.05,
            "THE gate that comes first. If the clairvoyant gap does not beat its interaction null "
            "here, the twelve-tape ceiling was a twelve-tape artifact, nothing about signals may "
            "be read, and the four failed conversion attempts were chasing something absent",
            gap=gap, null_mean=nul["null_mean"], null_p95=nul["null_p95"],
            p_value=nul["p_value"]),
        "f6_multiplicity_over_both_map_families": ge(
            float(len(usable)), float(len(KS) * (len(names) - 1)),
            "26 tests are run and the best reported; correcting fewer than were tried is the "
            "metric shopping this project already priced",
            n_tests=len(usable), map_families=list(KS), n_features=len(names)),
        "f7_placebo_keeps_the_mechanism": ge(
            float(N_PLACEBO), 100.0,
            "the placebo applies the SAME map to a shuffled feature, so a gain it also produces "
            "belongs to the mapping and not to the signal -- on twelve tapes two features lost to "
            "their own placebo",
            n_placebo=N_PLACEBO),
        "f8_no_map_exceeds_the_ceiling": ge(
            gap - max((v["gain_vs_open_loop"] for v in usable.values()), default=0.0), -1e-9,
            "a train-fitted map cannot beat a tape-knowing chooser; exceeding the ceiling means it "
            "saw the test tapes",
            best_gain=max((v["gain_vs_open_loop"] for v in usable.values()), default=0.0)),
        "f9_virgin_block_and_frozen_design": ge(
            float(len(set(TRAIN) & set(TEST)) * -1 + 1), 1.0,
            "train and test must not overlap, and the design was frozen before the block was "
            "opened; a rerun on these seeds is a collision by construction",
            block=[BLOCK[0], BLOCK[-1]], design_frozen_at=DESIGN_FROZEN_AT,
            custody=custody_falsifier(seeds, replay_of=None, exclude=args.output)),
        "d1_one_way_door": disclosure(
            "this block is consumed by this run; a defect found now burns it rather than "
            "licensing a rerun, which is the only reason a virgin block is worth more than a "
            "reused one",
            block="8700001-8700048", authorisation=str(args.contract)),
        "d2_fidelity_price": disclosure(
            "release and the 336 h lead time are OUR extensions with no source event; the buffer "
            "price is a declared assumption in endpoint units, not a monetary rate",
            reproduces_thesis=False),
    }
    summary = summarise(falsifiers)
    if not summary["all_passed"] and not replicated:
        verdict = "CEILING_DID_NOT_REPLICATE"
    elif not summary["all_passed"]:
        verdict = "BLOCKED_INSTRUMENT"
    elif winners:
        verdict = "SIGNAL_FOUND"
    else:
        verdict = "NO_PREFIX_SIGNAL_CAPTURES_THE_CEILING_AT_N48"

    print(f"\n  {'rasgo|k':>28} {'ganancia':>10} {'lcb95':>10} {'placebo':>10} {'holm':>7} {'cuota':>8}")
    for k in sorted(usable, key=lambda k: -usable[k]["gain_vs_open_loop"])[:10]:
        v = usable[k]
        print(f"  {k:>28} {v['gain_vs_open_loop']:10.6f} {v['lcb95']:10.6f} "
              f"{v['placebo_mean_gain']:10.6f} {v['holm_adjusted_p']:7.3f} "
              f"{v['share_of_ceiling']:8.1%}")
    print(f"\n  veredicto: {verdict}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:52s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "expanded_signal_search_v1",
        "claim_status": verdict,
        "scope": "VIRGIN_BLOCK_CONSUMED_DESIGN_FROZEN_BEFORE_OPENING",
        "run_role": "CEILING_REPLICATION_AND_SIGNAL_SEARCH",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "authorisation": str(args.contract), "design_frozen_at": DESIGN_FROZEN_AT,
        "block": {"range": [BLOCK[0], BLOCK[-1]], "train": list(TRAIN), "test": list(TEST),
                  "status_after_this_run": "CONSUMED"},
        "lambda": LAMBDA, "scenario": SCENARIO, "live_scenario": live,
        "ceiling": {"gap": gap, "open_loop_J": float(open_loop.mean()),
                    "clairvoyant_J": float(clair.mean()), "fixed_option": list(opts[fixed]),
                    "null": dict(nul), "replicated": replicated},
        "options": [list(o) for o in opts], "features": names,
        "tape_features": {str(s): feats[s] for s in seeds},
        "L_matrix": L.tolist(), "inventory_hours_matrix": IH.tolist(),
        "max_inventory_hours": max_ih,
        "rows": rows, "winners": winners, "best": best,
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/signal_search/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
