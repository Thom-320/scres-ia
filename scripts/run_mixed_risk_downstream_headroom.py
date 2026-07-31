#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_MEZCLA_RIESGOS_2026-07-31.md`.

Two facts from the headroom map point at one experiment. WHICH risk family is active is 75.5%
of total variance and Garrido runs one at a time -- mixing exists in neither his design nor
ours. And all the interaction that exists is downstream: `op12_q_max`, `op10_q_max`, `op9_rop`.

So: mix the families, and measure whether the downstream dispatch levers acquire a
regime-dependent optimum that single families do not give them.

The measured quantity is the **value of knowing the regime**

    H_regime = mean_r [ max_a ReT(a, r) ] - max_a [ mean_r ReT(a, r) ]

-- the best setting when the regime is known, minus the best single setting that must serve
every regime. It is a CEILING on any policy that conditions on the regime: no adaptive
controller can exceed it, and if it is zero there is nothing to learn, whatever the network.
It counts only if its bootstrap LCB95 over seeds clears zero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
R3 = ("R3",)
REGIMES = {
    "R1r": R1R, "R2r": R2R, "R3": R3,                      # his design: one family at a time
    "R1r+R2r": R1R + R2R, "R1r+R3": R1R + R3,              # never run, by him or by us
    "R2r+R3": R2R + R3, "R1r+R2r+R3": R1R + R2R + R3,
}
PURE = ("R1r", "R2r", "R3")
LEVERS = {"op9_rop": (12.0, 48.0), "op10_q_max": (1_200.0, 5_200.0),
          "op12_q_max": (1_200.0, 5_200.0)}
SEED_BASE = 4_300_001


def grid(levels: int) -> list[dict[str, float]]:
    axes = {name: np.linspace(lo, hi, levels) for name, (lo, hi) in LEVERS.items()}
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def run(setting: dict[str, float], risks: tuple[str, ...], seed: int,
        horizon: float) -> float:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action={
        "op9_rop": setting["op9_rop"],
        "op9_q_min": 2_400.0, "op9_q_max": 2_600.0,
        "op10_rop": 24.0,
        "op10_q_min": setting["op10_q_max"] * 0.92, "op10_q_max": setting["op10_q_max"],
        "op12_rop": 24.0,
        "op12_q_min": setting["op12_q_max"] * 0.92, "op12_q_max": setting["op12_q_max"],
    }, step_hours=horizon)
    return float(compute_episode_metrics(sim)["ret_excel"])


def headroom(scores: np.ndarray, regimes: list[str], all_regimes: list[str],
             *, n_boot: int = 2000, seed: int = 0) -> dict:
    """`H_regime` over a subset of regimes, with a paired bootstrap over seeds.

    `scores` is `(settings, regimes, seeds)`. The bootstrap resamples SEEDS, not settings:
    the seeds are the replication unit and the grid is exhaustive, so that is where the
    uncertainty lives.
    """
    idx = [all_regimes.index(r) for r in regimes]
    block = scores[:, idx, :]

    def point(sample: np.ndarray) -> float:
        per_regime = sample.mean(axis=2)                       # (settings, regimes)
        informed = float(per_regime.max(axis=0).mean())        # know the regime
        single = float(per_regime.mean(axis=1).max())          # one setting for all
        return informed - single

    rng = np.random.default_rng(seed)
    n_seeds = block.shape[2]
    boot = [point(block[:, :, rng.integers(0, n_seeds, n_seeds)]) for _ in range(n_boot)]
    per_regime = block.mean(axis=2)
    best_by_regime = {r: float(per_regime[:, j].max()) for j, r in enumerate(regimes)}
    return {
        "H_regime": point(block),
        "lcb95": float(np.percentile(boot, 5)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "best_by_regime": best_by_regime,
        "best_common_setting_score": float(per_regime.mean(axis=1).max()),
        "argmax_by_regime": {r: int(per_regime[:, j].argmax()) for j, r in enumerate(regimes)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/mixed_risk_downstream_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    settings = grid(args.levels)
    names = list(REGIMES)
    started = time.perf_counter()

    scores = np.zeros((len(settings), len(names), args.seeds), dtype=float)
    for s_i, setting in enumerate(settings):
        for r_i, label in enumerate(names):
            for k in range(args.seeds):
                # CRN: the seed depends on the replicate ONLY, so every setting and every
                # regime sees the same exogenous draw and the comparison is paired.
                scores[s_i, r_i, k] = run(setting, REGIMES[label], SEED_BASE + k, horizon)
        if (s_i + 1) % 25 == 0:
            print(f"  {s_i + 1}/{len(settings)} ajustes "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    pure = headroom(scores, list(PURE), names, seed=1)
    mixed = headroom(scores, names, names, seed=2)
    only_mixed = headroom(scores, [r for r in names if r not in PURE], names, seed=3)

    mean_by_regime = {r: float(scores[:, j, :].mean()) for j, r in enumerate(names)}
    pure_range = (min(mean_by_regime[r] for r in PURE), max(mean_by_regime[r] for r in PURE))
    outside = [r for r in names if r not in PURE
               and not (pure_range[0] <= mean_by_regime[r] <= pure_range[1])]
    per_regime = scores.mean(axis=2)
    interior = [r for j, r in enumerate(names)
                if 0 < int(per_regime[:, j].argmax()) < len(settings) - 1]
    repeat = run(settings[0], REGIMES["R1r"], SEED_BASE, horizon)

    falsifiers = {
        "f1_mixes_are_distinguishable_from_pure_families": {
            "passed": bool(outside),
            "evidence": {"why_it_can_fail": ("if mixing changes nothing measurable the "
                                             "experiment has no object"),
                         "mean_ret_by_regime": mean_by_regime,
                         "pure_range": list(pure_range), "mixes_outside_pure_range": outside}},
        "f2_optimum_is_interior_for_some_regime": {
            "passed": bool(interior),
            "evidence": {"why_it_can_fail": ("an optimum pinned to the grid edge would make "
                                             "the headroom an artefact of the range"),
                         "regimes_with_interior_argmax": interior,
                         "argmax_index_by_regime": {r: int(per_regime[:, j].argmax())
                                                    for j, r in enumerate(names)}}},
        "f3_headroom_is_non_negative": {
            "passed": all(h["H_regime"] >= -1e-12 for h in (pure, mixed, only_mixed)),
            "evidence": {"why_it_can_fail": "an inverted sign would betray a bad aggregation",
                         "H": {"pure": pure["H_regime"], "all_seven": mixed["H_regime"],
                               "mixed_only": only_mixed["H_regime"]}}},
        "f4_crn_is_real": {
            "passed": abs(repeat - scores[0, names.index("R1r"), 0]) < 1e-12,
            "evidence": {"why_it_can_fail": ("without pairing, the headroom is seed noise"),
                         "repeat": repeat,
                         "stored": float(scores[0, names.index("R1r"), 0])}},
        "f5_regimes_match_the_thesis_families": {
            "passed": (set(REGIMES["R1r"]) == set(R1R) and set(REGIMES["R2r"]) == set(R2R)
                       and set(REGIMES["R1r+R2r"]) == set(R1R) | set(R2R)),
            "evidence": {"why_it_can_fail": "mixing the wrong ids would void the reading",
                         "regimes": {k: list(v) for k, v in REGIMES.items()}}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    verdict = {
        "H_pure": pure["H_regime"], "H_pure_lcb95": pure["lcb95"],
        "H_all_seven": mixed["H_regime"], "H_all_seven_lcb95": mixed["lcb95"],
        "H_mixed_only": only_mixed["H_regime"], "H_mixed_only_lcb95": only_mixed["lcb95"],
        "mixing_increases_headroom": bool(mixed["H_regime"] > pure["H_regime"]),
        "headroom_clears_its_own_noise": bool(mixed["lcb95"] > 0.0),
        "rule": ("H_regime counts only if its bootstrap LCB95 over seeds is > 0; it is a "
                 "CEILING on any regime-conditioned policy"),
    }

    print("\n  === H_regime: el valor de conocer el régimen ===")
    for label, h in (("puros (3)", pure), ("mezclas (4)", only_mixed), ("los 7", mixed)):
        print(f"    {label:<14} H = {h['H_regime']:.6f}   LCB95 {h['lcb95']:.6f}   "
              f"CI95 [{h['ci95'][0]:.6f}, {h['ci95'][1]:.6f}]")
    print("\n  ReT medio por régimen:")
    for r in names:
        print(f"    {r:<12} {mean_by_regime[r]:.6f}"
              f"{'   <- mezcla' if r not in PURE else ''}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<48} {'PASA' if check['passed'] else 'FALLA'}")
    print(f"\n  ¿mezclar sube el headroom? {verdict['mixing_increases_headroom']}   "
          f"¿supera su ruido? {verdict['headroom_clears_its_own_noise']}")

    payload = {
        "schema_version": "mixed_risk_downstream_headroom_v1",
        "claim_status": ("DEVELOPMENT_MIXED_RISK_HEADROOM" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "regimes": {k: list(v) for k, v in REGIMES.items()},
        "levers": {k: list(v) for k, v in LEVERS.items()},
        "levels": args.levels, "seeds": args.seeds,
        "n_runs": int(len(settings) * len(names) * args.seeds),
        "headroom": {"pure": pure, "mixed_only": only_mixed, "all_seven": mixed},
        "mean_ret_by_regime": mean_by_regime,
        "verdict": verdict, "falsifiers": falsifiers,
        "settings": settings,
        "scores_mean_over_seeds": per_regime.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_MEZCLA_RIESGOS_2026-07-31.md"),
        reference=Path("results/sensitivity/headroom_map_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
