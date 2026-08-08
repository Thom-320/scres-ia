#!/usr/bin/env python3
"""Two comparators the marginal replay should have been, scored against the same transfer arms.

WHAT THIS REPAIRS. `results/comparator_drift/result.json` established that the comparator the paper
calls a state-blind marginal replay has two defects. Its visit histogram is created once, before the
seed loop, and updated inside it -- so (1) the 24 visits the carrier just chose on the case being
scored are in the histogram before the replay samples from it, and (2) the histogram grows from
4,608 pseudocounts to 8,640 real visits over the run, making the comparator nearly uniform sampling
at the start and an informed histogram at the end.

THE TWO ARMS, AND WHY BOTH.

`frozen_prior` counts the carrier's visits PER FACTOR LEVEL during the base-288 training phase and
freezes them before the extended grid is touched, expanding to the 4,608 configurations by product
with the two new factors uniform -- the same extension rule `extend_state` uses for the carrier, so
the only difference between arms is what is kept, never how it is spread. It contains no part of the
target case, does not accumulate during evaluation, and can be deployed WITHOUT running the carrier
on the case being scored. It is, literally, the transportable level-frequency prior whose sufficiency
this project claimed and then had to retract for lack of identification. This arm identifies it.

`loo_marginal` is the original comparator with `visits += 1` moved to AFTER the replay instead of
before. It keeps cross-case accumulation and removes exactly the current-case contamination, so the
pair separates the two defects instead of confounding them.

THE GRADE IS FIXED AND CANNOT IMPROVE. Seeds 8200001-8200060 are burned -- this block was opened for
the confirmation. This run is REPLAY/DEVELOPMENT. It cannot raise RQ2a, replace the preregistered
confirmation, or name a different winner. It qualifies an interpretation, and that is all it may do.

Preregistration: docs/PREREGISTRO_COMPARADOR_REPARADO_2026-08-08.md
Physics, budget, contexts, normaliser and RNG streams are imported unchanged from the original
runner; nothing here re-implements the estimand.
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

import scripts.run_grid_transfer_v1 as G  # noqa: E402
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

SOURCE = Path("results/grid_transfer_confirmation_v2/result.json")
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py",
           "scripts/run_grid_transfer_v1.py")
BURNED_BLOCK = (8_200_001, 8_200_060)
N_BOOT, BOOT_SEED = 5_000, 20260808
MODES = ("transfer", "cold", "marginal", "loo_marginal", "frozen_prior")


def level_prior(base_visits: list[int]) -> np.ndarray:
    """Factor-level visit counts from the base grid, expanded over the 4,608 configurations.

    Laplace-smoothed by one so every extended configuration keeps positive mass -- f5 fails
    otherwise, and a comparator that cannot reach 4,320 of the 4,608 configurations would be
    crippled rather than fair. The two new factors carry a flat count, which is the frequency
    analogue of `extend_state` giving their new levels a zero UCB count.
    """
    counts = {n: np.zeros(len(G.EXT_FACTORS[n])) for n in G.EXT_NAMES}
    for i in base_visits:
        cfg = G.BASE_CONFIGS[i]
        for n, v in cfg.items():
            counts[n][G.EXT_FACTORS[n].index(v)] += 1.0
    p = np.ones(len(G.EXT_CONFIGS))
    for j, cfg in enumerate(G.EXT_CONFIGS):
        for n in G.EXT_NAMES:
            p[j] *= counts[n][G.EXT_FACTORS[n].index(cfg[n])] + 1.0
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-cache", type=Path,
                    default=Path("results/surface_cache/garrido_transfer_confirmation_v2_base"))
    ap.add_argument("--ext-cache", type=Path,
                    default=Path("results/surface_cache/garrido_transfer_confirmation_v2_ext"))
    ap.add_argument("--budget", type=int, default=G.BUDGET)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_COMPARADOR_REPARADO_2026-08-08.md"))
    ap.add_argument("--max-seeds", type=int, default=0, help="smoke only; 0 means all")
    ap.add_argument("--out", type=Path, default=Path("results/comparator_repair/result.json"))
    args = ap.parse_args()

    base, base_ctx, seeds, *_ = G.load(args.base_cache, len(G.BASE_CONFIGS), "wrap288_v1")
    ext, ext_ctx, ext_seeds, *_ = G.load(args.ext_cache, len(G.EXT_CONFIGS),
                                         "wrap288_compat_extended_v1")
    contexts = [c for c in G.CONTEXT_ORDER if c in base_ctx and c in ext_ctx]
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]

    outside = [s for s in seeds if not BURNED_BLOCK[0] <= s <= BURNED_BLOCK[1]]
    print(f"  {len(seeds)} semillas x {len(contexts)} contextos x {len(MODES)} modos "
          f"x {len(G.ARMS)} familias")

    rows = {f"{a}_{m}": [] for a in G.ARMS for m in MODES}
    visits_orig = {a: np.ones(len(G.EXT_CONFIGS)) for a in G.ARMS}
    visits_loo = {a: np.ones(len(G.EXT_CONFIGS)) for a in G.ARMS}
    frozen_mass_zero, budget_failures = [], []
    started = time.perf_counter()

    for r, seed in enumerate(seeds):
        for kind in G.ARMS:
            trained = G.fresh_state(kind, G.BASE_FACTORS)
            rng = np.random.default_rng(90_000 + r)
            base_visits: list[int] = []
            for ctx in contexts:                       # train on the 288 grid -- unchanged
                s = G.Surface(base[(ctx, seed)])
                G.build(kind, trained, "base")(s, rng, args.budget)
                base_visits.extend(s.visited)

            # Frozen BEFORE the extended grid is touched. That ordering is the whole point.
            prior = level_prior(base_visits)
            if float(prior.min()) <= 0.0:
                frozen_mass_zero.append((kind, seed))

            carried = G.extend_state(kind, trained, G.EXT_FACTORS)
            aucs = {m: [] for m in MODES}
            for ctx in contexts:
                s = G.Surface(ext[(ctx, seed)])
                G.build(kind, carried, "ext")(s, np.random.default_rng(70_000 + r), args.budget)
                aucs["transfer"].append(s.auc(args.budget))

                cold = G.fresh_state(kind, G.EXT_FACTORS)
                s2 = G.Surface(ext[(ctx, seed)])
                G.build(kind, cold, "ext")(s2, np.random.default_rng(70_000 + r), args.budget)
                aucs["cold"].append(s2.auc(args.budget))

                # loo FIRST, on the histogram that does NOT yet contain this case.
                s4 = G.Surface(ext[(ctx, seed)])
                G.marginal_replay(visits_loo[kind], s4,
                                  np.random.default_rng(70_000 + r), args.budget)
                aucs["loo_marginal"].append(s4.auc(args.budget))

                s5 = G.Surface(ext[(ctx, seed)])
                G.marginal_replay(prior, s5, np.random.default_rng(70_000 + r), args.budget)
                aucs["frozen_prior"].append(s5.auc(args.budget))

                # Original ordering, reproduced exactly: the transfer visits enter first.
                for i in s.visited:
                    visits_orig[kind][i] += 1.0
                s3 = G.Surface(ext[(ctx, seed)])
                G.marginal_replay(visits_orig[kind], s3,
                                  np.random.default_rng(70_000 + r), args.budget)
                aucs["marginal"].append(s3.auc(args.budget))
                for i in s.visited:
                    visits_loo[kind][i] += 1.0        # loo catches up AFTER its own replay

                for m, surf in (("transfer", s), ("cold", s2), ("marginal", s3),
                                ("loo_marginal", s4), ("frozen_prior", s5)):
                    if len(surf.visited) != args.budget:
                        budget_failures.append((kind, m, ctx, seed, len(surf.visited)))
            for m in MODES:
                rows[f"{kind}_{m}"].append(float(np.mean(aucs[m])))
        print(f"  réplica {r + 1}/{len(seeds)} ({time.perf_counter() - started:.0f}s)", flush=True)

    boot_rng = np.random.default_rng(BOOT_SEED)

    def boot(d: np.ndarray) -> dict:
        st = d[boot_rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
        return {"mean": float(d.mean()), "lcb95": float(np.percentile(st, 2.5)),
                "ucb95": float(np.percentile(st, 97.5)), "n": int(d.size)}

    order = np.arange(len(seeds), dtype=float)

    def rho(v: np.ndarray) -> float:
        return float(np.corrcoef(order, v)[0, 1]) if len(v) > 2 else float("nan")

    contrasts, drift = {}, {}
    for kind in G.ARMS:
        t = np.asarray(rows[f"{kind}_transfer"])
        contrasts[kind] = {f"vs_{m}": boot(np.asarray(rows[f"{kind}_{m}"]) - t)
                           for m in ("cold", "marginal", "loo_marginal", "frozen_prior")}
        drift[kind] = {m: rho(np.asarray(rows[f"{kind}_{m}"])) for m in MODES}

    src = json.loads((ROOT / SOURCE).read_text()) if (ROOT / SOURCE).exists() else {}
    sealed_arms = src.get("per_arm", {})
    full_run = not args.max_seeds and len(seeds) == len(sealed_arms.get("ucb1_transfer", []) or [1])
    reproduced = {}
    if full_run:
        for kind in G.ARMS:
            for m in ("transfer", "cold", "marginal"):
                a = [round(x, 12) for x in sealed_arms.get(f"{kind}_{m}", [])]
                b = [round(x, 12) for x in rows[f"{kind}_{m}"]]
                reproduced[f"{kind}_{m}"] = a == b

    n_frozen_drifting = sum(1 for k in G.ARMS if abs(drift[k]["frozen_prior"]) > 0.25)
    n_loo_drifting = sum(1 for k in G.ARMS if drift[k]["loo_marginal"] < -0.15)
    loo_gap = {k: abs(float(np.mean(rows[f"{k}_loo_marginal"]))
                      - float(np.mean(rows[f"{k}_marginal"]))) for k in G.ARMS}

    falsifiers = {
        "f1_transfer_and_cold_reproduce_the_sealed_values_exactly": {
            "passed": (not full_run) or all(reproduced.values()),
            "checked": full_run, "detail": reproduced,
            "why_it_can_fail": ("adding arms would have perturbed the existing ones if any RNG "
                                "stream were shared; each arm draws its own default_rng")},
        "f2_frozen_prior_does_not_drift_with_run_order": {
            "passed": n_frozen_drifting == 0, "rho": {k: drift[k]["frozen_prior"] for k in G.ARMS},
            "threshold_abs_rho": 0.25,
            "why_it_can_fail": "if the frozen prior drifts it is not frozen and the arm is not what it claims"},
        "f3_loo_still_drifts_with_run_order": {
            "passed": n_loo_drifting >= 2, "rho": {k: drift[k]["loo_marginal"] for k in G.ARMS},
            "threshold_rho": -0.15, "n_drifting": n_loo_drifting,
            "why_it_can_fail": ("removing the current case would have removed the drift, which "
                                "would refute the cross-case-accumulation diagnosis outright")},
        "f4_loo_differs_from_the_original_by_no_more_than_the_mass_bound": {
            "passed": all(v < 0.01 for v in loo_gap.values()), "gap": loo_gap, "bound": 0.01,
            "why_it_can_fail": "a large gap would mean the 0.18-0.52% mass arithmetic is wrong"},
        "f5_frozen_prior_puts_mass_on_every_extended_configuration": {
            "passed": not frozen_mass_zero, "zero_mass_cases": frozen_mass_zero[:5],
            "why_it_can_fail": "a prior with zeros cannot reach 4,320 of the 4,608 configurations"},
        "f6_budgets_are_matched": {
            "passed": not budget_failures, "failures": budget_failures[:5]},
        "f7_no_seed_outside_the_burned_block": {
            "passed": not outside, "outside": outside[:5], "block": list(BURNED_BLOCK)},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    primary = contrasts["ucb1"]["vs_frozen_prior"]
    verdict = ("HALTED_FALSIFIER_FAILED" if not falsifiers["all_passed"] else
               "UCB1_BEATS_A_FROZEN_EX_ANTE_LEVEL_PRIOR" if primary["lcb95"] > 0 else
               "A_FROZEN_LEVEL_PRIOR_BEATS_UCB1" if primary["ucb95"] < 0 else
               "UCB1_INDISTINGUISHABLE_FROM_A_FROZEN_EX_ANTE_LEVEL_PRIOR")

    payload = {
        "schema_version": "comparator_repair_v1",
        "claim_status": verdict,
        "scope": "REPLAY_ON_THE_BURNED_CONFIRMATION_BLOCK_NO_NEW_SEEDS_NO_ADJUDICATION",
        "run_role": "REPLAY_REANALYSIS",
        "registration_status": "PREREGISTERED_REPAIR_ON_BURNED_SEEDS_CANNOT_RAISE_RQ2A",
        "endpoint": "auc_regret_norm",
        "estimand": "paired per-seed (comparator - transfer); primary is ucb1 vs frozen_prior",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "source": str(SOURCE), "source_self_sha256": src.get("self_sha256"),
        "seeds": seeds, "contexts": contexts, "budget": args.budget,
        "n_ext_configs": len(G.EXT_CONFIGS), "modes": list(MODES),
        "mean_auc": {k: float(np.mean(v)) for k, v in rows.items()},
        "per_arm": rows,
        "contrasts": contrasts,
        "drift_rho_with_run_order": drift,
        "reproduces_sealed_arms": reproduced,
        "what_this_cannot_do": [
            "raise the grade of RQ2a -- the seeds are burned and this is a replay",
            "replace the preregistered confirmation or name a different winner",
            "authorise a third comparator if the reading is unwelcome",
        ],
        "falsifiers": falsifiers,
    }
    seal_and_write(payload, ROOT / args.out, contract=ROOT / args.contract,
                   reference=ROOT / SOURCE)

    print(f"\n{'familia':10}{'vs cold':>22}{'vs marginal':>22}{'vs loo':>22}{'vs frozen':>22}")
    for k in G.ARMS:
        c = contrasts[k]
        cells = "".join(f"{c[f'vs_{m}']['mean']:>+10.5f}[{c[f'vs_{m}']['lcb95']:>+9.5f}]"
                        for m in ("cold", "marginal", "loo_marginal", "frozen_prior"))
        print(f"{k:10}{cells}")
    print(f"\nveredicto: {verdict}")
    print(f"falsadores: {'todos pasan' if falsifiers['all_passed'] else 'FALLO'}")
    for n, v in falsifiers.items():
        if isinstance(v, dict) and not v["passed"]:
            print(f"  FALLA {n}")
    print(f"-> {args.out}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
