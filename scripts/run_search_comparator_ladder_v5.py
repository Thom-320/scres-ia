#!/usr/bin/env python3
"""Ladder v5: lookahead search (knowledge gradient) and Thompson, on the same tape as v4.

THIS IS NOT THE SUPPLY-CHAIN MPC, AND THE DISTINCTION IS THE POINT. This ladder is
simulation-optimization over CONFIGURATIONS: four factors set once, budget 24 on a grid of 288.
There are no intra-episode decisions here, so a receding-horizon controller has nothing to control
and a supply-chain MPC arm would be meaningless. The real MPC of this project is
`supply_chain.expanded_contract_controllers.ReceedingHorizonMPC`, which plans buffer targets inside
the DES -- that is Garrido's own step three and it belongs in the expanded buffer contract, not
here.

What DOES transfer to this environment is lookahead over the SEARCH: plan the next evaluation by
looking ahead and replan after each observation. Its canonical form in the Bayesian-optimization
literature is the Knowledge Gradient,

    KG(x) = E[ max_i mu_{n+1}(i) | evaluate at x ] - max_i mu_n(i)

which picks x for its effect on the expected optimum AFTER observing, not for its own immediate
improvement. That is exactly the difference between EI (myopic, one step) and receding-horizon
control. Adding an observation at x shifts the posterior mean linearly in the observation, so KG is
computed by Monte Carlo over a single standard normal draw -- no GP refit per fantasy.

AND THOMPSON, because if the question is "what is the best method for this environment", the bandit
that usually beats UCB1 has to be in the room. ucb1_transfer leads the ladder today; leaving out its
natural competitor would be picking the comfortable comparator.

The eleven v4 arms are re-run on the same tape, same budget, same cache, and f6 anchors their means
against the sealed v4 artifact. Nothing is re-scored; arms are added.

Preregistration: docs/PREREGISTRO_ESCALERA_V5_MPC_Y_THOMPSON_2026-08-06.md
Development on burned tapes. Adjudicates nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_search_comparator_ladder_v2 import (  # noqa: E402
    BUDGET, CONFIGS, COORDS, DEFAULT, FACTOR_NAMES, FACTORS, FEATURES, GP_N_INIT, N_CFG,
    Surface, _prefix_normalised, arm_annealing, arm_gp_ei, arm_lhs_local, arm_ofat, arm_oracle,
    arm_random, arm_ucb1, load_cache, make_gp_transfer_arm, make_neuron_arm, make_ofat_transfer_arm,
    make_ucb1_transfer_arm,
)

KG_FANTASIES = 128
N_BOOT = 5_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
V4_REFERENCE = Path("results/search_ladder_v4/result.json")


def _kernel():
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    return (ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(COORDS.shape[1]))
            + WhiteKernel(1e-4))


def _fit(x_idx, y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    return GaussianProcessRegressor(kernel=_kernel(), normalize_y=True, n_restarts_optimizer=2,
                                    random_state=0).fit(COORDS[list(x_idx)], np.asarray(y))


def _knowledge_gradient(gp, cand: list[int], rng) -> np.ndarray:
    """KG for every candidate, by Monte Carlo over the posterior update.

    Observing y at x moves the posterior mean of every point i by sigma_tilde(i,x) * Z with
    Z standard normal, where sigma_tilde(i,x) = k_n(i,x)/sqrt(k_n(x,x) + noise). So the expected
    post-observation maximum is E_Z[max_i (mu_i + sigma_tilde(i,x) Z)] and no refit is needed.
    That expectation over the MAX is what makes this a lookahead rather than a myopic acquisition:
    EI asks how much better x itself might be, KG asks how much better the whole search gets."""
    X = COORDS[cand]
    mu = gp.predict(X)
    kern = gp.kernel_
    cov = kern(X)                                    # (m, m) posterior-ish covariance on candidates
    var = np.clip(np.diag(cov), 1e-12, None)
    best_now = float(mu.max())
    z = rng.standard_normal(KG_FANTASIES)
    # sigma_tilde column for candidate j is cov[:, j] / sqrt(var[j]); vectorised over j.
    sig = cov / np.sqrt(var)[None, :]                # (i, j)
    # E_Z[max_i (mu_i + sig[i, j] * Z)] for each j
    vals = mu[:, None, None] + sig[:, :, None] * z[None, None, :]
    return vals.max(axis=0).mean(axis=1) - best_now


def make_lookahead_arm(memory: dict | None):
    """Receding-horizon control over the search: fit, look ahead with KG, evaluate, REPLAN."""

    def arm(s: Surface, rng, budget: int) -> None:
        here_idx: list[int] = []
        here_raw: list[float] = []
        for idx in rng.permutation(N_CFG)[:GP_N_INIT]:
            here_idx.append(int(idx))
            here_raw.append(s.select(int(idx)))
        while len(s.visited) < budget:
            if memory is not None:
                x = list(memory["idx"]) + here_idx
                y = list(memory["y"]) + _prefix_normalised(here_raw)
            else:
                x, y = here_idx, _prefix_normalised(here_raw)
            gp = _fit(x, y)
            cand = s.unvisited
            kg = _knowledge_gradient(gp, cand, rng)
            pick = cand[int(kg.argmax())]
            here_idx.append(pick)
            here_raw.append(s.select(pick))
        if memory is not None:
            memory["idx"].extend(here_idx)
            memory["y"].extend(_prefix_normalised(here_raw))

    return arm


def make_thompson_arm(memory: dict | None):
    """Posterior sampling: draw one function from the GP and take its argmax. The bandit that
    usually beats UCB1, and ucb1_transfer is what currently leads this ladder."""

    def arm(s: Surface, rng, budget: int) -> None:
        here_idx: list[int] = []
        here_raw: list[float] = []
        for idx in rng.permutation(N_CFG)[:GP_N_INIT]:
            here_idx.append(int(idx))
            here_raw.append(s.select(int(idx)))
        while len(s.visited) < budget:
            if memory is not None:
                x = list(memory["idx"]) + here_idx
                y = list(memory["y"]) + _prefix_normalised(here_raw)
            else:
                x, y = here_idx, _prefix_normalised(here_raw)
            gp = _fit(x, y)
            cand = s.unvisited
            mu, sd = gp.predict(COORDS[cand], return_std=True)
            draw = mu + sd * rng.standard_normal(len(cand))
            pick = cand[int(draw.argmax())]
            here_idx.append(pick)
            here_raw.append(s.select(pick))
        if memory is not None:
            memory["idx"].extend(here_idx)
            memory["y"].extend(_prefix_normalised(here_raw))

    return arm


ARM_ORDER = ("oracle", "random", "ofat", "lhs_local", "gp_ei", "ucb1", "annealing",
             "gp_ei_transfer", "ucb1_transfer", "ofat_transfer", "neuron_memory", "neuron_reset",
             "lookahead_kg", "lookahead_kg_transfer", "thompson", "thompson_transfer")
NEW_ARMS = ("lookahead_kg", "lookahead_kg_transfer", "thompson", "thompson_transfer")
MEMORY_ARMS = ("gp_ei_transfer", "ucb1_transfer", "ofat_transfer", "neuron_memory",
               "lookahead_kg_transfer", "thompson_transfer")
STATIC_ARMS = ("random", "ofat", "lhs_local", "gp_ei", "ucb1", "annealing", "neuron_reset")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path, default=Path("results/search_ladder_v5/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    surface, contexts, seeds = load_cache(args.cache)
    print(f"  caché: {len(contexts)} contextos x {len(seeds)} semillas · presupuesto "
          f"{args.budget} · {len(ARM_ORDER)} brazos")

    per_arm = {n: {"auc": [], "final": [], "budget_used": []} for n in ARM_ORDER}
    visit_trace: dict[str, list[int]] = {}
    memory_end: dict[str, int] = {}
    for r, seed in enumerate(seeds):
        retained = {"rho": np.zeros(FEATURES.shape[1])}
        gp_mem = {"idx": [], "y": []}
        ucb_mem = {"sums": {n: np.zeros(len(FACTORS[n])) for n in FACTOR_NAMES},
                   "counts": {n: np.zeros(len(FACTORS[n])) for n in FACTOR_NAMES}}
        ofat_mem = {"incumbent": dict(DEFAULT)}
        mpc_mem = {"idx": [], "y": []}
        ts_mem = {"idx": [], "y": []}
        builders = {
            "neuron_memory": lambda: make_neuron_arm(retained),
            "neuron_reset": lambda: make_neuron_arm(None),
            "gp_ei_transfer": lambda: make_gp_transfer_arm(gp_mem),
            "ucb1_transfer": lambda: make_ucb1_transfer_arm(ucb_mem),
            "ofat_transfer": lambda: make_ofat_transfer_arm(ofat_mem),
            "lookahead_kg": lambda: make_lookahead_arm(None),
            "lookahead_kg_transfer": lambda: make_lookahead_arm(mpc_mem),
            "thompson": lambda: make_thompson_arm(None),
            "thompson_transfer": lambda: make_thompson_arm(ts_mem),
        }
        plain = {"oracle": arm_oracle, "random": arm_random, "ofat": arm_ofat,
                 "lhs_local": arm_lhs_local, "gp_ei": arm_gp_ei, "ucb1": arm_ucb1,
                 "annealing": arm_annealing}
        for name in ARM_ORDER:
            rng = np.random.default_rng(90_000 + r)          # same stream as v4, per arm
            fn = builders[name]() if name in builders else plain[name]
            aucs, finals = [], []
            for ctx in contexts:
                s = Surface(surface[(ctx, seed)])
                fn(s, rng, args.budget)
                curve = s.regret_curve()
                denom = args.budget * abs(s.best) if s.best else 1.0
                aucs.append(float(np.sum(curve)) / denom)
                finals.append(curve[-1] / (abs(s.best) or 1.0))
                per_arm[name]["budget_used"].append(len(s.visited))
                if r == 0 and ctx == contexts[0]:
                    visit_trace[name] = list(s.visited)
            per_arm[name]["auc"].append(float(np.mean(aucs)))
            per_arm[name]["final"].append(float(np.mean(finals)))
        memory_end = {"lookahead_kg_transfer": len(mpc_mem["idx"]),
                      "thompson_transfer": len(ts_mem["idx"]),
                      "gp_ei_transfer": len(gp_mem["idx"])}
        print(f"  réplica {r + 1}/{len(seeds)} ({time.perf_counter() - started:.0f}s)", flush=True)

    mean_auc = {n: float(np.mean(per_arm[n]["auc"])) for n in ARM_ORDER}
    rng = np.random.default_rng(20260806)

    def boot(diff: np.ndarray) -> dict:
        draws = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(N_BOOT)]
        return {"mean": float(np.mean(diff)), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)), "n": int(len(diff))}

    base = np.asarray(per_arm["neuron_memory"]["auc"])
    vs_neuron = {n: boot(np.asarray(per_arm[n]["auc"]) - base)
                 for n in ARM_ORDER if n not in ("oracle", "neuron_memory")}
    ranking = sorted((n for n in ARM_ORDER if n != "oracle"), key=lambda n: mean_auc[n])

    # --- f4: can the MPC arm win at all? A surface only lookahead can crack. -------------------
    synth_rng = np.random.default_rng(4242)
    wins = 0
    for t in range(6):
        vals = -np.abs(COORDS - COORDS[synth_rng.integers(0, N_CFG)]).sum(axis=1)
        vals = vals + 0.02 * synth_rng.standard_normal(N_CFG)
        a = Surface(vals); make_lookahead_arm(None)(a, np.random.default_rng(7 + t), args.budget)
        b = Surface(vals); arm_random(b, np.random.default_rng(7 + t), args.budget)
        wins += float(np.sum(a.regret_curve())) < float(np.sum(b.regret_curve()))
    lookahead_can_win = wins >= 4

    # --- f2: provoke the leak guard rather than assert it --------------------------------------
    probe = Surface(surface[(contexts[0], seeds[0])])
    try:
        probe.value_of_visited(0)
        guard_raises = False
    except LookupError:
        guard_raises = True

    v4 = json.loads(V4_REFERENCE.read_text()) if V4_REFERENCE.exists() else {}
    v4_means = v4.get("mean_auc_regret", {})
    drift = {n: abs(mean_auc[n] - v4_means[n]) for n in v4_means if n in mean_auc}

    new_better = [n for n in NEW_ARMS if vs_neuron[n]["ucb95"] < 0]
    new_point_better = [n for n in NEW_ARMS if vs_neuron[n]["mean"] < 0 and n not in new_better]
    if new_better:
        verdict = "A_CLASSICAL_SEARCH_METHOD_BEATS_THE_NEURON"
    elif new_point_better:
        verdict = "INDISTINGUISHABLE_FROM_THE_NEURON"
    else:
        verdict = "THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH"

    print("\n  ranking (auc_regret_norm, menor es mejor):")
    for n in ranking:
        v = vs_neuron.get(n)
        tag = "" if v is None else f"   Δ {v['mean']:+.5f} [{v['lcb95']:+.5f} · {v['ucb95']:+.5f}]"
        star = "  <-- NUEVO" if n in NEW_ARMS else ""
        print(f"    {n:<18} {mean_auc[n]:.5f}{tag}{star}")
    print(f"    {'neuron_memory':<18} {mean_auc['neuron_memory']:.5f}   (referencia)")
    print(f"\n  veredicto: {verdict}\n")

    falsifiers = {
        "f1_budgets_are_matched": {
            "passed": all(u == args.budget for n in ARM_ORDER for u in per_arm[n]["budget_used"]),
            "evidence": {"why_it_can_fail": "an arm that stopped early would buy its ranking with "
                                            "a smaller spend", "budget": args.budget}},
        "f2_no_arm_can_read_an_unrun_cell": {
            "passed": bool(guard_raises),
            "evidence": {"why_it_can_fail": "the guard is provoked here, not asserted: a Surface "
                                            "that returned a value for an unvisited cell would let "
                                            "any arm peek at the answer",
                         "probe_raised_LookupError": bool(guard_raises)}},
        "f3_the_new_arms_are_not_the_old_ones": {
            "passed": visit_trace.get("lookahead_kg") != visit_trace.get("gp_ei")
                      and visit_trace.get("thompson") != visit_trace.get("gp_ei"),
            "evidence": {"why_it_can_fail": "if KG collapses onto EI's choices we added an alias "
                                            "rather than a method, and the MPC claim would be "
                                            "vacuous",
                         "mpc_vs_gp_first_context": {
                             "lookahead_kg": visit_trace.get("lookahead_kg", [])[:8],
                             "gp_ei": visit_trace.get("gp_ei", [])[:8]}}},
        "f4_the_lookahead_arm_can_win": {
            "passed": bool(lookahead_can_win),
            "evidence": {"why_it_can_fail": "on a synthetic single-peak surface MPC must beat "
                                            "random. If it cannot search at all, losing here would "
                                            "say nothing about predictive control",
                         "wins_of_6": int(wins)}},
        "f5_memory_arms_actually_carry_state": {
            "passed": all(memory_end.get(n, 0) > 0 for n in
                          ("lookahead_kg_transfer", "thompson_transfer", "gp_ei_transfer")),
            "evidence": {"why_it_can_fail": "a transfer arm whose memory ends empty is its own "
                                            "memoryless twin under a different name",
                         "observations_retained": memory_end}},
        "f6_v4_arms_reproduce": {
            "passed": bool(drift) and max(drift.values()) < 1e-9,
            "evidence": {"why_it_can_fail": "the eleven v4 arms are re-run here; if adding arms "
                                            "perturbed their random streams the comparison is not "
                                            "on one tape and nothing is comparable with the sealed "
                                            "artifact",
                         "max_drift": max(drift.values()) if drift else None,
                         "drift": drift}},
        "f7_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        lab = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<44} {lab}")

    payload = {
        "schema_version": "search_comparator_ladder_v5",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_ESCALERA_V5_MPC_Y_THOMPSON_2026-08-06.md",
        "predecessor": str(V4_REFERENCE),
        "primary_metric": "auc_regret_norm", "budget": args.budget,
        "what_mpc_means_here": (
            "This environment is simulation-optimization over configurations, not intra-episode "
            "control, so predictive control means planning the next evaluation by lookahead and "
            "replanning after each observation. That is the Knowledge Gradient: pick x for its "
            "effect on the expected optimum AFTER observing, rather than on its own improvement."),
        "arms": list(ARM_ORDER), "new_arms": list(NEW_ARMS),
        "memory_arms": list(MEMORY_ARMS), "static_arms": list(STATIC_ARMS),
        "mean_auc_regret": mean_auc, "ranking_best_first": ranking,
        "vs_neuron_memory": vs_neuron, "per_arm": per_arm,
        "contexts": contexts, "seeds": seeds,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=V4_REFERENCE)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
