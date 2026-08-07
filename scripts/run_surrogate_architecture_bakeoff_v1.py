#!/usr/bin/env python3
"""Six surrogate architectures inside the search loop, on one tape, at matched capacity.

THE QUESTION. KAN has exactly one measured advantage and it is not as a policy: as a supervised
surrogate of the design surface it beats a 529-parameter MLP at 532, held out, in all six contexts.
Does that fit advantage convert into a better SEARCH? That is Garrido's Fig. 5 position -- the
approximator sitting between node 3 and node 8.

WHY THE CLASSICAL ARMS ARE IN. Gradient-boosted trees and a polynomial with pairwise interactions
are the two I would bet on: 288 rows and 4 features is where trees usually beat networks, and in
both earlier surface probes the classical beat both networks. Including them is what makes a KAN
win mean something -- beating only an MLP is a straw man.

NO GPU. 288 points and four coordinates; the fit is instant on CPU and the useful parallelism is
across seeds, not across tensors. Said plainly so nobody reads acceleration into this that is not
there.

Preregistration: docs/PREREGISTRO_SURROGATE_ARCHITECTURE_BAKEOFF_2026-08-07.md
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

from run_search_comparator_ladder_v2 import (  # noqa: E402
    BUDGET, COORDS, GP_N_INIT, N_CFG, Surface, arm_random, load_cache, make_neuron_arm,
)

N_BOOT = 5_000
KAN_WIDTH, MLP_WIDTH, FIT_STEPS = 5, 88, 300
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
REFERENCE_ARM = "neuron_5p"


def _fit_torch(kind: str, x: np.ndarray, y: np.ndarray):
    import torch
    torch.manual_seed(9201)
    if kind == "kan":
        from kan import KAN
        model = KAN(width=[x.shape[1], KAN_WIDTH, 1], grid=5, k=3,
                    auto_save=False, save_act=False, symbolic_enabled=False)
    else:
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(x.shape[1], MLP_WIDTH), nn.Tanh(), nn.Linear(MLP_WIDTH, 1))
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        ((model(xt) - yt) ** 2).mean().backward()
        opt.step()
    n = sum(p.numel() for p in model.parameters())

    def predict(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.tensor(z, dtype=torch.float32)).numpy().ravel()
    return predict, n


def _fit_sklearn(kind: str, x: np.ndarray, y: np.ndarray):
    if kind == "gbt":
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)
    elif kind == "spline_poly":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        m = make_pipeline(PolynomialFeatures(degree=3, include_bias=False),
                          RidgeCV(alphas=np.logspace(-4, 2, 25)))
    elif kind == "gp_matern":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        m = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(x.shape[1]))
            + WhiteKernel(1e-4), normalize_y=True, n_restarts_optimizer=2, random_state=0)
    else:
        raise ValueError(kind)
    m.fit(x, y)
    return (lambda z: np.asarray(m.predict(z)).ravel()), -1


def make_surrogate_arm(kind: str, param_sink: dict):
    """Fit on what has been run, score what has not, evaluate the argmax, repeat."""
    fitter = _fit_torch if kind in ("kan", "mlp_matched") else _fit_sklearn
    key = "mlp" if kind == "mlp_matched" else kind

    def arm(s: Surface, rng, budget: int) -> None:
        for idx in rng.permutation(N_CFG)[:GP_N_INIT]:
            s.select(int(idx))
        while len(s.visited) < budget:
            seen = sorted(s._seen)
            y = np.array([s.value_of_visited(i) for i in seen])
            lo, hi = y.min(), y.max()
            yn = (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)
            predict, n_par = fitter(key, COORDS[seen], yn)
            if n_par > 0:
                param_sink[kind] = n_par
            cand = s.unvisited
            s.select(cand[int(np.argmax(predict(COORDS[cand])))])
    return arm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/surrogate_architecture_bakeoff/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    surface, contexts, seeds = load_cache(args.cache)
    print(f"  {len(contexts)} contextos x {len(seeds)} semillas x {N_CFG} configuraciones "
          f"· presupuesto {BUDGET}")

    params: dict = {}
    arm_names = (REFERENCE_ARM, "kan", "mlp_matched", "gp_matern", "gbt", "spline_poly", "random")
    per_arm = {n: {"auc": [], "final": [], "budget_used": []} for n in arm_names}
    trace: dict = {}

    for r, seed in enumerate(seeds):
        retained = {"rho": np.zeros(COORDS.shape[1] + 1)}
        for name in arm_names:
            rng = np.random.default_rng(90_000 + r)
            fn = (make_neuron_arm(retained) if name == REFERENCE_ARM
                  else arm_random if name == "random"
                  else make_surrogate_arm(name, params))
            aucs, finals = [], []
            for ctx in contexts:
                s = Surface(surface[(ctx, seed)])
                fn(s, rng, BUDGET)
                curve = s.regret_curve()
                denom = BUDGET * abs(s.best) or 1.0
                aucs.append(float(np.sum(curve)) / denom)
                finals.append(curve[-1] / (abs(s.best) or 1.0))
                per_arm[name]["budget_used"].append(len(s.visited))
                if r == 0 and ctx == contexts[0]:
                    trace[name] = list(s.visited)
            per_arm[name]["auc"].append(float(np.mean(aucs)))
            per_arm[name]["final"].append(float(np.mean(finals)))
        print(f"  semilla {r + 1}/{len(seeds)} ({time.perf_counter() - started:.0f}s)", flush=True)

    mean_auc = {n: float(np.mean(per_arm[n]["auc"])) for n in arm_names}
    ceiling = {n: 100.0 * (1.0 - float(np.mean(per_arm[n]["final"]))) for n in arm_names}
    rng = np.random.default_rng(20260807)

    def boot(diff: np.ndarray) -> dict:
        draws = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(N_BOOT)]
        return {"mean": float(np.mean(diff)), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "p_two_sided": float(2 * min(np.mean(np.asarray(draws) > 0),
                                             np.mean(np.asarray(draws) < 0)))}

    base = np.asarray(per_arm[REFERENCE_ARM]["auc"])
    vs_ref = {n: boot(np.asarray(per_arm[n]["auc"]) - base)
              for n in arm_names if n != REFERENCE_ARM}
    ps = [vs_ref[n]["p_two_sided"] for n in vs_ref]
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[idx]))
        adj[idx] = running
    for name, a in zip(vs_ref, adj):
        vs_ref[name]["holm_adjusted_p"] = a

    kan_vs_mlp = boot(np.asarray(per_arm["kan"]["auc"]) - np.asarray(per_arm["mlp_matched"]["auc"]))
    best = min((n for n in arm_names if n != "random"), key=lambda n: mean_auc[n])
    vs_random = boot(np.asarray(per_arm[best]["auc"]) - np.asarray(per_arm["random"]["auc"]))

    if kan_vs_mlp["ucb95"] < 0:
        verdict = "KAN_BEATS_A_MATCHED_MLP_AT_SEARCH"
    elif kan_vs_mlp["lcb95"] > 0:
        verdict = "KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP"
    elif best not in ("kan", "mlp_matched", REFERENCE_ARM):
        verdict = "A_CLASSICAL_SURROGATE_LEADS"
    else:
        verdict = "KAN_AND_MLP_ARE_INDISTINGUISHABLE_AT_SEARCH"

    print("\n  ranking (auc_regret_norm, menor es mejor):")
    for n in sorted(arm_names, key=lambda k: mean_auc[k]):
        v = vs_ref.get(n)
        tag = "" if v is None else (f"   Δ {v['mean']:+.5f} [{v['lcb95']:+.5f} · {v['ucb95']:+.5f}]"
                                    f" holm {v['holm_adjusted_p']:.3f}")
        star = "  (referencia)" if n == REFERENCE_ARM else ""
        print(f"    {n:<14} {mean_auc[n]:.5f}   % techo {ceiling[n]:6.2f}%{tag}{star}")
    print(f"\n  KAN − MLP igualado: {kan_vs_mlp['mean']:+.5f} "
          f"[{kan_vs_mlp['lcb95']:+.5f} · {kan_vs_mlp['ucb95']:+.5f}]")
    print(f"  veredicto: {verdict}\n")

    probe = Surface(surface[(contexts[0], seeds[0])])
    try:
        probe.value_of_visited(0)
        guard = False
    except LookupError:
        guard = True
    n_kan, n_mlp = params.get("kan", 0), params.get("mlp_matched", 0)

    falsifiers = {
        "f1_kan_and_mlp_are_parameter_matched": {
            "passed": bool(n_kan and n_mlp and abs(n_kan - n_mlp) <= 0.10 * n_kan),
            "evidence": {"why_it_can_fail": "David's objection was 532 parameters against 31 called "
                                            "matched. This checks the counts, not the label",
                         "kan": n_kan, "mlp": n_mlp}},
        "f2_no_arm_can_read_an_unrun_cell": {
            "passed": bool(guard),
            "evidence": {"why_it_can_fail": "the guard is provoked, not asserted; a Surface that "
                                            "returned an unvisited value would let any arm peek",
                         "probe_raised": bool(guard)}},
        "f3_budgets_are_matched": {
            "passed": all(u == BUDGET for n in arm_names for u in per_arm[n]["budget_used"]),
            "evidence": {"why_it_can_fail": "an arm stopping early would buy its ranking with a "
                                            "smaller spend", "budget": BUDGET}},
        "f4_the_arms_are_not_the_same_policy": {
            "passed": len({tuple(v) for v in trace.values()}) == len(trace),
            "evidence": {"why_it_can_fail": "two arms with identical visit sequences are one arm "
                                            "under two names",
                         "distinct_traces": len({tuple(v) for v in trace.values()}),
                         "n_arms": len(trace)}},
        "f5_the_harness_can_detect_skill": {
            "passed": bool(vs_random["ucb95"] < 0),
            "evidence": {"why_it_can_fail": "if the best arm cannot beat random, the harness "
                                            "separates nothing and no tie here means anything",
                         "best_arm": best, "best_vs_random": vs_random}},
        "f6_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
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
        "schema_version": "surrogate_architecture_bakeoff_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_SURROGATE_ARCHITECTURE_BAKEOFF_2026-08-07.md",
        "primary_metric": "auc_regret_norm", "budget": BUDGET,
        "no_gpu": ("288 points and four coordinates; the fit is instant on CPU and the useful "
                   "parallelism is across seeds, not tensors"),
        "scope_note": ("This does NOT live in the MPC lane. That lane's sign is unstable across "
                       "endpoint, incumbent and tape block, and its preregistered guardrail "
                       "worst_product_fill was never applied, so it cannot define a neural "
                       "residual. Here nothing is deployed: a configuration is chosen, the oracle "
                       "is exact, and there is no pending service guardrail."),
        "arms": list(arm_names), "parameters": params,
        "mean_auc_regret": mean_auc, "percent_of_ceiling": ceiling,
        "vs_reference": vs_ref, "kan_minus_matched_mlp": kan_vs_mlp,
        "best_arm": best, "best_vs_random": vs_random,
        "per_arm": per_arm, "contexts": contexts, "seeds": seeds,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/search_surrogates/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
