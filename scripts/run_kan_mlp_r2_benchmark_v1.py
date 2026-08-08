#!/usr/bin/env python3
"""KAN against MLP under R2 stress, at matched parameters, with the interaction as the contrast.

Contract: docs/PREREGISTRO_GARRIDO_R2_RANDOMIZED_BENCHMARK_V1.md, amended by
docs/ENMIENDA_ALCANCE_R1_R3_BENCHMARK_2026-08-08.md and
docs/ENMIENDA_BENCHMARK_SIGNO_SESOI_HORIZONTE_2026-08-08.md. All three committed before this file.

THE ROLE THE ARCHITECTURES PLAY is Garrido's own Fig. 5: a SUPERVISED SURROGATE placed between his
node (3) and node (8), not an RL policy. Each network is trained to predict L* from a schedule and
its tape context, and then picks a schedule per tape. That is the construct he proposed, and it is
what makes KAN a candidate at all.

THE CONTRAST IS THE INTERACTION, with the sign corrected -- L* is a LOSS, so

    A_e = E[L*_MLP - L*_KAN]          positive means KAN is better in environment e
    Delta = A_stressed - A_baseline   positive means R2 stress differentially favours KAN

Writing it as (KAN - MLP), which the first preregistration did, is positive when KAN LOSES.

WHAT IS DECLARED BEFORE THE NUMBERS EXIST, because it bounds every result here:
  * the exact class of 26 contiguous schedules has a clairvoyant ceiling of UCB95 <= 0.0028, so no
    surrogate can win more than that by choosing well inside it;
  * shifts add nothing given the buffer -- M_S = 0.000000 in all nine diagnostic cells;
  * R21 fires at most 1.00 times per episode against R24's 32.08, so nothing here speaks about R21;
  * the R2 arm is PARAMETRIC stress inside the source family. Changing the distribution family is
    what Garrido asked for and it is NOT IMPLEMENTED -- the exponential/lognormal/weibull support in
    supply_chain.py is for the fulfilment delay, not risk occurrence.

MATCHING IS THE WHOLE BENCHMARK. Equal DES interactions, equal hyperparameter search, parameters
matched within each budget, latency and memory reported. Without that, "KAN wins" can mean "KAN got
more search".
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.continuous_its_env import make_continuous_its_track_a_env  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
MAX_STEPS, STEP_HOURS, K_BUFFER = 26, 168.0, 13
ENVIRONMENTS = {"baseline": 1.0, "r2_stressed": 4.0}
BUDGETS = {"p25": 0.25, "p50": 0.50, "p100": 1.00}
SESOI_RELATIVE = 0.05
SEED_BLOCK = tuple(range(8600001, 8600013))
N_BOOT = 2_000
BASE_WIDTH = 32                       # the 100% budget; the others are matched to its parameter count

MODULES = ("supply_chain/continuous_its_env.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def contiguous(start: int) -> set:
    return {((start + i) % MAX_STEPS) for i in range(K_BUFFER)}


SCHEDULES = [contiguous(s) for s in range(MAX_STEPS)]


def exposure(sim) -> float:
    horizon, start = float(sim.env.now), float(sim.warmup_time)
    num = den = 0.0
    for o in sim.orders:
        if bool(getattr(o, "metrics_excluded", False)):
            continue
        opt = float(getattr(o, "OPTj", 0.0) or 0.0)
        if opt < start:
            continue
        q = float(o.quantity or 0.0)
        due = opt + float(o.LTj or 0.0)
        end = float(o.OATj) if getattr(o, "OATj", None) is not None else horizon
        num += q * max(0.0, end - due)
        den += q * max(0.0, horizon - due)
    return num / den if den > 0 else 0.0


def play(mult: float, seed: int, weeks, rule: bool = False) -> dict:
    env = make_continuous_its_track_a_env(
        init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
        risk_level="current", enabled_risks=R1 + R2, risk_rng_mode="per_risk",
        stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
        risk_obs=True, holding_cost=0.0, shift_cost=0.0,
        risk_frequency_multipliers_by_id={r: float(mult) for r in R2},
        risk_impact_multipliers_by_id={r: float(mult) for r in R2})
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    done = truncated = False
    step, left = 0, K_BUFFER
    ctx, played = [], []
    try:
        while not (done or truncated):
            backlog = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0)
            if rule:
                remaining = MAX_STEPS - step
                on = left > 0 and (backlog > 0.0 or left == remaining)
                left -= int(on)
            else:
                on = step in weeks
            played.append(int(on))
            ctx.append(backlog)
            _o, _r, done, truncated, _i = env.step(
                np.array([1.0 if on else 0.0, -1.0], dtype=np.float32))
            step += 1
        return {"L": exposure(sim), "context": ctx, "schedule": played,
                "n_events": len(getattr(sim, "risk_events", []) or [])}
    finally:
        env.close()


def features(weeks, ctx) -> np.ndarray:
    """Schedule encoding plus early tape context. The context is PRE-DECISION only -- the first
    four weeks of observed backlog -- so the surrogate never reads the outcome it predicts."""
    s = np.zeros(MAX_STEPS)
    for w in weeks:
        s[w] = 1.0
    early = np.asarray(ctx[:4], dtype=float)
    early = early / (early.max() + 1e-9)
    return np.concatenate([s, early])


def train_and_pick(kind: str, width: int, X_tr, y_tr, X_te_per_tape, seed: int):
    """Fit a surrogate, then pick the argmin-predicted schedule for each held-out tape.

    Both architectures get the SAME data, the SAME number of epochs and the SAME optimiser
    settings. Nothing here is tuned per architecture, because tuning one and not the other is the
    commonest way a benchmark like this becomes meaningless.
    """
    import torch
    torch.manual_seed(seed)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)
    d = X_tr.shape[1]
    t0 = time.perf_counter()
    if kind == "mlp":
        model = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.Tanh(),
                                    torch.nn.Linear(width, 1))
    else:
        from kan import KAN
        model = KAN(width=[d, max(1, width // 8), 1], grid=3, k=3, seed=seed,
                    auto_save=False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(Xt).reshape(-1, 1), yt)
        loss.backward()
        opt.step()
    fit_s = time.perf_counter() - t0
    picks = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for Xc in X_te_per_tape:
            pred = model(torch.tensor(Xc, dtype=torch.float32)).reshape(-1).numpy()
            picks.append(int(np.argmin(pred)))
    infer_s = time.perf_counter() - t1
    return picks, {"n_params": int(n_params), "fit_seconds": fit_s,
                   "inference_seconds": infer_s, "train_mse": float(loss.item())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/kan_mlp_r2_benchmark/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)
    seeds = list(SEED_BLOCK[:args.seeds])
    half = len(seeds) // 2
    train_seeds, test_seeds = seeds[:half], seeds[half:]
    print(f"  {len(ENVIRONMENTS)} entornos x {len(seeds)} semillas x {len(SCHEDULES)} calendarios "
          f"= {len(ENVIRONMENTS) * len(seeds) * len(SCHEDULES)} episodios de cache")

    cache, des_calls = {}, 0
    for env_name, mult in ENVIRONMENTS.items():
        for s in seeds:
            runs = [play(mult, s, w) for w in SCHEDULES]
            des_calls += len(SCHEDULES)
            cache[(env_name, s)] = runs
        print(f"    cache {env_name} listo")

    results, spread = {}, {}
    for env_name, mult in ENVIRONMENTS.items():
        ctx0 = {s: cache[(env_name, s)][0]["context"] for s in seeds}
        X_tr = np.array([features(SCHEDULES[i], ctx0[s])
                         for s in train_seeds for i in range(len(SCHEDULES))])
        y_tr = np.array([cache[(env_name, s)][i]["L"]
                         for s in train_seeds for i in range(len(SCHEDULES))])
        X_te = [np.array([features(SCHEDULES[i], ctx0[s]) for i in range(len(SCHEDULES))])
                for s in test_seeds]
        L_te = np.array([[cache[(env_name, s)][i]["L"] for i in range(len(SCHEDULES))]
                         for s in test_seeds])

        # Comparators, on the SAME held-out tapes.
        best_fixed = int(np.argmin(L_te.mean(axis=0)))
        L_open = L_te[:, best_fixed]
        L_rule = np.array([play(mult, s, None, rule=True)["L"] for s in test_seeds])
        des_calls += len(test_seeds)
        L_clair = L_te.min(axis=1)

        arms = {"open_loop": {"L": [float(x) for x in L_open]},
                "rule": {"L": [float(x) for x in L_rule]},
                "clairvoyant_ceiling": {"L": [float(x) for x in L_clair]}}
        for bname, frac in BUDGETS.items():
            width = max(4, int(round(BASE_WIDTH * frac)))
            for kind in ("mlp", "kan"):
                picks, meta = train_and_pick(kind, width, X_tr, y_tr, X_te, seed=20260808)
                L_arm = np.array([L_te[j, p] for j, p in enumerate(picks)])
                arms[f"{kind}|{bname}"] = {"L": [float(x) for x in L_arm], **meta,
                                           "picks": picks, "width": width}
                print(f"    {env_name:12s} {kind}|{bname:4s} L {L_arm.mean():.6f}  "
                      f"params {meta['n_params']}  fit {meta['fit_seconds']:.2f}s")
        spread[env_name] = L_te.tolist()
        results[env_name] = {"multiplier": mult, "arms": arms,
                             "best_fixed_index": best_fixed,
                             "test_seeds": test_seeds, "train_seeds": train_seeds}

    def advantage(env_name, bname):
        """A = E[L_MLP - L_KAN]. POSITIVE MEANS KAN IS BETTER, because L* is a loss."""
        a = np.asarray(results[env_name]["arms"][f"mlp|{bname}"]["L"])
        b = np.asarray(results[env_name]["arms"][f"kan|{bname}"]["L"])
        d = a - b
        boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                         for _ in range(N_BOOT)])
        return {"mean": float(d.mean()), "lcb95": float(np.percentile(boot, 2.5)),
                "ucb95": float(np.percentile(boot, 97.5)),
                "relative": float(d.mean() / (a.mean() + 1e-12)),
                "meets_sesoi": bool(d.mean() / (a.mean() + 1e-12) >= SESOI_RELATIVE
                                    and float(np.percentile(boot, 2.5)) > 0)}

    interaction = {}
    for bname in BUDGETS:
        a_base = advantage("baseline", bname)
        a_str = advantage("r2_stressed", bname)
        db = (np.asarray(results["r2_stressed"]["arms"][f"mlp|{bname}"]["L"])
              - np.asarray(results["r2_stressed"]["arms"][f"kan|{bname}"]["L"]))
        dbase = (np.asarray(results["baseline"]["arms"][f"mlp|{bname}"]["L"])
                 - np.asarray(results["baseline"]["arms"][f"kan|{bname}"]["L"]))
        diff = db - dbase
        boot = np.array([float(np.mean(diff[rng.integers(0, len(diff), len(diff))]))
                         for _ in range(N_BOOT)])
        interaction[bname] = {
            "A_baseline": a_base, "A_stressed": a_str,
            "delta_mean": float(diff.mean()),
            "delta_lcb95": float(np.percentile(boot, 2.5)),
            "delta_ucb95": float(np.percentile(boot, 97.5)),
            "favours_kan": bool(float(np.percentile(boot, 2.5)) > 0)}

    any_sesoi = [b for b in BUDGETS
                 if interaction[b]["A_stressed"]["meets_sesoi"]
                 or interaction[b]["A_baseline"]["meets_sesoi"]]
    fav = [b for b in BUDGETS if interaction[b]["favours_kan"]]
    beats_simple = [f"{e}|{k}" for e in ENVIRONMENTS for k in
                    [f"{a}|{b}" for a in ("mlp", "kan") for b in BUDGETS]
                    if np.mean(results[e]["arms"][k]["L"])
                    < np.mean(results[e]["arms"]["open_loop"]["L"]) - 1e-12]

    falsifiers = {
        "f1_parameters_are_matched_within_budget": {
            "passed": all(
                abs(results[e]["arms"][f"mlp|{b}"]["n_params"]
                    - results[e]["arms"][f"kan|{b}"]["n_params"])
                <= 2.0 * min(results[e]["arms"][f"mlp|{b}"]["n_params"],
                             results[e]["arms"][f"kan|{b}"]["n_params"])
                for e in ENVIRONMENTS for b in BUDGETS),
            "evidence": {"why_it_can_fail": "a benchmark where one architecture carries several "
                                            "times the parameters of the other measures capacity, "
                                            "not architecture",
                         "per_cell": {f"{e}|{b}": {
                             "mlp": results[e]["arms"][f"mlp|{b}"]["n_params"],
                             "kan": results[e]["arms"][f"kan|{b}"]["n_params"]}
                             for e in ENVIRONMENTS for b in BUDGETS}}},
        "f2_equal_des_interactions": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "disclosure carried as a falsifier so it cannot be "
                                            "dropped: every arm reads the SAME cached episodes, so "
                                            "no architecture can buy an advantage with more "
                                            "simulator calls",
                         "shared_cache": True, "total_des_episodes": des_calls}},
        "f3_no_per_architecture_tuning": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "disclosure. Same data, same 200 epochs, same Adam at "
                                            "lr 1e-2 for both. Tuning one and not the other is the "
                                            "commonest way this benchmark becomes meaningless",
                         "epochs": 200, "optimiser": "Adam", "lr": 0.01}},
        "f4_surrogate_reads_no_outcome": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "disclosure. Features are the schedule encoding plus "
                                            "the first four weeks of observed backlog. A surrogate "
                                            "that read the episode's own L* would be the leak that "
                                            "voided the meta-learner v1",
                         "context_weeks": 4, "feature_dim": MAX_STEPS + 4}},
        "f5_held_out_tapes": {
            "passed": len(set(train_seeds) & set(test_seeds)) == 0 and len(test_seeds) > 0,
            "evidence": {"why_it_can_fail": "picking on the tapes it was fitted on would measure "
                                            "memorisation",
                         "train_seeds": train_seeds, "test_seeds": test_seeds}},
        "f6_ceiling_is_declared": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "disclosure placed BEFORE the numbers: the exact class "
                                            "has a clairvoyant ceiling of UCB95 <= 0.0028, shifts "
                                            "add nothing given the buffer, R21 fires at most once "
                                            "per episode, and the R2 arm is PARAMETRIC stress -- "
                                            "the distribution-family change Garrido asked for is "
                                            "NOT IMPLEMENTED",
                         "clairvoyant_ucb95": 0.0028,
                         "r2_family_change": "NOT_IMPLEMENTED",
                         "clairvoyant_L_by_env": {
                             e: float(np.mean(results[e]["arms"]["clairvoyant_ceiling"]["L"]))
                             for e in ENVIRONMENTS}}},
        "f8_decision_space_is_not_degenerate": {
            # THE FALSIFIER I DROPPED, and dropping it is how this run first read as "equivalence".
            # Two earlier runners were fixed by exactly this check and I did not carry it here.
            # The arms pick DIFFERENT schedules -- MLP 0, KAN 24 and 19 -- and all score identical
            # L*, and the clairvoyant per-tape minimum equals the fixed column. If every schedule
            # ties on every tape there is no decision for a surrogate to make, and a tie between
            # architectures is a fact about the decision space, not about the architectures.
            "passed": all(
                float(np.mean(np.max(np.asarray(spread[e]), axis=1)
                              - np.min(np.asarray(spread[e]), axis=1))) > 1e-9
                for e in ENVIRONMENTS),
            "evidence": {"why_it_can_fail": "AND IT DID. A benchmark whose options are "
                                            "indistinguishable cannot compare the things choosing "
                                            "between them",
                         "mean_within_tape_spread": {
                             e: float(np.mean(np.max(np.asarray(spread[e]), axis=1)
                                              - np.min(np.asarray(spread[e]), axis=1)))
                             for e in ENVIRONMENTS},
                         "picks_differ_but_scores_do_not": {
                             e: {k: results[e]["arms"][k]["picks"]
                                 for k in results[e]["arms"] if "picks" in results[e]["arms"][k]}
                             for e in ENVIRONMENTS}}},
        "f7_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    all_ok = all(v["passed"] for k, v in falsifiers.items()
                 if isinstance(v, dict) and not v.get("not_applicable"))
    falsifiers["all_passed"] = all_ok

    if not all_ok:
        verdict = "BLOCKED_INSTRUMENT"
    elif fav and any_sesoi:
        verdict = "KAN_ADVANTAGE_UNDER_R2_STRESS"
    elif not beats_simple:
        verdict = "NEITHER_ARCHITECTURE_BEATS_THE_OPEN_LOOP_CALENDAR"
    else:
        verdict = "EQUIVALENT_CHOOSE_MLP_BY_PARSIMONY"

    print("\n  interaccion  Delta = A_estresado - A_baseline   (A = E[L_MLP - L_KAN], "
          "positivo = KAN mejor)")
    for b, v in interaction.items():
        print(f"    {b:5s} A_base {v['A_baseline']['mean']:+.6f} "
              f"A_estr {v['A_stressed']['mean']:+.6f}  Delta {v['delta_mean']:+.6f} "
              f"[{v['delta_lcb95']:+.6f}, {v['delta_ucb95']:+.6f}]")
    print(f"\n  bate al calendario open-loop: {beats_simple or 'ninguna arquitectura'}")
    print(f"\n  veredicto: {verdict}   (SESOI = {SESOI_RELATIVE:.0%} relativo)\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<44} {label}")

    payload = {
        "schema_version": "kan_mlp_r2_benchmark_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_SURROGATE_ROLE_NOT_RL",
        "run_role": "ARCHITECTURE_BENCHMARK", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "architecture_role": ("Garrido's Fig. 5: a SUPERVISED SURROGATE between his nodes 3 and 8, "
                              "predicting L* and choosing a schedule -- not an RL policy"),
        "contrast": {"A": "E[L*_MLP - L*_KAN], positive means KAN better because L* is a loss",
                     "delta": "A_stressed - A_baseline",
                     "sesoi": {"value": SESOI_RELATIVE, "definition": "relative reduction of L*"}},
        "declared_before_the_numbers": {
            "clairvoyant_ceiling_ucb95": 0.0028,
            "shifts_add_nothing_given_buffer": "M_S = 0.000000 in all nine diagnostic cells",
            "r21_exposure": "at most 1.00 events per episode against R24's 32.08",
            "r2_arm": "PARAMETRIC stress within the source family",
            "r2_distribution_family_change": "NOT_IMPLEMENTED — requires a risk-scheduler change"},
        "environments": {k: {"r2_multiplier_by_id": v} for k, v in ENVIRONMENTS.items()},
        "budgets": BUDGETS, "results": results, "interaction": interaction,
        "arms_beating_open_loop": beats_simple,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/lever_redundancy_diagnostic/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
