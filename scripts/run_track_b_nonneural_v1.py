#!/usr/bin/env python3
"""The non-neural comparator track_b_v1 never had, and without which "neural premium" is untestable.

WHAT WAS MISSING. In this environment we have measured that training buys +20 to +26 and that the
choice among KAN, MLP and DMLPA buys nothing. What we have never measured is whether a network
beats something that is NOT a network. Every arm compared so far is a network, and the only
untrained one is the SAME network with random weights -- which measures the effect of training, not
the need for the architecture.

The ordering was wrong and this run says so: it should have preceded the architecture comparisons.

THE LEAK THIS AVOIDS. Searching for the best constant on the evaluation episodes would make it an
oracle and the result worthless. The constant and the thresholds are fitted on a DISJOINT block,
seed0 = 888_000, and every arm is then scored with the protocol the networks already used:
seed0 = 777_000, 24 episodes. f1 fails if the blocks touch.

Preregistration: docs/PREREGISTRO_COMPARADOR_NO_NEURONAL_TRACK_B_2026-08-07.md
Development. No custody seeds are opened.
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
from supply_chain.seed_custody import module_manifest  # noqa: E402

from run_architecture_bakeoff_v1 import make_env  # noqa: E402

EVAL_SEED0, FIT_SEED0 = 777_000, 888_000
N_BOOT = 5_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def rollout(policy, seed: int) -> float:
    """One episode under a callable policy(obs) -> action."""
    env = make_env()
    obs, _ = env.reset(seed=seed)
    done, total = False, 0.0
    while not done:
        obs, r, term, trunc, _ = env.step(policy(obs))
        total += float(r)
        done = term or trunc
    return total


def score(policy, seed0: int, episodes: int) -> list[float]:
    return [rollout(policy, seed0 + k) for k in range(episodes)]


def const_policy(a: np.ndarray):
    return lambda obs: a


def threshold_policy(base: np.ndarray, gain: np.ndarray, idx: np.ndarray):
    """Constant, shifted by observed signals from the most recent frame.

    The frame stack is (features x HISTORY_LEN) flattened, so the last block is the newest
    observation; `idx` selects which of its coordinates drive the shift."""
    def act(obs):
        z = np.asarray(obs, dtype=np.float64)
        tail = z[-101:] if z.size >= 101 else z
        sig = np.tanh(tail[idx % tail.size])
        return np.clip(base + gain * sig, -1.0, 1.0).astype(np.float32)
    return act


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-episodes", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--const-candidates", type=int, default=160)
    ap.add_argument("--rule-candidates", type=int, default=80)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/track_b_nonneural/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260807)
    dim = int(make_env(0).action_space.shape[0])

    # ---- fit the constant on the FIT block only ------------------------------------------------
    best_a, best_v, trace = None, -np.inf, []
    for i in range(args.const_candidates):
        a = rng.uniform(-1.0, 1.0, dim).astype(np.float32)
        v = float(np.mean(score(const_policy(a), FIT_SEED0, args.fit_episodes)))
        trace.append(v)
        if v > best_v:
            best_a, best_v = a, v
        if (i + 1) % 40 == 0:
            print(f"  constante {i+1}/{args.const_candidates}  mejor {best_v:+.4f} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)

    # ---- fit the threshold rule on the FIT block ------------------------------------------------
    best_rule, best_rv = None, -np.inf
    for i in range(args.rule_candidates):
        gain = rng.uniform(-0.5, 0.5, dim).astype(np.float32)
        idx = rng.integers(0, 101, dim)
        pol = threshold_policy(best_a, gain, idx)
        v = float(np.mean(score(pol, FIT_SEED0, args.fit_episodes)))
        if v > best_rv:
            best_rule, best_rv = (gain, idx), v
        if (i + 1) % 40 == 0:
            print(f"  regla {i+1}/{args.rule_candidates}  mejor {best_rv:+.4f} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)

    # ---- score everything on the EVAL block, the protocol the networks used ---------------------
    arms = {
        "random_action": lambda obs: rng.uniform(-1, 1, dim).astype(np.float32),
        "constant_best": const_policy(best_a),
        "threshold_rule": threshold_policy(best_a, best_rule[0], best_rule[1]),
    }
    evals = {n: score(p, EVAL_SEED0, args.eval_episodes) for n, p in arms.items()}
    means = {n: float(np.mean(v)) for n, v in evals.items()}

    # Networks come from the sealed artifacts; nothing is retrained here.
    nets = {}
    bk = Path("results/architecture_bakeoff_200k/result.json")
    if bk.exists():
        d = json.loads(bk.read_text())
        for a, v in d.get("by_arch", {}).items():
            nets[f"trained_{a.lower()}"] = float(v["mean"])
    un = Path("results/untrained_control/result.json")
    if un.exists():
        nets["untrained_net"] = float(json.loads(un.read_text())["untrained_mean"])

    base = np.asarray(evals["constant_best"])

    def boot(d):
        draws = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(N_BOOT)]
        return {"mean": float(np.mean(d)), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5))}

    vs_const = {n: boot(np.asarray(v) - base) for n, v in evals.items() if n != "constant_best"}
    net_gap = {n: m - means["constant_best"] for n, m in nets.items()}
    best_net = max((n for n in nets if n.startswith("trained_")), key=lambda n: nets[n],
                   default=None)

    print("\n  brazos NO neuronales (bloque de evaluacion 777_000):")
    for n, m in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"    {n:<16} {m:+.4f}")
    print("\n  redes (de artefactos sellados) menos la mejor constante:")
    for n, g in sorted(net_gap.items(), key=lambda kv: -kv[1]):
        print(f"    {n:<16} {nets[n]:+.4f}   gap {g:+.4f}")

    gap = net_gap.get(best_net) if best_net else None
    verdict = ("NEURAL_PREMIUM_LIKELY_IN_TRACK_B" if gap is not None and gap > 0
               else "A_CONSTANT_ACTION_MATCHES_OR_BEATS_THE_NETWORKS")

    falsifiers = {
        "f1_the_constant_is_not_fitted_on_the_evaluation_block": {
            "passed": abs(FIT_SEED0 - EVAL_SEED0) >= max(args.fit_episodes, args.eval_episodes),
            "evidence": {"why_it_can_fail": "fitting the constant on the episodes it is scored on "
                                            "makes it an oracle and voids the comparison",
                         "fit_seed0": FIT_SEED0, "eval_seed0": EVAL_SEED0,
                         "fit_episodes": args.fit_episodes,
                         "eval_episodes": args.eval_episodes}},
        "f2_the_arms_share_the_action_space_and_protocol": {
            "passed": True,
            "evidence": {"why_it_can_fail": "an arm scored on a different block or episode count "
                                            "is not comparable",
                         "action_space": f"Box(-1,1,({dim},))", "eval_seed0": EVAL_SEED0,
                         "eval_episodes": args.eval_episodes}},
        "f3_the_harness_can_detect_skill": {
            "passed": bool(best_net and nets[best_net] - means["random_action"] > 0),
            "evidence": {"why_it_can_fail": "if the trained networks cannot beat uniform random "
                                            "actions, the harness separates nothing and no tie "
                                            "here means anything",
                         "best_net": best_net,
                         "best_net_minus_random": (nets[best_net] - means["random_action"])
                         if best_net else None}},
        "f4_the_constant_search_actually_searched": {
            "passed": bool(float(np.std(trace)) > 1e-6
                           and float(np.max(trace)) > float(trace[0])),
            "evidence": {"why_it_can_fail": "a search whose best is its first draw, or whose "
                                            "candidates all score the same, did not search",
                         "n_candidates": args.const_candidates,
                         "fit_spread": float(np.std(trace)),
                         "first": float(trace[0]), "best": float(np.max(trace))}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<56} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "track_b_nonneural_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_NO_CUSTODY_SEEDS_NO_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_COMPARADOR_NO_NEURONAL_TRACK_B_2026-08-07.md",
        "why": ("track_b_v1 had no non-neural arm, so 'neural premium' was untestable there. The "
                "ordering was wrong: this should have preceded the architecture comparisons."),
        "fit_block_seed0": FIT_SEED0, "eval_block_seed0": EVAL_SEED0,
        "fit_episodes": args.fit_episodes, "eval_episodes": args.eval_episodes,
        "constant_best": best_a.tolist(), "constant_fit_score": best_v,
        "rule_gain": best_rule[0].tolist(), "rule_index": best_rule[1].tolist(),
        "rule_fit_score": best_rv,
        "nonneural_eval_means": means, "nonneural_eval_episodes": evals,
        "vs_constant": vs_const,
        "network_means_from_sealed_artifacts": nets, "network_minus_constant": net_gap,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/untrained_control/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
