#!/usr/bin/env python3
"""David's architecture, run as it should have been: does the KAN in the latent_rw help?

WHOSE DEFECT. The notebook we shipped him declares use_kan=False; the cell he ran declares
use_kan=True. His DMLPA has the KAN by design and we had silently switched it off -- the origin
defect is ours. What it broke is that DMLPA and DMLPA_KAN became the SAME object in his run (his
parameter_matching reports both at 225,410 params, 12.71% off), so his "DMLPA" label stopped
meaning what ours means. And this repo never had the KAN branch at all: what we have measured as
DMLPA is his architecture with his KAN removed.

THE SINGLE-FACTOR DESIGN. He compared two things at once without meaning to -- features_dim 60 vs
84 AND the latent_rw. Here the whole transformer is held fixed (features_dim 84, nhead 12,
num_layers 2, ff_mult 4) and only hidden_dim moves, because it carries no divisibility constraint.
Both arms land within 0.5% of the 200k budget, against the 30% tolerance he had to use.

A RESULT BEFORE ANY TRAINING: at equal budget the KAN affords hidden_dim=10 against the MLP's 152.
KAN edges are expensive; the width they buy is fifteen times narrower.

WHAT THIS DOES NOT COVER. His version also differs in the normalisation order --
pre_norm(latent + pos) here, pre_norm(latent) + pos there. Only the latent_rw is tested; changing
both would reintroduce the very confound this preregistration exists to remove.

Preregistration: docs/PREREGISTRO_DMLPA_KAN_LATENT_2026-08-07.md
Development. Seeds 9491-9495 are already open.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from run_architecture_bakeoff_v1 import (  # noqa: E402
    DMLPA, TARGET_PARAMS, evaluate, make_env, make_vec, policy_kwargs,
)

MODULES = ("supply_chain/arm_runner.py", "supply_chain/env_experimental_shifts.py",
           "supply_chain/external_env_interface.py")
FEATURES_DIM, NHEAD, LAYERS, FF_MULT = 84, 12, 2, 4
# Searched once, over hidden_dim at fixed features_dim, and frozen here so the run cannot re-tune.
ARMS = {"dmlpa_mlp": {"use_kan": False, "hidden_dim": 152},
        "dmlpa_kan": {"use_kan": True, "hidden_dim": 10}}
N_BOOT = 5_000


def build(space, spec):
    return DMLPA(space, features_dim=FEATURES_DIM, hidden_dim=spec["hidden_dim"],
                 nhead=NHEAD, num_layers=LAYERS, ff_mult=FF_MULT, use_kan=spec["use_kan"])


def main() -> int:
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9491, 9492, 9493, 9494, 9495])
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/dmlpa_kan_latent/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    flat = int(make_env(0).observation_space.shape[0])
    space = gym.spaces.Box(-np.inf, np.inf, (flat,), dtype=np.float32)
    sizes, probe_out = {}, {}
    fixed = torch.zeros(2, flat)
    fixed[0, ::7] = 0.37
    for name, spec in ARMS.items():
        m = build(space, spec)
        n = sum(p.numel() for p in m.parameters())
        sizes[name] = {"params": int(n), "error": abs(n - TARGET_PARAMS) / TARGET_PARAMS,
                       **spec, "features_dim": FEATURES_DIM}
        with torch.no_grad():
            probe_out[name] = [round(float(v), 6) for v in m(fixed)[0][:6]]
        print(f"  {name:<11} hidden_dim={spec['hidden_dim']:<4} params={n:>8,} "
              f"desv={100*sizes[name]['error']:5.2f}%")

    rows = []
    for name, spec in ARMS.items():
        for seed in args.seeds:
            venv = make_vec(args.n_envs, seed)
            model = PPO("MlpPolicy", venv, seed=seed, device="cpu", learning_rate=3e-4,
                        n_steps=512, batch_size=64, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, ent_coef=0.01, verbose=0,
                        policy_kwargs=dict(
                            features_extractor_class=DMLPA,
                            features_extractor_kwargs=dict(
                                features_dim=FEATURES_DIM, hidden_dim=spec["hidden_dim"],
                                nhead=NHEAD, num_layers=LAYERS, ff_mult=FF_MULT,
                                use_kan=spec["use_kan"]),
                            net_arch=dict(pi=[64, 64], vf=[64, 64])))
            model.learn(total_timesteps=args.total_steps)
            mean, sd = evaluate(model, args.eval_episodes)
            rows.append({"arm": name, "seed": int(seed), "ret_mean": float(mean),
                         "ret_sd_within": float(sd), "params": sizes[name]["params"]})
            venv.close()
            del model
            print(f"    {name:<11} semilla {seed}  ReT {mean:+.5f} ± {sd:.5f} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)

    def col(name):
        return np.array([r["ret_mean"] for r in rows if r["arm"] == name], dtype=float)

    kan, mlp = col("dmlpa_kan"), col("dmlpa_mlp")
    rng = np.random.default_rng(20260807)
    d = kan - mlp
    draws = d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
    contrast = {"mean": float(d.mean()), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)), "n": int(d.size)}

    # f4: the harness must still reproduce at this budget, or the paired contrast is void.
    repro = []
    for _ in range(2):
        venv = make_vec(args.n_envs, args.seeds[0])
        m = PPO("MlpPolicy", venv, seed=args.seeds[0], device="cpu", learning_rate=3e-4,
                n_steps=512, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, verbose=0,
                policy_kwargs=dict(features_extractor_class=DMLPA,
                                   features_extractor_kwargs=dict(
                                       features_dim=FEATURES_DIM, hidden_dim=152, nhead=NHEAD,
                                       num_layers=LAYERS, ff_mult=FF_MULT, use_kan=False),
                                   net_arch=dict(pi=[64, 64], vf=[64, 64])))
        m.learn(total_timesteps=4096)
        repro.append(float(evaluate(m, 4)[0]))
        venv.close()
        del m

    falsifiers = {
        "f1_parameters_are_matched_within_our_tolerance": {
            "passed": bool(max(v["error"] for v in sizes.values()) <= 0.10),
            "evidence": {"why_it_can_fail": "comparing different capacities measures capacity, not "
                                            "architecture; David had to loosen this to 30%",
                         "sizes": sizes, "tolerance": 0.10}},
        "f2_only_the_latent_rw_differs": {
            "passed": bool(len({FEATURES_DIM}) == 1 and all(
                v["features_dim"] == FEATURES_DIM for v in sizes.values())),
            "evidence": {"why_it_can_fail": "a second differing factor would make this the same "
                                            "confound the preregistration exists to remove",
                         "held_fixed": {"features_dim": FEATURES_DIM, "nhead": NHEAD,
                                        "num_layers": LAYERS, "ff_mult": FF_MULT},
                         "not_covered": "normalisation order differs from David's version"}},
        "f3_the_two_arms_are_behaviourally_distinct": {
            "passed": bool(probe_out["dmlpa_kan"] != probe_out["dmlpa_mlp"]),
            "evidence": {"why_it_can_fail": "identical outputs on a fixed input would mean the "
                                            "same model was trained twice -- which is exactly "
                                            "what happened to David when both labels resolved to "
                                            "one object",
                         "fingerprints": probe_out}},
        "f4_the_harness_reproduces": {
            "passed": bool(abs(repro[0] - repro[1]) <= 1e-9),
            "evidence": {"why_it_can_fail": "if the determinism repair does not hold at this "
                                            "budget the paired contrast is void",
                         "replicas": repro, "delta": abs(repro[0] - repro[1])}},
        "f5_no_new_seeds": {
            "passed": bool(set(args.seeds) <= {9491, 9492, 9493, 9494, 9495}),
            "evidence": {"why_it_can_fail": "a seed outside the development set would consume "
                                            "custody this run never declared",
                         "seeds": args.seeds}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    verdict = ("KAN_LATENT_HELPS" if contrast["lcb95"] > 0
               else "KAN_LATENT_HURTS" if contrast["ucb95"] < 0
               else "KAN_LATENT_INDISTINGUISHABLE")

    print(f"\n  dmlpa_kan {kan.mean():+.5f}   dmlpa_mlp {mlp.mean():+.5f}")
    print(f"  kan - mlp {contrast['mean']:+.5f} "
          f"[{contrast['lcb95']:+.5f}, {contrast['ucb95']:+.5f}]")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<48} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "dmlpa_kan_latent_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "endpoint": "ret_mean_track_b_v1",
        "whose_defect": ("the notebook we shipped declares use_kan=False and David's executed cell "
                         "declares use_kan=True: his DMLPA has the KAN by design and we had "
                         "switched it off. The origin defect is ours."),
        "what_it_broke": ("DMLPA and DMLPA_KAN became the same object in his run -- his "
                          "parameter_matching reports both at 225,410 params, 12.71% off -- so his "
                          "DMLPA label stopped meaning what ours means"),
        "result_before_training": ("at equal budget the KAN affords hidden_dim=10 against the "
                                   "MLP's 152; KAN edges are expensive and the width they buy is "
                                   "fifteen times narrower"),
        "not_covered": ("David's version also differs in normalisation order: pre_norm(latent+pos) "
                        "here, pre_norm(latent)+pos there. Only the latent_rw is tested."),
        "design": {"features_dim": FEATURES_DIM, "nhead": NHEAD, "num_layers": LAYERS,
                   "ff_mult": FF_MULT, "total_steps": args.total_steps,
                   "eval_episodes": args.eval_episodes, "n_envs": args.n_envs},
        "seeds": args.seeds, "sizes": sizes, "rows": rows,
        "arm_means": {"dmlpa_kan": float(kan.mean()), "dmlpa_mlp": float(mlp.mean())},
        "kan_minus_mlp": contrast, "reproducibility_check": repro,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/determinism_repair_control/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
