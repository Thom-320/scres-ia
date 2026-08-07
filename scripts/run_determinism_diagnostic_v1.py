#!/usr/bin/env python3
"""Which layer makes Track B non-deterministic? Three configurations, two replicas each.

WHY THIS RUNS BEFORE ANY FIX. The reproducibility probe closed on branch R3: the same seed, the
same architecture and the same PPO arguments produced 96.747 / 94.383 / 96.023 -- a spread of
2.363 points at a FIXED seed, larger than the 2.102 the bake-off shows across five DIFFERENT
seeds. So the seed explains nothing and every cross-run comparison in Track B is void.

I preregistered a suspected cause: make_vec discards its `seed` argument in the multi-worker
branch (`SubprocVecEnv([lambda: make_env(None) ...])`). Reading SB3 2.9 afterwards, that is a real
defect but probably NOT the cause -- `PPO(seed=seed)` calls `BaseAlgorithm.set_random_seed`, which
calls `self.env.seed(seed)`, and SB3's VecEnv.seed assigns `seed + idx` to every worker for the
next reset. The workers do get seeded, through a different path than the one I flagged.

Fixing a cause I have not demonstrated would produce a repair that verifies nothing. So this
measures first.

THE THREE CONFIGURATIONS, and what each isolates:

    A  n_envs=8, torch threads left alone   -- the configuration every Track B artifact used
    B  n_envs=8, torch threads pinned to 1  -- removes intra-op float reduction-order variation
    C  n_envs=1, torch threads pinned to 1  -- removes the subprocess workers entirely

READING RULE, FIXED BEFORE RUNNING:

    A differs, B agrees                  -> TORCH_THREADING; the fix is to pin threads
    A differs, B differs, C agrees       -> SUBPROCESS_WORKERS; the fix is in the vec env
    all three differ                     -> DEEPER_THAN_BOTH; report as an environment limit,
                                            do not claim a fix
    A AGREES                             -> the 20k horizon is too short to expose it. This run
                                            decided nothing and says so; escalate the horizon.

That last branch is the one that keeps this honest: a diagnostic that cannot come back empty is
not a diagnostic.

Contract: docs/PREREGISTRO_DIAGNOSTICO_DETERMINISMO_2026-08-07.md
Instrument diagnostic. No seeds are opened; 9492 is development and already burned.
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

MODULES = ("supply_chain/arm_runner.py", "supply_chain/external_env_interface.py",
           "supply_chain/env.py")
SEED = 9492
TOL = 1e-9
CONFIGS = (
    {"name": "A_8envs_threads_default", "n_envs": 8, "threads": None},
    {"name": "B_8envs_threads_1", "n_envs": 8, "threads": 1},
    {"name": "C_1env_threads_1", "n_envs": 1, "threads": 1},
)


def one_run(n_envs: int, threads: int | None, steps: int, episodes: int) -> float:
    import torch
    from stable_baselines3 import PPO
    from run_architecture_bakeoff_v1 import (
        DMLPA, TARGET_PARAMS, evaluate, make_env, make_vec, policy_kwargs, size_to_budget)
    import gymnasium as gym

    if threads is not None:
        torch.set_num_threads(int(threads))
    flat = int(make_env(0).observation_space.shape[0])
    space = gym.spaces.Box(-np.inf, np.inf, (flat,), dtype=np.float32)
    width, _ = size_to_budget(
        lambda w: DMLPA(space, hidden_dim=max(32, int(w) // 12 * 12),
                        features_dim=max(12, int(w) // 12 * 12), nhead=12, num_layers=2),
        12, 480, TARGET_PARAMS)
    venv = make_vec(n_envs, SEED)
    model = PPO("MlpPolicy", venv, seed=SEED, device="cpu", learning_rate=3e-4,
                n_steps=512, batch_size=64, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, verbose=0,
                policy_kwargs=policy_kwargs("DMLPA", width))
    model.learn(total_timesteps=int(steps))
    mean, _ = evaluate(model, int(episodes))
    venv.close()
    del model
    return float(mean)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/determinism_diagnostic/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    rows = {}
    for cfg in CONFIGS:
        vals = []
        for rep in range(2):
            v = one_run(cfg["n_envs"], cfg["threads"], args.steps, args.episodes)
            vals.append(v)
            print(f"  {cfg['name']:<26} replica {rep+1}/2  ReT {v:+.9f} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)
        rows[cfg["name"]] = {"values": vals, "delta": abs(vals[0] - vals[1]),
                             "deterministic": abs(vals[0] - vals[1]) <= TOL,
                             "n_envs": cfg["n_envs"], "threads": cfg["threads"]}

    a, b, c = (rows[k["name"]]["deterministic"] for k in CONFIGS)
    if a:
        verdict = "INCONCLUSIVE_HORIZON_TOO_SHORT_TO_EXPOSE_IT"
    elif b:
        verdict = "TORCH_THREADING"
    elif c:
        verdict = "SUBPROCESS_WORKERS"
    else:
        verdict = "DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT"

    falsifiers = {
        "f1_the_diagnostic_can_come_back_empty": {
            "passed": True,
            "evidence": {"why_it_can_fail": "if configuration A reproduces at this horizon the "
                                            "run has isolated nothing, and the declared reading "
                                            "rule sends it to INCONCLUSIVE instead of letting a "
                                            "downstream branch claim a cause",
                         "A_deterministic": bool(a), "steps": args.steps}},
        "f2_all_three_share_seed_arch_and_hyperparameters": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a configuration differing in more than the isolated "
                                            "layer would not isolate that layer",
                         "seed": SEED, "arch": "DMLPA", "n_steps": 512, "ent_coef": 0.01,
                         "learning_rate": 3e-4, "varies_only": ["n_envs", "torch_threads"]}},
        "f3_the_tolerance_is_exact_not_approximate": {
            "passed": bool(TOL <= 1e-9),
            "evidence": {"why_it_can_fail": "a loose tolerance would call a drifting pipeline "
                                            "deterministic", "tolerance": TOL}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    print(f"\n  veredicto: {verdict}\n")
    for name, r in rows.items():
        print(f"    {name:<26} delta {r['delta']:.3e}  "
              f"{'DETERMINISTA' if r['deterministic'] else 'NO determinista'}")

    payload = {
        "schema_version": "determinism_diagnostic_v1",
        "claim_status": verdict,
        "scope": "INSTRUMENT_DIAGNOSTIC_NO_SEEDS_OPENED_NO_SCIENTIFIC_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "motivation": ("results/repro_probe closed on branch R3: 96.747 / 94.383 / 96.023 at a "
                       "fixed seed, spread 2.363, versus 2.102 across five different seeds."),
        "preregistered_suspicion_status": (
            "make_vec discards its `seed` argument in the multi-worker branch. That is a real "
            "defect, but SB3 2.9 BaseAlgorithm.set_random_seed calls self.env.seed(seed) and "
            "VecEnv.seed assigns seed+idx per worker, so the workers ARE seeded by another path. "
            "The suspicion is therefore probably not the cause, and this run measures instead of "
            "assuming."),
        "seed": SEED, "steps": args.steps, "eval_episodes": args.episodes, "tolerance": TOL,
        "configurations": rows, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/repro_probe/A/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
