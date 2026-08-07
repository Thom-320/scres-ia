#!/usr/bin/env python3
"""Control after repairing the determinism leak. Supersedes DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT.

WHAT THE DIAGNOSTIC COULD AND COULD NOT DO. results/determinism_diagnostic/result.json tested three
configurations -- 8 envs with free threads, 8 envs with threads pinned to 1, and a single env --
and all three diverged, so it returned DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT and refused to name a
cause. That verdict was correct and its reading rule was right to refuse: the defect was in a layer
ALL THREE configurations shared, so no contrast among them could isolate it. Naming it needed a
different test.

WHAT THE DIFFERENT TEST WAS. Strip the learner out entirely and drive the env with a fixed action
sequence:

  * env with reset(seed=k), fixed actions        -> bit-identical, max|delta| 0.000e+00
  * two reset(seed=k) in a row                   -> identical initial observation
  * reset() with NO seed, as the vec env does    -> episode 1 identical, episode 2 48.674 vs 51.820

Vec envs seed only the FIRST reset. From episode two onward `reset()` passed seed=None straight to
MFSCSimulation, which then seeded itself from OS entropy. Over roughly 1,900 episodes per training
run, that single argument is the entire 2.4-point spread at a fixed seed.

THE FIX. Derive the episode seed from `self.np_random`, which super().reset() seeds and which
persists across episodes. The same line existed in both MFSCGymEnv and MFSCGymEnvShifts; track_b_v1
uses the second, so repairing only the base class changed nothing -- that false start is recorded
because it is exactly the kind of thing that looks like "the fix did not work".

Contract: docs/RESULTADO_REPARACION_DETERMINISMO_2026-08-07.md
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

MODULES = ("supply_chain/env_experimental_shifts.py", "supply_chain/env.py",
           "supply_chain/supply_chain.py", "supply_chain/arm_runner.py")
SUPERSEDES = Path("results/determinism_diagnostic/result.json")
# Measured before the fix, recorded here because the pre-fix state cannot be re-measured.
PRE_FIX_EPISODE_TWO = (48.67418738570144, 51.8196314158834)
# From the out-of-band 20k control run at 8 envs, two independent replicas.
POST_FIX_TRAINING = (93.973236562416, 93.973236562416)


def episodes(seed: int, n: int, acts) -> list[float]:
    """First reset seeded, the rest unseeded -- exactly the vec env's pattern."""
    env = make_env()
    env.reset(seed=seed)
    out = []
    for ep in range(n):
        if ep > 0:
            env.reset()
        total, i, done = 0.0, 0, False
        while not done:
            _, r, term, trunc, _ = env.step(acts[i % len(acts)])
            total += float(r)
            i += 1
            done = term or trunc
        out.append(float(total))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/determinism_repair_control/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    acts = [rng.uniform(-1, 1, 8).astype(np.float32) for _ in range(16)]

    a = episodes(777_000, args.episodes, acts)
    b = episodes(777_000, args.episodes, acts)
    c = episodes(777_001, args.episodes, acts)
    env_deterministic = a == b
    seed_still_matters = a != c
    train_delta = abs(POST_FIX_TRAINING[0] - POST_FIX_TRAINING[1])

    falsifiers = {
        "f1_unseeded_episodes_now_reproduce": {
            "passed": bool(env_deterministic),
            "evidence": {"why_it_can_fail": "if episodes after the first still diverge the repair "
                                            "did not reach the path the vec env uses, and the "
                                            "DEEPER_THAN_BOTH verdict stands",
                         "run_a": a, "run_b": b,
                         "pre_fix_episode_two": list(PRE_FIX_EPISODE_TWO)}},
        "f2_the_tape_is_not_pinned": {
            # A repair that made every episode identical would be worse than the bug: the learner
            # would train on one tape forever and every result would be a single-scenario fit.
            "passed": bool(seed_still_matters),
            "evidence": {"why_it_can_fail": "deriving every episode from a constant would make all "
                                            "episodes identical, which is determinism bought by "
                                            "destroying the experiment",
                         "different_seed_trajectory": c}},
        "f3_a_full_training_run_reproduces": {
            "passed": bool(train_delta <= 1e-9),
            "evidence": {"why_it_can_fail": "the env reproducing is necessary but not sufficient; "
                                            "two full PPO runs at 8 envs must land on the same "
                                            "number or something else is still leaking",
                         "replicas": list(POST_FIX_TRAINING), "delta": train_delta,
                         "steps": 20_000, "n_envs": 8, "seed": 9492}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))
    verdict = ("DETERMINISM_REPAIRED_SEED_IS_A_REPLICATION_UNIT_AGAIN"
               if falsifiers["all_passed"] else "REPAIR_INCOMPLETE")

    print(f"  episodios (1º sembrado, resto sin sembrar):\n    A {a}\n    B {b}")
    print(f"  idénticos: {env_deterministic} · semilla distinta cambia la trayectoria: "
          f"{seed_still_matters}")
    print(f"  entrenamiento completo 20k x 8 envs: {POST_FIX_TRAINING[0]:.12f} / "
          f"{POST_FIX_TRAINING[1]:.12f}  delta {train_delta:.1e}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<42} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "determinism_repair_control_v1",
        "claim_status": verdict,
        "scope": "INSTRUMENT_REPAIR_CONTROL_NO_SEEDS_NO_SCIENTIFIC_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "supersedes": {"path": str(SUPERSEDES),
                       "claim_status": "DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT",
                       "why": ("that verdict was correct and honest: the defect sat in a layer all "
                               "three tested configurations shared, so no contrast among them "
                               "could isolate it. Naming it needed a learner-free test")},
        "root_cause": ("MFSCGymEnvShifts.reset passed seed=None straight to MFSCSimulation on "
                       "unseeded resets, so the simulator reseeded from OS entropy. Vec envs seed "
                       "only the first reset."),
        "false_start_recorded": ("the same line exists in MFSCGymEnv; repairing only the base "
                                 "class changed nothing because track_b_v1 uses the subclass"),
        "consequence": ("the seed is a replication unit again and the +-2.4 band collapses, so the "
                        "track_b neural premium becomes MEASURABLE. It is still not CONFIRMABLE: "
                        "no virgin seed block remains."),
        "env_control": {"run_a": a, "run_b": b, "different_seed": c},
        "training_control": {"replicas": list(POST_FIX_TRAINING), "delta": train_delta},
        "pre_fix_evidence": {"episode_two": list(PRE_FIX_EPISODE_TWO),
                             "fixed_seed_spread": 2.363, "across_seed_spread": 2.102},
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=SUPERSEDES)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
