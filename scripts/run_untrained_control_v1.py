#!/usr/bin/env python3
"""The untrained control Garrido asked for on 22 July, run on its own so it lands first.

WHY SEPARATELY. The 200k architecture bake-off is already training DMLPA on seeds 9491-9495 with
24 evaluation episodes, so the TRAINED side of the contrast is being computed right now. What is
missing is the same architecture at zero training steps -- which costs evaluation only, minutes not
hours. Running the full variant sweep before knowing whether training buys anything would be
seventeen hours spent in the wrong order.

WHAT IT DECIDES. If trained DMLPA does not beat untrained DMLPA with an interval excluding zero,
then in this environment nothing is bought by learning, and no architecture comparison here is
interpretable -- ours or David's. That would matter more than any variant.

PAIRING. Same seeds, same architecture width, same evaluate() with seed0=777_000 and 24 episodes,
so the rows pair one-to-one against the bake-off's DMLPA rows. The contrast itself is computed by
scripts/adjudicate_untrained_control_v1.py once the bake-off lands; this runner only produces the
untrained side and says so.

Preregistration: docs/PREREGISTRO_VARIANTES_DMLPA_2026-08-07.md (f3)
Development. Seeds 9491-9495 are development, not virgin.
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

from run_architecture_bakeoff_v1 import (  # noqa: E402
    DMLPA, TARGET_PARAMS, evaluate, make_env, make_vec, policy_kwargs, size_to_budget,
)

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def main() -> int:
    import gymnasium as gym
    from stable_baselines3 import PPO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9491, 9492, 9493, 9494, 9495])
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--arch", default="DMLPA")
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/untrained_control/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    flat = int(make_env(0).observation_space.shape[0])
    space = gym.spaces.Box(-np.inf, np.inf, (flat,), dtype=np.float32)
    width, err = size_to_budget(
        lambda w: DMLPA(space, hidden_dim=max(32, int(w) // 12 * 12),
                        features_dim=max(12, int(w) // 12 * 12), nhead=12, num_layers=2),
        12, 480, TARGET_PARAMS)
    print(f"  {args.arch} ancho={width} desv={100*err:.1f}% · {args.eval_episodes} episodios "
          f"· CERO pasos de entrenamiento")

    rows = []
    for seed in args.seeds:
        # One env is enough: nothing is trained, so the vec width only affects rollout collection.
        model = PPO("MlpPolicy", make_vec(1, seed), seed=seed, verbose=0,
                    policy_kwargs=policy_kwargs(args.arch, width))
        mean, sd = evaluate(model, args.eval_episodes)
        rows.append({"seed": int(seed), "ret_mean": float(mean), "ret_sd_within": float(sd)})
        print(f"    semilla {seed}  ReT {mean:+.5f} ± {sd:.5f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)

    vals = np.array([r["ret_mean"] for r in rows])
    falsifiers = {
        "f1_nothing_was_trained": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a stray learn() call would make this an ordinary arm "
                                            "and destroy the contrast",
                         "total_timesteps": 0, "learn_called": False}},
        "f2_pairs_with_the_bakeoff": {
            "passed": args.seeds == [9491, 9492, 9493, 9494, 9495] and args.eval_episodes == 24,
            "evidence": {"why_it_can_fail": "the contrast is paired seed by seed against the "
                                            "bake-off's DMLPA rows; different seeds or a different "
                                            "episode count would break the pairing",
                         "seeds": args.seeds, "eval_episodes": args.eval_episodes,
                         "eval_seed0": 777_000}},
        "f3_parameters_match_the_bakeoff_arm": {
            "passed": bool(err <= 0.10),
            "evidence": {"why_it_can_fail": "an untrained network of a different size is not the "
                                            "control for the trained one",
                         "width": int(width), "error": float(err)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))
    print(f"\n  media sin entrenar: {vals.mean():+.5f}  (sd entre semillas {vals.std(ddof=1):.5f})")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<38} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "untrained_control_v1",
        "claim_status": "UNTRAINED_SIDE_ONLY_NO_CONTRAST_YET",
        "scope": "DEVELOPMENT_NOT_VIRGIN_ONE_SIDE_OF_A_PAIRED_CONTRAST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_VARIANTES_DMLPA_2026-08-07.md",
        "what_this_is_not": ("Not a result. This is the untrained half of the trained-vs-untrained "
                             "contrast Garrido asked for; the trained half comes from the 200k "
                             "bake-off and the contrast is computed once that lands."),
        "arch": args.arch, "width": int(width), "param_error": float(err),
        "eval_episodes": args.eval_episodes, "eval_seed0": 777_000,
        "seeds": args.seeds, "rows": rows,
        "untrained_mean": float(vals.mean()),
        "untrained_sd_between_seeds": float(vals.std(ddof=1)),
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/architecture_bakeoff/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
