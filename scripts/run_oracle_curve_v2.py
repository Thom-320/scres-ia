#!/usr/bin/env python3
"""Run the v2 matched-rights retained learning curve.

Contract: contracts/oracle_retained_learning_curve_v2.json (frozen 2026-07-26).

Sequencing enforced here, in this order:
  1. parity self-check   -- retained and reset arms differ ONLY in the retention flag;
  2. calibration         -- the best static bar is selected on a calibration block and
                            frozen BEFORE the test block is graded (the pilot selected it
                            in-sample on the very campaigns it then graded);
  3. training            -- disjoint root block, checkpoints selected on calibration mean
                            ReT so the test frontier is never used for selection;
  4. test                -- weights frozen, graded by exact lookup, three numbers reported.

Usage:
    .venv/bin/python scripts/run_oracle_curve_v2.py --total-timesteps 96000 --seeds 3
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.c6_perbatch_ceiling import OBJECTIVE  # noqa: E402
from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.oracle_capture import (  # noqa: E402
    BOOT_SEED, calendar_index, load_campaigns, pooled_capture,
)
from supply_chain.oracle_curve_v2 import (  # noqa: E402
    HistorySpec, MetaCampaignEnv, assert_arm_parity,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    simulate_full_des_frontier,
)

OUT = ROOT / "results/oracle_curve_v2"
FRONTIERS = ROOT / "results/fig5_surrogate_v1/frontiers"
TRAIN_ROOTS = range(7_680_001, 7_680_017)      # fresh, verified unused
CALIB_ROOTS = range(7_690_001, 7_690_005)      # fresh, verified unused
KAPPAS = (0.75, 0.90)
CAMPAIGNS = 12


# The 4-action COUNT scheduler, which is what the env, the enumerated 4^8 frontiers and the
# frozen comparator all use. c6_perbatch_ceiling.SCHED_PATTERN is the 8-action PER-BATCH
# mapping of the C6 gate; using it here silently reinterpreted action 1 as [P_C,P_H,P_H]
# instead of [P_H,P_C,P_H], so the reward was computed on different physics than the grading.
COUNT_SCHEDULER = scheduler()


def objective_of(skeleton, calendar) -> float:
    metrics = simulate_full_des_frontier(
        skeleton=skeleton, scheduler=COUNT_SCHEDULER,
        calendars=np.asarray([calendar], dtype=np.uint8), include_q_r1_metrics=True)
    return float(np.asarray(metrics[OBJECTIVE])[0])


def histories(roots, kappas=KAPPAS) -> list[HistorySpec]:
    return [HistorySpec(r, k, CAMPAIGNS) for r in roots for k in kappas]


def rollout_history(model, env: MetaCampaignEnv, spec: HistorySpec) -> list[dict]:
    """One meta-episode; the recurrent state persists iff the arm is the retained one."""
    obs, _ = env.reset(options={"history": spec})
    state, done = None, False
    first_step_of_history = True
    rows: list[dict] = []
    while not done:
        episode_start = np.array([first_step_of_history if env.retained
                                  else not env.last_calendar])
        action, state = model.predict(obs, state=state, episode_start=episode_start,
                                      deterministic=True)
        first_step_of_history = False
        obs, _reward, done, _trunc, info = env.step(int(action))
        if info.get("campaign_boundary"):
            rows.append(info)
    return rows


def mean_objective(model, env, specs) -> float:
    values = [row["objective"] for spec in specs for row in rollout_history(model, env, spec)]
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=96_000)
    parser.add_argument("--eval-every", type=int, default=12_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=7_680_101)
    parser.add_argument("--arm", choices=("retained", "reset", "both"), default="both")
    parser.add_argument("--output", type=Path, default=OUT / "curve_v2.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    from sb3_contrib import RecurrentPPO  # noqa: PLC0415

    sched = scheduler()
    train_specs, calib_specs = histories(TRAIN_ROOTS), histories(CALIB_ROOTS)
    started = time.perf_counter()

    # ---- 1. parity self-check, before anything is trained or read -------------
    probe_ret = MetaCampaignEnv(scheduler=sched, histories=train_specs, retained=True,
                                objective_fn=objective_of, rng_seed=0)
    probe_res = MetaCampaignEnv(scheduler=sched, histories=train_specs, retained=False,
                                objective_fn=objective_of, rng_seed=0)
    assert_arm_parity(probe_ret, probe_res)
    obs_ret, _ = probe_ret.reset(options={"history": train_specs[0]})
    obs_res, _ = probe_res.reset(options={"history": train_specs[0]})
    if not np.allclose(obs_ret[:-1], obs_res[:-1]):
        raise SystemExit("parity violation: arms differ outside the carried-prior slot")
    parity = {
        "configuration_identical": True,
        "differs_only_in_prior_slot": True,
        "retained_prior_slot": float(obs_ret[-1]),
        "reset_prior_slot": float(obs_res[-1]),
    }
    print(f"[parity] OK  retained prior slot {obs_ret[-1]:.4f} vs reset {obs_res[-1]:.4f}",
          flush=True)

    # ---- 2. the static bar is selected on calibration and frozen -------------
    calib_env = MetaCampaignEnv(scheduler=sched, histories=calib_specs, retained=True,
                                objective_fn=objective_of, rng_seed=1)
    all_calendars = np.load(FRONTIERS / "calendars.npz")["calendars"]
    calib_labels = []
    for spec in calib_specs:
        campaigns, _ = calib_env._load(spec)
        for campaign in campaigns:
            metrics = simulate_full_des_frontier(
                skeleton=campaign.skeleton, scheduler=COUNT_SCHEDULER,
                calendars=all_calendars, include_q_r1_metrics=True)
            calib_labels.append(np.asarray(metrics[OBJECTIVE], dtype=float))
    calib_stack = np.vstack(calib_labels)
    bar_row = int(calib_stack.mean(axis=0).argmax())
    bar_calendar = [(bar_row // 4 ** (7 - w)) % 4 for w in range(8)]
    print(f"[calibration] static bar {bar_calendar} selected on "
          f"{calib_stack.shape[0]} out-of-test campaigns ({time.perf_counter()-started:.0f}s)",
          flush=True)

    # ---- test block: the burned campaigns, whose exact frontiers exist -------
    test_campaigns = load_campaigns(FRONTIERS)
    bar = {c.key: float(c.labels[bar_row]) for c in test_campaigns}
    test_specs = sorted({HistorySpec(c.history_root, c.kappa, CAMPAIGNS)
                         for c in test_campaigns},
                        key=lambda s: (s.history_root, s.kappa))
    graded = {(c.history_root, c.campaign_index, c.persistence_mode): c
              for c in test_campaigns}

    def grade(model, env) -> dict:
        calendars = {}
        for spec in test_specs:
            for row in rollout_history(model, env, spec):
                mode = "binary_0.75" if spec.kappa < 0.85 else "binary_0.9"
                key = (row["history_root"], row["campaign_index"], mode)
                if key in graded:
                    calendars[key] = row["calendar"]
            for key, cal in list(calendars.items()):
                if calendar_index(cal) < 0:
                    raise AssertionError("illegal calendar")
        present = [c for c in test_campaigns if c.key in calendars]
        pooled = pooled_capture(present, calendars, bar, np.random.default_rng(BOOT_SEED))
        return {
            **pooled,
            "mean_label": float(np.mean([c.value_of(calendars[c.key]) for c in present])),
            "exact_optimum_hits": int(sum(
                1 for c in present if c.ceiling - c.value_of(calendars[c.key]) <= 1e-9)),
            "distinct_calendars": len({tuple(v) for v in calendars.values()}),
            "n_graded": len(present),
        }

    # ---- 3-4. train both arms, select on calibration, grade on test ---------
    arms: dict[str, list] = {}
    for name in ("recurrent_ppo_retained", "recurrent_ppo_reset"):
        arms[name] = []
    wanted = {"retained": [("recurrent_ppo_retained", True)],
              "reset": [("recurrent_ppo_reset", False)],
              "both": [("recurrent_ppo_retained", True), ("recurrent_ppo_reset", False)]}
    for arm, retained in wanted[args.arm]:
        for offset in range(args.seeds):
            seed = args.seed_start + offset
            train_env = MetaCampaignEnv(scheduler=sched, histories=train_specs,
                                        retained=retained, objective_fn=objective_of,
                                        rng_seed=seed)
            eval_env = MetaCampaignEnv(scheduler=sched, histories=test_specs,
                                       retained=retained, objective_fn=objective_of,
                                       rng_seed=seed)
            calib_eval = MetaCampaignEnv(scheduler=sched, histories=calib_specs,
                                         retained=retained, objective_fn=objective_of,
                                         rng_seed=seed)
            model = RecurrentPPO("MlpLstmPolicy", train_env, seed=seed, n_steps=96,
                                 batch_size=96, verbose=0)
            points, best = [], (-1e9, None)
            done_steps = 0
            while done_steps < args.total_timesteps:
                chunk = min(args.eval_every, args.total_timesteps - done_steps)
                model.learn(total_timesteps=chunk, reset_num_timesteps=False)
                done_steps += chunk
                selection = mean_objective(model, calib_eval, calib_specs)
                point = {"timesteps": done_steps, "calibration_mean_ret": selection,
                         **grade(model, eval_env)}
                points.append(point)
                if selection > best[0]:
                    best = (selection, point)
                print(f"[{arm} seed {seed}] t={done_steps} calib {selection:.4f} "
                      f"capture {point['pooled_ratio']:+.4f} "
                      f"cond {point['conditional_ratio']:+.4f} "
                      f"hits {point['exact_optimum_hits']} "
                      f"({time.perf_counter()-started:.0f}s)", flush=True)
            arms[arm].append({"seed": seed, "points": points,
                              "selected_checkpoint": best[1],
                              "selected_on": "calibration_mean_ret"})

    payload = {
        "schema": "oracle_curve_v2",
        "contract": "contracts/oracle_retained_learning_curve_v2.json",
        "claim_status": "DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parity_self_check": parity,
        "static_bar": {"calendar": bar_calendar, "frontier_row": bar_row,
                       "selected_on": "calibration block, out of test",
                       "calibration_campaigns": int(calib_stack.shape[0])},
        "blocks": {"train": [TRAIN_ROOTS.start, TRAIN_ROOTS.stop - 1],
                   "calibration": [CALIB_ROOTS.start, CALIB_ROOTS.stop - 1],
                   "test": "burned 7570801-24, the campaigns with exact frontiers",
                   "campaigns_per_history": CAMPAIGNS},
        "training": {"total_timesteps": args.total_timesteps,
                     "eval_every": args.eval_every,
                     "seeds": [args.seed_start + i for i in range(args.seeds)]},
        "arms": arms,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
