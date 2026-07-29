#!/usr/bin/env python3
"""Cross-fitted fixed-posture audit for the Track B-P 11D checkpoints.

The within-checkpoint audit used each evaluation episode's own full-trajectory
mean as its clamp, which is useful diagnostically but leaks that episode into
the selected constant. This audit estimates one Op3/Op5/Op9 posture per
training seed on disjoint calibration episodes, freezes it, and evaluates it
on the canonical held-out CRN episodes. It also reports a single global
posture pooled across calibration seeds for the retrained 8D baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_bp_timing_within import build_args, load, make_env, rollout


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-11d", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--calibration-episodes", type=int, default=12)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--calibration-seed-offset", type=int, default=60_000)
    p.add_argument("--eval-seed-offset", type=int, default=50_000)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--obs-config", default="v10_no_regime_forecast")
    p.add_argument("--enabled-risks", default="R21")
    p.add_argument("--risk-frequency-by-id", default="R21=8")
    p.add_argument("--risk-impact-by-id", default="R21=4")
    p.add_argument("--replenishment-lead-time", type=float, default=168.0)
    p.add_argument("--target-risk", default="R21")
    return p


def t_summary(values: list[float]) -> dict[str, object]:
    x = np.asarray(values, dtype=float)
    if len(x) > 1 and float(np.std(x, ddof=1)) > 0.0:
        ci = scipy_stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=scipy_stats.sem(x))
    else:
        ci = (float("nan"), float("nan"))
    return {
        "per_seed": [float(v) for v in x],
        "mean": float(x.mean()),
        "seed_clustered_ci95": [float(ci[0]), float(ci[1])],
        "seeds_positive": int((x > 0).sum()),
    }


def main() -> None:
    cli = parser().parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)
    sample = make_env(args, cli)
    cal_seeds = [1 + cli.calibration_seed_offset + i for i in range(cli.calibration_episodes)]
    eval_seeds = [1 + cli.eval_seed_offset + i for i in range(cli.eval_episodes)]

    models = {}
    postures: dict[int, np.ndarray] = {}
    for seed in cli.seeds:
        model, vec_norm = load(cli.run_11d, seed, sample)
        models[seed] = (model, vec_norm)
        traces = []
        for es in cal_seeds:
            traces.append(np.asarray(rollout(model, vec_norm, args, cli, es)["frac_trace"]))
        postures[seed] = np.concatenate(traces, axis=0).mean(axis=0)
        print(f"seed {seed} calibrated: {postures[seed].round(6).tolist()}", flush=True)

    global_posture = np.mean(np.stack(list(postures.values())), axis=0)
    rows: list[dict[str, object]] = []
    seed_deltas_frozen: list[float] = []
    seed_deltas_global: list[float] = []
    for seed in cli.seeds:
        model, vec_norm = models[seed]
        own = postures[seed].copy()
        deltas_own = []
        deltas_global = []
        for es in eval_seeds:
            factual = rollout(model, vec_norm, args, cli, es)["ret_excel"]
            frozen = rollout(
                model, vec_norm, args, cli, es,
                buffer_override=lambda _t, _f, v=own: v,
            )["ret_excel"]
            global_fixed = rollout(
                model, vec_norm, args, cli, es,
                buffer_override=lambda _t, _f, v=global_posture: v,
            )["ret_excel"]
            deltas_own.append(float(factual - frozen))
            deltas_global.append(float(factual - global_fixed))
            for arm, value in (
                ("self", factual),
                ("calibration_frozen_per_seed", frozen),
                ("calibration_frozen_global", global_fixed),
            ):
                rows.append({"train_seed": seed, "eval_seed": es, "arm": arm, "ret_excel": value})
        seed_deltas_frozen.append(float(np.mean(deltas_own)))
        seed_deltas_global.append(float(np.mean(deltas_global)))

    with (cli.output_dir / "frozen_posture_rows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (cli.output_dir / "calibrated_postures.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["train_seed", "op3_frac", "op5_frac", "op9_frac"])
        for seed in cli.seeds:
            writer.writerow([seed, *[float(v) for v in postures[seed]]])
        writer.writerow(["global", *[float(v) for v in global_posture]])

    arm_means = {
        arm: float(np.mean([float(r["ret_excel"]) for r in rows if r["arm"] == arm]))
        for arm in ("self", "calibration_frozen_per_seed", "calibration_frozen_global")
    }
    summary = {
        "config": {k: str(v) for k, v in vars(cli).items()},
        "calibrated_postures": {
            str(seed): [float(v) for v in posture] for seed, posture in postures.items()
        },
        "global_posture": [float(v) for v in global_posture],
        "arm_means": arm_means,
        "self_minus_calibration_frozen_per_seed": t_summary(seed_deltas_frozen),
        "self_minus_calibration_frozen_global": t_summary(seed_deltas_global),
    }
    (cli.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
