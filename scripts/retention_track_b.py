#!/usr/bin/env python3
"""Retained-vs-reset in Track B (continuous downstream control), with an
observability ablation.

Track A showed retained-vs-reset ~ 0 because the [6,3] action can't reach the
bottleneck AND the regime is observable. Track B's action space reaches the
downstream bottleneck (F11/F12) and has real headroom. Here we test whether
retained history L_{k-1} produces a LARGER, clearer effect there -- and whether it
emerges specifically when the regime one-hot + forecast features (v7 idx 30-36) are
MASKED so the regime must be inferred.

Training-tape only (native adaptive_benchmark_v2 regime). Symmetric adapt-then-eval:
retained carries theta across episodes/blocks; reset reloads theta_0 each block;
both adapt within block, frozen never adapts. Seed-clustered (seed = inferential unit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.evaluate_retained_reset_learning as ev  # noqa: E402
from supply_chain.external_env_interface import get_episode_terminal_metrics  # noqa: E402

REGIME_OBS_INDICES = "30,31,32,33,34,35,36"  # v7 regime one-hot + forecasts


def cluster_stats(per_seed: list[float]) -> dict:
    arr = np.array([x for x in per_seed if np.isfinite(x)], dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "ci95_lo": (mean - 1.96 * sem) if n > 1 else float("nan"),
        "ci95_hi": (mean + 1.96 * sem) if n > 1 else float("nan"),
        "per_seed": [float(x) for x in per_seed],
    }


def eval_episode_b(args, model, seed: int) -> dict:
    """Continuous-action episode eval (no block wrapper, no int() cast)."""
    env = ev.build_env(args)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _info = env.step(action)
        done = bool(term or trunc)
    m = get_episode_terminal_metrics(env)
    env.close()
    return {
        "ret": float(m["order_level_ret_mean"]),
        "fill": float(m["fill_rate_order_level"]),
    }


def run_condition(args, *, mask: str | None, seeds: list[int], cycles: int) -> dict:
    args.mask_obs_indices = mask
    per_seed_deltas = []  # mean (retained-reset) ReT per seed
    per_seed_rf = []
    for s_i, seed in enumerate(seeds):
        args.seed = seed  # independent network init per seed (seed = inferential unit)
        with tempfile.TemporaryDirectory() as tmp:
            init = Path(tmp) / "init.zip"
            ev.build_initial_model(args, init)
            frozen = ev.load_model(args, init)
            retained = ev.load_model(args, init)
            d_rr, d_rf = [], []
            for ci in range(cycles):
                ep_seed = 7_000 + s_i * 1000 + ci
                adapt_seed = ep_seed + ev.ADAPT_SEED_OFFSET
                fr = eval_episode_b(args, frozen, ep_seed)["ret"]
                ev.online_update(args, retained, seed=adapt_seed, regime=None)
                rt = eval_episode_b(args, retained, ep_seed)["ret"]
                reset = ev.load_model(args, init)
                ev.online_update(args, reset, seed=adapt_seed, regime=None)
                rs = eval_episode_b(args, reset, ep_seed)["ret"]
                d_rr.append(rt - rs)
                d_rf.append(rt - fr)
            per_seed_deltas.append(float(np.nanmean(d_rr)))
            per_seed_rf.append(float(np.nanmean(d_rf)))
    return {
        "retained_minus_reset": cluster_stats(per_seed_deltas),
        "retained_minus_frozen": cluster_stats(per_seed_rf),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="retention_track_b")
    p.add_argument(
        "--output-root", type=Path, default=Path("outputs/benchmarks/retention_track_b")
    )
    p.add_argument("--reward-mode", default="ReT_seq_v1")
    p.add_argument("--seeds", default="8101,8102,8103")
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--pretrain-timesteps", type=int, default=2000)
    p.add_argument("--online-timesteps-per-cycle", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=4)
    cli = p.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",") if s.strip()]

    def base_args():
        a = ev.build_parser().parse_args([])
        a.track = "b"
        a.algo = "ppo"
        a.decision_cadence = "weekly"  # per-step; avoids the int() block wrapper
        a.reward_mode = cli.reward_mode
        a.max_steps = cli.max_steps
        a.pretrain_timesteps = cli.pretrain_timesteps
        a.online_timesteps_per_cycle = cli.online_timesteps_per_cycle
        a.n_steps = cli.n_steps
        a.batch_size = cli.batch_size
        a.n_epochs = cli.n_epochs
        return a

    results = {}
    for cond, mask in (("obs_full", None), ("obs_hidden", REGIME_OBS_INDICES)):
        print(f"[track-b] running {cond} (mask={mask}) ...", flush=True)
        results[cond] = run_condition(
            base_args(), mask=mask, seeds=seeds, cycles=cli.cycles
        )

    run_dir = cli.output_root / cli.label
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "track": "b",
        "reward_mode": cli.reward_mode,
        "seeds": seeds,
        "cycles": cli.cycles,
        "online_timesteps_per_cycle": cli.online_timesteps_per_cycle,
        "regime_obs_indices_masked": REGIME_OBS_INDICES,
        "results": results,
    }
    (run_dir / "retention_track_b.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nTRACK B RETENTION (retained - reset ReT, seed-clustered)")
    for cond in ("obs_full", "obs_hidden"):
        rr = results[cond]["retained_minus_reset"]
        rf = results[cond]["retained_minus_frozen"]
        print(
            f"  {cond:10} ret-reset={rr['mean']:+.4f} +/-{rr['sem']:.4f} "
            f"ci95=[{rr['ci95_lo']:+.4f},{rr['ci95_hi']:+.4f}]  ret-frozen={rf['mean']:+.4f}"
        )
    print(f"Saved: {run_dir / 'retention_track_b.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
