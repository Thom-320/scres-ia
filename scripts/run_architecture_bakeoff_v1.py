#!/usr/bin/env python3
"""KAN vs MLP vs DMLPA at a matched parameter budget, on the richest decision problem we have.

Garrido's conclusions name three candidates by name: back-propagation networks, Kolmogorov-Arnold
networks, and simulation-optimization as a form of reinforcement learning. This measures the first
two as CONTROLLERS, which is the question David is asking, and it does it where an architecture can
plausibly matter: Track B, an 8-dimensional continuous action over 101 observation features and 104
weekly decisions, rather than Program O-R's four discrete actions over 21 features.

THE COMPARISON IS ONLY ABOUT ARCHITECTURE IF THE BUDGETS MATCH. David's own objection to the
anti-KAN preprint was that a comparison at unequal parameter counts measures capacity, not
architecture. Every arm here is auto-sized to a shared budget and the run aborts if any lands
outside the tolerance.

ONLY THE INDEPENDENT ARM. Optimizer seeds are independent replicates of training, which is what
makes a between-seed interval mean anything. A persistent arm -- weights crossing seeds -- answers
a different question (learning across runs) and has no independent replicates by construction, so
it cannot support an architecture comparison. It is measured separately in David's lab.

Development only. No virgin seeds, no adjudication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from supply_chain.external_env_interface import make_track_b_env  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402
from real_kan_extractor import RealKANFeaturesExtractor  # noqa: E402

OBS_VERSION, MAX_STEPS, HISTORY_LEN = "v10", 104, 16
TARGET_PARAMS, PARAM_TOLERANCE = 200_000, 0.10
ARCHS = ("KAN", "MLP", "DMLPA")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/env_experimental_shifts.py",
           "supply_chain/external_env_interface.py", "supply_chain/config.py",
           "supply_chain/seed_custody.py")


class HistoryStack(gym.Wrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        d = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (d * n,), dtype=np.float32)

    def reset(self, **kw):
        o, i = self.env.reset(**kw)
        self.buf = [o] * self.n
        return np.concatenate(self.buf).astype(np.float32), i

    def step(self, a):
        o, r, term, trunc, i = self.env.step(a)
        self.buf = self.buf[1:] + [o]
        return np.concatenate(self.buf).astype(np.float32), r, term, trunc, i


def make_env(seed=None):
    e = HistoryStack(make_track_b_env(observation_version=OBS_VERSION, max_steps=MAX_STEPS),
                     HISTORY_LEN)
    if seed is not None:
        e.reset(seed=seed)
    return e


def make_vec(n_envs, seed=None):
    import multiprocessing as mp
    if n_envs == 1 or "fork" not in mp.get_all_start_methods():
        return DummyVecEnv([lambda: make_env(seed)])
    return SubprocVecEnv([lambda: make_env(None) for _ in range(n_envs)], start_method="fork")


class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, hidden=256):
        super().__init__(observation_space, features_dim)
        d = int(observation_space.shape[0])
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, features_dim), nn.LayerNorm(features_dim))

    def forward(self, x):
        return self.net(x.float())


class DMLPA(BaseFeaturesExtractor):
    """David's transformer over the frame stack, with dim_feedforward stated explicitly.

    PyTorch defaults it to 2048, which at d_model=120 is ~492k parameters PER LAYER -- the reason
    his architecture could not be sized down to a shared budget at all.
    """

    def __init__(self, observation_space, factor=HISTORY_LEN, features_dim=120,
                 hidden_dim=100, nhead=12, num_layers=2, ff_mult=4):
        super().__init__(observation_space, features_dim)
        flat = int(observation_space.shape[0])
        self.obs_dimension, self.factor = flat // factor, factor
        self.latent_rw = nn.Sequential(nn.Linear(self.obs_dimension, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, features_dim))
        self.pre_norm = nn.LayerNorm(features_dim)
        layer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead, batch_first=True,
                                           dim_feedforward=ff_mult * features_dim)
        self.accumulated = nn.TransformerEncoder(layer, num_layers=num_layers)
        pe = torch.zeros(factor, features_dim)
        pos = torch.arange(factor).unsqueeze(1)
        div = torch.exp(torch.arange(0, features_dim, 2) * (-math.log(10000.0) / features_dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pos", pe.unsqueeze(0))

    def forward(self, x):
        x = x.float().view(x.shape[0], self.factor, self.obs_dimension)
        x = self.pre_norm(self.latent_rw(x) + self.pos)
        return self.accumulated(x)[:, -1, :]


def factories(flat_dim):
    space = gym.spaces.Box(-np.inf, np.inf, (flat_dim,), dtype=np.float32)
    return {
        "KAN": (lambda w: RealKANFeaturesExtractor(space, features_dim=64,
                                                   hidden_width=int(w), grid=3, k=3), 4, 64),
        "MLP": (lambda w: MLPExtractor(space, features_dim=64, hidden=int(w)), 8, 512),
        "DMLPA": (lambda w: DMLPA(space, hidden_dim=max(32, int(w) // 12 * 12),
                                  features_dim=max(12, int(w) // 12 * 12),
                                  nhead=12, num_layers=2), 12, 480),
    }


def size_to_budget(factory, lo, hi, budget):
    best, best_err = None, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = sum(p.numel() for p in factory(mid).parameters())
        err = abs(n - budget) / budget
        if err < best_err:
            best, best_err = mid, err
        lo, hi = (mid + 1, hi) if n < budget else (lo, mid - 1)
    return best, best_err


def policy_kwargs(arch, width):
    return {
        "KAN": {"features_extractor_class": RealKANFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 64, "hidden_width": width,
                                              "grid": 3, "k": 3}},
        "MLP": {"features_extractor_class": MLPExtractor,
                "features_extractor_kwargs": {"features_dim": 64, "hidden": width}},
        "DMLPA": {"features_extractor_class": DMLPA,
                  "features_extractor_kwargs": {"hidden_dim": max(32, width // 12 * 12),
                                                "features_dim": max(12, width // 12 * 12),
                                                "nhead": 12, "num_layers": 2}},
    }[arch] | {"net_arch": dict(pi=[64, 64], vf=[64, 64])}


def evaluate(model, episodes, seed0=777_000):
    rets = []
    for k in range(episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed0 + k)
        done, total = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
        rets.append(total)
    return float(np.mean(rets)), float(np.std(rets))


def ms_per_decision(model, n=200):
    env = make_env()
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        model.predict(obs, deterministic=True)
    t0 = time.time()
    for _ in range(n):
        model.predict(obs, deterministic=True)
    return 1000.0 * (time.time() - t0) / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=60_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9491, 9492, 9493, 9494, 9495])
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--n-envs", type=int, default=0, help="0 = cores-1")
    ap.add_argument("--arch", nargs="+", default=list(ARCHS))
    ap.add_argument("--output", type=Path,
                    default=Path("results/architecture_bakeoff/result.json"))
    args = ap.parse_args()
    import os
    n_envs = args.n_envs or max(1, min(8, (os.cpu_count() or 2) - 1))
    started = time.perf_counter()

    probe = make_env(0)
    flat_dim = probe.observation_space.shape[0]
    facs = factories(flat_dim)
    sizes = {}
    for name in args.arch:
        factory, lo, hi = facs[name]
        width, err = size_to_budget(factory, lo, hi, TARGET_PARAMS)
        sizes[name] = {"width": width, "error": err,
                       "params": sum(p.numel() for p in factory(width).parameters())}
        print(f"  {name:<6} ancho={width:<4} params={sizes[name]['params']:>9,} "
              f"desv={err:6.1%}", flush=True)
    worst = max(v["error"] for v in sizes.values())
    if worst > PARAM_TOLERANCE:
        raise SystemExit(f"presupuestos a {worst:.1%}: comparar eso mide capacidad, no arquitectura")
    print(f"  todas dentro del {PARAM_TOLERANCE:.0%} · {n_envs} envs · flat={flat_dim}\n", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch in args.arch:
        for seed in args.seeds:
            venv = make_vec(n_envs, seed)
            model = PPO("MlpPolicy", venv, seed=seed, device="cpu", learning_rate=3e-4,
                        n_steps=512, batch_size=64, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, ent_coef=0.01,
                        policy_kwargs=policy_kwargs(arch, sizes[arch]["width"]), verbose=0)
            t0 = time.time()
            model.learn(total_timesteps=args.total_steps, progress_bar=False)
            mean, sd = evaluate(model, args.eval_episodes)
            rows.append({"arch": arch, "seed": seed, "ret_mean": mean, "ret_sd_within": sd,
                         "params": sizes[arch]["params"], "train_s": time.time() - t0,
                         "ms_per_decision": ms_per_decision(model)})
            venv.close()
            # Checkpoint after every seed: a dropped ssh session must not cost the whole run.
            args.output.write_text(json.dumps({"partial": True, "rows": rows}, indent=1))
            print(f"  {arch:<6} seed {seed}  ReT {mean:+.5f} ± {sd:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    by_arch = {}
    for arch in args.arch:
        vals = np.array([r["ret_mean"] for r in rows if r["arch"] == arch])
        by_arch[arch] = {
            "mean": float(vals.mean()), "sd_between_seeds": float(vals.std(ddof=1)),
            "n": int(vals.size), "params": sizes[arch]["params"],
            "ms_per_decision": float(np.mean([r["ms_per_decision"] for r in rows
                                              if r["arch"] == arch])),
        }
    contrasts = {}
    for a in args.arch:
        for b in args.arch:
            if a >= b:
                continue
            va = np.array([r["ret_mean"] for r in rows if r["arch"] == a])
            vb = np.array([r["ret_mean"] for r in rows if r["arch"] == b])
            d = va - vb                       # mismas semillas: contraste pareado
            se = d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else float("nan")
            contrasts[f"{a}_minus_{b}"] = {
                "mean": float(d.mean()), "se": float(se),
                "lcb95": float(d.mean() - 1.96 * se), "ucb95": float(d.mean() + 1.96 * se)}

    payload = {
        "schema_version": "architecture_bakeoff_v1",
        "claim_status": "DEVELOPMENT_ARCHITECTURE_BAKEOFF_NO_ADJUDICATION",
        "scope": "DEVELOPMENT_ONLY_NO_VIRGIN_SEEDS_NO_LEARNER_AUTHORISATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "torch": torch.__version__, "cores": os.cpu_count(), "n_envs": n_envs},
        "design": {"env": "track_b_v1", "obs_version": OBS_VERSION, "history_len": HISTORY_LEN,
                   "max_steps": MAX_STEPS, "total_steps": args.total_steps,
                   "seeds": args.seeds, "eval_episodes": args.eval_episodes,
                   "target_params": TARGET_PARAMS, "arm": "independent_only"},
        "parameter_matching": sizes, "by_arch": by_arch, "contrasts": contrasts, "rows": rows,
        "reading_rule": ("Equal quality goes to the cheaper architecture: that is Delta_efficiency, "
                         "and ms_per_decision is only comparable within this single host."),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=1))
    print("\n  === resumen ===", flush=True)
    for a, v in by_arch.items():
        print(f"    {a:<6} ReT {v['mean']:+.5f} ± {v['sd_between_seeds']:.5f}  "
              f"params {v['params']:,}  {v['ms_per_decision']:.2f} ms/dec")
    for k, v in contrasts.items():
        print(f"    {k:<16} {v['mean']:+.5f} [{v['lcb95']:+.5f}, {v['ucb95']:+.5f}]")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
