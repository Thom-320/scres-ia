#!/usr/bin/env python3
"""Track C Gate C3: learner conversion under the passing calibration.

Pre-registered (docs/TRACK_C_PREREGISTRATION_2026-07-10.md + C3 addendum):
train PPO on the Track C env with a J_v3-aligned reward and evaluate frozen
checkpoints against the SAME-CONTRACT frozen constant on a virgin battery.

Arms:
  scratch — PPO from random init.
  bcwarm  — PPO initialized by behavior-cloning the Gate-C2 frozen
            detector-switcher (the CF20 lesson: PPO maintains what it cannot
            discover), then fine-tuned with the same budget.

Reward: base `ReT_excel_delta` (step change in cumulative Excel-ReT mass)
minus the J_v3 cost terms scaled by ORDERS_PER_STEP_REF, so the episode
return is proportional to J_v3 up to order-count noise.

Verdict (per arm): PPO − frozen constant on tapes 610001-610060,
two-way (seed × tape) bootstrap CI95 + seed direction.
WIN iff CI95 wholly > 0 AND >=4/5 seeds positive AND >=50%+ tapes positive.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from scripts.run_track_b_crossed_eval import episode_metrics_row  # noqa: E402
from scripts.run_track_c_gates import (  # noqa: E402
    MAX_STEPS,
    HazardDetector,
    Policy,
    run_episode,
    two_way_bootstrap_1d,
)
from supply_chain.track_c_env import make_track_c_env  # noqa: E402

# Expected orders per weekly step (6,000 orders / 954 weekly steps).
ORDERS_PER_STEP_REF = 6.29

CONFIRM_MIN = 610_001
CONFIRM_MAX = 610_060
BC_COLLECT_BASE = 620_001  # training-side tapes, never used for any verdict


class JV3RewardWrapper(gym.Wrapper):
    """reward_t = ret_excel_mass_delta_t − Σ λ·cost_t · ORDERS_PER_STEP_REF."""

    def __init__(self, env: gym.Env, lambdas: dict[str, float]):
        super().__init__(env)
        self.lam = {k: float(lambdas[k]) for k in ("lam_h", "lam_d", "lam_s")}
        self._prev = {"holding": 0.0, "dispatch": 0.0, "shift": 0.0, "n": 0}

    def reset(self, **kw):
        self._prev = {"holding": 0.0, "dispatch": 0.0, "shift": 0.0, "n": 0}
        return self.env.reset(**kw)

    def step(self, action):
        obs, base_r, term, trunc, info = self.env.step(action)
        # Under reward_mode="ReT_excel_delta" the base reward IS the step
        # change in cumulative Excel-ReT mass.
        mass_delta = float(base_r)
        econ = info.get("track_c_econ", {})
        n = int(econ.get("n_steps", 1))
        # Per-step (not running-mean) cost terms from the wrapper accumulators.
        step_cost = 0.0
        for key, lam_key in (
            ("holding_frac_mean", "lam_h"),
            ("dispatch_excess_mean", "lam_d"),
            ("shift_excess_mean", "lam_s"),
        ):
            running = float(econ.get(key, 0.0))
            prev_key = key.split("_")[0]
            step_val = running * n - self._prev[prev_key] * (n - 1)
            self._prev[prev_key] = running
            step_cost += self.lam[lam_key] * step_val
        self._prev["n"] = n
        reward = mass_delta - step_cost * ORDERS_PER_STEP_REF
        return obs, float(reward), term, trunc, info


def build_env(campaign: dict[str, Any], lead: float, lambdas: dict[str, float]):
    base = make_track_c_env(
        max_steps=MAX_STEPS,
        campaign_config=campaign,
        inventory_replenishment_lead_time=lead,
        reward_mode="ReT_excel_delta",
    )
    return JV3RewardWrapper(base, lambdas)


def collect_bc_data(pair: dict[str, Any], det: dict[str, Any],
                    campaign: dict[str, Any], lead: float,
                    n_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll the frozen detector-switcher, recording (obs, executed 11D action)."""
    calm = np.asarray(pair["calm"], dtype=np.float32)
    camp = np.asarray(pair["campaign"], dtype=np.float32)
    obs_list, act_list = [], []
    for ep in range(n_episodes):
        env = make_track_c_env(
            max_steps=MAX_STEPS, campaign_config=campaign,
            inventory_replenishment_lead_time=lead,
        )
        obs, _ = env.reset(seed=BC_COLLECT_BASE + ep)
        sim = env.unwrapped.sim
        detector = HazardDetector(det["theta"], det["halflife_weeks"])
        term = trunc = False
        while not (term or trunc):
            action = camp if detector.update(sim, float(sim.env.now)) else calm
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            act_list.append(np.asarray(action, dtype=np.float32).copy())
            obs, _r, term, trunc, _i = env.step(action)
        env.close()
    return np.vstack(obs_list), np.vstack(act_list)


def bc_pretrain(model: PPO, vec_norm: VecNormalize, obs_np: np.ndarray,
                act_np: np.ndarray, epochs: int, seed: int) -> None:
    device = model.policy.device
    obs_n = vec_norm.normalize_obs(obs_np)
    obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_np, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(obs_np))
        for start in range(0, len(obs_np), 128):
            idx = torch.as_tensor(order[start:start + 128], dtype=torch.long, device=device)
            pred = model.policy.get_distribution(obs_t[idx]).mode()
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()


def train_seed(task: tuple) -> dict[str, Any]:
    (arm, seed, out_dir_raw, campaign, lead, lambdas, timesteps,
     bc_pair, bc_det, bc_episodes, bc_epochs) = task
    out_dir = Path(out_dir_raw)
    # Normalize obs stats warm-up envs share the training tape stream.
    venv = DummyVecEnv([lambda: build_env(campaign, lead, lambdas)])
    vec_norm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", vec_norm,
        learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.98, clip_range=0.2, ent_coef=0.005,
        policy_kwargs={"net_arch": [256, 256]}, seed=seed, verbose=0,
        device="cpu",
    )
    if arm == "bcwarm":
        obs_np, act_np = collect_bc_data(bc_pair, bc_det, campaign, lead, bc_episodes)
        # Fit normalization stats on the BC corpus before cloning.
        for row in obs_np[:: max(1, len(obs_np) // 512)]:
            vec_norm.obs_rms.update(row[None, :])
        bc_pretrain(model, vec_norm, obs_np, act_np, bc_epochs, seed)
    model.learn(total_timesteps=int(timesteps), progress_bar=False)
    vec_norm.training = False
    seed_dir = out_dir / "models" / arm / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(seed_dir / "ppo_model.zip"))
    vec_norm.save(str(seed_dir / "vec_normalize.pkl"))
    return {"arm": arm, "seed": seed, "dir": str(seed_dir)}


def eval_ckpt(task: tuple) -> list[dict[str, Any]]:
    (arm, seed, seed_dir_raw, tapes, campaign, lead) = task
    seed_dir = Path(seed_dir_raw)
    model = PPO.load(str(seed_dir / "ppo_model.zip"), device="cpu")
    dummy = DummyVecEnv([lambda: build_env(campaign, lead,
                                           {"lam_h": 0, "lam_d": 0, "lam_s": 0})])
    vec_norm = VecNormalize.load(str(seed_dir / "vec_normalize.pkl"), dummy)
    vec_norm.training = False
    rows = []
    for tape in tapes:
        env = make_track_c_env(
            max_steps=MAX_STEPS, campaign_config=campaign,
            inventory_replenishment_lead_time=lead,
        )
        obs, info = env.reset(seed=tape)
        term = trunc = False
        while not (term or trunc):
            obs_n = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
            action, _ = model.predict(obs_n, deterministic=True)
            obs, _r, term, trunc, info = env.step(np.asarray(action[0], dtype=np.float32))
        row = episode_metrics_row(env.unwrapped.sim)
        econ = info.get("track_c_econ", {})
        row.update({f"econ_{k}": float(v) for k, v in econ.items()})
        rows.append({"arm": arm, "train_seed": seed, "eval_seed": tape, **row})
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gates-dir", type=Path, required=True,
                        help="Passing iteration's gate dir (frozen artifacts)")
    parser.add_argument("--arms", nargs="+", default=["scratch", "bcwarm"],
                        choices=["scratch", "bcwarm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-timesteps", type=int, default=60_000)
    parser.add_argument("--bc-episodes", type=int, default=30)
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument("--test-tapes", type=int, default=60)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    gates = args.gates_dir
    campaign = json.loads((gates / "campaign_config.json").read_text())["campaign"]
    lead = json.loads((gates / "campaign_config.json").read_text())["lead"]
    lambdas = json.loads((gates / "lambdas.json").read_text())
    const = json.loads((gates / "frozen_constant.json").read_text())["constant"]
    pair = json.loads((gates / "frozen_pair.json").read_text())
    det = json.loads((gates / "frozen_detector.json").read_text())

    tapes = list(range(CONFIRM_MIN, CONFIRM_MIN + args.test_tapes))
    assert max(tapes) <= CONFIRM_MAX

    if not args.skip_train:
        train_tasks = [
            (arm, seed, str(out), campaign, lead, lambdas, args.train_timesteps,
             pair, det, args.bc_episodes, args.bc_epochs)
            for arm in args.arms for seed in args.seeds
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for res in pool.map(train_seed, train_tasks, chunksize=1):
                print(f"trained {res['arm']} seed {res['seed']}", flush=True)

    # Reference rows: frozen constant + frozen detector-switcher on the battery.
    ref_rows = []
    for name, pol in (
        ("frozen_constant", Policy("frozen_constant", calm=tuple(const["calm"]))),
        ("detector_switcher", Policy(
            "detector_switcher", calm=tuple(pair["calm"]),
            campaign=tuple(pair["campaign"]),
            detector=(det["theta"], det["halflife_weeks"]))),
    ):
        for tape in tapes:
            r = run_episode(pol, tape, campaign=campaign, lead=lead)
            ref_rows.append({"arm": name, "train_seed": 0, "eval_seed": tape, **r})
        print(f"reference {name} done", flush=True)

    eval_tasks = [
        (arm, seed, str(out / "models" / arm / f"seed{seed}"), tapes, campaign, lead)
        for arm in args.arms for seed in args.seeds
    ]
    ppo_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for block in pool.map(eval_ckpt, eval_tasks, chunksize=1):
            ppo_rows.extend(block)
    print("checkpoint eval done", flush=True)

    all_rows = ref_rows + ppo_rows
    keys: list[str] = []
    for r in all_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (out / "c3_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    def J(r):
        return (float(r["ret_excel"])
                - lambdas["lam_h"] * float(r["econ_holding_frac_mean"])
                - lambdas["lam_d"] * float(r["econ_dispatch_excess_mean"])
                - lambdas["lam_s"] * float(r["econ_shift_excess_mean"]))

    const_by = {int(r["eval_seed"]): J(r) for r in ref_rows if r["arm"] == "frozen_constant"}
    result: dict[str, Any] = {
        "lambdas": lambdas,
        "reference_means_J": {
            name: float(np.mean([J(r) for r in ref_rows if r["arm"] == name]))
            for name in ("frozen_constant", "detector_switcher")
        },
    }
    for arm in args.arms:
        deltas = np.array([
            [J(r) - const_by[int(r["eval_seed"])]
             for r in ppo_rows if r["arm"] == arm and int(r["train_seed"]) == s]
            for s in args.seeds
        ])
        per_seed = deltas.mean(axis=1)
        flat = deltas.mean(axis=0)
        # two-way bootstrap over seeds x tapes
        rng = np.random.default_rng(0)
        boots = np.empty(10_000)
        n_s, n_t = deltas.shape
        for b in range(10_000):
            si = rng.integers(0, n_s, n_s)
            ti = rng.integers(0, n_t, n_t)
            boots[b] = deltas[np.ix_(si, ti)].mean()
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        win = bool(lo > 0 and (per_seed > 0).sum() >= 4 and (flat > 0).sum() >= int(0.8 * n_t))
        result[arm] = {
            "ppo_minus_constant_J": {
                "mean": float(deltas.mean()), "two_way_ci95": [lo, hi],
                "per_seed": [float(v) for v in per_seed],
                "seeds_positive": int((per_seed > 0).sum()),
                "tapes_positive": int((flat > 0).sum()), "n_tapes": n_t,
            },
            "win": win,
        }
    result["verdict"] = (
        "ADAPTIVE_WIN" if any(result[a]["win"] for a in args.arms) else "NO_WIN"
    )
    (out / "c3_verdict.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
