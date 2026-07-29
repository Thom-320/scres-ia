#!/usr/bin/env python3
"""Confirmatory PPO for Track A v2's conservation-respecting contract.

This runner consumes the 5D static gate output and trains PPO on
``MFSCGymEnvShifts(action_contract="track_a_v1")``. The nominal Track A action
space remains six signals, but ``op5_q`` is fixed at neutral because D11 shows
it is inert unless initial Op5 buffers are enabled, which would reintroduce the
exogenous top-up mechanism this lane is designed to avoid.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_a_v2_conservation_3d_gate import make_env  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    label: str
    action: tuple[float, ...]


def read_gate(gate_dir: Path) -> tuple[dict[str, Any], list[str], dict[str, Candidate]]:
    summary = json.loads((gate_dir / "gate_summary.json").read_text())
    regimes = list(summary["best_by_regime"].keys())
    candidates: dict[str, Candidate] = {}
    with (gate_dir / "static_runs.csv").open() as fh:
        for row in csv.DictReader(fh):
            label = row["candidate"]
            if label not in candidates:
                candidates[label] = Candidate(
                    label=label,
                    action=tuple(float(x) for x in json.loads(row["action"])),
                )
    return summary, regimes, candidates


def split_regime(regime: str) -> tuple[str, float, float]:
    family, phi_part, psi_part = regime.split("_", 2)
    return family, float(phi_part.replace("phi", "")), float(psi_part.replace("psi", ""))


def resource_from_action(action: np.ndarray) -> float:
    clipped = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    mults = 1.25 + 0.75 * clipped[:2]
    shift_sig = float(clipped[5]) if clipped.shape[0] > 5 else 0.0
    if shift_sig < -0.33:
        shift = 1
    elif shift_sig < 0.33:
        shift = 2
    else:
        shift = 3
    return float(np.clip(0.5 * ((float(np.mean(mults)) - 0.5) / 1.5) + 0.5 * ((shift - 1) / 2.0), 0.0, 1.0))


def run_episode(regime: str, action_fn, *, seed: int, max_steps: int) -> dict[str, Any]:
    family, phi, psi = split_regime(regime)
    env = make_env(family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed)
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    resources: list[float] = []
    actions: list[np.ndarray] = []
    try:
        while not (done or truncated):
            action = np.asarray(action_fn(obs, regime, env), dtype=np.float32)
            obs, _reward, done, truncated, info = env.step(action)
            resources.append(float(info.get("resource_composite", resource_from_action(action))))
            actions.append(action.copy())
        metrics = compute_episode_metrics(env.unwrapped.sim)
        action_mat = np.vstack(actions) if actions else np.zeros((0, 6), dtype=float)
        return {
            "regime": regime,
            "excel": float(metrics.get("ret_excel", np.nan)),
            "service_loss": float(metrics.get("service_loss_auc_ration_hours", np.nan)),
            "flow_fill": float(metrics.get("flow_fill_rate", np.nan)),
            "lost_rate": float(metrics.get("lost_rate", np.nan)),
            "resource": float(np.nanmean(resources)) if resources else np.nan,
            "action_mean": np.nanmean(action_mat, axis=0).tolist() if len(action_mat) else [],
            "metrics": metrics,
        }
    finally:
        env.close()


def eval_policy(regimes: list[str], action_fn, *, seed0: int, max_steps: int) -> dict[str, Any]:
    rows = [
        run_episode(regime, action_fn, seed=seed0 + i, max_steps=max_steps)
        for i, regime in enumerate(regimes)
    ]
    return {
        "excel": float(np.mean([r["excel"] for r in rows])),
        "service_loss": float(np.mean([r["service_loss"] for r in rows])),
        "flow_fill": float(np.mean([r["flow_fill"] for r in rows])),
        "lost_rate": float(np.mean([r["lost_rate"] for r in rows])),
        "resource": float(np.mean([r["resource"] for r in rows])),
        "by_regime": rows,
    }


def eval_static_frontier(
    regimes: list[str], candidates: dict[str, Candidate], *, seed0: int, max_steps: int
) -> list[dict[str, Any]]:
    out = []
    for cand in candidates.values():
        result = eval_policy(
            regimes,
            lambda _obs, _regime, _env, a=cand.action: np.asarray(a, dtype=np.float32),
            seed0=seed0,
            max_steps=max_steps,
        )
        out.append({"candidate": cand.label, "action": list(cand.action), **result})
    return out


def oracle_actions(
    regimes: list[str], candidates: dict[str, Candidate], *, seed0: int, max_steps: int
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    actions: dict[str, list[float]] = {}
    rows = []
    for i, regime in enumerate(regimes):
        local = eval_static_frontier([regime], candidates, seed0=seed0 + i, max_steps=max_steps)
        best = max(local, key=lambda r: r["excel"])
        actions[regime] = list(best["action"])
        rows.append({"regime": regime, **best})
    result = eval_policy(
        regimes,
        lambda _obs, regime, _env: np.asarray(actions[regime], dtype=np.float32),
        seed0=seed0,
        max_steps=max_steps,
    )
    result["by_regime_static_best"] = rows
    return actions, result


def collect_bc(regimes: list[str], teacher_actions: dict[str, list[float]], *, seed0: int, max_steps: int):
    obs_rows, act_rows = [], []
    for i, regime in enumerate(regimes):
        family, phi, psi = split_regime(regime)
        env = make_env(family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed0 + i)
        target = np.asarray(teacher_actions[regime], dtype=np.float32)
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        while not (done or truncated):
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            act_rows.append(target.copy())
            obs, _reward, done, truncated, _info = env.step(target)
        env.close()
    return np.vstack(obs_rows).astype(np.float32), np.vstack(act_rows).astype(np.float32)


def collect_returns_for_critic(
    regimes: list[str],
    teacher_actions: dict[str, list[float]],
    *,
    seed0: int,
    max_steps: int,
    gamma: float,
):
    """Roll out the teacher/BC-cloned policy and compute discounted return-to-go
    per step, for critic pretraining. Diagnosis this addresses: BC only
    initializes the ACTOR; the critic starts uncalibrated, so the first PPO
    advantage estimates are near-random and can erode a good BC-cloned actor
    before the critic catches up (observed: best checkpoint selected at
    5k-30k/40k steps in 4/5 seeds of the un-pretrained run, 2026-07-03).
    """
    obs_rows: list[np.ndarray] = []
    return_rows: list[float] = []
    for i, regime in enumerate(regimes):
        family, phi, psi = split_regime(regime)
        env = make_env(family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed0 + i)
        target = np.asarray(teacher_actions[regime], dtype=np.float32)
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        ep_obs: list[np.ndarray] = []
        ep_rewards: list[float] = []
        while not (done or truncated):
            ep_obs.append(np.asarray(obs, dtype=np.float32).copy())
            obs, reward, done, truncated, _info = env.step(target)
            ep_rewards.append(float(reward))
        env.close()
        returns = np.zeros(len(ep_rewards), dtype=np.float64)
        running = 0.0
        for t in reversed(range(len(ep_rewards))):
            running = ep_rewards[t] + gamma * running
            returns[t] = running
        obs_rows.extend(ep_obs)
        return_rows.extend(returns.tolist())
    return np.vstack(obs_rows).astype(np.float32), np.asarray(return_rows, dtype=np.float32)


def critic_pretrain(
    model: PPO, obs: np.ndarray, returns: np.ndarray, *, epochs: int, batch_size: int, seed: int
) -> dict[str, float]:
    """Fit ONLY the value-function parameters (mlp_extractor.value_net +
    value_net) against teacher-policy Monte Carlo returns, via a dedicated
    optimizer -- the BC-trained actor parameters are never touched here.
    Returns are z-scored against this rollout's own mean/std as an
    approximation of the reward-normalization scale VecNormalize will use
    once real training starts (its true running ret_rms doesn't exist yet).
    """
    device = model.policy.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ret_mean = float(returns.mean())
    ret_std = float(returns.std()) if float(returns.std()) > 1e-8 else 1.0
    norm_returns = (returns - ret_mean) / ret_std
    ret_t = torch.as_tensor(norm_returns, dtype=torch.float32, device=device)

    value_params = list(model.policy.mlp_extractor.value_net.parameters()) + list(
        model.policy.value_net.parameters()
    )
    optimizer = torch.optim.Adam(value_params, lr=3e-4)
    rng = np.random.default_rng(seed)

    def value_loss(idx=None):
        x = obs_t if idx is None else obs_t[idx]
        y = ret_t if idx is None else ret_t[idx]
        pred = model.policy.predict_values(x).squeeze(-1)
        return torch.nn.functional.mse_loss(pred, y)

    with torch.no_grad():
        initial = float(value_loss().detach().cpu())
    for _ in range(int(epochs)):
        order = rng.permutation(len(obs))
        for start in range(0, len(obs), int(batch_size)):
            idx = torch.as_tensor(order[start : start + int(batch_size)], dtype=torch.long, device=device)
            l = value_loss(idx)
            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(value_params, 0.5)
            optimizer.step()
    with torch.no_grad():
        final = float(value_loss().detach().cpu())
    return {
        "critic_pretrain_loss_initial": initial,
        "critic_pretrain_loss_final": final,
        "critic_pretrain_samples": int(len(obs)),
        "critic_pretrain_ret_mean": ret_mean,
        "critic_pretrain_ret_std": ret_std,
    }


def bc_train(model: PPO, obs: np.ndarray, actions: np.ndarray, *, epochs: int, batch_size: int, seed: int):
    device = model.policy.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)

    def loss(idx=None):
        x = obs_t if idx is None else obs_t[idx]
        y = act_t if idx is None else act_t[idx]
        pred = model.policy.get_distribution(x).mode()
        return torch.nn.functional.mse_loss(pred, y)

    with torch.no_grad():
        initial = float(loss().detach().cpu())
    for _ in range(int(epochs)):
        order = rng.permutation(len(obs))
        for start in range(0, len(obs), int(batch_size)):
            idx = torch.as_tensor(order[start : start + int(batch_size)], dtype=torch.long, device=device)
            l = loss(idx)
            model.policy.optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
    with torch.no_grad():
        final = float(loss().detach().cpu())
    return {"bc_loss_initial": initial, "bc_loss_final": final, "bc_samples": int(len(obs))}


def make_train_env(regime: str, seed: int, max_steps: int):
    family, phi, psi = split_regime(regime)
    return make_env(family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--output", default="outputs/experiments/track_a_v2_conservation_ppo_2026-07-03")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--checkpoint-interval", type=int, default=5000)
    ap.add_argument("--bc-epochs", type=int, default=150)
    ap.add_argument("--bc-batch-size", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--selection-seed0", type=int, default=8000)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--clip-range", type=float, default=0.1)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="SB3 default is 0.95. Raising toward 1.0 shifts advantage estimation toward "
        "observed multi-step returns (less value-function bootstrap dependence) -- tests "
        "whether delayed order-fulfillment credit (see reward-noise audit, 2026-07-03) is "
        "part of why PPO erodes the BC-cloned starting point.",
    )
    ap.add_argument("--teacher", choices=["best_static", "oracle_if_better"], default="oracle_if_better")
    ap.add_argument(
        "--critic-pretrain-epochs",
        type=int,
        default=0,
        help="Fit the value head on teacher-policy Monte Carlo returns before PPO "
        "fine-tuning starts (default 0 = off, matches original behavior). "
        "Addresses critic-lag: BC only initializes the actor, leaving the critic "
        "uncalibrated at the start of RL fine-tuning.",
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.seeds = "1"
        args.n_envs = 1
        args.timesteps = min(args.timesteps, 512)
        args.checkpoint_interval = min(args.checkpoint_interval, 256)
        args.bc_epochs = min(args.bc_epochs, 2)
        if args.critic_pretrain_epochs > 0:
            args.critic_pretrain_epochs = min(args.critic_pretrain_epochs, 2)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gate_summary, regimes, candidates = read_gate(Path(args.gate_dir))

    selection_frontier = eval_static_frontier(
        regimes, candidates, seed0=args.selection_seed0, max_steps=args.max_steps
    )
    heldout_frontier = eval_static_frontier(
        regimes, candidates, seed0=args.eval_seed0, max_steps=args.max_steps
    )
    best_selection = max(selection_frontier, key=lambda r: r["excel"])
    best_heldout = max(heldout_frontier, key=lambda r: r["excel"])
    oracle_by_regime, oracle_selection = oracle_actions(
        regimes, candidates, seed0=args.selection_seed0, max_steps=args.max_steps
    )
    if args.teacher == "oracle_if_better" and oracle_selection["excel"] > best_selection["excel"] + 1e-12:
        teacher_kind = "oracle_by_regime"
        teacher_actions = oracle_by_regime
    else:
        teacher_kind = "best_static"
        teacher_actions = {regime: list(best_selection["action"]) for regime in regimes}

    write_csv(out / "static_frontier_selection.csv", selection_frontier)
    write_csv(out / "static_frontier_heldout.csv", heldout_frontier)
    (out / "teacher.json").write_text(
        json.dumps(
            {
                "teacher_kind": teacher_kind,
                "best_selection": best_selection,
                "oracle_selection": oracle_selection,
                "teacher_actions": teacher_actions,
            },
            indent=2,
            default=float,
        )
    )

    bc_obs_raw, bc_actions = collect_bc(
        regimes, teacher_actions, seed0=args.selection_seed0 + 1000, max_steps=args.max_steps
    )

    learned: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for seed in [int(x.strip()) for x in args.seeds.split(",") if x.strip()]:
        env_fns = [
            (lambda r=regimes[i % len(regimes)], s=seed * 100 + i: make_train_env(r, s, args.max_steps))
            for i in range(int(args.n_envs))
        ]
        venv = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=True, clip_reward=10.0)
        venv.obs_rms.update(bc_obs_raw)
        bc_obs = venv.normalize_obs(bc_obs_raw.copy())
        model = PPO(
            "MlpPolicy",
            venv,
            seed=seed,
            verbose=0,
            n_steps=min(512, args.max_steps * 4),
            batch_size=64,
            learning_rate=args.learning_rate,
            n_epochs=10,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        bc_stats = bc_train(model, bc_obs, bc_actions, epochs=args.bc_epochs, batch_size=args.bc_batch_size, seed=seed)

        if args.critic_pretrain_epochs > 0:
            critic_obs_raw, critic_returns = collect_returns_for_critic(
                regimes,
                teacher_actions,
                seed0=args.selection_seed0 + 2000 + seed,
                max_steps=args.max_steps,
                gamma=model.gamma,
            )
            critic_obs = venv.normalize_obs(critic_obs_raw.copy())
            critic_stats = critic_pretrain(
                model,
                critic_obs,
                critic_returns,
                epochs=args.critic_pretrain_epochs,
                batch_size=args.bc_batch_size,
                seed=seed,
            )
            bc_stats.update(critic_stats)

        best_model_state = None
        best_vec_path = out / f"seed_{seed}_best_vecnormalize.pkl"
        best_score = -1e18
        best_step = 0
        elapsed = 0
        while elapsed < int(args.timesteps):
            chunk = min(int(args.checkpoint_interval), int(args.timesteps) - elapsed)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            elapsed += chunk
            sel = eval_policy(
                regimes,
                lambda obs, _regime, _env, m=model, n=venv: m.predict(
                    n.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0],
                    deterministic=True,
                )[0],
                seed0=args.selection_seed0,
                max_steps=args.max_steps,
            )
            score = sel["excel"]
            checkpoint_rows.append({"seed": seed, "step": elapsed, **{k: sel[k] for k in ("excel", "service_loss", "resource", "flow_fill")}})
            if score > best_score:
                best_score = score
                best_step = elapsed
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.policy.state_dict().items()
                }
                venv.save(best_vec_path)
        if best_model_state is not None:
            model.policy.load_state_dict(best_model_state)
        model.save(out / f"seed_{seed}_best_model.zip")
        held = eval_policy(
            regimes,
            lambda obs, _regime, _env, m=model, n=venv: m.predict(
                n.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0],
                deterministic=True,
            )[0],
            seed0=args.eval_seed0,
            max_steps=args.max_steps,
        )
        held.update({"seed": seed, "selected_step": best_step, "bc": bc_stats})
        learned.append(held)
        venv.close()

    dynamic = {
        "excel": float(np.mean([r["excel"] for r in learned])),
        "service_loss": float(np.mean([r["service_loss"] for r in learned])),
        "flow_fill": float(np.mean([r["flow_fill"] for r in learned])),
        "resource": float(np.mean([r["resource"] for r in learned])),
    }
    deltas = [r["excel"] - best_heldout["excel"] for r in learned]
    verdict = {
        "best_heldout_static": best_heldout,
        "dynamic": dynamic,
        "seed_deltas_vs_best_heldout_static": deltas,
        "raw_ret_win": dynamic["excel"] > best_heldout["excel"],
        "raw_ret_delta_vs_best_heldout_static": dynamic["excel"] - best_heldout["excel"],
        "positive_seeds": int(sum(d > 0 for d in deltas)),
        "seed_count": len(deltas),
    }
    payload = {
        "args": vars(args),
        "gate_summary": gate_summary,
        "regimes": regimes,
        "teacher_kind": teacher_kind,
        "teacher_actions": teacher_actions,
        "best_selection_static": best_selection,
        "oracle_selection": oracle_selection,
        "learned_per_seed": learned,
        "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    write_csv(out / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(
        out / "seed_health.csv",
        [
            {
                "seed": r["seed"],
                "excel": r["excel"],
                "delta_vs_best_static": r["excel"] - best_heldout["excel"],
                "resource": r["resource"],
                "selected_step": r["selected_step"],
                "bc_loss_initial": r["bc"]["bc_loss_initial"],
                "bc_loss_final": r["bc"]["bc_loss_final"],
            }
            for r in learned
        ],
    )
    (out / "report.md").write_text(
        "\n".join(
            [
                "# Track A v2 Conservation PPO",
                "",
                f"Teacher: `{teacher_kind}`.",
                f"Best heldout static: `{best_heldout['candidate']}` Excel={best_heldout['excel']:.6f}.",
                f"Dynamic Excel={dynamic['excel']:.6f}.",
                f"Delta vs best heldout static={verdict['raw_ret_delta_vs_best_heldout_static']:+.6f}.",
                f"Positive seeds: {verdict['positive_seeds']}/{verdict['seed_count']}.",
            ]
        )
    )
    print((out / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
