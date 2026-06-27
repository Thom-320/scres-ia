#!/usr/bin/env python3
"""Diagnose Pepe (RL) on Discrete(18) [6 buffer x 3 shift] + CD reward in the war cell.

Question: on Discrete(18) the agent CAN trivially match any static by emitting one constant
action. If it does not even match the best static, that is an optimization/architecture failure,
not an action-space limit. This script:
  1. trains PPO (and optionally DQN) with CD reward in the war cell, logging a learning curve,
  2. inspects the LEARNED POLICY (action histogram over the 18 buffer x shift configs),
  3. evaluates every CONSTANT action on the SAME env (the static lower bound the agent should match),
  4. reports CD (episode-mean), Excel ReT, flow_fill, lost_rate, and service-loss CVaR95,
all measured the SAME way for learned and constant policies (no aggregation mismatch).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.external_env_interface import make_discrete18_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv


def build(reward_mode, phi, psi, regime, max_steps, seed=None, kappa_train_frac=1.0):
    env = make_discrete18_track_a_env(
        reward_mode=reward_mode, observation_version="v4", risk_level=regime,
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi),
        stochastic_pt=False, year_basis="thesis", warmup_trigger="op9_arrival",
        downstream_q_source="figure_6_2", r14_defect_mode="thesis_strict_op6",
        risk_occurrence_mode="thesis_window", raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=float(P["raw_material_order_up_to_multiplier"]),
        ret_g24_shift_cost=1.0, ret_g24_kappa_train_frac=float(kappa_train_frac),
        step_size_hours=168.0, max_steps=int(max_steps),
    )
    if seed is not None:
        env.reset(seed=int(seed))
    return env


INV_PERIODS = [0, 168, 336, 504, 672, 1344]


def _sim_of(env):
    node = env
    for _ in range(6):
        s = getattr(node, "sim", None)
        if s is not None:
            return s
        node = getattr(node, "env", None)
        if node is None:
            return None
    return None


def eval_policy(env, act_fn, episodes, seed0):
    cds, actions, real = [], [], {k: [] for k in
        ["flow_fill_rate", "lost_rate", "backorder_qty_final", "ret_excel",
         "service_loss_auc_ration_hours"]}
    sl_per_ep = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        sig = []
        done = trunc = False
        info = {}
        while not (done or trunc):
            a = act_fn(obs)
            actions.append(int(a) if np.isscalar(a) else int(np.asarray(a).item()))
            obs, _r, done, trunc, info = env.step(a)
            sig.append(float(info.get("ret_garrido2024_sigmoid_step", float("nan"))))
        cds.append(float(np.nanmean(sig)) if sig else float("nan"))
        sim = _sim_of(env)
        m = compute_episode_metrics(sim) if sim is not None else {}
        for k in real:
            real[k].append(float(m.get(k, float("nan"))))
        sl_per_ep.append(float(m.get("service_loss_auc_ration_hours", float("nan"))))
    sl_sorted = sorted(x for x in sl_per_ep if x == x)
    k = max(1, int(round(0.05 * len(sl_sorted))))
    cvar95 = float(np.mean(sl_sorted[-k:])) if sl_sorted else float("nan")
    return {"cd": float(np.nanmean(cds)), "actions": actions,
            "real": {kk: float(np.nanmean(v)) for kk, v in real.items()},
            "service_loss_cvar95": cvar95}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--regime", default="current")
    ap.add_argument("--reward-mode", default="ReT_garrido2024")
    ap.add_argument("--kappa-train-frac", type=float, default=1.0,
                    help="cost weight in the CD reward; 1.0=full cost (rewards no-buffer), "
                         "0.2=light cost (rewards service/buffer).")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--checkpoints", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--output", default="outputs/experiments/audit_pepe_discrete18_2026-06-27")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    KF = args.kappa_train_frac
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # decode the 18 actions -> (buffer, shift) labels
    probe = build(args.reward_mode, args.phi, args.psi, args.regime, args.max_steps, seed=1, kappa_train_frac=KF)
    decode = probe.decode_discrete_action if hasattr(probe, "decode_discrete_action") else None
    labels = {}
    for a in range(18):
        try:
            lv, sh = (int(x) for x in decode(a))
            labels[a] = f"S{sh + 1}_I{INV_PERIODS[lv]}"
        except Exception:
            labels[a] = str(a)

    # --- constant-action baselines (the static lower bound the agent should match) ---
    const = {}
    for a in range(18):
        env = build(args.reward_mode, args.phi, args.psi, args.regime, args.max_steps, kappa_train_frac=KF)
        r = eval_policy(env, lambda o, aa=a: aa, args.eval_episodes, 5000)
        const[a] = {"label": labels[a], "cd": r["cd"], "real": r["real"],
                    "cvar95": r["service_loss_cvar95"]}
    best_const = max(const, key=lambda a: const[a]["cd"])

    # --- learned policy ---
    learned_runs = []
    curve = []
    for seed in seeds:
        venv = DummyVecEnv([lambda s=seed: build(args.reward_mode, args.phi, args.psi,
                                                  args.regime, args.max_steps, seed=s, kappa_train_frac=KF)])
        Algo = PPO if args.algo == "ppo" else DQN
        kw = dict(seed=seed, verbose=0)
        if args.algo == "ppo":
            kw.update(n_steps=min(1024, args.max_steps * 4), batch_size=64,
                      learning_rate=3e-4, n_epochs=10)
        else:
            kw.update(learning_rate=1e-3, buffer_size=20000, learning_starts=200,
                      batch_size=64, exploration_fraction=0.4, exploration_final_eps=0.05)
        model = Algo("MlpPolicy", venv, **kw)
        chunk = max(1, args.timesteps // args.checkpoints)
        for c in range(args.checkpoints):
            model.learn(total_timesteps=chunk, reset_num_timesteps=(c == 0))
            ev = build(args.reward_mode, args.phi, args.psi, args.regime, args.max_steps, kappa_train_frac=KF)
            r = eval_policy(ev, lambda o: model.predict(o, deterministic=True)[0],
                            args.eval_episodes, seed * 100 + 7)
            curve.append({"seed": seed, "ts": (c + 1) * chunk, "cd": r["cd"]})
        # final detailed eval
        ev = build(args.reward_mode, args.phi, args.psi, args.regime, args.max_steps, kappa_train_frac=KF)
        r = eval_policy(ev, lambda o: model.predict(o, deterministic=True)[0],
                        args.eval_episodes, seed * 100 + 99)
        hist = Counter(labels[a] for a in r["actions"])
        learned_runs.append({"seed": seed, "cd": r["cd"], "real": r["real"],
                             "cvar95": r["service_loss_cvar95"],
                             "action_hist": dict(hist.most_common(6))})

    learned_cd = float(np.nanmean([x["cd"] for x in learned_runs]))
    bc = const[best_const]
    summary = {"phi": args.phi, "psi": args.psi, "regime": args.regime, "algo": args.algo,
               "reward_mode": args.reward_mode, "timesteps": args.timesteps,
               "best_constant": {"action": labels[best_const], **bc},
               "learned_runs": learned_runs, "learned_cd_mean": learned_cd,
               "learned_minus_best_constant_cd": learned_cd - bc["cd"],
               "learning_curve": curve,
               "all_constants_cd": {labels[a]: round(const[a]["cd"], 4) for a in const}}
    (out / "audit.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== PEPE DISCRETE(18) AUDIT (war phi{args.phi}/psi{args.psi}, {args.algo}, "
          f"reward={args.reward_mode}) ===")
    print(f"best CONSTANT action = {labels[best_const]}: cd={bc['cd']:.4f} "
          f"flow_fill={bc['real']['flow_fill_rate']:.3f} lost={bc['real']['lost_rate']:.3f} "
          f"cvar95={bc['cvar95']:.0f}")
    print(f"LEARNED cd={learned_cd:.4f}  (learned - best_constant = {learned_cd - bc['cd']:+.4f})")
    print("\nlearning curve (cd over timesteps):")
    for c in curve:
        print(f"  seed{c['seed']} ts={c['ts']:>6} cd={c['cd']:.4f}")
    print("\nlearned action histogram (what Pepe actually picks):")
    for run in learned_runs:
        print(f"  seed{run['seed']}: {run['action_hist']}  cd={run['cd']:.4f} "
              f"flow_fill={run['real']['flow_fill_rate']:.3f} lost={run['real']['lost_rate']:.3f} "
              f"cvar95={run['cvar95']:.0f}")
    verdict = ("MATCHES/BEATS best static (learning works)" if learned_cd >= bc["cd"] - 1e-3
               else "FAILS to match best static -> optimization/architecture problem")
    print(f"\nVERDICT: {verdict}")
    print(f"WROTE {out}/audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
