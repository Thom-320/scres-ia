#!/usr/bin/env python3
"""Decompose the war-CD "win": is PPO's CD advantage real resilience or CD-index gaming?

Re-evaluates a war-cell (phi4/psi1.5) PPO policy vs the static CD frontier with:
  1. CONSISTENT cd aggregation (episode-mean of the per-step sigmoid) for BOTH PPO and static,
  2. REAL service metrics from compute_episode_metrics (war_cd_train read non-existent info keys
     and defaulted them to 0 -- that is why its flow_fill/lost/service all showed 0),
  3. the 5 Cobb-Douglas components (zeta inventory, epsilon backorders, phi spare-capacity,
     tau time, kappa cost) and their per-component log-contribution a*ln(zeta) - b*ln(eps) +
     c*ln(phi) - d*ln(tau) - n*ln(kappa).

Verdict logic: if the CD delta (PPO - best static) lives in epsilon (backorders) / service, it is
real resilience -> worth a confirmatory. If it lives in zeta/phi/kappa (inventory/capacity/cost
positioning) while epsilon/service do NOT improve, it is CD-index gaming -> report as such.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import scripts.war_cd_train as W  # reuse the exact war env builder
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.config import INVENTORY_BUFFERS
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

COMP_KEYS = ["zeta_avg", "epsilon_avg", "phi_avg", "tau_avg", "kappa_dot"]
STATIC_CONFIGS = [("S1_I168", 1, 168), ("S2_I168", 2, 168), ("S3_I336", 3, 336),
                  ("S1_I0", 1, 0), ("S3_I1344", 3, 1344)]


def static_env(regime, phi, psi, max_steps):
    return MFSCGymEnvShifts(
        reward_mode="ReT_garrido2024", observation_version="v4", step_size_hours=168.0,
        max_steps=int(max_steps), risk_level=str(regime), stochastic_pt=False,
        year_basis="thesis", warmup_trigger="op9_arrival", downstream_q_source="figure_6_2",
        r14_defect_mode="thesis_strict_op6", risk_occurrence_mode="thesis_window",
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi),
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=float(P["raw_material_order_up_to_multiplier"]),
        ret_g24_kappa_train_frac=1.0, ret_g24_shift_cost=1.0,
    )


def run_capture(env, action_fn, seed):
    """Return (cd_sigmoid_mean, terminal_components, real_service_metrics)."""
    obs, _ = env.reset(seed=int(seed), options=action_fn.get("reset_options"))
    sig = []
    info = {}
    done = trunc = False
    while not (done or trunc):
        a = action_fn["act"](obs)
        obs, _r, done, trunc, info = env.step(a)
        sig.append(float(info.get("ret_garrido2024_sigmoid_step", float("nan"))))
    comps = {k: float(info.get(k, float("nan"))) for k in COMP_KEYS}
    sim = getattr(env, "sim", None)
    svc = compute_episode_metrics(sim) if sim is not None else {}
    real = {k: float(svc.get(k, float("nan"))) for k in
            ["flow_fill_rate", "lost_rate", "lost_orders", "backorder_qty_final",
             "service_loss_auc_ration_hours", "ret_excel"]}
    cd_mean = float(np.nanmean(sig)) if sig else float("nan")
    return cd_mean, comps, real


def contrib(comps, e):
    lz = math.log(max(comps["zeta_avg"], 1e-9))
    le = math.log(max(comps["epsilon_avg"], 1e-9))
    lp = math.log(max(comps["phi_avg"], 1e-9))
    lt = math.log(max(comps["tau_avg"], 1e-9))
    lk = math.log(max(comps["kappa_dot"], 1e-9))
    return {
        "zeta(+inventory)": e["a"] * lz,
        "epsilon(-backorders)": -e["b"] * le,
        "phi(+spare_cap)": e["c"] * lp,
        "tau(-time)": -e["d"] * lt,
        "kappa(-cost)": -e["n"] * lk,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--regime", default="current")  # war_cd_train trained on risk_level=current
    ap.add_argument("--reward-mode", default="ReT_garrido2024_raw")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--timesteps", type=int, default=15000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--output", default="outputs/experiments/cd_component_decomposition_2026-06-27")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    probe = W.build_env(seed=1, reward_mode=args.reward_mode, phi=args.phi, psi=args.psi,
                        stochastic_pt=False, demand_multiplier=1.0, shift_cost=1.0,
                        kappa_train_frac=1.0, cvar_lambda=0.0, cvar_alpha=0.05,
                        max_steps=args.max_steps)
    e = {"a": float(probe.ret_g24_a_zeta), "b": float(probe.ret_g24_b_epsilon),
         "c": float(probe.ret_g24_c_phi), "d": float(probe.ret_g24_d_tau),
         "n": float(probe.ret_g24_n_kappa)}
    print(f"exponents: {e}")

    # --- PPO ---
    ppo_cd, ppo_comp_acc, ppo_real_acc = [], {k: [] for k in COMP_KEYS}, {}
    for seed in seeds:
        venv = DummyVecEnv([lambda s=seed: W.build_env(
            seed=s, reward_mode=args.reward_mode, phi=args.phi, psi=args.psi,
            stochastic_pt=False, demand_multiplier=1.0, shift_cost=1.0, kappa_train_frac=1.0,
            cvar_lambda=0.0, cvar_alpha=0.05, max_steps=args.max_steps)])
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=min(1024, args.max_steps),
                    batch_size=min(64, args.max_steps), learning_rate=3e-4, n_epochs=10)
        model.learn(total_timesteps=int(args.timesteps))
        ev = W.build_env(seed=seed * 7 + 1, reward_mode=args.reward_mode, phi=args.phi,
                         psi=args.psi, stochastic_pt=False, demand_multiplier=1.0, shift_cost=1.0,
                         kappa_train_frac=1.0, cvar_lambda=0.0, cvar_alpha=0.05,
                         max_steps=args.max_steps)
        for ep in range(args.eval_episodes):
            af = {"act": lambda o: model.predict(o, deterministic=True)[0], "reset_options": None}
            cd, comps, real = run_capture(ev, af, seed * 1000 + ep)
            ppo_cd.append(cd)
            for k in COMP_KEYS:
                ppo_comp_acc[k].append(comps[k])
            for k, v in real.items():
                ppo_real_acc.setdefault(k, []).append(v)
    ppo = {"cd_sigmoid_mean": float(np.nanmean(ppo_cd)),
           "components": {k: float(np.nanmean(ppo_comp_acc[k])) for k in COMP_KEYS},
           "real": {k: float(np.nanmean(v)) for k, v in ppo_real_acc.items()}}

    # --- statics (same env family, dict action) ---
    statics = {}
    for label, sh, period in STATIC_CONFIGS:
        bufs = {k: float(v) for k, v in INVENTORY_BUFFERS[period].items()} if period else None
        cds, comp_acc, real_acc = [], {k: [] for k in COMP_KEYS}, {}
        for ep in range(args.eval_episodes):
            env = static_env(args.regime, args.phi, args.psi, args.max_steps)
            af = {"act": lambda o, s=sh: {"assembly_shifts": int(s)},
                  "reset_options": {"initial_buffers": bufs, "initial_shifts": int(sh),
                                    "inventory_replenishment_period": (float(period) if period else None)}}
            cd, comps, real = run_capture(env, af, 1000 + ep)
            cds.append(cd)
            for k in COMP_KEYS:
                comp_acc[k].append(comps[k])
            for k, v in real.items():
                real_acc.setdefault(k, []).append(v)
        statics[label] = {"cd_sigmoid_mean": float(np.nanmean(cds)),
                          "components": {k: float(np.nanmean(comp_acc[k])) for k in COMP_KEYS},
                          "real": {k: float(np.nanmean(v)) for k, v in real_acc.items()}}

    best_static = max(statics, key=lambda L: statics[L]["cd_sigmoid_mean"])
    bs = statics[best_static]
    ppo_contrib = contrib(ppo["components"], e)
    bs_contrib = contrib(bs["components"], e)
    delta_contrib = {k: ppo_contrib[k] - bs_contrib[k] for k in ppo_contrib}
    total_logscore_delta = sum(delta_contrib.values())

    summary = {"phi": args.phi, "psi": args.psi, "regime": args.regime, "exponents": e,
               "ppo": ppo, "statics": statics, "best_static": best_static,
               "ppo_minus_best_static_cd": ppo["cd_sigmoid_mean"] - bs["cd_sigmoid_mean"],
               "component_contrib_delta_ppo_minus_best": delta_contrib,
               "total_logscore_delta": total_logscore_delta}
    (out / "decomposition.json").write_text(json.dumps(summary, indent=2))

    print("\n=== CD COMPONENT DECOMPOSITION (war phi%.1f/psi%.1f, %s) ===" % (args.phi, args.psi, args.regime))
    print(f"PPO cd_sigmoid_mean={ppo['cd_sigmoid_mean']:.4f}  best_static={best_static} "
          f"cd={bs['cd_sigmoid_mean']:.4f}  Δcd={summary['ppo_minus_best_static_cd']:+.4f}")
    print("\nPer-component log-score contribution Δ (PPO − best static):")
    for k, v in sorted(delta_contrib.items(), key=lambda x: -abs(x[1])):
        print(f"  {k:24} Δ={v:+.5f}")
    print(f"\nREAL service metrics (the thing war_cd_train failed to measure):")
    print(f"  {'metric':28} {'PPO':>12} {best_static:>12}")
    for k in ["flow_fill_rate", "lost_rate", "backorder_qty_final", "ret_excel"]:
        print(f"  {k:28} {ppo['real'].get(k, float('nan')):>12.4f} {bs['real'].get(k, float('nan')):>12.4f}")
    # verdict
    eps_delta = delta_contrib["epsilon(-backorders)"]
    real_service_better = (ppo["real"].get("flow_fill_rate", 0) > bs["real"].get("flow_fill_rate", 0)
                           or ppo["real"].get("backorder_qty_final", 9e9) < bs["real"].get("backorder_qty_final", 9e9))
    driver = max(delta_contrib, key=lambda k: abs(delta_contrib[k]))
    print(f"\nVERDICT: dominant CD driver = {driver}; epsilon(backorder) Δ={eps_delta:+.5f}; "
          f"real service better than static = {real_service_better}")
    print("  => " + ("REAL RESILIENCE (epsilon/service drives it) -> worth a confirmatory"
                     if (driver.startswith('epsilon') or real_service_better)
                     else "CD-INDEX GAMING (inventory/capacity/cost drives it, service not better)"))
    print(f"\nWROTE {out}/decomposition.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
