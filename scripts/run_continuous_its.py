#!/usr/bin/env python3
"""Continuous I_{t,S} Track-A experiment: does a continuous (rampable) buffer + foresight beat
the best constant-fraction static on Excel ReT / CVaR?

Experiment 1 (foresight-given): obs v6 (forecast + regime visible), MLP policy.
Evaluates the learned continuous policy vs a grid of CONSTANT-fraction static policies
(the continuous analog of the 18-config static grid), on the same bars (Excel ReT, flow,
lost, CVaR, CD), and reports whether the learned policy is ADAPTIVE (frac varies over the
episode) or collapses to a constant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

DEFAULT_CONST_FRACS = [round(i / 10, 2) for i in range(11)]
SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def parse_static_fracs(value: str) -> list[float]:
    fracs = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not fracs:
        raise argparse.ArgumentTypeError("Expected at least one static buffer fraction.")
    bad = [frac for frac in fracs if frac < 0.0 or frac > 1.0]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Static buffer fractions must be in [0,1], got {bad}."
        )
    return fracs


def parse_init_choice(value: str) -> str | float:
    lowered = str(value).strip().lower()
    if lowered in {"best_excel", "best_cvar", "best_cd"}:
        return lowered
    try:
        parsed = float(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a float in [0,1] or one of best_excel,best_cvar,best_cd."
        ) from exc
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("Initial fraction must be in [0,1].")
    return parsed


def build(
    reward,
    obs_v,
    phi,
    psi,
    regime,
    max_steps,
    init_frac,
    *,
    stochastic_pt=False,
    ret_tail_transform="identity",
    ret_tail_gamma=1.0,
    ret_tail_beta=2.0,
    ret_tail_cap_kappa=None,
    ret_tail_inv_kappa=None,
    ret_cd_balanced_n_kappa=0.05,
    ret_excel_cvar_alpha=0.5,
    ret_excel_cvar_tail_level=0.05,
    ret_excel_cvar_window=50,
    seed=None,
):
    ret_tail_kwargs = {}
    if ret_tail_cap_kappa is not None:
        ret_tail_kwargs["ret_tail_cap_kappa"] = float(ret_tail_cap_kappa)
    if ret_tail_inv_kappa is not None:
        ret_tail_kwargs["ret_tail_inv_kappa"] = float(ret_tail_inv_kappa)
    env = make_continuous_its_track_a_env(
        reward_mode=reward, observation_version=obs_v, risk_level=regime,
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi),
        stochastic_pt=bool(stochastic_pt),
        ret_tail_transform=ret_tail_transform,
        ret_tail_gamma=float(ret_tail_gamma),
        ret_tail_beta=float(ret_tail_beta),
        ret_cd_balanced_n_kappa=float(ret_cd_balanced_n_kappa),
        ret_excel_cvar_alpha=float(ret_excel_cvar_alpha),
        ret_excel_cvar_tail_level=float(ret_excel_cvar_tail_level),
        ret_excel_cvar_window=int(ret_excel_cvar_window),
        max_steps=int(max_steps), step_size_hours=168.0,
        init_frac=init_frac,
        **ret_tail_kwargs)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def eval_pol(env, act_fn, episodes, seed0):
    cds, fr, sl = [], [], []
    real = {k: [] for k in ["flow_fill_rate", "lost_rate", "ret_excel",
                            "backorder_qty_final", "service_loss_auc_ration_hours"]}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        sig, fracs = [], []
        done = trunc = False
        info = {}
        while not (done or trunc):
            a = act_fn(obs)
            obs, _r, done, trunc, info = env.step(a)
            sig.append(float(info.get("ret_garrido2024_sigmoid_step", float("nan"))))
            fracs.append(float(info.get("continuous_its_frac", float("nan"))))
        cds.append(float(np.nanmean(sig)) if sig else float("nan"))
        fr.append(fracs)
        m = compute_episode_metrics(env.unwrapped.sim)
        for k in real:
            real[k].append(float(m.get(k, float("nan"))))
        sl.append(float(m.get("service_loss_auc_ration_hours", float("nan"))))
    sl_sorted = sorted(x for x in sl if x == x)
    k = max(1, int(round(0.05 * len(sl_sorted))))
    cvar = float(np.mean(sl_sorted[-k:])) if sl_sorted else float("nan")
    flat = [f for ep in fr for f in ep if f == f]
    return {"cd": float(np.nanmean(cds)),
            "real": {kk: float(np.nanmean(v)) for kk, v in real.items()},
            "cvar95": cvar, "frac_mean": float(np.mean(flat)) if flat else float("nan"),
            "frac_std": float(np.std(flat)) if flat else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="control_v1")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--regime", default="current")
    ap.add_argument("--stochastic-pt", action="store_true")
    ap.add_argument(
        "--ret-tail-transform",
        choices=["identity", "power", "exp_norm"],
        default="identity",
    )
    ap.add_argument("--ret-tail-gamma", type=float, default=1.0)
    ap.add_argument("--ret-tail-beta", type=float, default=2.0)
    ap.add_argument("--ret-tail-cap-kappa", type=float, default=None)
    ap.add_argument("--ret-tail-inv-kappa", type=float, default=None)
    ap.add_argument("--ret-cd-balanced-n-kappa", type=float, default=0.05)
    ap.add_argument("--ret-excel-cvar-alpha", type=float, default=0.5)
    ap.add_argument("--ret-excel-cvar-tail-level", type=float, default=0.05)
    ap.add_argument("--ret-excel-cvar-window", type=int, default=50)
    ap.add_argument(
        "--init-frac",
        type=parse_init_choice,
        default="best_excel",
        help=(
            "Pre-warmup buffer fraction for PPO, or best_excel/best_cvar/best_cd "
            "to select it from the static two-stage surface."
        ),
    )
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument(
        "--static-fracs",
        type=parse_static_fracs,
        default=DEFAULT_CONST_FRACS,
        help=(
            "Comma-separated constant buffer fractions for the continuous static "
            "frontier. Default: 0.0,0.1,...,1.0."
        ),
    )
    ap.add_argument(
        "--static-init-fracs",
        type=parse_static_fracs,
        default=DEFAULT_CONST_FRACS,
        help=(
            "Comma-separated initial prepositioning fractions for the static "
            "two-stage surface. Default: 0.0,0.1,...,1.0."
        ),
    )
    ap.add_argument("--output", default="outputs/experiments/continuous_its_2026-06-27")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # constant-fraction static grid (continuous analog of the 18-config grid)
    const = {}
    for init_f in args.static_init_fracs:
        for f in args.static_fracs:
            for sh, sig in SHIFT_SIGS.items():
                env = build(args.reward_mode, args.observation_version, args.phi, args.psi,
                            args.regime, args.max_steps, init_f,
                            stochastic_pt=args.stochastic_pt,
                            ret_tail_transform=args.ret_tail_transform,
                            ret_tail_gamma=args.ret_tail_gamma,
                            ret_tail_beta=args.ret_tail_beta,
                            ret_tail_cap_kappa=args.ret_tail_cap_kappa,
                            ret_tail_inv_kappa=args.ret_tail_inv_kappa,
                            ret_cd_balanced_n_kappa=args.ret_cd_balanced_n_kappa,
                            ret_excel_cvar_alpha=args.ret_excel_cvar_alpha,
                            ret_excel_cvar_tail_level=args.ret_excel_cvar_tail_level,
                            ret_excel_cvar_window=args.ret_excel_cvar_window)
                r = eval_pol(
                    env,
                    lambda o, ff=f, ss=sig: np.array([ff, ss], dtype=np.float32),
                    args.eval_episodes,
                    5000,
                )
                const[f"init{init_f}_f{f}_S{sh}"] = {
                    "init_frac": float(init_f),
                    "weekly_frac": float(f),
                    "shift": float(sh),
                    "cd": r["cd"],
                    **r["real"],
                    "cvar95": r["cvar95"],
                }
    best_excel = max(const, key=lambda k: const[k]["ret_excel"])
    best_cvar = min(const, key=lambda k: const[k]["cvar95"])
    best_cd = max(const, key=lambda k: const[k]["cd"])
    if args.init_frac == "best_excel":
        learned_init_frac = float(const[best_excel]["init_frac"])
    elif args.init_frac == "best_cvar":
        learned_init_frac = float(const[best_cvar]["init_frac"])
    elif args.init_frac == "best_cd":
        learned_init_frac = float(const[best_cd]["init_frac"])
    else:
        learned_init_frac = float(args.init_frac)

    # learned continuous policy
    learned = []
    for seed in seeds:
        n_envs = max(1, int(args.n_envs))
        venv = DummyVecEnv(
            [
                lambda s=seed + i: build(
                    args.reward_mode,
                    args.observation_version,
                    args.phi,
                    args.psi,
                    args.regime,
                    args.max_steps,
                    learned_init_frac,
                    stochastic_pt=args.stochastic_pt,
                    ret_tail_transform=args.ret_tail_transform,
                    ret_tail_gamma=args.ret_tail_gamma,
                    ret_tail_beta=args.ret_tail_beta,
                    ret_tail_cap_kappa=args.ret_tail_cap_kappa,
                    ret_tail_inv_kappa=args.ret_tail_inv_kappa,
                    ret_cd_balanced_n_kappa=args.ret_cd_balanced_n_kappa,
                    ret_excel_cvar_alpha=args.ret_excel_cvar_alpha,
                    ret_excel_cvar_tail_level=args.ret_excel_cvar_tail_level,
                    ret_excel_cvar_window=args.ret_excel_cvar_window,
                    seed=s,
                )
                for i in range(n_envs)
            ]
        )
        rollout_steps = max(8, min(1024, args.max_steps * 4))
        model = PPO(
            "MlpPolicy",
            venv,
            seed=seed,
            verbose=0,
            n_steps=rollout_steps,
            batch_size=min(64, rollout_steps * n_envs),
            learning_rate=float(args.learning_rate),
            ent_coef=float(args.ent_coef),
            n_epochs=10,
        )
        model.learn(total_timesteps=int(args.timesteps))
        ev = build(args.reward_mode, args.observation_version, args.phi, args.psi,
                   args.regime, args.max_steps, learned_init_frac,
                   stochastic_pt=args.stochastic_pt,
                   ret_tail_transform=args.ret_tail_transform,
                   ret_tail_gamma=args.ret_tail_gamma,
                   ret_tail_beta=args.ret_tail_beta,
                   ret_tail_cap_kappa=args.ret_tail_cap_kappa,
                   ret_tail_inv_kappa=args.ret_tail_inv_kappa,
                   ret_cd_balanced_n_kappa=args.ret_cd_balanced_n_kappa,
                   ret_excel_cvar_alpha=args.ret_excel_cvar_alpha,
                   ret_excel_cvar_tail_level=args.ret_excel_cvar_tail_level,
                   ret_excel_cvar_window=args.ret_excel_cvar_window)
        r = eval_pol(ev, lambda o: model.predict(o, deterministic=True)[0], args.eval_episodes,
                     seed * 100 + 9)
        learned.append({"seed": seed, "cd": r["cd"], **r["real"], "cvar95": r["cvar95"],
                        "frac_mean": r["frac_mean"], "frac_std": r["frac_std"]})

    le = float(np.nanmean([x["ret_excel"] for x in learned]))
    lcv = float(np.nanmean([x["cvar95"] for x in learned]))
    args_payload = dict(vars(args))
    args_payload["init_frac"] = str(args.init_frac)
    summary = {"args": args_payload, "constants": const, "best_const_excel": best_excel,
               "best_const_cvar": best_cvar, "best_const_cd": best_cd,
               "learned_init_frac": learned_init_frac,
               "learned": learned, "learned_excel_mean": le,
               "learned_cvar_mean": lcv}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    be, bc = const[best_excel], const[best_cvar]
    print(f"\n=== CONTINUOUS I_t,S ({args.reward_mode}, obs={args.observation_version}, "
          f"war phi{args.phi}/psi{args.psi}) ===")
    print(f"selected learned init_frac={learned_init_frac:.2f} via {args.init_frac}")
    print(f"best const by Excel: {best_excel} excel={be['ret_excel']:.5f} flow={be['flow_fill_rate']:.3f} "
          f"lost={be['lost_rate']:.3f} cvar={be['cvar95']:.2e}")
    print(f"best const by CVaR : {best_cvar} cvar={bc['cvar95']:.2e} excel={bc['ret_excel']:.5f}")
    bd = const[best_cd]
    print(f"best const by C-D  : {best_cd} cd={bd['cd']:.4f} excel={bd['ret_excel']:.5f} "
          f"cvar={bd['cvar95']:.2e}")
    for x in learned:
        print(f"LEARNED seed{x['seed']}: excel={x['ret_excel']:.5f} flow={x['flow_fill_rate']:.3f} "
              f"lost={x['lost_rate']:.3f} cvar={x['cvar95']:.2e}  frac_mean={x['frac_mean']:.3f} "
              f"frac_std={x['frac_std']:.3f} {'(ADAPTIVE)' if x['frac_std']>0.05 else '(CONSTANT)'}")
    print(f"\nlearned Excel {le:.5f} vs best-const {be['ret_excel']:.5f} -> "
          f"{'WIN' if le>be['ret_excel'] else 'no'};  "
          f"learned CVaR {lcv:.2e} vs best-const {bc['cvar95']:.2e} -> "
          f"{'WIN' if lcv<bc['cvar95'] else 'no'}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
