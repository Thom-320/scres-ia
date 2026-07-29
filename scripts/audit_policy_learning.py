#!/usr/bin/env python3
"""Audit: IS the policy learning, and WHAT is it learning?

For the continuous_its lane (or any continuous Box(2) frac/shift policy). Answers two questions
that frac_std alone cannot:

  (1) IS IT LEARNING?  learned vs random-init (same net, untrained) vs best-constant, on the eval
      metric, plus a checkpointed training curve. If learned ≈ random or ≈ constant -> not learning
      anything useful.
  (2) WHAT IS IT LEARNING?  Interrogate the policy's action (buffer frac, shift) as a function of the
      OBSERVATION over eval episodes:
        - linear-probe R^2 of frac ~ obs  -> how state-dependent the policy is (≈0 = effectively a
          constant; high = conditions on state).
        - corr(frac, each obs feature)    -> WHICH signals drive the buffer (esp. risk_forecast_*,
          regime one-hot, backorder_rate).
        - anticipation lead/lag: corr(frac[t], risk_forecast_168h[t+k]) for k in {-2..+2}
          -> does the buffer rise BEFORE forecast risk (anticipatory, k>0) or AFTER (reactive, k<0)?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.external_env_interface import get_observation_fields
from supply_chain.episode_metrics import compute_episode_metrics

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def build(reward, obs_v, phi, psi, regime, max_steps, init_frac, seed=None, risk_obs=False):
    env = make_continuous_its_track_a_env(
        reward_mode=reward, observation_version=obs_v, risk_level=regime,
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi),
        stochastic_pt=False, max_steps=int(max_steps), step_size_hours=168.0, init_frac=init_frac,
        risk_obs=bool(risk_obs))
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def eval_metric(env, act_fn, episodes, seed0):
    excels, sl = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = trunc = False
        while not (done or trunc):
            obs, _r, done, trunc, _i = env.step(act_fn(obs))
        m = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(m.get("ret_excel", float("nan"))))
        sl.append(float(m.get("service_loss_auc_ration_hours", float("nan"))))
    return float(np.nanmean(excels)), float(np.nanmean(sl))


def interrogate(env, model, fields, episodes, seed0):
    """Collect per-step (obs, frac, shift) and analyze state-dependence + anticipation."""
    OBS, FRAC, SHIFT = [], [], []
    per_ep_fc, per_ep_frac = [], []  # for lead/lag (aligned within an episode)
    fc_idx = fields.index("risk_forecast_168h_norm") if "risk_forecast_168h_norm" in fields else None
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = trunc = False
        ep_fc, ep_frac = [], []
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)
            OBS.append(np.asarray(obs, dtype=np.float64))
            obs2, _r, done, trunc, info = env.step(a)
            FRAC.append(float(info.get("continuous_its_frac", float("nan"))))
            SHIFT.append(float(info.get("continuous_its_shift", float("nan"))))
            if fc_idx is not None:
                ep_fc.append(float(np.asarray(obs, dtype=np.float64)[fc_idx]))
                ep_frac.append(float(info.get("continuous_its_frac", float("nan"))))
            obs = obs2
        if fc_idx is not None and len(ep_fc) > 6:
            per_ep_fc.append(np.array(ep_fc)); per_ep_frac.append(np.array(ep_frac))
    OBS = np.array(OBS); FRAC = np.array(FRAC); SHIFT = np.array(SHIFT)

    # state-dependence: linear-probe R^2 of frac ~ obs (and shift ~ obs)
    def probe_r2(y):
        if np.std(y) < 1e-9:
            return 0.0
        X = np.column_stack([OBS, np.ones(len(OBS))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    frac_r2 = probe_r2(FRAC); shift_r2 = probe_r2(SHIFT)

    # which features drive frac (|corr|)
    corrs = {}
    for j, name in enumerate(fields):
        col = OBS[:, j]
        if np.std(col) < 1e-9 or np.std(FRAC) < 1e-9:
            corrs[name] = 0.0
        else:
            corrs[name] = float(np.corrcoef(col, FRAC)[0, 1])
    top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:8]

    # anticipation lead/lag: corr(frac[t], forecast[t+k])
    leadlag = {}
    if per_ep_frac:
        for k in (-2, -1, 0, 1, 2):
            xs, ys = [], []
            for fc, fr in zip(per_ep_fc, per_ep_frac):
                if k >= 0:
                    a_, b_ = fr[: len(fr) - k], fc[k:]
                else:
                    a_, b_ = fr[-k:], fc[: len(fc) + k]
                xs.append(a_); ys.append(b_)
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            leadlag[k] = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 1e-9 and np.std(ys) > 1e-9 else 0.0
    return {"frac_r2_obs": frac_r2, "shift_r2_obs": shift_r2, "frac_std": float(np.std(FRAC)),
            "top_frac_feature_corrs": top, "frac_vs_forecast_leadlag": leadlag,
            "n_steps": int(len(FRAC))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--regime", default="current")
    ap.add_argument("--init-frac", type=float, default=1.0)
    ap.add_argument("--risk-obs", action="store_true",
                    help="augment obs with realized active/recent risk one-hots (R11..R3)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--checkpoints", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--output", default="outputs/experiments/policy_learning_audit_2026-06-27")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fields = list(get_observation_fields(args.observation_version))

    _probe = build(reward=args.reward_mode, obs_v=args.observation_version, phi=args.phi, psi=args.psi, regime=args.regime, max_steps=args.max_steps, init_frac=args.init_frac, risk_obs=args.risk_obs)
    if getattr(_probe, 'obs_field_names', None):
        fields = list(_probe.obs_field_names)
    cfg = dict(reward=args.reward_mode, obs_v=args.observation_version, phi=args.phi,
               psi=args.psi, regime=args.regime, max_steps=args.max_steps, init_frac=args.init_frac, risk_obs=args.risk_obs)

    # random-init baseline (untrained net)
    rnd_env = build(**cfg, seed=args.seed)
    rnd_model = PPO("MlpPolicy", DummyVecEnv([lambda: build(**cfg, seed=args.seed)]), seed=args.seed,
                    verbose=0, n_steps=min(1024, args.max_steps * 4), batch_size=64)
    rnd_excel, rnd_sl = eval_metric(rnd_env, lambda o: rnd_model.predict(o, deterministic=True)[0],
                                    args.eval_episodes, 9000)

    # best constant (the trivial policy the learner should beat)
    best_c_excel = -1.0
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        for sh, sig in SHIFT_SIGS.items():
            env = build(**cfg)
            ex, _ = eval_metric(env, lambda o, ff=f, ss=sig: np.array([ff, ss], dtype=np.float32),
                                args.eval_episodes, 5000)
            best_c_excel = max(best_c_excel, ex)

    # train with checkpoint curve
    venv = DummyVecEnv([lambda: build(**cfg, seed=args.seed)])
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0, n_steps=min(1024, args.max_steps * 4),
                batch_size=64, learning_rate=3e-4, n_epochs=10)
    curve = []
    chunk = max(1, args.timesteps // args.checkpoints)
    for c in range(args.checkpoints):
        model.learn(total_timesteps=chunk, reset_num_timesteps=(c == 0))
        ev = build(**cfg)
        ex, _ = eval_metric(ev, lambda o: model.predict(o, deterministic=True)[0], args.eval_episodes, 700)
        curve.append({"ts": (c + 1) * chunk, "excel": ex})
    model.save(str(out / "model.zip"))

    learn_env = build(**cfg)
    learned_excel, learned_sl = eval_metric(
        learn_env, lambda o: model.predict(o, deterministic=True)[0], args.eval_episodes, 9000)

    interro = interrogate(build(**cfg), model, fields, args.eval_episodes, 9000)

    verdict_learning = (learned_excel > rnd_excel + 1e-5 and learned_excel >= best_c_excel - 1e-5)
    state_dependent = interro["frac_r2_obs"] > 0.15
    ll = interro["frac_vs_forecast_leadlag"]
    anticip = (ll.get(1, 0) + ll.get(2, 0)) > (ll.get(-1, 0) + ll.get(-2, 0)) + 0.05

    summary = {"config": cfg, "timesteps": args.timesteps,
               "learning": {"learned_excel": learned_excel, "random_excel": rnd_excel,
                            "best_constant_excel": best_c_excel, "curve": curve,
                            "is_learning": bool(verdict_learning)},
               "what_it_learns": interro,
               "verdicts": {"beats_random_and_constant": bool(verdict_learning),
                            "state_dependent_policy": bool(state_dependent),
                            "anticipatory_frac_leads_forecast": bool(anticip)}}
    (out / "audit.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n=== POLICY LEARNING AUDIT ({args.reward_mode}, {args.observation_version}, "
          f"h{args.max_steps}, war φ{args.phi}/ψ{args.psi}) ===")
    print("[1] IS IT LEARNING?")
    print(f"    learned Excel={learned_excel:.5f} | random-init={rnd_excel:.5f} | "
          f"best-constant={best_c_excel:.5f}")
    print(f"    curve: {[ (c['ts'], round(c['excel'],5)) for c in curve ]}")
    print(f"    -> {'LEARNING (beats random+constant)' if verdict_learning else 'NOT clearly learning (≈ random or ≈ constant)'}")
    print("[2] WHAT IS IT LEARNING?")
    print(f"    frac R^2 ~ obs = {interro['frac_r2_obs']:.3f}  (state-dependence; ≈0 = constant)  "
          f"frac_std={interro['frac_std']:.3f}")
    print(f"    top frac drivers: {[(n, round(c,2)) for n,c in interro['top_frac_feature_corrs'][:5]]}")
    print("    frac vs forecast lead/lag (k>0 = anticipatory): "
          + str({k: round(v, 2) for k, v in interro["frac_vs_forecast_leadlag"].items()}))
    print(f"    -> state_dependent={state_dependent}  anticipatory={anticip}")
    print(f"WROTE {out}/audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
