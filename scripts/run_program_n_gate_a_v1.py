#!/usr/bin/env python3
"""Gate A: does Track B's neural premium survive a paired interval against the RULE?

Contract: `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`
Preregistration: `docs/PREREGISTRO_PUERTA_A_TRACK_B_CUSTODIA_2026-08-09.md`
PI-authorised block 9200001-9200120, three disjoint sub-blocks.

WHY THIS RETRAINS. `results/track_b_nonneural` is the only place in the repository where a network
beats a non-neural comparator: MLP 98.743 against a fitted threshold rule at 97.142. But the
bake-off stored one mean per (architecture, seed) and never called `model.save`, so that +1.60 has
no interval and nothing exists to re-evaluate. Retraining is the only way to get paired episodes.

THE ESTIMAND IS AGAINST THE RULE. Beating the best constant is not a neural premium; a threshold
rule already does that (+0.575 [LCB95 +0.330]). The contrast that decides is `mlp - threshold_rule`,
paired by tape, and the contrast against the constant is reported beside it as a diagnostic.

TWO PLACEBOS THAT KEEP THE NETWORK AND DESTROY ONLY TIME. `shuffled` permutes the frames in the
history stack; `frozen` fills every slot with the CURRENT frame, so present state survives and
history does not. Both reuse the trained weights. If the network beats the rule but not these, the
premium is capacity rather than memory -- which is a different answer to Garrido's Q1, not a
failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import gymnasium as gym                                                          # noqa: E402
from stable_baselines3 import PPO                                                # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv                         # noqa: E402

from run_architecture_bakeoff_v1 import HISTORY_LEN, MAX_STEPS, OBS_VERSION      # noqa: E402
from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402
from supply_chain.external_env_interface import make_track_b_env                 # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest         # noqa: E402

TRAIN_SEEDS = tuple(range(9200001, 9200006))
FIT_SEEDS = tuple(range(9200011, 9200035))
EVAL_SEEDS = tuple(range(9200051, 9200099))
SESOI = 0.01
TARGET_PARAMS, PARAM_TOLERANCE = 200_000, 0.10
N_BOOT = 20_000
OUT = Path("results/program_n/gate_a_track_b/result.json")
MODEL_DIR = Path("results/program_n/gate_a_track_b/models")
CONTRACT = Path("docs/PREREGISTRO_PUERTA_A_TRACK_B_CUSTODIA_2026-08-09.md")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/env_experimental_shifts.py",
           "supply_chain/external_env_interface.py", "supply_chain/falsifiers.py")


class HistoryStack(gym.Wrapper):
    """Frame stack with two declared ablations of TIME only.

    `mode="shuffled"` permutes the frames, keeping the multiset and destroying the order.
    `mode="frozen"` fills every slot with the current frame, so present state survives intact and
    history does not. Neither touches the observation's content, which is what makes the contrast
    a memory contrast rather than an information contrast.
    """

    def __init__(self, env, n: int, mode: str = "real", seed: int = 0):
        super().__init__(env)
        self.n, self.mode = n, mode
        self._rng = np.random.default_rng(seed)
        d = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (d * n,), dtype=np.float32)

    def _view(self):
        if self.mode == "frozen":
            frames = [self.buf[-1]] * self.n
        elif self.mode == "shuffled":
            frames = [self.buf[i] for i in self._rng.permutation(self.n)]
        else:
            frames = self.buf
        return np.concatenate(frames).astype(np.float32)

    def reset(self, **kw):
        o, i = self.env.reset(**kw)
        self.buf = [o] * self.n
        return self._view(), i

    def step(self, a):
        o, r, term, trunc, i = self.env.step(a)
        self.buf = self.buf[1:] + [o]
        return self._view(), r, term, trunc, i


def make_env(mode: str = "real", seed: int | None = None):
    e = HistoryStack(make_track_b_env(observation_version=OBS_VERSION, max_steps=MAX_STEPS),
                     HISTORY_LEN, mode=mode, seed=seed or 0)
    if seed is not None:
        e.reset(seed=seed)
    return e


def rollout(policy, seed: int, mode: str = "real") -> float:
    env = make_env(mode=mode, seed=None)
    obs, _ = env.reset(seed=int(seed))
    done, total = False, 0.0
    while not done:
        obs, r, term, trunc, _ = env.step(policy(obs))
        total += float(r)
        done = term or trunc
    env.close()
    return total


def const_policy(a: np.ndarray):
    return lambda obs: a


def threshold_policy(base: np.ndarray, gain: np.ndarray, idx: np.ndarray):
    """Constant, shifted by signals from the most recent frame -- the arm the network must beat."""
    def act(obs):
        z = np.asarray(obs, dtype=np.float64)
        tail = z[-101:] if z.size >= 101 else z
        return np.clip(base + gain * np.tanh(tail[idx % tail.size]), -1.0, 1.0).astype(np.float32)
    return act


def net_policy(model):
    def act(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a
    return act


def train_one(seed: int, total_steps: int):
    venv = DummyVecEnv([lambda: make_env(seed=seed)])
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", learning_rate=3e-4,
                n_steps=1024, batch_size=256, verbose=0,
                policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])})
    model.learn(total_timesteps=total_steps, progress_bar=False)
    venv.close()
    return model


def paired(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(20260809)
    boot = rng.choice(d, size=(N_BOOT, d.size), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(boot, 2.5)),
            "ucb95": float(np.percentile(boot, 97.5)),
            "favourable": int((d > 0).sum()), "n": int(d.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--train-seeds", type=int, default=len(TRAIN_SEEDS))
    ap.add_argument("--fit-candidates", type=int, default=200)
    ap.add_argument("--refine-steps", type=int, default=100)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()
    started = time.time()

    dim = make_env().action_space.shape[0]
    rng = np.random.default_rng(20260809)

    # --- comparators fitted on the FITTING block only -------------------------------------
    def fit_score(policy) -> float:
        return float(np.mean([rollout(policy, s) for s in FIT_SEEDS[:8]]))

    best_a, best_a_v = None, -np.inf
    for _ in range(args.fit_candidates):
        a = rng.uniform(-1, 1, dim).astype(np.float32)
        v = fit_score(const_policy(a))
        if v > best_a_v:
            best_a, best_a_v = a, v
    # Local refinement. The pre-flight fitted the constant at 59.4 against a rule at 87.3, which
    # would have handed the network a premium over a comparator nobody had actually optimised.
    for _ in range(args.refine_steps):
        cand = np.clip(best_a + rng.normal(0, 0.15, dim), -1, 1).astype(np.float32)
        v = fit_score(const_policy(cand))
        if v > best_a_v:
            best_a, best_a_v = cand, v
    best_rule, best_rule_v = None, -np.inf
    for _ in range(args.fit_candidates):
        gain = rng.uniform(-0.5, 0.5, dim).astype(np.float32)
        idx = rng.integers(0, 101, dim)
        v = fit_score(threshold_policy(best_a, gain, idx))
        if v > best_rule_v:
            best_rule, best_rule_v = (gain, idx), v

    # --- training on the TRAINING block only ----------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models, params = [], []
    for seed in TRAIN_SEEDS[:args.train_seeds]:
        model = train_one(int(seed), args.total_steps)
        path = MODEL_DIR / f"mlp_seed{seed}.zip"
        model.save(path)                       # the bake-off never did this; that is why we retrain
        models.append(model)
        params.append(int(sum(p.numel() for p in model.policy.parameters())))

    untrained = PPO("MlpPolicy", DummyVecEnv([lambda: make_env(seed=7)]), seed=7, device="cpu",
                    policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])})

    # --- evaluation on the FRESH block, every arm on the same tapes -----------------------
    arms = {
        "random_action": lambda obs: rng.uniform(-1, 1, dim).astype(np.float32),
        "constant_best": const_policy(best_a),
        "threshold_rule": threshold_policy(best_a, *best_rule),
        "untrained_net": net_policy(untrained),
    }
    evals = {name: [rollout(p, s) for s in EVAL_SEEDS] for name, p in arms.items()}
    # The network arm is the mean over training replicates on each tape, so the interval is over
    # TAPES with optimizer noise averaged in, not over five numbers.
    evals["mlp"] = [float(np.mean([rollout(net_policy(m), s) for m in models])) for s in EVAL_SEEDS]
    for mode in ("shuffled", "frozen"):
        evals[f"mlp_{mode}_history"] = [
            float(np.mean([rollout(net_policy(m), s, mode=mode) for m in models]))
            for s in EVAL_SEEDS]

    means = {k: float(np.mean(v)) for k, v in evals.items()}
    vs_rule = {k: paired(v, evals["threshold_rule"]) for k, v in evals.items()
               if k != "threshold_rule"}
    vs_const = {k: paired(v, evals["constant_best"]) for k, v in evals.items()
                if k != "constant_best"}
    premium = vs_rule["mlp"]

    blocks = [set(TRAIN_SEEDS), set(FIT_SEEDS), set(EVAL_SEEDS)]
    overlap = sum(len(a & b) for i, a in enumerate(blocks) for b in blocks[i + 1:])
    param_error = max(abs(p - TARGET_PARAMS) / TARGET_PARAMS for p in params)

    checks = {
        "f1_blocks_are_disjoint": F.lt(
            overlap, 1,
            "fitting a comparator on the tapes where it is scored is the leak that inflates every "
            "number below it"),
        "f2_training_actually_moved_the_policy": F.gt(
            paired(evals["mlp"], evals["untrained_net"])["lcb95"], 0.0,
            "200k steps may simply not be enough, in which case nothing below is about memory"),
        "f3_rule_beats_the_constant": F.gt(
            vs_const["threshold_rule"]["lcb95"], 0.0,
            "if the fitted rule cannot beat the best constant it is a straw man, and beating it "
            "would prove nothing"),
        "f4_quality_premium_over_the_rule": F.ge(
            premium["lcb95"], SESOI,
            "this is exactly what the uncustodied run never measured: the +1.60 was a difference "
            "of means with no interval, and it may not survive one"),
        "f5_beats_both_history_placebos": F.gt(
            min(paired(evals["mlp"], evals["mlp_shuffled_history"])["lcb95"],
                paired(evals["mlp"], evals["mlp_frozen_history"])["lcb95"]), 0.0,
            "if the same weights do as well with time destroyed, the premium is capacity and not "
            "memory"),
        "f6_budget_is_matched": F.lt(
            param_error, PARAM_TOLERANCE,
            "an unmatched budget measures capacity rather than architecture"),
        "f7_a_control_must_differ": F.lt(
            vs_rule["random_action"]["ucb95"], 0.0,
            "random must lose to the rule that decides this gate. Checking random against the "
            "CONSTANT was fragile and the pre-flight caught it: a thinly fitted constant lost to "
            "random, failing a control for a reason that had nothing to do with the harness"),
    }
    checks["d1_endpoint_declared_deviation"] = F.disclosure(
        "the endpoint is track_b_v1's own ret_mean rather than the programme's Cobb-Douglas "
        "primary, so the number stays comparable with the +1.60 this gate exists to collect. "
        "Changing the endpoint and the design at once would make the difference unattributable",
        evidence={"target_params": TARGET_PARAMS, "history_len": HISTORY_LEN})
    checks["custody"] = custody_falsifier(sorted(set(TRAIN_SEEDS + FIT_SEEDS + EVAL_SEEDS)))
    summary = F.summarise(checks)

    if not checks["f1_blocks_are_disjoint"]["passed"]:
        status = "BLOCKED_INSTRUMENT"
    elif not checks["f3_rule_beats_the_constant"]["passed"]:
        status = "NO_VALID_NONNEURAL_COMPARATOR"
    elif checks["f4_quality_premium_over_the_rule"]["passed"] and \
            checks["f5_beats_both_history_placebos"]["passed"]:
        status = "TRACK_B_QUALITY_PREMIUM_CONFIRMED_UNDER_CUSTODY"
    elif checks["f4_quality_premium_over_the_rule"]["passed"]:
        status = "PREMIUM_IS_CAPACITY_NOT_MEMORY"
    else:
        status = "TRACK_B_QUALITY_PREMIUM_DID_NOT_SURVIVE_CUSTODY"

    payload = {
        "schema_version": "program_n_gate_a_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DEVELOPMENT",
        "scope": "DEVELOPMENT_PAIRED_ON_A_FRESH_BLOCK_NO_CONFIRMATORY_GRADE",
        "endpoint": "track_b_v1_ret_mean",
        "seeds": sorted(set(TRAIN_SEEDS + FIT_SEEDS + EVAL_SEEDS)),
        "blocks": {"train": list(TRAIN_SEEDS[:args.train_seeds]),
                   "fit": [FIT_SEEDS[0], FIT_SEEDS[-1]],
                   "eval": [EVAL_SEEDS[0], EVAL_SEEDS[-1]]},
        "sesoi": SESOI, "total_steps": args.total_steps,
        "comparator_fit": {"candidates": args.fit_candidates,
                           "refine_steps": args.refine_steps,
                           "constant_fit_score": best_a_v,
                           "rule_fit_score": best_rule_v},
        "params_per_model": params, "param_error": param_error,
        "means": means, "per_tape": evals,
        "quality_premium_over_rule": premium,
        "vs_rule": vs_rule, "vs_constant": vs_const,
        "uncustodied_reference": {
            "artifact": "results/track_b_nonneural/result.json",
            "constant_best": 96.567, "threshold_rule": 97.142, "trained_mlp": 98.743,
            "note": "difference of means with no interval; the number this gate tries to collect"},
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.time() - started,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/track_b_nonneural/result.json"))

    print(f"veredicto: {status}\n")
    for k, v in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24}{v:9.4f}")
    print(f"\n  PRIMA vs regla: {premium['mean']:+.4f} "
          f"[{premium['lcb95']:+.4f}, {premium['ucb95']:+.4f}]  "
          f"{premium['favourable']}/{premium['n']} tapes  vs SESOI {SESOI}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:40} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
