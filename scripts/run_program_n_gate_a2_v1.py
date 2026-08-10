#!/usr/bin/env python3
"""Gate A2: does the premium survive a WIDENED non-neural class, including one with memory?

Contract: `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`
Preregistration: `docs/PREREGISTRO_PUERTA_A2_COMPARADOR_ARREGLADO_2026-08-09.md`
PI-authorised block 9300001-9300120. Gate A's block is burned and is not reused.

WHY A SUCCESSOR. Gate A's network won by +0.4699 [+0.2372, +0.7024] and the verdict was
NO_VALID_NONNEURAL_COMPARATOR, because f3 asked the rule to beat the constant and it did not. The
reason was the opposite of what f3 guarded: the constant is SATURATED at 98.21 against the rule's
98.30, so the adaptive version has nothing left to improve. That falsifier could not tell a weak
comparator from a saturated one.

THE FIX IS TWO CHANGES. The best non-neural arm is now a MAXIMUM over the family, chosen on the
fitting block; and its validity is checked against an ABSOLUTE FLOOR -- it must beat random and the
untrained network by margin -- so it can fail because the family collapsed, never because one arm
ties another on the same side.

AND THE CLASS IS WIDENED, because Gate A showed a well-searched constant nearly exhausts what we
knew how to write. A linear feedback law is added, and an EWMA rule that uses ORDERED history --
the comparator any memory claim actually has to beat, and the one Gate A lacked when its network
beat the frozen placebo but not the shuffled one.

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

TRAIN_SEEDS = tuple(range(9300001, 9300006))
FIT_SEEDS = tuple(range(9300011, 9300035))
EVAL_SEEDS = tuple(range(9300051, 9300099))
SESOI = 0.01
TARGET_PARAMS, PARAM_TOLERANCE = 200_000, 0.10
N_BOOT = 20_000
OUT = Path("results/program_n/gate_a2_track_b/result.json")
MODEL_DIR = Path("results/program_n/gate_a2_track_b/models")
CONTRACT = Path("docs/PREREGISTRO_PUERTA_A2_COMPARADOR_ARREGLADO_2026-08-09.md")
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


def rollout(policy_factory, seed: int, mode: str = "real") -> float:
    """`policy_factory` is called ONCE per episode. Stateful arms (EWMA) must start clean on every
    tape; reusing one filter across tapes would leak the previous episode into the next."""
    policy = policy_factory() if callable(getattr(policy_factory, "__call__", None)) and \
        getattr(policy_factory, "_is_factory", False) else policy_factory
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


def linear_feedback_policy(W: np.ndarray, b: np.ndarray):
    """a = clip(W . obs_last + b). A classical feedback law, memoryless, on the newest frame."""
    def act(obs):
        z = np.asarray(obs, dtype=np.float64)
        tail = z[-101:] if z.size >= 101 else z
        return np.clip(W @ tail + b, -1.0, 1.0).astype(np.float32)
    return act


def ewma_policy(base: np.ndarray, gain: np.ndarray, idx: np.ndarray, lam: float):
    """base shifted by an EXPONENTIALLY WEIGHTED MEAN of the selected signals.

    This is the arm Gate A lacked. Its network beat the frozen-history placebo but not the
    shuffled one, so having history helped while the ORDER of it was not demonstrated. An
    exponential filter is the simplest structured way to use order, and it is what a memory claim
    actually has to beat -- a network that only ties this has not shown memory value.
    """
    state = {"m": None}

    def act(obs):
        z = np.asarray(obs, dtype=np.float64)
        tail = z[-101:] if z.size >= 101 else z
        sig = tail[idx % tail.size]
        state["m"] = sig if state["m"] is None else lam * state["m"] + (1.0 - lam) * sig
        return np.clip(base + gain * np.tanh(state["m"]), -1.0, 1.0).astype(np.float32)
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

    # Linear feedback: a classical control law on the newest frame.
    best_lin, best_lin_v = None, -np.inf
    for _ in range(args.fit_candidates):
        W = (rng.normal(0, 0.15, (dim, 101)) * (rng.random((dim, 101)) < 0.1)).astype(np.float64)
        v = fit_score(linear_feedback_policy(W, best_a.astype(np.float64)))
        if v > best_lin_v:
            best_lin, best_lin_v = W, v

    # EWMA: the memory-using structured arm. `lam` is fitted with everything else.
    def _ewma_factory(gain, idx, lam):
        f = lambda: ewma_policy(best_a, gain, idx, lam)                      # noqa: E731
        f._is_factory = True
        return f

    best_ewma, best_ewma_v = None, -np.inf
    for _ in range(args.fit_candidates):
        gain = rng.uniform(-0.5, 0.5, dim).astype(np.float32)
        idx = rng.integers(0, 101, dim)
        lam = float(rng.uniform(0.3, 0.95))
        v = fit_score(_ewma_factory(gain, idx, lam))
        if v > best_ewma_v:
            best_ewma, best_ewma_v = (gain, idx, lam), v

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
        "linear_feedback": linear_feedback_policy(best_lin, best_a.astype(np.float64)),
        "ewma_rule": _ewma_factory(*best_ewma),
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
    # THE FIX. The best non-neural arm is the maximum over the family ON THE FITTING BLOCK, never
    # one arm by decree. Gate A designated the rule and then failed because the constant, not the
    # rule, was the strongest -- a verdict about my choice of arm rather than about the network.
    fit_scores = {"constant_best": best_a_v, "threshold_rule": best_rule_v,
                  "linear_feedback": best_lin_v, "ewma_rule": best_ewma_v}
    best_nonneural = max(fit_scores, key=lambda k: fit_scores[k])
    vs_best = {k: paired(v, evals[best_nonneural]) for k, v in evals.items()
               if k != best_nonneural}
    vs_rule = {k: paired(v, evals["threshold_rule"]) for k, v in evals.items()
               if k != "threshold_rule"}
    vs_const = {k: paired(v, evals["constant_best"]) for k, v in evals.items()
                if k != "constant_best"}
    vs_ewma = {k: paired(v, evals["ewma_rule"]) for k, v in evals.items() if k != "ewma_rule"}
    premium = vs_best["mlp"]

    blocks = [set(TRAIN_SEEDS), set(FIT_SEEDS), set(EVAL_SEEDS)]
    overlap = sum(len(a & b) for i, a in enumerate(blocks) for b in blocks[i + 1:])
    param_error = max(abs(p - TARGET_PARAMS) / TARGET_PARAMS for p in params)

    checks = {
        "f1_blocks_are_disjoint": F.lt(
            overlap, 1,
            "fitting a comparator on the tapes where it is scored is the leak that inflates every "
            "number below it"),
        "f2_training_moved_the_policy": F.gt(
            paired(evals["mlp"], evals["untrained_net"])["lcb95"], 0.0,
            "200k steps may simply not be enough, in which case nothing below is about memory"),
        # THE REPAIRED CHECK. Gate A asked one non-neural arm to beat another and failed because
        # the constant was saturated, not weak. Validity is now an ABSOLUTE floor: the family's
        # best must beat random and the untrained network. That can fail if the class collapses,
        # and cannot fail because two arms on the same side tie.
        "f3_nonneural_family_beats_the_floor": F.gt(
            min(paired(evals[best_nonneural], evals["random_action"])["lcb95"],
                paired(evals[best_nonneural], evals["untrained_net"])["lcb95"]), 0.0,
            "if the widened non-neural family cannot clear random and an untrained network, the "
            "whole class is noise and beating it would prove nothing"),
        "f4_quality_premium_over_the_best_nonneural": F.ge(
            premium["lcb95"], SESOI,
            "with the class widened this is easier to fail than in Gate A, and failing would say "
            "that gate's advantage existed only against a class that was too narrow"),
        "f5_beats_the_memory_comparator": F.gt(
            vs_ewma["mlp"]["lcb95"], 0.0,
            "the EWMA rule uses ORDERED history. A network that only ties it has not shown memory "
            "value, whatever it does against the memoryless arms"),
        "f6_beats_both_history_placebos": F.gt(
            min(paired(evals["mlp"], evals["mlp_shuffled_history"])["lcb95"],
                paired(evals["mlp"], evals["mlp_frozen_history"])["lcb95"]), 0.0,
            "in Gate A the shuffled placebo already crossed zero, so this is expected to be the "
            "hardest of the three memory checks"),
        "f7_budget_is_matched": F.lt(
            param_error, PARAM_TOLERANCE,
            "an unmatched budget measures capacity rather than architecture"),
        "f8_a_control_must_differ": F.lt(
            paired(evals["random_action"], evals[best_nonneural])["ucb95"], 0.0,
            "random must lose to the family's best; a harness that cannot separate those agrees "
            "with everything"),
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
    elif not checks["f3_nonneural_family_beats_the_floor"]["passed"]:
        status = "NONNEURAL_FAMILY_COLLAPSED"
    elif not checks["f4_quality_premium_over_the_best_nonneural"]["passed"]:
        status = "NO_QUALITY_PREMIUM_AGAINST_THE_WIDENED_CLASS"
    elif not checks["f5_beats_the_memory_comparator"]["passed"]:
        status = "PREMIUM_IS_CAPACITY_NOT_MEMORY"
    elif checks["f6_beats_both_history_placebos"]["passed"]:
        status = "TRACK_B_MEMORY_PREMIUM_CONFIRMED"
    else:
        status = "PREMIUM_OVER_STRUCTURED_MEMORY_BUT_PLACEBOS_UNRESOLVED"

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
        "best_nonneural_on_fit": best_nonneural, "fit_scores": fit_scores,
        "quality_premium_over_best_nonneural": premium,
        "vs_best_nonneural": vs_best, "vs_rule": vs_rule, "vs_constant": vs_const,
        "vs_ewma": vs_ewma,
        "uncustodied_reference": {
            "artifact": "results/track_b_nonneural/result.json",
            "constant_best": 96.567, "threshold_rule": 97.142, "trained_mlp": 98.743,
            "note": "difference of means with no interval; the number this gate tries to collect"},
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.time() - started,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/program_n/gate_a_track_b/result.json"))

    print(f"veredicto: {status}\n")
    for k, v in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24}{v:9.4f}")
    print(f"\n  mejor no-neuronal en AJUSTE: {best_nonneural} "
          f"({', '.join(f'{k}={v:.2f}' for k, v in fit_scores.items())})")
    print(f"  PRIMA vs mejor no-neuronal: {premium['mean']:+.4f} "
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
