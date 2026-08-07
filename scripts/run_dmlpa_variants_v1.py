#!/usr/bin/env python3
"""Five variants of David's DMLPA at matched parameters, plus the untrained control.

THE ONE I WOULD BET ON IS NOT A HYPERPARAMETER. DMLPA.forward returns x[:, -1, :] -- the last
frame's token -- from a TransformerEncoder with no causal mask. The summary that attention spread
across all sixteen tokens is thrown away. Mean-pooling is one line and has a mechanism; the other
three variants are budget reallocations with weaker stories.

f3 IS THE POINT. Garrido asked on 22 July for learning to be measured as trained against untrained,
and we never ran it. If trained DMLPA does not beat the SAME architecture at zero training steps
with an interval excluding zero, then nothing here is bought by learning and no architecture
comparison means anything -- we would be ranking noise. That result would matter more than any
variant.

Preregistration: docs/PREREGISTRO_VARIANTES_DMLPA_2026-08-07.md
Development. Seeds 9491-9495 are declared development, not virgin.
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

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

from run_architecture_bakeoff_v1 import (  # noqa: E402
    DMLPA, HISTORY_LEN, PARAM_TOLERANCE, TARGET_PARAMS, evaluate, make_env, make_vec,
    size_to_budget,
)

N_BOOT = 5_000
BASE = "dmlpa_base"
UNTRAINED = "dmlpa_untrained"
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


class DMLPAMeanPool(DMLPA):
    """Identical to David's, except the sequence is summarised by its mean rather than its last
    token. The encoder is not causal, so `[:, -1, :]` is an arbitrary read of a bidirectional
    summary; this is a modelling choice rather than a hyperparameter."""

    def forward(self, x):
        import torch  # noqa: F401
        from einops import rearrange
        z = rearrange(x.float(), "b (d k) -> b d k", d=self.factor)
        z = self.latent_rw(z)
        z = self.pre_norm(z + self.pos)
        return self.accumulated(z).mean(dim=1)


def variant_specs(space):
    """Each entry: (factory(width), kwargs-for-policy, search bounds)."""
    def dim(w):
        return max(12, int(w) // 12 * 12)

    return {
        BASE: (lambda w: DMLPA(space, hidden_dim=max(32, dim(w)), features_dim=dim(w),
                               nhead=12, num_layers=2), DMLPA,
               lambda w: dict(hidden_dim=max(32, dim(w)), features_dim=dim(w),
                              nhead=12, num_layers=2), 12, 480),
        "dmlpa_meanpool": (lambda w: DMLPAMeanPool(space, hidden_dim=max(32, dim(w)),
                                                   features_dim=dim(w), nhead=12, num_layers=2),
                           DMLPAMeanPool,
                           lambda w: dict(hidden_dim=max(32, dim(w)), features_dim=dim(w),
                                          nhead=12, num_layers=2), 12, 480),
        "dmlpa_1layer": (lambda w: DMLPA(space, hidden_dim=max(32, dim(w)), features_dim=dim(w),
                                         nhead=12, num_layers=1), DMLPA,
                         lambda w: dict(hidden_dim=max(32, dim(w)), features_dim=dim(w),
                                        nhead=12, num_layers=1), 12, 600),
        "dmlpa_ff2": (lambda w: DMLPA(space, hidden_dim=max(32, dim(w)), features_dim=dim(w),
                                      nhead=12, num_layers=2, ff_mult=2), DMLPA,
                      lambda w: dict(hidden_dim=max(32, dim(w)), features_dim=dim(w),
                                     nhead=12, num_layers=2, ff_mult=2), 12, 600),
        "dmlpa_nhead4": (lambda w: DMLPA(space, hidden_dim=max(32, dim(w)), features_dim=dim(w),
                                         nhead=4, num_layers=2), DMLPA,
                         lambda w: dict(hidden_dim=max(32, dim(w)), features_dim=dim(w),
                                        nhead=4, num_layers=2), 12, 480),
    }


def main() -> int:
    import gymnasium as gym
    from stable_baselines3 import PPO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9491, 9492, 9493, 9494, 9495])
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--n-envs", type=int, default=0)
    ap.add_argument("--arms", nargs="+",
                    help="subset of variants to run; the base is always included")
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/dmlpa_variants/result.json"))
    args = ap.parse_args()
    import os
    n_envs = args.n_envs or max(1, min(8, (os.cpu_count() or 2) - 1))
    started = time.perf_counter()

    probe = make_env(0)
    flat = int(probe.observation_space.shape[0])
    space = gym.spaces.Box(-np.inf, np.inf, (flat,), dtype=np.float32)
    specs = variant_specs(space)
    if args.arms:
        keep = {BASE, *args.arms}
        unknown = set(args.arms) - set(specs)
        if unknown:
            raise SystemExit(f"variantes desconocidas: {sorted(unknown)}")
        specs = {k: v for k, v in specs.items() if k in keep}
        print(f"  subconjunto: {list(specs)}")

    sizes = {}
    for name, (factory, cls, kw, lo, hi) in specs.items():
        w, err = size_to_budget(factory, lo, hi, TARGET_PARAMS)
        n = sum(p.numel() for p in factory(w).parameters())
        sizes[name] = {"width": int(w), "params": int(n), "error": float(err)}
        print(f"  {name:<16} ancho={w:<4} params={n:>9,} desv={100*err:5.1f}%")
    worst = max(v["error"] for v in sizes.values())
    if worst > PARAM_TOLERANCE:
        raise SystemExit(f"presupuesto no igualado: peor desviacion {100*worst:.1f}%")
    print(f"  todas dentro del {100*PARAM_TOLERANCE:.0f}% · {n_envs} envs · flat={flat}\n")

    # Behavioural fingerprint: one fixed batch through each extractor under one fixed seed.
    import torch
    torch.manual_seed(0)
    probe_x = torch.randn(4, flat)
    fingerprints = {}
    for name, (factory, _, _, _, _) in specs.items():
        torch.manual_seed(9201)
        with torch.no_grad():
            fingerprints[name] = factory(sizes[name]["width"])(probe_x).numpy().ravel()[:32]

    arms = list(specs) + [UNTRAINED]
    per_arm = {a: [] for a in arms}
    for name in arms:
        src = BASE if name == UNTRAINED else name
        factory, cls, kw, _, _ = specs[src]
        width = sizes[src]["width"]
        for seed in args.seeds:
            model = PPO("MlpPolicy", make_vec(n_envs, seed), seed=seed, verbose=0,
                        policy_kwargs={"features_extractor_class": cls,
                                       "features_extractor_kwargs": kw(width),
                                       "net_arch": dict(pi=[64, 64], vf=[64, 64])})
            if name != UNTRAINED:
                model.learn(total_timesteps=args.total_steps, progress_bar=False)
            mean, sd = evaluate(model, args.eval_episodes)
            # The vec env is NOT closed by garbage collection here: measured 8 worker processes
            # left behind per cell, 24 alive and 4.41 GB after three cells on a 16 GB machine.
            # Thirty cells would have reached ~240 processes and OOMed around cell twelve.
            try:
                model.env.close()
            except Exception:
                pass
            del model
            per_arm[name].append(mean)
            tag = " (SIN ENTRENAR)" if name == UNTRAINED else ""
            print(f"    {name:<16} semilla {seed}  ReT {mean:+.5f} ± {sd:.5f}"
                  f"  ({time.perf_counter()-started:.0f}s){tag}", flush=True)

    rng = np.random.default_rng(20260807)

    def boot(d):
        draws = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(N_BOOT)]
        return {"mean": float(np.mean(d)), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "p_two_sided": float(2 * min(np.mean(np.asarray(draws) > 0),
                                             np.mean(np.asarray(draws) < 0)))}

    base = np.asarray(per_arm[BASE])
    vs_base = {n: boot(np.asarray(per_arm[n]) - base) for n in arms if n != BASE}
    variants = [n for n in specs if n != BASE]
    ps = [vs_base[n]["p_two_sided"] for n in variants]
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[idx]))
        adj[idx] = run
    for n, a in zip(variants, adj):
        vs_base[n]["holm_adjusted_p"] = a

    learning = boot(base - np.asarray(per_arm[UNTRAINED]))
    winners = [n for n in variants
               if vs_base[n]["lcb95"] > 0 and vs_base[n]["holm_adjusted_p"] < 0.05]
    verdict = ("A_DMLPA_VARIANT_SEPARATES" if winners
               else "ARCHITECTURE_IS_NOT_THE_LEVER_WITHIN_DMLPA")

    print("\n  medias (ReT, mayor es mejor):")
    for n in sorted(arms, key=lambda k: -float(np.mean(per_arm[k]))):
        v = vs_base.get(n)
        tag = "" if v is None else (f"   Δ {v['mean']:+.4f} [{v['lcb95']:+.4f} · {v['ucb95']:+.4f}]"
                                    + (f" holm {v['holm_adjusted_p']:.3f}" if n in variants else ""))
        print(f"    {n:<16} {float(np.mean(per_arm[n])):+.5f}{tag}")
    print(f"\n  entrenado − sin entrenar: {learning['mean']:+.4f} "
          f"[{learning['lcb95']:+.4f} · {learning['ucb95']:+.4f}]")
    print(f"  veredicto: {verdict}\n")

    falsifiers = {
        "f1_all_variants_are_parameter_matched": {
            "passed": worst <= PARAM_TOLERANCE,
            "evidence": {"why_it_can_fail": "David's objection: without matching we measure "
                                            "capacity, not architecture", "sizes": sizes}},
        "f2_the_variants_are_actually_different": {
            # Parameter counts CANNOT separate these: base, meanpool and nhead4 all land on
            # 187,404. The first version keyed on (params, name), which put the name inside the
            # set and made the check unfailable. This compares what the extractors actually
            # COMPUTE on one fixed input under one fixed seed.
            "passed": len({tuple(np.round(v, 6)) for v in fingerprints.values()}) == len(specs),
            "evidence": {"why_it_can_fail": "two variants that compute the same function are one "
                                            "network under two names, and parameter counts do not "
                                            "reveal it -- three of these share a count",
                         "distinct_outputs": len({tuple(np.round(v, 6))
                                                  for v in fingerprints.values()}),
                         "n_variants": len(specs), "sizes": sizes}},
        "f3_training_beats_not_training": {
            "passed": bool(learning["lcb95"] > 0),
            "evidence": {"why_it_can_fail": "Garrido asked for this on 22 July and it was never "
                                            "run. If training does not beat the same architecture "
                                            "at zero steps, nothing here is bought by learning and "
                                            "no architecture comparison is interpretable",
                         **learning}},
        "f4_budgets_are_matched": {
            "passed": True,
            "evidence": {"why_it_can_fail": "an arm trained longer would buy its ranking",
                         "total_steps": args.total_steps,
                         "note": "untrained arm trains 0 steps BY DESIGN and is not a competitor"}},
        "f5_no_fresh_seeds": {
            "passed": True, "not_applicable": True,
            "evidence": {"seeds": args.seeds,
                         "status": "USED_DEVELOPMENT_NOT_VIRGIN, same block as the bake-off"}},
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        lab = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<44} {lab}")

    payload = {
        "schema_version": "dmlpa_variants_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_NOT_VIRGIN_NO_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_VARIANTES_DMLPA_2026-08-07.md",
        "design": {"env": "track_b_v1", "history_len": HISTORY_LEN,
                   "total_steps": args.total_steps, "seeds": args.seeds,
                   "eval_episodes": args.eval_episodes, "target_params": TARGET_PARAMS},
        "sizes": sizes, "means": {a: float(np.mean(per_arm[a])) for a in arms},
        "per_arm": per_arm, "vs_base": vs_base,
        "trained_minus_untrained": learning, "winners": winners,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/architecture_bakeoff/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
