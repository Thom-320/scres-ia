#!/usr/bin/env python3
"""Does the Alzheimer effect survive a normaliser that cannot see the unrun surface?

`run_meta_learner_over_configs_v1.py:189` scales the learner's target by `min/max` over ALL 288
configurations, including the ones the arm never ran. That is load-bearing: `ret_excel_risk_
conditional` is ~0.009 in the R1r contexts, so without the oracle rescaling every gradient step
would be about -0.49 and `rho` would collapse uniformly negative. By this lane's own rule --
`docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md` -- a leak shared by two arms does not invalidate
their contrast, but `memory_vs_ofat` and `memory_vs_random` are exposed, because OFAT and random
never receive that information.

This runner replays the identical CRN surface under two normalisers, and refuses to be interpreted
until it reproduces the sealed artifact under the old one.

Contract: docs/PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md
Seeds: burned block 5_300_001-012, declared replay of `garrido_q2_des288`. No fresh roots.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from scripts.seal_garrido_surface_cache_v1 import verify_sealed_slice  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
}
FACTOR_NAMES = tuple(FACTORS)
CONFIGS = tuple(dict(zip(FACTOR_NAMES, combo))
                for combo in itertools.product(*FACTORS.values()))
DEFAULT = {"buffer_hours": 0.0, "shifts": 1, "op9_rop": 24.0, "op12_rop": 24.0}
CONTEXTS = {
    "R1r": (R1R, {}), "R2r": (R2R, {}), "R1r+R2r": (R1R + R2R, {}),
    "R1r|esc": (R1R, {r: 3.0 for r in R1R}),
    "R2r|esc": (R2R, {r: 3.0 for r in R2R}),
    "R1r+R2r|esc": (R1R + R2R, {r: 3.0 for r in R1R + R2R}),
}
METRIC = "ret_excel_risk_conditional"
SEED_BASE = 5_300_001
NORMALISERS = ("oracle", "prefix")
STRATEGIES = ("ofat", "random", "neuron_reset", "neuron_memory")
NEURAL = ("neuron_reset", "neuron_memory")
#: The sealed anchor. Written by a previous version of this tree; agreement is not something this
#: script can arrange with itself.
SEALED_V2 = Path("results/garrido_meta_learner_v2/result.json")
SEALED_V2_MEANS = {"neuron_memory": 6.986111111111111, "neuron_reset": 14.888888888888891,
                   "ofat": 12.416666666666666, "random": 19.541666666666668}
MODULES = ("supply_chain/supply_chain.py", "supply_chain/config.py",
           "supply_chain/episode_metrics.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def evaluate(config: dict, context: str, seed: int, horizon: float) -> dict:
    """One cell of the surface. Stores the PANEL, not a scalar, so a later endpoint change does
    not invalidate the cache."""
    risks, freq = CONTEXTS[context]
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    sim.run()
    panel = compute_episode_metrics(sim)
    drivers = [float(panel["excel_case_pct_autotomy"]) / 100.0,
               float(panel["excel_case_pct_recovery"]) / 100.0,
               float(panel["excel_case_pct_risk_no_recovery"]) / 100.0,
               float(panel["excel_case_pct_fill_rate"]) / 100.0]
    keep = ("flow_fill_rate", "lost_orders", "delivered_rations", "demanded_rations",
            "ret_excel", "ret_excel_full_ledger", METRIC)
    return {"value": float(panel[METRIC]), "drivers": drivers,
            "panel": {k: float(panel[k]) for k in keep if k in panel}}


def features(config: dict) -> np.ndarray:
    coords = [float(FACTORS[n].index(config[n])) / (len(FACTORS[n]) - 1) for n in FACTOR_NAMES]
    return np.concatenate([np.array(coords), [1.0]])


def cache_is_compatible(blob: dict, current_manifest: dict) -> bool:
    """Accept a sealed cache when only this audit harness changed.

    The evaluator's imported physics modules are the scientific identity.  Adding a falsifier to
    this entry script must not trigger 20,736 duplicate DES episodes; an unsealed or physically
    mismatched cache still fails closed.
    """
    cached_manifest = blob.get("module_manifest", {})
    same_declared_modules = (
        cached_manifest.get("modules") == current_manifest.get("modules")
        and cached_manifest.get("missing") == current_manifest.get("missing")
    )
    if not same_declared_modules:
        return False
    if blob.get("schema_version") == "garrido_surface_cache_v1":
        try:
            verify_sealed_slice(blob)
        except (KeyError, ValueError, TypeError):
            return False
        return True
    return cached_manifest == current_manifest


class Fig5Neuron:
    def __init__(self, dim: int, lr: float = 0.35):
        self.rho = np.zeros(dim)
        self.lr = lr

    def predict(self, x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(self.rho @ x, -30, 30))))

    def update(self, x: np.ndarray, y: float) -> None:
        self.rho += self.lr * (y - self.predict(x)) * x


def search(values: list[float], strategy: str, normaliser: str, neuron, rng,
           budget: int) -> dict:
    """One context. `values` is the CRN surface; `neuron` is carried in by the caller so that
    `neuron_memory` retains rho across contexts and `neuron_reset` does not."""
    n_cfg = len(values)
    best = max(values)
    oracle_lo = min(values)
    oracle_span = best - oracle_lo if best > oracle_lo else 1.0
    seen, curve, running, visited = set(), [], -np.inf, []
    observed: list[float] = []
    current, fi, li = dict(DEFAULT), 0, 0
    factor_best: tuple[float, dict] | None = None
    coord_changes: list[int] = []

    for _ in range(budget):
        if strategy == "random":
            idx = int(rng.integers(0, n_cfg))
        elif strategy == "ofat":
            if fi >= len(FACTOR_NAMES):
                idx = CONFIGS.index(current)
            else:
                name = FACTOR_NAMES[fi]
                cand = dict(current, **{name: FACTORS[name][li]})
                coord_changes.append(sum(1 for n in FACTOR_NAMES if cand[n] != current[n]))
                idx = CONFIGS.index(cand)
                if factor_best is None or values[idx] > factor_best[0]:
                    factor_best = (values[idx], cand)
                li += 1
                if li >= len(FACTORS[name]):
                    current, fi, li = factor_best[1], fi + 1, 0
                    factor_best = None
        else:
            unseen = [i for i in range(n_cfg) if i not in seen] or list(range(n_cfg))
            if len(seen) < 3:
                idx = int(rng.choice(unseen))
            else:
                preds = [neuron.predict(features(CONFIGS[i])) for i in unseen]
                idx = unseen[int(np.argmax(preds))]
        seen.add(idx)
        visited.append(int(idx))
        value = values[idx]
        running = max(running, value)
        curve.append(best - running)
        observed.append(value)
        if neuron is not None:
            if normaliser == "oracle":
                neuron.update(features(CONFIGS[idx]), (value - oracle_lo) / oracle_span)
            else:
                # PREFIX: only what this arm has already run. No update until two DISTINCT values
                # exist, because before that the target is not defined -- declared in the contract.
                lo, hi = min(observed), max(observed)
                if hi > lo:
                    neuron.update(features(CONFIGS[idx]), (value - lo) / max(hi - lo, 1e-12))

    within = next((i + 1 for i, r in enumerate(curve) if r <= 0.01 * abs(best)), budget + 1)
    censored = within > budget
    chosen = max(seen, key=lambda i: values[i])
    denom = budget * abs(best) if best != 0 else 1.0
    return {"regret_curve": curve, "final_regret": curve[-1],
            "auc_regret_norm": float(sum(curve) / denom),
            "runs_to_within_1pct": within, "censored": bool(censored), "best": best,
            "chosen_index": int(chosen), "chosen_config": dict(CONFIGS[chosen]),
            "chosen_value": values[chosen], "visited_sequence": visited,
            "ofat_coordinate_changes": coord_changes}


def run_all(surface, seeds, budget, normaliser, rescale=None) -> dict:
    """Every strategy over every replicate. `rescale` applies a positive affine map per context,
    which f2 uses: an arm that reads a surface-wide statistic in a scale-dependent way changes
    its visit sequence."""
    ctx_order = list(CONTEXTS)
    out = {s: [] for s in STRATEGIES}
    for r, seed in enumerate(seeds):
        for strategy in STRATEGIES:
            rng = np.random.default_rng(90_000 + r)
            neuron = Fig5Neuron(len(FACTOR_NAMES) + 1) if strategy in NEURAL else None
            per_ctx = {}
            for ctx in ctx_order:
                values = [c["value"] for c in surface[(ctx, seed)]]
                if rescale is not None:
                    a, b = rescale[ctx]
                    values = [a * v + b for v in values]
                if strategy == "neuron_reset":
                    neuron = Fig5Neuron(len(FACTOR_NAMES) + 1)
                per_ctx[ctx] = search(values, strategy, normaliser, neuron, rng, budget)
            out[strategy].append(per_ctx)
    return out


def boot_paired(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    draws = rng.integers(0, d.size, size=(int(n_boot), d.size))
    stats = d[draws].mean(axis=1)
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(stats, 2.5)),
            "ucb95": float(np.percentile(stats, 97.5)), "n": int(d.size)}


def per_replicate(res: dict, strategy: str, key: str) -> np.ndarray:
    """Mean over the six contexts, per replicate: the replicate is the inference unit."""
    return np.array([float(np.mean([c[key] for c in rep.values()])) for rep in res[strategy]])


def separability(surface, seeds) -> dict:
    """Additive model vs pairwise interactions on the per-configuration means, per context.

    On a separable surface OFAT is near-optimal by construction, so if this returns nothing the
    mechanism that justifies the whole lane is false.
    """
    out = {}
    coords = np.array([[float(FACTORS[n].index(c[n])) / (len(FACTORS[n]) - 1)
                        for n in FACTOR_NAMES] for c in CONFIGS])
    add = np.hstack([np.ones((len(CONFIGS), 1)), coords])
    inter = np.hstack([add] + [(coords[:, [i]] * coords[:, [j]])
                               for i, j in itertools.combinations(range(4), 2)])
    for ctx in CONTEXTS:
        y = np.array([float(np.mean([surface[(ctx, s)][i]["value"] for s in seeds]))
                      for i in range(len(CONFIGS))])
        r2 = {}
        for name, X in (("additive", add), ("interactions", inter)):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            ss = ((y - X @ beta) ** 2).sum()
            r2[name] = float(1.0 - ss / ((y - y.mean()) ** 2).sum())
        out[ctx] = dict(r2, gain=r2["interactions"] - r2["additive"])
    return out


def optimum_moves(surface, seeds) -> dict:
    """Where the true argmax sits, per context, on the mean surface -- and whether it moves."""
    out = {}
    for ctx in CONTEXTS:
        y = [float(np.mean([surface[(ctx, s)][i]["value"] for s in seeds]))
             for i in range(len(CONFIGS))]
        out[ctx] = {"argmax_config": dict(CONFIGS[int(np.argmax(y))]),
                    "best_mean": float(max(y))}
    keys = {ctx: tuple(v["argmax_config"][n] for n in FACTOR_NAMES) for ctx, v in out.items()}
    return {"by_context": out, "distinct_argmax": len(set(keys.values())),
            "n_contexts": len(keys),
            "moves": bool(len(set(keys.values())) > 1)}


def twin_surface_falsifier(surface, seed: int, budget: int) -> dict:
    """Check that changing only unvisited tail cells cannot change a prefix path.

    The affine-rescaling falsifier is blind to an oracle leak that is invariant to scale.  For each
    normaliser we first record a reference path, alter two cells that no arm on that path visited,
    and replay the same seed/RNG stream.  A legitimate prefix arm must be identical; the oracle
    arm is expected to react because its global min/max changed.  The test is intentionally run on
    one burned seed: it is a structural spy test, not a new scientific replication.
    """
    out = {}
    ctx_order = list(CONTEXTS)
    for normaliser in NORMALISERS:
        reference = run_all(surface, [seed], budget, normaliser)
        twin = {key: [dict(cell) for cell in cells] for key, cells in surface.items()}
        changed = {}
        for ctx in ctx_order:
            protected = {
                index
                for strategy in STRATEGIES
                for index in reference[strategy][0][ctx]["visited_sequence"]
            }
            tail = [index for index in range(len(CONFIGS)) if index not in protected]
            if len(tail) < 2:
                return {"passed": False, "reason": f"not enough hidden tail in {ctx}"}
            low_index, high_index = tail[:2]
            low = dict(twin[(ctx, seed)][low_index])
            high = dict(twin[(ctx, seed)][high_index])
            low["value"] = -10.0
            high["value"] = 10.0
            twin[(ctx, seed)][low_index] = low
            twin[(ctx, seed)][high_index] = high
            changed[ctx] = {"protected": len(protected), "tail_indices": [low_index, high_index]}

        replay = run_all(twin, [seed], budget, normaliser)
        same = {
            strategy: {
                ctx: reference[strategy][0][ctx]["visited_sequence"]
                == replay[strategy][0][ctx]["visited_sequence"]
                for ctx in ctx_order
            }
            for strategy in STRATEGIES
        }
        out[normaliser] = {
            "path_unchanged": same,
            "all_paths_unchanged": all(flag for values in same.values() for flag in values.values()),
            "changed_cells": changed,
        }

    prefix_passed = bool(out["prefix"]["all_paths_unchanged"])
    oracle_reacted = not bool(out["oracle"]["all_paths_unchanged"])
    return {
        "passed": prefix_passed and oracle_reacted,
        "seed": int(seed),
        "budget": int(budget),
        "prefix_passed": prefix_passed,
        "oracle_reacted": oracle_reacted,
        "by_normaliser": out,
        "why_it_can_fail": "an affine-invariant oracle leak can leave the scale test green while changing the path when hidden tails change",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/garrido_normaliser_audit/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [args.seed_base + i for i in range(args.repeats)]
    started = time.perf_counter()

    # ---- surface, cached per (context, seed) ------------------------------------------------
    surface: dict[tuple[str, int], list[dict]] = {}
    manifest = module_manifest(MODULES, script=__file__)
    for ctx in CONTEXTS:
        for seed in seeds:
            slice_path = args.cache_dir / ctx.replace("|", "_").replace("+", "_") / f"{seed}.json"
            if slice_path.is_file():
                blob = json.loads(slice_path.read_text())
                if cache_is_compatible(blob, manifest):
                    surface[(ctx, seed)] = blob["cells"]
                    continue
                print(f"  cache descartada (deriva de modulos): {slice_path}")
            cells = [evaluate(cfg, ctx, seed, horizon) for cfg in CONFIGS]
            slice_path.parent.mkdir(parents=True, exist_ok=True)
            slice_path.write_text(json.dumps(
                {"grid_id": "wrap288_v1", "context": ctx, "seed": seed,
                 "horizon_hours": horizon, "metric": METRIC,
                 "module_manifest": manifest, "cells": cells}, sort_keys=True))
            surface[(ctx, seed)] = cells
        print(f"  superficie {ctx} lista ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- both normalisers on the identical surface ------------------------------------------
    runs = {norm: run_all(surface, seeds, args.budget, norm) for norm in NORMALISERS}
    rng = np.random.default_rng(20260805)

    summary = {}
    for norm in NORMALISERS:
        res = runs[norm]
        cell = {"means_runs_to_within_1pct": {
            s: float(np.mean(per_replicate(res, s, "runs_to_within_1pct"))) for s in STRATEGIES}}
        cell["means_auc_regret_norm"] = {
            s: float(np.mean(per_replicate(res, s, "auc_regret_norm"))) for s in STRATEGIES}
        cell["censoring_rate"] = {
            s: float(np.mean([c["censored"] for rep in res[s] for c in rep.values()]))
            for s in STRATEGIES}
        contrasts = {}
        for key in ("auc_regret_norm", "runs_to_within_1pct"):
            mem = per_replicate(res, "neuron_memory", key)
            # Lower is better for both, so the ADVANTAGE of memory is comparator minus memory.
            contrasts[key] = {
                "memory_vs_reset": boot_paired(per_replicate(res, "neuron_reset", key), mem,
                                               args.n_boot, rng),
                "memory_vs_ofat": boot_paired(per_replicate(res, "ofat", key), mem,
                                              args.n_boot, rng),
                "memory_vs_random": boot_paired(per_replicate(res, "random", key), mem,
                                                args.n_boot, rng)}
        cell["contrasts"] = contrasts
        summary[norm] = cell

    # ---- falsifiers --------------------------------------------------------------------------
    anchor = summary["oracle"]["means_runs_to_within_1pct"]
    anchor_delta = {s: abs(anchor[s] - SEALED_V2_MEANS[s]) for s in STRATEGIES}
    sealed_ok = SEALED_V2.is_file()

    rescale = {ctx: (float(rng.uniform(2.0, 5.0)), float(rng.uniform(-1.0, 1.0)))
               for ctx in CONTEXTS}
    rescaled = {norm: run_all(surface, seeds, args.budget, norm, rescale=rescale)
                for norm in NORMALISERS}

    def seqs(res):
        return [c["visited_sequence"] for s in STRATEGIES for rep in res[s]
                for c in rep.values()]

    f2 = {norm: seqs(runs[norm]) == seqs(rescaled[norm]) for norm in NORMALISERS}
    f3 = all(
        [c["visited_sequence"] for rep in runs["oracle"][s] for c in rep.values()]
        == [c["visited_sequence"] for rep in runs["prefix"][s] for c in rep.values()]
        for s in ("ofat", "random"))

    sep = separability(surface, seeds)
    opt = optimum_moves(surface, seeds)
    twins = twin_surface_falsifier(surface, seeds[0], args.budget)

    falsifiers = {
        "f1_harness_reproduces_the_sealed_artifact": {
            "passed": bool(sealed_ok and max(anchor_delta.values()) < 1e-9),
            "evidence": {"why_it_can_fail": "any drift in the simulator, the metric, the CRN or "
                                            "the RNG consumption order since v2 was sealed breaks "
                                            "the equality -- and then the prefix number would be "
                                            "uninterpretable, because the difference could be "
                                            "physics rather than the normaliser",
                         "sealed_artifact": str(SEALED_V2), "sealed_present": sealed_ok,
                         "sealed_means": SEALED_V2_MEANS, "observed_means": anchor,
                         "abs_delta": anchor_delta}},
        "f2_visit_sequences_are_scale_invariant": {
            "passed": bool(all(f2.values())),
            "evidence": {"why_it_can_fail": "a positive affine rescaling of the surface leaves "
                                            "every legitimate decision rule unchanged; an arm "
                                            "whose sequence moves is reading the surface in a "
                                            "scale-dependent way",
                         "by_normaliser": f2, "rescale": rescale}},
        "f3_non_neural_arms_are_untouched": {
            "passed": bool(f3),
            "evidence": {"why_it_can_fail": "ofat and random never call the normaliser, so their "
                                            "sequences must be byte-identical across arms; if "
                                            "they differ the harness changed something else"}},
        "f4_censoring_is_reported": {
            "passed": bool(summary["oracle"]["censoring_rate"]["random"] > 0.0),
            "evidence": {"why_it_can_fail": "if nothing were censored the old primary would be "
                                            "unbiased and changing the estimand would be "
                                            "unjustified",
                         "rates": {n: summary[n]["censoring_rate"] for n in NORMALISERS}}},
        "f6_surface_twins_do_not_change_prefix_paths": {
            "passed": bool(twins["passed"]),
            "evidence": twins,
        },
        "f5_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))
    falsifiers["not_applicable"] = sorted(
        k for k, v in falsifiers.items() if isinstance(v, dict) and v.get("not_applicable"))

    pref = summary["prefix"]["contrasts"]["auc_regret_norm"]["memory_vs_reset"]
    if not falsifiers["all_passed"]:
        verdict = "HALTED_FALSIFIER_FAILED"
    elif pref["lcb95"] > 0:
        verdict = "ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER"
    else:
        verdict = "ALZHEIMER_EFFECT_DOES_NOT_SURVIVE_AN_HONEST_NORMALISER"

    for norm in NORMALISERS:
        s = summary[norm]
        print(f"\n  === normalizador {norm} ===")
        print("  runs_to_within_1pct:", {k: round(v, 3)
                                         for k, v in s["means_runs_to_within_1pct"].items()})
        print("  auc_regret_norm:    ", {k: round(v, 5)
                                         for k, v in s["means_auc_regret_norm"].items()})
        print("  censura:            ", {k: round(v, 3) for k, v in s["censoring_rate"].items()})
        for key in ("auc_regret_norm", "runs_to_within_1pct"):
            for name, c in s["contrasts"][key].items():
                print(f"    {key:<20} {name:<18} {c['mean']:+.5f} "
                      f"[{c['lcb95']:+.5f}, {c['ucb95']:+.5f}]")
    print("\n  g1 argmax se mueve:", opt["moves"],
          f"({opt['distinct_argmax']}/{opt['n_contexts']} distintos)")
    print("  g2 ganancia de interaccion por contexto:",
          {k: round(v["gain"], 4) for k, v in sep.items()})
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name in ("all_passed", "not_applicable") or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<46} {label}")

    payload = {
        "schema_version": "garrido_normaliser_audit_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_REPLAY_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "BURNED_REPLAY_AUDIT", "replay_of": args.replay_of,
        "module_manifest": manifest,
        "primary_metric": "auc_regret_norm",
        "primary_rationale": (
            "runs_to_within_1pct imputes budget+1 and is censored at very different rates per arm, "
            "so its mean is not comparable across arms. AUC of the normalised regret curve is "
            "defined in every cell and uncensored."),
        "budget": args.budget, "repeats": args.repeats, "seeds": seeds,
        "contexts": list(CONTEXTS), "n_configurations": len(CONFIGS),
        "normalisers": list(NORMALISERS), "strategies": list(STRATEGIES),
        "summary": summary,
        "gates": {"g1_optimum_moves_across_contexts": opt,
                  "g2_surface_separability": sep},
        "falsifiers": falsifiers,
        "twin_surface_falsifier": twins,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=SEALED_V2)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
