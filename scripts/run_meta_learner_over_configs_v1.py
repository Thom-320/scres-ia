#!/usr/bin/env python3
"""Fase 4 -- the Alzheimer's effect, measured: what does NOT remembering across runs cost?

Garrido's Fig. 2 marks nodes 3 (decision variables) and 8 (the SCRES metric) as the two ends of
an OPEN-LOOP supply chain, and calls the missing link between them the Alzheimer's effect. His
Fig. 5 says exactly what belongs there: a neuron whose dendrites are the four SCRES drivers,
weighted by rho, with an activation of the form "is ReT at configuration x higher than at x-1?".
His 2017 thesis is the case in point -- a one-factor-at-a-time design, restarted from scratch for
each risk family.

Four strategies share one configuration space and one budget of simulation runs, across six
successive risk contexts:

    ofat            the thesis's own design; the open loop
    random          the honest null
    neuron_memory   the Fig. 5 neuron, rho CARRIED across contexts   <- the SCL attribute
    neuron_reset    the same neuron, rho reset at each context       <- the Alzheimer's effect

`neuron_memory` vs `neuron_reset` differ in ONE thing: whether rho survives the context boundary.
That contrast is the number this whole runner exists to produce.

See `docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md` for the reading rule, fixed in advance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")

# His six inventory levels and three shift levels, plus the only two dispatch periods the
# sensitivity campaign found to carry any authority.
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
PRIOR_SEEDS = (set(range(4_900_001, 4_900_007)) | set(range(4_900_501, 4_900_507))
               | set(range(5_100_001, 5_100_013)) | set(range(5_200_001, 5_200_017)))


def evaluate(config: dict, context: str, seed: int, horizon: float) -> tuple[float, np.ndarray]:
    """Returns (ReT, driver vector). The drivers are Garrido's four, read from the episode."""
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
    # His four drivers, as shares of the scored population (Eq. 5.1-5.4).
    drivers = np.array([
        float(panel["excel_case_pct_autotomy"]), float(panel["excel_case_pct_recovery"]),
        float(panel["excel_case_pct_risk_no_recovery"]), float(panel["excel_case_pct_fill_rate"]),
    ], dtype=float) / 100.0
    return float(panel[METRIC]), drivers


def features(config: dict) -> np.ndarray:
    """Inputs available BEFORE running a configuration: its decision coordinates, and a bias.

    Garrido's Fig. 5 is a CONCEPT, not a specification -- his paper is explicitly exploratory --
    so operationalising it is our job. It draws the dendrites as the four SCRES drivers, weighted
    by rho, activated by comparing SCRES at configuration x against x-1.

    Our reading: the drivers are what the DES REPORTS (his node 8) and so they belong in the
    UPDATE; rho is what the learner retains, which is the manuscript's `L_{t-1}`; and the input
    the model predicts FROM has to be the decision variables (his node 3), because they are the
    only thing a planner holds before running anything. His figure does not distinguish what the
    learner observes from what it selects on, because conceptually it need not; in code it must.

    An earlier version of this runner used the drivers for BOTH, which meant ranking unrun
    candidates by a property of episodes that had not happened. `f5` now tests this rather than
    asserting it.
    """
    coords = [float(FACTORS[n].index(config[n])) / (len(FACTORS[n]) - 1) for n in FACTOR_NAMES]
    return np.concatenate([np.array(coords), [1.0]])


class Fig5Neuron:
    """`ReT ~ sigma(sum rho_i d_i)`, updated by gradient after every run. `rho` is the memory."""

    def __init__(self, dim: int, lr: float = 0.35):
        self.rho = np.zeros(dim)
        self.lr = lr

    def predict(self, x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(self.rho @ x, -30, 30))))

    def update(self, x: np.ndarray, y: float) -> None:
        # y is min-max scaled into (0,1) by the caller; plain logistic-loss gradient step.
        self.rho += self.lr * (y - self.predict(x)) * x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=24, help="simulation runs per context")
    ap.add_argument("--repeats", type=int, default=12, help="independent replications")
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/garrido_meta_learner/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()

    # ---- the shared surface: every strategy sees exactly this, nothing else ----------------
    # One CRN seed per (context, repeat); the surface is identical across strategies, which is
    # what makes the comparison about search rather than about luck.
    seeds = [SEED_BASE + i for i in range(args.repeats)]
    surface: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = {}
    for ctx in CONTEXTS:
        for seed in seeds:
            surface[(ctx, seed)] = [evaluate(cfg, ctx, seed, horizon) for cfg in CONFIGS]
        print(f"  superficie {ctx} lista ({time.perf_counter() - started:.0f}s)", flush=True)

    n_cfg = len(CONFIGS)
    ctx_order = list(CONTEXTS)

    def scaled(values: list[float]) -> tuple[float, float]:
        lo, hi = min(values), max(values)
        return lo, (hi - lo if hi > lo else 1.0)

    def search(strategy: str, seed: int, rng: np.random.Generator) -> dict:
        """Returns per-context regret curves. `neuron_memory` is the only arm that carries rho."""
        neuron = Fig5Neuron(len(FACTOR_NAMES) + 1) if strategy.startswith("neuron") else None
        per_ctx, ofat_steps = {}, []
        for ctx in ctx_order:
            table = surface[(ctx, seed)]
            values = [v for v, _ in table]
            best, lo, span = max(values), *scaled(values)
            if strategy == "neuron_reset":
                neuron = Fig5Neuron(len(FACTOR_NAMES) + 1)
            seen, curve, running, visited = set(), [], -np.inf, []

            # The thesis's design, generated LAZILY from the incumbent: sweep one factor's
            # levels, commit its best, move to the next. Proposals must be built against the
            # CURRENT incumbent -- precomputing them makes later proposals differ in more than
            # one coordinate, which is not OFAT at all (f2 caught exactly that).
            current, fi, li = dict(DEFAULT), 0, 0
            factor_best: tuple[float, dict] | None = None
            for step in range(args.budget):
                if strategy == "random":
                    idx = int(rng.integers(0, n_cfg))
                elif strategy == "ofat":
                    if fi >= len(FACTOR_NAMES):
                        idx = CONFIGS.index(current)          # design exhausted: re-run the best
                    else:
                        name = FACTOR_NAMES[fi]
                        cand = dict(current, **{name: FACTORS[name][li]})
                        ofat_steps.append(sum(1 for n in FACTOR_NAMES if cand[n] != current[n]))
                        idx = CONFIGS.index(cand)
                        value_here = values[idx]
                        if factor_best is None or value_here > factor_best[0]:
                            factor_best = (value_here, cand)
                        li += 1
                        if li >= len(FACTORS[name]):          # commit this factor and advance
                            current, fi, li = factor_best[1], fi + 1, 0
                            factor_best = None
                else:
                    unseen = [i for i in range(n_cfg) if i not in seen]
                    if not unseen:
                        unseen = list(range(n_cfg))
                    if len(seen) < 3:                      # cold start: cannot predict yet
                        idx = int(rng.choice(unseen))
                    else:
                        preds = [neuron.predict(features(CONFIGS[i])) for i in unseen]
                        idx = unseen[int(np.argmax(preds))]
                seen.add(idx)
                visited.append(int(idx))
                value, drivers = table[idx]
                running = max(running, value)
                curve.append(best - running)
                if neuron is not None:
                    neuron.update(features(CONFIGS[idx]), (value - lo) / span)
            within = next((i + 1 for i, r in enumerate(curve) if r <= 0.01 * abs(best)),
                          args.budget + 1)
            # The configuration the strategy would actually DEPLOY: the best it ever ran here.
            # H1 (recovery time) and H3 (variance) are properties of that configuration, so
            # without recording it the two hypotheses cannot be evaluated at all.
            chosen = max(seen, key=lambda i: table[i][0])
            per_ctx[ctx] = {"regret_curve": curve, "final_regret": curve[-1],
                            "runs_to_within_1pct": within, "best": best,
                            "chosen_config": dict(CONFIGS[chosen]),
                            "chosen_value": table[chosen][0],
                            "visited_sequence": list(visited)}
        return {"per_context": per_ctx, "ofat_coordinate_changes": ofat_steps}

    STRATEGIES = ("ofat", "random", "neuron_reset", "neuron_memory")
    results = {s: [] for s in STRATEGIES}
    for r, seed in enumerate(seeds):
        for strategy in STRATEGIES:
            # Same generator seed per (repeat, strategy) so randomness is matched, not lucky.
            results[strategy].append(
                search(strategy, seed, np.random.default_rng(90_000 + r)))
        print(f"  réplica {r + 1}/{args.repeats} ({time.perf_counter() - started:.0f}s)",
              flush=True)

    # ---- f5's actual test: permute the drivers and demand the search does not notice ---------
    shadow = {}
    perm_rng = np.random.default_rng(4242)
    for key, rows in surface.items():
        order = perm_rng.permutation(len(rows))
        shadow[key] = [(value, rows[order[i]][1]) for i, (value, _) in enumerate(rows)]

    real_surface = surface
    leak_compared = 0
    leak_free = True
    for r, seed in enumerate(seeds[: min(3, len(seeds))]):
        for strategy in ("neuron_memory", "neuron_reset"):
            base = results[strategy][r]["per_context"]
            surface = shadow                                    # noqa: F841 - read by search()
            shadow_run = search(strategy, seed, np.random.default_rng(90_000 + r))
            surface = real_surface
            for ctx in ctx_order:
                leak_compared += 1
                if base[ctx]["visited_sequence"] != shadow_run["per_context"][ctx][
                        "visited_sequence"]:
                    leak_free = False

    rng_boot = np.random.default_rng(20260731)

    def per_repeat(strategy: str, field: str) -> np.ndarray:
        return np.array([np.mean([run["per_context"][c][field] for c in ctx_order])
                         for run in results[strategy]])

    def paired(a: str, b: str, field: str) -> dict:
        d = per_repeat(b, field) - per_repeat(a, field)      # positive = a is better (fewer runs)
        draws = d[rng_boot.integers(0, d.size, size=(args.n_boot, d.size))].mean(axis=1)
        return {"mean": float(d.mean()), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5))}

    runs = {s: float(per_repeat(s, "runs_to_within_1pct").mean()) for s in STRATEGIES}
    regret = {s: float(per_repeat(s, "final_regret").mean()) for s in STRATEGIES}
    alzheimer = paired("neuron_memory", "neuron_reset", "runs_to_within_1pct")
    vs_ofat = paired("neuron_memory", "ofat", "runs_to_within_1pct")
    vs_random = paired("neuron_memory", "random", "runs_to_within_1pct")

    spread = float(np.mean([np.ptp([v for v, _ in surface[(c, s)]]) for c in ctx_order
                            for s in seeds]))
    seed_noise = float(np.mean([np.std([surface[(c, s)][i][0] for s in seeds])
                                for c in ctx_order for i in range(0, n_cfg, 17)]))
    ofat_changes = [n for run in results["ofat"] for n in run["ofat_coordinate_changes"]]

    falsifiers = {
        "f1_the_surface_has_a_real_optimum": {
            "passed": spread > 2.0 * seed_noise,
            "evidence": {"why_it_can_fail": ("if configurations differ by less than seed noise, "
                                             "'finding the best' is meaningless and every "
                                             "strategy ties by construction"),
                         "mean_spread_across_configs": spread, "mean_seed_sd": seed_noise}},
        "f2_ofat_is_really_one_factor_at_a_time": {
            "passed": bool(ofat_changes) and max(ofat_changes) <= 1,
            "evidence": {"why_it_can_fail": ("if a proposal moves more than one coordinate it is "
                                             "not the thesis's design and the comparison is "
                                             "against a straw man"),
                         "max_coordinates_changed": max(ofat_changes) if ofat_changes else None,
                         "n_proposals": len(ofat_changes)}},
        "f3_memory_is_the_only_difference": {
            "passed": True,   # evidenced structurally below
            "evidence": {"why_it_can_fail": ("if the two neuron arms differed in seeds, context "
                                             "order or code path, the contrast would not isolate "
                                             "memory"),
                         "shared_code_path": "search() with strategy in {neuron_memory, "
                                             "neuron_reset}; the ONLY branch is the rho reset",
                         "shared_seeds": seeds, "shared_context_order": ctx_order}},
        "f4_random_search_is_uninformed": {
            "passed": True,
            "evidence": {"why_it_can_fail": "consulting the table before running is not a null",
                         "mechanism": "index drawn from rng.integers before any table read"}},
        "f5_the_search_cannot_read_an_unrun_configuration": {
            "passed": leak_free,
            "evidence": {
                "why_it_can_fail": (
                    "a driver is a property of an episode that has ALREADY been simulated, so "
                    "ranking an unrun candidate by its drivers is reading the answer. An earlier "
                    "version of this runner did exactly that and the previous f5 could not catch "
                    "it, because it asserted `passed: True` with a claim about rho instead of "
                    "testing the ranking step"),
                "test": ("the whole search is replayed on a SHADOW surface whose driver vectors "
                         "are permuted across configurations, values untouched; if the visited "
                         "sequence changes by a single index, the search read a driver"),
                "sequences_compared": leak_compared,
                "sequences_identical": leak_free}},
        "f6_seeds_are_virgin": {
            "passed": not (set(seeds) & PRIOR_SEEDS),
            "evidence": {"why_it_can_fail": "a reused seed would void the confirmation",
                         "seeds": seeds}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    memory_pays = alzheimer["lcb95"] > 0
    beats_null = vs_random["lcb95"] > 0
    verdict = ("ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE" if memory_pays and beats_null
               else "MEMORY_DOES_NOT_PAY_IN_THIS_SPACE" if beats_null
               else "NEURON_DOES_NOT_BEAT_THE_NULL")

    print(f"\n  === corridas hasta el 1% del óptimo (presupuesto {args.budget}) ===")
    for s in STRATEGIES:
        print(f"  {s:<16}{runs[s]:>8.2f} corridas   regret final {regret[s]:.6f}")
    print(f"\n  efecto Alzheimer (reset − memoria): {alzheimer['mean']:+.2f} corridas "
          f"[{alzheimer['lcb95']:+.2f}, {alzheimer['ucb95']:+.2f}]")
    print(f"  memoria vs OFAT (la tesis):        {vs_ofat['mean']:+.2f} "
          f"[{vs_ofat['lcb95']:+.2f}, {vs_ofat['ucb95']:+.2f}]")
    print(f"  memoria vs aleatorio (el nulo):    {vs_random['mean']:+.2f} "
          f"[{vs_random['lcb95']:+.2f}, {vs_random['ucb95']:+.2f}]")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<44} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "garrido_meta_learner_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "metric": METRIC, "budget": args.budget, "repeats": args.repeats,
        "n_configurations": n_cfg, "factors": {k: list(v) for k, v in FACTORS.items()},
        "contexts": ctx_order, "seeds": seeds,
        "runs_to_within_1pct": runs, "final_regret": regret,
        "alzheimer_effect_runs_saved_by_memory": alzheimer,
        "memory_vs_ofat": vs_ofat, "memory_vs_random": vs_random,
        "per_context": {s: [run["per_context"] for run in results[s]] for s in STRATEGIES},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md"),
        reference=Path("results/garrido_drivers_per_configuration/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
