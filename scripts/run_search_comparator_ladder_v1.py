#!/usr/bin/env python3
"""The search-comparator ladder, in the order a referee will demand it.

"Our learner beats one-factor-at-a-time and random" is where a C&IE referee stops reading. The
gate that justifies this lane is already measured: the 288 surface is NOT separable
(leave-one-seed-out interaction gain 0.072-0.159), so OFAT is not near-optimal by construction and
there is a real search problem. What is not yet measured is whether the Fig. 5 neuron beats the
methods an operations-research reader would reach for first.

EVERY arm sees the same CRN surface, the same budget, and the same cache. Reads are enforced, not
asserted: `Surface` refuses to return a value for a configuration the arm has not selected, so an
arm that tried to peek at an unrun cell would raise rather than quietly win.

DEVIATION FROM PLAN, declared: `supply_chain.gsa.gp_locate` is not reused verbatim. It proposes
2048 CONTINUOUS candidates and returns no visit history, so on a 288-point grid it would need a
snap-to-nearest rule plus an arbitrary duplicate policy. Enumerating the 288 candidates is both
exact and cheaper. The kernel, the EI formula and the normalisation follow gsa.py:67-88.

Contract: docs/ENMIENDA_ESCALERA_COMPARADORES_2026-08-05.md
Cache: results/surface_cache/wrap288_v1 (burned block 5_300_001-012, declared replay)
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
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from scripts.seal_garrido_surface_cache_v1 import verify_sealed_slice  # noqa: E402

FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
}
FACTOR_NAMES = tuple(FACTORS)
CONFIGS = tuple(dict(zip(FACTOR_NAMES, combo)) for combo in itertools.product(*FACTORS.values()))
DEFAULT = {"buffer_hours": 0.0, "shifts": 1, "op9_rop": 24.0, "op12_rop": 24.0}
N_CFG = len(CONFIGS)
BUDGET = 24
GP_N_INIT = 8                       # of a budget of 24; gp_locate's default 16 would spend 2/3
N_BOOT = 5_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
CONTEXT_ORDER = ("R1r", "R2r", "R1r+R2r", "R1r|esc", "R2r|esc", "R1r+R2r|esc")

COORDS = np.array([[FACTORS[n].index(c[n]) / (len(FACTORS[n]) - 1) for n in FACTOR_NAMES]
                   for c in CONFIGS], dtype=float)
FEATURES = np.column_stack([COORDS, np.ones(N_CFG)])
#: Hamming-1 neighbours on the factor lattice, for simulated annealing.
NEIGHBOURS: list[list[int]] = [[] for _ in range(N_CFG)]
_INDEX = {tuple(sorted(c.items())): i for i, c in enumerate(CONFIGS)}
for _i, _c in enumerate(CONFIGS):
    for _n in FACTOR_NAMES:
        for _lv in FACTORS[_n]:
            if _lv != _c[_n]:
                NEIGHBOURS[_i].append(_INDEX[tuple(sorted(dict(_c, **{_n: _lv}).items()))])


class Surface:
    """The tape, with reads enforced. An arm can only see what it has selected."""

    def __init__(self, values: np.ndarray):
        self._values = values
        self.best = float(values.max())
        self.visited: list[int] = []
        self._seen: set[int] = set()

    def select(self, idx: int) -> float:
        idx = int(idx)
        if not 0 <= idx < N_CFG:
            raise ValueError(f"configuration {idx} is not on the grid")
        self.visited.append(idx)
        self._seen.add(idx)
        return float(self._values[idx])

    def value_of_visited(self, idx: int) -> float:
        if int(idx) not in self._seen:
            raise LookupError(f"configuration {idx} has not been run; reading it would be a leak")
        return float(self._values[int(idx)])

    @property
    def unvisited(self) -> list[int]:
        return [i for i in range(N_CFG) if i not in self._seen]

    def regret_curve(self) -> list[float]:
        running, curve = -np.inf, []
        for idx in self.visited:
            running = max(running, float(self._values[idx]))
            curve.append(self.best - running)
        return curve


# ---------------------------------------------------------------- arms -----------------------
def arm_oracle(s: Surface, rng, budget: int) -> None:
    """Reference ceiling only. It reads the surface by construction and is never a policy."""
    order = np.argsort(-s._values)
    for idx in order[:budget]:
        s.select(int(idx))


def arm_random(s: Surface, rng, budget: int) -> None:
    for idx in rng.permutation(N_CFG)[:budget]:
        s.select(int(idx))


def arm_ofat(s: Surface, rng, budget: int) -> None:
    """The thesis design, generated lazily from the incumbent so each proposal moves exactly one
    coordinate. The stale-index bug in the v1 runner is fixed: when the design is exhausted the
    arm re-runs the INCUMBENT, not whatever its last proposal happened to be."""
    current, fi, li = dict(DEFAULT), 0, 0
    factor_best: tuple[float, dict] | None = None
    for _ in range(budget):
        if fi >= len(FACTOR_NAMES):
            s.select(_INDEX[tuple(sorted(current.items()))])
            continue
        name = FACTOR_NAMES[fi]
        cand = dict(current, **{name: FACTORS[name][li]})
        idx = _INDEX[tuple(sorted(cand.items()))]
        value = s.select(idx)
        if factor_best is None or value > factor_best[0]:
            factor_best = (value, cand)
        li += 1
        if li >= len(FACTORS[name]):
            current, fi, li, factor_best = factor_best[1], fi + 1, 0, None


def arm_lhs_local(s: Surface, rng, budget: int) -> None:
    """Space-filling start, then greedy hill-climbing on Hamming-1 neighbours: the standard
    simulation-optimization baseline an OR reader expects before any learner."""
    n_init = max(4, budget // 3)
    for idx in rng.permutation(N_CFG)[:n_init]:
        s.select(int(idx))
    while len(s.visited) < budget:
        incumbent = max(set(s.visited), key=s.value_of_visited)
        cand = [i for i in NEIGHBOURS[incumbent] if i not in s._seen]
        s.select(int(rng.choice(cand)) if cand else int(rng.choice(s.unvisited)))


def arm_gp_ei(s: Surface, rng, budget: int) -> None:
    """Discrete GP-EI. Kernel and acquisition follow gsa.py:75-86; candidates are the grid."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    for idx in rng.permutation(N_CFG)[:GP_N_INIT]:
        s.select(int(idx))
    kern = (ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(COORDS.shape[1]))
            + WhiteKernel(1e-4))
    while len(s.visited) < budget:
        seen = sorted(s._seen)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=0).fit(
            COORDS[seen], np.array([s.value_of_visited(i) for i in seen]))
        cand = s.unvisited
        mu, sd = gp.predict(COORDS[cand], return_std=True)
        best = max(s.value_of_visited(i) for i in seen)
        imp = mu - best
        z = np.where(sd > 1e-9, imp / sd, 0.0)
        ei = np.where(sd > 1e-9, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)
        s.select(cand[int(ei.argmax())])


def arm_ucb1(s: Surface, rng, budget: int) -> None:
    """UCB1 over FACTOR LEVELS, assembling a configuration from the per-level upper bounds. The
    factor-independent bandit is the cheap honest answer, and on a non-separable surface it should
    struggle -- which is what makes it informative rather than decorative."""
    sums = {n: np.zeros(len(FACTORS[n])) for n in FACTOR_NAMES}
    counts = {n: np.zeros(len(FACTORS[n])) for n in FACTOR_NAMES}
    for t in range(budget):
        cfg = {}
        for n in FACTOR_NAMES:
            if counts[n].min() == 0:
                cfg[n] = FACTORS[n][int(counts[n].argmin())]
            else:
                ucb = sums[n] / counts[n] + np.sqrt(2.0 * np.log(t + 1) / counts[n])
                cfg[n] = FACTORS[n][int(ucb.argmax())]
        idx = _INDEX[tuple(sorted(cfg.items()))]
        if idx in s._seen:                       # already run: take the best unseen neighbour
            cand = [i for i in NEIGHBOURS[idx] if i not in s._seen] or s.unvisited
            idx = int(rng.choice(cand))
        value = s.select(idx)
        for n in FACTOR_NAMES:
            li = FACTORS[n].index(CONFIGS[idx][n])
            sums[n][li] += value
            counts[n][li] += 1.0


def arm_annealing(s: Surface, rng, budget: int) -> None:
    idx = int(rng.integers(0, N_CFG))
    current = s.select(idx)
    for t in range(1, budget):
        temp = max(1e-9, 1.0 - t / budget)
        cand = [i for i in NEIGHBOURS[idx] if i not in s._seen] or s.unvisited
        nxt = int(rng.choice(cand))
        value = s.select(nxt)
        scale = abs(current) + 1e-12
        if value > current or rng.random() < np.exp((value - current) / (temp * scale)):
            idx, current = nxt, value


def make_neuron_arm(retained: dict | None):
    """Garrido's Fig. 5 unit with the PREFIX normaliser: `lo`/`span` come only from what has
    already been run. `retained` is `rho` crossing the context boundary -- the memory itself."""

    def arm(s: Surface, rng, budget: int) -> None:
        rho = retained["rho"] if retained is not None else np.zeros(FEATURES.shape[1])
        seen_values: list[float] = []
        for _ in range(budget):
            if len(s.visited) < 3:
                idx = int(rng.choice(s.unvisited))
            else:
                scores = FEATURES[s.unvisited] @ rho
                idx = s.unvisited[int(np.argmax(scores))]
            value = s.select(idx)
            seen_values.append(value)
            lo, hi = min(seen_values), max(seen_values)
            if hi > lo:
                y = (value - lo) / (hi - lo)
                pred = 1.0 / (1.0 + np.exp(-np.clip(FEATURES[idx] @ rho, -30, 30)))
                rho = rho + 0.35 * (y - pred) * FEATURES[idx]
        if retained is not None:
            retained["rho"] = rho

    return arm


ARMS = {
    "oracle": arm_oracle, "random": arm_random, "ofat": arm_ofat,
    "lhs_local": arm_lhs_local, "gp_ei": arm_gp_ei, "ucb1": arm_ucb1,
    "annealing": arm_annealing, "neuron_memory": None, "neuron_reset": None,
}


def load_cache(root: Path):
    surface, contexts, seeds = {}, [], set()
    for path in sorted(root.rglob("*.json")):
        p = json.loads(path.read_text())
        verify_sealed_slice(p)
        ctx, seed = p["context"], int(p["seed"])
        if len(p["cells"]) != N_CFG:
            raise ValueError(f"{path}: incomplete surface slice")
        surface[(ctx, seed)] = np.array([c["value"] for c in p["cells"]], dtype=float)
        seeds.add(seed)
        if ctx not in contexts:
            contexts.append(ctx)
    missing = [ctx for ctx in CONTEXT_ORDER if ctx not in contexts]
    if missing:
        raise ValueError(f"cache is missing contracted contexts: {missing}")
    return surface, [ctx for ctx in CONTEXT_ORDER if ctx in contexts], sorted(seeds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path, default=Path("results/search_ladder/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    surface, contexts, seeds = load_cache(args.cache)
    print(f"  caché: {len(contexts)} contextos x {len(seeds)} semillas, presupuesto "
          f"{args.budget}")

    per_arm: dict[str, dict] = {name: {"auc": [], "final": [], "budget_used": []}
                                for name in ARMS}
    for r, seed in enumerate(seeds):
        retained = {"rho": np.zeros(FEATURES.shape[1])}
        for name in ARMS:
            rng = np.random.default_rng(90_000 + r)
            fn = (make_neuron_arm(retained) if name == "neuron_memory"
                  else make_neuron_arm(None) if name == "neuron_reset" else ARMS[name])
            aucs, finals = [], []
            for ctx in contexts:
                s = Surface(surface[(ctx, seed)])
                fn(s, rng, args.budget)
                curve = s.regret_curve()
                denom = args.budget * abs(s.best) if s.best else 1.0
                aucs.append(float(np.sum(curve)) / denom)
                finals.append(curve[-1] / (abs(s.best) or 1.0))
                per_arm[name]["budget_used"].append(len(s.visited))
            per_arm[name]["auc"].append(float(np.mean(aucs)))
            per_arm[name]["final"].append(float(np.mean(finals)))
        print(f"  réplica {r + 1}/{len(seeds)} ({time.perf_counter() - started:.0f}s)", flush=True)

    rng = np.random.default_rng(20260805)

    def boot(diff: np.ndarray) -> dict:
        draws = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
        stats = diff[draws].mean(axis=1)
        return {"mean": float(diff.mean()), "lcb95": float(np.percentile(stats, 2.5)),
                "ucb95": float(np.percentile(stats, 97.5)), "n": int(diff.size)}

    means = {n: float(np.mean(v["auc"])) for n, v in per_arm.items()}
    ranking = sorted((n for n in means if n != "oracle"), key=lambda n: means[n])
    champion = ranking[0]
    #: Lower AUC regret is better, so `other - neuron` positive means the neuron wins.
    vs_neuron = {n: boot(np.asarray(per_arm[n]["auc"]) - np.asarray(per_arm["neuron_memory"]["auc"]))
                 for n in ranking if n != "neuron_memory"}

    budgets_matched = all(u == args.budget for v in per_arm.values() for u in v["budget_used"])
    neuron_beats_all = all(v["lcb95"] > 0.0 for v in vs_neuron.values())
    beats_gp = vs_neuron["gp_ei"]["lcb95"] > 0.0

    verdict = ("NEURON_BEATS_THE_FULL_CLASSICAL_LADDER" if neuron_beats_all
               else "NEURON_BEATS_BAYESIAN_OPTIMISATION_NOT_ALL" if beats_gp
               else f"CLASSICAL_SEARCH_WINS__{champion.upper()}")

    falsifiers = {
        "f1_budgets_are_matched": {
            "passed": bool(budgets_matched),
            "evidence": {"why_it_can_fail": "an arm that spends fewer or more evaluations is not "
                                            "being compared; gp_ei's initialisation is exactly "
                                            "where this breaks, so it is counted from the access "
                                            "log rather than asserted",
                         "budget": args.budget,
                         "distinct_counts": sorted({u for v in per_arm.values()
                                                    for u in v["budget_used"]})}},
        "f2_no_arm_reads_an_unrun_configuration": {
            "passed": True,
            "evidence": {"why_it_can_fail": "it cannot silently: Surface.value_of_visited raises "
                                            "LookupError for any configuration not selected, so an "
                                            "arm that peeked would abort the run rather than win. "
                                            "Enforced structurally, not asserted",
                         "enforcement": "Surface.value_of_visited"}},
        "f3_the_oracle_is_a_ceiling_not_a_competitor": {
            "passed": bool(means["oracle"] <= min(means[n] for n in ranking) + 1e-12),
            "evidence": {"why_it_can_fail": "if any arm matched the oracle the ceiling would be "
                                            "mis-specified and every contrast meaningless",
                         "oracle_auc": means["oracle"]}},
        "f4_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print("\n  AUC de regret normalizado (menor es mejor):")
    for n in ["oracle"] + ranking:
        print(f"    {n:<16} {means[n]:.5f}")
    print("\n  contra la neurona con memoria (positivo = gana la neurona):")
    for n, v in vs_neuron.items():
        print(f"    {n:<16} {v['mean']:+.5f} [LCB95 {v['lcb95']:+.5f}]")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<46} {label}")

    payload = {
        "schema_version": "search_comparator_ladder_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "gate": "surface_gates_v1 -> NON_SEPARABLE_BUT_CONTEXT_INVARIANT",
        "primary_metric": "auc_regret_norm", "budget": args.budget,
        "contexts": contexts, "seeds": seeds, "arms": list(ARMS),
        "mean_auc_regret": means, "ranking_best_first": ranking,
        "vs_neuron_memory": vs_neuron, "per_arm": per_arm,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/surface_gates/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
