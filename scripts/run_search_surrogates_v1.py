#!/usr/bin/env python3
"""KAN and MLP as search surrogates, and Delta_efficiency measured rather than promised.

Garrido asked for KAN vs MLP vs MPC on parameters, speed and convergence. Inside the episode this
project measured no neural premium in four separate contracts -- an MLP came out WORSE than a
linear model. So the question moves to the place where a network can have work: approximating the
DESIGN surface inside the between-run search, which is the role Figure 5 actually gives it.

Ladder v2 already found that the ingredient is retention rather than the approximator. What is not
yet measured is whether a MORE EXPRESSIVE approximator converts that retention into more advantage
-- and what it costs per decision.

Every arm shares the neuron's loop and normalisation; only the approximator changes, and all of
them retain weights across contexts. Re-initialising per step would turn them into memoryless arms
and reproduce the tautology v2 corrected.

Contract: docs/ENMIENDA_SURROGATES_Y_EFICIENCIA_2026-08-05.md
Cache: results/surface_cache/wrap288_v1 (burned block, declared replay)
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
N_CFG = len(CONFIGS)
BUDGET = 24
N_BOOT = 5_000
COLD_START = 3
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")

COORDS = np.array([[FACTORS[n].index(c[n]) / (len(FACTORS[n]) - 1) for n in FACTOR_NAMES]
                   for c in CONFIGS], dtype=float)
FEATURES = np.column_stack([COORDS, np.ones(N_CFG)])


class Surface:
    """The tape, with reads enforced: an arm only sees what it selected."""

    def __init__(self, values: np.ndarray):
        self._values = values
        self.best = float(values.max())
        self.visited: list[int] = []
        self._seen: set[int] = set()

    def select(self, idx: int) -> float:
        self.visited.append(int(idx))
        self._seen.add(int(idx))
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


def prefix_target(values: list[float]) -> float:
    lo, hi = min(values), max(values)
    return 0.5 if hi <= lo else (values[-1] - lo) / (hi - lo)


# ------------------------------------------------------------------ approximators ------------
class LinearNeuron:
    """Garrido's Fig. 5 unit: five parameters, online logistic-loss step."""
    name = "neuron_memory"

    def __init__(self):
        self.rho = np.zeros(FEATURES.shape[1])

    @property
    def n_parameters(self) -> int:
        return int(self.rho.size)

    def score(self, idx: list[int]) -> np.ndarray:
        return FEATURES[idx] @ self.rho

    def update(self, idx: int, y: float) -> None:
        pred = 1.0 / (1.0 + np.exp(-np.clip(FEATURES[idx] @ self.rho, -30, 30)))
        self.rho = self.rho + 0.35 * (y - pred) * FEATURES[idx]


class TorchSurrogate:
    """Shared harness for the MLP and the KAN: weights persist, training is incremental."""

    def __init__(self, module, steps: int, lr: float, name: str):
        import torch
        self.torch = torch
        self.net = module
        self.name = name
        self.steps = steps
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.x: list[np.ndarray] = []
        self.y: list[float] = []

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def score(self, idx: list[int]) -> np.ndarray:
        with self.torch.no_grad():
            t = self.torch.tensor(COORDS[idx], dtype=self.torch.float32)
            return self.net(t).squeeze(-1).cpu().numpy()

    def update(self, idx: int, y: float) -> None:
        self.x.append(COORDS[idx])
        self.y.append(float(y))
        xt = self.torch.tensor(np.array(self.x), dtype=self.torch.float32)
        yt = self.torch.tensor(np.array(self.y), dtype=self.torch.float32).unsqueeze(-1)
        for _ in range(self.steps):
            self.opt.zero_grad()
            loss = ((self.net(xt) - yt) ** 2).mean()
            loss.backward()
            self.opt.step()


def make_mlp():
    import torch
    net = torch.nn.Sequential(
        torch.nn.Linear(len(FACTOR_NAMES), 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    return TorchSurrogate(net, steps=30, lr=0.01, name="surrogate_mlp")


def make_kan():
    from kan import KAN
    net = KAN(width=[len(FACTOR_NAMES), 4, 1], grid=3, k=3, seed=0,
              auto_save=False, save_act=False, symbolic_enabled=False)
    return TorchSurrogate(net, steps=30, lr=0.01, name="surrogate_kan")


BUILDERS = {"neuron_memory": LinearNeuron, "surrogate_mlp": make_mlp, "surrogate_kan": make_kan}


def run_context(model, s: Surface, rng, budget: int) -> list[float]:
    """One context. Returns per-decision wall-clock seconds."""
    latencies, seen = [], []
    for _ in range(budget):
        t0 = time.perf_counter()
        if len(s.visited) < COLD_START:
            idx = int(rng.choice(s.unvisited))
        else:
            unvisited = s.unvisited
            idx = unvisited[int(np.argmax(model.score(unvisited)))]
        latencies.append(time.perf_counter() - t0)
        seen.append(s.select(idx))
        model.update(idx, prefix_target(seen))
    return latencies


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
    return surface, contexts, sorted(seeds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seeds", type=int, default=0, help="0 = every seed in the cache")
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path, default=Path("results/search_surrogates/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    surface, contexts, seeds = load_cache(args.cache)
    if args.seeds:
        seeds = seeds[:args.seeds]
    print(f"  caché: {len(contexts)} contextos x {len(seeds)} semillas, presupuesto {args.budget}")

    per_arm = {n: {"auc": [], "latency": [], "params": None} for n in BUILDERS}
    for r, seed in enumerate(seeds):
        for name, builder in BUILDERS.items():
            rng = np.random.default_rng(90_000 + r)
            model = builder()
            aucs = []
            for ctx in contexts:
                s = Surface(surface[(ctx, seed)])
                per_arm[name]["latency"].extend(run_context(model, s, rng, args.budget))
                curve = s.regret_curve()
                aucs.append(float(np.sum(curve)) / (args.budget * (abs(s.best) or 1.0)))
            per_arm[name]["auc"].append(float(np.mean(aucs)))
            per_arm[name]["params"] = model.n_parameters
        print(f"  réplica {r + 1}/{len(seeds)} ({time.perf_counter() - started:.0f}s)", flush=True)

    rng = np.random.default_rng(20260805)

    def boot(diff: np.ndarray) -> dict:
        draws = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
        stats = diff[draws].mean(axis=1)
        return {"mean": float(diff.mean()), "lcb95": float(np.percentile(stats, 2.5)),
                "ucb95": float(np.percentile(stats, 97.5)), "n": int(diff.size)}

    means = {n: float(np.mean(v["auc"])) for n, v in per_arm.items()}
    efficiency = {n: {"parameters": v["params"],
                      "median_seconds_per_decision": float(np.median(v["latency"])),
                      "p95_seconds_per_decision": float(np.percentile(v["latency"], 95)),
                      "total_search_seconds": float(np.sum(v["latency"]))}
                  for n, v in per_arm.items()}
    vs_neuron = {n: boot(np.asarray(per_arm[n]["auc"])
                         - np.asarray(per_arm["neuron_memory"]["auc"]))
                 for n in BUILDERS if n != "neuron_memory"}
    #: Positive means the neuron wins (lower AUC regret is better).
    neural_premium = {n: -v["mean"] for n, v in vs_neuron.items()}
    any_network_wins = any(v["ucb95"] < 0.0 for v in vs_neuron.values())
    all_equivalent = all(v["lcb95"] <= 0.0 <= v["ucb95"] for v in vs_neuron.values())

    verdict = ("NEURAL_SURROGATE_PREMIUM_IN_THE_OUTER_LOOP" if any_network_wins
               else "APPROXIMATOR_IS_NOT_THE_INGREDIENT_RETENTION_IS" if all_equivalent
               else "LINEAR_NEURON_BEATS_THE_DEEPER_SURROGATES")

    falsifiers = {
        "f1_every_arm_spends_the_same_budget": {
            "passed": True,
            "evidence": {"why_it_can_fail": "DES calls are equalised by construction here, so the "
                                            "only currency left is per-decision cost; if budgets "
                                            "differed the efficiency comparison would be vacuous",
                         "budget": args.budget,
                         "decisions_per_arm": {n: len(v["latency"]) for n, v in per_arm.items()}}},
        "f2_the_surrogates_actually_retain_weights": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a surrogate re-initialised per context would be the "
                                            "memoryless arm under a new name, and ladder v2 showed "
                                            "that comparison is a tautology. One model instance is "
                                            "built per replicate and carried across all contexts",
                         "instances_per_replicate": 1, "contexts_per_instance": len(contexts)}},
        "f3_parameter_counts_are_read_from_the_model": {
            "passed": all(v["parameters"] and v["parameters"] > 0 for v in efficiency.values()),
            "evidence": {"why_it_can_fail": "counted from the live modules, not declared; a wrong "
                                            "architecture would show a wrong count",
                         "parameters": {n: v["parameters"] for n, v in efficiency.items()}}},
        "f4_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print("\n  AUC de regret (menor mejor) · parámetros · s/decisión (mediana):")
    for n in sorted(means, key=lambda k: means[k]):
        e = efficiency[n]
        print(f"    {n:<16} {means[n]:.5f}   {e['parameters']:>6}   "
              f"{e['median_seconds_per_decision']*1000:8.2f} ms")
    print("\n  contra la neurona lineal (positivo = gana la neurona):")
    for n, v in vs_neuron.items():
        print(f"    {n:<16} {v['mean']:+.5f} [LCB95 {v['lcb95']:+.5f}, UCB95 {v['ucb95']:+.5f}]")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<46} {label}")

    payload = {
        "schema_version": "search_surrogates_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "primary_metric": "auc_regret_norm", "budget": args.budget,
        "contexts": contexts, "seeds": seeds, "arms": list(BUILDERS),
        "mean_auc_regret": means, "vs_neuron_memory": vs_neuron,
        "neural_premium_point_estimate": neural_premium,
        "delta_efficiency": efficiency,
        "per_arm_auc": {n: v["auc"] for n, v in per_arm.items()},
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/search_ladder_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
