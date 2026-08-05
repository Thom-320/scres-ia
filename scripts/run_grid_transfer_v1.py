#!/usr/bin/env python3
"""Grid transfer 288 -> 4,608: does retained state survive a change of design space?

The surface gates closed cross-context transfer -- H_regime +0.0038 against a 0.05 bar, with the
optimum common to all six contexts. Grid transfer is what is left, and it is the axis that
separates a method that learned the SHAPE of the surface from one that memorised POINTS.

Train each memory-carrying method on its six-context career over the 288 grid, then let the SAME
retained state search the 4,608 grid. The control is the same method starting cold. The
coordinate space goes from four factors to six, and that is the difficulty: the 288 grid IS the
subgrid op3_rm = op5_rm = 0, so every training observation lives in the six-dimensional space with
its last two coordinates at zero and no arm is handed information it did not earn.

BO IS NOT ASSUMED TO FAIL. A warm-started GP is a full arm here. "A GP prior cannot cross a change
of design space" is our hypothesis, and hypotheses get measured.

Contract: docs/ENMIENDA_TRANSFERENCIA_REJILLA_2026-08-05.md
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

BASE_FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
}
RAW_LEVELS = (0.0, 17_500.0, 70_000.0, 140_000.0)
EXT_FACTORS = dict(BASE_FACTORS, op3_rm=RAW_LEVELS, op5_rm=RAW_LEVELS)
EXT_NAMES = tuple(EXT_FACTORS)
BASE_CONFIGS = tuple(dict(zip(BASE_FACTORS, c))
                     for c in itertools.product(*BASE_FACTORS.values()))
EXT_CONFIGS = tuple(dict(zip(EXT_NAMES, c)) for c in itertools.product(*EXT_FACTORS.values()))
BUDGET, N_BOOT, COLD_START = 24, 5_000, 3
GP_N_INIT = 8
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def coords(configs, factors) -> np.ndarray:
    """Every configuration in the SIX-dimensional space. The 288 grid simply has its last two
    coordinates at zero, which is what makes the two spaces commensurable without invention."""
    rows = []
    for cfg in configs:
        rows.append([factors[n].index(cfg[n]) / (len(factors[n]) - 1) if n in cfg else 0.0
                     for n in EXT_NAMES])
    return np.asarray(rows, dtype=float)


BASE_COORDS = coords(BASE_CONFIGS, BASE_FACTORS)
EXT_COORDS = coords(EXT_CONFIGS, EXT_FACTORS)
BASE_FEAT = np.column_stack([BASE_COORDS, np.ones(len(BASE_CONFIGS))])
EXT_FEAT = np.column_stack([EXT_COORDS, np.ones(len(EXT_CONFIGS))])
EXT_INDEX = {tuple(sorted(c.items())): i for i, c in enumerate(EXT_CONFIGS)}


class Surface:
    def __init__(self, values: np.ndarray):
        self._v = values
        self.best = float(values.max())
        self.visited: list[int] = []
        self._seen: set[int] = set()

    def select(self, i: int) -> float:
        self.visited.append(int(i))
        self._seen.add(int(i))
        return float(self._v[int(i)])

    def value_of_visited(self, i: int) -> float:
        if int(i) not in self._seen:
            raise LookupError(f"{i} was never run; reading it would be a leak")
        return float(self._v[int(i)])

    @property
    def unvisited(self) -> list[int]:
        return [i for i in range(self._v.size) if i not in self._seen]

    def auc(self, budget: int) -> float:
        run, curve = -np.inf, []
        for i in self.visited:
            run = max(run, float(self._v[i]))
            curve.append(self.best - run)
        return float(np.sum(curve)) / (budget * (abs(self.best) or 1.0))


def prefix_target(seen: list[float]) -> float:
    lo, hi = min(seen), max(seen)
    return 0.5 if hi <= lo else (seen[-1] - lo) / (hi - lo)


# ------------------------------------------------------------------- arms ---------------------
def neuron_arm(state, feat):
    def run(s: Surface, rng, budget):
        seen = []
        for _ in range(budget):
            if len(s.visited) < COLD_START:
                idx = int(rng.choice(s.unvisited))
            else:
                u = s.unvisited
                idx = u[int(np.argmax(feat[u] @ state["rho"]))]
            seen.append(s.select(idx))
            y = prefix_target(seen)
            pred = 1.0 / (1.0 + np.exp(-np.clip(feat[idx] @ state["rho"], -30, 30)))
            state["rho"] = state["rho"] + 0.35 * (y - pred) * feat[idx]
    return run


def ucb1_arm(state, configs, factors):
    def run(s: Surface, rng, budget):
        sums, counts = state["sums"], state["counts"]
        seen_idx, seen_raw = [], []
        for t in range(budget):
            cfg = {}
            for n in factors:
                if counts[n].min() == 0:
                    cfg[n] = factors[n][int(counts[n].argmin())]
                else:
                    ucb = sums[n] / counts[n] + np.sqrt(2.0 * np.log(t + 1) / counts[n])
                    cfg[n] = factors[n][int(ucb.argmax())]
            idx = EXT_INDEX[tuple(sorted(cfg.items()))] if len(factors) == len(EXT_NAMES) \
                else configs.index(cfg)
            if idx in s._seen:
                idx = int(rng.choice(s.unvisited))
            seen_idx.append(idx)
            seen_raw.append(s.select(idx))
        lo, hi = min(seen_raw), max(seen_raw)
        for idx, raw in zip(seen_idx, seen_raw):
            y = 0.5 if hi <= lo else (raw - lo) / (hi - lo)
            for n in factors:
                li = factors[n].index(configs[idx][n])
                sums[n][li] += y
                counts[n][li] += 1.0
    return run


def ofat_arm(state, configs, factors):
    def run(s: Surface, rng, budget):
        current = dict(state["incumbent"])
        for n in factors:                          # a transferred incumbent may lack new factors
            current.setdefault(n, factors[n][0])
        current = {n: current[n] for n in factors}
        fi, li, best = 0, 0, None
        names = tuple(factors)
        for _ in range(budget):
            if fi >= len(names):
                s.select(_index_of(current, configs))
                continue
            cand = dict(current, **{names[fi]: factors[names[fi]][li]})
            idx = _index_of(cand, configs)
            value = s.select(idx)
            if best is None or value > best[0]:
                best = (value, cand)
            li += 1
            if li >= len(factors[names[fi]]):
                current, fi, li, best = best[1], fi + 1, 0, None
        state["incumbent"] = dict(configs[max(set(s.visited), key=s.value_of_visited)])
    return run


def _index_of(cfg, configs):
    key = tuple(sorted(cfg.items()))
    if key in EXT_INDEX and len(configs) == len(EXT_CONFIGS):
        return EXT_INDEX[key]
    return configs.index(cfg)


def gp_arm(state, coords_all):
    def run(s: Surface, rng, budget):
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        here_i, here_raw = [], []
        for idx in rng.permutation(len(coords_all))[:GP_N_INIT]:
            here_i.append(int(idx))
            here_raw.append(s.select(int(idx)))
        kern = (ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(coords_all.shape[1]))
                + WhiteKernel(1e-4))
        while len(s.visited) < budget:
            lo, hi = min(here_raw), max(here_raw)
            here_y = [0.5 if hi <= lo else (v - lo) / (hi - lo) for v in here_raw]
            x = np.vstack([state["x"], coords_all[here_i]]) if state["x"] is not None \
                else coords_all[here_i]
            y = np.asarray(list(state["y"]) + here_y)
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=0).fit(x, y)
            cand = s.unvisited
            mu, sd = gp.predict(coords_all[cand], return_std=True)
            imp = mu - max(y)
            z = np.where(sd > 1e-9, imp / sd, 0.0)
            ei = np.where(sd > 1e-9, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)
            pick = cand[int(ei.argmax())]
            here_i.append(pick)
            here_raw.append(s.select(pick))
        lo, hi = min(here_raw), max(here_raw)
        here_y = [0.5 if hi <= lo else (v - lo) / (hi - lo) for v in here_raw]
        new_x = coords_all[here_i]
        state["x"] = new_x if state["x"] is None else np.vstack([state["x"], new_x])
        state["y"] = list(state["y"]) + here_y
    return run


def fresh_state(kind: str, factors):
    if kind == "neuron":
        return {"rho": np.zeros(len(EXT_NAMES) + 1)}
    if kind == "ucb1":
        return {"sums": {n: np.zeros(len(factors[n])) for n in factors},
                "counts": {n: np.zeros(len(factors[n])) for n in factors}}
    if kind == "ofat":
        return {"incumbent": {n: factors[n][0] for n in factors}}
    return {"x": None, "y": []}


def extend_state(kind: str, state, factors):
    """Carry the trained state into the six-factor space without inventing anything."""
    if kind == "neuron":
        return {"rho": np.array(state["rho"], copy=True)}       # already 7-dimensional
    if kind == "ucb1":
        sums = {n: np.zeros(len(factors[n])) for n in factors}
        counts = {n: np.zeros(len(factors[n])) for n in factors}
        for n in state["sums"]:
            sums[n][: state["sums"][n].size] = state["sums"][n]
            counts[n][: state["counts"][n].size] = state["counts"][n]
        return {"sums": sums, "counts": counts}                 # new levels keep count 0
    if kind == "ofat":
        inc = dict(state["incumbent"])
        for n in factors:
            inc.setdefault(n, factors[n][0])
        return {"incumbent": inc}
    return {"x": None if state["x"] is None else np.array(state["x"], copy=True),
            "y": list(state["y"])}


ARMS = ("neuron", "ucb1", "ofat", "gp")


def build(kind, state, grid):
    if grid == "base":
        return {"neuron": lambda: neuron_arm(state, BASE_FEAT),
                "ucb1": lambda: ucb1_arm(state, BASE_CONFIGS, BASE_FACTORS),
                "ofat": lambda: ofat_arm(state, BASE_CONFIGS, BASE_FACTORS),
                "gp": lambda: gp_arm(state, BASE_COORDS)}[kind]()
    return {"neuron": lambda: neuron_arm(state, EXT_FEAT),
            "ucb1": lambda: ucb1_arm(state, EXT_CONFIGS, EXT_FACTORS),
            "ofat": lambda: ofat_arm(state, EXT_CONFIGS, EXT_FACTORS),
            "gp": lambda: gp_arm(state, EXT_COORDS)}[kind]()


def load(root: Path, n_expected: int):
    surface, contexts, seeds = {}, [], set()
    for path in sorted(root.rglob("*.json")):
        p = json.loads(path.read_text())
        if len(p["cells"]) != n_expected:
            continue
        surface[(p["context"], int(p["seed"]))] = np.array([c["value"] for c in p["cells"]],
                                                           dtype=float)
        seeds.add(int(p["seed"]))
        if p["context"] not in contexts:
            contexts.append(p["context"])
    return surface, contexts, sorted(seeds)


def marginal_replay(visit_counts: np.ndarray, s: Surface, rng, budget: int) -> None:
    """Reproduce the transferred arm's marginal visit distribution while ignoring the state.

    THE decisive placebo. If the transferred arm does not beat this, what crossed the grid
    boundary was a lookup table over configurations, not the shape of the surface.
    """
    p = visit_counts / visit_counts.sum()
    for _ in range(budget):
        u = s.unvisited
        w = p[u]
        w = w / w.sum() if w.sum() > 0 else None
        s.select(int(rng.choice(u, p=w)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-cache", type=Path,
                    default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--ext-cache", type=Path,
                    default=Path("results/surface_cache/wrap288_compat_extended_v1"))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path, default=Path("results/grid_transfer/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    base, base_ctx, seeds = load(args.base_cache, len(BASE_CONFIGS))
    ext, ext_ctx, ext_seeds = load(args.ext_cache, len(EXT_CONFIGS))
    if not ext:
        raise SystemExit("the extended cache is empty or still building")
    contexts = [c for c in base_ctx if c in ext_ctx]
    seeds = [s for s in seeds if s in ext_seeds]
    print(f"  base {len(base)} rebanadas · extendida {len(ext)} · "
          f"{len(contexts)} contextos x {len(seeds)} semillas")

    # ---- f1: the null subgrid must reproduce the 288 cache bit for bit ----------------------
    null_idx = [i for i, c in enumerate(EXT_CONFIGS)
                if c["op3_rm"] == 0.0 and c["op5_rm"] == 0.0]
    base_order = [EXT_INDEX[tuple(sorted(dict(c, op3_rm=0.0, op5_rm=0.0).items()))]
                  for c in BASE_CONFIGS]
    mismatches, checked, max_abs = 0, 0, 0.0
    for ctx in contexts:
        for seed in seeds:
            d = np.abs(ext[(ctx, seed)][base_order] - base[(ctx, seed)])
            checked += d.size
            max_abs = max(max_abs, float(d.max()))
            mismatches += int((d > 0.0).sum())

    # ---- f2: do the new factors move anything? ----------------------------------------------
    spread = {}
    for ctx in contexts:
        mean = np.mean([ext[(ctx, s)] for s in seeds], axis=0)
        by_base = {}
        for i, c in enumerate(EXT_CONFIGS):
            key = tuple(c[n] for n in BASE_FACTORS)
            by_base.setdefault(key, []).append(mean[i])
        spread[ctx] = float(np.mean([max(v) - min(v) for v in by_base.values()]))

    # ---- the transfer experiment -------------------------------------------------------------
    rows = {f"{a}_{m}": [] for a in ARMS for m in ("transfer", "cold", "marginal")}
    visits = {a: np.ones(len(EXT_CONFIGS)) for a in ARMS}
    for r, seed in enumerate(seeds):
        for kind in ARMS:
            trained = fresh_state(kind, BASE_FACTORS)
            rng = np.random.default_rng(90_000 + r)
            for ctx in contexts:                       # train on the 288 grid
                Surface(base[(ctx, seed)])
                s = Surface(base[(ctx, seed)])
                build(kind, trained, "base")(s, rng, args.budget)

            carried = extend_state(kind, trained, EXT_FACTORS)
            aucs = {"transfer": [], "cold": [], "marginal": []}
            for ctx in contexts:
                s = Surface(ext[(ctx, seed)])
                build(kind, carried, "ext")(s, np.random.default_rng(70_000 + r), args.budget)
                aucs["transfer"].append(s.auc(args.budget))
                for i in s.visited:
                    visits[kind][i] += 1.0

                cold = fresh_state(kind, EXT_FACTORS)
                s2 = Surface(ext[(ctx, seed)])
                build(kind, cold, "ext")(s2, np.random.default_rng(70_000 + r), args.budget)
                aucs["cold"].append(s2.auc(args.budget))

                s3 = Surface(ext[(ctx, seed)])
                marginal_replay(visits[kind], s3, np.random.default_rng(70_000 + r), args.budget)
                aucs["marginal"].append(s3.auc(args.budget))
            for mode in aucs:
                rows[f"{kind}_{mode}"].append(float(np.mean(aucs[mode])))
        print(f"  réplica {r + 1}/{len(seeds)} ({time.perf_counter() - started:.0f}s)", flush=True)

    rng = np.random.default_rng(20260805)

    def boot(diff: np.ndarray) -> dict:
        draws = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
        st = diff[draws].mean(axis=1)
        return {"mean": float(diff.mean()), "lcb95": float(np.percentile(st, 2.5)),
                "ucb95": float(np.percentile(st, 97.5)), "n": int(diff.size)}

    means = {k: float(np.mean(v)) for k, v in rows.items()}
    contrasts = {}
    for kind in ARMS:
        t = np.asarray(rows[f"{kind}_transfer"])
        contrasts[kind] = {
            "vs_cold": boot(np.asarray(rows[f"{kind}_cold"]) - t),
            "vs_marginal_replay": boot(np.asarray(rows[f"{kind}_marginal"]) - t)}
    transfers = {k: (v["vs_cold"]["lcb95"] > 0.0 and v["vs_marginal_replay"]["lcb95"] > 0.0)
                 for k, v in contrasts.items()}
    winners = [k for k, ok in transfers.items() if ok]
    verdict = ("GRID_TRANSFER_ESTABLISHED__" + "_".join(sorted(winners)).upper() if winners
               else "NO_GRID_TRANSFER")

    falsifiers = {
        "f1_the_null_subgrid_reproduces_the_288_cache": {
            "passed": mismatches == 0,
            "evidence": {"why_it_can_fail": "exposing the two factors must not have moved the "
                                            "physics; the 288 cache was written by an earlier run "
                                            "so this anchor is external",
                         "cells_checked": checked, "mismatches": mismatches,
                         "max_abs_delta": max_abs}},
        "f2_the_new_factors_move_the_endpoint": {
            "passed": all(v > 0.0 for v in spread.values()),
            "evidence": {"why_it_can_fail": "this project has already measured 4.56M units of raw "
                                            "material moving exactly zero ReT; if the new factors "
                                            "are decoration the extension is padding",
                         "mean_within_base_config_spread": spread}},
        "f3_transfer_beats_its_marginal_replay": {
            "passed": any(v["vs_marginal_replay"]["lcb95"] > 0.0 for v in contrasts.values()),
            "evidence": {"why_it_can_fail": "if no arm beats a state-blind replay of its own visit "
                                            "marginals, what crossed the boundary was a lookup "
                                            "table and not the shape of the surface",
                         "contrasts": {k: v["vs_marginal_replay"] for k, v in contrasts.items()}}},
        "f4_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print("\n  AUC de regret sobre 4.608 (menor mejor):")
    for kind in ARMS:
        print(f"    {kind:<8} transfer {means[f'{kind}_transfer']:.5f}   "
              f"frío {means[f'{kind}_cold']:.5f}   marginal {means[f'{kind}_marginal']:.5f}")
    print("\n  ventaja de transferir (positivo = transferir ayuda):")
    for kind in ARMS:
        c = contrasts[kind]
        print(f"    {kind:<8} vs frío {c['vs_cold']['mean']:+.5f} "
              f"[{c['vs_cold']['lcb95']:+.5f}]   vs marginal "
              f"{c['vs_marginal_replay']['mean']:+.5f} "
              f"[{c['vs_marginal_replay']['lcb95']:+.5f}]")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<48} {label}")

    payload = {
        "schema_version": "grid_transfer_v1", "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "budget": args.budget, "contexts": contexts, "seeds": seeds,
        "n_base_configs": len(BASE_CONFIGS), "n_ext_configs": len(EXT_CONFIGS),
        "mean_auc": means, "contrasts": contrasts, "transfers": transfers,
        "per_arm": rows, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/search_ladder_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
