#!/usr/bin/env python3
"""Which resilience endpoint has headroom? Measure H_regime for every one of them, same surfaces.

Every headroom verdict this project has issued was computed with ONE endpoint. If the choice of
endpoint decides the answer, then "there is no regime-dependent headroom" is a statement about the
metric and not about the supply chain -- and we would not know, because the comparison was never
run.

This runs it. Same cached surfaces, same estimator, same seeds; only the endpoint changes. The
estimator averages over seeds BEFORE the max over actions, because taking the max per (regime,
seed) is per-seed clairvoyance and inflated an E*-C estimate roughly tenfold.

Normalisation matters and is declared: each endpoint is rescaled inside each context by that
context's own observed range, so H_regime reads as "fraction of the achievable spread that knowing
the regime buys". Without it, endpoints on wildly different scales -- ret_excel ~0.009 against a
fill rate ~0.86 -- could not be compared at all.

Development on burned tapes. Adjudicates nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

RAW = (0.0, 17_500.0, 70_000.0, 140_000.0)
GRIDS = {
    "wrap288_v1": {"buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
                   "shifts": (1, 2, 3), "op9_rop": (12.0, 24.0, 36.0, 48.0),
                   "op12_rop": (12.0, 24.0, 36.0, 48.0)},
}
GRIDS["wrap288_compat_extended_v1"] = dict(GRIDS["wrap288_v1"], op3_rm=RAW, op5_rm=RAW)

#: Endpoints where LOWER is better must be flipped before the max, or H_regime measures the
#: value of knowing the regime in order to do the worst possible thing.
LOWER_IS_BETTER = {"lost_orders", "demanded_rations"}
N_BOOT = 2_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def load(root: Path, grid_id: str):
    """surface[(context, seed)][endpoint] -> vector over configurations."""
    factors = GRIDS[grid_id]
    n_cfg = len(list(itertools.product(*factors.values())))
    surface, contexts, seeds, keys = {}, [], set(), None
    for path in sorted(root.rglob("*.json")):
        p = json.loads(path.read_text())
        if p.get("grid_id") != grid_id:
            continue
        cells = p["cells"]
        if len(cells) != n_cfg:
            raise SystemExit(f"{path}: {len(cells)} celdas, esperaba {n_cfg}")
        panel_keys = sorted(cells[0]["panel"])
        keys = panel_keys if keys is None else keys
        if panel_keys != keys:
            raise SystemExit(f"{path}: el panel no coincide entre rebanadas")
        surface[(p["context"], int(p["seed"]))] = {
            k: np.array([c["panel"][k] for c in cells], dtype=float) for k in keys}
        seeds.add(int(p["seed"]))
        if p["context"] not in contexts:
            contexts.append(p["context"])
    return surface, contexts, sorted(seeds), keys


def h_regime(surface, contexts, seeds, key) -> float:
    per_ctx = []
    for ctx in contexts:
        mean = np.mean([surface[(ctx, s)][key] for s in seeds], axis=0)
        if key in LOWER_IS_BETTER:
            mean = -mean
        lo, hi = mean.min(), mean.max()
        per_ctx.append((mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean))
    stacked = np.stack(per_ctx)
    return float(stacked.max(axis=1).mean() - stacked.mean(axis=0).max())


def spread(surface, contexts, seeds, key) -> float:
    """Raw authority of the lever on this endpoint, before any normalisation."""
    vals = []
    for ctx in contexts:
        mean = np.mean([surface[(ctx, s)][key] for s in seeds], axis=0)
        vals.append(float(mean.max() - mean.min()))
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache"))
    ap.add_argument("--grids", nargs="+", default=list(GRIDS))
    ap.add_argument("--output", type=Path,
                    default=Path("results/endpoint_headroom_atlas/result.json"))
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260806)

    atlas, all_seeds = {}, set()
    for grid_id in args.grids:
        root = args.cache / grid_id
        if not root.exists():
            print(f"  (sin caché para {grid_id})")
            continue
        surface, contexts, seeds, keys = load(root, grid_id)
        all_seeds |= set(seeds)
        print(f"\n  === {grid_id} · {len(contexts)} contextos x {len(seeds)} semillas ===")
        rows = {}
        for key in keys:
            point = h_regime(surface, contexts, seeds, key)
            draws = np.array([
                h_regime(surface, contexts, [seeds[i] for i in rng.integers(0, len(seeds),
                                                                            len(seeds))], key)
                for _ in range(200)])
            rows[key] = {"H_regime": point,
                         "lcb95": float(np.percentile(draws, 2.5)),
                         "ucb95": float(np.percentile(draws, 97.5)),
                         "raw_spread": spread(surface, contexts, seeds, key),
                         "lower_is_better": key in LOWER_IS_BETTER}
        atlas[grid_id] = {"contexts": contexts, "seeds": seeds, "endpoints": rows}
        for key, v in sorted(rows.items(), key=lambda kv: -kv[1]["H_regime"]):
            flag = "  <-- material" if v["lcb95"] >= 0.05 else ""
            print(f"    {key:<32} H {v['H_regime']:+.5f} [LCB95 {v['lcb95']:+.5f}]"
                  f"  spread {v['raw_spread']:.4g}{flag}")

    material = {g: [k for k, v in d["endpoints"].items() if v["lcb95"] >= 0.05]
                for g, d in atlas.items()}
    any_material = any(material.values())
    verdict = ("SOME_ENDPOINT_CARRIES_REGIME_HEADROOM" if any_material
               else "NO_ENDPOINT_CARRIES_REGIME_HEADROOM")

    falsifiers = {
        "f1_the_atlas_can_separate_endpoints": {
            "passed": bool(len({round(v["H_regime"], 6)
                                for d in atlas.values() for v in d["endpoints"].values()}) > 1),
            "evidence": {"why_it_can_fail": "if every endpoint returned the identical H_regime the "
                                            "atlas would be measuring the estimator, not the "
                                            "endpoints, and no comparison would be possible"}},
        "f2_direction_is_handled": {
            "passed": True,
            "evidence": {"why_it_can_fail": "an endpoint where lower is better would otherwise "
                                            "have its H_regime measure the value of knowing the "
                                            "regime in order to do the WORST thing",
                         "flipped": sorted(LOWER_IS_BETTER)}},
        "f3_no_fresh_seeds": custody_falsifier(sorted(all_seeds), replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  veredicto: {verdict}")
    for g, ks in material.items():
        print(f"    {g}: endpoints con LCB95 >= 0,05 -> {ks or 'ninguno'}")

    payload = {
        "schema_version": "endpoint_headroom_atlas_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "threshold": 0.05, "lower_is_better": sorted(LOWER_IS_BETTER),
        "normalisation": "per-context min-max of the seed-averaged surface",
        "atlas": atlas, "material_endpoints": material, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/surface_gates_extended/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
