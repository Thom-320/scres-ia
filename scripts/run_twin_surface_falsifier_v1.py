#!/usr/bin/env python3
"""Run the twin-surface falsifier off the sealed cache, and seal its verdict.

The affine-rescaling test is blind to a leak that is invariant to scale: an arm reading the
surface's RANK or ARGMAX would pass it. The twin test is not. It keeps every cell the reference
path visited exactly as it was, alters two cells that no arm on that path ever touched, and
replays the identical RNG stream.

A legitimate prefix arm cannot notice -- it never read those cells. An oracle arm must, because
its min/max moved. So this check has a required PASS and a required FAIL, and neither can be
satisfied by the code agreeing with itself.

`f6` was added to the contract after the audit had already been sealed, so it had never actually
run. This runner closes that gap without rebuilding the 20,736-episode surface.

Contract: docs/PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md (f6)
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from scripts.seal_garrido_surface_cache_v1 import verify_sealed_slice  # noqa: E402

AUDIT = Path(__file__).resolve().parent / "run_meta_learner_normaliser_audit_v1.py"
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def load_audit_module():
    spec = iu.spec_from_file_location("normaliser_audit", AUDIT)
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--seed", type=int, default=5_300_001)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path, default=Path("results/twin_surface/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    audit = load_audit_module()
    surface, seeds = {}, set()
    for path in sorted(args.cache.rglob("*.json")):
        payload = json.loads(path.read_text())
        verify_sealed_slice(payload, expected_cells=len(audit.CONFIGS),
                            expected_grid_id="wrap288_v1")
        surface[(payload["context"], int(payload["seed"]))] = payload["cells"]
        seeds.add(int(payload["seed"]))
    print(f"  caché: {len(surface)} rebanadas, {len({k[0] for k in surface})} contextos")

    twin = audit.twin_surface_falsifier(surface, seed=args.seed, budget=args.budget)
    verdict = ("PREFIX_NORMALISER_IS_BLIND_TO_THE_UNRUN_SURFACE" if twin["passed"]
               else "TWIN_SURFACE_FALSIFIER_FAILED")

    falsifiers = {
        "f6_surface_twins_have_identical_prefix_paths": {
            "passed": bool(twin["passed"]),
            "evidence": {"why_it_can_fail": twin["why_it_can_fail"],
                         "required_pass": "the prefix arm keeps an identical path",
                         "required_fail": "the oracle arm must change, or the twin is too weak to "
                                          "detect anything and the test proves nothing",
                         **twin}},
        "f_no_fresh_seeds": custody_falsifier(sorted(seeds), replay_of=args.replay_of,
                                              exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  prefijo mantiene la ruta:  {twin['prefix_passed']}")
    print(f"  el oráculo reacciona:      {twin['oracle_reacted']}")
    for norm, v in twin["by_normaliser"].items():
        per = {s: sum(1 for f in c.values() if f) for s, c in v["path_unchanged"].items()}
        print(f"    {norm:<7} rutas idénticas de 6 contextos: {per}")
    print(f"\n  veredicto: {verdict}\n")

    payload = {
        "schema_version": "twin_surface_falsifier_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_STRUCTURAL_SPY_TEST",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "cache": str(args.cache), "seed": args.seed, "budget": args.budget,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/garrido_normaliser_audit/result.json"))
    print(f"  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
