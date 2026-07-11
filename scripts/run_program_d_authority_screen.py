#!/usr/bin/env python3
"""Program D — lever D1 (Op9 rationing rule): Phase 1 liveness + Phase 2 authority screen.

For each backorder_priority_rule, evaluate the frozen garrido_proxy_v1 physics
over N paired calibration tapes (strict-CRN, so the exogenous risk/demand stream
is identical across rules for a given seed — the rule only re-orders the standing
backlog). Reports:

  Phase 1 (liveness): does the rule change Ut/Bt/ReT at all? (non-identity assert)
  Phase 2 (authority): best constant rule vs worst constant rule, paired bootstrap
    CI95 on ret_excel and Ut. Authority present iff the best-worst gap CI95 excludes
    zero. This also fixes the strong same-contract comparator (best constant rule)
    that any later state-contingent policy (Phase 3 branching) must beat.

No PPO, no virgin tapes. See docs/PROGRAM_D_LEVER_DISCOVERY_PREREG_2026-07-11.md.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.config import (  # noqa: E402
    BACKORDER_PRIORITY_RULE_OPTIONS,
    HOURS_PER_WEEK,
)

PROXY_PATH = Path("supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json")
DEFAULT_OUTPUT = Path("results/program_d/d1_authority_screen")
FAMILIES = ("R1", "R2", "mixed")
RISK_LEVELS = ("current", "increased")


def proxy_base_kwargs() -> dict:
    proxy = json.loads(PROXY_PATH.read_text(encoding="utf-8"))
    base = dict(proxy["sim_kwargs"])
    base.pop("risk_level", None)
    base.pop("seed_stream_mode", None)
    return base, sha256(PROXY_PATH.read_bytes()).hexdigest()


def run_one(rule: str, seed: int, risk_level: str, horizon: float, base: dict) -> dict:
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={},
        seed=seed,
        horizon=horizon,
        risk_level=risk_level,
        strict_exogenous_crn=True,
        backorder_priority_rule=rule,
        **base,
    )
    sim.run()
    metrics = sim.compute_order_level_ret()
    return {
        "rule": rule,
        "seed": seed,
        "risk_level": risk_level,
        "ret_excel": float(metrics["mean_ret_excel_formula"]),
        "unattended_orders": int(sim.total_unattended_orders),
        "backorders": int(sim.total_backorders),
        "n_orders": int(metrics["n_orders"]),
        "cumulative_backorder_qty": float(sim.cumulative_backorder_qty),
    }


def paired_bootstrap_ci(deltas: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = float(np.mean(deltas[idx]))
    return float(np.mean(deltas)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=700_000)
    parser.add_argument("--horizon-weeks", type=int, default=104)
    parser.add_argument("--warmup-hours", type=float, default=8_000.0)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base, proxy_sha = proxy_base_kwargs()
    horizon = args.warmup_hours + args.horizon_weeks * HOURS_PER_WEEK

    rows: list[dict] = []
    for i in range(args.tapes):
        seed = args.seed_base + 1 + i
        risk_level = RISK_LEVELS[(i // len(FAMILIES)) % len(RISK_LEVELS)]
        for rule in BACKORDER_PRIORITY_RULE_OPTIONS:
            row = run_one(rule, seed, risk_level, horizon, base)
            rows.append(row)
        print(f"[d1] tape {i+1}/{args.tapes} seed={seed} done", flush=True)

    # Reshape to seed x rule matrices.
    seeds = sorted({r["seed"] for r in rows})
    rules = list(BACKORDER_PRIORITY_RULE_OPTIONS)
    ret = {rule: np.array([next(r["ret_excel"] for r in rows if r["seed"] == s and r["rule"] == rule) for s in seeds]) for rule in rules}
    ut = {rule: np.array([next(r["unattended_orders"] for r in rows if r["seed"] == s and r["rule"] == rule) for s in seeds]) for rule in rules}

    # Phase 1 liveness: at least one rule differs from spt_contingent on some seed.
    live = any(not np.allclose(ret[rule], ret["spt_contingent"]) for rule in rules if rule != "spt_contingent")

    # Phase 2 authority: best vs worst constant rule by mean ret_excel.
    mean_ret = {rule: float(np.mean(ret[rule])) for rule in rules}
    mean_ut = {rule: float(np.mean(ut[rule])) for rule in rules}
    best_rule = max(mean_ret, key=mean_ret.get)
    worst_rule = min(mean_ret, key=mean_ret.get)

    ret_delta = ret[best_rule] - ret[worst_rule]
    ut_delta = ut[worst_rule] - ut[best_rule]  # fewer lost is better for best_rule
    ret_gap, ret_lo, ret_hi = paired_bootstrap_ci(ret_delta, args.n_boot, 0xD1A)
    ut_gap, ut_lo, ut_hi = paired_bootstrap_ci(ut_delta.astype(float), args.n_boot, 0xD1B)

    static_range = max(mean_ret.values()) - min(mean_ret.values())
    delta_authority = 0.02 * static_range  # preregistered threshold
    authority = bool(ret_lo > 0.0)

    # Best rule vs the thesis default (spt_contingent) — the managerial headline.
    vs_default = ret[best_rule] - ret["spt_contingent"]
    vd_gap, vd_lo, vd_hi = paired_bootstrap_ci(vs_default, args.n_boot, 0xD1C)

    verdict = {
        "kind": "program_d_d1_authority_screen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": "garrido_proxy_v1",
        "proxy_sha256": proxy_sha,
        "n_tapes": args.tapes,
        "seed_base": args.seed_base,
        "horizon_weeks": args.horizon_weeks,
        "primary_metric": "ret_excel",
        "co_primary": ["unattended_orders"],
        "rules": rules,
        "phase1_liveness_pass": live,
        "mean_ret_excel_by_rule": mean_ret,
        "mean_unattended_orders_by_rule": mean_ut,
        "best_constant_rule": best_rule,
        "worst_constant_rule": worst_rule,
        "best_minus_worst_ret_excel": {"mean": ret_gap, "ci95": [ret_lo, ret_hi]},
        "best_minus_worst_unattended_reduction": {"mean": ut_gap, "ci95": [ut_lo, ut_hi]},
        "best_minus_thesis_default_ret_excel": {"mean": vd_gap, "ci95": [vd_lo, vd_hi]},
        "static_ret_range": static_range,
        "delta_authority_threshold": delta_authority,
        "authority_present": authority,
        "interpretation": (
            "AUTHORITY_PRESENT: the rationing rule moves ret_excel with CI95>0; "
            "proceed to Phase 3 branching to test state-contingent value."
            if authority else
            "NO_STATIC_AUTHORITY: constant rule choice does not move ret_excel; "
            "lever D1 is dead in this contract."
        ),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }

    with (args.output_dir / "d1_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
