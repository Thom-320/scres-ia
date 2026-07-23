#!/usr/bin/env python3
"""Service-aware retained screen: variant arms on burned roots vs the frozen c256.

Contract: contracts/q_r1_service_aware_retained_screen_v1.json (frozen before execution).
Only the VARIANT retained arm is evaluated here; the comparator side (frozen c256 retained
and reset arms) is reused from the frozen Pareto raw rows, CRN-valid because campaigns are
rebuilt identically. V0 is a documented no-op: the conditional bank is already
regime-consistent (probability_c = share if regime_c else 1-share); the defect is the
posterior-EXPECTED aggregation, which V1/V2/V3 target.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_q_r1_comparator_v2_preflight import development_states  # noqa: E402
from scripts.run_q_r1_successor_abc import fixed_theta_belief  # noqa: E402
from supply_chain.q_r1_comparator_v2 import (  # noqa: E402
    ComparatorV2Config,
    comparator_v2_calendar,
)
from supply_chain.q_r1_retained_learning import evaluate_calendar  # noqa: E402


def variant_grid() -> list[ComparatorV2Config]:
    """The frozen grid: V1 (4) + V2 (2) + V3 (3) = 9 configs. V0 no-op documented."""
    configs: list[ComparatorV2Config] = []
    for alpha in (0.10, 0.25):
        for floor in (0.70, 0.80):
            configs.append(ComparatorV2Config(
                horizon=4, conditional_paths=256, mode="constraint_aware",
                worst_product_floor=floor, fill_tail_alpha=alpha,
                value_indifference_tolerance=0.0, tie_breaker="legacy"))
    for slope in (0.4, 0.6):
        configs.append(ComparatorV2Config(
            horizon=4, conditional_paths=256, mode="constraint_aware",
            worst_product_floor=0.70, belief_floor_slope=slope,
            value_indifference_tolerance=0.0, tie_breaker="legacy"))
    for lam in (0.5, 1.0, 2.0):
        configs.append(ComparatorV2Config(
            horizon=4, conditional_paths=256, mode="scenario",
            worst_product_floor=0.0, penalty_lambda=lam, penalty_floor=0.80,
            fill_tail_alpha=0.10, value_indifference_tolerance=0.0,
            tie_breaker="legacy"))
    assert len(configs) == 9
    return configs


class _Args:
    def __init__(self, seed_start: int, histories: int) -> None:
        self.seed_start = seed_start
        self.histories = histories
        self.campaigns = 12
        self.states = 12


def burned_states() -> list:
    rows = []
    for start in (7_570_801, 7_570_807, 7_570_813, 7_570_819):
        rows.extend(development_states(_Args(start, 6)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-indices", nargs="+", type=int, required=True)
    parser.add_argument("--hard-cap-seconds", type=float, default=43_200.0)
    parser.add_argument("--max-states", type=int, default=None,
                        help="smoke only: truncate the state list")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    configs = variant_grid()
    chosen = [configs[i] for i in args.config_indices]
    states = burned_states()
    if args.max_states is not None:
        states = states[: args.max_states]
    sched = scheduler()
    started = time.perf_counter()
    rows: list[dict] = []
    for config in chosen:
        for campaign, retained_prior in states:
            if time.perf_counter() - started > args.hard_cap_seconds:
                raise TimeoutError("service-aware screen shard hard cap exceeded")
            calendar, detail = comparator_v2_calendar(
                campaign=campaign,
                belief=fixed_theta_belief(retained_prior),
                scheduler=sched,
                config=config,
            )
            metrics = evaluate_calendar(campaign=campaign, calendar=calendar, scheduler=sched)
            rows.append({
                "config_id": config.config_id,
                "history_root": int(campaign.history_root),
                "campaign_index": int(campaign.campaign_index),
                "persistence_mode": campaign.persistence_mode,
                "retained_prior": float(retained_prior),
                "variant": metrics,
                "variant_calendar": list(calendar),
                "fallbacks": sum(
                    1 for step in detail.get("diagnostics", [])
                    if step.get("fallback_used")
                ),
            })
            print(f"  {config.config_id[-28:]} root {campaign.history_root} "
                  f"c{campaign.campaign_index} ({time.perf_counter()-started:.0f}s)",
                  flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": "q_r1_service_aware_screen_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "contracts/q_r1_service_aware_retained_screen_v1.json",
        "config_indices": args.config_indices,
        "configs": [c.config_id for c in chosen],
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_performed": False,
        "learner_return_used": False,
    }, indent=1, sort_keys=True) + "\n")
    print(f"rows={len(rows)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
