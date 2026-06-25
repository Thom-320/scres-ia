#!/usr/bin/env python3
"""Headroom diagnostic: does the best [6,3] static policy change across regimes?

If the argmax static policy (by order-level ReT / fill) is the SAME under current /
increased / severe, then there is no regime-dependent optimum -- a retained learner has
nothing to accumulate that a reset learner or a single static policy lacks. That confirms
the weak retained-vs-reset signal is structural headroom (cause #1), independent of reward.

No training; pure static-policy evaluation under common random numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.evaluate_retained_reset_learning as ev  # noqa: E402
from supply_chain.scenario_tape import RegimePhase  # noqa: E402

REGIMES = ("current", "increased", "severe")


def regime_for(level: str) -> RegimePhase:
    return RegimePhase(
        disruption_phase=0, demand_phase=0, disruption_level=level, demand_multiplier=1.0
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="3101,3102,3103")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/headroom/argmax_stability.json"),
    )
    cli = p.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",") if s.strip()]

    args = ev.build_parser().parse_args([])
    args.max_steps = cli.max_steps

    table: dict[str, dict] = {}
    for level in REGIMES:
        regime = regime_for(level)
        rows = []
        for action in range(18):
            ret, fill, loss = [], [], []
            for seed in seeds:
                r = ev.run_episode(
                    args=args, condition=f"static_{action}", seed=seed,
                    cycle_index=0, policy_fn=lambda _o, _i, a=action: a, regime=regime,
                )
                ret.append(r["order_level_ret_mean"])
                fill.append(r["fill_rate_order_level"])
                loss.append(r["service_loss_area"])
            rows.append(
                {
                    "action": action,
                    "policy": ev.static_action_name(action),
                    "ret": float(np.nanmean(ret)),
                    "fill": float(np.nanmean(fill)),
                    "service_loss": float(np.nanmean(loss)),
                }
            )
        best_ret = max(rows, key=lambda x: x["ret"])
        best_fill = max(rows, key=lambda x: x["fill"])
        table[level] = {"rows": rows, "argmax_ret": best_ret, "argmax_fill": best_fill}

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print("\nHEADROOM DIAGNOSTIC — best static [6,3] policy per regime")
    print(f"{'regime':>10} {'argmax(ReT)':>22} {'ReT':>7} {'argmax(fill)':>22} {'fill':>7}")
    ret_winners, fill_winners = set(), set()
    for level in REGIMES:
        br, bf = table[level]["argmax_ret"], table[level]["argmax_fill"]
        ret_winners.add(br["policy"])
        fill_winners.add(bf["policy"])
        print(f"{level:>10} {br['policy']:>22} {br['ret']:7.3f} {bf['policy']:>22} {bf['fill']:7.3f}")
    print(
        f"\nregime-dependent optimum? ReT-argmax distinct policies={len(ret_winners)}, "
        f"fill-argmax distinct policies={len(fill_winners)}"
    )
    print(
        "  >1 distinct => optimal action varies by regime (retention has something to learn)\n"
        "  ==1          => single static policy is best everywhere => no retention headroom (cause #1)"
    )
    print(f"Saved: {cli.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
