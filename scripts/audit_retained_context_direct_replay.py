#!/usr/bin/env python3
"""Direct-SimPy replay audit for the R0 oracle arm on burned histories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
    simulate_full_des_frontier,
)
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402
from supply_chain.retained_context_discovery import build_campaign_history  # noqa: E402


def main() -> None:
    sched = scheduler()
    max_error = 0.0
    failures = []
    episodes = 0
    for kappa in (0.5, 0.75, 0.9):
        for root in range(7_570_001, 7_570_004):
            history = build_campaign_history(
                history_root=root,
                campaigns=12,
                kappa=kappa,
                scheduler=sched,
                regime_persistence=0.9,
                dominant_share=0.9,
            )
            for campaign in history:
                prior = 1.0 - 1e-6 if campaign.initial_regime == "P_C" else 1e-6
                calendar, _decisions = state_rich_calendar(
                    skeleton=campaign.skeleton.as_dict(),
                    scheduler=sched,
                    config=StateRichConfiguration("belief_mpc", 3),
                    regime_persistence=0.9,
                    dominant_share=0.9,
                    initial_belief_c=prior,
                )
                transduced = simulate_full_des_frontier(
                    skeleton=campaign.skeleton,
                    scheduler=sched,
                    calendars=np.asarray([calendar], dtype=np.uint8),
                )
                sim, panel = run_program_o_full_des_episode(
                    seed=campaign.tape_seed,
                    calendar=calendar,
                    scheduler=sched,
                    regime_persistence=0.9,
                    dominant_share=0.9,
                    downstream_freight_physics_mode="fixed_clock_physical_v1",
                    initial_regime=campaign.initial_regime,
                )
                direct = direct_full_des_vector(sim, panel)
                episode_error = max(
                    abs(float(transduced[key][0]) - float(direct[key]))
                    for key in MATRIX_KEYS
                )
                max_error = max(max_error, episode_error)
                if episode_error > 1e-8:
                    failures.append(
                        {
                            "kappa": kappa,
                            "history_root": root,
                            "campaign_index": campaign.campaign_index,
                            "max_error": episode_error,
                        }
                    )
                episodes += 1
    payload = {
        "schema_version": "retained_context_direct_replay_audit_v1",
        "status": "EXPLORATORY_NO_CLAIM",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": "3 of 24 history roots per kappa; every campaign; oracle arm",
        "episodes": episodes,
        "max_abs_error": max_error,
        "tolerance": 1e-8,
        "failures": failures,
        "passed": not failures,
    }
    output = ROOT / "results/retained_context/r0_primary_v1/direct_replay_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
