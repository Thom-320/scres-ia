from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_program_s_s1_preopen import audit
from scripts.build_program_s_morris_design import build
from scripts.preflight_program_s_transducer import (
    BURNED_SEED,
    SCHEDULER,
    cell,
    forced_tape,
)
from research.paper2_exhaustive_search.program_s_transducer import (
    exact_short_horizon_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_risk_aware_transducer_is_exact_for_r24_priority_and_risk_ret() -> None:
    payload = exact_short_horizon_gate(
        seed=BURNED_SEED,
        scheduler=SCHEDULER,
        cell=cell("LOC_SURGE"),
        risk_tapes_by_horizon={1: forced_tape("LOC_SURGE", 1)},
        horizons=(1,),
    )
    assert payload["pass"] is True
    assert payload["horizons"]["1"]["calendars"] == 4
    assert payload["horizons"]["1"]["unique_skeleton_hashes"] == 1
    assert payload["horizons"]["1"]["max_matrix_abs_error"] == 0.0


def test_live_transducer_preflight_admits_all_three_masks_without_seed_opening() -> None:
    payload = json.loads(
        (ROOT / "results/program_s/s1_transducer_preflight_v1/result.json").read_text()
    )
    assert payload["verdict"] == "PASS_S1_TRANSDUCER_PREFLIGHT_ALL_MASKS_ELIGIBLE"
    assert set(payload["eligible_masks"]) == {
        "PRODUCTION_QUALITY_SURGE",
        "LOC_SURGE",
        "CROSS_ECHELON_SURGE",
    }
    assert payload["scientific_seed_blocks_opened"] == []
    assert max(
        row["h8_replay"]["max_matrix_abs_error"]
        for row in payload["masks"].values()
    ) <= 1e-10


def test_morris_design_is_deterministic_optimized_and_capacity_anchored() -> None:
    live = json.loads(
        (
            ROOT
            / "research/paper2_exhaustive_search/program_s_morris_design_v1.json"
        ).read_text()
    )
    assert build() == live
    assert len(live["groups"]) == 6
    assert sum(len(group["trajectories"]) for group in live["groups"]) == 60
    assert all(
        all(anchor["baseline_capacity_multiplier"] == 1.0 for anchor in group["mandatory_capacity_1_anchors"])
        for group in live["groups"]
    )
    assert live["scientific_seed_block_opened"] is False


def test_s1_is_technically_ready_but_q_priority_blocks_seed_opening() -> None:
    payload = audit()
    assert payload["technically_ready"] is True
    assert payload["program_q_vps_priority_active"] is True
    assert payload["scientific_seed_authorization"] is False
    assert payload["verdict"] == "HOLD_S1_TECHNICALLY_READY_PROGRAM_Q_HAS_VPS_PRIORITY"

