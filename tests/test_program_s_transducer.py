from __future__ import annotations

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
    native = json.loads(
        (
            ROOT
            / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
        ).read_text()
    )
    wartime = json.loads(
        (
            ROOT
            / "research/paper2_exhaustive_search/program_s_wartime_morris_design_v1_1.json"
        ).read_text()
    )
    assert build("THESIS_NATIVE_INDEPENDENT") == native
    assert build("RESEARCHER_WARTIME_COUPLED") == wartime
    assert len(native["groups"]) == 3
    assert sum(len(group["trajectories"]) for group in native["groups"]) == 30
    assert native["promotion_authorized"] is True
    assert wartime["promotion_authorized"] is False
    assert wartime["scientific_seed_block"] is None
    assert all(
        all(anchor["baseline_capacity_multiplier"] == 1.0 for anchor in group["mandatory_capacity_1_anchors"])
        for group in native["groups"]
    )
    assert all(
        point["phi_by_risk"].get("R23", 1.0) == 1.0
        and point["psi_by_risk"].get("R23", 1.0) == 1.0
        for design in (native, wartime)
        for group in design["groups"]
        for trajectory in group["trajectories"]
        for point in trajectory["points"]
    )
    assert native["scientific_seed_block_opened"] is False


def test_s1_is_technically_ready_but_q_priority_blocks_seed_opening() -> None:
    payload = audit()
    assert payload["technically_ready"] is True
    assert payload["program_q_vps_priority_active"] is True
    assert payload["scientific_seed_authorization"] is False
    assert payload["verdict"] == "HOLD_S1_TECHNICALLY_READY_PROGRAM_Q_HAS_VPS_PRIORITY"
