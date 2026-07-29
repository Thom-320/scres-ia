import json
from pathlib import Path

from scripts.validate_program_o_exact_transducer import connected_components


ROOT = Path(__file__).resolve().parent.parent


def test_connected_grid_component():
    component = connected_components(
        {"rho75_share75", "rho75_share90", "rho90_share75", "rho90_share90"}
    )
    assert component == [["rho75_share75", "rho75_share90", "rho90_share75", "rho90_share90"]]


def test_validation_freeze_keeps_hobs_and_learner_blocked():
    freeze = json.loads(
        (ROOT / "research/paper2_exhaustive_search/program_o_exact_transducer_validation_freeze_20260714.json").read_text()
    )
    assert freeze["validation_seeds"] == [7400025, 7400048]
    assert freeze["bootstrap"]["static_calendar_reselected_in_every_cell_and_resample"] is True
    assert freeze["h_obs_authorized"] is False
    assert freeze["learner_authorized"] is False
    assert freeze["paper3_authorized"] is False
