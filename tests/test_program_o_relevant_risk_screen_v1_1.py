import json
from pathlib import Path

from scripts.run_program_o_relevant_risk_screen_v1_1 import event_value


ROOT = Path(__file__).resolve().parent.parent


class Event:
    risk_id = "R21"
    affected_ops = [3, 5, 6, 7, 9]


def test_event_value_reads_dataclass_field_and_mapping():
    assert event_value(Event(), "affected_ops") == [3, 5, 6, 7, 9]
    assert event_value({"affected_ops": [13]}, "affected_ops") == [13]


def test_v11_contract_freezes_phi_one_and_fixed_controller_model():
    contract = json.loads(
        (ROOT / "contracts/program_o_relevant_risk_sensitivity_v1_1.json").read_text()
    )
    assert "phi=1" in contract["gates"]["G1"]
    assert contract["primary_cell"]["controller_model_parameters"] == {
        "regime_persistence": 0.75,
        "dominant_share": 0.90,
    }
    assert contract["future_grid_if_authorized"]["selection_for_program_o_r"] is False
    assert "after the common neutral" in contract["risks"]["activation"]


def test_v11_has_no_g2_execution_surface():
    source = (ROOT / "scripts/run_program_o_relevant_risk_screen_v1_1.py").read_text()
    assert 'choices=("g0", "g1")' in source
    assert "G2" not in source.split("def main()", 1)[1]


def test_historical_stop_and_cvar_correction_are_fail_closed():
    stop = json.loads(
        (ROOT / "results/program_o/relevant_risk_sensitivity_v1/stop_after_g0_v1.json").read_text()
    )
    correction = json.loads(
        (ROOT / "results/program_o/cvar_gate_instrument_audit_v1/interpretation_correction_v1.json").read_text()
    )
    assert stop["g2_executed"] is False
    assert correction["corrective_rerun_authorized"] is False
    assert correction["historical_verdict_unchanged"] == "STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION"
