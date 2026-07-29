from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_garrido_h2_h3_confirmation_v1 import holm_passes


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts/garrido_h2_h3_confirmation_v1.json").read_text()
)
DEVELOPMENT = json.loads(
    (ROOT / "contracts/garrido_h2_h3_corrective_v1.json").read_text()
)


def test_confirmation_roots_are_fresh_unique_and_still_sealed() -> None:
    roots = CONTRACT["execution"]["confirmation_tape_roots"]
    development = set(DEVELOPMENT["execution"]["tape_roots"])
    assert len(roots) == len(set(roots)) == 12
    assert not development.intersection(roots)
    assert CONTRACT["confirmation_roots_opened"] is False


def test_confirmation_keeps_the_physical_contract_fixed() -> None:
    for key in (
        "hours_per_year",
        "common_evaluation_start_hours",
        "strict_exogenous_crn",
        "periodic_release_mode",
        "assembly_batch_release_mode",
        "raw_material_flow_mode",
        "raw_material_order_up_to_multiplier",
        "downstream_q_source",
        "buffer_rule",
    ):
        assert CONTRACT["execution"][key] == DEVELOPMENT["execution"][key]


def test_confirmation_controls_all_six_panels_and_service_concordance() -> None:
    inference = CONTRACT["confirmation_inference"]
    assert len(inference["primary_panels"]) == 6
    assert inference["familywise_alpha"] == 0.05
    assert inference["multiplicity"].startswith("Holm")
    assert set(inference["mandatory_panel_concordance"]) == {
        "ret_excel_full_ledger",
        "flow_fill_rate",
        "delivered_rations",
        "unresolved_orders",
        "generated_orders",
    }


def test_holm_step_down_stops_after_first_failure() -> None:
    result = holm_passes(
        {"a": 0.001, "b": 0.02, "c": 0.06},
        0.05,
    )
    assert result["a"]["pass"] is True
    assert result["b"]["pass"] is True
    assert result["c"]["pass"] is False
