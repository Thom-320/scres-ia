import hashlib
import json
from pathlib import Path

from scripts.bound_program_o_affected_orders import row_ceiling


ROOT = Path(__file__).resolve().parent.parent


def json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_row_ceiling_counts_smallest_rescuable_prefix():
    result = row_ceiling([0.0, 0.5, 1.0, 1.0], gate=0.2)
    assert result["minimum_perfectly_rescued_rows_for_gate"] == 1
    assert result["minimum_fraction_visible_rows_for_gate"] == 0.25
    assert result["all_rows_perfect_rescue_upper_delta"] == 0.375


def test_program_o_contract_keeps_future_tapes_sealed_and_learner_blocked():
    contract = json.loads(
        (ROOT / "contracts/program_o_multi_ration_product_mix_v1.json").read_text()
    )
    assert contract["governing_metric"] == "ret_excel_request_snapshot_v2"
    assert contract["frozen_endpoint_gate"] == 0.01
    assert all(
        row["status"] == "SEALED_NOT_AUTHORIZED"
        for row in contract["tape_blocks"].values()
    )
    assert contract["learner_authorized"] is False
    assert contract["paper3_authorized"] is False


def test_completed_bound_is_content_addressed_if_present():
    path = ROOT / "results/program_o/affected_order_bound_v1/result.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    expected = payload.pop("content_sha256")
    assert json_sha256(payload) == expected
    assert payload["metric"] == "ret_excel_request_snapshot_v2"
    assert payload["claim_boundary"]["h_pi_established"] is False
    assert payload["claim_boundary"]["learner_authorized"] is False
