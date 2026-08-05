from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_estar_hcompute_contract import validate_contract
from supply_chain.estar_bridge import check_flags_off_golden, load_tape


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/garrido_expanded_des_e_star_v2_hcompute.json"
TAPE = ROOT / (
    "results/expanded_contract_comparators_v2_preflight_1dc40c1/preflight/"
    "R1r_actual_tapes.json"
)


def _load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_e_star_v2_is_design_only_and_no_fresh_execution() -> None:
    contract = _load_contract()
    assert contract["status"] == "DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT"
    assert contract["authority"]["scientific_execution_authorized"] is False
    assert contract["authority"]["fresh_roots_opened"] is False
    assert contract["authority"]["neural_training_authorized"] is False
    assert contract["h_compute"]["burned_only"] is True


def test_contract_validator_is_fail_closed() -> None:
    contract = _load_contract()
    registry = json.loads(
        (ROOT / "research/seed_custody_registry.json").read_text(encoding="utf-8")
    )
    assert validate_contract(contract, registry) == {"ok": True, "errors": []}


def test_flags_off_golden_vector_passes() -> None:
    contract = _load_contract()
    tape = load_tape(TAPE)
    result = check_flags_off_golden(
        tape, contract["flags_off_bridge"]["golden_payload_sha256"]
    )
    assert result["passed"] is True


def test_validator_rejects_cvar_as_promoting_metric() -> None:
    contract = _load_contract()
    registry = json.loads(
        (ROOT / "research/seed_custody_registry.json").read_text(encoding="utf-8")
    )
    contract["metric_hierarchy"]["cvar"]["may_promote_alone"] = True
    result = validate_contract(contract, registry)
    assert result["ok"] is False
    assert any("cvar.may_promote_alone" in error for error in result["errors"])
