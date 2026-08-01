from __future__ import annotations

import json

from scripts.adjudicate_neural_headroom_gate_v1 import main


def test_e1_headroom_gate_holds_when_required_placebo_is_missing(tmp_path, monkeypatch):
    import scripts.adjudicate_neural_headroom_gate_v1 as gate

    output = tmp_path / "result.json"
    monkeypatch.setattr(gate, "OUTPUT", output)
    assert main() == 0
    payload = json.loads(output.read_text())
    assert payload["claim_status"] == "HOLD_E1_PLACEBO_NOT_OPENED"
    assert payload["placebo_required"] is True
    assert payload["placebo_status"] == "NOT_OPENED_BY_E1_CONTRACT"
    assert payload["training_authorized"] is False
