from __future__ import annotations

import json

from scripts.adjudicate_neural_headroom_gate_v1 import main


def test_e1_headroom_gate_refuses_training_when_headroom_is_zero(tmp_path, monkeypatch):
    import scripts.adjudicate_neural_headroom_gate_v1 as gate

    output = tmp_path / "result.json"
    monkeypatch.setattr(gate, "OUTPUT", output)
    assert main() == 0
    payload = json.loads(output.read_text())
    assert payload["claim_status"] == "NO_GO_NEURAL_PREMIUM_E1_HEADROOM_CLOSED"
    assert payload["training_authorized"] is False
