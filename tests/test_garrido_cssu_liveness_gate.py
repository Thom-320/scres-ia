from __future__ import annotations

import json
from pathlib import Path

from scripts.run_cssu_liveness_gate import main


def test_cssu_liveness_gate_passes_interface_and_holds_op11(tmp_path, monkeypatch):
    import scripts.run_cssu_liveness_gate as gate

    output = tmp_path / "result.json"
    monkeypatch.setattr(gate, "OUTPUT", output)
    assert main() == 0
    payload = json.loads(output.read_text())
    assert payload["claim_status"] == "GATE_A_PASS_GATE_B_HOLD"
    assert payload["gate_a"]["status"] == "PASS"
    assert payload["gate_b"]["status"] == "HOLD_OP11_PHYSICS_UNSPECIFIED"
    assert payload["falsifiers"]["all_passed"] is True
