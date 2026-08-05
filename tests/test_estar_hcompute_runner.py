from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/garrido_expanded_des_e_star_v2_hcompute.json"
TAPE = ROOT / (
    "results/expanded_contract_comparators_v2_preflight_1dc40c1/preflight/"
    "R1r_actual_tapes.json"
)
RUNNER = ROOT / "scripts/run_estar_hcompute_preflight.py"


def test_burned_preflight_uses_expanded_bridge_and_stays_non_scientific(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            str(RUNNER),
            "--contract",
            str(CONTRACT),
            "--run-role",
            "BURNED_COMPUTE_PREFLIGHT",
            "--replay-of",
            "expanded_contract_comparators_v2_preflight_1dc40c1",
            "--tape-file",
            str(TAPE),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["claim_status"] == "H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE"
    assert payload["expanded_des_bridge_ready"] is True
    assert payload["fresh_seeds_opened"] is False
    assert payload["learner_trained"] is False
    assert payload["command_argv"]
    assert payload["hardware_and_protocol"]["protocol"][
        "warmups_excluded_from_hot_statistics"
    ] is True
    assert payload["timing"][0]["planners"][0]["cold_seconds"] >= 0.0
    assert "solver_iterations" in payload["timing"][0]["planners"][0]
    assert payload["falsifiers"]["all_passed"] is True
    assert all(
        row["passed"]
        for name, row in payload["falsifiers"].items()
        if name != "all_passed"
    )
    assert payload["h_compute_adjudicated"] is True
    assert payload["h_compute_adjudication"]["calls_gate"] is True
    assert payload["h_compute_adjudication"]["calls_two_consecutive_increases"] is True
