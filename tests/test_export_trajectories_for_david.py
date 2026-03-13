from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_export_trajectories_includes_state_constraints_and_reward_terms(
    tmp_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "scripts/export_trajectories_for_david.py",
        "--episodes",
        "1",
        "--seed-start",
        "5",
        "--risk-level",
        "increased",
        "--reward-mode",
        "control_v1",
        "--observation-version",
        "v2",
        "--output-dir",
        str(tmp_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd="/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia",
        check=True,
        capture_output=True,
        text=True,
    )
    assert "state_constraint_context.npy" in completed.stdout
    assert "reward_terms.npy" in completed.stdout

    state_constraints = np.load(tmp_path / "state_constraint_context.npy")
    reward_terms = np.load(tmp_path / "reward_terms.npy")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    reward_fields = json.loads(
        (tmp_path / "reward_terms_fields.json").read_text(encoding="utf-8")
    )

    assert state_constraints.ndim == 2
    assert reward_terms.ndim == 2
    assert state_constraints.shape[0] == reward_terms.shape[0]
    assert metadata["reward_mode"] == "control_v1"
    assert metadata["observation_version"] == "v2"
    assert metadata["state_constraint_context_shape"][1] == state_constraints.shape[1]
    assert metadata["reward_terms_shape"][1] == reward_terms.shape[1]
    assert metadata["obs_shape"][1] == 18
    assert reward_fields["fields"][0] == "reward_total"
