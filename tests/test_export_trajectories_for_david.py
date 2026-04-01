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
    state_fields = json.loads(
        (tmp_path / "state_constraint_fields.json").read_text(encoding="utf-8")
    )

    assert state_constraints.ndim == 2
    assert reward_terms.ndim == 2
    assert state_constraints.shape[0] == reward_terms.shape[0]
    assert metadata["reward_mode"] == "control_v1"
    assert metadata["observation_version"] == "v2"
    assert metadata["state_constraint_context_shape"][1] == state_constraints.shape[1]
    assert metadata["reward_terms_shape"][1] == reward_terms.shape[1]
    assert metadata["obs_shape"][1] == 18
    assert len(state_fields["fields"]) == state_constraints.shape[1]
    assert "pending_backorders_count" in state_fields["fields"]
    assert "pending_backorder_qty" in state_fields["fields"]
    assert "unattended_orders_total" in state_fields["fields"]
    assert "cum_backorder_rate_rations_theatre" in state_fields["fields"]
    assert "cum_disruption_fraction_op13" in state_fields["fields"]
    assert reward_fields["fields"][0] == "reward_total"
    assert "ret_unified_step" in reward_fields["fields"]


def test_export_trajectories_supports_v4_unified_contract(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/export_trajectories_for_david.py",
        "--episodes",
        "1",
        "--seed-start",
        "7",
        "--risk-level",
        "increased",
        "--reward-mode",
        "ReT_unified_v1",
        "--observation-version",
        "v4",
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.run(
        cmd,
        cwd="/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia",
        check=True,
        capture_output=True,
        text=True,
    )

    observations = np.load(tmp_path / "observations.npy")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    env_spec = json.loads((tmp_path / "env_spec.json").read_text(encoding="utf-8"))

    assert metadata["observation_version"] == "v4"
    assert metadata["reward_mode"] == "ReT_unified_v1"
    assert metadata["obs_shape"][1] == 24
    assert observations.shape[1] == 24
    assert env_spec["observation_version"] == "v4"
    assert env_spec["observation_fields"][-4:] == [
        "rations_sb_dispatch_norm",
        "assembly_shifts_active_norm",
        "op1_down",
        "op2_down",
    ]


def test_export_trajectories_supports_v5_cycle_contract(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/export_trajectories_for_david.py",
        "--episodes",
        "1",
        "--seed-start",
        "9",
        "--risk-level",
        "increased",
        "--reward-mode",
        "ReT_unified_v1",
        "--observation-version",
        "v5",
        "--policy",
        "heuristic_cycle_guard",
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.run(
        cmd,
        cwd="/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia",
        check=True,
        capture_output=True,
        text=True,
    )

    observations = np.load(tmp_path / "observations.npy")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    env_spec = json.loads((tmp_path / "env_spec.json").read_text(encoding="utf-8"))

    assert metadata["observation_version"] == "v5"
    assert metadata["policy"] == "heuristic_cycle_guard"
    assert metadata["obs_shape"][1] == 30
    assert observations.shape[1] == 30
    assert env_spec["observation_version"] == "v5"
    assert env_spec["observation_fields"][-6:] == [
        "op1_cycle_phase_norm",
        "op2_cycle_phase_norm",
        "workweek_phase_sin_norm",
        "workweek_phase_cos_norm",
        "workday_phase_sin_norm",
        "workday_phase_cos_norm",
    ]


def test_export_trajectories_preserves_direct_garrido_action_context(
    tmp_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "scripts/export_trajectories_for_david.py",
        "--episodes",
        "1",
        "--seed-start",
        "11",
        "--risk-level",
        "increased",
        "--reward-mode",
        "ReT_unified_v1",
        "--observation-version",
        "v4",
        "--policy",
        "garrido_cf_s2",
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.run(
        cmd,
        cwd="/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia",
        check=True,
        capture_output=True,
        text=True,
    )

    direct_context = np.load(tmp_path / "direct_action_context.npy")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    direct_fields = json.loads(
        (tmp_path / "direct_action_context_fields.json").read_text(encoding="utf-8")
    )

    assert metadata["policy"] == "garrido_cf_s2"
    assert metadata["uses_direct_des_actions"] is True
    assert direct_context.shape[1] == len(direct_fields["fields"])
    assert "assembly_shifts" in direct_fields["fields"]
    assert np.isfinite(direct_context[:, 0]).all()
    assert float(np.nanmean(direct_context[:, 0])) == 2.0
