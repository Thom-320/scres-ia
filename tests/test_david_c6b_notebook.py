from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "scresia_david_C6B_physical_perbatch_FINAL.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def _all_source() -> str:
    return "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])


def test_all_code_cells_compile() -> None:
    for index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"c6b-cell-{index}")


def test_run_all_defaults_are_serious_and_retained_across_phases() -> None:
    source = _all_source()
    assert 'os.environ.get("DAVID_C6B_PROFILE", "serious")' in source
    assert 'TRAINING_CONTINUITY = "retained_across_phases"' in source
    assert '"serious": dict(timesteps_per_phase=200_192, phases=3' in source
    assert "MODEL_INITIALIZATION_SEED = 9201" in source
    assert "FRAME_STACK = 24" in source


def test_notebook_encodes_the_causal_c6b_contract() -> None:
    source = _all_source()
    assert "24 epochs físicos distintos" in source
    assert "all(b > a for a, b in zip(times, times[1:]))" in source
    assert "action_space = spaces.Discrete(2)" in source
    assert "incremental/vector OAT parity failure" in source
    assert "incremental/vector backlog-counter parity failure" in source


def test_all_relevant_comparators_and_memory_ablation_are_present() -> None:
    source = _all_source()
    for name in (
        "recurrent_ppo_mlp",
        "ppo_dmlpa_stack24",
        "ppo_dmlpa_stack1",
        "recurrent_ppo_dmlpa_stack24",
        "sac_discrete_dmlpa_stack24",
    ):
        assert name in source
    assert "DiscreteSACAgent" in source
    assert "BoxToDiscrete" not in source
    assert "src_key_padding_mask=~valid" in source
    assert "memory_delta_LCB05" in source


def test_claim_and_custody_guards_are_visible() -> None:
    source = _all_source()
    assert "C6B_RETAINED_TRAJECTORY_DEVELOPMENT_ONLY" in source
    assert "SMOKE_ONLY_NO_SCIENTIFIC_CONCLUSION" in source
    assert "OBSERVED_RETAINED_WIN_REQUIRES_INDEPENDENT_REPLICATION" in source
    assert "NO_GO_ON_SINGLE_RETAINED_TRAJECTORY_UNDER_TESTED_ENVELOPE" in source
    assert "Requiere validación de Garrido" in source
    assert "assert_dev_tape" in source
    assert "No cambies" in source
    assert "PUEDES EDITAR AQUÍ" in source


def test_notebook_is_verbose_and_explains_progress_to_operator() -> None:
    source = _all_source()
    assert "NOTEBOOK 6 · PLAN DE EJECUCIÓN" in source
    assert "TRABAJO {job_number}/{total_jobs}" in source
    assert "SIGUE CORRIENDO" in source
    assert "ETA aproximado restante" in source
    assert "RESULTADOS MEDIOS POR MODELO Y CELDA" in source
    assert 'verbose=1' in source


def test_notebook_interprets_learning_and_builds_sendable_audit_zip() -> None:
    source = _all_source()
    assert "random_binary" in source
    assert "learned_signal_vs_random_all_cells" in source
    assert "RESUMEN_PARA_THOMAS.txt" in source
    assert "REPORTE_VISUAL_PARA_PANTALLAZO.html" in source
    assert "C6B_AUDITORIA_PARA_ENVIAR" in source
    assert "files.sha256" in source
    assert "AUTO_DOWNLOAD_AUDIT" in source
    assert "google.colab import files" in source
    assert "c6b-download" in source


def test_agent_is_built_once_and_state_is_retained_between_phases() -> None:
    source = _all_source()
    assert "# ÚNICA construcción" in source
    assert "for phase_id in PHASE_IDS" in source
    assert "assert id(agent) == agent_identity" in source
    assert "assert optimizer_identity(agent) == retained_optimizer_identity" in source
    assert "assert id(agent.replay_buffer) == retained_buffer_identity" in source
    assert "assert before_sha == previous_phase_sha" in source
    assert "reset_num_timesteps=(phase_id == 1)" in source
    assert "steps_before + TIMESTEPS_PER_PHASE" in source
    assert "agents[(kind, optimizer_seed)]" not in source


def test_single_retained_trajectory_is_not_misreported_as_three_seeds() -> None:
    source = _all_source()
    assert "INDEPENDENT_INITIALIZATIONS = 1" in source
    assert "ROBUSTNESS_GATE_ELIGIBLE = INDEPENDENT_INITIALIZATIONS >= 3" in source
    assert "observed_goal_all_cells_single_trajectory" in source
    assert 'row["strong_cell_pass"] = bool(row["observed_cell_win"] and ROBUSTNESS_GATE_ELIGIBLE)' in source
    assert "tape_bootstrap_lcb" in source
    assert "two_way_lcb" not in source
