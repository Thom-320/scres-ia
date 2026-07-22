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


def test_run_all_defaults_are_serious_and_multiseed() -> None:
    source = _all_source()
    assert 'os.environ.get("DAVID_C6B_PROFILE", "serious")' in source
    assert '"serious": dict(timesteps=200_192' in source
    assert "optimizer_seeds=[9201, 9202, 9203]" in source
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
    assert "C6B_DEVELOPMENT_ONLY_NOT_PROMOTABLE" in source
    assert "SMOKE_ONLY_NO_SCIENTIFIC_CONCLUSION" in source
    assert "C6B_DEVELOPMENT_PASS_TO_PREREGISTRATION" in source
    assert "C6B_DEVELOPMENT_NO_GO_UNDER_TESTED_ENVELOPE" in source
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
