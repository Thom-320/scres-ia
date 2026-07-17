"""Static fail-closed checks for David's Program O-R development notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "david_sandbox_program_o_ret.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def test_all_code_cells_compile() -> None:
    for index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_david_architectures_and_audit_are_visible() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    required = (
        "class FriendDMLPAFaithful",
        "class FriendDMLPAPositional",
        "FEATURE EXTRACTOR COMPLETO",
        "total_parameters",
        "build_sac_discrete_dmlpa",
        "RecurrentPPO",
        "HistoryStackWrapper",
    )
    for token in required:
        assert token in source


def test_default_is_bounded_multiseed_preliminary_run() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    assert 'PRESET = "preliminary"' in source
    assert '"preliminary": dict(total_timesteps=50_000' in source
    assert "optimizer_seeds=[9201, 9202, 9203]" in source
    assert "eval_tapes_per_cell=12" in source
    assert 'MODEL_KINDS_TO_RUN = ["ppo_dmlpa_positional"]' in source


def test_scientific_seed_ranges_are_rejected() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    assert "(747000000, 748999999)" in source
    assert "(7480001, 7480999)" in source
    assert "assert_dev_seed" in source


def test_classical_comparator_is_selected_by_panel_mean() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    assert "classical_ret.mean(axis=0).argmax()" in source
    assert "best_classical_index" in source
    assert "regime_persistence=0.75, dominant_share=0.90" in source
