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
        "DiscreteSACAgent",
        "RecurrentPPO",
        "HistoryStackWrapper",
        "source_sha256",
        "source_origin",
        "notebook_class_ast",
        "never hash a class repr as if it were code",
    )
    for token in required:
        assert token in source


def test_default_is_bounded_multiseed_screen() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    assert 'os.environ.get("DAVID_PRESET", "screen")' in source
    assert '"screen": dict(total_timesteps=50_000' in source
    assert "optimizer_seeds=[9201, 9202, 9203]" in source
    assert "eval_tapes_per_cell=12" in source
    for model in (
        "ppo_mlp", "ppo_mlp_history", "recurrent_ppo",
        "ppo_dmlpa_faithful", "ppo_dmlpa_positional",
        "sac_discrete_dmlpa_faithful", "sac_discrete_dmlpa_positional",
    ):
        assert f'"{model}"' in source


def test_scientific_seed_ranges_are_rejected() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    assert "949100001" in source
    assert "949299999" in source
    assert "el sandbox solo admite namespaces 9491*/9492*" in source
    assert "assert_dev_seed" in source


def test_classical_comparator_is_selected_by_panel_mean() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    assert "classical_ret.mean(axis=0).argmax()" in source
    assert "best_classical_index" in source
    assert "regime_persistence=0.75, dominant_share=0.90" in source
