import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_full_notebook_exposes_requested_models_and_only_dev_tapes() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/program_o_david_model_lab_FULL.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for model in (
        "RECURRENT_PPO",
        "PPO_DMPLA",
        "DISCRETE_SAC_DMPLA",
        "CUSTOM",
    ):
        assert model in source
    assert "949100001" in source
    assert "949200001" in source
    assert "7480101" not in source
    assert "748100001" not in source
    assert "DavidLabEncoder" in source
    assert "POSITIONAL_MODE" in source


def test_all_notebook_code_cells_compile() -> None:
    path = ROOT / "notebooks/program_o_david_model_lab_FULL.ipynb"
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell{index}", "exec")
