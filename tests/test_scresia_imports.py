from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_scresia_namespace_imports_from_repo_cwd() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "from scresia.supply_chain import MFSCSimulation; "
        "from scresia.supply_chain.config import SIMULATION_HORIZON; "
        "from scresia.scripts.run_thesis_faithful import THESIS_BACKBONE; "
        "assert MFSCSimulation is not None; "
        "assert SIMULATION_HORIZON == 161_280; "
        "assert THESIS_BACKBONE['protocol'] == 'thesis_1to1'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_scresia_namespace_imports_when_repo_folder_is_named_scresia(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    linked_repo = tmp_path / "scresia"
    os.symlink(repo_root, linked_repo, target_is_directory=True)
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "from scresia.supply_chain import MFSCSimulation; "
        "from scresia.supply_chain.config import SIMULATION_HORIZON; "
        "from scresia.scripts.run_thesis_faithful import THESIS_BACKBONE; "
        "assert MFSCSimulation is not None; "
        "assert SIMULATION_HORIZON == 161_280; "
        "assert THESIS_BACKBONE['protocol'] == 'thesis_1to1'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
