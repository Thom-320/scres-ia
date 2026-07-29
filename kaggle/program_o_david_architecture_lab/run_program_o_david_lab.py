from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


GIT_URL = "https://github.com/Thom-320/scres-ia.git"
GIT_BRANCH = "codex/program-o-ret-only-learner"
ROOT = Path("/kaggle/working/scres-ia")
EXECUTED = Path("/kaggle/working/david_sandbox_program_o_ret_executed.ipynb")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    run(["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_URL, str(ROOT)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    run([sys.executable, str(ROOT / "scripts/build_david_sandbox_notebook.py")], cwd=ROOT)
    os.environ.setdefault("DAVID_PRESET", "screen")
    run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=-1",
            "--output",
            str(EXECUTED),
            str(ROOT / "notebooks/david_sandbox_program_o_ret.ipynb"),
        ],
        cwd=ROOT,
    )
    print(f"Executed notebook: {EXECUTED}", flush=True)
    print(f"Bundles: {ROOT / 'outputs/david_sandbox'}", flush=True)


if __name__ == "__main__":
    main()
