#!/usr/bin/env python3
"""Rebuild the complete Submission A package and deterministic review PDF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "submission_a_program_q"
DEFAULT_OUTPUT = PAPER / "output" / "pdf" / "submission_a_program_q_draft.pdf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_submission_a_program_q.py")],
        cwd=ROOT,
        check=True,
    )
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise SystemExit("tectonic is required to build the review PDF")
    environment = os.environ.copy()
    # Freeze the review-build date to the publication-first evidence freeze.
    environment["SOURCE_DATE_EPOCH"] = "1784635200"
    with tempfile.TemporaryDirectory(prefix="submission-a-pdf-") as temporary:
        subprocess.run(
            [tectonic, "main.tex", "--outdir", temporary],
            cwd=PAPER,
            env=environment,
            check=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(temporary) / "main.pdf", args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
