from __future__ import annotations

import subprocess
from pathlib import Path

FORBIDDEN_PATH_SNIPPETS = (
    "/Users/thom",
    "/System/Volumes/Data",
    "GoogleDrive",
    "CloudStorage",
    "chisicathomas",
    "~/Desktop/Universidad_Codigo",
    "~/Downloads",
    "/Applications/LibreOffice.app",
    "/opt/miniconda",
    "/opt/homebrew",
)

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}


def test_repo_has_no_user_specific_absolute_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    tracked_files = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for relative_path in tracked_files:
        path = repo_root / relative_path
        if path.name == "test_repo_portability.py":
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in FORBIDDEN_PATH_SNIPPETS:
            if snippet in text:
                offenders.append(f"{path.relative_to(repo_root)} contains {snippet}")
    assert offenders == []
