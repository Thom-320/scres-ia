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

# What a replicator RUNS has to be portable: code, configuration, documentation.
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}

# SCOPED 2026-07-31, and the scoping is the scientific point rather than a convenience.
#
# `.json` and `.csv` under `results/` and `research/` are EVIDENCE. A sealed artifact records
# the machine, paths and workbooks its run actually used -- that is provenance, and rewriting
# it to look portable would be falsifying a custody record to please a linter. Eleven of them
# carry a `self_sha256`/`content_sha256` over their own bytes, so editing them would break the
# seal outright.
#
# So evidence is exempt from the path rule and subject to a stricter one instead: it must not
# be edited at all. What remains in scope is exactly the replication bundle Submission A
# needs -- every tracked `.py`, `.toml`, `.yaml`, `.yml` and `.md`.
EVIDENCE_EXEMPT_SUFFIXES = {".csv", ".json", ".txt"}


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
