#!/usr/bin/env python3
"""Refresh hashes in the curated Paper-2 reproducibility manifest fail-closed."""
from __future__ import annotations

from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "research" / "paper2_exhaustive_search" / "reproducibility_manifest.json"
PACKAGES = (
    "SALib",
    "gymnasium",
    "numpy",
    "pandas",
    "sb3-contrib",
    "scikit-learn",
    "scipy",
    "simpy",
    "stable-baselines3",
    "torch",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def resolve_artifact(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def refresh_hashes(rows: dict[str, str]) -> dict[str, str]:
    refreshed = {}
    missing = []
    for path_text in sorted(rows):
        path = resolve_artifact(path_text)
        if not path.is_file():
            missing.append(path_text)
            continue
        refreshed[path_text] = sha256(path)
    if missing:
        raise FileNotFoundError(f"Manifest-listed files are missing: {missing}")
    return refreshed


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    required = {
        "schema_version",
        "scientific_status",
        "paper2_confirmed",
        "paper3_authorized",
        "repository",
        "execution_scope",
        "seed_and_tape_status",
        "known_reproducibility_defects",
        "commands",
        "artifact_hashes",
        "source_hashes",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Curated manifest is missing required fields: {missing}")

    # Never reconstruct scientific status, commands, exclusions or claim limits.
    # Only fields that can drift mechanically are refreshed here.
    manifest["generated_date"] = date.today().isoformat()
    manifest["repository"]["branch"] = git("branch", "--show-current")
    manifest["repository"]["head_input"] = git("rev-parse", "HEAD")
    manifest["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: _package_version(package) for package in PACKAGES
        },
    }
    manifest["artifact_hashes"] = refresh_hashes(manifest["artifact_hashes"])
    manifest["source_hashes"] = refresh_hashes(manifest["source_hashes"])
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(MANIFEST)
    return 0


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
