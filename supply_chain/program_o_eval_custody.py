"""Fail-closed file custody helpers for Program O-R evaluation artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256_manifest(root: Path, paths: list[Path], manifest: Path) -> dict[str, str]:
    """Write a sorted relative-path manifest, refusing paths outside *root*."""
    root = root.resolve()
    manifest = manifest.resolve()
    rows: dict[str, str] = {}
    for path in sorted({Path(p).resolve() for p in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(f"custody input missing: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"custody input outside root: {path}") from exc
        if path == manifest:
            raise ValueError("a SHA-256 manifest cannot include itself")
        rows[relative.as_posix()] = sha256(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in rows.items()),
        encoding="utf-8",
    )
    return rows


def verify_sha256_manifest(root: Path, manifest: Path) -> dict[str, str]:
    """Recompute every listed digest; malformed, missing, duplicate, or extra rows fail."""
    root = root.resolve()
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"missing SHA-256 manifest: {manifest}")
    rows: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"malformed SHA-256 manifest row {line_number}")
        expected, relative_text = parts
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe SHA-256 manifest path: {relative_text}")
        key = relative.as_posix()
        if key in rows:
            raise ValueError(f"duplicate SHA-256 manifest path: {key}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"SHA-256 manifest path escapes root: {key}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"SHA-256 manifest input missing: {key}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {key}: expected {expected}, got {actual}"
            )
        rows[key] = actual
    if not rows:
        raise ValueError("empty SHA-256 manifest")
    return rows
