#!/usr/bin/env python3
"""Locate the external primary sources by CONTENT, never by one machine's absolute path.

Garrido-Ríos's thesis, his 2024 paper and the three canonical workbooks are copyrighted
primary sources: they cannot live in the repository, but every claim we make rests on them, so
their identity has to be checkable. Until 2026-07-31 that was done by pinning absolute paths
under one user's home directory. Two failure modes followed, both observed:

* **A moved file reads as a corrupted one.** `reproducibility_manifest.json` pinned
  `.../Supernote/Document/20_RESEARCH/PhD-Papers/garrido2024 scres+AI.pdf`; the folder was
  renamed to `01_RESEARCH`, the check reported the source as missing, and one test stayed red
  for weeks over a rename. The bytes never changed -- sha256 `3e3bc8f8…` is identical at both
  locations.
* **The repository stopped being portable.** Those pins are a chunk of the 90 user-specific
  absolute paths `tests/test_repo_portability.py` fails on, and they would make the Submission
  A replication bundle unusable on any other machine.

So identity is the hash and the filename; WHERE the file sits is environment. Resolution order:

1. `$SCRES_EXTERNAL_SOURCES` -- colon-separated directories, searched recursively. Set this if
   your copies live somewhere unusual.
2. A short list of home-relative roots, bounded in depth.

`verify()` returns one of three statuses per source, and the distinction is the point:
`verified` (found, hash matches), `mismatch` (found, hash differs -- a real failure), and
`unavailable` (not on this machine). **Unavailable is not pass and not fail: it is unverified,
and must be reported as such.** A check that turns "I could not look" into "I looked and it was
fine" is the exact failure mode this project keeps finding in its own audits.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

# filename -> sha256. No paths: the content is the identity.
EXTERNAL_SOURCES: dict[str, str] = {
    "Raw_data1+Re.xlsx":
        "30b88c9b9fe68ef527dbfcc70d8e653ea7bd152ab891b3fc0ecf53cb6f043486",
    "Raw_data2+Re.xlsx":
        "4bd462771fefff16fc5666a851256b3780198d474832dec1423c0b6f94be86b0",
    "Rsult_1.xlsx":
        "1901f683f6014cf75237c17233b8eba04f541b956f2d19dcecf2edc00e83b00a",
    "garrido et al 2024 factory resilience.pdf":
        "1260863dc295232faf24b820e1f67d53f25f81ffa2d221f7ef02a02310519c43",
    "garrido2024 scres+AI.pdf":
        "3e3bc8f82e20b891ee163fb8a035dd37be4312fa11f58dde77452dc1bb903ae6",
    "WRAP_Theses_Garrido_Rios_2017.pdf":
        "de9192d233b0c728ece6156b754fc64543146868121358b8a95c73b3edaa55cf",
    "v.0_neuralNet-scres.docx":
        "b111070a05c8f4d1afa058454138bed9b4b74900ab87eaaf6eb5186b6e8293f2",
    "v.0_neuralNet-scres.pdf":
        "521b12770e94f3e70c4c88ce1e38613f4e0aad3e1dab114632c9c89dbfad182d",
}

ENV_VAR = "SCRES_EXTERNAL_SOURCES"
MAX_DEPTH = 9  # cloud mounts nest deeply: Library/<mount>/<account>/My Drive/a/b/c/file


def search_roots() -> list[Path]:
    """Directories to search, most specific first. Nothing here is hard-coded to a user."""
    roots: list[Path] = []
    for entry in os.environ.get(ENV_VAR, "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry.strip()).expanduser())
    home = Path.home()
    roots.extend([home / "Downloads", home / "Documents", home / "Library"])
    return [root for root in roots if root.is_dir()]


def _find(filename: str, root: Path) -> Path | None:
    """Depth-bounded search. `Library` holds cloud mounts and must not be walked forever."""
    base = len(root.parts)
    for candidate in root.rglob(filename):
        if len(candidate.parts) - base <= MAX_DEPTH and candidate.is_file():
            return candidate
    return None


def resolve(filename: str) -> Path | None:
    for root in search_roots():
        try:
            found = _find(filename, root)
        except (OSError, PermissionError):  # cloud placeholders, protected folders
            continue
        if found is not None:
            return found
    return None


def verify(filename: str | None = None) -> dict[str, dict]:
    """`verified` / `mismatch` / `unavailable` for each external source."""
    wanted = {filename: EXTERNAL_SOURCES[filename]} if filename else EXTERNAL_SOURCES
    out: dict[str, dict] = {}
    for name, expected in wanted.items():
        path = resolve(name)
        if path is None:
            out[name] = {"status": "unavailable", "expected_sha256": expected,
                         "note": f"not found under {ENV_VAR} or the home-relative roots"}
            continue
        actual = sha256(path.read_bytes()).hexdigest()
        out[name] = {
            "status": "verified" if actual == expected else "mismatch",
            "expected_sha256": expected, "actual_sha256": actual,
            # The location is diagnostic output, never an input to the check.
            "found_at_depth": len(path.parts) - len(path.parents[len(path.parts) - 2].parts),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = verify()
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        for name, row in sorted(report.items()):
            print(f"  {row['status']:<12} {name}")
    return 1 if any(r["status"] == "mismatch" for r in report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
