#!/usr/bin/env python3
"""Emit the Kaggle kernels for the architecture arms David is not running himself.

WHY KAGGLE HERE AND NOT FOR THE CONFIRMATION. These arms are development: they open no virgin
seeds, and a session that dies costs a restart rather than an irreplaceable block. That is the
opposite of the grid-transfer confirmation, which stays local precisely because a 9-hour cap plus
session death would produce a partial confirmation this project refuses to rescue.

GPU IS OFF ON PURPOSE. The bottleneck is the DES -- simpy, pure Python, CPU -- and no GPU touches
it. Kaggle's GPU kernels also hand out FEWER vCPU than the CPU ones, so enabling it would cut the
number of parallel environments and make the run slower. Measured: 105 steps/s with three parallel
envs at HISTORY_LEN=16, which puts David's 1,000,000-step default at ~4.4 h against the 9 h cap.

A CAVEAT THAT MUST TRAVEL WITH THE RESULTS. `ms_per_decision` is not comparable across machines.
Splitting arms between Kaggle and a laptop keeps the QUALITY comparison valid -- same environment,
same seeds, same budget -- but Delta_efficiency has to come from one machine or be reported only
for the arms that shared one.

Output: kaggle/david_kan_lab_<arch>/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ARCHS = ("MLP", "DMLPA")
OWNER = "thomaschisica"
BUILDER = Path("scripts/build_david_kan_lab_notebook.py")


def kernel_metadata(arch: str, slug: str) -> dict:
    return {
        "id": f"{OWNER}/{slug}",
        "title": slug,
        "code_file": f"{slug.replace('-', '_')}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        # OFF deliberately: see the module docstring. The bottleneck is CPU-bound simulation and
        # Kaggle's GPU kernels give fewer vCPU, so turning it on would REDUCE throughput.
        "enable_gpu": False,
        "enable_tpu": False,
        # Required: the notebook sparse-clones the repository (9 MB of 860).
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("kaggle"))
    args = ap.parse_args()

    made = []
    for arch in ARCHS:
        slug = f"scresia-david-kan-lab-{arch.lower()}"
        folder = args.out / f"david_kan_lab_{arch.lower()}"
        folder.mkdir(parents=True, exist_ok=True)
        meta = kernel_metadata(arch, slug)
        notebook = folder / meta["code_file"]
        subprocess.check_call([sys.executable, str(BUILDER), "--arch", arch,
                               "--output", str(notebook), "--quiet"])
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        made.append((arch, folder, slug))

    print("\n  kernels listos:")
    for arch, folder, slug in made:
        print(f"    {arch:<6} {folder}  ->  kaggle.com/{OWNER}/{slug}")
    print("\n  para publicarlos (necesita ~/.kaggle/kaggle.json):")
    for _, folder, _ in made:
        print(f"    kaggle kernels push -p {folder}")
    print("\n  ms_per_decision NO es comparable entre máquinas. Si estos dos corren en Kaggle y")
    print("  KAN corre en otra parte, la comparación de CALIDAD sigue siendo válida pero")
    print("  Delta_efficiency sólo se reporta dentro de la máquina que compartieron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
