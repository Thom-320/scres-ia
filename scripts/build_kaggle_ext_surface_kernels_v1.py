#!/usr/bin/env python3
"""Two Kaggle kernels that take six of the forty extended-surface partitions.

WHY KAGGLE GETS THE SURFACE AND NOT THE SCIENCE. The comparator-repair run is the scientific
critical path, but it is a single-process loop whose `marginal` and `loo_marginal` arms accumulate
their histogram ACROSS seeds -- sharding it by seed would change the very object under study -- and
a Kaggle vCPU is slower than an M1 core, so it would finish later there than here. The extended
surface is the opposite: embarrassingly parallel by slice, twelve hours deep, and blocking nothing.

WHY THE SHARDS ARE COMMITTED FIRST. `run_surface` skips a slice whose file already exists, but it
builds that list at start-up from the local filesystem. Kaggle gets a fresh clone, so unless the
finished slices are in git it would recompute all of them. They are 2 MB for 504 files.

WHY A REPARTITION AND NOT JUST "ALSO RUN IT". All forty partitions are claimed -- local 0-29, VPS
30-39 -- and two pools on overlapping partitions duplicate work rather than share it. Local drops to
0-23 at the same moment Kaggle takes 24-29, which also frees local cores for the repair run. The
only loss is the up-to-eight slices in flight when local restarts.

A THIRD ARCHITECTURE IS A BONUS, NOT A RISK. The base surface reproduced bit-exactly from macOS
arm64 to Linux x86_64. Kaggle is a third platform, and the tolerance rule frozen in
`docs/ENMIENDA_TOLERANCIA_EQUIVALENCIA_CROSS_PLATFORM_2026-08-08.md` was written before any of these
results were seen, so a last-bit difference is adjudicated rather than argued about.

Output: kaggle/ext_surface_<n>/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "thomaschisica"
REPO = "Thom-320/scres-ia"
BRANCH = "codex/expanded-contract-comparators-v2"
OF = 40

#: kernel slug -> the partitions it owns. Disjoint from local (0-23) and the VPS (30-39).
KERNELS = {
    "scresia-ext-surface-a": (24, 26, 28),
    "scresia-ext-surface-b": (25, 27, 29),
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def notebook(slug: str, parts: tuple[int, ...]) -> dict:
    shard_arg = ",".join(str(p) for p in parts)
    cells = [
        md(f"""# Superficie extendida — particiones {shard_arg} de {OF}

Replay de verificación de `garrido_transfer_confirmation_v2_ext`: cada rebanada se recalcula celda a
celda y se compara **bit a bit** contra el caché sellado. No abre semillas, no entrena, no adjudica
nada científico — es procedencia.

Las cuarenta particiones están repartidas: local 0-23, este par de kernels 24-29, VPS 30-39. Son
disjuntas a propósito; dos pools sobre particiones solapadas duplican trabajo en vez de repartirlo.

Las rebanadas ya terminadas viajan en el repositorio, así que este kernel se las salta y sólo calcula
lo que falta.

Regla de tolerancia: `docs/ENMIENDA_TOLERANCIA_EQUIVALENCIA_CROSS_PLATFORM_2026-08-08.md`,
congelada antes de ver ningún resultado de la extendida.
"""),
        code(f"""# 1) Repo y dependencias
import os, subprocess, sys, time, json, shutil, platform
from pathlib import Path

ROOT = Path('/kaggle/working/scres-ia')
if not ROOT.exists():
    subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', '{BRANCH}',
                           'https://github.com/{REPO}.git', str(ROOT)])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'simpy>=4.1', 'numpy', 'pandas', 'scipy', 'scikit-learn'])

# The host identity travels with the result: a bit-exact claim across architectures is worth
# nothing if nobody recorded which architectures.
env = {{'platform': platform.platform(), 'machine': platform.machine(),
       'python': platform.python_version(), 'cpus': os.cpu_count(),
       'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()}}
print(json.dumps(env, indent=1))
Path('/kaggle/working/host_env_{slug}.json').write_text(json.dumps(env, indent=1))
done = len(list(Path('results/frozen_path_equivalence_v2/shards').glob('ext__*.json')))
print(f'rebanadas ext ya en el repo: {{done}}/360')"""),
        code(f"""# 2) Una partición por proceso. Independientes: en Linux esto es fork y no hay
#    ProcessPoolExecutor de por medio, que es lo que colgaba en macOS.
parts = {list(parts)!r}
procs = []
for p in parts:
    log = open(f'/kaggle/working/ext_{{p}}.log', 'w')
    procs.append(subprocess.Popen(
        [sys.executable, '-u', 'scripts/verify_frozen_path_equivalence_v2.py',
         '--phase', 'surface', '--surface', 'ext', '--of', '{OF}', '--shards', str(p)],
        stdout=log, stderr=subprocess.STDOUT))
print(f'lanzados {{len(procs)}} procesos sobre particiones {{parts}}', flush=True)

t0 = time.time()
while any(pr.poll() is None for pr in procs):
    time.sleep(120)
    n = len(list(Path('results/frozen_path_equivalence_v2/shards').glob('ext__*.json')))
    print(f'  [{{(time.time()-t0)/60:.0f}} min] ext en disco: {{n}}/360', flush=True)
print('codigos de salida:', [pr.returncode for pr in procs])"""),
        code(f"""# 3) Sólo las rebanadas nuevas vuelven, con su diagnóstico de diferencias.
out = Path('/kaggle/working/ext_shards_{slug}')
out.mkdir(exist_ok=True)
mismatched = []
for f in Path('results/frozen_path_equivalence_v2/shards').glob('ext__*.json'):
    d = json.loads(f.read_text())
    if d.get('mismatches'):
        mismatched.append({{'file': f.name, 'mismatches': d['mismatches'],
                           'max_abs_delta': d.get('max_abs_delta')}})
    shutil.copy(f, out / f.name)
print(f'rebanadas empaquetadas: {{len(list(out.glob("*.json")))}}')
print(f'con diferencias: {{len(mismatched)}}')
for m in mismatched[:10]:
    print(' ', m)
shutil.make_archive('/kaggle/working/ext_shards_{slug}', 'zip', out)
print('ZIP -> /kaggle/working/ext_shards_{slug}.zip')"""),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                        "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def kernel_metadata(slug: str) -> dict:
    return {
        "id": f"{OWNER}/{slug}", "title": slug,
        "code_file": f"{slug.replace('-', '_')}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        # OFF on purpose: the work is pure-Python cell replay and Kaggle's GPU kernels hand out
        # fewer vCPU, so enabling it would reduce throughput.
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("kaggle"))
    args = ap.parse_args()
    for slug, parts in KERNELS.items():
        d = args.out / slug.replace("-", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{slug.replace('-', '_')}.ipynb").write_text(
            json.dumps(notebook(slug, parts), indent=1) + "\n")
        (d / "kernel-metadata.json").write_text(
            json.dumps(kernel_metadata(slug), indent=1) + "\n")
        print(f"{d}  particiones {parts}")
    print(f"\nempujar con:  kaggle kernels push -p kaggle/<dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
