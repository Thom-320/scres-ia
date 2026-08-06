#!/usr/bin/env python3
"""Emit the four parallel Kaggle kernels for Garrido's step 3 on the expanded buffer contract.

WHY FOUR AND WHY SHARDED THIS WAY. The run is 12 tapes x 5 scenarios x 2 families x 216 postures at
a 52-week horizon, and the cost is the DES itself. The shards differ ONLY in tape identity, so the
paired contrast -- arms against the best static posture ON THE SAME TAPE -- stays intact inside each
shard and the pooled analysis just concatenates per-tape rows. Sharding by family would have been
wrong on its own: R1r and R2r are different estimands, not replicates, so each family is sharded
within itself.

GPU IS OFF ON PURPOSE. The bottleneck is simpy in pure Python. Kaggle's GPU kernels hand out FEWER
vCPU than the CPU ones, so enabling it would cut the worker count and make the run slower.

THE METRIC IS NOT THE RUNNER'S DEFAULT. `--metric ret_excel_full_ledger` is passed explicitly on
every shard: ret_excel is measured to reward abandonment -- the split maximising it delivers 50% of
rations, the one minimising it delivers 80% -- so a controller could win this experiment by not
serving. f4 of the preregistration fails if any shard is left on the default.

Preregistration: docs/PREREGISTRO_PASO3_GARRIDO_MPC_EXPANDIDO_2026-08-06.md
Output: kaggle/step3_<shard>/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "thomaschisica"
REPO = "Thom-320/scres-ia"
BRANCH = "codex/expanded-contract-comparators-v2"
METRIC = "ret_excel_full_ledger"
HORIZON_WEEKS, EPOCH_WEEKS, SCENARIOS, TAPES_PER_SHARD = 52, 4, 5, 6

#: shard -> (family, seed_start). Tapes are the replicate axis; families are separate estimands.
SHARDS = {
    "s1_r1r_a": ("R1r", 1_420_001),
    "s2_r1r_b": ("R1r", 1_421_001),
    "s3_r2r_a": ("R2r", 1_422_001),
    "s4_r2r_b": ("R2r", 1_423_001),
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def notebook(shard: str, family: str, seed_start: int) -> dict:
    out_dir = f"results/step3_{shard}"
    cells = [
        md(f"""# Paso 3 de Garrido — shard `{shard}` ({family}, semillas {seed_start}+)

**MPC de replay y DDMRP proyectado contra las 216 posturas estáticas**, sobre el contrato expandido
de buffers (`op3_rm`, `op5_rm`, `op9_rations`).

Este shard corre **{TAPES_PER_SHARD} tapes x {SCENARIOS} escenarios** de la familia **{family}**,
horizonte {HORIZON_WEEKS} semanas, época {EPOCH_WEEKS}. Los cuatro shards difieren **sólo en qué
tapes tocan**; el contraste pareado vive dentro del shard y el análisis agrupado concatena filas.

**GPU apagada a propósito:** el cuello es el DES en Python puro, y los kernels GPU de Kaggle dan
menos vCPU.

**Métrica decisora `{METRIC}`, no el default del runner.** `ret_excel` está medido premiando el
abandono, así que un controlador podría ganar dejando de servir.

Preregistro: `docs/PREREGISTRO_PASO3_GARRIDO_MPC_EXPANDIDO_2026-08-06.md`
"""),
        code(f"""# 1) Repo + dependencias
import os, subprocess, sys, time, json, shutil
from pathlib import Path

ROOT = Path('/kaggle/working/scres-ia')
if not ROOT.exists():
    subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', '{BRANCH}',
                           'https://github.com/{REPO}.git', str(ROOT)])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'simpy>=4.1', 'numpy', 'pandas'])
print('commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
print('cpus  :', os.cpu_count())"""),
        code(f"""# 2) El shard. Todo lo que decide algo va explícito en la línea de comandos.
WORKERS = max(1, (os.cpu_count() or 2) - 1)
cmd = [sys.executable, 'scripts/run_expanded_contract_comparators_v2.py',
       '--phase', 'full',
       '--families', '{family}',
       '--tapes', '{TAPES_PER_SHARD}',
       '--scenarios', '{SCENARIOS}',
       '--seed-start', '{seed_start}',
       '--horizon-weeks', '{HORIZON_WEEKS}',
       '--epoch-weeks', '{EPOCH_WEEKS}',
       '--metric', '{METRIC}',
       '--workers', str(WORKERS),
       '--output-dir', '{out_dir}']
print(' '.join(cmd), flush=True)
t0 = time.time()
subprocess.check_call(cmd)
print(f'listo en {{time.time() - t0:.0f}}s')"""),
        code(f"""# 3) Lectura rápida y ZIP para enviar
res = json.loads(Path('{out_dir}/result.json').read_text())
print('claim_status:', res.get('claim_status'))
print('metric      :', res.get('metric'))
for fam, block in res.get('family_results', {{}}).items():
    print(f'\\n== {{fam}} · {{block.get("candidate_count")}} candidatos')
    for arm, c in block.get('comparisons', {{}}).items():
        ci = c.get('ci95', [None, None])
        print(f'   {{arm:<26}} delta {{c.get("delta_mean")}}  IC95 {{ci}}')

shutil.make_archive('/kaggle/working/step3_{shard}', 'zip', '{out_dir}')
print('\\nZIP -> /kaggle/working/step3_{shard}.zip')"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"language": "python",
                                                        "display_name": "Python 3",
                                                        "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def kernel_metadata(slug: str) -> dict:
    return {
        "id": f"{OWNER}/{slug}", "title": slug,
        "code_file": f"{slug.replace('-', '_')}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        # OFF deliberately: the bottleneck is CPU-bound simulation and Kaggle's GPU kernels give
        # fewer vCPU, so turning it on would REDUCE throughput.
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True,                    # the notebook clones the repository
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("kaggle"))
    args = ap.parse_args()

    made = []
    for shard, (family, seed_start) in SHARDS.items():
        slug = f"scresia-step3-{shard.replace('_', '-')}"
        folder = args.out / f"step3_{shard}"
        folder.mkdir(parents=True, exist_ok=True)
        meta = kernel_metadata(slug)
        (folder / meta["code_file"]).write_text(
            json.dumps(notebook(shard, family, seed_start), indent=1) + "\n")
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        made.append((shard, family, seed_start, folder, slug))

    print("\n  kernels listos (los cuatro corren EN PARALELO, semillas disjuntas):")
    for shard, family, seed_start, folder, slug in made:
        print(f"    {shard:<10} {family:<4} semillas {seed_start}+  {folder}")
    print("\n  publicar (necesita ~/.kaggle/kaggle.json):")
    for *_, folder, _ in made:
        print(f"    kaggle kernels push -p {folder}")
    print(f"\n  métrica decisora: {METRIC}  (NO el default ret_excel, que premia el abandono)")
    print("  al terminar, agrupar con: scripts/merge_step3_shards.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
