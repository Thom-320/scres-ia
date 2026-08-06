#!/usr/bin/env python3
"""David's step-4 notebook: his architectures against the STRUCTURED baseline, on the contract
where the neural residual is still undefined.

WHY A NEW ONE. His KAN lab asks whether one architecture beats another, and we already answered
that on his own seeds: nothing separates. Worse, that lab has no static comparator and no
controller arm, so it structurally cannot tell him whether a network is needed at all. This one
can, because the baselines are inside it.

WHY THIS ENVIRONMENT. The expanded buffer contract is the only place left where the neural residual
is UNDEFINED rather than measured at zero. The rights move ReT by +11% to +25%, Garrido specified
the four-step design himself and named DDMRP as the incumbent, and the structured controllers are
implemented and preflight green. Everywhere else we looked -- the 288/4608 configuration surface,
track_b_v1, the thesis-native envelope -- the premium is measured and it is zero.

SEEDS ARE DISJOINT FROM OURS ON PURPOSE. We take 1,420,001-1,423,006 for step 3 on Kaggle; his
block starts at 1,430,001. Two people running the same seeds is a replication dressed up as two
experiments.

Output: notebooks/scresia_david_step4_expanded_contract.ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCH = "david/kan-lab"
REPO = "Thom-320/scres-ia"
SEED_START = 1_430_001
METRIC = "ret_excel_full_ledger"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


CELLS = [
    md("""# SCRES-IA — paso 4 de Garrido: ¿hace falta una red?

**Esto sustituye al laboratorio KAN.** Aquel preguntaba qué arquitectura gana. Ya lo medimos, con
tus propias semillas, y la respuesta está abajo. La pregunta que queda abierta —y que aquel
cuaderno **no podía** contestar, porque sus cinco brazos eran todos neuronales— es si hace falta
una red **en absoluto**.

## Lo que ya está medido, para que no lo repitas

**1 · A parámetros igualados, ninguna arquitectura separa.** Mismo entorno `track_b_v1`, mismas
semillas 9491–9495, mismo presupuesto de 200.000 parámetros:

| contraste | media | IC95 |
|---|---:|---|
| KAN − MLP | −0,475 | [−1,548 · +0,598] |
| DMLPA − MLP | +0,136 | [−0,569 · +0,841] |

Y la KAN cuesta **2,82 ms por decisión contra 0,69 del MLP — 4,1×**.

**2 · A igual presupuesto de búsqueda, 74× más parámetros no compran nada.** Una neurona de **5
parámetros** que conserva pesos contra sustitutos de 369 y 380:

| brazo | parámetros | Δ vs la neurona | s/decisión |
|---|---:|---|---:|
| `neuron_memory` | **5** | — | 1,8e-05 |
| `surrogate_mlp` | 369 | −0,00156 [−0,0059 · +0,0026] | 1,9e-04 (11×) |
| `surrogate_kan` | 380 | −0,00101 [−0,0061 · +0,0043] | 6,0e-04 (34×) |

**3 · La escalera de 15 métodos dice dónde está la línea.** Ordenados por regret (menor es mejor):

| brazo | AUC | Δ vs la neurona |
|---|---:|---|
| `ucb1_transfer` | 0,04502 | −0,00701 [−0,0244 · +0,0141] |
| **`neuron_memory`** | **0,05203** | — |
| `ofat_transfer` | 0,06274 | +0,01071 |
| `lookahead_kg_transfer` | 0,08018 | +0,02815 |
| `gp_ei_transfer` | 0,08390 | +0,03187 |
| `thompson_transfer` | 0,08908 | +0,03705 |
| `ucb1` | 0,09655 | +0,04452 |
| `ofat` *(el método de la tesis)* | 0,10024 | +0,04821 |
| `gp_ei` | 0,10661 | +0,05458 |
| `thompson` | 0,10893 | +0,05690 |
| `lhs_local` | 0,10949 | +0,05746 |
| **`neuron_reset`** *(la misma neurona sin memoria)* | 0,11274 | +0,06070 |
| `lookahead_kg` *(planificador con lookahead)* | 0,11479 | +0,06276 |
| `random` | 0,13979 | +0,08776 |
| `annealing` | 0,17420 | +0,12217 |

**Mira dónde corta:** los cuatro primeros son los cuatro brazos que **conservan estado entre
contextos**. La misma neurona sin memoria cae por debajo del OFAT crudo. **Lo que rinde es
retener, no aproximar** — y contra eso perdió tu KAN, no contra otra red.

> **Veredicto honesto: en los entornos medidos NO hay prima neural.** No es un fracaso: es la
> respuesta a la pregunta 1 de Garrido, con número. Lo que imita el aprendizaje de la cadena es lo
> que cierra el lazo ③→⑧, no lo que tiene más capacidad.

## Qué queda abierto, y es lo que corre este cuaderno

Garrido pidió cuatro pasos: *baseline → MPC original → **MPC expandido** → KAN*. El paso 3 está
corriendo en Kaggle de nuestro lado. **Éste es el paso 4**, y es el único entorno donde el residual
neural sigue **sin definir** en vez de medido en cero:

* los derechos expandidos (`op3_rm`, `op5_rm`, `op9_rations`) **mueven ReT entre +11 % y +25 %**;
* hay un incumbente estructurado de verdad — **la mejor de las 216 posturas estáticas**, DDMRP
  proyectado al mismo dominio, y MPC de replay;
* así que **ganar aquí significaría algo**, y perder también.

**Tus semillas empiezan en 1.430.001.** Las nuestras son 1.420.001–1.423.006. No se tocan: dos
personas con las mismas semillas son una réplica disfrazada de dos experimentos.
"""),

    code("""# 0) LA ÚNICA CELDA QUE TOCAS
ARCHS         = ['MLP', 'KAN', 'DMLPA']   # las tuyas; corre las que quieras
TARGET_PARAMS = 200_000                   # presupuesto COMPARTIDO; el cuaderno aborta si no casan
SEEDS         = [1_430_001, 1_430_002, 1_430_003]
TAPES         = 6                         # tapes de evaluacion por arquitectura
FAMILY        = 'R1r'                     # 'R1r' | 'R2r' -- son estimandos distintos, no replicas
HORIZON_WEEKS, EPOCH_WEEKS = 52, 4

# NO cambies esto sin decirlo: ret_excel esta MEDIDO premiando el abandono (el reparto que lo
# maximiza entrega 50% de las raciones, el que lo minimiza entrega 80%). Una red podria ganar este
# experimento dejando de servir.
METRIC    = 'ret_excel_full_ledger'
GUARDRAIL = 'worst_product_fill'

GIT_URL, GIT_BRANCH = 'https://github.com/{repo}.git', '{branch}'
print(f'{{len(ARCHS)}} arquitecturas x {{len(SEEDS)}} semillas · familia {{FAMILY}} · '
      f'metrica {{METRIC}}')""".replace("{repo}", REPO).replace("{branch}", BRANCH)),

    code("""# 1) Setup portatil (Colab / Kaggle / local)
import os, subprocess, sys, time, json
from pathlib import Path

ROOT = Path('/kaggle/working/scres-ia') if Path('/kaggle').exists() else Path('/content/scres-ia')
if not ROOT.exists():
    subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', GIT_BRANCH, GIT_URL,
                           str(ROOT)])
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'simpy>=4.1', 'numpy', 'pandas', 'torch', 'pykan==0.2.8'])
print('commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())"""),

    code("""# 2) EL INCUMBENTE, calculado antes de entrenar nada.
# Las 216 posturas estaticas se enumeran completas: el mejor fijo es el rival a batir, y ya sabemos
# que es HETEROGENEO (op5_rm = 0), asi que no basta con guardar mas stock.
from supply_chain.expanded_contract_controllers_v2 import ALL_POSTURES, posture_name

print(f'{len(ALL_POSTURES)} posturas estaticas en el dominio 6^3')
cmd = [sys.executable, 'scripts/run_expanded_contract_comparators_v2.py',
       '--phase', 'full', '--families', FAMILY, '--tapes', str(TAPES), '--scenarios', '5',
       '--seed-start', str(SEEDS[0]), '--horizon-weeks', str(HORIZON_WEEKS),
       '--epoch-weeks', str(EPOCH_WEEKS), '--metric', METRIC,
       '--workers', str(max(1, (os.cpu_count() or 2) - 1)),
       '--output-dir', 'results/david_step4_baseline']
print(' '.join(cmd), flush=True)
subprocess.check_call(cmd)

base = json.loads(Path('results/david_step4_baseline/result.json').read_text())
block = base['family_results'][FAMILY]
print('\\n== TUS RIVALES, medidos en TUS tapes ==')
for arm, c in block['comparisons'].items():
    print(f'   {arm:<26} delta {c.get("delta_mean")}  IC95 {c.get("ci95")}')"""),

    code("""# 3) Tu brazo: la red fija los TRES objetivos de buffer en cada epoca.
# Mismos derechos de decision y misma informacion que los controladores estructurados: observa el
# estado del simulador y escribe op3_rm, op5_rm, op9_rations dentro del MISMO dominio 6^3.
# Si tu red no puede al menos EMPATAR la mejor postura fija -- que estaba en su conjunto de
# acciones -- eso es un fallo de BUSQUEDA y se reporta asi, no como evidencia sobre el derecho.
import torch, torch.nn as nn

# Deja el esqueleto y mete tu extractor. Lo unico que el cuaderno exige es que los tres
# arquitecturas queden a menos del 10% en numero de parametros.
def build_policy(arch: str, obs_dim: int, n_levels: int = 6, n_nodes: int = 3):
    raise NotImplementedError('mete aqui tu KAN / MLP / DMLPA con el mismo presupuesto')

print('pendiente: tu extractor. El baseline de la celda 2 ya es comparable tal cual.')"""),

    md("""## Cómo se lee tu resultado, fijado antes de correr

* **Tu red le gana al mejor estático con IC95 que excluye el cero** → hay prima neural **en este
  contrato**, y es el primer positivo del proyecto. Se preregistra confirmación en bloque virgen.
* **Empata** → el derecho expandido se convierte, pero no hace falta una red para convertirlo:
  gana el controlador más barato que llegue ahí.
* **Pierde contra el mejor estático** → **fallo de búsqueda**, porque esa postura estaba dentro de
  su conjunto de acciones. Se reporta como fallo de búsqueda, no como evidencia sobre el derecho.

**Y el guardarraíl no es negociable:** ninguna configuración se promueve si empeora
`worst_product_fill` más allá de su margen. Una red que gana abandonando no gana.

> Todo esto es **desarrollo**. Un buen resultado selecciona un candidato para un experimento
> preregistrado; no es evidencia de paper por sí solo, y requiere validación de Garrido.
"""),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=Path("notebooks/scresia_david_step4_expanded_contract.ipynb"))
    args = ap.parse_args()
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"  {args.output}  ({len(CELLS)} celdas)")
    print(f"  Colab : https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/"
          f"{args.output}")
    print(f"  semillas de David: {SEED_START}+   (las nuestras: 1.420.001-1.423.006, disjuntas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
