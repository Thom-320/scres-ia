# Diagnóstico de los 38 fallos de la suite

**Fecha:** 2026-08-24 · **Suite base:** 2260 passed, 38 failed, 7 skipped, 2 xfailed (1338 s).

Los 38 fallos no son un único problema. Son **cuatro clases distintas**, y solo una
toca la ciencia.

## Clase A — Entorno: `.venv` ausente (20 fallos) · CORREGIDA

**Síntoma:** `FileNotFoundError: '.venv/bin/python'` y
`No module named pip` al ejecutar `python -I -B -m pip freeze --all`.

**Causa verificada:** los harnesses de custodia invocan el intérprete por la ruta
relativa `.venv/bin/python` y toman una huella de dependencias con `pip freeze`.
El repositorio no tenía `.venv`, y el entorno real
(`/home/ubuntu/.venv-scres-cert`) fue creado con `uv`, que no instala `pip`.

**Corrección aplicada:**

1. `uv pip install pip` en el entorno de certificación.
2. `.venv` como symlink a `/home/ubuntu/.venv-scres-cert`.

`.venv/` ya está en `.gitignore`, así que esto no entra en el repositorio.

**Verificado:** de los 20 fallos de este grupo, **13 pasan ahora**. Quedan 7 que
son de la clase C.

**Por qué symlink y no un venv nuevo:** reinstalar las dependencias en un venv
independiente falló dos veces (`torch==2.13.0+cpu` y `packaging==26.3` no
resolvían en el índice por defecto). Más importante: **un venv distinto tendría
huellas de dependencia distintas**, que es justo lo que estos tests certifican.
El symlink garantiza que la huella sea la del entorno que realmente ejecuta la
ciencia.

## Clase B — Clasificación de estado del simulador (10 fallos) · CORREGIDA

**Síntoma:** `Markov-completeness runtime audit failed: simulator live-field
classification is incomplete; live simulator read is unclassified`.

**Causa verificada:** 18 atributos nuevos del simulador —extensiones de conmutación
CSSU, de expedición y de topología LOC— no estaban asignados a ningún rol
científico. El auditor falla cerrado ante cualquier atributo sin clasificar, que
es su comportamiento correcto: un campo mutable fuera de la clave de Markov es
exactamente lo que produce una fusión falsa de estados.

**Corrección aplicada** en `scripts/run_paper2_bottleneck_exact_transducer.py`:

- **8 campos a `IMMUTABLE_CONTRACT_FIELDS`** — configuración de ejecución asignada
  una sola vez en `__init__` (`supply_chain.py:507-531`) y nunca reasignada:
  `cssu_min_dwell_days`, `cssu_switch_cost_rations`, `expedite_budget_hours`,
  `expedite_reduction_hours`, `expedite_charge_hours`, `loc_topology_mode`,
  `loc_graph`, `_cssu_capacity_ledger`.
- **10 campos a `INERT_FROZEN_FIELDS`** — mutables escritos por código vivo, que se
  pliegan a la clave de forma conservadora: `cssu_switch_count`,
  `cssu_switch_cost_paid`, `cssu_switch_cost_unpaid`, `cssu_blocked_by_dwell_count`,
  `expedite_budget_remaining`, `expedite_events`, `loc_arcs_down`,
  `loc_arc_down_events`, `_cssu_last_switch_at`, `_pending_expeditions`.

**No basta con clasificar: hay que demostrarlo.** Añadí 9 invariantes que
verifican que esos campos están realmente inertes bajo este contrato
(`cssu_dwell_inert`, `expedite_budget_zero`, `loc_topology_serial`, etc.). Si un
contrato futuro los activa, la auditoría falla ruidosamente en lugar de dejar
estado vivo fuera de la clave en silencio.

**Verificado:** `classification_complete=True`, sin atributos sin clasificar, sin
solapamientos, y los 9 invariantes nuevos pasan.

## Clase C — Anchors de contenido desactualizados (7 fallos) · NO CORREGIDA

Aquí es donde hay que tener cuidado, porque **la corrección fácil es la
equivocada**.

**Síntoma:** hashes dorados que no coinciden.

| Test | Esperado | Obtenido |
|---|---|---|
| `test_cssu_capacity_bridge` | `f3fe61b1…` | `9cb65c7a…` |
| `test_loc_graph_bridge` | `371c5975…` | `eb65748e…` |
| `test_program_j_..._structure_audit` | `d8fd9347…` | `2f348e59…` |

**Lo verificado hasta ahora:**

- Los ficheros de física (`supply_chain.py`, `episode_metrics.py`,
  `scientific_payload.py`, `config.py`) **no han cambiado** desde el commit que
  congeló el hash CSSU (`3b70dd9`). `git diff 3b70dd9..HEAD` sobre ellos está vacío.
- El binding del audit de Program J apunta a la versión de `supply_chain.py` del
  commit `5cb8fb8` (2026-07-31). Después llegaron `84937de`, `646c65a`, `852c64b`
  y `3b70dd9`, con 216 líneas añadidas y 6 eliminadas.
- Los otros 8 bindings de ese audit **siguen correctos**; solo deriva
  `supply_chain.py`.
- Los `result_bindings` (`verdict.json`, `raw_matrices.npz`) **también siguen
  correctos**: los resultados no se han tocado.

**Lo que esto significa:** el audit de Program J ata el *hash del fuente*, no el
comportamiento. El fuente cambió legítimamente al añadir extensiones posteriores.
Que los resultados sigan coincidiendo es evidencia de que el comportamiento no
cambió, pero no es una prueba.

**Por qué no re-congelo los hashes:** re-congelar un anchor es exactamente la
acción que el propio test prohíbe sin justificación —«re-freeze in a commit that
says why»—. Hacerlo yo convertiría un guardián en decoración. Esta decisión es
del PI, y necesita evidencia de que la física no cambió, no solo de que el fuente
sí.

**Experimento en curso:** ejecutar `test_cssu_capacity_bridge` en un worktree del
propio commit `3b70dd9` que congeló el hash. Si allí también falla, el hash se
congeló mal desde el principio. Si allí pasa, algo posterior sí alteró el
comportamiento y hay que encontrarlo.

## Clase D — Artefactos ausentes (1 fallo)

`test_paper2_exhaustive_search_registry` referencia ficheros que no existen en el
árbol, por ejemplo
`outputs/experiments/track_b_same_contract_challenge_2026-07-10/summary.json`.
Es un problema de custodia de artefactos, no de código.

## Resumen

| Clase | Fallos | Estado |
|---|---|---|
| A — entorno `.venv`/`pip` | 20 | corregida (13 verificados; 7 son clase C) |
| B — clasificación de estado | 10 | corregida y verificada |
| C — anchors desactualizados | 7 | **decisión del PI, no la tomo yo** |
| D — artefactos ausentes | 1 | pendiente de custodia |
