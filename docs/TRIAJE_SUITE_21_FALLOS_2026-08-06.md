# Triaje — la suite pasó de 0 fallos el 31 de julio a 21 hoy

**Ninguno viene del trabajo de hoy.** Verificado en un worktree limpio sobre el tip de ayer
(`dd0faaa`): los mismos ficheros fallan allí, en igual o mayor número. La comparación es limpia para
ficheros versionados; el worktree base no tiene los `results/` sin commitear de hoy, y varios de
estos tests hashean artefactos, así que no es perfecta.

**Punto de partida:** `docs/`/memoria registran **27 → 0** tras la reparación de custodia del
31 de julio. Hoy: **21 fallos, 2.260 pasados**.

## 1. Arreglado

| test | causa | acción |
|---|---|---|
| `test_seed_custody_registry::test_seed_registry_is_fail_closed_before_submission_receipt` | el bloque `garrido_grid_transfer_v2_confirmation` tiene estado `BURNED_CONFIRMATION_COMPLETE`, introducido por `ed16b9e` y nunca añadido a la lista blanca del test | añadido, con la justificación escrita: el estado es **estrictamente más específico** que `BURNED` y no permite nada que `BURNED` no permitiera |

**Por qué importaba:** es el test que impide abrir semillas sin registro, y llevaba días en rojo por
el commit de un resultado legítimo. Hoy se abrió el bloque `1.420.001+` y **la verificación la hice
a mano con `check_seeds`, no el arnés.**

## 2. Decisión del PI — deriva de hash contra manifiesto de reproducibilidad

| test | qué falla |
|---|---|
| `test_paper2_exhaustive_search_registry::test_reproducibility_manifest_hashes_every_listed_artifact_and_source` | `scripts/bound_program_o_affected_orders.py` ya no casa su hash congelado |
| `test_paper2_exhaustive_search_registry::test_canonical_v2_metric_governance_...` | `validation["passed"] is False` |

**Causa identificada.** El fichero fue congelado por `9eb2433` y luego editado por **`ca84f39`**
(3 de agosto, *«Sweep 37 runners: --contract required everywhere it can be»*). El diff son **tres
líneas**: `default=DEFAULT_CONTRACT` → `required=True`, más el comentario que preserva el default
histórico.

Aquel commit reportó *«20 failed, 1736 passed — identical counts to before it»*. **Estos fallos ya
estaban dentro de esos 20**, así que comparar totales no podía detectarlo.

**No se arregla actualizando el hash esperado.** Un manifiesto de reproducibilidad existe para cazar
exactamente esto; si el remedio es reescribir el número cuando salta, deja de cazar nada. Hay dos
salidas legítimas y son distintas:

* **Re-congelar con justificación documentada**, aceptando que el cambio preserva el comportamiento
  del artefacto sellado. Es lo probable —el diff sólo endurece el CLI— **pero la invocación
  histórica, sin `--contract`, hoy ya no es ejecutable.**
* **Declarar rota la procedencia** de los claims que ese manifiesto ancla.

Program O está **cerrado con veredicto terminal**, así que esto no cambia ninguna conclusión viva.
Lo que cambia es si podemos decir *«este artefacto lo produjo este código»* señalando el árbol
actual. Para ese fichero, hoy, **no podemos**.

## 3. Pendientes de triar

| grupo | n aprox. | nota |
|---|---:|---|
| `test_paper2_bound_execution_harness` | ~6 | `ca84f39` los dejó fuera **a propósito**: se invocan desde tests que no pasan `--contract`, y arreglarlos exige tocar tests ya dentro del conjunto histórico de fallos |
| `test_paper2_bottleneck_full_frontier::test_checkpoint_resume_invalidates_on_dependency_hash_drift` | 1 | misma familia de deriva de hash |
| `test_program_j_request_snapshot_v2_structure_audit` | 1 | deriva de hash de fichero |
| resto | ~10 | sin triar |

## 4. La lección, y no es sobre estos tests

**Una suite con 21 en rojo es una suite que nadie lee**, y por eso el fallo de custodia sobrevivió:
se pierde en el ruido. El coste real no fue el test — fue que **la custodia dejó de ser automática
justo la semana en que abrimos un bloque nuevo**.

La regla que sale de aquí: *«los totales no empeoraron»* **no es una verificación** cuando el
conjunto base ya es no vacío. Hay que comparar **conjuntos de fallos**, no cardinales.
