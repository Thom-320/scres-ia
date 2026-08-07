# Resultado — el paso 3, por primera vez con el guardarraíl que su contrato exigía

**Artefacto:** `results/step3_split_pooled/result.json` (sello `1cfc4000b163a986…`),
**`NO_STRUCTURED_CONTROLLER_CONVERTS`** · preregistro
`docs/PREREGISTRO_PASO3_SPLIT_V1_2026-08-07.md`, commiteado antes de correr · cuatro shards,
**24 tapes**, 216 posturas estáticas completas, `ret_excel_full_ledger`, **cero semillas nuevas**.

## 1. Lo que cambia respecto al agregado

`f4_the_preregistered_guardrail_is_available_and_can_veto` **PASA**:

| | |
|---|---|
| `worst_product_fill` presente | **5.256 de 5.256 filas** |
| varía | **sí**, rango `[0,6406, 0,9940]` |
| distinto del agregado | **sí** |
| topología | `split_v1` en **todas** las filas |

Es la primera vez que el screen del paso 3 se corre **con el guardarraíl bloqueante que su
preregistro nombraba**. En el contrato agregado era imposible: con un solo reclamante,
`worst_product_fill` **es** `flow_fill_rate`.

Ese falsador estaba **hardcodeado a `passed: False`** en el script de fusión. Era cierto cuando se
escribió —el runner no persistía el campo— pero **una constante no es una comprobación**: habría
seguido en falso después de que el campo existiera. Ahora lee las filas.

## 2. El resultado

| familia | mejor postura de 216 | brazo | Δ contra el mejor estático | IC95 | tapes a favor |
|---|---|---|---:|---|---:|
| **R1r** | `[0, 0, 336]` (0,009150) | `greedy_pi_best_found_v2` | +0,000024 | [+0,000004, +0,000049] | 9/12 |
| | | `replay_mpc_v2` | **−0,000021** | [−0,000046, +0,000004] | 3/12 |
| | | `ddmrp_projected_v2` | **−0,000303** | [−0,000342, −0,000266] | 0/12 |
| **R2r** | `[0, 672, 168]` (0,241743) | `greedy_pi_best_found_v2` | +0,001002 | [+0,000088, +0,002202] | 6/12 |
| | | `ddmrp_projected_v2` | **0,000000** | [0,000000, 0,000000] | 0/12 |
| | | `replay_mpc_v2` | **−0,000991** | [−0,003701, +0,000683] | 2/12 |

**Ningún controlador estructurado convierte.** El único brazo positivo es el techo de información
perfecta, excluido por diseño (`f5` comprueba que no se cuele en el conjunto de ganadores — la
primera versión del script lo dejó entrar y fabricó el titular contrario).

**Nadie gana abandonando:** el fill agregado de los tres brazos queda en 0,9963–0,9966 (R1r) y
0,9244–0,9258 (R2r), y el guardarraíl estaba disponible para vetar. No tuvo a quién vetar.

## 3. `f6` sigue fallando, y ya está adjudicado

`ddmrp_projected_v2` emite **una sola postura**. No es un defecto abierto: está explicado en
`docs/ENMIENDA_PASO3_ALCANCE_Y_ADJUDICACION_DDMRP_2026-08-07.md` — la postura se pega al techo del
dominio, y `results/buffer_saturation_diagnostic/` midió que **por encima de ese techo la métrica
es plana exactamente** (×10 → 0,000000, `saturated_upward: true` en los tres nodos). El Δ de
exactamente `0,000000` en R2r es esa degeneración, vista de frente.

## 4. Lo que ahora se puede afirmar, y con qué alcance

> **En el contrato del paso 3, con dos reclamantes y el guardarraíl de peor producto disponible y
> capaz de vetar, ningún controlador estructurado —MPC de replay ni DDMRP proyectado— supera a la
> mejor de 216 posturas estáticas bajo `ret_excel_full_ledger`.**

El hueco **A1 queda cerrado**: `NO_STRUCTURED_CONTROLLER_CONVERTS` deja de ser un diagnóstico de
desarrollo con un guardarraíl ausente y pasa a ser **el screen preregistrado, aplicado**.

Lo que sigue sin ser: una confirmación. Las 24 tapes son cintas ya quemadas y no queda ningún
bloque virgen en el proyecto.

## 5. Nota de comparabilidad

`split_v1` reproduce `ret_excel_full_ledger` y `flow_fill_rate` del contrato agregado con
**delta 0,000e+00** sobre 8 cintas de las dos familias, medido antes de escribir el preregistro.
Por eso este artefacto **acompaña** a `results/step3_pooled/` en vez de superseder lo: mismo ReT,
una columna más.
