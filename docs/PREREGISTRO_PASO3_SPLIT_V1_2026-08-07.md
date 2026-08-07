# Preregistro — el paso 3 con el guardarraíl que su contrato exigía

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_expanded_contract_comparators_v2.py
--cssu-topology split_v1`. Mismas cintas, mismas semillas, mismo dominio, mismos riesgos, misma
métrica. **Cero semillas nuevas.**

## 1. Por qué existe

`docs/PREREGISTRO_PASO3_GARRIDO_MPC_EXPANDIDO_2026-08-06.md` nombró `worst_product_fill` como
guardarraíl bloqueante. Hoy quedó medido (`results/step3_expressiveness/result.json`) que en el
contrato **agregado** hay **un solo reclamante** —141 pedidos, `cssu_destination` = `None`, sin
atributo de producto— así que `worst_product_fill` **es** `flow_fill_rate` y no puede vetar nada.
No fue un campo que se cayó: es una dimensión que el contrato no tiene.

El DES **sí** tiene la dimensión, detrás de un flag: `cssu_topology_mode='split_v1'` reparte el
mismo flujo de pedidos entre dos reclamantes `A`/`B` por hash estable.

## 2. El hecho que hace legítimo comparar las dos corridas

Medido antes de escribir esto, sobre **8 cintas de las dos familias**:

> `ret_excel_full_ledger` y `flow_fill_rate` bajo `split_v1` reproducen los del agregado con
> **delta = 0,000e+00** — exactamente cero, no «dentro de tolerancia».

`split_v1` **particiona sin cambiar la física**. Por eso esta corrida no supersede al agregado: lo
acompaña, con una columna más.

## 3. El guardarraíl se mueve — comprobado antes de gastar 3 horas

Seis posturas × tres cintas R2r:

| cinta | rango de ReT | rango de `flow_fill` | rango de **`worst_product_fill`** |
|---|---:|---:|---:|
| 1422001 | 1,6e−03 | 0,527 | **0,623** |
| 1422002 | 8,3e−04 | 0,646 | **0,613** |
| 1422003 | 1,3e−03 | 0,497 | **0,667** |

La postura lo lleva de 0,04 a 0,72. **Puede vetar**, que era la condición de `f7` del preregistro
original.

## 4. La advertencia, declarada ahora y no descubierta después

En estas cintas `affected_cssu` es `None`, así que **los riesgos no se localizan** en un CSSU, y el
espacio de acción del paso 3 —posturas de buffer— **no tiene palanca A/B**. El peor-producto se
mueve por postura y por asignación de hash, **no por una decisión de asignación**.

Consecuencia que se acepta de antemano: el guardarraíl **podrá vetar**; que además **discrimine
entre brazos** es lo que esta corrida contesta, y puede contestar que no. Si todos los brazos se
mueven juntos, el resultado es *«el guardarraíl es computable y vetador pero no separa
controladores en este espacio de acción»*, y se reporta así.

## 5. Diseño — idéntico al agregado salvo el flag

`--phase full --metric ret_excel_full_ledger --cssu-topology split_v1`, 52 semanas, época 4,
5 escenarios, 6 tapes por shard, **216 posturas estáticas**:

| shard | familia | semillas |
|---|---|---|
| `s1_r1r_a` | R1r | 1.420.001+ |
| `s2_r1r_b` | R1r | 1.421.001+ |
| `s3_r2r_a` | R2r | 1.422.001+ |
| `s4_r2r_b` | R2r | 1.423.001+ |

## 6. Regla de lectura, fijada ahora

Primaria: `ret_excel_full_ledger`, contraste pareado por tape contra el mejor de las 216 posturas
estáticas, IC95 bootstrap. **Bloqueante: `worst_product_fill`** — un brazo que gane en ReT y pierda
el guardarraíl **no gana**.

* ningún controlador supera al mejor estático → **`NO_STRUCTURED_CONTROLLER_CONVERTS_UNDER_THE_REAL_GUARDRAIL`**, y ahora sí es el screen preregistrado;
* alguno supera **y** pasa el guardarraíl → residual real, que **no autoriza entrenar**, sólo preregistrar;
* alguno supera **y** falla el guardarraíl → **`A_CONTROLLER_WINS_BY_ABANDONING_A_CLAIMANT`**, que es el hallazgo más informativo de los tres.

## 7. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f_ret_reproduces_bit_identically` | la columna `ret_excel_full_ledger` de cada `(tape, postura)` debe coincidir **exactamente** con la del artefacto agregado. **Falla si `split_v1` cambia la física**, y entonces las dos corridas no son comparables y ésta no acompaña a nada |
| `f_two_claimants_are_actually_emitted` | falla si algún tape emite un solo destino, y entonces el guardarraíl vuelve a ser el agregado |
| `f_the_guardrail_varies_across_arms` | falla si `worst_product_fill` es constante entre brazos; entonces es computable pero decorativo, y se dice |
| `f_no_new_seeds` | 1.420.001–1.423.006, ya abiertas |
| `f_metric_is_explicit` | `--metric` es obligatorio desde hoy; `ret_excel` premia el abandono y no puede heredarse |

## 8. Alcance

Desarrollo sobre cintas ya quemadas. **No supersede el veredicto agregado**, no adjudica el
manuscrito, no autoriza entrenamiento.
