# Preregistro — readjudicar `headroom_gsa` bajo el objetivo declarado por el PI

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_gsa_resilience_only_v1.py`.
Cero semillas nuevas: los bloques `3000001`, `4200001`, `4500001` ya están abiertos.

## 1. Qué cambió, y qué NO cambió

**Cambió el objetivo, por decisión del PI del 2026-08-07:** *«la medida es la resiliencia, es lo
único que nos importa»*. El resultado histórico `results/headroom_gsa/oos_guardrail_check.json`
tiene `qualifies_new_lane: false` con esta razón literal:

> `worst_cssu_fill_delta -0.13 << -0.02 fairness guardrail`

y su veredicto lo llama, con sus propias palabras, *"the Program-G concentration/**fairness**
artifact"*. Es un guardarraíl **distributivo**, no de resiliencia. Bajo el objetivo declarado deja
de ser compuerta y pasa a resultado reportado.

**NO cambió nada del instrumento.** Y hay un hecho que quita del medio la objeción obvia:
`ret_order_metrics` (`supply_chain/program_g.py:320`) dice *"Unattended orders are marked lost →
score 0 → Ut"*. **La métrica de esta lane no censura**: el abandono ya se paga a cero. Así que
esto **no** es el caso en que «sólo importa la resiliencia» premia abandonar — eso vale para
`ret_excel` visible, no aquí. La distinción es la que decide si esta readjudicación es legítima, y
por eso va primero.

## 2. Lo que la corrida histórica ya midió

H_obs positivo y estable fuera de muestra en **tres bloques independientes**:

| bloque | H_obs | IC95 | `ret_quantity_delta` | `worst_cssu_fill_delta` | `attended_delta` |
|---|---:|---|---:|---:|---:|
| `GP_search_3000001` | 0,0131 | [0,0102, 0,0160] | +0,0142 | −0,1435 | −0,09 |
| `FRESH_4200001` | 0,0114 | [0,0087, 0,0141] | +0,0128 | −0,1282 | −0,14 |
| `FRESH_4500001` | 0,0100 | [0,0072, 0,0129] | +0,0130 | −0,1258 | −0,28 |

θ localizado: `signal_q 0,532 · lead 2 · surge_mult 1,946 · persistence short · commonality 0,887
· r22_prob 0,107`.

Pero ese artefacto es del **2026-07-14** y su propio `claim_limit` dice que **no puede promover un
contrato ni probar H_PI/H_obs**. Por eso esto re-ejecuta, no re-lee.

## 3. Diseño

Re-ejecutar `headroom_at` en el θ localizado sobre los tres bloques, `n_tapes = 200` cada uno
(coste medido: 32 ms/cinta → ~20 s en total). Cuatro brazos por cinta:

| brazo | qué es |
|---|---|
| `static` | el **mejor** calendario periódico evaluado sobre las mismas cintas (baseline in-sample = el más exigente) |
| `oracle` | el máximo sobre las 3⁴ = 81 secuencias de acciones |
| `obs` | la política de creencia `_belief_policy` |
| **`placebo`** | **la secuencia que `_belief_policy` produjo en OTRA cinta**, aplicada a ésta — misma distribución de acciones, cero información alineada |

El placebo es obligatorio por norma del proyecto y **no existía** en `headroom_sensitivity.py`; se
construye aquí con el idioma que el repositorio ya usa (permutación de cintas, derangement).

**Reportado siempre, nunca bloqueante:** `worst_cssu_fill`, `attended`, `lost`, `ret_quantity`.
Esa es la decisión del PI, y queda escrita como decisión, no como hallazgo.

## 4. Regla de lectura, fijada ahora

La lane **califica** si y sólo si, con `LCB95` bootstrap sobre cintas:

1. `H_obs > 0` con `LCB95 > 0` en **los tres bloques**, y
2. `obs − placebo > 0` con `LCB95 > 0` en **los tres bloques**, y
3. `f2` pasa (abajo).

Cualquier fallo → **`GSA_DOES_NOT_QUALIFY_EVEN_UNDER_RESILIENCE_ONLY`**, y la lane se cierra por
número, no por preferencia.

## 5. Falsadores, con por qué cada uno **puede** fallar

| falsador | por qué puede fallar |
|---|---|
| `f1_the_historical_cell_still_reproduces` | re-calcula H_PI en el θ localizado y exige el `0,014446` sellado dentro de `2e-3`. **Falla si la física derivó** desde el 14-jul — es el hueco A2 aplicado a esta lane, y si falla se detiene todo |
| `f2_the_gain_is_not_bought_by_attending_fewer` | correlación de Pearson por cinta entre `(obs − static)` en ReT y `(attended_obs − attended_static)`. **Si la correlación es negativa y material (< −0,3), la ganancia se compra atendiendo menos** y la lane muere aunque el guardarraíl ya no bloquee. Es el falsador que puede matarla |
| `f3_an_uninformed_placebo_does_not_reproduce_it` | mismo reparto de acciones, sin alineación. Si el placebo iguala a `obs`, el valor está en la cadencia y no en la información — que es exactamente lo que ya se midió en `op12` |
| `f4_the_static_baseline_is_the_argmax` | comprueba que el estático elegido es el mejor de los calendarios sobre estas cintas. Si no lo fuera, el headroom estaría inflado por un comparador débil |
| `f5_no_new_seeds` | bloques `3000001`, `4200001`, `4500001`, ya abiertos. Falla si aparece una semilla fuera |

## 6. Alcance

Desarrollo sobre bloques ya abiertos. **No adjudica el manuscrito** y no autoriza entrenar nada.
Si califica, lo que abre es el derecho a **preregistrar** una lane con oracle-first, no a entrenar.
