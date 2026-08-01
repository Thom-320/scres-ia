# Preregistro — G3-obs: ¿el techo se convierte con observaciones desplegables?

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_g3_obs_conversion.py`.
**Semillas:** bloque **quemado** `5.200.001–16`. **Ninguna nueva** — `authority_ladder_v1`
(`fresh_roots_opened: false`) lo prohíbe y `f8` lo verifica.
**Márgenes heredados:** `docs/PREREGISTRO_G3C_ACOPLAMIENTO_TEMPORAL_2026-08-01.md` §4, sin cambios.

## 1. La admisión que estrecha la pregunta antes de empezar

La reauditoría midió un brazo que llamé **clarividente**. Al escribir este contrato hay que
clasificar su señal honestamente, y la clasificación me desfavorece:

> Ese brazo leía `cssu_demanded − cssu_delivered`: **demanda acumulada menos entregas**. Es una
> cantidad **del ledger**, que cualquier operador conoce, y **no lee el futuro** —el objetivo del
> siguiente riesgo le es invisible, y la acción se activa 24 h después de decidirse.

**Entonces no era un techo clarividente: era ya una política observable** — y, más incómodo aún,
**era un umbral de dos ramas**. Eso significa que `STRUCTURED_CONTROL_SUFFICES` es el desenlace
**esperado**, no el temido, y este contrato existe para comprobarlo con rigor en vez de para
buscar una excepción.

Lo que queda genuinamente abierto son **dos** preguntas:

1. **¿Sobrevive el valor a límites de observación realistas** —ventana finita en vez de acumulado
   de todo el horizonte, retardo de información, ruido— o depende de una contabilidad perfecta
   desde el primer día?
2. **¿Queda residual sobre el mejor umbral simple** que una política más rica (árbol/tabular)
   capture, o el `if` de dos ramas agota el valor?

## 2. Estimandos, separados

Con `Y = worst_claimant_fill` (escalar; `service_first_v2` es regla de selección, nunca estimando):

* **`H_obs` = `V(mejor política observable) − V(mejor constante)`** — la conversión.
* **`residual_over_simple` = `V(mejor política rica) − V(mejor umbral simple)`** — el único
  contraste que podría abrir sitio a algo aprendido.
* **`degradation_by_realism` = `V(acumulado) − V(ventana/retardo/ruido)`** — el precio de la
  observabilidad realista.

**Ningún estimando se llama headroom desplegable**, y ninguno autoriza entrenar.

## 3. Diseño

* **Celdas**: `FIFO_PARTIAL × {R1r+R2r|base, R1r+R2r|freq3_imp2}`, **no fungible**, paso diario
  24 h con la latencia de activación de 24 h ya existente.
* **Partición de semillas**: **8 de desarrollo / 8 de test, disjuntas**. Todo umbral, bin o
  parámetro se **ajusta sólo en desarrollo** y se evalúa en test. Sin esto, «el mejor umbral» se
  elegiría sobre su propio resultado — el defecto que ya cometí en la corrida de Cobb-Douglas.
* **Señal observable**, calculada al momento de decidir y nunca del futuro:
  `s_t = (no servido A − no servido B) / (no servido A + no servido B + ε)` sobre una **ventana
  móvil de `W` días**.

| brazo | qué prueba |
|---|---|
| `best_constant` | el nulo viejo, como contexto |
| `threshold_cumulative` | **la regla de la reauditoría**, ahora correctamente etiquetada *observable* |
| `threshold_windowed` | ventana finita: ¿hace falta la contabilidad desde el día 1? |
| `threshold_delayed` | la señal llega con `D` días de retardo |
| `threshold_noisy` | la señal viene con ruido multiplicativo |
| `tabular_5bin` | política rica: 5 bins de `s` → `α`, ajustada en desarrollo |
| `uninformed_placebo` | misma cadencia y soporte, sin leer nada |
| `wrong_claimant` | misma información, dirección invertida |

## 4. Guardarraíles — con los márgenes firmados, no a margen cero

Un brazo **falla** si el **UCB95** del daño `(mejor constante − brazo)` **excede `δ`**:

| guardarraíl | `δ` |
|---|---:|
| `flow_fill_rate` | 0,005 |
| `lost_orders` | 0,25 órdenes/episodio |
| `backorder_qty_final` | 1,0 % relativo |
| masa, capacidad creada, recursos programados | 0,0 exacto |

**SESOI primario `+0,010`.** `ret_excel` es **diagnóstico**, nunca guardarraíl: está medido
premiando el abandono.

## 5. Potencia — y el desenlace legal si no la hay

Con 8 semillas de test y **prohibido abrir más**, la potencia puede no alcanzar. El runner
**calcula y publica el efecto mínimo detectable (MDE)** al 90 % de potencia, una cola, a partir de
la SD observada de las diferencias pareadas.

* **`MDE ≤ 0,010`** → el contraste es interpretable.
* **`MDE > 0,010`** → **`STOP_G3_OBS_UNDERPOWERED`**, se publica el número y **no se afirma nada
  sobre el primario**. No se afloja el SESOI ni se toman semillas de otro bloque para llegar.

## 6. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_signal_is_causal_and_pre_action` | la señal debe construirse **antes** de la decisión y no contener el objetivo futuro del riesgo. Si filtra, es un oráculo disfrazado |
| `f2_real_signal_beats_shuffled_delayed_and_wrong` | el orden `real > retardada > barajada > equivocada` debe respetarse. Si una permutación iguala a la real, no hay información |
| `f3_thresholds_fit_on_development_only` | **la puerta contra la selección sobre test**; sin ella «el mejor umbral» se elige mirando su resultado |
| `f4_every_guardrail_has_a_signed_margin` | ningún guardarraíl a margen cero salvo identidades algebraicas — la reparación del `f7` que detuvo la reauditoría |
| `f5_power_is_published_pass_or_fail` | el MDE se publica **pase o falle**; ocultarlo convertiría un nulo sin potencia en una afirmación |
| `f6_actuator_is_live` | `α` debe moverse y respetar la latencia; ya me falló una vez y cazó un defecto real |
| `f7_no_gain_by_abandonment` | ahora **con margen**: `UCB95(daño) ≤ δ` |
| `f8_no_fresh_seeds_opened` | gobernanza: cualquier semilla fuera del bloque quemado viola `authority_ladder_v1` |

## 7. Reglas terminales, fijadas antes de correr

* **`H_obs ≥ SESOI` con `LCB95 > 0`, márgenes respetados, y `residual_over_simple` NO material**
  → **`STRUCTURED_CONTROL_SUFFICES_G3_OBS`**. **Es el desenlace esperado y es un éxito del
  contrato**: el valor existe, es desplegable, y **un `if` de dos ramas lo agota**. No se entrena
  nada, y para el paper vale tanto como un positivo.
* **`residual_over_simple ≥ SESOI` con `LCB95 > 0`** → `G3_OBS_RESIDUAL_OVER_SIMPLE_RULE`.
  Autoriza **pasar a G3c**, nunca una afirmación de prima.
* **`H_obs < SESOI`, o el realismo lo destruye** → `OBSERVABLE_CONVERSION_FAILS`. El techo no era
  convertible y el carril se cierra sin construir física nueva.
* **cualquier margen violado** → `STOP_G3_OBS_GUARDRAIL`. **Sin segundo rescate.**

## 8. Lo que no afirma

No reabre Program O ni Program Q. No dice nada sobre `N ≥ 3` ni sobre asimetría física: G3a sigue
siendo el único contrato capaz de expresar una equivarianza A↔B real. Y no convierte la
reauditoría en un resultado: sigue `HALTED`.
