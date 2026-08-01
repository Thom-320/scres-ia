# Preregistro — ¿el nulo de contención fue de la física, o de la clase de política?

**Escrito y commiteado ANTES de correr.** Runner:
`scripts/reanalyze_contention_policy_class.py`.
**No abre ninguna semilla nueva:** reutiliza el bloque **ya quemado** `5.200.001–16` de
`contention_headroom_v1`. Bajo `contracts/authority_ladder_v1.json`
(`fresh_roots_opened: false`, `scientific_execution_authorized: false`) la reproducción de tapes
quemados está permitida; abrir raíces frescas **no**. Este contrato se mantiene dentro de lo
permitido, y `f5` lo verifica.

## De dónde sale esta pregunta

Escribí que el `H_regime = 0` de contención venía de que los reclamantes son **simétricos por
construcción** (`stable_cssu_destination` reparte con un bit de hash). Cinco revisiones externas
señalaron que la simetría **no** implica que el óptimo esté en 0,5 —sólo implica equivarianza,
`V(α) = V(1−α)`— y **nuestro propio artefacto de `ret_excel` es el contraejemplo**: superficie
simétrica con `argmax` en 0,1 y 0,9. **Mi argumento, en su forma fuerte, está refutado.**

Al ir al código a comprobarlo aparece algo más concreto y más grave:

> El objetivo de cada riesgo se elige **por evento**: `rng.choice(("A","B"))` en `R22` (`:5930`),
> `R23` (`:5984`) y `R24` (`:6030`). **El estrés alterna dentro del episodio.**

Y el barrido que midió el nulo era **de constantes** — su propio docstring lo dice:
*«Constants only: the question here is whether headroom EXISTS, not whether a policy captures
it»*. **Una constante no puede seguir a un reclamante que alterna.** La equivarianza obliga a que
los estados espejo tengan acciones espejo; una única `α` fija no puede ser a la vez `0,9` y `0,1`,
así que **el valor dependiente del estado se cancela en la agregación por construcción de la clase
de política, antes de que la física tenga ocasión de mostrarlo.**

## La hipótesis

> **R.** Sobre los **mismos tapes quemados**, una política **equivariante** que reasigna a diario
> hacia el reclamante con mayor demanda insatisfecha supera a la **mejor constante** en
> `worst_claimant_fill`, con `LCB95 > 0`, y **un placebo no informado con la misma cadencia y la
> misma distribución marginal de `α` no lo consigue**.

**Lo que mediría:** `H_PI` de la **clase equivariante** — un techo clarividente, no un headroom
desplegable. **No autoriza entrenar nada.** Es el diagnóstico que decide si G3a tiene premisa.

## Diseño

* **Semillas**: `5.200.001+`, el bloque quemado. Ninguna nueva.
* **Celdas**: `{FIFO_PARTIAL, R24_AGE_PARTIAL}` × `{R1r+R2r|base, R1r+R2r|freq3_imp2}`,
  **no fungible**. `SPT_FULL` se excluye: su actuador está **medido muerto** (fracción viva
  0,0000).
* **Brazos**:
  1. `best_constant` — rejilla 0,1…0,9, mejor constante por celda;
  2. `equivariant_clairvoyant` — cada día, `α → 0,9` hacia el reclamante con mayor
     `demanded − delivered`; `0,5` en empate. Usa estado **verdadero y actual**;
  3. `uninformed_placebo` — misma cadencia y **misma distribución marginal de `α`**, pero el
     destino sale de una permutación derivada de la semilla, **independiente del estado**;
  4. `label_swap` — el brazo 2 con las etiquetas A↔B intercambiadas.
* **Endpoint primario, escalar**: `worst_claimant_fill`. **No** una clave lexicográfica, y **no**
  `ret_excel`, que está medido premiando el abandono. Diagnósticos al lado: `flow_fill_rate`,
  `ret_excel_full_ledger`, `ret_excel_risk_conditional`.
* **Inferencia**: bootstrap agrupado por semilla, 5.000 remuestreos, LCB95.
* **Cadencia declarada**: paso diario de 24 h, con acción sujeta a la latencia de activación de
  24 h que ya existe en el modelo.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_adaptive_action_is_live` | **control positivo**: el brazo adaptativo debe producir `α` que efectivamente varía y despachos distintos del mejor constante. Si el actuador está muerto o la latencia lo anula, no hay experimento |
| `f2_placebo_does_not_reproduce` | **el falsador central**: si un placebo no informado con la misma cadencia iguala al clairvoyant, el valor está en **variar**, no en **qué** hace variar — que es exactamente lo ya medido en `op12`. Puede fallar, y si falla la hipótesis muere |
| `f3_label_swap_equivariance` | intercambiar A↔B debe dar el mismo resultado. Si no, la ventaja viene de un sesgo de desempate, del orden de cola o del hash — no del mecanismo |
| `f4_constant_arm_reproduces_sealed_artifact` | **custodia**: el brazo constante debe recuperar los números sellados de `contention_headroom_v1_1`. Si no, la comparación no es contra el nulo publicado |
| `f5_no_fresh_seeds_opened` | **gobernanza**: cualquier semilla fuera del bloque quemado violaría `authority_ladder_v1` |
| `f6_endpoint_is_scalar_not_lexicographic` | un `LCB95` sobre una tupla no significa nada; el primario debe ser un escalar |
| `f7_no_gain_by_abandonment` | el `worst_claimant_fill` no puede subir a costa de perder pedidos; se reportan `lost_orders` y el ledger completo |

## Regla de lectura, fijada de antemano

* **Clairvoyant > mejor constante con `LCB95 > 0`, placebo batido, equivarianza limpia** →
  `POLICY_CLASS_WAS_THE_BINDING_CONSTRAINT`. El nulo anterior era **de instrumento**, y G3a pasa a
  tener premisa medida. **Sigue sin autorizar entrenamiento**: es un techo clarividente.
* **Clairvoyant ≈ mejor constante** → `PHYSICS_IS_FLAT_FOR_THE_EQUIVARIANT_CLASS`. La clase de
  política no era la restricción, y **mi diagnóstico de simetría muere del todo, barato**.
* **Clairvoyant > constante pero el placebo también** → `VALUE_IS_IN_VARYING_NOT_IN_STATE`,
  replicando `op12`. Resultado con contenido y **cierra el carril**.

**Alcance:** esto no reabre Program O ni Program Q, no toca ningún artefacto fechado, y no
promueve nada a confirmación. Un falsador que falle **no oculta la corrida**: nada se promueve,
todo se registra.
