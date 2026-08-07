# Enmienda — los dos falsadores del paso 3 no fallan por un defecto de código

**Escrita ANTES de correr el diagnóstico.** Runner: `scripts/run_step3_expressiveness_diagnostic_v1.py`.

## 1. Lo que el registro de huecos daba por supuesto

`docs/REGISTRO_DE_HUECOS_2026-08-07.md` A1:

> *"El preregistro lo nombra guardarraíl bloqueante; el runner sólo persiste `flow_fill_rate`, un
> agregado que **no ve un producto abandonado**. **Cierre:** que el runner persista fill por
> producto y re-correr. ~5 h."*

Esa lectura **supone que la dimensión existe y se cayó al persistir**. Antes de gastar cinco horas
de cómputo hay que comprobar el supuesto, que es justo lo que no se hizo.

## 2. Los dos hechos que se miden

**E1 — ¿cuántos reclamantes emite de verdad el contrato del paso 3?** Si emite exactamente uno,
entonces `worst_product_fill` **es** `flow_fill_rate`, el guardarraíl preregistrado **no es
expresable** en este contrato, y no hay nada que persistir. Re-correr no compraría nada.

**E2 — ¿dónde cae la postura proyectada de DDMRP dentro del dominio compartido 6³?** El artefacto
sellado reporta **una sola** postura, `[1344, 1344, 504]`. Si se pega al techo del dominio, el
brazo emite una postura porque **el dominio no puede expresar su objetivo**, no porque el actuador
esté roto. Es el mismo hecho que `results/ddmrp_unprojected_v1/` midió por el otro lado: DDMRP sin
proyectar sostiene **+1,02 M / +1,27 M** unidades de más para una métrica full-ledger
**bit a bit idéntica**.

## 3. Por qué esto es una categoría conocida y no una excusa

Es exactamente la forma de `f3b_true_equivariance_is_not_testable_here` en el carril de contención,
cuyo propio texto dice: *"it cannot fail here, and that is the finding: the model has no parameter
that distinguishes A from B"*. **Un guardarraíl que el modelo no puede expresar no es un
guardarraíl que se olvidó.**

## 4. El falsador que decide, y puede tumbar esta enmienda

`f2_the_guardrail_would_be_expressible_if_the_domain_had_it` **pasa sólo si el simulador NO
expresa la dimensión**. Si emite más de un reclamante, o si los pedidos llevan un atributo de
producto, entonces:

* el registro de huecos tenía razón,
* esta enmienda está equivocada,
* y **la re-corrida preregistrada de cinco horas hay que hacerla**.

El veredicto en ese caso es `GUARDRAIL_IS_EXPRESSIBLE_THE_RERUN_IS_STILL_REQUIRED`, y se acata.

## 5. Lo que esta enmienda NO hace

No adjudica el paso 3. **No levanta `NO_STRUCTURED_CONTROLLER_CONVERTS`** y no autoriza nada. Si
el guardarraíl resulta inexpresable, lo que sigue es una decisión del PI entre dos caminos, y
ninguno es «seguir como si nada»:

| camino | qué implica |
|---|---|
| **declarar el guardarraíl inexpresable en este contrato** | el screen del paso 3 se adjudica sobre `ret_excel_full_ledger`, que **ya** puntúa a cero los pedidos no servidos, así que el abandono ya está pagado. El alcance del veredicto se estrecha por escrito a «un solo reclamante» |
| **extender el contrato a un dominio multiproducto** | es territorio del Programa O, con su propia física y sus propias cintas. Es un experimento distinto, no una reparación de éste |

Y para DDMRP, si satura: **el paso 3 no puede sostener ningún claim sobre DDMRP en ninguna
dirección**, que es lo que su propio `ddmrp_domain_note` ya dice. La adjudicación que Garrido pidió
necesita un dominio con techo por encima del objetivo de DDMRP.

## 6. Alcance

Diagnóstico. Cintas del paso 3 ya quemadas, cero semillas. Reemplaza cinco horas de cómputo por una
enmienda escrita — **o demuestra que el cómputo hace falta**, que es el único desenlace en el que
esta enmienda se retira.
