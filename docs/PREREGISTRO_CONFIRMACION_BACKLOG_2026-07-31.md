# Preregistro — confirmación de `backlog` sobre semillas vírgenes

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_backlog_confirmation.py`.
Antecedente sellado: `results/sensitivity/observable_sweep_op12_v1/result.json`
(`bfe60ca8d4aa7266…`, cinco falsadores PASA).

## Qué encontró el barrido, y por qué no basta

De siete observables, seis utilizables, **uno solo** —`backlog`, la cola de pedidos pendientes—
dio captura **positiva** de la brecha del oráculo (+43,2%) **y** batió a su propio placebo
(−64,8%). Es el primer positivo de toda la puerta.

**Tres razones por las que ese número, tal cual, no autoriza nada:**

1. **Un positivo de seis es exactamente lo que produce el azar** sin corrección por
   comparaciones múltiples. El barrido eligió el ganador *después* de mirar.
2. **No tiene intervalo de confianza.** Es una media sobre 6 semillas de prueba × 7 regímenes,
   sin dispersión reportada. Un +43,2% de una brecha de 3,5e-5 puede ser ruido de una décima.
3. **El umbral se ajustó y se leyó en el mismo barrido.** Las semillas de prueba eran disjuntas
   de las de ajuste (`f4` PASA), pero la *elección del observable* usó ambas.

## La hipótesis única, declarada

> Con la regla **congelada** del artefacto sellado (`low = 21,0 h`, `high = 30,0 h`,
> `threshold = 116 361,6`), sobre **semillas vírgenes disjuntas** de las de ajuste y de prueba,
> la política reactiva sobre `backlog` supera a la mejor constante (21,0 h) con
> **LCB95 > 0** en la diferencia pareada, **y** supera a su placebo sobre las mismas semillas.

Una sola hipótesis. No se elige observable aquí: `backlog` está fijado por el artefacto previo.

## Diseño

* **Semillas**: `5 100 001…` — vírgenes; `f1` verifica disyunción con `4 900 001…006`
  (ajuste) y `4 900 501…506` (prueba). 12 semillas × 7 regímenes = 84 pares.
* **Pareado**: la diferencia se toma **por (régimen, semilla)** contra la constante evaluada en
  esa misma semilla. Mismo flujo exógeno en ambos brazos (`strict_exogenous_crn`).
* **Bootstrap agrupado por semilla** (10 000 remuestreos): la semilla es la unidad
  independiente; los siete regímenes dentro de una semilla están correlacionados y no pueden
  contarse como observaciones separadas.
* **Placebo**: la misma regla movida por la traza de `backlog` de **otro episodio**, con su
  propia diferencia pareada y su propio LCB95.
* **Métrica**: `ret_excel_risk_conditional`, la misma del barrido.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_seeds_are_virgin` | reutilizar semillas convertiría esto en releer el mismo dato |
| `f2_rule_is_the_frozen_one` | los tres parámetros se **leen del JSON sellado**, no se retipean; un valor distinto sería otra política |
| `f3_rule_switches_on_new_seeds` | una regla que no conmuta iguala trivialmente a la constante y el test sería vacuo |
| `f4_paired_difference_has_variance` | varianza cero haría el IC degenerado y el LCB no significaría nada |
| `f5_placebo_is_not_the_signal` | si la traza placebo coincidiera con la real, el control no controlaría nada |

## Regla de lectura, fijada de antemano

* **LCB95 > 0 y bate al placebo** → `CONFIRMED_BACKLOG_SENSOR`. La puerta declarada queda
  **abierta**: una clase de política más rica pasa a estar autorizada, porque el sensor mínimo
  ya extrae señal.
* **LCB95 ≤ 0** → el +43,2% era ruido de selección. La puerta se sostiene por segunda razón
  independiente y el negativo se fortalece.

**Y esto vale en ambos casos:** un PASS autoriza *gasto*, no una afirmación de headroom
material. La brecha total del oráculo es **3,5e-5**; capturarla entera sigue estando **~290×
bajo la barra de 0,01**. Lo que un PASS diría es que el acoplamiento es *extraíble*, no que
valga la pena extraerlo. No confundiré esas dos cosas al reportar.
