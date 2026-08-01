# Preregistro — Fase 1B: un presupuesto de expedición **escaso**, y cuándo gastarlo

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_expedite_headroom_v1.py`.
Sucede a la Fase 1A y **prueba la hipótesis que ella generó**, no una idea nueva.

## De dónde sale, y por qué cambia respecto al plan original

La Fase 1A midió contención **severa** —113.632 raciones renunciadas por episodio, ~17 % del
flujo— con `H_regime = 1,5e-04`. La disputa era real; **la decisión no**. Y la razón es
localizable: los dos reclamantes son **simétricos por construcción** (destino asignado por hash
50/50), así que su demanda esperada es idéntica **en todos los regímenes** y el reparto óptimo no
tiene por qué moverse.

> **La hipótesis que dejó: la contención crea headroom sólo cuando los reclamantes son
> asimétricos de una forma que varía con el estado.**

El plan original de la Fase 1B era «expedición con coste convexo». **Lo cambio, y digo por qué:**
un coste convexo exige inventar un coeficiente que ninguna fuente fija, y —peor— la expedición
sin escasez es almuerzo gratis, así que el óptimo sería «expedir siempre», invariante, y otro
`H_regime ≈ 0` que no habría enseñado nada.

**Un presupuesto escaso es la versión asimétrica del mismo mecanismo, y no inventa ningún
número.** Hay `B` horas de expedición al año; las semanas **no son intercambiables**; hay que
elegir **cuándo**. Es contención sobre el tiempo, con la asimetría que a la Fase 1A le faltaba, y
es militarmente literal: el transporte aéreo prioritario es finito.

## Mecanismo

`expedite_budget_hours` (por defecto **0,0** — la función queda apagada y nada cambia). Cuando un
epoch arma una expedición, la siguiente pierna de transporte elegible recorta su tiempo de
proceso y **debita** el presupuesto. Piernas: `op8_pt`, `op10_pt`, `op12_pt` — las tres del
camino crítico al teatro, 24 h nominales cada una. El gancho está en `_pt()`, el único punto por
el que pasan todos los tiempos de proceso.

## El instrumento: `H_PI`, no `H_regime`

Un presupuesto obliga a repartir **en el tiempo**, así que la pregunta no es «¿varía la mejor
constante con el régimen?» sino **«¿vale algo saber cuándo gastar?»**. Ése es el mismo
`H_PI` con el que Program O midió 0,1515:

    H_PI = [mejor calendario conocido de antemano] − [mejor tasa constante]

ambos **con el mismo presupuesto**. Se reporta también `H_regime`, para poder comparar en la
misma escala que la Fase 1A.

**Y el control que hoy demostró ser imprescindible: el placebo.** Un calendario **aleatorio** con
el **mismo gasto total**. Esta mañana, en `op12`, el placebo no informado batió a la señal real:
el valor estaba en *variar*, no en *saber*. Si aquí el placebo iguala al clarividente, el valor
está en **gastar**, no en **cuándo**, y lo diré exactamente así.

## Diseño

* **Presupuestos**: `B ∈ {0, 168, 336, 672, 1344}` horas/año (0 = brazo de control; 1344 h ≈ 56
  días-pierna, holgado pero lejos de ilimitado).
* **Regímenes (4)**: {R1r, R2r, R1r+R2r} base, más R1r+R2r escalado ×3 en frecuencia.
* **Brazos**: `clarividente` (calendario óptimo con conocimiento del episodio), `constante`
  (tasa fija, optimizada sobre la misma rejilla), `placebo` (calendario aleatorio, mismo total).
* **Semillas**: `5 400 001…` vírgenes, CRN entre brazos.
* **Métricas**: `ret_excel_risk_conditional` primaria; `ret_excel_visible_clipped_0_1` y
  `flow_fill_rate` al lado.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_budget_is_conserved` | si se gasta más de `B`, el clarividente gana por hacer trampa y `H_PI` no mide nada |
| `f2_expediting_actually_shortens_delivery` | si `CTj` no baja al expedir, el mecanismo no existe y todo lo demás es ruido |
| `f3_the_clairvoyant_respects_the_same_budget` | un clarividente con más presupuesto que la constante compara dos cosas distintas |
| `f4_the_constant_is_optimised_not_assumed` | comparar contra una constante arbitraria infla `H_PI`; se optimiza sobre la misma rejilla |
| `f5_placebo_spends_the_same_total` | un placebo que gasta menos perdería por presupuesto, no por ignorancia |
| `f6_budget_zero_reproduces_the_baseline` | el brazo `B = 0` debe ser **idéntico** al DES sin la función; si no, el gancho cambió algo que no debía |
| `f7_seeds_are_virgin` | reutilizar semillas invalidaría la confirmación |

## Regla de lectura, fijada de antemano

* **`H_PI ≥ 0,01`, `LCB95 > 0`, y el clarividente bate al placebo** →
  `TIMING_HEADROOM_FOUND`: existe una decisión temporal que aprender, y la Fase 3 queda
  autorizada **sobre esta palanca**.
* **`H_PI ≥ 0,01` pero el placebo empata** → el valor es **open-loop**: gastar, no cuándo. Sería
  la **segunda** aparición del mismo patrón que `op12` en un mecanismo distinto, y eso pasa de
  ser una anécdota a ser un resultado sobre esta cadena.
* **`H_PI < 0,01`** → ni siquiera un clarividente con presupuesto saca 1 % de esta cadena. Con la
  Fase 1A, cerraría las dos formas de contención que sabemos construir aquí, y la conclusión del
  paper dejaría de ser «no encontramos» para ser **«medimos que no lo hay, y por qué»**.

**Lo que no afirma:** nada sobre RL. `H_PI` es un techo clarividente; que una política aprendida
lo alcance es otra pregunta, y va después.
