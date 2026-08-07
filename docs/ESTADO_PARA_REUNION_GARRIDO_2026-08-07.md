# Estado para la reunión con Garrido — 7 de agosto de 2026

Todo lo de aquí sale de un artefacto sellado. Cada fila cita el suyo. Lo que no está medido se
declara como no medido.

## 1. Sus dos preguntas, contestadas

### Q1 — ¿qué familia de algoritmos imita mejor el aprendizaje de la cadena?

**La que retiene, no la que aproxima.** Tres mediciones independientes, con controles distintos:

| medición | resultado | artefacto |
|---|---|---|
| escalera de **15 métodos**, presupuesto 24 igualado, 12 semillas × 6 contextos | los **seis primeros son los seis que conservan estado**; la misma neurona **sin memoria** cae por debajo del OFAT crudo de la tesis | `results/search_ladder_v5/` |
| sustitutos a presupuesto igualado | una neurona de **5 parámetros** empata a MLP de 369 y KAN de 380 | `results/search_surrogates/` |
| bake-off a **200.000 parámetros igualados** | **KAN − MLP = −0,475 [−1,548 · +0,598]** — nada separa | `results/architecture_bakeoff/` |

**No hay prima neural en ningún entorno medido.** Y eso es la respuesta a Q1, no un fracaso: lo que
imita el aprendizaje de la cadena es **lo que cierra el lazo ③→⑧**, que es exactamente su Fig. 5 — y
le bastan cinco parámetros.

**El efecto Alzheimer tiene precio medido:** la memoria ahorra **7,90 corridas [6,88 · 8,93]** frente
a la misma neurona reseteada, y **5,43 [4,01 · 6,78]** frente al OFAT de la tesis
(`results/garrido_meta_learner_v2/`).

> **Las cifras 7,24 / 12,42 / 13,54 están retiradas** — vinieron de un runner con fuga de
> normalizador. Si aparecen en algún borrador, hay que quitarlas.

### Q2 — ¿cómo se integra esa familia en el DES?

Como **optimización por simulación sobre configuraciones**: el resultado observable de una corrida
alimenta el estado retenido que elige la configuración siguiente. Es su Fig. 2 leída entre corridas,
no dentro del episodio.

**Confirmación prospectiva, semillas vírgenes, ya cerrada:** transferencia de rejilla 288 → 4.608,
`GRID_TRANSFER_CONFIRMED__UCB1`, δ **+0,03073 [LCB +0,01990]** y **+0,05744 [+0,04989]**, 60
semillas vírgenes (`results/grid_transfer_v1/`).

## 2. Su método del oráculo — y una advertencia

Pidió *«qué porcentaje del techo teórico captura el modelo»*. Aquí está — **y ordena distinto que
nuestra métrica primaria**, así que conviene decirlo antes de que lo note un revisor:

| brazo | **% del techo** | AUC (nuestra primaria) |
|---|---:|---:|
| `lookahead_kg_transfer` | **99,70 %** | 0,08018 |
| `gp_ei_transfer` | 99,67 % | 0,08390 |
| **`neuron_memory`** | **99,50 %** | **0,05203** |
| `ofat_transfer` | 98,94 % | 0,06274 |
| `ucb1_transfer` | 97,64 % | **0,04502** |
| `ofat` (tesis) | 97,55 % | 0,10024 |
| azar | 93,38 % | 0,13979 |

No se contradicen: **el % del techo mide dónde acabas; el AUC mide cuánto cuesta llegar.** Con su
métrica gana el planificador con memoria; con la nuestra, el bandido con memoria. **En ambas, todo
lo que retiene va por delante de todo lo que no.**

## 3. Su diseño de cuatro pasos

| paso | estado |
|---|---|
| 1 · baseline (simulación de Garrido, políticas estáticas) | **hecho** — las 216 posturas enumeradas |
| 2 · MPC sobre las variables originales | **hecho**, y reclasificado como no válido en su v1; reparado en v2 |
| 3 · **MPC sobre las variables expandidas** | **CORRIENDO** — 4 shards, 6 tapes × 5 escenarios × 216 posturas × 52 semanas, semillas vírgenes 1.420.001+ |
| 4 · KAN bajo la misma configuración ampliada | **cuaderno entregado a David**, con los rivales embebidos |

**El paso 3 es el que define dónde puede vivir la prima neural**, porque el residual se mide contra
el mejor controlador estructurado. Hasta que aterrice, **cualquier afirmación sobre residual neural
en este contrato es prematura**.

## 4. El preprint de KAN

[arXiv:2407.16674](https://arxiv.org/abs/2407.16674) **sigue siendo preprint** — sin journal-ref, sin
DOI externo, y sus autores lo etiquetan «Technical Report». **Dos años sin publicar.**

**Pero no lo usemos para desestimarlo**: replicamos su hallazgo en nuestra propia cadena. La postura
defendible es *«coincidimos con ellos en un dominio que ellos no tocaron»*. Y su ablación juega a
nuestro favor: la ventaja de la KAN viene de **la base B-spline**, y un MLP con B-splines la iguala.

**Hueco de novedad: intacto.** Nada publicado combina KAN × resiliencia de cadena × DES.
(`docs/REVISION_LITERATURA_KAN_2026-08-06.md`)

## 5. La KAN: pierde como política, gana como surrogate

Su argumento tenía dos patas. **Una cae y la otra se sostiene**, y la frontera entre ambas es
justamente la contribución.

**Ahorro de parámetros como política de control: medido, y no se sostiene.** A 200.000 parámetros
igualados, KAN − MLP = **−0,475 [−1,548 · +0,598]**, y cuesta **4,1×** por decisión.

**Como surrogate supervisado de la superficie de diseño: gana, y con margen.** A **532 contra 529
parámetros**, evaluado sobre un cuarto **retenido** (`results/kan_interpretability/`):

| contexto | KAN R²_out | MLP R²_out |
|---|---:|---:|
| R1r | 0,9978 | 0,9418 |
| **R2r** | **0,9777** | **0,7424** |
| R1r+R2r | 0,9945 | 0,9425 |
| R1r\|esc | 0,9915 | 0,9312 |
| R2r\|esc | 0,9673 | 0,8912 |
| R1r+R2r\|esc | 0,9908 | 0,9599 |

**No es una contradicción, es la distinción que hace el propio preprint KANbeFair:** la ventaja de
la KAN está en **representación de funciones**. Una superficie de diseño *es* una función; una
política de control no lo es.

**Y las curvas univariadas son legibles y no son artefacto de la base.** Reajustado sobre superficie
**barajada**, el R² retenido sale **negativo (−0,556 a −1,938)** y la distancia de curvas es
0,317–0,428. Las formas son la cadena, no el spline. Eso es la interpretabilidad que él vende,
hecha concreta y con su control.

De paso confirma el resultado de turnos por otra vía: el recorrido de `shifts` en las curvas es
**0,05–0,28** frente a **1,5–2,6** de `op9_rop`. El turno es la variable que menos mueve la
superficie.

## 6. Lo demás que pidió

* **Turnos como proxy Pareto:** en **5 de 6 contextos no hay ahorro** — la config más barata dentro
  del 1 % usa los mismos turnos que el óptimo. Y bajo **R1r los turnos no compran nada** (0,00926 con
  1, 2 o 3), mientras bajo R2r sí. **El valor del turno lo decide la familia de riesgo.**
  (`results/shift_pareto_diagnostic/`)
* **Riesgo de acumular inventario al final del horizonte:** las filas del paso 3 traen
  `terminal_stock`; **se reporta en cuanto aterricen los shards**.
* **Métrica de resiliencia:** 162 derivaciones de sus dos métricas + 661 recurvaturas. **Bajo la
  curvatura que él publicó no hay headroom** (0,0000 en 288, 0,0195 en la extendida). El headroom
  sólo aparece con curvatura que él no declaró, y **sólo la amante del riesgo llega al umbral** —
  la aversión mueve el headroom **hacia cero**.

## 7. Lo que le debemos y va con retraso

Compromisos del 28 de julio, vencidos el 3 de agosto:

| responsable | compromiso | estado |
|---|---|---|
| Thomas | nodos y variables de decisión | **parcial** — rejilla 4.608 hecha; **capacidad finita en `wdc`/`al`/`sb` sin cablear** |
| **David** | KAN con grid search **en el entorno ampliado** | **no hecho** — su corrida va 2/5 y está en el entorno **viejo** |
| Thomas | preprint de KAN | **entregado hoy**, con retraso |
| Thomas | literatura KAN + resiliencia | **entregado hoy**, con retraso |
