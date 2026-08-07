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

**No hay prima neural como política o buscador en los entornos medidos.** Sí aparece una ventaja
descriptiva de KAN como *surrogate* supervisado de una superficie ya construida (§5), que es una
tarea distinta. La respuesta provisional a Q1 es por tanto funcional, no arquitectónica: lo que
imita el aprendizaje de la cadena es **lo que cierra el lazo ③→⑧ y conserva estado entre
corridas**. En la búsqueda, cinco parámetros bastan para empatar a aproximadores mayores.

**El efecto Alzheimer sobrevive al normalizador de prefijo**, en un *replay* de desarrollo sobre
cintas ya usadas: AUC de *regret* `memoria−reset` **+0,06070 [LCB95 +0,04556]** y
`memoria−OFAT` **+0,04821 [LCB95 +0,03325]**. El secundario censurado equivale a 5,83 y 5,33
corridas ahorradas, respectivamente (`results/garrido_normaliser_audit_v3/`). Las cifras 7,90 y
5,43 corresponden al normalizador oráculo y no son el titular corregido.

> **Las cifras 7,24 / 12,42 / 13,54 están retiradas** — vinieron de un runner con fuga de
> normalizador. Si aparecen en algún borrador, hay que quitarlas.

### Q2 — ¿cómo se integra esa familia en el DES?

Como **optimización por simulación sobre configuraciones**: el resultado observable de una corrida
alimenta el estado retenido que elige la configuración siguiente. Es su Fig. 2 leída entre corridas,
no dentro del episodio.

**Confirmación prospectiva ya cerrada:** transferencia de rejilla 288 → 4.608,
`GRID_TRANSFER_CONFIRMED__UCB1`, δ **+0,03073 [LCB +0,01990]** contra su réplica marginal y
**+0,05744 [+0,04989]** contra arranque en frío, n = 60
(`results/grid_transfer_confirmation_v2/`). La custodia demuestra coincidencia exacta con el
bloque declarado y **ninguna colisión conocida**; el inventario central se declara incompleto, por
lo que no prueba virginidad absoluta.

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
| 3 · **MPC sobre las variables expandidas** | **HECHO** — `NO_STRUCTURED_CONTROLLER_CONVERTS`. 4 shards, 12 tapes/familia, 5 escenarios, **las 216 posturas**, 5.256 filas, hashes de prefijo casando. El bloque 1.420.001+ no consta aún en el inventario central y no se describe como virgen |
| 4 · KAN bajo la misma configuración ampliada | **cuaderno entregado a David**, con los rivales embebidos |

### El resultado del paso 3

Contraste pareado contra **la mejor de las 216 posturas estáticas**, métrica
`ret_excel_full_ledger` (`results/step3_pooled/`):

| familia | brazo | Δ | IC95 | tapes |
|---|---|---:|---|---:|
| R1r | *greedy PI (techo, no compite)* | +0,000024 | [+0,000004 · +0,000049] | 9/12 |
| | **`replay_mpc_v2`** | **−0,000021** | [−0,000046 · +0,000004] | 3/12 |
| R2r | *greedy PI (techo, no compite)* | +0,001002 | [+0,000088 · +0,002202] | 6/12 |
| | **`replay_mpc_v2`** | **−0,000991** | [−0,003701 · +0,000683] | 2/12 |

### Y el mecanismo, que es lo que hay que contar

`results/buffer_saturation_diagnostic/`: perturbación uno-a-uno alrededor de la postura de
referencia. **×10 mueve la métrica exactamente `+0,000000` en los tres nodos y en las dos
familias**; llevar a cero sí duele (−0,0026 en `op9`/R1r, **−0,0508** en `op9`/R2r).

> **Más buffer no compra nada; menos buffer sí duele.** El contrato está **saturado hacia arriba**,
> así que **no hay nada por encima del incumbente que capturar — para ningún controlador,
> estructurado o neuronal**. El negativo es una propiedad medida del contrato, no un fracaso de los
> métodos, y es también por qué el techo clarividente vale sólo +0,001.

**Eso acota el paso 4 antes de entrenar.** Si la KAN gana algo material aquí, el sitio para buscar
el error es el experimento.

### DDMRP: lo que se midió primero no era DDMRP

En la corrida agrupada el brazo emitía **una sola postura**, `(1344, 1344, 504)`, por un desajuste
de escala: la materia prima corre en millones y la escalera de la Tabla 6.16 tapa **28× por
debajo**, así que la proyección lo aplastaba contra el peldaño superior.

Se le sacó del dominio compartido (`results/ddmrp_unprojected_v1/`, enmienda del 6-ago): ahora emite
**114 / 209 / 282 objetivos distintos** y supera el techo antiguo en **312 de 312** puntos. **La
asimetría se declara**: tiene MÁS derechos que los demás brazos, así que una victoria suya no sería
superioridad del método — pero **una derrota es evidencia más fuerte**. Y pierde: **−0,000303 en
R1r (0/12 tapes)** y exactamente indistinguible en R2r.

## 4. El preprint de KAN

[arXiv:2407.16674](https://arxiv.org/abs/2407.16674) **sigue siendo preprint** — sin journal-ref, sin
DOI externo, y sus autores lo etiquetan «Technical Report». **Dos años sin publicar.**

**Pero no lo usemos para desestimarlo**: replicamos su hallazgo en nuestra propia cadena. La postura
defendible es *«coincidimos con ellos en un dominio que ellos no tocaron»*. Y su ablación juega a
nuestro favor: la ventaja de la KAN viene de **la base B-spline**, y un MLP con B-splines la iguala.

**Hueco de novedad, formulado de manera auditable:** la búsqueda documentada no encontró un
estudio *peer-reviewed* que combine KAN, medición de SCRES y DES. No es una afirmación universal de
ausencia y debe acompañarse de fecha, bases y consulta (`docs/REVISION_LITERATURA_KAN_2026-08-06.md`).

## 5. La KAN: pierde como política, gana como surrogate

Su argumento tenía dos patas. **Una cae y la otra se sostiene**, y la frontera entre ambas es
justamente la contribución.

**Ahorro de parámetros como política de control: medido, y no se sostiene.** A 200.000 parámetros
igualados, KAN − MLP = **−0,475 [−1,548 · +0,598]**, y cuesta **4,1×** por decisión.

**Como surrogate supervisado de la superficie de diseño: ventaja descriptiva clara.** A **532
contra 529 parámetros**, evaluado sobre una partición aleatoria retenida
(`results/kan_interpretability/`, desarrollo, sin adjudicación):

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

Las curvas mostradas son **cortes de respuesta** con las otras coordenadas fijadas en su mediana;
no son, por sí solas, las funciones de arista internas de la KAN. El control barajado confirma que
la superficie real contiene señal predictiva (R² retenido −0,556 a −1,938 al barajar), pero una
sola partición y una distancia de curvas no demuestran estabilidad ni fidelidad interpretativa.
Antes de citarlas como mecanismo hacen falta CV agrupada/por ejes, escalado ajustado sólo en
entrenamiento y estabilidad de las formas entre folds.

De paso confirma el resultado de turnos por otra vía: el recorrido de `shifts` en las curvas es
**0,05–0,28** frente a **1,5–2,6** de `op9_rop`. El turno es la variable que menos mueve la
superficie.

## 6. Lo demás que pidió

* **Turnos como proxy Pareto:** en **5 de 6 contextos no hay ahorro** — la config más barata dentro
  del 1 % usa los mismos turnos que el óptimo. Y bajo **R1r los turnos no compran nada** (0,00926 con
  1, 2 o 3), mientras bajo R2r sí. **El valor del turno lo decide la familia de riesgo.**
  (`results/shift_pareto_diagnostic/`)
* **Riesgo de acumular inventario al final del horizonte — medido, y la respuesta tiene dos caras.**
  Dentro del dominio compartido **ningún controlador acumula**: todos terminan con MENOS inventario
  que la media estática (R1r 4,52 M el MPC contra 4,63 M estática; R2r 4,43 M contra 4,50 M), y el
  clarividente es el más bajo de todos. **Pero DDMRP sin proyectar sí acumula**, y mucho: termina
  con **+1,02 M (R1r) y +1,27 M (R2r)** sobre su versión proyectada — **y su `ret_excel_full_ledger`
  es bit-idéntico**. Es decir, **un millón largo de unidades extra que compran exactamente cero
  resiliencia.** Su preocupación se materializa precisamente cuando se le quita el tope al método,
  y es la demostración más limpia de la saturación.
* **Métrica de resiliencia:** 162 derivaciones de sus dos métricas + 661 recurvaturas. Bajo la
  curvatura publicada no aparece el *headroom* requerido (0,0000 en 288, 0,0195 en la extendida).
  La lectura de curvatura sigue siendo **descriptiva**: el artefacto endurecido falla el falsador de
  monotonicidad, por lo que no autoriza el veredicto «únicamente la postura amante del riesgo».

## 7. Lo que le debemos y va con retraso

Compromisos del 28 de julio, vencidos el 3 de agosto:

| responsable | compromiso | estado |
|---|---|---|
| Thomas | nodos y variables de decisión | **parcial** — rejilla 4.608 hecha; **capacidad finita en `wdc`/`al`/`sb` sin cablear** |
| **David** | KAN con grid search **en el entorno ampliado** | **no hecho** — su corrida va 2/5 y está en el entorno **viejo** |
| Thomas | preprint de KAN | **entregado hoy**, con retraso |
| Thomas | literatura KAN + resiliencia | **entregado hoy**, con retraso |
