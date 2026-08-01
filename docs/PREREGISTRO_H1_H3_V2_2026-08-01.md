Preregistro — H1 y H3, segunda formulación

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_h1_h3_v2.py`.
Sucede a `docs/RESULTADO_H1_H3_2026-08-01.md`, que se detuvo con dos falsadores en FALLA.

## Por qué hay una segunda formulación, y qué la hace legítima

Cambiar el estimando **después** de ver un resultado es precisamente lo que un preregistro existe
para disciplinar. Así que digo exactamente qué cambia, por qué, y qué **no** rescata:

| falsador que falló | causa | qué se cambia |
|---|---|---|
| `f3` — `system_ttr` censurado al 100 % | ningún clúster de recuperación cierra nunca; la media vale 0 por vacuidad | **métrica nueva**: `service_loss_auc_ration_hours`, sin censura por construcción |
| `f1` — brazos idénticos | se comparaba **una sola configuración modal** por brazo | **diseño nuevo**: comparar la configuración que cada estrategia eligió **en cada celda** (contexto × réplica) |

**Dato de factibilidad comprobado antes de diseñar** (y lo declaro porque lo miré): de las **72**
celdas, **42 (58 %) despliegan configuraciones distintas** entre híbrido y estático, con **23**
configuraciones distintas en total. La comparación es real; lo que la anulaba era colapsar todo a
la moda.

## H1 — segunda formulación

> **H1' .** La configuración que despliega el aprendiz con memoria pierde **menos servicio
> acumulado** que la que despliega el diseño estático de la tesis, bajo el mismo régimen de
> riesgo y las mismas semillas.

**Métrica:** `service_loss_auc_ration_hours` = `Σ_j qty_j · max(0, fin_j − (OPTj + LT))`, **menor
es mejor**. Está ya en el panel, integra **todos** los pedidos y **no puede censurarse**: un
pedido que nunca se sirve acumula pérdida hasta el horizonte en vez de desaparecer.

**Y digo lo que NO es:** no es un «tiempo de recuperación». Es la **integral del servicio
perdido**, que mezcla magnitud y duración. Es una operacionalización defendible de *«se recupera
mejor»*, pero **no** es el mismo estimando que `H1` enuncia, y el paper tiene que decirlo así.
Se reporta `service_loss_auc_per_order` al lado.

**Diseño:** para cada celda (contexto × réplica) se evalúa la configuración que **cada brazo
eligió ahí**, sobre semillas vírgenes `5 800 001…`, pareado por (contexto, réplica, semilla).

* **primario**: las **72** celdas, incluidas las 30 idénticas. Es **conservador** — esas aportan
  exactamente cero y arrastran el contraste hacia el nulo — y es lo que un lector esperaría.
* **secundario**: las **42** celdas con configuraciones distintas, que es donde hay algo que medir.

Ambos se reportan. **No elijo cuál titular después de verlos.**

## H3 — reescrita, y con el cambio declarado

El borrador dice: *«Learning-enabled models reduce performance variance across heterogeneous
disruption intensities»*. Esa forma **no es comprobable aquí**: exige que el aprendiz despliegue
algo distinto y, con el óptimo invariante medido en toda la campaña, converge al mismo punto.

> **H3' .** El aprendiz con memoria reduce la **varianza del COSTE DE BÚSQUEDA entre contextos**
> frente al aprendiz reiniciado y frente al diseño OFAT.

Es fiel al espíritu —«el aprendizaje estabiliza el desempeño ante disrupciones heterogéneas»— con
el desempeño leído donde este entorno **sí** lo hace variar: **cuánto cuesta encontrar el óptimo**
en cada contexto. **Es un cambio de constructo, no una reparación**, y así va etiquetado.

**Estimando:** por réplica, la **varianza entre los seis contextos** de `runs_to_within_1pct`;
diferencia pareada por réplica, bootstrap. Sale del artefacto **ya sellado**
`results/garrido_meta_learner_v2/result.json` — **sin simulación nueva**.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_some_cells_deploy_different_configurations` | si ninguna celda difiere, `H1'` vuelve a ser la tautología que detuvo la v1 |
| `f2_identical_cells_contribute_exactly_zero` | una celda con configuraciones idénticas **debe** dar diferencia 0,0; cualquier otra cosa significa que el pareado está roto |
| `f3_service_loss_auc_is_not_censored` | si hubiese pedidos fuera de la integral, la métrica heredaría el defecto que vinimos a evitar |
| `f4_the_metric_discriminates_between_deployed_configs` | si las 23 configuraciones puntúan igual, no hay nada que comparar |
| `f5_contexts_differ_in_search_difficulty` | `H3'` necesita que el coste de búsqueda **varíe** entre contextos; si es plano, su varianza es ruido |
| `f6_h3_reads_the_sealed_artifact_unmodified` | recalcular la búsqueda aquí permitiría moverla; se verifica el sello |
| `f7_seeds_are_virgin` | reutilizar semillas invalidaría la confirmación |

## Regla de lectura, fijada de antemano

* **`H1'`**: híbrido con menor `service_loss_auc` que estático, `LCB95 > 0` en la diferencia
  pareada del conjunto **primario** → **sostenida**. Si sólo se sostiene en el subconjunto de
  celdas distintas, se dice **exactamente eso** y no se titula como global.
* **`H3'`**: varianza del coste de búsqueda del híbrido **menor** que la del reiniciado, con
  `LCB95 > 0` → **sostenida**. El contraste contra OFAT se reporta como secundario.
* **Cualquier falsador en FALLA** → no se reporta esa hipótesis como medida, igual que ayer.

**Y una advertencia que me impongo:** el estático sigue siendo **el diseño de Garrido, bien
implementado**. Que en el 42 % de las celdas coincida con el aprendiz ya es un resultado sobre lo
bueno que es ese diseño, y va dicho aunque `H1'` salga a favor del híbrido.
