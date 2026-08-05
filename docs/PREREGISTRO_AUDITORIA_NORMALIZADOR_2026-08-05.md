# Preregistro — auditoría del normalizador del meta-aprendiz (P0.2), y los dos gates del carril

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_meta_learner_normaliser_audit_v1.py`.
**Semillas:** bloque quemado `5.300.001–012`, réplica declarada de `garrido_q2_des288`. **Ninguna
nueva.** Screen de **desarrollo**; no adjudica y no autoriza nada.

## 1. El defecto que motiva esta auditoría

En [`run_meta_learner_over_configs_v1.py:189`](../scripts/run_meta_learner_over_configs_v1.py:189):

```python
values = [v for v, _ in table]                 # las 288, incluidas las NO corridas
best, lo, span = max(values), *scaled(values)
...
neuron.update(features(CONFIGS[idx]), (value - lo) / span)   # línea 236
```

El objetivo del aprendiz se normaliza con el **min/max de la superficie completa**, que incluye
configuraciones que el brazo nunca ejecutó. Es carga estructural, no cosmética: `ret_excel_risk_
conditional` vale ~0,009 en los contextos R1r, así que sin ese reescalado cada paso de gradiente
sería `(0,009 − σ(0)) ≈ −0,49` y `ρ` colapsaría uniformemente negativo. **La neurona funciona en
parte porque se le entrega el rango de la superficie.**

Por qué importa exactamente aquí: `docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md` ya fijó la
regla al retirar la v1 — una fuga compartida por dos brazos no invalida su contraste, pero
**`memory_vs_ofat` y `memory_vs_random` sí quedan expuestos, porque OFAT y azar no reciben esa
información**. Es la tercera fuga de la misma familia. El sucesor
(`run_garrido_q2_des288_v1.py:85`) ya la corrigió con `TARGET_SCALES` fijas a priori.

## 2. Lo que se mide

**Brazo A — `oracle`**: el normalizador tal como está hoy. Sirve de **reproducción**, no de ciencia.

**Brazo B — `prefix`**: `lo` y `span` calculados **sólo sobre los valores ya observados en ese
contexto**. Regla de arranque, fijada aquí: **no se aplica actualización hasta haber observado al
menos dos valores distintos**; a partir de ahí `span = max(hi − lo, 1e-12)`. El normalizador es
entonces no estacionario, que es exactamente la situación honesta de un planificador.

Todo lo demás es idéntico: mismas semillas, misma superficie CRN, mismo `default_rng(90_000 + r)`
por réplica, mismo presupuesto 24, mismos cuatro brazos de búsqueda.

## 3. El ancla externa, sin la cual esto no prueba nada

Antes de leer cualquier número del brazo B:

> **`f1_harness_reproduces_the_sealed_artifact`** — con el normalizador `oracle`, este arnés debe
> reproducir `results/garrido_meta_learner_v2/result.json` **en las cuatro medias**: memoria
> 6,986111, reset 14,888889, OFAT 12,416667, azar 19,541667.
>
> *Por qué puede fallar:* cualquier deriva del simulador, de la métrica, del CRN o del orden de
> consumo del RNG desde que se selló ese artefacto rompe la igualdad. Y si falla, **el número del
> brazo B no significa nada**, porque no sabríamos si la diferencia viene del normalizador o de la
> física. Es el único chequeo aquí que no puede satisfacerse por acuerdo del código consigo mismo:
> el ancla la escribió una versión anterior del árbol.

## 4. Primarios, y por qué se cambia el que había

`runs_to_within_1pct` **imputa `budget + 1 = 25`** cuando el objetivo no se alcanza. Las tasas de
censura difieren brutalmente por brazo, así que la media no es comparable entre brazos.

**Primario nuevo: `auc_regret_norm`** — área bajo la curva de regret dividida por
`budget · |best|`, sin censura y definida para toda celda. **Secundario:** regret simple final.
**Terciario, sólo con sus tasas de censura declaradas:** `runs_to_within_1pct`.

Unidad de inferencia: la **réplica** (agrupa sus seis contextos). Bootstrap de bloques de réplica,
5.000 remuestreos, LCB95.

## 5. Los dos gates del carril, computados sobre la misma superficie

Se declaran aquí porque son **go/no-go de todo el programa**, no resultados accesorios.

* **`g1_optimum_moves_across_contexts`** — el argmax de la superficie debe diferir entre contextos
  por encima del ruido de semilla. *Puede fallar*, y la evidencia previa dice que **fallará**: sobre
  los `chosen_config` de v2, `buffer_hours = 1344` en 247/288 y `op9_rop = 12` en 223/288. Si el
  óptimo no se mueve, «la memoria gana» significa **«la respuesta es siempre la misma»**, y hay que
  escribirlo así.
* **`g2_surface_is_non_separable`** — R² de un modelo aditivo en los cuatro factores contra uno con
  interacciones pareadas, sobre las medias por configuración. *Puede fallar.* Si la superficie es
  separable, **OFAT es casi óptimo por construcción** y ningún método de búsqueda puede ganarle por
  mucho; el mecanismo que justifica el carril sería falso.

## 6. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_harness_reproduces_the_sealed_artifact` | ancla externa; ver §3 |
| `f2_prefix_normaliser_never_reads_an_unrun_value` | reproducir con la superficie **reescalada afínmente** por un factor positivo aleatorio por contexto: la secuencia de visitas del brazo `prefix` debe ser **invariante** y la del brazo `oracle` también (el reescalado afín preserva su min/max relativo), pero un brazo que leyera un estadístico global **no invariante** cambiaría. Se reporta la comparación explícita |
| `f3_the_two_arms_differ_only_in_the_normaliser` | los brazos no neuronales (`ofat`, `random`) deben dar secuencias **idénticas** bajo ambos normalizadores. Si difieren, el arnés cambió algo más |
| `f4_censoring_is_reported_not_hidden` | tasas de censura por brazo presentes y distintas de cero para `random`; si salieran todas cero, el primario antiguo no estaría censurado y el cambio de estimando sobraría |
| `f6_surface_twins_have_identical_prefix_paths` | se conservan idénticas las celdas visitadas por la corrida de referencia y se alteran sólo dos celdas de la cola no observada; `prefix` debe conservar su trayectoria y el brazo `oracle` debe reaccionar al cambio de min/max |
| `f5_no_fresh_seeds` | custodia central, réplica declarada de `garrido_q2_des288` |

## 7. Reglas de lectura, fijadas de antemano

* **`prefix` conserva el efecto con LCB95 > 0 en `memory − reset`** → la fuga era cosmética para ese
  contraste, y se puede decir **con una medición**, no con un argumento.
* **`prefix` lo reduce pero mantiene LCB95 > 0** → se publica el número corregido y **se retira el
  anterior**, como ya se hizo dos veces en este carril.
* **`prefix` lo elimina** → `ALZHEIMER_EFFECT_DOES_NOT_SURVIVE_AN_HONEST_NORMALISER`. El positivo
  del paper cae, y el carril pasa a ser la auditoría de fugas. **Es un desenlace publicable y hay
  que reportarlo igual.**
* **En los tres casos**, `memory_vs_ofat` y `memory_vs_random` sólo se reportan del brazo `prefix`.

**Alcance:** desarrollo sobre tapes quemados. No adjudica, no abre semillas, no autoriza aprendices.
