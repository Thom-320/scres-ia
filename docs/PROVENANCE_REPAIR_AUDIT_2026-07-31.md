# Provenance repair — what traces, what contradicts, what I did not touch

**Status:** `AUDIT_COMPLETE_TWO_BLOCKERS_UNRESOLVED`. Executes item 3 of the 2026-07-09
strategy assessment. **No number in the manuscript was changed.** Provenance is not repaired
by editing figures until they agree; it is repaired by tracing each figure to an artifact and
declaring the ones that do not trace.

---

## 0. Headline

| | resultado |
|---|---|
| captions en secciones de resultados | **20** |
| que declaran conteo de semillas | **8 de 20** |
| que citan un artefacto fuente | **0 de 20** |
| contradicciones numéricas encontradas | **2**, ambas bloqueantes |
| placeholders restantes | **4**, todos dependientes de terceros |
| «validación ±15% inventada» | **ya no existe** — verificado, ninguna forma |

**Ninguna tabla ni figura de resultados cita su artefacto.** Ése es el defecto de procedencia,
más que cualquier discrepancia individual: un revisor no puede llegar del número al dato.

## 1. Contradicción bloqueante A — la celda `increased/h104` tiene dos valores

| dónde | mejor estático | Δ |
|---|---:|---:|
| tabla `tab:cross_scenario`, `04_results.tex:639` | **0.003108** | **+0.000552** |
| prosa, `04_results.tex:672` y `:697` | **0.003118** | **+0.000542** |

Ambas parejas son internamente consistentes (`0.003660 − 0.003108 = 0.000552`;
`0.003660 − 0.003118 = 0.000542`), así que **son dos comparadores distintos, no un error de
tecleo.**

**Y la nota al pie declara la convención que resuelve cuál va:** *«taking the stronger static
wherever two runs disagree»*. El estático más fuerte es **0.003118**, que da el delta
**menor**. **La tabla reporta el comparador más débil, es decir el resultado más favorable a
nosotros, en contra de la convención que el propio texto declara.**

No lo corrijo, y la razón importa: **ninguno de los dos valores tiene artefacto citado**, y no
encontré ninguno en `results/` que los contenga. Cambiar la tabla para que coincida con la
prosa sería fabricar procedencia, exactamente el defecto que este trabajo debe cerrar.

**Acción requerida:** localizar la corrida de frontera densa que produjo `0.003118` y
re-generar la fila desde ella, o re-correr la celda. **No enviar mientras persista.**

## 2. Contradicción bloqueante B — Track A lleva dos conteos de semillas

| dónde | qué dice |
|---|---|
| tabla `tab:track_a_boundary`, `04_results.tex:103` | «BC-warm-started PPO (`per_op_conflict`, **3 seeds**)», valor 0.155247 |
| figura 13, `04_results.tex:129` | «best learned PPO (**5 seeds**)» |
| figura 15, `04_results.tex:143` | «learning curves across **five** PPO seeds» |

El artefacto que sí encontré respalda **3**: `docs/PER_OP_CONFLICT_METRICS_AUDIT_2026-06-29.md`
documenta `--seeds 1,2,3`, la carpeta `..._3seed_richmetrics_2026-06-29`, y describe la
corrida como *«the richer 3-seed rerun»*, con el mismo 0.155254 del estático.

**Pero eso no cierra la contradicción**, porque la figura 13(a) podría estar graficando una
corrida PPO **distinta** de la fila de la tabla. Si son dos corridas, el texto debe decirlo;
si es una, un caption está mal. **No puedo decidirlo sin la fuente de la figura.**

**Nota:** el audit numérico del 2026-07-07 marcó esta tabla como «OK» citando dos documentos —
pero **verificó la tabla y no los captions de las figuras**, que es donde vive la
contradicción.

## 3. Lo que sí verifiqué como limpio

* **`±15%` inventado: eliminado.** Buscado en todas sus formas (`15\%`, `15 %`, `pm 15`,
  `±15`) en `main.tex` y las nueve secciones. **Cero coincidencias.**
* **Track B, 10 semillas:** la tabla principal traza a
  `docs/track_b_q1_stats_2026-07-02_final_10seed/effect_sizes.csv`, verificado en el audit de
  julio sobre siete métricas.
* **Ablación de espacio de acción, 5 semillas:** traza a
  `docs/track_b_q1_stats_2026-07-02_final/e4_ablation_summary.csv`.
* **La matriz `tab:cross_scenario` declara su protocolo** en el caption: CRN-emparejado,
  CI95 agrupado por semilla sobre cinco deltas por semilla. Eso es correcto y es más de lo
  que hacen la mayoría de los captions.

## 4. Lo que falta y no es contradicción, sino ausencia

**12 de 20 captions no declaran conteo de semillas.** Incluye
`tab:track_a_boundary` (la tabla lo dice en la celda, no en el caption), la reconstrucción
DES, la sensibilidad de costos con 120 pares, y las dos tablas de `05_results.tex`.

**0 de 20 citan artefacto.** La convención mínima para C&IE debería ser una línea por caption:
`Source: results/<ruta>/result.json (sha …)`. Con el sellado que ya existe en
`arm_runner.seal_and_write`, eso es mecánico para todo resultado nuevo; para los antiguos hay
que localizar cada bundle.

## 5. Los 4 placeholders, y por qué no los toco

`main.tex:52` autores y roles · `:62` financiamiento · `:73` re-pin del commit `10c7de9` ·
`:80` agradecimiento a Garrido.

**Los cuatro dependen de terceros** — PI y Garrido — y rellenarlos yo sería inventar
autoría, financiamiento y agradecimientos. El del commit sí es mío pero debe fijarse **al
congelar la submission**, no antes.

## 6. Veredicto

**La procedencia NO está reparada, y el bloqueo es real:** dos contradicciones numéricas cuyo
lado correcto no puedo determinar sin el artefacto de origen, y ninguna tabla de resultados
enlazada a su fuente.

Lo que sí queda cerrado: el `±15%` inventado no existe, y las dos contradicciones están
**localizadas al número y a la línea**, que es lo que hacía falta para poder repararlas.

**Recomendación:** de las dos, **A es la urgente** — reporta el comparador que nos favorece
en contra de la convención declarada en su propia nota al pie. Si un revisor la encuentra
antes que nosotros, cuesta mucho más que corregirla ahora.
