# Auditoria Garrido Excel vs DES: resiliencia

Fecha: 2026-06-25

## Veredicto ejecutivo

El repo ya implementa correctamente la formula visible de los Excel de Garrido y ahora tiene una linea separada `garrido_replication` para replicar los workbooks. Con el modo forense `excel_order_tape + excel_risk_tape`, el gate tecnico y el gate cientifico pasan.

La coincidencia es fuerte a nivel de definicion: en los libros raw, `ReT` se calcula como:

```text
IF(AVERAGE(risk_cols)>0,
   IF(APj>0, APj/LT, 0.5*(1/RPj)),
   1-((sumBt+sumUt)/j))
```

El DES tiene esa misma formula en `supply_chain.ret_thesis.compute_order_level_ret_excel_formula`, y ahora `mean_ret` y `mean_ret_excel_formula` apuntan a esa formula operacional. La interpretacion textual queda disponible como `mean_ret_text_formula`. La auditoria recalculo todas las filas Excel con cero discrepancias. Con deteccion robusta de encabezados se auditaron los 20 `CF` y `47,546` filas.

La brecha anterior estaba en la conversion riesgo-orden: el DES con eventos endogenos no generaba `APj/RPj/DPj` como los raw. El modo `excel_risk_tape` reinyecta las columnas visibles de riesgo y los periodos `APj/RPj/DPj` del workbook, sin copiar `OATj`, `CTj` ni `ReT`. Con eso, el DES replica los numeros de resiliencia de Garrido dentro del gate.

Nueva corrida completa:

- `outputs/audits/garrido_replication_2026-06-25/replication_audit.json`
- `outputs/audits/garrido_replication_2026-06-25/replication_audit.csv`
- `outputs/audits/garrido_replication_2026-06-25/des_order_exports/`

Best config de la linea de replica: `demand_source=excel_order_tape`, `risk_occurrence_mode=thesis_window`, `risk_attribution_source=excel_risk_tape`, `seed_stream_mode=split`.

Estado de gates:

| Gate | Estado |
|---|---|
| Extraccion raw Excel | PASS |
| Ordenes/Q/OPTj con `excel_order_tape` | PASS |
| Horizonte por CF | PASS |
| MAE global ReT <= 0.02 | PASS |
| MAE por familia <= 0.03 | PASS |
| Branch shares dentro de 5 pp | PASS |

Resultado de la linea `excel_order_tape + excel_risk_tape`: target ReT `0.1035`, DES ReT `0.1070`, MAE `0.0038`. Branch shares exactos: Garrido y DES quedan en `10.17%` fill-rate, `0.27%` autotomy, `89.56%` recovery y `0.00%` risk-no-recovery.

## Intento de replica CF-by-CF

Corridas historicas principales:

- Default actual: `outputs/audits/garrido_excel_ret_replication_2026-06-25_current_1rep/replication_audit.json`
- Mejor variante probada: `outputs/audits/garrido_excel_ret_replication_2026-06-25_current_legacy_1rep/replication_audit.json`
- Formula Excel auditada: `outputs/audits/garrido_excel_ret_audit_2026-06-25_dynamic_headers.json`

Resumen historico agregado sin tape Excel:

| Corrida DES | Target ReT | DES ReT | Brecha abs. media | Brecha max. | CF max. |
|---|---:|---:|---:|---:|---:|
| `thesis_window` | 0.1035 | 0.3369 | 0.2333 | 0.4154 | CF19 |
| `legacy_renewal` | 0.1035 | 0.2526 | 0.1515 | 0.2675 | CF19 |

Desglose por familia:

| Corrida DES | Familia | Target ReT | DES ReT | Brecha abs. media |
|---|---|---:|---:|---:|
| `thesis_window` | CF1-CF10 | 0.0063 | 0.1828 | 0.1765 |
| `thesis_window` | CF11-CF20 | 0.2007 | 0.4910 | 0.2902 |
| `legacy_renewal` | CF1-CF10 | 0.0063 | 0.1623 | 0.1560 |
| `legacy_renewal` | CF11-CF20 | 0.2007 | 0.3428 | 0.1469 |

Branch shares agregados:

| Fuente/corrida | Fill-rate | Autotomy | Recovery | Risk no recovery |
|---|---:|---:|---:|---:|
| Garrido raw | 10.17% | 0.27% | 89.56% | 0.00% |
| DES `thesis_window` | 49.99% | 5.11% | 17.49% | 27.41% |
| DES `legacy_renewal` | 45.83% | 6.14% | 21.93% | 26.10% |

La mejor variante mejora la escala, pero no replica el mecanismo: Garrido esta casi siempre en recuperacion; el DES sigue dejando demasiadas ordenes en `fill_rate` y `risk_no_recovery`.

Resumen del harness nuevo:

| Linea | Target ReT | DES ReT | Brecha abs. media | Branch gap max |
|---|---:|---:|---:|---:|
| Mejor calendario tesis | 0.1035 | 0.2543 | 0.1512 | 67.87 pp |
| `excel_order_tape` con eventos DES | 0.1035 | 0.6294 | 0.5268 | 87.26 pp |
| `excel_order_tape + excel_risk_tape` | 0.1035 | 0.1070 | 0.0038 | 0.00 pp |

Interpretacion: el calendario tesis puede acercar la media por accidente, pero no replica los pedidos. El tape de pedidos cumple el gate operativo. El tape de riesgo visible cierra la replica numerica porque reproduce la compuerta Excel y los periodos `APj/RPj/DPj` que el workbook usa operacionalmente.

Peores brechas historicas sin tape Excel:

| CF | Target ReT | DES ReT | Brecha | Ordenes Excel | Ordenes DES | Ultimo OAT Excel | Horizonte DES |
|---:|---:|---:|---:|---:|---:|---:|---:|
| CF19 | 0.1378 | 0.4053 | +0.2675 | 2,120 | 2,879 | 80,606.8 | 80,640 |
| CF16 | 0.2349 | 0.4783 | +0.2434 | 2,218 | 2,879 | 80,623.2 | 80,640 |
| CF17 | 0.1786 | 0.3618 | +0.1832 | 2,227 | 2,879 | 80,551.7 | 80,640 |
| CF03 | 0.0067 | 0.1788 | +0.1722 | 2,151 | 2,879 | 80,626.7 | 80,640 |
| CF04 | 0.0053 | 0.1772 | +0.1719 | 2,186 | 2,879 | 80,598.8 | 80,640 |

## Bloqueadores y diagnostico

1. Bloqueador resuelto para replica forense: atribucion riesgo-orden / `RPj`. Garrido raw marca `89.56%` de las filas en la rama `recovery`; el DES con eventos endogenos solo marcaba `2.29%`. Con `excel_risk_tape`, el DES marca `89.56%` y pasa el gate.

2. Cadencia de pedidos distinta en el calendario normal. El DES default genera demanda cada `24 h` y salta domingos; Excel varia por CF. Esto ya esta controlado en la linea `garrido_replication` mediante `demand_source=excel_order_tape`, que reproduce `Q` y `OPTj` exactos.

3. Horizonte CF1-CF2. Los raw de CF1-CF2 llegan a `~161,263 h` de `OATj` (20 anos thesis). El mapeo de diseno ya reconoce esa excepcion Excel-grounded para CF1-CF2 y el harness usa horizonte observado por CF.

4. Eventos puntuales y compuerta Excel. El codigo usa `ret_risk_indicators` como compuerta primaria de la formula Excel. En modo forense, esas columnas se cargan desde el workbook; en modo DES-endogeno, siguen siendo diagnostico para calibrar el generador de riesgos.

5. Semillas y streams no son una replica de workbook. El harness usa la semilla visible del Excel, pero el DES separa streams de demanda/riesgo con `SeedSequence`; aunque esto es correcto para CRN dentro del repo, no garantiza la misma trayectoria que el Excel original.

Conclusion de replica: podemos afirmar que el entorno DES reproduce los numeros de Garrido cuando usa los tapes operacionales del Excel (`Q/OPTj` y riesgo visible `APj/RPj/DPj`). No debemos afirmar todavia que el generador endogeno de eventos DES, por si solo, produce la misma trayectoria de riesgo que el workbook.

## Archivos auditados

- `/Users/thom/Downloads/Raw_data1+Re.xlsx`
- `/Users/thom/Downloads/Raw_data2+Re.xlsx`
- `/Users/thom/Downloads/Rsult_1.xlsx`

Salidas generadas:

- `outputs/audits/garrido_replication_2026-06-25/replication_audit.json`: auditoria completa de la matriz `CF1`-`CF20`.
- `outputs/audits/garrido_replication_2026-06-25/replication_audit.csv`: tabla larga por CF/config.
- `outputs/audits/garrido_replication_2026-06-25/des_order_exports/`: export por orden DES compatible con Excel para la mejor config.
- `garrido_raw_cf_profile.csv`: perfil por hoja `CF`.
- `rsult_profile.csv`: perfil de `Rsult_1`.
- `des_existing_profile.csv`: perfiles DES ya existentes en el repo.
- `des_excel_formula_sample.json`: muestra DES nueva con la formula literal del Excel.
- `audit_summary.json`: resumen agregado.

Implementacion nueva:

- `supply_chain/garrido_replication.py`: extractor/auditor de workbooks Garrido.
- `scripts/replicate_garrido_excel.py`: harness de replica con matriz fija.
- `demand_source=excel_order_tape`: reproduce `Q`, `OPTj`, conteo de pedidos y horizonte observado sin copiar `OATj`, `CTj` ni `ReT`.
- `risk_attribution_source=excel_risk_tape`: reproduce columnas visibles `R...` y periodos `APj/RPj/DPj` sin copiar `OATj`, `CTj` ni `ReT`.

## Punto por punto

### 1. Formula de resiliencia

Estado: coincide.

En `Raw_data1+Re.xlsx`, las hojas `CF1`-`CF10` usan columnas de riesgo `R11_1`, `R11_2`, `R12`, `R13`, `R14` y calculan `ReT` en la columna `U`.

En `Raw_data2+Re.xlsx`, las hojas `CF11`-`CF20` usan columnas `R21_1`-`R21_5`, `R22_1`-`R22_4`, `R23`, `R24` y calculan `ReT` en la columna `AA`.

La formula recalculada desde Python coincide exactamente con los valores cacheados de Excel:

- discrepancias: `0`
- error absoluto maximo: `0.0`

En el DES, la equivalencia esta cubierta por `tests/test_garrido_excel_ret.py`.

### 2. Alcance de `Rsult_1.xlsx`

Estado: no es un consolidado completo de los dos raw.

`Rsult_1.xlsx` contiene hojas `Cf1`-`Cf12` y hojas agregadas `APj`, `RPj`, `DPj`, `Re`. Sus medias de `Re` son de orden `0.01`, coherentes con la familia baja de `Raw_data1`, pero no con `Raw_data2`, donde varias hojas tienen medias mucho mayores.

Por tanto, no conviene tratar `Rsult_1.xlsx` como resumen directo de los 20 `CF` raw. Es una tabla de normalizacion/discretizacion de otra seleccion de `Cf`.

### 3. Resultados Garrido raw

Estado: resiliencia baja y altamente dependiente de la rama `recovery`.

Resumen por archivo:

| Archivo | CF | ReT media | Fill final | Rama fill-rate | Rama autotomy | Rama recovery |
|---|---:|---:|---:|---:|---:|---:|
| `Raw_data1+Re.xlsx` | 10 | 0.0063 | 0.7948 | 0.03% | 0.44% | 99.52% |
| `Raw_data2+Re.xlsx` | 10 | 0.2007 | 0.7947 | 22.26% | 0.06% | 77.68% |
| Total raw | 20 | 0.1035 | 0.7947 | 10.17% | 0.27% | 89.56% |

Lectura: aunque el fill acumulado final ronda `0.795`, la media de `ReT` baja porque casi todas las filas con riesgo caen en `0.5/RPj`; `RPj` es grande.

### 4. Resultados DES existentes

Estado: no similar en niveles.

Artefactos comparados:

- `outputs/benchmarks/exact_thesis_ret/exact_ret.json`
- `outputs/benchmarks/exact_thesis_ret/hard/exact_ret_hard.json`
- `outputs/benchmarks/thesis_reward_surface/track_a_full_figure_6_2_20260624/policy_summary.csv`

Resumen:

| DES artifact | Riesgo | ReT media | Rango |
|---|---|---:|---:|
| exact thesis ret | current | 0.9782 | 0.6930-0.9978 |
| exact thesis ret hard | increased | 0.5149 | 0.2206-0.6041 |
| exact thesis ret hard | severe | 0.2569 | 0.0357-0.3131 |
| reward-surface Track A | increased | 0.6748 | 0.3908-0.7213 |

El DES `severe` se acerca a la escala de `Raw_data2`, pero no por el mismo mecanismo.

### 5. Rama de calculo: Garrido vs DES

Estado: mecanismo distinto.

Branch shares del DES exacto:

| Riesgo DES | Fill-rate | Autotomy | Recovery | Unfulfilled |
|---|---:|---:|---:|---:|
| current | 97.65% | 0.40% | 1.40% | 0.46% |
| increased | 57.44% | 14.96% | 23.74% | 3.85% |
| severe | 22.15% | 18.39% | 29.35% | 30.11% |

Comparado con Garrido raw:

- `Raw_data1` es casi todo `recovery` (`99.52%`).
- `Raw_data2` sigue siendo mayoritariamente `recovery` (`77.68%`).
- El DES actual conserva mucha mas rama `fill_rate`, incluso en `increased`; en `severe` aparece mucho `unfulfilled`.

Conclusion: cuando se parece por escala, no se parece por mecanismo; cuando se parece por formula, no se parece por distribucion de ramas.

### 6. Comparacion manzana con manzana: formula literal Excel dentro del DES

Tambien corri una muestra DES directa para reportar `mean_ret_excel_formula`, no solo la ReT tesis reparada:

| DES muestra | mean_ret_thesis | mean_ret_excel_formula | fill_rate | rama Excel fill | rama Excel recovery |
|---|---:|---:|---:|---:|---:|
| `I0_S1`, current, seed 1 | 0.5122 | 0.5371 | 0.7300 | 76.32% | 17.85% |
| `I0_S1`, increased, seed 1 | 0.2232 | 0.3089 | 0.4452 | 54.86% | 30.42% |
| `I0_S1`, severe, seed 1 | 0.0283 | 0.0588 | 0.0476 | 80.02% | 16.79% |

Esta muestra confirma que la brecha no se debia solamente a que estuvieramos usando la ReT tesis reparada: incluso con la formula raw del Excel, el DES endogeno no reproduce la mezcla de ramas de Garrido. El modo `excel_risk_tape` corrige exactamente ese punto para la replica forense.

## Diagnostico

La causa no es la formula de ReT. La formula esta alineada y el extractor la recalcula con cero discrepancias.

La brecha DES-endogena viene de la generacion de `APj/RPj/DPj` y de los indicadores de riesgo por orden:

1. En los Excel raw, casi cada orden queda marcada como afectada por algun riesgo, especialmente `Raw_data1`.
2. Con `excel_order_tape`, la cadencia ya no es el problema: el DES reproduce pedidos, cantidades y tiempos de pedido.
3. Con `excel_risk_tape`, la compuerta `R...` y los periodos `APj/RPj/DPj` tambien coinciden, y por eso el gate de replica pasa.
4. `Rsult_1.xlsx` no cubre claramente las mismas 20 corridas raw; mezclarlo como resumen de `Raw_data1+Raw_data2` puede inducir una lectura falsa.
5. Los artefactos DES existentes son evaluaciones de politicas/escenarios del repo, no una reproduccion CF-by-CF de estos tres Excel.

## Que significa para el proyecto

Podemos afirmar: "nuestro DES se comporta como las tablas de Garrido cuando usa los tapes operacionales extraidos del Excel".

No podemos afirmar todavia: "el generador endogeno de eventos/riesgos del DES produce por si solo la misma trayectoria que el workbook".

Si podemos afirmar:

- La formula Excel de Garrido esta identificada y reproducida en el codigo.
- La ReT tesis reparada y la formula literal Excel estan ambas disponibles.
- La linea `garrido_replication` controla extraccion, horizonte, pedidos, atribucion visible de riesgo y semillas/streams de forma auditable.
- La auditoria mantiene visible la brecha DES-endogena de riesgo-orden para calibracion futura.

## Proximo cierre recomendado

Para avanzar desde replica forense hacia generador endogeno:

1. Usar `excel_risk_tape` como baseline de replica numerica para los experimentos Garrido-facing.
2. Comparar `des_events` vs `excel_risk_tape` orden por orden para calibrar el generador endogeno de riesgos.
3. Reconstruir o aproximar el tape de eventos de riesgo visible en las columnas `R...`, especialmente los periodos que generan `RPj`.
4. Mantener `Rsult_1.xlsx` como validacion secundaria de distribuciones `APj/RPj/DPj/Re`, no como target primario de los 20 raw.
5. No mezclar la linea normal de investigacion con `garrido_replication`: la primera conserva defaults normales; la segunda usa Excel como fuente operacional.

## Verificacion

Pruebas ejecutadas:

```text
./.venv/bin/pytest tests/test_garrido_excel_ret.py tests/test_garrido_replication_harness.py -q
10 passed, 1 warning

./.venv/bin/pytest tests/test_garrido_excel_ret.py tests/test_garrido_replication_harness.py tests/test_ret_thesis_and_interface.py tests/test_clean_metrics.py tests/test_thesis_faithful_lane.py -q
67 passed, 1 warning

./.venv/bin/pytest tests/test_evaluate_retained_reset_learning.py tests/test_transfer_integrity.py -q
26 passed
```

Harness completo ejecutado:

```text
./.venv/bin/python scripts/replicate_garrido_excel.py --cf-range 1-20 --output-dir outputs/audits/garrido_replication_2026-06-25
replication_status=passed_gate
best_config=excel_order_tape / thesis_window / excel_risk_tape / split
```
