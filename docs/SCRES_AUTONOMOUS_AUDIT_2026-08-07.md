# Auditoría autónoma SCRES-IA — 2026-08-07

**Propósito:** dejar trazabilidad de la verificación del DES, del margen experimental y de la decisión metodológica antes de escribir el artículo.

**Estado:** documento de resultados y decisiones; no es el manuscrito.

**Fecha de ejecución:** `2026-08-07T03:57:55Z`.

**Regla de lectura:**

- **Hecho verificado:** aparece en una fuente local o fue producido por una corrida identificada.
- **Interpretación:** inferencia razonable a partir de hechos verificados, no un resultado causal nuevo.
- **Propuesta:** experimento o modificación futura; no debe citarse como evidencia.
- **No reclamable:** el artefacto contiene información relacionada, pero sus gates no permiten la afirmación.

## 1. Identidad del código y estado de ramas

### Hechos verificados

- `origin/main` resuelve a `89acc813fd9917a0b8e03441850830959177750a`.
- `origin/codex/expanded-contract-comparators-v2` resuelve a `3d5082007276ca0a5d1304585b9f066d8ff6030f`.
- El worktree de auditoría `/home/ubuntu/scres-ia-expanded-v2` estaba en detached HEAD exactamente en `3d5082007276ca0a5d1304585b9f066d8ff6030f`.
- La comparación de grafos reportó `8` commits detrás y `769` commits por delante de `origin/main`.
- El diff frente a `origin/main` contiene `4.919` archivos, `60.345.217` inserciones y `1.404` eliminaciones.
- Después de retirar el symlink `.venv` creado sólo para esta auditoría, el worktree quedó limpio.
- Los hashes SHA-256 de `scripts/run_expanded_contract_comparators_v2.py`, `supply_chain/supply_chain.py` y `supply_chain/episode_metrics.py` coinciden entre `/home/ubuntu/scres-ia` y el worktree v2.
- No se observó un proceso de entrenamiento o corrida científica activo. Sí existe una sesión independiente `claude --remote-control scres`; no se la interrumpió.

### Interpretación

La rama v2 no es un pequeño parche de `main`: es una rama de investigación con miles de artefactos. Por ello, un resultado de v2 no debe presentarse como resultado de `main`, y un checkout local sin Git no puede usarse para probar procedencia histórica.

## 2. Evidencia visual de Garrido 2024

La API de visión devolvió un error `404` porque el modelo configurado `gpt-5.6-luna-pro` no está disponible. No se inventó una lectura visual. Como respaldo, se procesaron las cinco imágenes adjuntas con OCR local (`tesseract`).

### Hechos verificados en las imágenes

- La primera imagen reproduce la Fig. 1: el flujo de construcción de un DES para SCRES, desde elegir la red hasta registrar los resultados.
- La segunda reproduce la Fig. 2: muestra las variables de decisión del diseño experimental y la métrica SCRES, con el aprendizaje `L` pasando de una corrida/configuración a la siguiente.
- La tercera reproduce la Fig. 3: la clasificación de niveles de IA de Garrido.
- La cuarta reproduce la Fig. 4: los drivers de SCRES (`AP`, `RP`, `DP-RP` y `FR`) y la métrica `ReT`.
- La quinta reproduce la Fig. 5: variables de decisión → pesos/función de activación → salida de SCRES.

### Corrección metodológica importante

La Fig. 2 y la Fig. 5 son compatibles con un **bucle externo de optimización de simulación** sobre configuraciones: una corrida produce una métrica y el estado aprendido informa la siguiente configuración. No obligan a interpretar la propuesta como control RL intra-episodio. Esta lectura es más fiel al documento que entrenar una política dentro de una trayectoria y llamarla automáticamente respuesta a Q2.

## 3. Qué dice directamente la tesis de 2017

Se usó la extracción local de `WRAP_Theses_Garrido_Rios_2017.pdf`, no un resumen de terceros.

### Hechos verificados

- La Tabla 6.11 da las frecuencias actuales para R11–R14, R21–R24 y R3, y el horizonte estimado de hasta 20 años/161.280 horas.
- La Tabla 6.12 especifica los niveles actuales/aumentados de riesgo. Por ejemplo, R11 usa `U(1,168)` frente a `U(1,42)`, R21 `U(1,16128)` frente a `U(1,4032)`, R24 `U(1,672)` frente a `U(1,336)` y R3 `U(1,161280)` frente a `U(1,80640)`.
- Las Tablas 6.13–6.15 definen los diseños de riesgo para Cf1–Cf30.
- La Tabla 6.16 define los buffers en Op3, Op5 y Op9 para 168, 336, 504, 672 y 1.344 horas.
- La tesis declara que los buffers se reponen cada `t` igual a uno de esos niveles, con `S=1` en el escenario de inventario.
- Las Tablas 6.20–6.23 definen `S=1,2,3`, cantidades/lot sizes y los diseños Cf61–Cf90.
- El warm-up descrito en §6.8.2 se activa cuando llega el primer pedido `Q=5.000` raciones a Op9; no es simplemente un número fijo de horas.
- La tesis almacena por orden `OPTj`, `OATj`, `CTj`, `LTj`, backorders, unattended orders, `APj`, `RPj`, `DPj` y riesgo asociado.

### Implicación

Los tres libros adjuntos no representan toda la matriz de la tesis. Los libros raw suministrados cubren Cf1–Cf20; el `Rsult_1.xlsx` es un resumen secundario. No es correcto afirmar réplica completa de Cf1–Cf90 sólo porque el replay de Cf1–Cf20 pase.

## 4. Verificación de los Excel adjuntos

### Corrida ejecutada

```text
python scripts/audit_garrido_workbooks.py \
  --raw-workbooks Raw_data1+Re.xlsx Raw_data2+Re.xlsx \
  --rsult-workbook Rsult_1.xlsx
```

Después se ejecutó el harness de replay con las cuatro combinaciones de fuentes de demanda/riesgo y los dos modos de semillas. Los archivos de entrada fueron los adjuntos reales bajo `.hermes/desktop-attachments/`.

### Hechos verificados

El artefacto `outputs/audits/garrido_workbook_fidelity_attached_2026-08-06/audit_report.md` reporta:

- `47.546` filas de órdenes auditadas.
- `0` discrepancias al recomputar la fórmula de ReT.
- Gap máximo de fórmula `0.0`.
- `Raw_data1+Re.xlsx` es la fuente primaria para Cf1–Cf10.
- `Raw_data2+Re.xlsx` es la fuente primaria para Cf11–Cf20.
- `Rsult_1.xlsx` es un libro agregado/secundario.

La fórmula Excel, por tanto, está reproducida exactamente en el ledger raw. Esto valida la aritmética y la extracción, no la equivalencia física entre dos simuladores.

### Lane forense de replay

La mejor combinación observada fue:

```text
demand_source       = excel_order_tape
risk_attribution    = excel_risk_tape
risk_occurrence     = thesis_window
seed_stream         = split
```

El resultado `outputs/audits/garrido_replication_attached_forensic_2026-08-06/replication_audit.json` pasó los gates de extracción, horizonte operativo, órdenes operativas, ramas, familias y ReT.

Pero “pasó el gate” no significa igualdad perfecta de todos los escalares agregados:

- `n_configurations = 20`.
- `mean_abs_ret_gap = 0.00230584`.
- `max_abs_ret_gap = 0.01024516`, en Cf12.
- Las cantidades de órdenes, `Q`, `OPTj`, ramas y proporciones de casos sí coinciden en ese lane.

La afirmación defendible es **replay forense de las cintas y fórmula compatible**, no “el DES endógeno ya replica la tesis”.

### Lane endógeno

Se volvió a ejecutar el mismo harness con demanda calendárica y eventos de riesgo generados por el DES:

```text
demand_source       = thesis_calendar
risk_attribution    = des_events
risk_occurrence     = thesis_window
seed_stream         = split
```

Resultado real:

- `replication_status = failed_gate`.
- `mean_abs_ret_gap = 0.11940672`.
- `max_abs_ret_gap = 0.35587088`, en Cf16.
- `max_branch_share_gap_pct = 15.2927`.
- Las órdenes generadas no conservan el conteo de las cintas raw y no se puede comparar trayectoria-a-trayectoria con el Excel.

Esto no prueba por sí solo que el DES sea “incorrecto”: Simulink y Python no comparten PRNG ni necesariamente la misma implementación de eventos. Sí prueba que **no tenemos una validación endógena aprobada** contra los resultados de Garrido. El código debe tratar el lane forense y el lane endógeno como dos objetos distintos.

## 5. Tests de software ejecutados

En el worktree v2, con Python 3.11.15, NumPy 2.4.6, pandas 3.0.3, PyTorch 2.12.1+cu130 y las dependencias del entorno:

```text
pytest -q \
  tests/test_garrido_replication_harness.py \
  tests/test_garrido_excel_ret.py \
  tests/test_ret_excel_request_snapshot_contract.py \
  tests/test_expanded_contract_comparators_v2.py \
  tests/test_garrido_v0_recovery_gate_v2.py \
  tests/test_garrido_v0_recovery_surface.py \
  tests/test_ret_metric_invariants.py
```

Resultado: **46 passed, 3 skipped, 2 xfailed**.

Segundo bloque:

```text
pytest -q \
  tests/test_garrido_risk_headroom_sensitivity.py \
  tests/test_garrido_neural_headroom_gate.py \
  tests/test_learning_frontier_exists.py \
  tests/test_compare_garrido_dynamic_vs_static.py \
  tests/test_run_garrido2024_direct_rl.py \
  tests/test_garrido_wrap_scres_ai_contract.py \
  tests/test_thesis_faithful_lane.py
```

Resultado: **60 passed**.

Estos tests prueban contratos y funciones. No convierten por sí solos resultados históricos en evidencia confirmatoria.

## 6. Headroom thesis-native y sensibilidad de riesgos

### 6.1 Dominio de decisión original

El contrato v2 enumera completamente `6^3 = 216` posturas de buffers en Op3, Op5 y Op9, con niveles `[0,168,336,504,672,1344]` horas. El resultado `results/step3_pooled/result.json` usa 12 tapes por familia R1r/R2r y 216 posturas por tape.

Hechos verificados del resultado:

| Familia | Mejor postura estática | replay MPC vs mejor estática | Greedy PI best-found |
|---|---|---:|---:|
| R1r | `[0,0,336]` | `−0.00002138`, IC95 `[−0.00004636,+0.00000431]` | `+0.00002374`, LCB `+0.00000432` |
| R2r | `[0,672,168]` | `−0.00099085`, IC95 `[−0.00370060,+0.00068258]` | `+0.00100162`, LCB `+0.00008828` |

El greedy PI está etiquetado correctamente como `GREEDY_PI_BEST_FOUND_NOT_EXACT_CEILING`; no es una cota exacta ni un baseline admisible para declarar superioridad.

DDMRP proyectado perdió en R1r (`−0.00030292`, 0/12 tapes) y fue indistinguible de la estática en R2r. La versión sin proyección tiene más derechos que el resto y no es una comparación justa de superioridad.

### 6.2 Saturación física

`results/buffer_saturation_diagnostic/result.json` muestra que multiplicar por 10 el buffer de cualquiera de los tres nodos no cambia `ret_excel_full_ledger` en las dos familias. Reducir a cero sí cambia el resultado:

- R1r: Op9 `−0.00257251` frente a la referencia.
- R2r: Op9 `−0.05081715` frente a la referencia.
- Op3/Op5 sólo tienen efectos negativos al retirarlos en algunas familias.

**Interpretación:** el dominio thesis-native tiene una barrera inferior y una superficie saturada hacia arriba. Añadir más de la misma variable no crea por sí solo un problema de control no trivial.

### 6.3 Sensibilidad de riesgos

La corrida `results/garrido_risk_headroom_sensitivity_v1/` evaluó 18 candidatos de política, 45 perfiles de riesgo y 6 semillas: `4.860` evaluaciones. Incluyó frecuencias actuales, aumentadas, impactos one-at-a-time y multiplicadores de impacto. El auditor independiente reporta:

- `n_rows = 4860`.
- `recomputed_passing_door_count = 0`.
- `recomputed_status = DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER`.
- `status = PASS_GARRIDO_RISK_AUDIT` para la auditoría del artefacto.

La sensibilidad encontró configuraciones con mejor ReT en subperfiles, pero ninguna pasó simultáneamente los guardrails de servicio, recursos, órdenes perdidas y cola. No se debe seleccionar el máximo de ReT ignorando esos secundarios.

### 6.4 Superficie recovery del v.0

El gate v2 de shocks pasó como gate de liveness:

- seis riesgos muestran efecto incremental de servicio;
- seis contextos muestran cambio al variar la postura;
- la fracción observada de recuperación es `0.8857`.

Eso autorizó construir la superficie de 216 posturas, pero no autorizó entrenar.

La reauditoría independiente de `run_garrido_v0_surface_gates_v1.py` sobre las seis semillas de desarrollo produjo exactamente:

```text
claim_status = STOP_NO_RECOVERY_LEARNING_HEADROOM
G0 = pass
G1 = pass
G2 = fail: sólo R24 cumple; se requieren 4 de 6 contextos
G3 = fail: ganancia contextual vs postura común = 0 horas
```

En las seis folds leave-one-seed-out, la postura contextual seleccionada fue la misma postura común `[0,672,168]` para R11, R14, R21, R22, R23 y R24. R24 sí tuvo `Delta_CV_R2 = 0.2469` con LCB `0.1675`, pero eso no alcanzó el gate de cuatro contextos y no generó valor operativo de cambiar la postura por contexto.

**Conclusión:** la superficie recovery está viva, pero no tiene el headroom contextual exigido para justificar PPO, MLP o KAN. Entrenar ahora sería optimizar ruido o volver a elegir después la métrica/ventana.

## 7. Métrica: qué conservar y qué corregir

### Hechos verificados

- `ret_excel` reproduce el endpoint de los libros de Garrido y debe conservarse como métrica de fidelidad.
- El contrato v2 exige secundarios: `ret_excel_full_ledger`, `ret_thesis`, fill rate de flujo, órdenes perdidas, raciones entregadas, unresolved, strategic injected y terminal stock.
- Los resultados de la clase de política muestran un falsador real: un aumento de `worst_claimant_fill` puede comprarse abandonando órdenes. En `contention_policy_class`, el falsador de “no gain by abandonment” falla.
- El Cobb–Douglas del trabajo de factory resilience de Garrido no es automáticamente la métrica ReT de la tesis MFSC. Usarlo como objetivo principal sería una extensión nuestra, no una recuperación de la regla de la tesis.

### Propuesta

Usar dos endpoints con funciones separadas:

1. **Fidelidad:** `ret_excel`/`ret_excel_request_snapshot_v2`, siempre acompañado de los secundarios y del ledger completo.
2. **Decisión:** un escalar pre-registrado que penalice explícitamente pérdida de órdenes, unresolved, inequidad entre productos y recursos. La maximización de ReT desnuda no puede ser el único objetivo si la fórmula recompensa abandonar demanda.

No elegir entre ReT y Cobb–Douglas después de ver qué arquitectura gana. Si ambos se reportan, la selección primaria debe quedar congelada antes de abrir el bloque confirmatorio y debe reportar si las conclusiones cambian.

## 8. ¿Qué familia responde a Garrido 2024?

### Desarrollo ya observado

El resultado `results/garrido_normaliser_audit_v3/result.json` reporta, en replay de desarrollo sobre 12 semillas y con normalización honesta por prefijo:

- `neuron_memory − neuron_reset`: AUC de regret `+0.06070`, LCB95 `+0.04556`.
- `neuron_memory − OFAT`: `+0.04821`, LCB95 `+0.03325`.
- La memoria requiere menos corridas para estar dentro de 1% del mejor conocido que reset y OFAT.

La escalera `search_ladder_v5` compara 15 métodos y pasa sus falsadores de igualdad de presupuesto, no lectura de celdas no visitadas y retención de estado. `search_surrogates` muestra que una aproximación grande no es el ingrediente necesario: una neurona de cinco parámetros puede empatar aproximadores mayores en esta superficie.

El resultado `grid_transfer_confirmation_v2` reporta para UCB1 con transferencia `+0.03073` contra la réplica marginal, LCB `+0.01990`, y `+0.05744` contra arranque en frío, LCB `+0.04989`, con `n=60`. La custodia declara el inventario central incompleto, por lo que no se debe llamar a esto confirmación virgen absoluta.

### Arquitecturas neuronales

El bake-off `results/architecture_bakeoff/result.json` iguala aproximadamente 200.000 parámetros:

- KAN: 204.816 parámetros, media `97.0489`.
- MLP: 199.215 parámetros, media `97.5240`.
- DMLPA: 187.404 parámetros, media `97.6597`.
- KAN−MLP: `−0.4751`, IC95 `[−1.5484,+0.5982]`.
- KAN cuesta aproximadamente 4,1 veces más por decisión en ese host.

Es un resultado de desarrollo y el endpoint está en la escala del bake-off, no en la escala ReT del Excel. No demuestra ventaja de KAN. La evidencia favorable para KAN como surrogate de una superficie no autoriza afirmar ventaja como política de control.

### Respuesta provisional a Q1 y Q2

- **Q1, dentro del dominio observado:** el ingrediente que reproduce el aprendizaje es la retención de estado entre corridas/configuraciones; no una arquitectura neuronal específica.
- **Q2:** el mecanismo compatible con la Fig. 2 es un lazo externo `configuración → DES → métrica → estado actualizado → siguiente configuración`. La memoria puede implementarse con bandit, Bayesian optimization, surrogate neuronal o una política de simulación-optimización; la comparación debe ser funcional y de presupuesto, no una carrera de nombres de redes.

Estas respuestas son resultados de desarrollo, no universalidades fuera del dominio estudiado.

## 9. Contradicciones y artefactos que no deben entrar al paper como están

1. `garrido_v0_recovery_gate_v2` pasa el liveness gate, mientras `garrido_v0_surface_gates_v1` falla el headroom contextual. El primer resultado no autoriza el segundo.
2. `results/program_o/state_rich_comparator_fit_v1/result.json` contiene `full_des_h_pi_established=true` pero también `status=STOP_RESOURCE_OR_GUARDRAIL_CONFOUND`, `selected_config=null` y `h_obs_established=false`. El estado positivo debe tratarse como inconsistente/stale hasta reconciliarlo con su custody audit.
3. El artefacto `program_o/exact_transducer_validation_v1` dice que el transductor exacto fue validado, pero su `claim_boundary` mantiene `full_des_h_pi_established=false` y `learner_authorized=false`. Debe describirse como validación del transductor, no como resultado final del DES completo.
4. DDMRP proyectado y DDMRP sin proyectar no tienen los mismos derechos de acción. Una victoria del segundo no sería comparación justa; su derrota sí es un diagnóstico útil, no una prueba de superioridad de MPC.
5. Los resultados de la escalera y de memoria reutilizan deliberadamente bloques de semillas ya consumidos. Son desarrollo/replay, no confirmación prospectiva.
6. El entrenamiento `results/program_o/ret_only_learner_v1/vps_run` tiene diez modelos recurrentes, pero el evaluador declara `provisional_primary_pass=false`, `base_cells=false` y `terminal_verdict=PENDING_DIRECT_FULL_DES_REPLAY_AND_INTEGRITY_AUDIT`. Es un artefacto de entrenamiento pendiente, no evidencia publicable.
7. Las frases “RL beats Garrido”, “neural premium over MPC”, “KAN advantage” y “full thesis replication” no están licenciadas por los artefactos auditados.

## 10. Plan ejecutable que queda

### Prioridad A — cerrar la validación de base

1. Mantener el replay Excel como lane forense y publicar sus límites.
2. Reconciliar el lane endógeno con una comparación por distribuciones y eventos, no con igualdad de trayectoria, o congelar la afirmación como “reconstrucción no validada endógenamente”.
3. Completar, si se necesita una afirmación de tesis completa, Cf21–Cf90 con el mismo procedimiento y semillas documentadas. Los tres libros actuales no permiten esa afirmación.

### Prioridad B — responder Garrido sin forzar PPO

1. Congelar un benchmark de optimización externa: random, OFAT, UCB1, Bayesian optimization, lookahead limitado, surrogate MLP y surrogate KAN.
2. Igualar presupuesto de evaluaciones, usar cintas disjuntas y separar ajuste de evaluación.
3. Reportar dos estimandos: AUC de regret y porcentaje del techo admisible, porque ordenan distinto.
4. Contrastar memoria contra reset con un placebo de frecuencia y un placebo informado. El efecto de retención es más fiel a la pregunta de Garrido que “PPO supera a una política estática”.

### Prioridad C — sólo si se busca una afirmación de control RL

1. No entrenar sobre la superficie v.0 actual: G2/G3 fallaron.
2. Definir una extensión física defendible antes de abrir semillas: dos productos no fungibles, capacidad compartida finita, decisiones de asignación observables, conservación de masa y sin premiar abandono.
3. Construir el placebo fungible y el placebo no informado.
4. Exigir un `H_PI` seguro con LCB por encima del SESOI y guardrails de pérdida, servicio, equidad y recursos.
5. Sólo entonces comparar control constante, umbral, MPC/heurística, PPO+MLP recurrente y KAN. KAN será una comparación arquitectónica, no una novedad garantizada.

### Criterio de parada

Si la extensión corregida no supera los gates físicos, el resultado publicable no es “RL fracasó”, sino: **el valor de aprendizaje aparece en la optimización de configuraciones y no en una política intra-episodio bajo el DES thesis-native**. Eso responde a Garrido y evita fabricar headroom mediante una métrica conveniente.

## 11. Estado final de esta auditoría

- Inventario y ramas: **verificado**.
- Imágenes y figuras de Garrido 2024: **inspeccionadas por OCR; visión API bloqueada por modelo inexistente**.
- Fórmula Excel y extracción raw: **0 discrepancias, verificado**.
- Replay forense de Cf1–Cf20: **gates pasan, con límites documentados**.
- DES endógeno contra las cintas: **gate falla, no reclamar réplica exacta**.
- Headroom thesis-native ampliado: **no se encontró una política estructurada que convierta; no entrenar todavía**.
- Superficie v.0 recovery: **liveness pasa; headroom contextual falla**.
- Memoria entre corridas: **señal positiva de desarrollo/replay; no confirmatoria virgen**.
- KAN como política: **sin ventaja detectada en el bake-off; más lento**.
- KAN como surrogate: **prometedor, pero requiere CV agrupada y estabilidad antes de reclamar interpretabilidad**.
- Código modificado en esta auditoría: **no**.

**Decisión:** no escribir todavía el paper ni lanzar otra campaña neuronal. El siguiente experimento de alto valor es una extensión física pre-registrada con objetivo anti-abandono y un baseline externo de optimización de simulación; si no pasa el gate de headroom, se cierra honestamente la rama RL intra-episodio y se centra el manuscrito en aprendizaje/retención sobre configuraciones.

## 12. Revisión independiente posterior

Después de cerrar esta primera auditoría, tres revisiones independientes volvieron a inspeccionar los artefactos. Sus resultados no sustituyen la verificación local; los puntos siguientes que se incorporan aquí fueron además comprobados contra los archivos indicados.

### 12.1 La semántica de `sumBt` sigue abierta

Ejecuté de nuevo `scripts/audit_garrido_ledger_conventions.py` contra los libros adjuntos.
El resultado fue `HALTED_FALSIFIER_FAILED` porque la hipótesis de que `sumBt` está saturado en el tope 60 no pasa:

- 47.780 filas leídas por este auditor de ledger, incluyendo las filas de los libros completos.
- Sólo 2 filas discriminan entre las convenciones de simultaneidad.
- Ninguna convención reconstruye `sumBt` en más de 1,09% de las filas.
- `DPj = CTj` en 42.814/42.814 filas retrasadas.
- `RPj <= DPj` en 42.814/42.814 filas retrasadas.
- Hay 128 filas con autotomía.
- El máximo de `RPj` es 7.116,10 h y sólo 0,0047% de los valores queda dentro de 1% de ese máximo.

La lectura correcta no es «ya conocemos el significado de `sumBt`». La lectura es:
la convención de simultaneidad es irrelevante para casi todas las filas, la hipótesis de saturación en 60 falló, y el modelo de backorders del repositorio no explica la columna fuente. `sumBt` queda como semántica no resuelta y no debe usarse para afirmar fidelidad conductual.

### 12.2 Inversión entre bloques y endpoints

El artefacto correctivo `results/metric_audit/ret_metric_repair_confirmation_v1/result.json` contiene una señal positiva para R2r bajo el endpoint acotado:

- `ret_excel_clipped_0_1`: `+0,0124747`, IC95 `[+0,0091086,+0,0159091]`, 15/16 tapes.
- `ret_excel_full_ledger`: `−0,0044835`, IC95 `[−0,0066005,−0,0023880]`, 2/16 tapes.

Sobre el bloque posterior de Paso 3, calculé directamente desde los `rows.json` de R2r y contra el mismo incumbente congelado `[336,0,168]`:

- `ret_excel_clipped_0_1`: `−0,0121712`, 1/12 tapes positivos.
- `ret_excel_full_ledger`: `+0,0084152`, 12/12 tapes positivos.
- `flow_fill_rate`: `+0,0047571`.
- `delivered_rations`: `+42.784` unidades de media.

El análisis agrupado de Paso 3 usa además el mejor estático dentro de los mismos tapes, no siempre el incumbente congelado; contra ese estimando obtuvo `−0,0009908` en R2r para el ledger completo.

Esto no es una contradicción de código que deba ocultarse. Es una demostración de que el signo depende de **endpoint, incumbente y bloque de tapes**. La afirmación «MPC convierte» no puede formularse sin fijar los tres.

### 12.3 Guardrails incompletos en Paso 3

El preregistro exige `worst_product_fill` como guardrail bloqueante, pero `scripts/merge_step3_shards.py` persiste y aplica `flow_fill_rate`, que es agregado.
El propio `results/step3_pooled/result.json` registra la desviación.

Por ello, incluso el veredicto negativo `NO_STRUCTURED_CONTROLLER_CONVERTS` debe etiquetarse como desarrollo con guardrail más débil que el preregistrado.
No autoriza una conclusión fuerte sobre DDMRP ni sobre una ventaja neural.

### 12.4 Nulos de aprendizaje ya disponibles

La revisión independiente confirmó un nulo especialmente limpio:

- En C30, sobre 60 tapes no usados para calibración, el PPO canónico quedó por debajo del estático de contrato común:
  `PPO − static = −0,000018049`, IC95 `[−0,000028615,−0,000008087]`.
- Sólo 2/10 checkpoints y 2/60 tapes fueron positivos.
- La comparación RecurrentPPO anterior usó 3 semillas y 30k pasos frente a 60k del PPO canónico; no debe publicarse como comparación justa hasta igualar presupuesto.

Esto fortalece una posible contribución nula: el aprendizaje retenido entre configuraciones puede tener valor, mientras que una política PPO intra-episodio no muestra prima sobre un estático bajo contrato común.

### 12.5 Implicación operacional

La tercera re-confirmación virgen sugerida por la revisión independiente no se lanzó.
El registro central actual declara `NO_NEW_SEEDS_AUTHORIZED`, y abrir raíces nuevas sin actualizar el contrato sería una violación de custodia, no una mejora metodológica.
Antes de abrirlas hay que congelar endpoint, incumbente, guardrail `worst_product_fill`, SESOI y registro de semillas.
