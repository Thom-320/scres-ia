# SCRES — fase B + respuesta experimental a Garrido 2024

**Fecha de corte de este documento:** `2026-08-07` (UTC)
**Worktree:** `/home/ubuntu/scres-ia-expanded-v2`
**Commit base usado para anclar los artefactos auditados:** `3d5082007276ca0a5d1304585b9f066d8ff6030f`
**Propósito:** dejar resultados, límites y decisiones de la fase experimental. **No es el paper.**

## Regla de lectura

- **Hecho verificado:** aparece en un artefacto local leído o en una ejecución identificada.
- **Inferencia:** lectura metodológica derivada de hechos verificados; no es un resultado causal nuevo.
- **Propuesta:** trabajo posterior; no es evidencia.
- **No reclamable:** el artefacto no permite esa afirmación, aunque contenga una cifra relacionada.

## Estado de promoción

| Lane | Resultado | Alcance permitido |
|---|---|---|
| Fidelidad de workbooks Garrido | 47.546 filas, 0 discrepancias de fórmula ReT | Compatibilidad aritmética/extracción; no equivalencia física del DES |
| Replay forense Garrido Cf1–Cf20 | gap medio ReT `0,00230584`, máximo `0,01024516` (Cf12) | Replay de cintas; no réplica endógena aprobada |
| DES endógeno contra Excel | `failed_gate`; gap medio `0,11940672`, máximo `0,35587088` (Cf16) | No afirmar réplica conductual |
| Optimización externa / transferencia | UCB1 con transferencia supera réplica marginal y arranque en frío en el bloque declarado | Respuesta a Q1/Q2 como desarrollo y bloque reservado; no RL intraepisodio |
| Vía B, `ret_excel_full_ledger` | `STOP_PRIMARY_FULL_LEDGER_HAS_NO_HEADROOM` en development y validation | Nulo primario publicable como nulidad del contrato full-ledger |
| Vía B, `service_safe` | `REPORT_SERVICE_SAFE_METRIC_EXPLORATORY` en ambos bloques | Reparación exploratoria; no promoción ni claim confirmatorio |
| Superficie v.0 de recovery | liveness pasa; G2/G3 de headroom contextual fallan | No autoriza afirmar adaptación contextual |
| Cobb–Douglas | prima puntual `+0,06248035`, IC95 `[-0,06063115,+0,18559186]` | No hay prima neural confirmada |

El manifiesto reproducible de esta síntesis está en:
`results/scres_phase_audit_2026-08-07/manifest.json`.

## A. Entrenamiento de Vía B bajo el contrato reparado

### Contrato y semántica

**Hecho verificado:** el contrato es `contracts/program_b_service_safe_learner_v1.json`.

- arquitectura: `PPO_MLP` y `RecurrentPPO_MLP`;
- semillas de learner: `8201, 8202, 8203`;
- `50.000` pasos por semilla;
- cintas de entrenamiento: bloque de desarrollo ya quemado `748100001–748350250`;
- cintas de evaluación: bloque de validación ya quemado `7400097–7400120`;
- dispositivo: CPU;
- no se autoriza entrenamiento confirmatorio fresco ni raíces DES nuevas;
- endpoint reparado: `clip(ration_ret_visible,0,1) * (1 - omitted_quantity / demanded_quantity)`;
- `ret_excel_full_ledger` se conserva como secundario y permanece idénticamente cero en los bloques full-ledger.

La reparación evita que una política reciba crédito por cantidades omitidas/abandonadas. El resultado `service_safe` no debe llamarse ReT de Garrido ni métrica thesis-native.

### Jobs lanzados

**Hecho verificado:** se relanzaron exactamente los cinco jobs faltantes:

```text
PPO_MLP          seed 8202
PPO_MLP          seed 8203
RecurrentPPO_MLP seed 8201
RecurrentPPO_MLP seed 8202
RecurrentPPO_MLP seed 8203
```

El PPO seed `8201` no se relanzó: la instrucción de ejecución prohibía duplicarlo y el proceso `72005` no apareció ya en la comprobación viva posterior. Se respetó la custodia y no se abrió un sexto job equivalente.

En el chequeo `2026-08-07T08:00:52Z`, los cinco procesos seguían vivos y sus hijos Python consumían aproximadamente `22,8–23,0%` de CPU cada uno, con ~`259 MB` RSS por proceso. Dos hijos ya habían leído ~`49,5 GB` cada uno y aún no había checkpoints `.zip`; el estado `D` y la espera `mem_cgroup_handle_over_high`/folio indican presión de memoria/lectura, no un cierre limpio. Esto es una observación operativa, no un resultado científico; no se mató ni reinició ningún job vivo. Por tanto, **aún no se reporta rendimiento de learner**. La evaluación obligatoria será `scripts/evaluate_program_b_service_safe_learner.py` sobre las 24 cintas de validation después de que existan todos los `.zip`.

### Headroom diagnóstico antes del learner

**Hecho verificado:** el oráculo seguro, que no es una política implementable ni un learner, produjo:

| Comparador | Desarrollo: media / LCB95 unilateral | Validación: media / LCB95 unilateral |
|---|---:|---:|
| Incumbente congelado | `0,234888 / 0,191023` | `0,318732 / 0,266086` |
| Incumbente in-sample | `0,233845 / 0,185776` | `0,274952 / 0,220231` |

Son diagnósticos del máximo seguro por tape, no evidencia de que PPO o RecurrentPPO lo alcancen. El criterio de lectura posterior es paired, por tape, con el incumbente congelado; el in-sample se informa separado para no contaminar la comparación ex ante.

**Decisión:** la campaña de desarrollo sí puede completar la evaluación de los learners bajo el contrato reparado. No se autoriza convertirla en confirmación, abrir semillas ni añadir KAN sin un nuevo contrato.

## B. Garrido 2024: reproducción y lazo externo

### Fidelidad de los números

**Hechos verificados:**

1. Los workbooks adjuntos contienen `47.546` filas auditadas y la fórmula Excel de ReT se recompone con `0` discrepancias y gap máximo `0,0`.
2. El replay forense conserva la cinta de órdenes y pasa sus gates de extracción, horizonte, órdenes y ramas, pero deja un gap ReT medio `0,00230584` y máximo `0,01024516`.
3. El lane endógeno, con demanda calendárica y eventos generados por el DES, falla: gap medio `0,11940672`, máximo `0,35587088` en Cf16 y diferencia máxima de participación de ramas `15,2927%`.
4. La semántica de `sumBt` continúa abierta: el falsador de saturación en 60 falló y el modelo de backorders del repositorio no explica la columna fuente.

**Límite:** los workbooks adjuntos cubren Cf1–Cf20. No se puede llamar a esto réplica completa de Cf1–Cf90.

### Q1 — qué ingrediente imita el aprendizaje

**Hechos verificados en los artefactos de búsqueda:**

- El ladder `search_ladder_v5` iguala presupuesto, impide leer celdas no visitadas y verifica que los brazos de memoria realmente retienen observaciones.
- La lectura exacta del ranking por AUC de regret normalizado (menor es mejor) es:
  `ucb1_transfer (0,045023)`, `neuron_memory (0,052033)`, `ofat_transfer (0,062743)`, `lookahead_kg_transfer (0,080182)`, …, `neuron_reset (0,112736)`.
- Los surrogates tienen AUC `0,047999` (KAN), `0,047450` (MLP) y `0,049011` (neurona de memoria), sin adjudicación confirmatoria.
- El bake-off de políticas a parámetros aproximadamente igualados reporta `KAN−MLP = -0,4751`, IC95 `[-1,5484,+0,5982]`; no demuestra una prima de KAN como política.

**Inferencia:** el resultado no sostiene “la red gana a todos los comparadores”. En el ladder corregido con comparadores clásicos que también conservan estado, `ucb1_transfer` tiene mejor AUC que `neuron_memory`. Lo que sobrevive de forma más limpia es la **retención entre configuraciones**, no una arquitectura neuronal particular.

El `claim_status` textual `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH` no debe leerse como victoria contra el ladder completo: el ranking y los contrastes almacenados son la evidencia primaria y colocan a UCB1 con transferencia por delante en AUC.

### Q2 — cómo se integra en el DES

**Hecho verificado:** `grid_transfer_v2` y `grid_transfer_confirmation_v2` implementan el lazo externo:

```text
configuración -> DES -> métrica -> estado retenido -> siguiente configuración
```

En el bloque reservado declarado (`n=60` réplicas):

- UCB1 con transferencia − réplica marginal: media `+0,03073311`, LCB95 `+0,01989687`;
- UCB1 con transferencia − arranque en frío: media `+0,05743819`, LCB95 `+0,04988858`;
- los falsadores de presupuesto, manifest de módulos y reproducción del subgrid pasan.

**Caveat de custodia:** el inventario central se declara incompleto. El artefacto demuestra que no hay colisión conocida y que el bloque es el declarado, pero no prueba virginidad absoluta. El alcance correcto es `CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK` tal como está registrado, con esta salvedad explícita.

**Respuesta experimental a Garrido:** la evidencia favorece leer la IA de las Figuras 2 y 5 como optimización de simulación sobre configuraciones. No obliga a reinterpretarla como control RL dentro de un episodio.

## C. Headroom y comparación justa de políticas

### Dominio thesis-native

**Hechos verificados:**

- La frontera de buffers enumera `6^3 = 216` posturas.
- En `step3_pooled`, `replay_mpc_v2` queda por debajo de la mejor estática: R1r `-0,00002138` (IC95 incluye cero) y R2r `-0,00099085` (IC95 incluye cero).
- El brazo greedy PI tiene mejoras pequeñas, pero está etiquetado como techo encontrado y no como competidor admisible.
- Multiplicar buffers por 10 no cambia `ret_excel_full_ledger`; retirar Op9 produce aproximadamente `-0,00257` en R1r y `-0,05082` en R2r.
- La auditoría de sensibilidad de riesgos tiene `4.860` filas y `0` puertas que pasan simultáneamente (`DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER`).

**Inferencia:** la superficie original tiene una barrera inferior y saturación superior. No es metodológicamente válido forzar una victoria de PPO, MPC o KAN cuando el endpoint principal no ofrece margen operativo defendible.

### Vía B full-ledger y reparación

**Hechos verificados:**

- Development y validation full-ledger: `STOP_PRIMARY_FULL_LEDGER_HAS_NO_HEADROOM`; `ret_full` primario es idénticamente cero.
- La salida original marcó que la partición fungible no era bit-idéntica por inventario separado, pero no creó headroom de `ret_full`.
- Tras la reparación, `service_safe` conserva el nulo fungible bit a bit y reporta valores exploratorios no nulos.

**Decisión:** el nulo full-ledger se conserva como resultado. La reparación se reporta como cambio de métrica auditable, no como “rescate” retroactivo del resultado original. El learner sólo puede evaluarse como exploratorio bajo `service_safe`, junto con órdenes perdidas, unresolved, fill por producto, stock terminal y carga real.

### Cobb–Douglas y resiliencia

**Hecho verificado:** en el experimento de Cobb–Douglas, la diferencia del oráculo de media por celda frente al baseline clásico con interacciones es `+0,06248035`, pero el IC95 a `t(4)` es `[-0,06063115,+0,18559186]`. La condición conjunta de SESOI e intervalo positivo no pasa.

**Conclusión:** el punto estimado no licencia “había una prima neural”. Las redes quedan por debajo del baseline clásico en ese artefacto. La etiqueta `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED` es la etiqueta del runner; la inferencia citable debe respetar el intervalo que cruza cero.

## D. v.0: H1–H4 con límites de estimando

| Hipótesis | Evidencia disponible | Veredicto de esta fase |
|---|---|---|
| H1 — learning effect | Transferencia de estado externa positiva para UCB1 en el bloque reservado; PPO intraepisodio C30 anterior fue negativo contra estático; B learner aún sin evaluación | **Apoyada sólo para el lazo externo/retención; no para superioridad RL intraepisodio** |
| H2 — adaptation | Recovery liveness pasa, pero la superficie v.0 tiene G2 fallido: sólo R24 supera el umbral; G3 falla con ganancia contextual de `0` horas y la misma postura común | **No apoyada como adaptación contextual** |
| H3 — volatility | La sensibilidad de riesgos cambia outcomes y tiene `4.860` filas, pero ninguna puerta segura bajo todos los guardrails | **Variación observada; no hay claim de política robusta** |
| H4 — path dependency | Los brazos de memoria cruzan contextos y superan a sus controles reset en algunas comparaciones; el efecto es de historia de búsqueda entre configuraciones | **Señal compatible con dependencia de trayectoria del buscador; no prueba todavía dependencia causal del DES endógeno** |

La separación evita confundir memoria del optimizador externo, memoria de la política y path-dependence física del DES.

## E. Decisiones y trabajo restante de esta fase

1. **Esperar y monitorizar** los cinco entrenamientos ya lanzados; no duplicar seed `8201` PPO ni abrir semillas nuevas.
2. **Evaluar** ambos directorios con `scripts/evaluate_program_b_service_safe_learner.py` sobre las mismas 24 cintas de validation.
3. **Adjudicar** por comparación pareada contra el incumbente congelado. Un learner no puede promoverse si falla una guardrail, si la ventaja sólo aparece en `ret_excel_clipped_0_1`, o si la LCB del endpoint `service_safe` no supera el SESOI contractual.
4. Si el learner queda negativo o nulo, **conservar el nulo y cerrar la afirmación de control intraepisodio**; no cambiar otra vez el endpoint para perseguir una señal.
5. Mantener el lane Garrido externo como companion experimental: OFAT, random, UCB1, Bayesian/lookahead y surrogates a presupuesto igualado, siempre con memoria y reset comparables.
6. No escribir el paper en esta fase. Los artefactos y esta nota son la trazabilidad previa.

## Artifacts

- Auditoría consolidada: `results/scres_phase_audit_2026-08-07/manifest.json`
- Runner de auditoría read-only: `scripts/audit_scres_phase_v1.py`
- Contrato B reparado: `contracts/program_b_service_safe_learner_v1.json`
- Gates B: `results/program_b_gate_v2/`
- Evaluador B: `scripts/evaluate_program_b_service_safe_learner.py` (incluye gate de guardrails por tape)
- Comparación final sin promoción: `scripts/merge_program_b_policy_comparison_v1.py`
- Observación operativa de runtime: `docs/PROGRAM_B_LEARNER_RUNTIME_NOTE_2026-08-07.md`
- Ladder clásico exploratorio: `scripts/evaluate_program_b_classical_baselines.py`, diseño en `docs/NOTE_PROGRAM_B_CLASSICAL_BASELINES_2026-08-07.md`
- Outer loop: `results/grid_transfer_v2/`, `results/grid_transfer_confirmation_v2/`
- Ladder y surrogates: `results/search_ladder_v5/`, `results/search_surrogates/`
- v.0 gates: `results/garrido_v0_recovery_gate_v2/`, `results/garrido_v0_surface_gates_v1/`
- Sensibilidad/headroom: `results/garrido_risk_headroom_sensitivity_v1/`, `results/step3_pooled/`, `results/headroom/cd_surface_prediction_premium/`
