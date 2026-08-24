# Revisión Codex: diagnóstico + literatura

**Fecha:** 2026-08-24  
**Alcance:** SCRES-IA / `scres-ia-expanded-v2`  
**Encargo respetado:** diagnóstico y literatura; no se ejecutaron experimentos ni entrenamientos.

## Trazabilidad y regla de lectura

Leí el brief, los 19 textos extraídos y los 10 informes de contexto, el manifiesto, el registro y la documentación/contratos/resultados relevantes del repositorio. El brief fija que Q está cerrado, que no se reabre con nuevas semillas, endpoint o margen, y que el Gate 0 es todavía una propuesta [`/home/ubuntu/scres-sources/BRIEF_REVISION_LITERATURA.md:20-44`](../BRIEF_REVISION_LITERATURA.md). El propio índice de acceso distingue 19 textos locales de 10 artículos `MANUAL`; no presento esos 10 como PDFs leídos íntegramente [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md:9-15`](../pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md).

Uso tres etiquetas:

- **HECHO:** consta en un resultado/contrato o en el texto citado.
- **INFERENCIA:** lectura causal razonable, pero no adjudicada por el contrato.
- **PROPUESTA:** diseño futuro; no cambia O, O-R ni Q.

El registro no equivale a 84 papers de frontera recientes: su propio conteo dice 63 filas del manifiesto, 25 filas de frontera, 84 DOI únicos y una entrada sin DOI [`/home/ubuntu/scres-sources/registry/BIBLIOGRAFIA_REGISTRO.json:1-9`](../registry/BIBLIOGRAFIA_REGISTRO.json); el manifiesto dice además 25 PDFs descargados y 38 entradas `MANUAL` [`/home/ubuntu/scres-sources/reports/MANIFIESTO_PDFS.md:10-15`](./MANIFIESTO_PDFS.md). En la cobertura de abajo, “hueco” significa que el subcampo no aparece como bloque explícito y verificable en ese registro, no que ningún concepto relacionado aparezca incidentalmente.

## 1. DIAGNÓSTICO

### Conclusión corta

No hay una contradicción entre “la adaptación funciona” y “no hay prima neural”. El proyecto ha probado dos preguntas distintas:

1. si una política con feedback supera a una frontera open-loop;
2. si una red añade valor sobre el mejor control estructurado bajo el mismo contrato.

La primera tiene evidencia positiva en Q; la segunda no. La documentación canónica dice explícitamente que Q mostró valor state-dependent frente a open-loop, equivalencia práctica con la mejor familia estructurada probada, pero no prima neural, seguridad por peor producto ni superioridad sobre un belief-MPC específico [`/home/ubuntu/scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md:73-78`](../../scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md). Por tanto, el diagnóstico correcto es **medición más física**, con una tercera causa operacional: todavía no se autoriza entrenar una nueva variante después de congelar Q.

### 1.1 Causas de MEDICIÓN

#### M1. El estimando de seguridad y el endpoint que usa una ruta de screening no son el mismo

**HECHO.** El programa define `worst_claimant_fill` como endpoint primario; el fill agregado y `ReT` full-ledger son guardarraíles o diagnósticos [`/home/ubuntu/scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md:109-124`](../../scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md). Sin embargo, el artefacto de Paso 3 registra `flow_fill_rate` como guardrail aplicado mientras `worst_product_fill` era el preregistrado; el propio JSON dice que un brazo que pase ahí **no** ha pasado el screen preregistrado [`/home/ubuntu/scres-ia-expanded-v2/results/step3_pooled/result.json:112-118`](../../scres-ia-expanded-v2/results/step3_pooled/result.json). La auditoría del repositorio llega a la misma conclusión: el veredicto negativo de esa ruta tiene un guardrail más débil y no autoriza una conclusión fuerte sobre una ventaja neural [`/home/ubuntu/scres-ia-expanded-v2/docs/SCRES_AUTONOMOUS_AUDIT_2026-08-07.md:374-380`](../../scres-ia-expanded-v2/docs/SCRES_AUTONOMOUS_AUDIT_2026-08-07.md).

**INFERENCIA.** Un agregado puede declarar éxito mientras abandona al producto débil. Eso no es una sutileza estadística: es cambiar el objeto de decisión. En Q, el informe de adjudicación observa precisamente que el learner compra fill agregado desbalanceando el producto débil y empeorando `max_backlog_age` y `service_loss_auc` frente al clásico [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md:9-20`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md). La falta de prima puede ser, en parte, que la política se está optimizando/evaluando sobre una métrica que no coincide con la guardrail de seguridad.

**Respaldo de literatura.** HPRS formula explícitamente una jerarquía `safety ⊃ target ⊃ comfort`: el target depende del safety y el comfort se ignora cuando entra en conflicto [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/a14-hprs2024-frontiers.txt`, abstract y §1]. La lección aplicable es que `worst_product_fill` no debe quedar como diagnóstico posterior de una recompensa plana.

#### M2. El comparador correcto deja un residuo mucho menor que el comparador que generó la impresión inicial

**HECHO.** En Q, `H_OL` es learner menos la mejor media de los 65.536 calendarios open-loop, mientras `Delta_N` es learner menos la mejor media de diez configuraciones clásicas; ambas familias se reseleccionan dentro de cada bootstrap [`/home/ubuntu/scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json:72-88`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json). El contrato pregunta justamente por superioridad sobre la frontera completa y por prima frente a la familia clásica, no por superar un baseline estático débil [`/home/ubuntu/scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json:12-19`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json).

**INFERENCIA.** La comparación contra open-loop identifica valor del feedback, pero no valor de la red. Al introducir un comparador state-rich y una familia clásica fuerte, el margen disponible se convierte en un residual. El resultado consolidado del repositorio es coherente con esa lectura: controles estructurados bastan en los screens ejecutados, el MLP de predicción de buffer queda por debajo del lineal, las redes de la superficie quedan por debajo del spline y `Delta_N` es negativo en las tres celdas [`/home/ubuntu/scres-ia-expanded-v2/docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md:11-20`](../../scres-ia-expanded-v2/docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md).

**Respaldo de literatura.** La revisión de R&S de Hong, Fan y Luo separa fixed-precision/PCS de fixed-budget y advierte que el objetivo de selección debe fijar explícitamente la familia de alternativas y la unidad de inferencia [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/a10-hong2021-fem-review-rs.txt`, §2-§3 y §6]. La medición original no demuestra que el learner sea inútil; demuestra que la afirmación “la red supera a un estático restringido” no es el estimando final.

#### M3. El Gate 0 propuesto tiene winner’s curse; no es evidencia ejecutada ni una explicación retroactiva de Q

**HECHO.** El Gate 0 escrito como `mean_t[max calendar] − max_c[mean ReT]` selecciona y evalúa sobre los mismos tapes; la auditoría confirma que no encontró una implementación ejecutada de esa fórmula y que el gate sigue sin adjudicar [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md:1-9`](../pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md). Si se usa el mismo tape para elegir el máximo y estimar su media, el primer término es `E_t[max_k X_tk]`, no la evaluación out-of-sample de un `k*` seleccionado en A y congelado en B [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md:11-21`](../pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md).

**INFERENCIA.** Sin split, el gate puede autorizar gasto porque exagera el headroom. Esto no invalida la adjudicación cerrada de Q; sí impediría que un futuro screen se presentara como evidencia física limpia. Hong et al. tratan el reuso de observaciones de búsqueda para evaluar el ganador como un problema abierto de garantías de selección [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/a10-hong2021-fem-review-rs.txt`, §7].

#### M4. La potencia y la inferencia son suficientes para cerrar el contrato, pero no para convertir cualquier señal en un claim general

**HECHO.** El artefacto de potencia de Q seleccionó `N=256` con potencia conjunta preabierta de `0.8755`, usando datos quemados; el bloque de confirmación figura como no abierto y requiere autorización independiente [`/home/ubuntu/scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json:37-70`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json). La regla de prima exige `LCB95(Delta_N) >= 0.01` en las tres celdas simultáneamente; si no, sólo permite equivalencia si el IC simultáneo queda dentro de `[-0.01,+0.01]` en todas [`/home/ubuntu/scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json:78-88`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json). El brief posterior declara Q cerrado y adjudicado; por eso el artefacto de potencia es evidencia del diseño preabierto, no una autorización para abrir datos nuevos [`/home/ubuntu/scres-sources/BRIEF_REVISION_LITERATURA.md:30-44`](../BRIEF_REVISION_LITERATURA.md).

**HECHO adicional.** En O, la validación correctiva confirmó ventaja de media del belief-MPC, pero falló la no-inferioridad conjunta de `CVaR10` en dos de las tres celdas [`/home/ubuntu/scres-ia-expanded-v2/docs/PROGRAM_O_TERMINAL_OUTCOME_CERTIFICATE_2026-07-15.md:6-31`](../../scres-ia-expanded-v2/docs/PROGRAM_O_TERMINAL_OUTCOME_CERTIFICATE_2026-07-15.md). Una adjudicación anterior fue retirada porque el comparador se seleccionó mirando las cintas de validación y porque el crítico simultáneo no estaba estandarizado [`/home/ubuntu/scres-ia-expanded-v2/docs/PROGRAM_O_FIXED_CLOCK_HOBS_VALIDATION_VERDICT_2026-07-15.md:3-30`](../../scres-ia-expanded-v2/docs/PROGRAM_O_FIXED_CLOCK_HOBS_VALIDATION_VERDICT_2026-07-15.md).

**INFERENCIA.** El diseño conservador evita falsos positivos, pero hace que “no se estableció prima” sea la conclusión correcta incluso cuando hay una señal de adaptación. No debe llamarse “falta de potencia” sin más: Q tiene un diseño de potencia; el problema es que una señal de media, una guardrail de cola y tres celdas simultáneas responden a claims distintos.

#### M5. La propia métrica de resiliencia tiene ramas ciegas y escalas difíciles de agregar

**HECHO.** La tesis define `ReT` como una función condicional a trozos (Eq. 5.5), con ramas de autotomía, recuperación, no recuperación y no disrupción [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/WRAP_Theses_Garrido_Rios_2017.txt`, §5.6.2-§5.6.3, Eq. 5.1-5.5]. La revisión del bundle verifica que la rama `Re(DP−RP)` tiene valor cero en todos los casos por el peso `Re^min`, y que `Re(RP)` es recíproca en el tiempo de recuperación [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/CLAUDE_COMMON_REVIEW_2026-08-24.md:46-64`](../pdfs_frontier/context_reports/CLAUDE_COMMON_REVIEW_2026-08-24.md).

**INFERENCIA.** Un control puede mejorar el agregado cambiando la composición de órdenes entre ramas, sin mejorar la vulnerabilidad profunda; y la media de un recíproco puede concentrar varianza en recuperaciones muy rápidas. Esto hace que `ReT` sea un endpoint de eficacia útil pero insuficiente como único endpoint de seguridad. El problema es de medición del objeto físico, no prueba de que la física carezca de headroom.

#### M6. La infraestructura y la gobernanza todavía no sostienen una nueva evidencia confirmatoria

**HECHO.** La suite explícita del repositorio tiene 2.260 tests pasados, 38 fallidos, 7 omitidos y 2 `xfail`; además se reporta un desacuerdo entre el anchor/hash CSSU actual y el golden [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md:19-25`](../pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md). La auditoría también deja `NO_NEW_SEEDS_AUTHORIZED` hasta congelar endpoint, incumbente, guardrail, SESOI y semillas [`/home/ubuntu/scres-ia-expanded-v2/docs/SCRES_AUTONOMOUS_AUDIT_2026-08-07.md:393-397`](../../scres-ia-expanded-v2/docs/SCRES_AUTONOMOUS_AUDIT_2026-08-07.md). Q mantiene el candidato histórico congelado, con reentrenamiento prohibido y sidecars no promocionables [`/home/ubuntu/scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json:21-35`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json).

**INFERENCIA.** Esta es una causa operacional de que todavía no exista una prima nueva: el fix-pack de observación, reward, `gamma`, horizonte, pasos y LSTM está propuesto, pero no se ha convertido en una campaña prospectiva autorizada. No es una excusa científica ni una licencia para reabrir Q.

### 1.2 Causas de FÍSICA / headroom real

#### F1. El headroom de feedback existe, pero el residual neural después del control estructurado es pequeño o nulo

**HECHO.** El informe de Q resume tres capas: el learner mejora mucho frente a open-loop; el control clásico con feedback también mejora la cola; frente al classical, el `worst_product_fill` del learner tiene LCB95 negativo en las tres celdas [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md:9-20`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md). El estado canónico del programa, de forma deliberadamente más prudente, sólo autoriza “mejor familia estructurada probada” y niega prima neural, seguridad por peor producto y superioridad sobre belief-MPC [`/home/ubuntu/scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md:73-78`](../../scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md).

**INFERENCIA.** La explicación física más parsimoniosa es que la información contingente que importa ya está capturada por una política estructurada, un DP/rollout o un belief-MPC. Una red puede aproximar esa superficie, pero no tiene un residuo material que comprar. Esto también encaja con R0: `STRUCTURED_CONTROL_SUFFICES_G3_OBS`, MLP peor que lineal, redes de superficie por debajo del spline y `Delta_N` negativo [`/home/ubuntu/scres-ia-expanded-v2/docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md:11-20`](../../scres-ia-expanded-v2/docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md).

#### F2. El contrato actual puede hacer que la secuencia sea fácil para el comparador y difícil para la red

**HECHO.** Q fija 21 campos observables no anticipativos, ocho decisiones semanales, cuatro acciones discretas, recompensa intermedia cero y recompensa terminal; la historia sólo es preprocesamiento arquitectónico [`/home/ubuntu/scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json:12-19`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json).

**INFERENCIA.** Si esos 21 campos ya contienen el belief suficiente para el control estructurado, la memoria recurrente no abre una dimensión física nueva. Si, por el contrario, hay aliasamiento temporal, el contrato exige probarlo antes: la literatura de POMDP recurrente de Ni et al. muestra que memoria, contexto y separación actor/crítico son mecanismos de representación, no garantía de prima [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/a6-ni2021-arxiv-recurrent-pomdp.txt`, abstract y §2-§4]. La documentación común también advierte que la literatura no identifica cuál de observación, reward, horizonte, presupuesto o comparador domina en SCRES sin un experimento nuevo [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md:27-31`](../pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md).

#### F3. El reward terminal y el shaping pueden estar suprimiendo crédito temporal, pero todavía no está demostrado que sean la causa dominante

**HECHO.** El fix-pack identifica como supresores candidatos la observación, reward terminal, `gamma`, pasos, tamaño LSTM, comparador y potencia [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md:22-34`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md). Forbes y Müller definen PBRS con `F=gamma Phi(s')−Phi(s)` y muestran que la condición de frontera en terminal/truncación, además de la escala del potencial, afecta la preservación del óptimo y la dinámica de aprendizaje [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/a2-forbes2024-arxiv-pbrs-intrinsic.txt`, abstract y §3-§5; `/home/ubuntu/scres-sources/pdfs_frontier/context_texts/a3-mueller2025-arxiv-pbrs-effectiveness.txt`, §2-§4].

**INFERENCIA.** Un horizonte largo con una sola señal terminal puede volver a la red un aproximador con poco gradiente útil, mientras que el MPC recibe directamente la estructura del problema. Es un mecanismo físico/algorítmico plausible, no un resultado ya adjudicado: no se puede atribuir la ausencia de prima a PBRS, `gamma` o LSTM sin la campaña nueva que el contrato actual prohíbe abrir.

#### F4. La política puede estar resolviendo un trade-off real media-cola que la recompensa no declara como lexicográfico

**HECHO.** El informe de Q separa eficacia y safety: el learner sube el fill agregado, pero la cola del producto débil y las métricas de backlog empeoran frente al clásico [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md:15-20`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md). HPRS ofrece un mecanismo explícito para no aceptar ese intercambio: el target sólo recibe valor condicionado al safety y el comfort se ignora cuando entra en conflicto [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/a14-hprs2024-frontiers.txt`, abstract, §1 y §3].

**INFERENCIA.** Si la física permite mejorar la media sacrificando el producto débil, una red optimizada con scalar reward no tiene por qué encontrar la solución que el revisor llamará “segura”. En ese caso no falta necesariamente capacidad neural: falta que el contrato de control represente el orden de preferencias del problema.

#### F5. El entorno y el índice pueden ofrecer menos diversidad causal de la que la arquitectura necesita

**HECHO.** Garrido et al. 2024 usan la demanda como fuente principal de incertidumbre en el modelo de resiliencia de fábrica y construyen el índice a partir de máximos muestrales y de la composición del conjunto de estrategias [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/garrido2024_factory_resilience.txt`, §3.1, §3.4-§5]; el texto SCRES+AI se presenta como exploratorio y no como prueba empírica formal [`/home/ubuntu/scres-sources/pdfs_frontier/context_texts/garrido2024_scres+AI.txt`, abstract y §3-§5].

**INFERENCIA.** Si las realizaciones de demanda, disrupción y composición no generan suficientes estados aliasados o mecanismos alternativos, una red sólo aprende una regla promedio. La frontera local no permite decidir si esto es “poca física” o “física no observada”; por eso el primer futuro screen debe medir headroom con un oráculo y un comparador limpio antes de entrenar.

### Diagnóstico integrado

La evidencia actual permite esta frase, y no una más fuerte: **hay valor de adaptación/feedback frente a open-loop, pero no se ha identificado un residuo neural material sobre la mejor familia estructurada, y parte de las rutas históricas de screening no midieron siempre el endpoint de seguridad preregistrado**. La documentación canónica prohíbe convertir ese resultado en “la prima es imposible” [`/home/ubuntu/scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md:173-183`](../../scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md).

## 2. COBERTURA: huecos frente a la frontera 2021–2026

### 2.0 Qué cubre y qué no cubre el registro

El manifiesto está organizado en seis bloques: control/creencias, DES/RL/SCM, resiliencia/disrupción, CVaR/distribucional, KAN y loop externo/simopt/bandits [`/home/ubuntu/scres-sources/reports/MANIFIESTO_PDFS.md:85-92`](./MANIFIESTO_PDFS.md). Hay buenos puntos de partida —Hong 2021 para R&S, Boute/Gijsbrechts y OWMR para inventario, CVaR/distributional RL y KAN—, pero faltan subcampos metodológicos que hoy son centrales para sostener un claim de política segura y generalizable.

Los identificadores de esta sección son DOI o arXiv verificables. Cuando una publicación de proceedings no tiene un DOI que deba afirmar con seguridad, doy título, autores/venue/año y `DOI_POR_VERIFICAR`, como exige el brief.

### H1. Safe RL y constrained RL

**Hueco.** El registro tiene riesgo/CVaR, pero no un bloque de safe policy optimization con restricciones explícitas, recuperación de violaciones y benchmark de seguridad. Eso importa porque el endpoint principal del proyecto es una guardrail de peor producto, no sólo una media.

- *Penalized Proximal Policy Optimization for Safe Reinforcement Learning* (2022), [arXiv:2205.11814](https://arxiv.org/abs/2205.11814).
- *Constrained Update Projection Approach to Safe Policy Optimization* (2022), [arXiv:2209.07089](https://arxiv.org/abs/2209.07089).
- Ji et al., *Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark*, NeurIPS Datasets and Benchmarks 2023, [arXiv:2310.12567](https://arxiv.org/abs/2310.12567), DOI [10.52202/075280-0831](https://doi.org/10.52202/075280-0831).

**Qué falta al pipeline:** una restricción de `worst_product_fill` o CVaR tratada como safety durante selección y evaluación, con presupuesto de violaciones y no-inferioridad, no como métrica secundaria después de maximizar `ReT`.

### H2. Offline RL

**Hueco.** No aparece una línea de aprendizaje offline desde logs/tapes, conservatismo fuera de soporte o comparación entre políticas aprendidas de datos históricos. Es relevante si no se puede abrir exploración o entrenamiento en el DES.

- Fujimoto y Gu, *A Minimalist Approach to Offline Reinforcement Learning*, NeurIPS 2021, [arXiv:2106.06860](https://arxiv.org/abs/2106.06860).
- Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling*, NeurIPS 2021, [arXiv:2106.01345](https://arxiv.org/abs/2106.01345).
- Kostrikov, Nair y Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022, [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).

**Qué falta al pipeline:** separar datos de búsqueda, entrenamiento y evaluación; declarar soporte/coverage de las acciones y medir si una política offline sólo copia la frontera histórica. Esto no autoriza entrenar Q; es una lane futura.

### H3. Sim-to-real, calibración de simuladores e incertidumbre de inputs

**Hueco.** El bundle discute DES y resiliencia, pero no hay una ruta moderna de calibración, domain randomization o transferencia que convierta un simulador sintético en evidencia operativa.

- Mehta et al., *A User’s Guide to Calibrating Robotic Simulators*, CoRL, PMLR 155 (2021), `DOI_POR_VERIFICAR`.
- Chen et al., *Understanding Domain Randomization for Sim-to-real Transfer* (2021), [arXiv:2110.03239](https://arxiv.org/abs/2110.03239).
- *Grounded Action Transformation for Sim-to-Real Reinforcement Learning*, *Machine Learning* 110 (2021), DOI [10.1007/s10994-021-05982-z](https://doi.org/10.1007/s10994-021-05982-z).

**Qué falta al pipeline:** calibrar distribuciones de demanda, retraso, impacto y recuperación contra observaciones externas; separar incertidumbre aleatoria de incertidumbre epistemológica; y evaluar la política en un conjunto de inputs plausible, no sólo en el DES nominal.

### H4. Evaluación off-policy (OPE)

**Hueco.** Hay evaluación por rollouts del simulador, pero no una capa explícita de OPE, cobertura de soporte, cross-fitting o estimadores doubly robust para comparar políticas sin abrir exploración.

- Uehara, Shi y Kallus, *A Review of Off-Policy Evaluation in Reinforcement Learning* (2022), [arXiv:2212.06355](https://arxiv.org/abs/2212.06355).
- Xu et al., *Doubly Robust Off-Policy Actor-Critic: Convergence and Optimality* (2021), [arXiv:2102.11866](https://arxiv.org/abs/2102.11866).
- Kallus y Uehara, *Efficiently Breaking the Curse of Horizon in Off-Policy Evaluation with Double Reinforcement Learning*, *Operations Research* (2022), DOI [10.1287/opre.2021.2249](https://doi.org/10.1287/opre.2021.2249).

**Qué falta al pipeline:** un estimador separado para calidad y safety, diagnóstico de overlap entre la política histórica y el challenger, e intervalos que incorporen incertidumbre del modelo de transición. OPE no sustituye la confirmación DES, pero evita llamar “evidencia” a una extrapolación fuera de soporte.

### H5. RL distribucionalmente robusto y robustez a shift

**Hueco.** El registro incluye CVaR/distributional RL, pero no la versión moderna de distribución robusta frente a un ambiguity set o a shift de soporte; CVaR nominal no es DRO.

- Zhou et al., *Finite-Sample Regret Bound for Distributionally Robust Offline Tabular Reinforcement Learning*, AISTATS 2021, PMLR 130, `DOI_POR_VERIFICAR`.
- Yu, Gehring, Schäfer y Anandkumar, *Robust Reinforcement Learning: A Constrained Game-theoretic Approach*, L4DC 2021, `DOI_POR_VERIFICAR`.
- Wang, Si, Blanchet y Zhou, *Sample Complexity of Variance-Reduced Distributionally Robust Q-Learning*, *Journal of Machine Learning Research* 25 (2024), `DOI_POR_VERIFICAR`.
- Lu et al., *Distributionally Robust Reinforcement Learning with Interactive Data Collection: Fundamental Hardness and Near-Optimal Algorithms* (2024), [arXiv:2404.03578](https://arxiv.org/abs/2404.03578).

**Qué falta al pipeline:** un ambiguity set de demanda/disrupción compartido por learner y classical, con la misma métrica de peor caso y una distinción clara entre robustez de cola y robustez de distribución.

### H6. Causal RL e intervención

**Hueco.** El registro trata covariables, control y resiliencia, pero no identifica qué intervención —asignación, producción, reparación o expedición— causa la mejora, ni usa DAGs para separar confusión de régimen.

- Gasse, Grasset, Gaudron y Oudeyer, *Causal Reinforcement Learning using Observational and Interventional Data* (2021), [arXiv:2106.14421](https://arxiv.org/abs/2106.14421), DOI [10.48550/arXiv.2106.14421](https://doi.org/10.48550/arXiv.2106.14421).
- Mutti et al., *Provably Efficient Causal Model-Based Reinforcement Learning for Systematic Generalization*, AAAI 2023, DOI [10.1609/aaai.v37i8.26109](https://doi.org/10.1609/aaai.v37i8.26109).
- *Causal Reinforcement Learning Based on Bayesian Networks Applied to Industrial Settings*, *Engineering Applications of Artificial Intelligence* 125 (2023) 106657, DOI [10.1016/j.engappai.2023.106657](https://doi.org/10.1016/j.engappai.2023.106657).
- Deng et al., *Causal Reinforcement Learning: A Survey* (2023), [arXiv:2307.01452](https://arxiv.org/abs/2307.01452).

**Qué falta al pipeline:** placebos y contrafactuales organizados por un DAG, intervenciones que rompan la asociación régimen-acción y un claim de mecanismo, no sólo de score.

### H7. Benchmarks modernos de inventario

**Hueco.** El registro sí tiene Beer Game, roadmap y OWMR; lo que falta es una batería reciente y reproducible para multi-echelon, perecederos, pricing, backlogging/lost sales y riesgo, con seeds, presupuestos y CIs comparables.

- Wang et al., *Solving a Joint Pricing and Inventory Control Problem for Perishables via Deep Reinforcement Learning*, *Complexity* (2021), DOI [10.1155/2021/6643131](https://doi.org/10.1155/2021/6643131).
- Wu et al., *Distributional Reinforcement Learning for Inventory Management in Multi-echelon Supply Chains*, *Digital Chemical Engineering* 6 (2023) 100073, DOI [10.1016/j.dche.2022.100073](https://doi.org/10.1016/j.dche.2022.100073).
- *Multi-echelon Inventory Optimization Using Deep Reinforcement Learning*, *Central European Journal of Operations Research* (2024), DOI [10.1007/s10100-023-00872-2](https://doi.org/10.1007/s10100-023-00872-2).

**Qué falta al pipeline:** un benchmark congelado con familias de heurísticas, MILP/DP cuando existan, no-inferioridad de servicio, métricas por producto, evaluación bajo demanda no estacionaria y presupuesto de cómputo igualado.

### H8. MARL con comunicación explícita

**Hueco.** El registro contiene CTDE/MARL y transformers, pero no comunicación aprendida, event-triggering, robustez adversarial de mensajes ni comparación de topologías de comunicación.

- Zhu, Dastani y Wang, *A Survey of Multi-Agent Deep Reinforcement Learning with Communication* (2022), [arXiv:2203.08975](https://arxiv.org/abs/2203.08975); versión de journal 2024, DOI [10.1007/s10458-023-09633-6](https://doi.org/10.1007/s10458-023-09633-6).
- Shibata, Jimbo y Matsubara, *Deep Reinforcement Learning of Event-triggered Communication and Control for Multi-agent Cooperative Transport* (2021), [arXiv:2103.15260](https://arxiv.org/abs/2103.15260).
- Yu et al., *Robust Communicative Multi-Agent Reinforcement Learning with Active Defense*, AAAI 2024, DOI [10.1609/aaai.v38i16.29708](https://doi.org/10.1609/aaai.v38i16.29708).
- Zhu et al., *HyperComm: Hypergraph-based Communication in Multi-Agent Reinforcement Learning*, *Neural Networks* 178 (2024) 106432, DOI [10.1016/j.neunet.2024.106432](https://doi.org/10.1016/j.neunet.2024.106432).

**Qué falta al pipeline:** definir quién observa qué, coste/latencia de cada mensaje, ablación no-communication y una prueba de que comunicar belief cambia la frontera, no sólo el score escalarizado.

### H9. Foundation models para OR y optimización híbrida LLM–solver

**Hueco.** KAN no es un foundation model para OR. El registro no cubre modelos grandes que generen formulaciones, llamen a solvers, aprendan heurísticas o traduzcan lenguaje/estado a decisiones verificables.

- Li et al., *Large Language Models for Supply Chain Optimization* (2023), [arXiv:2307.03875](https://arxiv.org/abs/2307.03875).
- AhmadiTeshnizi, Gao y Udell, *OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models* (2024), [arXiv:2402.10172](https://arxiv.org/abs/2402.10172).
- Huang et al., *ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling* (2024), [arXiv:2405.17743](https://arxiv.org/abs/2405.17743).

**Qué falta al pipeline:** un baseline solver/hybrid auditable, verificación de factibilidad, coste de llamadas y separación entre generación de una política y prueba de su desempeño en DES. No hay motivo para introducir foundation models antes de cerrar el endpoint y la frontera clásica.

### H10. R&S y simulation optimization moderna: contexto, covariables y presupuesto adaptativo

**Hueco.** El registro contiene la revisión de Hong 2021 y simopt/bandits, pero no cubre de forma suficiente la ola reciente de contextual R&S, covariates-to-decision, Gaussian-process CR&S y asignación adaptativa de simulaciones. Es el hueco más directamente conectado con el winner’s curse del Gate 0.

- Du, Gao y Chen, *A Contextual Ranking and Selection Method for Personalized Medicine*, *Manufacturing & Service Operations Management* 26 (2024) 167–181, DOI [10.1287/msom.2022.0232](https://doi.org/10.1287/msom.2022.0232).
- *Contextual Ranking and Selection with Gaussian Processes*, *ACM Transactions on Modeling and Computer Simulation* 34 (2024), DOI [10.1145/3633456](https://doi.org/10.1145/3633456), [arXiv:2201.07782](https://arxiv.org/abs/2201.07782).
- Keslin et al., *Ranking and Contextual Selection*, *Operations Research* 73 (2025) 2695–2707, DOI [10.1287/opre.2023.0378](https://doi.org/10.1287/opre.2023.0378).
- Li et al., *Efficient Simulation Budget Allocation for Contextual Ranking and Selection with Quadratic Models*, *European Journal of Operational Research* 328 (2026) 862–876, DOI [10.1016/j.ejor.2025.08.042](https://doi.org/10.1016/j.ejor.2025.08.042).

**Qué falta al pipeline:** contexto de tape como covariable, PCS/PGS o regret condicionado por contexto, selección en A y evaluación en B, y asignación de simulaciones a las celdas/alternativas difíciles en vez de gastar un N uniforme por todo.

## 3. QUÉ APLICAR: diez decisiones de diseño ordenadas por retorno esperado

El orden prioriza primero correcciones que pueden cambiar la conclusión sin entrenar y después mecanismos que abren un contrato nuevo. Ninguna decisión autoriza modificar los programas cerrados; la línea roja del bundle exige SHA, semillas/tapes vírgenes, estimando y gates congelados antes de tocar observación, reward, `gamma`, arquitectura, acción o comparador [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md:50-67`](../pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md).

| Orden | Decisión | Respaldo | Cambio concreto en el pipeline |
|---:|---|---|---|
| **1** | **Hacer que `worst_product_fill` sea endpoint bloqueante real y que la recompensa respete `safety → target → comfort`.** | HPRS, abstract/§1/§3, `a14-hprs2024-frontiers.txt`; Wu et al. 2023, DOI `10.1016/j.dche.2022.100073`. | Cambiar el esquema de resultados para persistir y aplicar siempre la métrica por producto; dejar agregado/`ReT` como diagnóstico. En una campaña nueva, comparar sin shaping, PBRS plano y potencial jerárquico. No promover una política por media si viola safety. |
| **2** | **Separar selección y evaluación de cualquier frontera adaptativa.** | Hong et al., `a10-hong2021-fem-review-rs.txt`, §7; auditoría de R&S [`CLAUDE_COMMON_REVIEW...:22-29`](../pdfs_frontier/context_reports/CLAUDE_COMMON_REVIEW_2026-08-24.md). | Implementar el Gate 0 corregido: seleccionar `k*` en tapes A, congelarlo y evaluar en B; reportar `G_PI_naive` sólo como diagnóstico, nunca como gate. Es un diseño futuro, no una re-adjudicación. |
| **3** | **Mantener el comparador fuerte, reseleccionarlo dentro de cada bootstrap y usar CRN por diferencias pareadas.** | Hong et al., §2-§3/§7; contrato Q [`program_q_frozen_policy_replication_v1.json:72-86`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json). | Congelar la familia admisible antes de abrir resultados; calcular learner−best-classical y learner−best-open-loop en la misma unidad tape/semilla; prohibir “best per tape”. Esto convierte una aparente derrota en un resultado de benchmark interpretable y evita crear un baseline débil. |
| **4** | **Diseñar la potencia con IZ/SESOI, PCS/PGS y asignación secuencial, no sólo con un N uniforme.** | Hong et al., §2/§5/§6; Fan et al., `a12-fan2025-jorsc-large-scale-so.txt`, sección de factor screening/CSB. | Fijar `delta=0.01`, error familiar y potencia conjunta antes de mirar datos; comprar replicaciones donde el intervalo no decide; usar CSB para separar obs/reward/horizonte/LSTM antes de un factorial confundido. |
| **5** | **Descomponer `ReT` por régimen y composición antes de usarlo como prueba de resiliencia.** | Garrido-Ríos, `WRAP_Theses_Garrido_Rios_2017.txt`, §5.6.2-§5.6.3, Eq. 5.1-5.5; revisión [`CLAUDE_COMMON_REVIEW...:33-64`](../pdfs_frontier/context_reports/CLAUDE_COMMON_REVIEW_2026-08-24.md). | Persistir rama por orden, efecto intra-régimen y efecto de composición; auditar la rama de no recuperación y tres agregadores de `Re(RP)`. No reemplazar retrospectivamente el endpoint Q; usarlo para definir un contrato nuevo no ciego a la cola. |
| **6** | **Tratar PBRS, truncación y terminal como parte del contrato, no como un ajuste cosmético.** | Forbes 2024, `a2-forbes2024-arxiv-pbrs-intrinsic.txt`, abstract/§3-§5; Müller 2025, `a3-mueller2025-arxiv-pbrs-effectiveness.txt`; HPRS, §3. | Separar `terminated` de `truncated`, fijar `Phi(terminal)=0`, congelar el potencial antes de entrenar y comprobar que el ranking de políticas congeladas no cambia con/sin shaping. Reportar `gamma` y número de pasos como factores, no esconderlos en una red. |
| **7** | **Medir un oráculo exacto de micro-MFSC antes de gastar en una red.** | Hong et al., §2/§6 sobre R&S; CONFIG real contenido en `a11-luo2024-scis-survey-mbrl.txt`, abstract/§1, que permite declarar infactibilidad en optimización black-box restringida. | Construir una instancia pequeña con valor óptimo por iteración, medir brecha del mejor control estructurado y tratar `Delta_N` como problema restringido. Si la región factible de safety no existe en el espacio declarado, cerrar la lane sin atribuirlo a mala sintonía. |
| **8** | **Calibrar inputs y convertir la variabilidad de demanda/disrupción en contexto observable o en uncertainty set.** | Mehta et al. 2021, PMLR 155; Chen et al. 2021, `arXiv:2110.03239`; Garrido 2024, `garrido2024_factory_resilience.txt`, §3.1/§3.4. | Ajustar distribuciones de llegada, duración, impacto y recovery con datos o escenarios externos; separar aleatoriedad del DES de incertidumbre de modelo; validar políticas en inputs hold-out y con domain randomization sólo si el conjunto es físicamente defendible. |
| **9** | **Usar memoria sólo después de demostrar aliasamiento y exigir presupuesto de aprendizaje igualado.** | Ni et al. 2021, `a6-ni2021-arxiv-recurrent-pomdp.txt`, abstract/§2-§4; Uehara et al. 2022, `arXiv:2212.06355`. | Crear pares de historias indistinguibles en el estado actual pero separables por historia; comparar feed-forward, recurrente y belief-MPC con iguales pasos, seeds, checkpoints y coste. Si no hay aliasamiento, no presentar LSTM como mecanismo causal. |
| **10** | **Preferir un residual estructurado/híbrido y factorizar la acción antes de escalar a MARL o foundation models.** | Boute 2021 y Gijsbrechts 2022, `a8-boute...txt`/`a7-gijsbrechts...txt`; Kaynov et al. 2023, `b12-ijpe2023-owmr-deep-rl.txt`; Zhu et al. 2024, DOI `10.1007/s10458-023-09633-6`. | Definir `policy = incumbent structured + bounded residual`, con máscara de factibilidad y coste de llamadas DES. Comparar contra el incumbent bajo el mismo presupuesto; sólo introducir comunicación/MARL si una ablación sin mensajes muestra un cuello de botella multiagente verificable. |

## 4. CLAIM LADDER

La escalera distingue lo que ya está respaldado, lo que puede resolver un único screen adicional y lo que requiere una campaña nueva. Los costes CPU de futuros peldaños son **estimaciones de planificación**, no resultados ejecutados; el bundle propone aproximadamente 6 CPU-h para el Gate 0, 42 CPU-h para el smoke B0/B1 y 80–150 CPU-h para tres brazos HPRS [`/home/ubuntu/scres-sources/pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md:36-40,57-62`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md); el uso real dependerá de la instrumentación.

| Peldaño | Claim defendible/alcanzable | Evidencia que lo sostiene | Qué falta | CPU adicional |
|---|---|---|---|---|
| **C0 — hoy** | **No se ha establecido una prima neural material `>=0.01` bajo el contrato Q.** Esto no dice que sea imposible. | Brief [`BRIEF_REVISION_LITERATURA.md:30-37`](../BRIEF_REVISION_LITERATURA.md); contrato Q y estado canónico [`program_q_frozen_policy_replication_v1.json:78-88`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json), [`PROGRAMA_PRIMA_NEURAL...:173-183`](../../scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md). | Nada para el claim negativo acotado; sí hace falta no sobreinterpretarlo como imposibilidad. | **0**. |
| **C1 — hoy** | **El feedback/state dependence tiene valor frente a la frontera open-loop en las celdas probadas.** | Q: `H_OL` y equivalencia pasan 3/3 según el brief; O-R reporta learner sobre 65.536 calendarios en calibración, pero no sobre el mejor classical [`REPOSITORY_SOURCE_OF_TRUTH.md:164-176`](../../scres-ia-expanded-v2/docs/REPOSITORY_SOURCE_OF_TRUTH.md). | Generalización a otras físicas, tapes y contratos; no convertir adaptación en prima neural. | **0**. |
| **C2 — hoy** | **La mejor política neural observada es prácticamente equivalente, no superior, a la mejor familia estructurada dentro del margen Q; y no es segura por peor producto frente al classical.** | Q exige simultáneamente `[-0.01,+0.01]` para equivalencia; la adjudicación/reportes dicen que la guardrail `worst_product_fill` falla frente al classical [`SINTESIS_FIXPACK_Y_PRIORIDAD.md:9-20`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md), [`program_q_frozen_policy_replication_v1.json:82-98`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json). | La evidencia de cola debe permanecer etiquetada como guardrail fallida, no como inexistencia de toda capacidad adaptativa. | **0**. |
| **C3 — un experimento barato, no ejecutado** | **Puede resolverse si queda headroom físico `>0.01` antes de entrenar:** Gate 0 con split A/B y comparador congelado. | Diseño corregido [`AUDITORIA_GATE0_SPLIT_TAPES.md:15-27`](../pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md); R&S contextual de Du/Keslin/Li. | Abrir un contrato nuevo y medir `G_PI_split`. Si no pasa, el claim más fuerte será “la lane no tiene headroom físico bajo este contrato”; si pasa, sólo autoriza estudiar aprendizaje. | **Bajo**, aproximadamente 6 CPU-h como estimación propuesta; no produce por sí solo una prima neural. |
| **C4 — una campaña nueva completa** | **Una política neural puede ser no-inferior en `worst_product_fill` y competitiva en `ReT` frente al mejor estructurado, bajo un contrato explícito de safety.** | Hay headroom de feedback y literatura de safe RL/HPRS que ofrece el mecanismo; no hay todavía evidencia de que la red lo capture. | Reward/endpoint jerárquico, split de selección, comparador nested, seeds/tapes vírgenes, presupuesto igualado y confirmación independiente. | **Alto**: el smoke propuesto ronda 42 CPU-h; la confirmación completa será mayor y no está presupuestada como número observado. |
| **C5 — ambicioso pero alcanzable con nueva evidencia** | **Prima neural `>=0.01` simultánea sobre la mejor familia classical y safety no-inferior en las tres celdas.** | Es la regla formal de Q [`program_q_frozen_policy_replication_v1.json:78-88`](../../scres-ia-expanded-v2/contracts/program_q_frozen_policy_replication_v1.json); no es un resultado actual. | Entrenamiento nuevo autorizado, control de winner’s curse, potencia conjunta y réplica confirmatoria. Un cambio de reward no puede venderse como réplica de Q. | **Muy alto**; varias campañas de entrenamiento/evaluación y al menos un bloque confirmatorio fresco. |
| **C6 — fuera del alcance actual** | **Prima generalizable, simulador validado o transferencia sim-to-real.** | No existe en el repositorio; el source of truth prohíbe presentar “validated digital twin”, universalidad o ventaja PPO general como claims [`REPOSITORY_SOURCE_OF_TRUTH.md:89-126`](../../scres-ia-expanded-v2/docs/REPOSITORY_SOURCE_OF_TRUTH.md). | Calibración externa, inputs hold-out, varios entornos, OPE/transfer, fallos de cola y evidencia operacional. | **Prohibitivo para la campaña actual**; no debe prometerse con un solo experimento. |

**Respuesta concreta a “un experimento adicional”.** Si “uno” significa un screen sin entrenamiento, el claim máximo honesto es C3: medir/descartar headroom físico sin winner’s curse. Si significa una campaña nueva que incluye entrenamiento, C4 es alcanzable como claim de competitividad/no-inferioridad; C5 exige además una confirmación independiente y no puede presentarse como simple parche de Q.

## 5. RIESGOS: objeciones previsibles de un revisor Q1 y blindaje

| Objeción del revisor | Por qué sería válida aquí | Blindaje exigible |
|---|---|---|
| **“El endpoint publicado no es el endpoint preregistrado.”** | El Paso 3 aplica `flow_fill_rate` donde el preregistro pedía `worst_product_fill` [`results/step3_pooled/result.json:112-118`](../../scres-ia-expanded-v2/results/step3_pooled/result.json). | Rehacer el esquema de salida antes de cualquier campaña; hacer que una violación por producto sea bloqueante; conservar agregado sólo como secundario. Etiquetar el resultado histórico como screen debilitado. |
| **“El learner gana porque el comparador está artificialmente restringido.”** | La propia historia canónica documenta que la ventaja de PPO no sobrevive al challenge same-contract y que la contribución es de benchmark/contrato, no de superioridad adaptativa [`REPOSITORY_SOURCE_OF_TRUTH.md:19-34`](../../scres-ia-expanded-v2/docs/REPOSITORY_SOURCE_OF_TRUTH.md). | Publicar la frontera completa, la familia clásica completa, presupuesto/latencia/observación y regla de reselección; separar `H_OL` de `Delta_N`. |
| **“Se seleccionó el ganador con los mismos datos usados para evaluarlo.”** | El Gate 0 propuesto tiene exactamente esa forma y todavía no se ha ejecutado [`AUDITORIA_GATE0_SPLIT_TAPES.md:5-13`](../pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md). | Split A/B o selección en desarrollo y evaluación virgen; declarar el split antes de abrir tapes; no usar `E[max]` como si fuera performance out-of-sample. |
| **“La prueba está subpotenciada para una afirmación conjunta de cola.”** | Q exige un margen de 0.01 en tres celdas y O falló CVaR10 simultánea en dos celdas aunque la media favorecía al MPC [`PROGRAM_O_TERMINAL_OUTCOME_CERTIFICATE_2026-07-15.md:17-31`](../../scres-ia-expanded-v2/docs/PROGRAM_O_TERMINAL_OUTCOME_CERTIFICATE_2026-07-15.md). | Fijar SESOI/IZ, potencia conjunta y unidad de resampling; reportar intervalos por celda y familia; no transformar una media positiva en safety pass. |
| **“El resultado depende de una inferencia defectuosa.”** | El veredicto O original fue retirado por selección del comparador en validación y crítico simultáneo no estandarizado [`PROGRAM_O_FIXED_CLOCK_HOBS_VALIDATION_VERDICT_2026-07-15.md:3-30`](../../scres-ia-expanded-v2/docs/PROGRAM_O_FIXED_CLOCK_HOBS_VALIDATION_VERDICT_2026-07-15.md). | Mantener la corrección como parte de la historia, usar sólo el certificado correctivo y registrar qué claims quedaron cerrados/falsados. No rescatar un resultado con una nueva fórmula post hoc. |
| **“El reward favorece una política que abandona un producto.”** | Q muestra mejora de fill agregado acompañada de deterioro del producto débil y backlog [`SINTESIS_FIXPACK_Y_PRIORIDAD.md:15-20`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md). | Safety jerárquico o CMDP, `worst_product_fill` bloqueante, métricas de backlog y costes de violación; comparar sin shaping y con PBRS terminalmente correcto. |
| **“La memoria o la arquitectura no se compararon con presupuesto justo.”** | La auditoría reporta una comparación RecurrentPPO de 30k pasos/3 seeds contra PPO de 60k pasos y advierte que no es justa [`SCRES_AUTONOMOUS_AUDIT_2026-08-07.md:384-391`](../../scres-ia-expanded-v2/docs/SCRES_AUTONOMOUS_AUDIT_2026-08-07.md). | Igualar seeds, pasos, checkpoints, hiperparámetros permitidos, coste de simulación y selección de checkpoint; tratar arquitectura como factor preregistrado, no como rescate. |
| **“El simulador es sintético y no está calibrado.”** | Garrido 2024 y los papers DT–MARL del bundle motivan mecanismos, pero no convierten por sí solos el DES en un digital twin validado; el texto SCRES+AI es exploratorio [`garrido2024_factory_resilience.txt`, §6.2; `garrido2024_scres+AI.txt`, abstract/§5; `b4-guzman2026-cie-circular.txt`, abstract y §5]. | Calibración de inputs, sensibilidad, conjunto de incertidumbre, escenarios hold-out y lenguaje “DES estilizado” hasta tener datos externos. No reclamar sim-to-real. |
| **“Hay demasiadas referencias no leídas o mal catalogadas.”** | El bundle avisa que 10 papers son `MANUAL`, que `a11-luo2024-scis-survey-mbrl.txt` contiene CONFIG y no una encuesta MBRL, y que varios papers de la lane topológica no tienen TXT local [`CONTEXTO_COMUN_HARNESSES_2026-08-24.md:9-15,42-48`](../pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md); [`CLAUDE_COMMON_REVIEW_2026-08-24.md:136-155`](../pdfs_frontier/context_reports/CLAUDE_COMMON_REVIEW_2026-08-24.md). | En la revisión final, distinguir “leído localmente”, “registro DOI” y “MANUAL”; corregir el catálogo; no atribuir a Kim/Fan/Akashi/Kotecha/Burtea/Mousa evidencia que no se leyó en el PDF. |
| **“La reproducibilidad de software no está cerrada.”** | La suite no está verde y existe un anchor/hash CSSU que no reproduce el golden [`CONTEXTO_COMUN_HARNESSES_2026-08-24.md:19-25`](../pdfs_frontier/context_reports/CONTEXTO_COMUN_HARNESSES_2026-08-24.md). | Publicar commit, contratos, hashes, seeds, manifest de tapes, runner y falsadores; detener claims científicos nuevos hasta resolver o aislar el artefacto. |
| **“Se está reabriendo el experimento hasta encontrar una arquitectura que gane.”** | Q/O están cerrados y la línea roja prohíbe re-adjudicar con otra semilla, métrica, física, comparador o margen [`BRIEF_REVISION_LITERATURA.md:30-52`](../BRIEF_REVISION_LITERATURA.md); la síntesis también fija el anti-p-hacking [`SINTESIS_FIXPACK_Y_PRIORIDAD.md:31-40`](../pdfs_frontier/context_reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md). | Tratar cada cambio como contrato nuevo; preregistrar antes de abrir datos; aceptar por escrito los falsadores y el resultado negativo; nunca llamar “réplica” a un cambio de reward/observación/arquitectura. |

### Implicación editorial

Para *Computers & Industrial Engineering*, la contribución defendible hoy es un trabajo de **contrato de benchmark, frontera de comparadores y límite negativo de la prima neural**, no un paper que afirme superioridad de PPO. Esa formulación coincide con la fuente canónica del repositorio y con la conclusión permitida: localizar cuándo el valor contingente es absorbido por una regla, un DP o un controlador estructurado, sin afirmar que la prima sea imposible [`/home/ubuntu/scres-ia-expanded-v2/docs/REPOSITORY_SOURCE_OF_TRUTH.md:19-34`](../../scres-ia-expanded-v2/docs/REPOSITORY_SOURCE_OF_TRUTH.md); [`/home/ubuntu/scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md:173-183`](../../scres-ia-expanded-v2/docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md).
