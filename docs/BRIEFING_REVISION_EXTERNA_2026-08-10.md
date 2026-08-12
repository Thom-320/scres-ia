# Briefing para revisión externa — ¿cuál es el camino al mejor claim?

**Fecha:** 2026-08-10 · **Rama:** `codex/expanded-contract-comparators-v2` · **HEAD:** `21553715`
**Para:** un revisor externo con acceso de lectura al repositorio.
**Pregunta que se le hace:** dado todo lo medido, ¿cuál es la ruta con mayor probabilidad de
producir (a) una **prima neural** defendible y (b) evidencia de alguna **hipótesis de Garrido**?

---

## 0. El objetivo, que no cambia

Garrido, Pongutá & Adarme (ICCL 2024, LNCS 15168, pp. 80–94) preguntan dos cosas:

1. **¿Qué categoría de algoritmos de IA imita mejor el atributo de aprendizaje de la cadena (SCL)?**
2. **¿Cómo se integra esa familia en la estructura interna de un modelo DES para evaluar SCRES?**

Su Fig. 2 marca el hueco: los nodos ③ (recolección de datos) y ⑧ (V&V) son **los dos extremos de
un lazo abierto**. Un algoritmo de IA colocado entre ellos lo cierra. Lo llaman el **efecto
Alzheimer**: la red modelada no retiene lo aprendido de corridas anteriores. Su Fig. 5 es el puente
— una neurona cuyas dendritas son los cuatro *drivers* SCRES `d_i`, ponderados por `ρ`, con una
activación del tipo *«¿es la medida SCRES en la configuración x mayor que en la x−1?»*.

**Detalle que ha desviado el proyecto durante meses y que el revisor debe tener presente:** su
Fig. 3 sitúa el problema en **nivel 3, reconocimiento de patrón**, y su Fig. 5 compara
**configuraciones sucesivas**, no decisiones dentro de un episodio. Es aprendizaje en el **bucle
externo**. Nosotros llevamos ~15 programas probando RL como **controlador dentro del episodio**.

---

## 1. Lo que SÍ funcionó (todo con artefacto sellado)

### 1.1 Bucle externo — el efecto Alzheimer tiene precio medido

| artefacto | veredicto | número |
|---|---|---|
| `results/garrido_meta_learner/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | 7,24 corridas con memoria vs **13,54** reseteada; 12,42 el OFAT de la tesis |
| `results/manuscript/h2_learning_curve/result.json` | `H2_SUPPORTED_LEARNING_CURVE` | pendiente **+0,0422 [+0,0347, +0,0499]**, con nulo propio en −0,005 |
| `results/retention_simultaneous/result.json` | `RETENTION_SURVIVES_SIMULTANEOUS_INFERENCE_6_OF_6_ON_AUC_BUT_1_OF_6_ON_FINAL_SIMPLE_REGRET` | 6/6 en AUC, **1/6** en *simple regret* final |
| `results/search_ladder_v5/result.json` | `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH` | la neurona aguanta contra búsqueda con anticipación |

**Éste es el resultado más fuerte del repositorio y responde directamente al hueco declarado.**
Nótese la asimetría de `retention_simultaneous`: 6/6 en AUC pero 1/6 en el regret final. Es una
debilidad real que el revisor debe pesar.

### 1.2 Prima neural de PREDICCIÓN — confirmada en bloque virgen (2026-08-09)

`results/program_n/gate_b_confirmation_v3/result.json` → `SURFACE_PREMIUM_CAPTURED`, 7/7
falsadores, semillas 9600001–9600008 nunca antes abiertas. Re-adjudicado en
`results/program_n/gate_b_readjudication/result.json` contra el **mejor comparador no neuronal de
cada clase de información**, que es lo que exige el contrato marco:

```
mlp_tuned  vs linear_interactions   +0.1081 [+0.0601, +0.1561]   PASA
recurrent  vs linear_lagged         +0.1487 [+0.1069, +0.1905]   PASA
```

El segundo par es el más limpio del proyecto: red y comparador ven **exactamente** la misma
información (la resiliencia de la configuración x−1). Es la Fig. 5 de Garrido implementada como
**predictor**, batiendo a un modelo clásico con el mismo conjunto de información.

**Límites que el revisor debe imponer a este claim, ya medidos:**

* **La arquitectura NO replica.** KAN gana en dos corridas, MLP en una, ninguno en la sensibilidad.
  La afirmación defendible es de **familia**, no de arquitectura.
* **Es específico del endpoint Cobb-Douglas.** En la superficie legada `ret_excel`
  (`results/program_n/gate_b_sensitivity_ret_excel/result.json`) **ningún** brazo neuronal bate al
  mejor clásico: `kan − tree = −0,0029 [−0,0839, +0,0782]` (empate) y `linear_lagged` encabeza todo.
* Es **predicción**, no control. No dice que una política decida mejor.

### 1.3 Headroom real, y su mecanismo causal

`Program O`: contención sobre un recurso escaso **no fungible** da `H_PI = 0,1515` (LCB95 0,1156),
y el control decisivo —hacer el recurso **fungible**— da headroom **exactamente 0**. Es el único
mecanismo con headroom material medido en todo el proyecto, ~1.000× la escala `1e-4` del sobre
nativo de la tesis.

### 1.4 El único sitio donde un aprendiz batió a un planificador

`contention_v1` (banco sintético, `supply_chain/contention_bench_v1.py`): el aprendiz batió al
belief-MPC por **+0,0136 [LCB95 +0,0124]**. Diferencia estructural única: el régimen es
**semi-Markov con permanencia mínima**, así que un filtro bayesiano de primer orden está **mal
especificado por construcción**. Es sintético y no carga ninguna afirmación sobre la MFSC.

---

## 2. Lo que NO funcionó, con el número y la causa

### 2.1 Control: cerrado en todas partes

| lane | veredicto | por qué perdió |
|---|---|---|
| Track B / Puerta A2 (`results/program_n/gate_a2_track_b`) | `NO_QUALITY_PREMIUM_AGAINST_THE_WIDENED_CLASS` | una **realimentación lineal** (99,127) bate al MLP (98,567): −0,559 [−0,748, −0,386], 7/48 tapas |
| `program_v/prelearner_gate_v1` | privilegiado − Bayes = **+0,00076 [UCB95 +0,0023]** | no queda margen para una red |
| `headroom/g3_obs_conversion_v2` | `STRUCTURED_CONTROL_SUFFICES_G3_OBS` | idem |
| `headroom/g2_autotomy_threshold` | `THRESHOLD_RULE_SUFFICES` | idem |

**Diagnóstico central, y es la hipótesis que más quiero que el revisor ataque:** en todos los
entornos que construimos, el estado latente tiene 2 o 3 estados y su **modelo generativo es
conocido**. Ahí un filtro bayesiano escrito a mano es óptimo o casi, y una red sólo puede empatar.
El contraejemplo (§1.4) dice que la prima vive donde **la creencia exacta no es calculable en forma
cerrada**.

### 2.2 Puerta C (amortización): cerrada antes de entrenar

`results/program_n/gate_c0_expert_audit` → `NO_QUALIFYING_EXPERT`.
Un experto merece amortizarse sólo si es **caro** *y* **mejor**. Los dos candidatos del árbol
fallan mitades distintas:

* **`k3_strong_mpc`** es mejor (`ret_order +0,01242 [+0,00546, +0,01928]`) pero **no planifica**:
  instrumentado sobre 320 decisiones da **0 evaluaciones de candidato y 0 llamadas al simulador**.
  Es `paced_policy(α, β, γ)`, una regla en forma cerrada, **20× más barata** que la red que la
  imitaría. `Δ_amortización` es negativo por construcción.
* **`estar_direct_des_mpc`** sí planifica (192 llamadas al DES, 44.359× la latencia de la red) pero
  su calidad **nunca se había medido**. Se midió:
  `results/program_n/gate_c_prereq_mpc_quality` → `PLANNER_OBJECTIVE_IS_FLAT_NO_QUALITY_TO_MEASURE`.
  El objetivo vale **exactamente −3100,0 para las ocho acciones en las 24 tapas**. Sus 9.984
  evaluaciones empatan, comete el primer nivel del grid y aterriza en el peor resultado físico:
  **−50,46 [−52,62, −48,30]** pedidos contra la mejor constante, 0/24 tapas, y **pierde contra una
  secuencia aleatoria** por −15,92 [−18,50, −13,33].
  El ledger físico **sí** se mueve (`n_lost` 251 → 200,5 → 242,9, óptimo interior en 0,125): la
  acción hace algo y **la recompensa no lo ve**.

### 2.3 El sobre nativo de la tesis no tiene headroom, y hay tres razones medidas

1. La línea de ensamblaje corre a un **margen de capacidad del 2,6 %** (17.948 vs ~17.500
   raciones/semana a S=1): no hay decisión de asignación.
2. La rama ReT de mayor peso (**autotomía**, peso 1,0) es **estructuralmente inalcanzable**:
   `GARRIDO_FULFILLMENT_DELAY_HOURS = 54 > LT = 48`.
3. En `op12` el **placebo desinformado bate a la regla condicionada al estado**: el valor está en
   que el periodo varíe, no en qué lo hace variar.

Certificado formal de agotamiento: `results/paper2_search/paper2_exhaustion_certificate_2026-07-15.json` +
`docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md`, con `positive_instance_found: FALSE`.

### 2.4 La métrica primaria de la tesis está rota como objetivo de entrenamiento

`ret_excel` **premia el abandono**: el reparto que la maximiza entrega 50 % de fill y el que la
minimiza entrega 80 %. Bajo el régimen de riesgo se vuelve no monótona por censura dependiente de
la política (18,6 % vs 3,9 % de pedidos omitidos). Además es **dependiente de la cadencia de
`step()`**: trayectorias idénticas puntúan 37 % distinto. **Nunca se entrena sobre ella**; se usa
sólo como sensibilidad legada. El endpoint primario es un índice **Cobb-Douglas** portado de
Garrido 2024 (IJPR), con sus propios defectos de escala documentados.

### 2.5 Tres nombres que afirmaban más de lo que medían (todos caídos el mismo día)

* **`train_cell_mean_comparator` llamado «techo»** — lo superan brazos neuronales en las **cuatro**
  corridas de la Puerta B. La cifra «+0,0625 de margen disponible» está retirada
  (`docs/CORRECCION_TECHO_SUPERFICIE_CD_2026-08-09.md`).
* **`strong_mpc`** — no planifica (§2.2).
* **`H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE`** — certificaba elegibilidad de **coste**, medida
  con un *fixture* de cronometraje y acciones sintéticas (`100.0 + índice`); nadie había comprobado
  que el objetivo respondiera.

**Se le pide explícitamente al revisor que busque más casos de este patrón.**

---

## 3. Lo que podría funcionar — hipótesis abiertas, para que el revisor las ordene o las sustituya

### H-A · El entorno donde el aprendizaje sí pagaría (Fase 4 del plan)

Componer únicamente ingredientes que **cada programa ya demostró**:

| ingrediente | de dónde | evidencia |
|---|---|---|
| contención por recurso **no fungible** | Program O | `H_PI = 0,1515`, nulo fungible **exactamente 0** |
| régimen latente **semi-Markov** (permanencia mínima) | `contention_v1` | única celda donde un aprendiz batió al belief-MPC |
| acción **por nodo** | Track B | donde la red bate a la regla de umbral |
| resiliencia **vectorial** con restricción de cola | Program Q + Fig. 4(e) de Garrido | Q ganó al lazo abierto y **cayó por el guardrail de peor producto** |
| planificador **caro** online | E* | pero hay que **arreglarle el objetivo** (§2.2) |

La pieza nunca probada: Garrido define ReT como un **vector de cuatro componentes**, no un escalar.
Un objetivo vectorial con restricción de cola es precisamente donde una regla estructurada es
difícil de escribir a mano.

### H-B · Segunda superficie con potencia, para convertir la prima de predicción en claim de método

La sensibilidad `ret_excel` empató (`kan − tree = −0,0029`, IC cruzando cero). La pregunta abierta
es si eso es **ausencia de efecto** o **falta de potencia**. Un cálculo de potencia y una tercera
superficie decidirían si el claim es «una red bate al lineal en superficies de resiliencia» o sólo
«en Cobb-Douglas».

### H-C · Reparar el objetivo del E*

El ledger físico responde a la acción y la recompensa no. Un objetivo que **vea** `n_lost` haría del
E* el primer sustrato con planificador genuinamente caro **y** decisión real — reabriendo la
Puerta C y, potencialmente, el control.

### H-D · Fortalecer el bucle externo, que es donde ya ganamos

Es el hueco que Garrido declara y el resultado que ya tenemos. La debilidad conocida es
`retention_simultaneous`: 6/6 en AUC pero **1/6** en regret final. Cerrar esa asimetría podría valer
más que cualquier lane de control nueva.

---

## 4. Disciplina del repositorio, para que el revisor sepa qué puede y qué no puede pedir

* **Preregistro antes de correr**; los falsadores deben decir **por qué pueden fallar** y deben
  poder **pasar** (nada de tests de signo sobre cantidades que cruzan cero — nos costó dos bloques
  de semillas).
* **Placebo desinformado** en toda medición de headroom.
* **Semillas vírgenes y disjuntas** para cada confirmación; `research/seed_custody_registry.json`
  tiene `new_seed_opening: false` — abrir bloque exige excepción explícita del PI.
* **Nunca se edita un contrato congelado ni un artefacto fechado en sitio**; se corrige por
  enmienda y por sucesión (`research/supersession_registry.json`, 22 aristas, 0 problemas).
* **Un resultado retirado se conserva y se etiqueta; no se borra.**
* Se mide **a través del pipeline** (`arm_runner.py`), nunca con scripts ad hoc.
* Suite completa: `pytest tests/ -q` → **2350 passed, 2 skipped, 2 xfailed** (~15 min).

---

## 5. Dónde mirar primero

```
CLAUDE.md                                          objetivo y jerarquía de decisión
docs/PROMISING_LANES_REGISTRY.md                   registro vivo de lanes (nunca se pierde uno)
docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md   certificado de agotamiento del sobre nativo
docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md contrato marco de la prima neural
research/supersession_registry.json                qué artefacto superó a cuál y con qué regla
research/seed_custody_registry.json                custodia de semillas
results/program_n/                                 las tres puertas de 2026-08-09/10
supply_chain/falsifiers.py                         el instrumento de falsación compartido
supply_chain/contention_bench_v1.py                el banco con verdad conocida por construcción
```

Los `claim_status` de cada `result.json` son la fuente de verdad; los documentos narran, los
artefactos deciden.
