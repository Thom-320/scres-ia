# REVISIÓN CLAUDE — Estado del arte SCRES × Aprendizaje por Refuerzo y estrategia de publicación Q1

**Fecha:** 2026-08-24 · **Autor del informe:** Claude (Opus 5), rol: revisor de literatura + estratega editorial
**Encargo:** estado del arte + estrategia de publicación. **No se ejecutó ningún experimento.**
**Línea roja respetada:** nada de lo que sigue re-adjudica O, O-R ni Q.

---

## 0. Alcance de lectura y convenciones

### 0.1 Qué leí realmente

| Fuente | Estado |
|---|---|
| `pdfs_frontier/context_texts/*.txt` (19) | Leídos: los 4 nucleares (tesis 2017, Garrido 2024 ×2, Ding 2026) y los 15 de frontera. En los ficheros largos leí abstract/introducción + las secciones y ecuaciones que cito, no las 500 páginas de la tesis línea a línea. Cada cita indica archivo y, cuando aplica, línea o ecuación. |
| `pdfs_frontier/context_reports/*.md` (10) | Leídos íntegros. |
| `reports/MANIFIESTO_PDFS.md` | Leído (cabecera, resumen, tabla, instrucciones). |
| `registry/BIBLIOGRAFIA_REGISTRO.json` | Leído programáticamente: 84 DOIs únicos, 1 sin DOI, 63 filas manifiesto + 25 frontera. |
| `reports/REPORT_A_auditorias.md`, `SECOND_OPINION_CLAUDE.md`, `SECOND_OPINION_CODEX.md`, `CORRECCION_PROGRAM_Q_STATUS.md`, `SUITE_CERTIFICACION.md`, `VERIFICACION_EJECUTADA.md` | Leídos (secciones relevantes). Son la única fuente de los números del repositorio. |
| **PDFs binarios** | **NO abiertos.** Trabajé sobre los TXT extraídos. Donde el TXT pierde paginación, cito sección/ecuación, no página. |
| **Repositorio `scres-ia-expanded-v2`** | **NO leído.** Ningún número del repo fue recomputado por mí. |
| **Los 10 papers MANUAL** (Kim IISE, Fan JIPR, Liu POM, Kotecha, Mousa, Burtea & Tsay, Akashi, Ampratwum, Cheng EJOR, Zhou WSC) | **NO leídos.** Sólo tienen DOI verificado. No los uso como evidencia; sólo como posición en el mapa, y así lo marco. |

### 0.2 Etiquetas de estatus epistémico

- **[H-TXT]** — hecho verificable en un TXT del bundle, con archivo y sección.
- **[H-REP]** — hecho que sólo consta en un informe del repositorio (`reports/`, `context_reports/`) o en el brief. Es testimonio de terceros. **No lo recomputé.**
- **[H-API]** — hecho verificado por mí hoy contra Crossref (`api.crossref.org/works/<DOI>`) o arXiv (`export.arxiv.org/api/query`).
- **[INF]** — inferencia mía a partir de lo anterior.
- **[PROP]** — propuesta mía, sin respaldo empírico todavía.

---

## 1. MAPA DEL ESTADO DEL ARTE

La intersección SCRES × RL no es un campo: son **cinco literaturas que se citan poco entre sí**. Presentarlas como una sola es el error de encuadre más frecuente en los papers del bundle, y es la grieta por donde entra nuestra contribución.

### 1.1 Estrato A — La métrica de resiliencia (sin aprendizaje)

Es la capa donde vive Garrido y donde nace el proyecto.

- **[H-TXT]** La tesis (`WRAP_Theses_Garrido_Rios_2017.txt:3121-3134`, Eq 5.5) define `ReT` como una **función condicional a trozos**: cada orden *j* cae en exactamente una de cuatro ramas — autotomía `Re(AP_j)=Re^max·(AP_j/LT)` (Eq 5.1), recuperación `Re(RP_j)=Re·(1/RP_j)` (Eq 5.2), no-recuperación `Re(DP_j−RP_j)=Re^min·((DP_j−RP_j)/CT_j)` (Eq 5.3), y no-disrupción vía fill rate `1−(B_t+U_t)/D_t` (Eq 5.4). El marco se apoya en el *tail autotomy effect* (`:2701-2716`) y en tres clases de riesgo R1/R2/R3 (`:4081-4131`).
- **[H-TXT]** La propia tesis declara que la rama Eq 5.3 **vale cero en todos los casos** por el peso `Re^min` asignado (`:3043-3049`, literal: *"though the value of resilience is zero in all cases due to the weighting parameter assigned"*). Es decir: durante el período de máxima vulnerabilidad la métrica es **estructuralmente insensible al control**.
- **[H-TXT]** Garrido 2024 IJPR (`garrido2024_factory_resilience.txt:690-780`) propone un índice de resiliencia fabril `R` sigmoide sobre una Cobb-Douglas de cinco argumentos (Eq 5–6), con exponentes calibrados forzando cada argumento a 1/5 sobre los **máximos muestrales** de 10.000 corridas (ej. `ζ^max≈3.612 ⇒ a=0,024`), y un término de coste **normalizado por el coste medio del conjunto de las siete substrategias** (`:766-769`). El ranking Eq (9) (`:840-857`) es `R(S12) ≻ R(S11) ≻ R(S32) ≻ R(S22) ≻ R(S31) ≻ R(S13) ≻ R(S21)`, derivado de boxplots y regresión cuantílica τ=0,5, sin IC ni garantía tipo PCS.
- **[H-TXT]** Garrido 2024 ICCL (`garrido2024_scres+AI.txt`, §4.2 y Fig. 2/5) diagnostica el **"Alzheimer effect"**: en los DES de SCRES los nodos ➌ (data gathering) y ➑ (V&V) son los extremos de un lazo abierto; propone insertar una NN entre ellos y redefine SCRES como *capacidad adaptativa progresiva derivada del aprendizaje continuo*.

**Qué está resuelto en este estrato:** que la resiliencia se puede operacionalizar como un escalar derivado de salidas DES, con fases (absorber / recuperar / no-recuperar / normalidad). Consenso.

**Qué está abierto y en disputa:** **cuál** escalar. Y aquí hay un problema serio y transversal, no un detalle:

- **[INF]** Los cuatro escalares de resiliencia del bundle son **escalarizaciones no identificadas**: (i) `ReT` Eq 5.5 mezcla cuatro ramas de escalas y sentidos distintos, con pesos de mezcla que son función de la política; (ii) `R` Eq 6 de Garrido 2024 depende del **conjunto de comparadores** por el denominador de coste — añadir o quitar una substrategia cambia `R` de todas; (iii) `R̂` de Ding 2026 es un **producto de cinco índices topológicos dividido por la distancia media** (`1-s2.0-S0925527326000861-main.txt`, Eq 55) sin acotación ni análisis dimensional; (iv) el *resilience score* 0,892 de Kong 2026 (`b10-kong2026-eai-transformer.txt`, abstract) es un compuesto con pesos no declarados.
- **[INF]** Consecuencia: **un ranking de políticas sobre cualquiera de estos escalares no es un estimando bien definido** hasta que se fijen conjunto de comparadores, pesos y escala. Ninguno de los cuatro papers lo hace.

### 1.2 Estrato B — DRL para inventario: la pregunta ya está contestada

Este es el estrato con **consenso más fuerte y más ignorado** por la literatura de SCRES-MARL.

- **[H-TXT]** Gijsbrechts, Boute, Van Mieghem & Zhang 2022 MSOM (`a7-...txt`, abstract): A3C **"can match performance of state-of-the-art heuristics and other approximate dynamic programming methods"** en lost-sales, dual-sourcing y multi-echelon. El verbo es *match*, no *beat*. Y declaran que *"the initial tuning was computationally- and time-demanding"*.
- **[H-TXT]** Kaynov, van Knippenberg, Menkovski, van Breemen & van Jaarsveld 2024 IJPE (`b12-...txt`, abstract): DRL supera benchmarks en ~1–3 % (lost sales) y ~12–20 % (backorder parcial), pero **"For complete back-ordering, the algorithm cannot consistently outperform the benchmark."** Además: la *allocation rule* debe ser parte del entorno (racionamiento aleatorio), o el agente aprende a explotarla.
- **[H-TXT]** Boute, Gijsbrechts, van Jaarsveld & Vanvuchelen 2022 EJOR (`a8-...txt`, abstract e Introducción): roadmap de decisiones de diseño; la motivación explícita es que *"the abundance of choices ... combined with the intense computational effort to tune and evaluate each choice, may hamper their application in practice"*.

**Resuelto (consenso fuerte):**
1. DRL resuelve problemas de inventario intratables a nivel casi-óptimo. Sí.
2. DRL **empata** con heurísticas estructurales bien calibradas cuando éstas existen; gana sobre todo donde no hay heurística buena (demanda no estacionaria, multi-echelon profundo, backorder parcial). Sí.
3. La regla de asignación/racionamiento es un supresor oculto que puede fabricar o destruir la ventaja. Sí (Kaynov 2024, Boute 2022).

**Abierto:**
- **Qué es un comparador justo.** Gijsbrechts lo plantea; nadie en el estrato C lo implementa. Es el hueco de método más grande del campo.

### 1.3 Estrato C — MARL para reconfiguración topológica: el estrato caliente y el más débil metodológicamente

- **[H-TXT]** **Ding, Ming, Wang, Yan & Zhang 2026, IJPE 297:109995** (`1-s2.0-S0925527326000861-main.txt`, abstract): SCSoS bajo riesgo de disrupción, tres estrategias **filling / repairing / recruiting** sobre una SCDN, modelo POMDP, MAPPO, comparado con QMIX y MADDPG. **Es literalmente nuestra "lane topológica", ya publicada.**
- **[H-TXT]** Pero: Ding declara que *"determining the reconfiguration behavior of each element can be considered a complete episode, **with each episode consisting of a single-step interaction**"* (`:1247-1248`), y la Eq (54) suma las recompensas de los tres agentes *"during the action taken in that single step"*. **[INF]** Con episodio de un paso, el problema resuelto es un **bandit contextual multiagente**, no un problema de asignación de crédito temporal; la superioridad reportada de MAPPO sobre MADDPG/QMIX no es evidencia sobre el aspecto secuencial de la reconfiguración.
- **[H-TXT]** Los resultados comparativos de Ding §4.2.1 se reportan como números puntuales (≈50 vs 46; ≈160 vs 155) **sin IC y sin número de semillas declarado**.
- **[H-TXT]** Guzmán, Andrés & Torres-Polo 2026, CIE 218:112044 (`b4-...txt`, abstract): DT + MARL cooperativo de 5 agentes; y **declaran explícitamente que su contribución primaria es metodológica**: *"a controlled evaluation protocol with matched seeds, fixed horizons, and 95% confidence intervals is introduced to enable reproducible comparison across baselines, disruption scenarios, and sector archetypes"*, más un protocolo Value-of-Data. Reportan mejoras frente a un baseline **No-Op** y transferibilidad a cuatro arquetipos *"without retuning"*.
- **[H-TXT]** Kong 2026 EAI (`b10-...txt`, abstract): transformer + MARL, CTDE; reporta WMAPE 14,38 %, AUC de riesgo 0,941, 95,2 % de fulfillment, 7,1 días de recuperación, *resilience score* 0,892, y −19,3 % TTR frente a Transformer-MAPPO. **Sin IC, sin semillas declaradas, con endpoint escalarizado.**
- **[H-REP]** Kim 2023/2024 IISE (transshipments Dec-POMDP), Liu 2024 POM (MADRL multi-echelon a escala), Kotecha 2025 y Mousa 2024 (CompChemEng, GNN+MARL y análisis de fallos CTDE), Fan 2023 JIPR y Akashi 2023 CNSM (reparación de red con recursos escasos), Ampratwum 2024 COMPSAC: descritos en `REPORT_FRONTERA_2021-2026.md` §B. **No leídos por mí.** Los sitúo en el mapa; no los cito como evidencia.

**Consenso del estrato C:** MAPPO/CTDE es el caballo de batalla; la observación parcial es el formalismo aceptado; los baselines son heurísticas fijas, No-Op o algoritmos MARL rivales.

**Disputa real (y esto es lo importante):** **la calidad de la evidencia**. Guzmán 2026 declara que su aporte principal es el protocolo de evaluación — eso es, de facto, **una admisión pública en CIE de que el estándar de evaluación del subcampo es insuficiente**. Ding y Kong ilustran el problema. **[INF]** El subcampo está en la fase en que las ganancias reportadas no son separables del ruido de semilla, de la escalarización de la métrica y de la selección del comparador.

**Ausencia total, verificada en los tres papers que sí leí:** **[H-TXT]** ninguno de Ding 2026, Guzmán 2026 ni Kong 2026 compara contra un controlador con modelo exacto, ni reporta una cota superior (clarividente / información perfecta) del valor alcanzable, ni un endpoint de equidad por producto. Comparan contra No-Op, reglas fijas u otros algoritmos de aprendizaje.

### 1.4 Estrato D — La maquinaria de RL que el campo importa

- **PBRS.** **[H-TXT]** Okudo & Yamada 2021 IEEE Access (`a1-...txt`): shaping por subobjetivos, más fácil de especificar por humanos, supera baseline y subgoals aleatorios en tres dominios. **[H-TXT]** Müller & Kudenko 2025 (`a3-...txt`, abstract): la **efectividad** del PBRS depende de los Q-values iniciales y de la recompensa externa; derivan un **desplazamiento lineal constante `b`** del potencial que mejora la eficiencia muestral *sin* cambiar las preferencias codificadas ni tocar Q-init; y muestran que **escalar** el potencial es intrínsecamente limitado en MDPs terminales. **[H-TXT]** Forbes et al. 2024 (`a2-...txt`): extienden PBRS a potenciales de variables arbitrarias en entornos episódicos, con condición de frontera suficiente para preservar el conjunto de políticas óptimas. **[H-TXT]** HPRS / Berducci et al. 2025 Frontiers (`a14-...txt`): tarea como **conjunto parcialmente ordenado** safety ⊃ target ⊃ comfort, con el reward de target *función de* el de safety, y prueba de preservación de optimalidad; *"benefits from comfort requirements when aligned ... and ignores them when in conflict"*.
- **γ y horizonte.** **[H-TXT]** Wang & Jiang 2023/2025 (`a4-...txt`, abstract): MDPs *fast-slow*; congelar estados lentos por T pasos y resolver el nivel superior en escala temporal lenta con γ más favorable; análisis de regret; **"simply omitting slow states is often a poor heuristic"**; validado en control de inventario con costes fijos de pedido. **[H-TXT]** Sharma et al. 2021 Symmetry (`a5-...txt`, abstract): γ dependiente de la transición, convergencia probada **sólo para espacios finitos** con Q-learning/SARSA, y palanca explícita de aversión al riesgo (Cliff Walking).
- **POMDP recurrente.** **[H-TXT]** Ni, Eysenbach & Salakhutdinov 2021/2022 ICML (`a6-...txt`, abstract): con arquitectura e hiperparámetros cuidados, RL model-free recurrente iguala o supera a métodos especializados en 18/21 entornos; el mensaje es *"la implementación importa"*, no *"la memoria es gratis"*.

**Consenso:** PBRS preserva la política óptima bajo las condiciones de Ng; su **efectividad** es una pregunta separada y depende de Q-init (Müller). RL recurrente model-free es una línea base fuerte si se configura bien (Ni). γ cercano a 1 sobre horizontes largos es un problema real con soluciones publicadas (Wang & Jiang).

**Abierto:** **[INF]** nada de esto se ha trasplantado a SCRES con **verificación empírica de invarianza**. En particular, si el harness trunca episodios, `Φ(terminal) ≠ 0` y el shaping deja de ser invariante — un modo de fallo estructural, no de hiperparámetro.

### 1.5 Estrato E — La capa inferencial: existe, es madura, y SCRES-RL no la usa

- **[H-TXT]** Hong, Fan & Luo 2021 FEM (`a10-...txt`, abstract): taxonomía **fixed-precision (hypothesis-testing) vs fixed-budget (dynamic-programming)**; el mismo texto (`:2498-2510`) define los tres objetivos posibles con covariables: `PCS(x)` condicional, `E[PCS(X)]` y `min_x PCS(x)`; y (`:2713`) recoge que **reusar las observaciones de la fase de búsqueda rompe las garantías del procedimiento de selección** (Eckman & Henderson).
- **[H-TXT]** Fan, Hong, Jiang & Luo 2025 JORSC (`a12-...txt`, abstract y `:631-641`): revisión de SO a gran escala; *variable/factor screening* como fase preliminar que *"identify the effective variables ... and statistically eliminate ineffective ones"*, con CSB controlando **error tipo I y potencia**.
- **[H-TXT]** El fichero `a11-luo2024-scis-survey-mbrl.txt` **no es** una encuesta de MBRL: es **Xu, Jiang, Svetozarevic & Jones, "Constrained Efficient Global Optimization of Expensive Black-box Functions" (CONFIG, arXiv 2211.00162v4)**, cuyo abstract declara que el método *"naturally provides a scheme to declare infeasibility when the original black-box optimization problem is infeasible"*. El defecto de catalogación sigue vivo en el bundle y ya estaba señalado en `CLAUDE_COMMON_REVIEW_2026-08-24.md §11.1`. **Cualquier afirmación del programa apoyada en "a11 = survey MBRL" carece de respaldo en este bundle.**

**Consenso de la comunidad de simulación:** una comparación entre alternativas caras exige zona de indiferencia declarada, garantía PCS/PGS, y *clean-up* tras la selección.

**[INF] El hecho estructural del campo:** de los papers de los estratos A y C que sí pude leer (tesis 2017, Garrido 2024 ×2, Ding 2026, Guzmán 2026, Kong 2026), **cero** usan procedimientos de ranking & selection con garantía. Guzmán 2026 es el más cercano (semillas emparejadas + IC95) y aun así **[INF]** IC marginales por brazo no son lo mismo que un IC sobre la diferencia pareada, y "transferible a 4 arquetipos sin retuning" es un claim de **superficie de decisión** que exigiría `min_x PCS(x)`.

### 1.6 Resumen del mapa

| Pregunta | Estado | Evidencia |
|---|---|---|
| ¿DRL resuelve inventario intratable? | **Resuelta: sí** | Gijsbrechts 2022 MSOM; Kaynov 2024 IJPE |
| ¿DRL supera heurísticas estructurales fuertes? | **Resuelta: no de forma fiable; empata** | Gijsbrechts 2022 (*match*); Kaynov 2024 (backorder completo: no supera) |
| ¿PBRS preserva el óptimo? | **Resuelta: sí bajo condiciones (Φ terminal 0)** | Müller 2025; Forbes 2024; HPRS 2025 |
| ¿Es MARL/CTDE viable para reconfiguración de red? | **Resuelta operativamente; disputada en calidad de evidencia** | Ding 2026 (sin IC, episodio de 1 paso); Guzmán 2026 (protocolo declarado); Kong 2026 (score escalar sin IC) |
| ¿Cuál es el techo de valor alcanzable? | **ABIERTA — nadie la formula** | ninguno de los 19 TXT reporta cota clarividente |
| ¿Qué comparador es justo? | **ABIERTA — planteada, no resuelta** | Gijsbrechts 2022 §comparador; nadie del estrato C la aborda |
| ¿Cómo medir resiliencia sin escalarización no identificada? | **ABIERTA y empeorando** | Eq 5.5 tesis; Eq 6 Garrido 2024; Eq 55 Ding 2026; score 0,892 Kong 2026 |
| ¿Seguridad de cola / equidad entre productos de la política aprendida? | **ABIERTA — casi virgen** | ningún endpoint por-producto en el bundle |
| ¿Validez estadística de "ganamos un X %"? | **ABIERTA — la literatura de R&S dice que la práctica actual es insuficiente** | Hong 2021 §7; Fan 2025 §screening |

---

## 2. DÓNDE ESTÁ NUESTRO HUECO

Voy a ser duro, como se pidió. Primero mato lo que no es nuestro.

### 2.1 Novedad APARENTE — cosas que creemos nuestras y ya están publicadas

| Supuesta contribución | Ya publicado en | Veredicto |
|---|---|---|
| **"MARL para reconfiguración de SC con filling/repairing/recruiting bajo POMDP"** | **[H-TXT]** Ding, Ming, Wang, Yan & Zhang 2026, *IJPE* 297:109995, DOI `10.1016/j.ijpe.2026.109995` — mismas tres estrategias, mismo formalismo POMDP, MAPPO, misma familia de baselines | **No es novedad.** Es la lane topológica, publicada antes que nosotros, en el journal que es nuestra segunda opción. Si la ejecutamos, somos segundos. |
| **"Protocolo reproducible con semillas emparejadas e IC95 para DT+MARL"** | **[H-TXT]** Guzmán, Andrés & Torres-Polo 2026, *CIE* 218:112044, DOI `10.1016/j.cie.2026.112044` — lo declaran como su **contribución primaria** | **No es novedad**, salvo por lo que ellos **no** hacen (ver §2.2). Es peligroso: es nuestro journal objetivo y ya tiene un paper cuyo aporte es "el protocolo". |
| **"PBRS para densificar una recompensa terminal esparsa en inventario"** | **[H-API]** De Moor, Gijsbrechts & Boute 2022, *EJOR* 301(2):535-545, DOI `10.1016/j.ejor.2021.10.045`, *"Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management"* | **No es novedad.** El ítem (b) del fix-pack está publicado, en EJOR, por el mismo grupo que escribió nuestro roadmap de referencia. |
| **"Sustituir MLP/LSTM por Transformer o GNN"** | **[H-TXT]** Kong 2026 (transformer-MARL); **[H-REP]** Kotecha 2025, Ampratwum 2024, Fan 2023 (GNN) | **No es novedad.** Es ingeniería de arquitectura sobre un patrón ya establecido. |
| **"El learner supera a la planificación open-loop"** | **[H-API]** Kegenbekov & Jackson 2021, *Algorithms* 14(8):240, DOI `10.3390/a14080240`; y es la premisa entera del estrato B | **Dirección esperada.** Nuestra versión es más fuerte (mejor de 65.536 calendarios enumerados), pero el *signo* no sorprende a nadie. |
| **"Cerrar el loop Alzheimer insertando una NN entre ➌ y ➑"** | **[H-TXT]** Es la propuesta explícita de Garrido, Pongutá & Adarme 2024 (ICCL LNCS 15168) — el paper-fuente del proyecto | **Ejecutar la propuesta de otro no es novedad**, salvo que el resultado sea sorprendente. Y nuestro resultado **sí lo es**, pero no en la dirección que el paper-fuente esperaba (ver §3). |
| **"KAN para SCRES"** | Sugerido por Garrido 2024; **[H-REP]** nuestro bake-off da KAN−MLP = −0,475 con IC [−1,548, +0,598] que **cruza 0** (`REPORT_A_auditorias.md:179`) | **No tenemos nada.** No mencionarlo como contribución. |
| **"Prima neural en SCRES"** | — | **[H-REP]** `neural_premium = false` en 3/3 celdas; Δ_N puntual −0,0015 a −0,0027 (`CORRECCION_PROGRAM_Q_STATUS.md`; `REPORT_A:149`). **No existe. No lo afirmemos jamás.** |

### 2.2 Novedad REAL — lo que hacemos y nadie más hace

Cuatro cosas. Ninguna es un algoritmo. Todas son de **medición**, y por eso son defendibles.

**(A) Comparador = máximo sobre una frontera de acciones enumerada exhaustivamente, reseleccionado dentro de cada resample bootstrap.**

- **[H-REP]** `H_OL = learner − max(65.536 calendarios)` con `reselect_open_loop_65536_inside_every_resample: true` (`REPORT_A_auditorias.md:148,287`). El espacio `Discrete(4)^8 = 65.536` se enumera completo, por tape.
- **[H-REP]** El brief declara además que el Gate O-0 de Program O está ejecutado y pasa con `static_reselected_over_65536_in_every_resample = true` y **20.447.232 evaluaciones de calendario**. **[INF]** 20.447.232 = 65.536 × 312, consistente con 312 unidades de evaluación (tape × celda); esa descomposición es aritmética mía, no un dato del repo.
- **[H-TXT]** Los papers del estrato C comparan contra 1–4 alternativas fijas (No-Op, reglas, QMIX/MADDPG). **Ninguno enumera la frontera.**
- **[INF]** Por qué importa: reseleccionar el máximo **dentro** del resample convierte el comparador en un rival adaptativo y **elimina el cherry-pick del comparador**, que es exactamente el sesgo que Eckman & Henderson describen (`a10-...txt:2713`) y que infla las mejoras reportadas en el estrato C. Es un dispositivo anti-winner's-curse aplicado al *baseline*, no al método propio. No lo he visto en ningún paper del bundle.

**(B) Techo de información perfecta medido ANTES de entrenar, con placebo causal que devuelve exactamente cero.**

- **[H-REP]** `H_PI` full-DES *safe* = **0,15151**, LCB95 simultánea **0,11562**, con **fungible-null exactamente 0** como control causal, 27 placebos de información PASS (LCB mínima 0,00716), igualdad de recursos verificada, 1.451 replays físicos con 0 fallos (`REPORT_A_auditorias.md:139-140`).
- **[H-REP]** El mismo instrumento aplicado a otra lane da **cero puertas**: 45 perfiles × 18 posturas × 6 semillas = **4.860 evaluaciones**, `H_profile_raw` máximo 0,00024 — **41,5× por debajo del SESOI 0,01** (`REPORT_A:97,159-161`). Y en Program L, `H_PI ≤ 0,005` en full-DES frente a 0,15 estilizado — **colapso 30×** (`REPORT_A:75`).
- **[INF]** El placebo fungible es la pieza que nadie tiene: demuestra que el headroom medido es **atribuible al mecanismo** (recurso compartido no fungible entre dos productos, Op5–Op7) y no a holgura de la métrica. Un headroom sin placebo es indistinguible de un artefacto de medición.
- **[INF]** Ningún paper del bundle mide un techo. Todos asumen que hay valor que capturar. Nosotros lo **enumeramos** y sabemos cuándo vale 0,15 y cuándo 0,0002.

**(C) Equivalencia preregistrada con zona de indiferencia declarada, en lugar de una prueba de superioridad.**

- **[H-REP]** `Delta_N` con IC95 simultáneos `[−0,00627, +0,00310]`, `[−0,00552, +0,00408]`, `[−0,00268, +0,00186]` — los tres **contenidos** en la banda ±0,01 preregistrada; N=256, potencia conjunta 0,8755, max-t studentizado (`SECOND_OPINION_CODEX.md:32`; `REPORT_A:104-105,207`).
- El precedente más cercano en la literatura: **[H-TXT]** Kaynov 2024 IJPE reporta que en backorder completo el algoritmo *"cannot consistently outperform the benchmark"* — pero como observación de pasada, sin estimando de equivalencia, sin δ declarado y sin potencia.
- **[INF]** Un resultado de equivalencia **formal** (δ preregistrado, potencia calculada ex-ante, inferencia simultánea sobre celdas) en SCRES-RL: no aparece en ninguno de los 19 TXT. **Esta es novedad real y es metodológica** — que es exactamente el tipo de aporte que CIE acaba de premiar con Guzmán 2026.

**(D) Guardrail de equidad por producto, prospectivo, que se le permite fallar — y falla, con mecanismo documentado.**

- **[H-REP]** `worst_product_fill` frente al clásico: puntos −0,01036 / −0,01573 / −0,00451 con t = −1,82 / **−3,41** / −0,45 y LCB95 −0,0227 / −0,0257 / −0,0263; frente a open-loop: +0,1455 / +0,1963 / +0,4267 (t = +5,96 / +8,16 / +11,7) (`SECOND_OPINION_CLAUDE.md:16-24`).
- **[H-REP]** Mecanismo: el learner **iguala exactamente** al clásico en agregados (`ret_full` y `quantity_ret_full` con punto = se = 0,0 en 12/18 endpoints) y a la vez empeora `max_backlog_age` (+122,7), `service_loss_auc` (+908k) y `unresolved_orders` (+0,20) (`SECOND_OPINION_CLAUDE.md:33-37`). Es **sustitución media↔mínimo**: compra fill agregado desbalanceando el producto débil.
- **[INF]** Esto **generaliza más allá de nuestro entorno**: cualquier paper que optimice un escalar de resiliencia agregado — Ding Eq 55, Kong 0,892, Garrido `R` Eq 6, el propio `ReT` Eq 5.5 — es vulnerable a esta sustitución y **no tiene el endpoint que la detectaría**. Es un modo de fallo del campo, no un defecto nuestro.
- **[INF]** Y hay un agravante que hace la crítica todavía más fuerte: **[H-TXT]** la rama Eq 5.3 de `ReT` vale cero por construcción durante el período de no-recuperación (`WRAP...:3043-3049`) — es decir, la métrica es ciega exactamente en la ventana donde vive el daño de cola. La sustitución media↔mínimo ocurre en la zona muerta de la métrica.

### 2.3 Formulación honesta del hueco

> **[INF]** El campo mide un escalar no identificado, contra un puñado de heurísticas fijas, sin techo, sin potencia declarada y sin endpoint de equidad. Nuestro hueco no es un algoritmo. Es un **aparato de medición con capacidad de falsación para SCRES-RL** — techo enumerado + placebo causal + frontera de comparadores exhaustiva reseleccionada + equivalencia con zona de indiferencia + guardrail de equidad — y el **resultado negativo/de equivalencia que ese aparato produce** cuando se aplica a un DES fiel a la tesis.

**[INF]** Corolario incómodo pero honesto: nuestra contribución es **más fuerte cuanto menos algoritmo tenga**. Si añadimos una arquitectura nueva, competimos en el terreno donde Ding, Kong y Kotecha llegaron antes. Si nos quedamos en la medición, competimos donde no hay nadie.

---

## 3. EL CLAIM PUBLICABLE

### 3.1 Qué tenemos realmente (todo [H-REP], no recomputado por mí)

| Resultado | Estado | Números |
|---|---|---|
| Adaptación vs open-loop (`H_OL`) | **PASA 3/3** | Efectos puntuales +0,0757 / +0,0626 / +0,1045 (`REPORT_A:105`); en confirmación, rango +0,062 a +0,106 con LCB95 simultánea positiva, 10/10 semillas positivas, 84,8–95,7 % de tapes favorables (`SECOND_OPINION_CLAUDE.md:105-107`). *Nota de precisión: la redacción de ese informe no deja claro si +0,062–0,106 son puntos o LCBs; hay que fijarlo contra el artefacto antes de escribir el abstract.* |
| Equivalencia vs mejor clásico (`Delta_N`) | **PASA 3/3** | IC95 simultáneos `[−0,00627,+0,00310]`, `[−0,00552,+0,00408]`, `[−0,00268,+0,00186]` ⊂ ±0,01; N=256, potencia conjunta 0,8755 |
| Techo de información perfecta (`H_PI`) | **MEDIDO** | 0,15151, LCB95 0,11562, placebo fungible exactamente 0, 27 placebos PASS |
| Integridad / reproducibilidad | **PASA** | 21.696 replays, 0 fallos, error máx. `ret_visible` 5,55e-16 |
| Prima neural | **NO ESTABLECIDA** | `neural_premium = false` 3/3; Δ_N puntual negativo |
| No-inferioridad de cola (`worst_product_fill`) | **FALLA 3/3** | LCB95 −0,0227 / −0,0257 / −0,0263 |
| Suite de certificación | **NO VERDE** | 2.260 passed / 38 failed / 7 skipped / 2 xfailed; anchor CSSU `9cb65c7a` ≠ golden `f3fe61b1` |

### 3.2 El claim que un Q1 aceptaría

**No es "nuestra red mejora la resiliencia". Es una descomposición del valor del aprendizaje en dos componentes, una medida contra un techo, y un modo de fallo de seguridad.**

> **CLAIM PRIMARIO — "El valor está en la realimentación, no en la aproximación neural; y el objetivo agregado compra la media a costa del mínimo."**

Cuatro componentes, en este orden:

1. **Positivo grande y bien medido:** el control en lazo cerrado supera al **mejor de 65.536 calendarios open-loop enumerados exhaustivamente**, reseleccionado dentro de cada resample.
2. **Equivalencia preregistrada:** ese mismo controlador es **estadísticamente equivalente** al mejor de 10 controladores clásicos con realimentación y estado rico, dentro de una zona de indiferencia δ = 0,01 fijada antes de abrir semillas, en 3/3 celdas, con potencia conjunta declarada.
3. **Techo medido, no supuesto:** el headroom de información perfecta es 0,15151 (LCB 0,11562) y **el placebo del mecanismo devuelve exactamente 0**; el aprendizaje captura la porción de lazo cerrado de ese techo y no deja residuo atribuible a la aproximación neural.
4. **Seguridad, negativo y mecanístico:** bajo un endpoint agregado de resiliencia, la política aprendida **compra fill medio desbalanceando el producto más débil** — la no-inferioridad de cola falla en 3/3 celdas, con `max_backlog_age` +122,7 y `service_loss_auc` +908k frente al clásico.

### 3.3 Borrador de abstract (así lo escribiría)

**Versión inglesa (para el envío):**

> Discrete-event simulation (DES) models of supply chain resilience (SCRES) are open-loop: they cannot retain what the network learned from past disruptions — the "Alzheimer effect". Embedding a learning agent between the data-gathering and validation stages of the DES is the standard proposed remedy, but the SCRES literature evaluates such agents against a handful of fixed heuristics, on scalarised resilience indices, without measuring how much value is available to be captured in the first place. We introduce a falsification-grade evaluation protocol for learning-based SCRES control and apply it to a military-supply-chain DES with two non-fungible products sharing finite capacity. The protocol has four elements: (i) a perfect-information headroom screen computed before any training, together with a mechanism placebo that must return exactly zero; (ii) a comparator defined as the maximum over an exhaustively enumerated action frontier (4^8 = 65,536 open-loop calendars), re-selected inside every bootstrap resample to remove comparator cherry-picking; (iii) a pre-registered indifference zone (δ = 0.01) so that "no difference" is tested as equivalence with declared power rather than inferred from a non-significant superiority test; and (iv) a per-product equity guardrail that is allowed to fail. Applied to three pre-registered operating cells (N = 256, joint power 0.876, simultaneous studentised max-t, 21,696 independent bit-exact replays), the protocol yields three results. First, closed-loop control beats the best of the 65,536 enumerated open-loop calendars in 3/3 cells with all 10 seeds positive. Second, the recurrent learner is statistically **equivalent** to the best of ten state-rich classical feedback controllers: all three simultaneous confidence intervals lie inside ±0.01. Third, the learner nonetheless **fails** per-product non-inferiority in 3/3 cells, because it buys aggregate fill by unbalancing the weakest product. The measured perfect-information headroom (0.15151, LCB 0.11562, mechanism placebo exactly 0) shows this is not a power artefact: the closed-loop component of the ceiling is captured, and no residual is attributable to neural approximation. We conclude that in this class of SCRES problems the value of "closing the Alzheimer loop" is the value of feedback, not of function approximation, and that scalarised resilience indices are structurally unable to detect the mean-versus-minimum substitution that learned policies exploit.

**Versión castellana (para el PI):**

> Los modelos DES de resiliencia de cadena de suministro son de lazo abierto: no retienen lo aprendido de disrupciones pasadas — el "efecto Alzheimer". Insertar un agente de aprendizaje entre las etapas de recolección de datos y de validación es el remedio propuesto, pero la literatura de SCRES evalúa esos agentes contra un puñado de heurísticas fijas, sobre índices de resiliencia escalarizados, y sin medir cuánto valor hay disponible para capturar. Proponemos un protocolo de evaluación con capacidad de falsación y lo aplicamos a un DES de cadena militar con dos productos no fungibles que comparten capacidad finita. El protocolo tiene cuatro piezas: (i) una pantalla de headroom de información perfecta calculada **antes** de entrenar, con un placebo de mecanismo que debe devolver exactamente cero; (ii) un comparador definido como el máximo sobre una frontera de acciones enumerada exhaustivamente (4^8 = 65.536 calendarios), reseleccionado dentro de cada resample; (iii) una zona de indiferencia preregistrada (δ = 0,01), de modo que "no hay diferencia" se contrasta como **equivalencia con potencia declarada** y no se infiere de un test de superioridad no significativo; y (iv) un guardrail de equidad por producto al que se le permite fallar. Resultados: el lazo cerrado supera al mejor de los 65.536 calendarios enumerados en 3/3 celdas; el learner recurrente es **equivalente** al mejor de diez controladores clásicos con realimentación (los tres IC simultáneos dentro de ±0,01); y **falla** la no-inferioridad por producto en 3/3 celdas porque compra fill agregado desbalanceando el producto débil. El techo medido (0,15151, LCB 0,11562, placebo exactamente 0) descarta que sea un artefacto de potencia. Conclusión: en esta clase de problemas SCRES, el valor de "cerrar el loop Alzheimer" es el valor de la realimentación, no el de la aproximación de funciones; y los índices escalarizados de resiliencia son estructuralmente incapaces de detectar la sustitución media↔mínimo que la política aprendida explota.

### 3.4 Por qué esto es más publicable que una prima neural marginal — argumento explícito

**[INF]** Cuatro razones, y las cuatro son verificables contra el bundle:

1. **Una prima neural marginal sería inmediatamente sospechosa.** Gijsbrechts 2022 MSOM (*"can match"*) y Kaynov 2024 IJPE (*"cannot consistently outperform"*) establecen la expectativa previa de empate contra buenas heurísticas. Un +0,01 nuestro invitaría a un revisor a preguntar por winner's curse, por N, por selección de comparador — y perderíamos. **Una equivalencia bien medida no tiene ese flanco: no hay nada que inflar.**
2. **La equivalencia es un estimando, no una ausencia de evidencia.** El error clásico ("p > 0,05 luego son iguales") es lo que la banda ±0,01 con potencia 0,876 evita. Que la literatura de SCRES-RL no lo haga nunca es precisamente el hueco.
3. **El headroom enumerado convierte el negativo en una medición, no en un fracaso.** "No hay prima neural" es débil. "No hay prima neural **y el techo total es 0,15151, del cual el lazo cerrado captura la parte accesible, con placebo del mecanismo en exactamente 0**" es una afirmación cuantitativa sobre la física del problema. Y viene acompañada de un caso donde el mismo instrumento dice 0,00024 (4.860 evaluaciones, 0 puertas), lo que demuestra que el instrumento **discrimina**.
4. **El fallo de cola es la parte más citable.** Es un modo de fallo con mecanismo, que aplica a todos los papers que optimizan un escalar agregado, y que ninguno de ellos puede detectar con sus endpoints. Un revisor de CIE que acaba de ver publicar a Guzmán 2026 reconoce este tipo de aporte.

### 3.5 Lo que un revisor atacará, y qué necesitamos tener listo

| Ataque previsible | Respuesta que debemos poder dar |
|---|---|
| *"Empataste con un oráculo. ¿Y qué?"* | **Debilidad real.** Gijsbrechts 2022 exige comparador desplegable. Necesitamos un brazo secundario **preregistrado** con MPC de modelo estimado o con la misma observación parcial, y reportar la equivalencia **acotada por ambos lados**: equivalente al oráculo Y equivalente/superior al desplegable. Sin esto el claim es atacable. Ver §5, ítem 3. |
| *"Tres celdas, un entorno."* | Sección explícita de validez externa + respuesta documental de Garrido a RT1–RT5 (¿son `current`/`increased` escenarios MFSC defendibles?). **[H-REP]** hoy están `UNANSWERED_CLAIM_SCOPE_ONLY` (`REPORT_A:317`). Si no se responden, todo lo positivo queda como *researcher stress extension*. |
| *"Tu endpoint es inestable."* | **[H-REP]** `REPORT_A:238-242` documenta que el mismo incumbente congelado da **signos opuestos** según endpoint × bloque (`ret_excel_clipped` +0,0125 vs −0,0122; `full_ledger` −0,0045 vs +0,0084). Si un revisor lo descubre después, nos hunde. **Hay que publicarlo nosotros como tabla de sensibilidad y como hallazgo sobre la familia ReT.** Ver §5, ítem 1. |
| *"El shaping/PBRS rompe la invarianza si truncas."* | Verificación empírica de invarianza (ranking de políticas congeladas idéntico con y sin shaping bajo la misma truncación) antes de reportar cualquier brazo con PBRS. Müller 2025 §3.2/§5. |
| *"Reusaste datos de selección."* | Clean-up de winner's curse con presupuesto fresco y reporte del delta de sesgo (Hong 2021 `a10:2713`; Eckman & Henderson 2021). Ver §5, ítem 5. |
| *"Dices bit-exact pero tu suite está roja."* | **[H-REP]** 38 tests fallando y el anchor CSSU `9cb65c7a` ≠ `f3fe61b1`. **Bloqueante.** Ver §5, ítem 6. |

---

## 4. JOURNAL

Criterios: (a) encaje temático, (b) qué exigiría de nosotros, (c) probabilidad honesta de aceptación **con el paquete de §3 completado según §5**. Las probabilidades son juicio mío **[INF]**, no datos.

### 4.1 *Computers & Industrial Engineering* (CIE) — Elsevier, 0360-8352

- **Encaje:** muy alto. **[H-API]** publica exactamente nuestra intersección: Guzmán 2026 (DT-MARL circular, `10.1016/j.cie.2026.112044`), Habibi, Chakrabortty & Abbasi 2023 (resiliencia de red con propagación de disrupción, `10.1016/j.cie.2023.109531`), Tian et al. 2024 (IACPPO reposición de almacén, `10.1016/j.cie.2023.109829`), Park & Lee 2025 (control predictivo de disrupción, `10.1016/j.cie.2025.111312`), Sriprateep et al. 2026 (DRL híbrido para semiconductores resiliente, `10.1016/j.cie.2025.111583`).
- **La ventaja decisiva:** **[H-TXT]** Guzmán 2026 declara que su contribución primaria es **metodológica** (protocolo de evaluación controlada). Eso es prueba documental de que el editor de CIE compra un paper cuyo aporte es el protocolo. Nosotros nos posicionamos como el paso siguiente: *"extendemos el protocolo de semillas emparejadas + IC95 con (i) un techo medido, (ii) una frontera de comparadores enumerada, (iii) una zona de indiferencia con potencia, y (iv) un guardrail de equidad — y mostramos qué cambia en las conclusiones"*. Es un delta incremental pero real, **con ancla dentro del propio journal**.
- **Qué exigiría:** relevancia de ingeniería aplicada e implicaciones gerenciales explícitas (CIE las pide siempre); un caso computacional sustantivo; figuras de sistema. Tolera resultados negativos si se enmarcan como soporte a la decisión ("cuándo NO invertir en aprendizaje"). Exigirá que citemos a Guzmán, Habibi, Tian y Sriprateep.
- **Riesgo:** que el editor lo lea como "no hay método nuevo". Se mitiga poniendo el protocolo como contribución 1 y el resultado como contribución 2, no al revés.
- **Probabilidad honesta:** **~45–55 %** con el paquete completo; **~20–25 %** si se envía como resultado negativo sin el envoltorio de protocolo.

### 4.2 *International Journal of Production Economics* (IJPE) — Elsevier, 0925-5273

- **Encaje:** alto en tema, **peligroso en posicionamiento**. **[H-TXT/H-API]** Es donde salió **Ding 2026** (`10.1016/j.ijpe.2026.109995`) — nuestro competidor directo — y también Kaynov 2024 (`10.1016/j.ijpe.2023.109088`), Preil & Krapp 2022 (`10.1016/j.ijpe.2022.108578`), Aldrighetti et al. 2021 (`10.1016/j.ijpe.2021.108103`), Asghari et al. 2026 (`10.1016/j.ijpe.2026.110161`).
- **Qué exigiría:** interpretación económica explícita (coste, no sólo servicio); implicación gerencial fuerte; y casi seguro **"¿por qué no hacéis MARL sobre una red, como Ding?"**. Nuestra respuesta ("porque primero hay que medir el techo") es buena pero es una respuesta confrontacional dirigida a un paper del propio journal — y probablemente a sus revisores.
- **Ventaja:** si añadiéramos la lane topológica con nuestro aparato de medición, IJPE sería la elección natural. Pero eso es otro paper y otros 200+ CPU-h.
- **Probabilidad honesta:** **~30–40 %** con el paper actual; **~45 %** si incluyera un mini-experimento topológico con el protocolo.

### 4.3 *European Journal of Operational Research* (EJOR) — Elsevier, 0377-2217

- **Encaje:** alto en la parte metodológica. **[H-API]** Publica el roadmap de Boute 2022 (`10.1016/j.ejor.2021.07.016`), De Moor 2022 (`10.1016/j.ejor.2021.10.045`), Dehaybe et al. 2024 (`10.1016/j.ejor.2023.10.007`), Temizöz et al. 2025 (`10.1016/j.ejor.2025.01.026`), van der Haar et al. 2026 (`10.1016/j.ejor.2026.04.006`), y Cheng et al. 2023 sobre validez finite-sample.
- **Qué exigiría:** novedad **de método OR** o de teoría. Nuestra maquinaria estadística es *aplicación cuidadosa* de R&S, no teoría nueva. Para EJOR habría que **implementar el procedimiento secuencial con su garantía** (KN/FHN con IZ, PCS y PGS reportados) y no sólo invocarlo. Tasa de rechazo editorial alta.
- **Probabilidad honesta:** **~20–25 %** tal como está; **~35 %** si se implementa el procedimiento secuencial con garantía y se presenta como "diseño de evaluación para RL en OM".

### 4.4 *Omega* — Elsevier, 0305-0483

- **Encaje:** medio. **[H-API]** Publica SCRES formal: Ivanov 2024 (analogía del sistema inmune, `10.1016/j.omega.2024.103081`), Sawik 2022 (`10.1016/j.omega.2022.102596`), Liu et al. 2022 (viabilidad multi-echelon, `10.1016/j.omega.2022.102683`).
- **Qué exigiría:** modelado analítico + insight gerencial de alto nivel. Nuestro trabajo es simulación + estadística; Omega es muy selectiva y prefiere el ángulo formal.
- **Probabilidad honesta:** **~12–18 %.** No recomiendo.

### 4.5 *IISE Transactions* — Taylor & Francis, 2472-5854

- **Encaje:** medio-alto en método. **[H-REP]** Publica Kim et al. 2023/2024 (MARL transshipments bajo disrupción). La comunidad de simulación/R&S es nativa de IISE.
- **Qué exigiría:** rigor metodológico de nivel de IE: procedimiento secuencial con garantía demostrada, no usada; probablemente un resultado teórico sobre la validez del gate. También exigiría que la parte SCRES sea secundaria.
- **Probabilidad honesta:** **~15–20 %** hoy; sube si el paper se reescribe como "diseño de experimentos para comparar controladores aprendidos vs estructurados en simulación", que es un paper distinto.

### 4.6 *Transportation Research Part E* (TRE) — Elsevier, 1366-5545

- **Encaje:** medio, con estiramiento de alcance. **[H-API]** Publica Lv 2025 (resiliencia con hipergrafo dinámico y RL cuántico, `10.1016/j.tre.2025.104458`) y Ding 2023 (recuperación de disrupciones aéreas con RL, `10.1016/j.tre.2023.103295`).
- **Qué exigiría:** un ángulo de transporte/logística de red genuino. Nuestro MFSC es una cadena militar con planta y CEDI; el componente de transporte no es el objeto de estudio. Un editor puede rechazarlo por alcance.
- **Probabilidad honesta:** **~18–22 %**, y con riesgo de desk-reject por scope.

### 4.7 *Computers and Chemical Engineering* — Elsevier, 0098-1354

- **Encaje:** bajo. **[H-REP]** Es el hogar de Kotecha 2025, Mousa 2024 y Burtea & Tsay 2024, que son referencias buenas para nosotros — pero la audiencia es process systems engineering y espera una instancia de planta de proceso.
- **Probabilidad honesta:** **~8–12 %.** No recomiendo.

### 4.8 *Annals of Operations Research* (AOR) — Springer

- **Encaje:** amplio; **[H-REP]** el registro ya incluye Katsaliaki 2021 (`10.1007/s10479-020-03912-1`) y Belhadi 2021 (`10.1007/s10479-021-03956-x`), ambos AOR.
- **Qué exigiría:** menos. Es el destino de rescate más probable.
- **Advertencia honesta:** su percepción de prestigio y su cuartil varían por listado y por año; presentarlo como equivalente a CIE/IJPE ante un comité de evaluación sería optimista.
- **Probabilidad honesta:** **~50–60 %**, con menor rédito.

### 4.9 *International Journal of Production Research* (IJPR) — Taylor & Francis, 0020-7543

- **Encaje:** alto, y con una ventaja política concreta: **[H-TXT]** es donde se publicó **Garrido, Pongutá & García-Reyes 2024** (`garrido2024_factory_resilience.txt`, DOI `10.1080/00207543.2024.2425771`) — el PI ya tiene track record ahí. **[H-API]** También publica Rolf et al. 2022 (revisión RL en SCM, `10.1080/00207543.2022.2140221`), Bussieweke et al. 2024, Ivanov 2023, Shukla et al. 2025, Zhang et al. 2025 (MARL federado).
- **Qué exigiría:** encuadre de *production research*, implicaciones gerenciales y una declaración de novedad clara. IJPR es receptiva a simulación + resiliencia + RL y su volumen es alto.
- **Probabilidad honesta:** **~40–50 %.**

### 4.10 Recomendación

**Enviar a *Computers & Industrial Engineering*.** Razones, en orden de peso:

1. **Ancla interna.** **[H-TXT]** Guzmán 2026, publicado en CIE en 2026, declara su contribución primaria como un **protocolo de evaluación**. Eso reduce drásticamente el riesgo editorial de nuestro encuadre "protocolo primero, resultado después", y nos da una frase de posicionamiento inatacable: extendemos su protocolo con las cuatro piezas que le faltan (techo, frontera enumerada, zona de indiferencia, equidad por producto).
2. **CIE publica DES + IA + resiliencia + coste** y espera implicaciones de ingeniería aplicada, que es exactamente el registro de "cuándo NO conviene invertir en aprendizaje".
3. **Tolerancia al resultado de equivalencia/negativo** mayor que EJOR u Omega, que quieren método o teoría nueva.
4. **Evitamos el choque frontal con Ding 2026 en su propio journal**, sin renunciar a citarlo y criticarlo técnicamente (episodio de un paso, ausencia de IC).

**Escalera de fallback:** CIE → **IJPR** (encaje del autor + linaje Garrido 2024) → **IJPE** (sólo si añadimos un experimento topológico con el protocolo) → **AOR** (rescate).

**Companion opcional [PROP]:** la parte puramente estadística (split-tape, sesgo de selección cuantificado, descomposición de `ReT`) da un paper corto de *Winter Simulation Conference* o *Journal of Simulation*, que además sirve de cita de apoyo para el envío a CIE.

---

## 5. QUÉ FALTA PARA ENVIAR

Ordenado por dependencia, no por comodidad. Los esfuerzos son estimaciones mías **[INF]**; el CPU son órdenes de magnitud, no compromisos.

### BLOQUEANTES (sin esto, no se envía)

**1. Resolver la identificación del endpoint.**
Fijar **un** endpoint primario y publicar como tabla de sensibilidad la inversión de signo documentada en **[H-REP]** `REPORT_A_auditorias.md:238-242` (mismo incumbente congelado: `ret_excel_clipped` +0,0125 IC[+0,009,+0,015] 15/16 vs −0,0122 1/12; `full_ledger` −0,0045 vs +0,0084 12/12).
*Por qué es bloqueante:* si un revisor descubre después que el signo depende del endpoint × bloque, el paper muere. Publicándolo nosotros, se convierte en un hallazgo sobre la familia `ReT`.
**Esfuerzo:** 1–2 semanas de analista. **CPU:** ~0 (reanálisis de trazas existentes).

**2. Descomposición Kitagawa/Oaxaca de `ReT` en efecto intra-régimen vs efecto composición.**
`Δ mean(ReT) = Σ_r w̄_r·Δμ_r + Σ_r μ̄_r·Δw_r`, con IC95 pareado por semilla (CRN), reportados por separado y **nunca sumados en un endpoint único**. Justificación: **[H-TXT]** Eq 5.5 es una mezcla cuyos pesos son función de la política (`WRAP...:3121-3134`).
*Por qué es bloqueante:* es la evidencia mecanística de la sustitución media↔mínimo — el componente 4 del claim — y a la vez la defensa de la identificación del endpoint agregado.
**Esfuerzo:** 1 semana. **CPU:** ~0 si las trazas registran el régimen por orden; si no, una re-evaluación con políticas congeladas (bajo).

**3. Comparador desplegable como brazo secundario PREREGISTRADO.**
MPC con modelo **estimado** (no exacto) y/o con la **misma observación parcial** que el learner. Motivación explícita: **[H-TXT]** Gijsbrechts 2022 §comparador justo. **Regla dura:** no degradar el primario post-hoc — eso es p-hacking (`SECOND_OPINION_CLAUDE.md:59-65`); se añade como secundario declarado.
*Por qué es bloqueante:* sin él, "empatamos con un oráculo" es el flanco más fácil de atacar.
**Esfuerzo:** 1–2 semanas + **20–40 CPU-h**.

**4. IC pareados con CRN sobre las diferencias, no IC marginales por brazo.**
Y verificar que CRN no invalida las constantes de los procedimientos que suponen independencia (Hong 2021 §3.1).
**Esfuerzo:** 3–5 días. **CPU:** 0. *Es el mejor ratio valor/coste de toda la lista.*

**5. Clean-up de winner's curse.**
Toda política, checkpoint o configuración seleccionada mirando datos se re-evalúa con presupuesto **fresco e independiente**, y se reporta el **delta de sesgo** = (valor de selección − valor limpio). Base: **[H-TXT]** `a10-hong2021-fem-review-rs.txt:2713` (reusar datos de búsqueda rompe las garantías).
**Esfuerzo:** 1 semana + **10–30 CPU-h**.

**6. Reparar los anclajes de integridad antes de reclamar reproducibilidad.**
**[H-REP]** `SUITE_CERTIFICACION.md`: 2.260 passed / **38 failed**; anchor CSSU calculado `9cb65c7a…` vs golden `f3fe61b1…`; fallos de entorno por `.venv/bin/python` inexistente y `pip freeze` sin pip.
*Por qué es bloqueante:* el componente de integridad del claim (21.696 replays, 0 fallos, error 5,55e-16) es incompatible con una suite roja si un revisor pide el repositorio.
**Esfuerzo:** 1–2 semanas de ingeniería. **CPU-ciencia:** 0.

**7. Preregistro con hash de cualquier brazo nuevo (Q2), antes de generar la primera semilla.**
Debe fijar: estimandos separados `efficacy` / `safety` / `authorization`; margen de cola δ = SESOI = 0,01 como no-inferioridad; familia de endpoints (depurando los 12 degenerados con SE = 0 declarado ex-ante); regla de agregación; criterio de parada; rango de semillas disjunto con stream RNG separado; y **compromiso escrito de publicar el resultado gane o pierda**.
**Esfuerzo:** 3–5 días.

**8. Obtener y leer los 10 papers MANUAL vía CRAI.**
Kim (`10.1080/24725854.2023.2217248`), Fan JIPR (`10.1186/s43065-023-00072-x`), Liu POM (`10.1177/10591478241305863`), Kotecha (`10.1016/j.compchemeng.2025.109111`), Mousa (`10.1016/j.compchemeng.2024.108783`), Burtea & Tsay (`10.1016/j.compchemeng.2023.108518`), Akashi (`10.23919/cnsm59352.2023.10327883`), Ampratwum (`10.1109/compsac61105.2024.00111`), Cheng EJOR (`10.1016/j.ejor.2022.11.038`), Zhou & Peng WSC (`10.1109/wsc60868.2023.10407663`).
*Por qué es bloqueante:* la sección de trabajo relacionado del paper **no puede** describir Kim, Kotecha o Mousa sin haberlos leído; y Cheng es la cita técnica del diseño secuencial. Hoy nadie del proyecto los ha leído.
**Esfuerzo:** 1 día del PI (CRAI EZProxy) + 3–4 días de lectura.

**9. Respuesta documental de Garrido a RT1–RT5.**
**[H-REP]** Hoy `UNANSWERED_CLAIM_SCOPE_ONLY` (`REPORT_A:317-328`). Sin respuesta, el alcance de todo lo positivo queda limitado a *researcher stress extension*.
**Esfuerzo:** depende del PI, no de CPU. **Empezar ya**, es la ruta crítica más larga que no controlamos.

### FUERTEMENTE RECOMENDADOS (suben materialmente la probabilidad de aceptación)

**10. Gate 0 con corrección split-tape, y reportar el sesgo de selección como número.**
Diseño ya auditado en `AUDITORIA_GATE0_SPLIT_TAPES.md`: seleccionar `k*` sobre tapes A, evaluar sólo `k*` sobre tapes B, reportar `G_PI_naive`, `G_PI_split` y su diferencia.
*Valor añadido:* el delta de sesgo es en sí mismo un **resultado metodológico citable** — cuantifica cuánto habría inflado la oportunidad un gate sin split.
**Esfuerzo:** 3 días. **CPU:** ~6 h.

**11. Procedimiento de R&S secuencial con garantía finite-sample.**
KN/FHN con IZ δ = 0,01, α = 0,05, `n0 = 10`, asignación adaptativa por varianza; reportar **PCS y PGS**, no sólo un t-test. Base: Hong 2021 §2.1.2–2.1.3 (PCS-IZ **no** implica PGS) + Cheng 2023 EJOR + Zhong & Hong 2022 (`10.1287/opre.2020.2065`).
*Valor añadido:* convierte "tuvimos potencia 0,8755" en "tenemos una garantía finite-sample". Es lo que subiría el paper de CIE a EJOR/IISE si quisiéramos.
**Esfuerzo:** 1–2 semanas + CPU modesto (el diseño secuencial **ahorra** frente a N fijo).

**12. Métricas de cola y auditoría de la zona muerta.**
CVaR₁₀ del coste y cuantil del tiempo de recuperación como secundarios preregistrados; y `f_dead` = fracción de pasos-orden que caen en la rama `Re(DP_j−RP_j)` que **[H-TXT]** vale cero por construcción. Gate de falsación honesto: si `f_dead < 0,02` en las tres celdas, la crítica de ceguera se descarta; si `f_dead ≥ 0,10`, queda demostrado que `ReT` no puede ser el único endpoint de seguridad.
**Esfuerzo:** 1 semana. **CPU:** nulo–bajo.

**13. Figuras que venden el paper.**
(a) Forest plot por celda de los tres IC de `Delta_N` contra la banda ±0,01 — **ésta es LA figura**; (b) curvas de aprendizaje por checkpoint; (c) barra de descomposición `H_PI` → componente lazo-cerrado capturada → residuo neural ≈ 0; (d) diagrama de sustitución media↔mínimo (fill agregado vs `worst_product_fill`).
**Esfuerzo:** 1 semana.

**14. Verificación empírica de invarianza del shaping** si algún brazo usa PBRS: ranking idéntico de políticas congeladas con y sin shaping bajo la misma truncación de horizonte, y `Φ(terminal) = 0` explícito. Müller 2025 §3.2/§5.1–5.2.
**Esfuerzo:** 3 días. **CPU:** bajo.

### OPCIONALES (sólo si sobra tiempo, y con gate previo)

**15. Brazo HPRS jerárquico** (safety ⊃ target ⊃ comfort, con `safety = worst_product_fill`, `target = ReT`, `comfort = coste`) como **remedio propuesto** al fallo de cola, en contrato nuevo con semillas vírgenes. Falsación fuerte: si el brazo jerárquico mejora la cola exactamente igual que un potencial aditivo plano, la jerarquía no es el mecanismo y el claim se retira.
**Esfuerzo:** **80–150 CPU-h**. *Sólo si Gate 0 pasa.*

**16. Certificado de infactibilidad vía EGO restringido** (CONFIG, `a11-...txt`): maximizar `Delta_N` sujeto a `worst_product_fill ≥ −δ` sobre el espacio de hiperparámetros preregistrado; si declara infactibilidad, "la seguridad falló" pasa de veredicto narrativo a **resultado negativo certificado**.
**Esfuerzo:** **60–120 CPU-h**. Alto valor retórico, hay que aceptar el compromiso por escrito antes de correrlo.

### NO HACER PARA ESTE PAPER

**17. La lane topológica 8–13 nodos.** **[H-TXT]** Es Ding 2026 IJPE. Entrar cuesta 200+ CPU-h y nos deja segundos con un aparato de medición que aún no hemos publicado. **Publicar primero el aparato; la lane topológica es el paper 2, y entonces el aparato ya es citable como nuestro.**

**18. Re-adjudicar Q, O u O-R con otras semillas, otro endpoint u otro margen.** Prohibido por contrato y, además, sería exactamente el margin-shopping que el propio proyecto identificó como su riesgo #1 (`SECOND_OPINION_CLAUDE.md:128-130`).

### Estimación agregada

**[INF]** ~6–10 semanas de trabajo de analista + ~1–2 semanas de ingeniería de suite + **~40–100 CPU-h** para lo bloqueante y lo recomendado. **Ningún entrenamiento nuevo es estrictamente necesario para el claim primario** — el claim ya está en los artefactos; lo que falta es blindarlo. La ruta crítica que no controlamos es el ítem 9 (respuesta de Garrido a RT1–RT5).

---

## 6. REFERENCIAS ADICIONALES (2021+, fuera del registro)

**Verificación:** los 38 DOIs/arXiv IDs siguientes fueron consultados por mí el **2026-08-24** contra `api.crossref.org/works/<DOI>` o `export.arxiv.org/api/query?id_list=<id>`, con `User-Agent: SCRES-IA/1.0 (mailto:thomas.chisica@urosario.edu.co)`. Autor, año, venue y título provienen de esa respuesta. **[H-API]**
**Deduplicación:** comprobado programáticamente contra los 84 DOIs de `registry/BIBLIOGRAFIA_REGISTRO.json` — **0 duplicados**.
**Advertencia:** verifiqué el registro bibliográfico, **no leí estos 38 papers**. No deben citarse con afirmaciones sobre su contenido interno sin leerlos.

### 6.1 El revisor de CIE los esperará: SCRES + IA en el propio journal (5)

| # | Referencia | DOI |
|---|---|---|
| 1 | Habibi F., Chakrabortty R.K., Abbasi A. (2023). *Evaluating supply chain network resilience considering disruption propagation.* **Computers & Industrial Engineering** 183:109531. | `10.1016/j.cie.2023.109531` |
| 2 | Tian R., Lu M., Wang H., Wang B., Tang Q. (2024). *IACPPO: A deep reinforcement learning-based model for warehouse inventory replenishment.* **Computers & Industrial Engineering** 187:109829. | `10.1016/j.cie.2023.109829` |
| 3 | Park S., Lee H. (2025). *Predictive supply chain disruption control framework using causal network-based multi-stream deep learning.* **Computers & Industrial Engineering** 207:111312. | `10.1016/j.cie.2025.111312` |
| 4 | Sriprateep K., Pitakaso R., Khonjun S., Enkvetchakul P., Jirasirilerd G. (2026). *Sustainable and resilient semiconductor supply chain optimization via hybrid deep reinforcement and generative learning.* **Computers & Industrial Engineering** 211:111583. | `10.1016/j.cie.2025.111583` |
| 5 | Ivanov D. (2024). *Supply chain resilience: Conceptual and formal models drawing from immune system analogy.* **Omega** 127:103081. | `10.1016/j.omega.2024.103081` |

### 6.2 DRL en inventario/OM — el estándar contra el que nos medirán (7)

| # | Referencia | DOI |
|---|---|---|
| 6 | De Moor B.J., Gijsbrechts J., Boute R.N. (2022). *Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management.* **EJOR** 301(2):535-545. **Crítico: es el precedente publicado del ítem PBRS de nuestro fix-pack.** | `10.1016/j.ejor.2021.10.045` |
| 7 | Dehaybe H., Catanzaro D., Chevalier P. (2024). *Deep Reinforcement Learning for inventory optimization with non-stationary uncertain demand.* **EJOR** 314(2):433-445. | `10.1016/j.ejor.2023.10.007` |
| 8 | Temizöz T., Imdahl C., Dijkman R., Lamghari-Idrissi D., van Jaarsveld W. (2025). *Deep Controlled Learning for Inventory Control.* **EJOR** 324(1):104-117. | `10.1016/j.ejor.2025.01.026` |
| 9 | van der Haar J.F., van Jaarsveld W., Basten R.J., Boute R.N. (2026). *Industrializing deep reinforcement learning for operational spare parts inventory management.* **EJOR** 334(1):128-140. | `10.1016/j.ejor.2026.04.006` |
| 10 | Preil D., Krapp M. (2022). *Bandit-based inventory optimisation: Reinforcement learning in multi-echelon supply chains.* **IJPE** 252:108578. | `10.1016/j.ijpe.2022.108578` |
| 11 | Wang Y., Minner S. (2024). *Deep reinforcement learning for demand fulfillment in online retail.* **IJPE** 269:109133. | `10.1016/j.ijpe.2023.109133` |
| 12 | Kegenbekov Z., Jackson I. (2021). *Adaptive Supply Chain: Demand–Supply Synchronization Using Deep Reinforcement Learning.* **Algorithms** 14(8):240. | `10.3390/a14080240` |

### 6.3 RL y resiliencia/disrupción en producción y logística (5)

| # | Referencia | DOI |
|---|---|---|
| 13 | Rolf B., Jackson I., Müller M., Lang S., Reggelin T. (2022). *A review on reinforcement learning algorithms and applications in supply chain management.* **IJPR** 61(20):7151-7179. **Es LA revisión que un revisor exigirá.** | `10.1080/00207543.2022.2140221` |
| 14 | Asghari M., Jaber M.Y., Searcy C., Afshari H. (2026). *Disruption-resilient lot-sizing and scheduling using multimodal deep reinforcement learning and distributionally robust evaluation.* **IJPE** 301:110161. | `10.1016/j.ijpe.2026.110161` |
| 15 | Shukla A., Kakde S.T., Mitra R., Mandal J., Tiwari M.K. (2025). *Actor-critic driven deep reinforcement learning for optimising agri-food supply chain.* **IJPR** 63(24):9913-9932. | `10.1080/00207543.2025.2529550` |
| 16 | Zhang B., Tan W.J., Cai W., Zhang A.N. (2025). *Information sharing and confidentiality: a federated multi-agent reinforcement learning approach for supply chain coordination.* **IJPR** 64(11):4217-4235. | `10.1080/00207543.2025.2598025` |
| 17 | Lv B. (2025). *Supply chain resilience modeling based on dynamic hypergraph and quantum reinforcement learning for low-altitude-ground networks.* **Transportation Research Part E** 204:104458. | `10.1016/j.tre.2025.104458` |

### 6.4 SCRES conceptual y cuantitativa — el marco que nos sitúa (6)

| # | Referencia | DOI |
|---|---|---|
| 18 | Aldrighetti R., Battini D., Ivanov D., Zennaro I. (2021). *Costs of resilience and disruptions in supply chain network design models: A review and future research directions.* **IJPE** 235:108103. | `10.1016/j.ijpe.2021.108103` |
| 19 | Dolgui A., Ivanov D. (2021). *Ripple effect and supply chain disruption management: new trends and research directions.* **IJPR** 59(1):102-109. | `10.1080/00207543.2021.1840148` |
| 20 | Ivanov D. (2023). *Two views of supply chain resilience.* **IJPR** 62(11):4031-4045. | `10.1080/00207543.2023.2253328` |
| 21 | Ivanov D., Dolgui A. (2022). *The shortage economy and its implications for supply chain and operations management.* **IJPR** 60(24):7141-7154. | `10.1080/00207543.2022.2118889` |
| 22 | Sawik T. (2022). *Stochastic optimization of supply chain resilience under ripple effect: A COVID-19 pandemic related study.* **Omega** 109:102596. | `10.1016/j.omega.2022.102596` |
| 23 | Chen S., Lin J., Zhuo X., Yin L., Shen J. (2026). *Impact of network nestedness on resistance and recovery of supply chain resilience.* **IJPE** 291:109854. | `10.1016/j.ijpe.2025.109854` |

### 6.5 Ranking & selection, simulación y validez estadística — el corazón de nuestro método (5)

| # | Referencia | DOI |
|---|---|---|
| 24 | Eckman D.J., Henderson S.G. (2021). *Fixed-Confidence, Fixed-Tolerance Guarantees for Ranking-and-Selection Procedures.* **ACM TOMACS** 31(2):1-33. **Es la cita exacta del problema de reusar datos de selección.** | `10.1145/3432754` |
| 25 | Zhong Y., Hong L.J. (2022). *Knockout-Tournament Procedures for Large-Scale Ranking and Selection in Parallel Computing Environments.* **Operations Research** 70(1):432-453. | `10.1287/opre.2020.2065` |
| 26 | Hong L.J., Jiang G., Zhong Y. (2022). *Solving Large-Scale Fixed-Budget Ranking and Selection Problems.* **INFORMS Journal on Computing** 34(6):2930-2949. | `10.1287/ijoc.2022.1221` |
| 27 | Eckman D.J., Henderson S.G., Shashaani S. (2023). *SimOpt: A Testbed for Simulation-Optimization Experiments.* **INFORMS Journal on Computing** 35(2):495-508. | `10.1287/ijoc.2023.1273` |
| 28 | Cakmak S., Zhou E., Gao S. (2021). *Contextual Ranking and Selection with Gaussian Processes.* **2021 Winter Simulation Conference (WSC)**, pp. 1-12. | `10.1109/wsc52266.2021.9715499` |

### 6.6 Rigor experimental en RL — lo que nos separa del estrato C (5)

| # | Referencia | DOI / arXiv |
|---|---|---|
| 29 | Agarwal R., Schwarzer M., Castro P.S., Courville A., Bellemare M.G. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* arXiv:2108.13264 (NeurIPS 2021 — **venue POR VERIFICAR**). **La cita canónica de "vuestras barras de error no significan nada".** | `arXiv:2108.13264` / `10.48550/arXiv.2108.13264` |
| 30 | Patterson A., Neumann S., White M., White A. (2023). *Empirical Design in Reinforcement Learning.* arXiv:2304.01315 (JMLR 2024 — **venue POR VERIFICAR**). | `arXiv:2304.01315` / `10.48550/arXiv.2304.01315` |
| 31 | Eimer T., Lindauer M., Raileanu R. (2023). *Hyperparameters in Reinforcement Learning and How To Tune Them.* arXiv:2306.01324 (ICML 2023 — **venue POR VERIFICAR**). | `arXiv:2306.01324` / `10.48550/arXiv.2306.01324` |
| 32 | Yu C., Velu A., Vinitsky E., Gao J., Wang Y., Bayen A., Wu Y. (2021). *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games.* arXiv:2103.01955 (NeurIPS 2022 D&B). **Es el paper de MAPPO que Ding, Kim, Liu y Kotecha usan; hay que citar la fuente, no sólo a sus usuarios.** | `arXiv:2103.01955` / `10.48550/arXiv.2103.01955` |
| 33 | Gu S., Yang L., Du Y., Chen G., Walter F., Wang J., Knoll A. (2024). *A Review of Safe Reinforcement Learning: Methods, Theories, and Applications.* **IEEE TPAMI**. | `10.1109/tpami.2024.3457538` |

### 6.7 Estructura de red, riesgo y control aprendido (5)

| # | Referencia | DOI / arXiv |
|---|---|---|
| 34 | Kosasih E.E., Brintrup A. (2021/2022). *A machine learning approach for predicting hidden links in supply chain with graph neural networks.* **IJPR** 60(17):5380-5393. | `10.1080/00207543.2021.1956697` |
| 35 | Alvo M., Russo D., Kanoria Y. (2023). *Deep Reinforcement Learning for Inventory Networks: Toward Reliable Policy Optimization.* arXiv:2306.11246. **Venue de publicación POR VERIFICAR.** | `arXiv:2306.11246` / `10.48550/arXiv.2306.11246` |
| 36 | Madeka D., Torkkola K., Eisenach C., Luo D., Foster D.P., Kakade S.M. (2022). *Deep Inventory Management.* arXiv:2210.03137. **Venue POR VERIFICAR.** | `arXiv:2210.03137` / `10.48550/arXiv.2210.03137` |
| 37 | Stranieri F., Stella F. (2022). *Comparing Deep Reinforcement Learning Algorithms in Two-Echelon Supply Chains.* arXiv:2204.09603. **Venue POR VERIFICAR.** | `arXiv:2204.09603` / `10.48550/arXiv.2204.09603` |
| 38 | Ma X., Ma S., Xia L., Zhao Q. (2022). *Mean-Semivariance Policy Optimization via Risk-Averse Reinforcement Learning.* **JAIR** 75:569-595. **Soporte para el eje de cola/aversión al riesgo.** | `10.1613/jair.1.13833` |

### 6.8 Nota sobre "DOI_POR_VERIFICAR"

Ningún DOI de esta lista está sin verificar: los 33 DOIs Crossref respondieron `200` y los 5 arXiv IDs resolvieron con título y fecha correctos. Lo que **sí** marqué como `POR VERIFICAR` son las **sedes de publicación de las cinco entradas arXiv** (#29, #30, #31, #35, #36, #37): el arXiv ID y el DOI `10.48550/*` son firmes, pero la referencia a NeurIPS / JMLR / ICML o al journal final proviene de mi conocimiento y no la verifiqué contra una API hoy. Antes de enviar, resolverlas con DBLP o con la página del autor.

---

## 7. Cierre — lo que NO pude verificar y dos defectos vivos del bundle

### 7.1 Afirmaciones que sólo constan en informes (nunca las recomputé)

Todos los números de Program O, O-R, Q, Track A/B, Step3, bake-off, Cobb-Douglas, el estado de la suite y el desajuste de hash CSSU provienen de `reports/` y `context_reports/`. Los trato como **[H-REP]**. Si alguno no reproduce contra el artefacto JSON original, la parte del claim que dependa de él cae.

En particular hay **una inconsistencia que hay que resolver antes de escribir el abstract**: `SECOND_OPINION_CLAUDE.md:22,105` describe la política congelada de Q como "belief-MPC", mientras que `SECOND_OPINION_CODEX.md:9-12,47` la describe como RecurrentPPO con LSTM 64, MLP [64,64] y 200.192 pasos, y sitúa al belief-MPC dentro de la familia de 10 clásicos (cuyos ganadores fueron `min_cost_flow__2` y `max_pressure__0`, ni siquiera belief-MPC). **Son descripciones incompatibles del brazo primario.** No puedo resolverla desde este bundle. **Es bloqueante para el abstract**, porque el claim entero es "el learner empata al mejor clásico".

### 7.2 Papers que el programa cita con contenido específico y nadie del proyecto ha leído

Kim 2023 IISE, Fan 2023 JIPR, Liu 2024 POM, Kotecha 2025, Mousa 2024, Burtea & Tsay 2024, Akashi 2023, Ampratwum 2024, Cheng 2023 EJOR y Zhou & Peng 2023 WSC. **[H-REP]** `REPORT_FRONTERA_2021-2026.md` §B/§A les atribuye aportes concretos; ninguno tiene TXT en `context_texts/`. La frase "la lane topológica ya es *science-backed*" descansa hoy, verificablemente, sólo sobre Ding 2026, Guzmán 2026 y Kong 2026 — y Ding tiene la reserva grave del episodio de un solo paso.

### 7.3 Defecto de catalogación aún vivo

**[H-TXT]** `context_texts/a11-luo2024-scis-survey-mbrl.txt` **no** es la encuesta de MBRL de Luo et al. 2024 (SCIS): su contenido es **Xu, Jiang, Svetozarevic & Jones, CONFIG, arXiv 2211.00162v4**. Ya estaba señalado en `CLAUDE_COMMON_REVIEW §11.1` y sigue sin corregirse en el manifiesto ni en el registro (donde el DOI `10.1007/s11432-022-3696-5` figura asociado a ese slug). Toda afirmación del programa que se apoye en "a11 = survey MBRL" carece de respaldo en este bundle. *(El desajuste a2/a3 Müller/Forbes que ese mismo informe reportaba **sí** está corregido en la versión actual de `LECTURA_OBLIGATORIA_HARNESSES.md:39-40`.)*

---

**Fin del informe.** Ninguna afirmación de este documento autoriza reabrir, reinterpretar ni re-adjudicar Program O, O-R o Q. Todo lo prospectivo (§5) exige preregistro con hash, semillas y tapes vírgenes, y gates congelados antes de entrenar.
