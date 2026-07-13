# Certificado de frontera y estrategia para los Artículos 2 y 3 — SCRES-IA MFSC DES

**Fecha:** 2026-07-13 · **Autor:** sesión PI raíz (FS local + VPS `ovh-agent-lab` + stack RL/DES/SA completo)
**Retorno terminal:** `SEARCH_ENVELOPE_BOUNDARY_CERTIFIED`
**Artefactos-máquina que gobiernan sobre este texto:** `results/paper2_search/{boundary_certificate,failure_taxonomy,candidate_registry,voi_ceiling_atlas,seed_burn_ledger,artifact_index}.json` y `docs/PAPER2_PAPER3_PROVENANCE_RECONCILIATION.md`.

> Nota de método: el texto principal va en español; los títulos, preguntas de investigación, hipótesis, contribuciones y pasajes listos-para-manuscrito van en inglés académico, según el encargo. Se etiqueta cada afirmación como `THESIS-SUPPORTED`, `REPOSITORY-EVIDENCED`, `LITERATURE-BACKED`, `PI-VALIDATED`, `COMMITTEE-INFERENCE` o `SPECULATION`.

---

## 1. Veredicto ejecutivo

**No existe, dentro del sobre físico admisible de la tesis (Garrido-Ríos 2017, Op1–Op13) ni de sus extensiones mínimamente justificables, un contrato de decisión que exponga *headroom* adaptativo observable, convertible, con recursos iguales y sin sacrificio de equidad, por encima de la mejor política estática/periódica de mismo contrato bajo la métrica canónica ReT.** `REPOSITORY-EVIDENCED + PI-VALIDATED`.

- **Artículo 2 positivo (control adaptativo desplegable): NO sostenible hoy.** El candidato más fuerte jamás medido (convoy finito DRA-2b) tiene `H_PI = 0.0221` de ReT pero `H_obs ≈ 0` (Programa E: 0/10 semillas PPO, árbol, heurística y mezcla convexa no convierten). Un barrido fresco de 64 celdas (esta sesión) confirma la firma: `H_PI` material (media 0.0135; 69% de celdas ≥ 0.01) con `H_obs` medio **−0.0079** y η medio **−0.79**.
- **Artículo 3 positivo (aprendizaje retenido): NO autorizado.** Requiere primero un control adaptativo confirmado (arriba, ausente). K3 —el único "triunfo neuronal" aparente— fue retractado: PPO emitió un único calendario fijo de 8 semanas que un cronograma estático período-8 reproduce y supera al MPC con recursos idénticos (`RETRACT_K3_…_STATIC_PERIOD8_CONFOUND`, reproducido bit-a-bit esta sesión).
- **La única ruta honesta a un Artículo 2/3 *positivo* pasa por dos hechos de dominio que la tesis no aporta** (familias `R09` vencimiento-de-misión/admisión y `R03` desvío-de-ruta con flota persistente). Se entregan como preguntas falsables exactas para Garrido (§16).

**Portafolio recomendado (fuerte, honesto, publicable bajo cualquier resultado):**
- **Artículo 2 (inmediatamente ejecutable):** *un techo de valor-de-información y un atlas de invariantes físicas que predice, sin entrenar, qué derechos de decisión están dominados por calendario/constante.* Distinto del Artículo 1.
- **Artículo 3 (condicional pero totalmente especificado):** *el inverso constructivo* — las condiciones necesarias para valor adaptativo desplegable y la elicitación pre-registrada a Garrido sobre `R03/R09` que las **convierte** (positivo) o las **cierra-certifica**. Fallback autosuficiente: *reversión de ranking inducida por métrica / validez de constructo de resiliencia* (Programa G).

**Recomendación decisiva única:** construir el Artículo 2 como certificado de frontera (no requiere nuevos datos), y **antes** de cualquier compromiso, hacer una única reunión de validación con Garrido con las dos preguntas de §16; su respuesta determina si el Artículo 3 es un estudio de conversión positivo o un cierre-certificado. No entrenar ninguna red hasta que exista `H_obs > 0` desplegable.

---

## 2. Estado verificado del repositorio

`REPOSITORY-EVIDENCED`. Detalle completo en `docs/PAPER2_PAPER3_PROVENANCE_RECONCILIATION.md`.

| Eje | Estado verificado |
|---|---|
| Capacidad | FS local ✅, git ✅, VPS `ovh-agent-lab` 135.148.42.12 (6 vCPU/11 GB, ocioso) ✅, Python 3.11.15 + gymnasium 1.3.0 + sb3 2.9.0 + **sb3_contrib (MaskablePPO)** + simpy 4.1.2 + torch 2.12.1 + SALib ✅. **No** hay bloqueo por *tooling*. |
| HEAD local | `codex/paper2-maintenance-headroom @ ef6b53b7`, **7 commits sin publicar** (`373aa6ab..ef6b53b7` = J/K/K2/K3). `git branch -r --contains ef6b53b7` → NONE. |
| Ciencia publicada más avanzada | `origin/program-g/structured-spatial-headroom @ 9b758d45` (hasta Programa I). |
| Divergencia | Tres cabezas sobre `9b758d45`: (a) HEAD local +7 (ciencia J–K3), (b) `origin/audit/provenance-gate-2026-07-13 @ ac6bb790` +2 (una *gate* de sesión previa solo-conector), (c) `origin/main @ c6e6d08b` (base obsoleta). |
| Gate previo | La sesión solo-conector devolvió `BLOCKED_PROVENANCE` porque le faltaban exactamente los objetos locales que **esta** sesión posee; queda **superada en el eje de procedencia**. |
| K3 | Retracción reproducida bit-a-bit (`fixed_minus_mpc_ordered_D0=[0,0,0]`; fijo período-8 supera al MPC por ReT≈0.0177). |

**Acción de procedencia requerida (aprobación del usuario):** etiquetar+publicar los 7 commits locales (tag `program-k3-retraction-2026-07-13` en `ef6b53b7`). Hoy la evidencia decisiva de la retracción K3 vive **solo en disco local** — punto único de fallo.

---

## 3. Frontera del Artículo 1

`REPOSITORY-EVIDENCED` (`docs/manuscript_current/submission/elsevier/`). El Artículo 1 es **benchmark-céntrico** y responde *cuándo funciona / cuándo falla* el RL en control operacional de resiliencia. Sus **cinco contribuciones**:

1. Un benchmark DES+RL con anclaje a Garrido, métricas ReT compatibles con la tesis y fronteras estáticas densas.
2. Resultado **negativo** de benchmark bajo el alcance original buffer/turno (Track A): ningún aprendiz convierte el pequeño *headroom* medido.
3. La ventaja PPO frente a una frontera **restringida** se **revierte** cuando la política constante se optimiza sobre el **mismo contrato completo** (sensibilidad de familia-de-comparador; PPO−estático `−0.000018`, IC95 bajo cero).
4. Diagnóstico de alcance-de-comparador (ablación de espacio-acción, tabla estática por régimen, reentrenamiento con observación enmascarada, matriz régimen×horizonte).
5. Auditoría adversarial de *prevención* causal: una señal pre-evento aparentemente positiva fue diagnosticada como artefacto de medición y retractada; el techo de previsión perfecta no halla *headroom* anticipatorio.

**Implicación para los Artículos 2/3:** el Artículo 2 **no puede** reutilizar el marco de elegibilidad/comparador ni la frontera de prevención (todo eso es Artículo 1, sobre Track A/B). El aprendizaje retenido está **reservado como *future work* del Artículo 1**. Por tanto el Artículo 2 debe anclarse en las familias **D–K** (convoy finito, espacial CSSU, mantenimiento, reabastecimiento) que el Artículo 1 nunca toca, y el Artículo 3 no puede ser un mero L(t−1) de retención.

---

## 4. Mapa de fallos de todas las líneas previas

`REPOSITORY-EVIDENCED` (17 familias, `results/paper2_search/failure_taxonomy.json`). Flecha de fallo = primer eslabón que se rompe en la escalera *autoridad física → reversión de ranking → H_PI → H_obs → conversión OOS → valor aprendido → valor retenido*.

| Familia | Verdicto | H_PI (ReT) | H_obs | Flecha de fallo |
|---|---|---|---|---|
| A — buffer/turno (Track A) | BOUNDARY | pequeño | ninguno | conversión observable |
| B — despacho aguas-abajo (Track B) | RETIRE tras reversión de mismo-contrato | restringido | revierte | comparador restringido |
| C — campañas no-estacionarias con costo | FAIL fase-oráculo | — | — | oráculo/precio |
| D1 — prioridad de backorder | STOP | 0.0011 | tree +23.25% pérdidas | magnitud + conversión + guardarraíl pérdidas |
| DRA1 — asignación CSSU | STOP | 8.8e-5 | — | magnitud + acción dominante |
| **DRA2/2b — convoy finito** | STOP pre-árbol | **0.0221** | — | **solo-info-perfecta** + horizonte + servicio + recursos |
| E — conversión observable convoy | STOP validación | 0.0116 (restr.) | −5e-5 | **info insuficiente / colapso del aprendiz (0/10)** |
| F — portafolio de mitigación | STOP screen | 0.0226 (máx) | 0.0044 (máx) | conversión observable (0/24 celdas) |
| G — compromiso espacial | STOP | 0.0164 | **−0.0212** | reversión de métrica + concentración + calendario ABAB dominante |
| H — rescate belief-state | STOP (cota floja) | 0.0164 | +0.0023 (IC cruza 0) | limitado-por-información + equidad |
| I — control-tesis (branching) | STOP | 3.95e-5 | — | magnitud + horizonte + recursos |
| I — región de escasez concurrente | FAIL equidad | — | +0.010 OOS | **starvation de un teatro (worst-CSSU)** |
| J — mantenimiento finito | STOP | 4.7e-5 | −4.9e-5 | magnitud + **calendario abierto dominante** + PPO 0/6 |
| K — perecibilidad | EXPLORATORIO inválido | — | — | **física incompatible con la tesis** (3 años no-perecedero) |
| K2 — reabastecimiento con costo | EVPI-dominado | 15820 (costo J) | −202 | solo-info-perfecta + comparador clásico fuerte |
| **K3 — reabastecimiento presupuestado** | **RETRACTED** | — | 0.0 vs fijo | **confusión de cronograma abierto período-8 + frontera estática incompleta** |
| post-K3 — migración de cuello de botella | STOP | — | −0.0013 | acción robusta dominante + servicio −3.03% |

**Mecanismos recurrentes:** (i) *constante/calendario robusto dominante* (D1, DRA1, G, J, K3, bottleneck); (ii) *información perfecta sin conversión observable* (DRA2b, E, F, H, K2); (iii) *reversión de métrica o guardarraíl* (D1, G, región-I); (iv) *compra de recursos/comparador* (DRA2b, I, K3); (v) *liveness de endpoint sub-umbral* (D1, DRA1, I, J); (vi) *colapso del aprendiz a calendario abierto* (E, J, K3); (vii) *física fuera del producto de la tesis* (K).

---

## 5. Marco formal de *headroom*

`COMMITTEE-INFERENCE` (formalización) sobre la escalera del encargo. Para estado físico `X`, observación desplegable `O=g(X)`, acción `A`, presupuesto multidimensional `B`, tape exógena `ω`, resultado canónico `Y` (mayor = mejor):

- Valor de open-loop más fuerte: `V_OL*(B) = max_{σ∈Σ_OL} E_ω[Y(σ;ω)]`.
- Techo de información perfecta: `V_PI*(B) = E_ω[max_{σ∈Σ(B,ω)} Y]`.
- **Headroom clairvoyant (EVPI):** `H_PI = V_PI* − V_OL*`.
- **Headroom observable (VSS-análogo):** `H_obs = max_{π∈Π_NA(O)} E_ω[Y(π)] − V_OL*`.
- **Valor de política desplegable:** `H_policy = J(π_obs) − J(π_static*)`.
- **Valor aprendido:** `H_learned = E[Y(π_RL)] − max{V_OL*, V_heur*, V_tree*, V_belief*, V_MPC*, V_DP*}`.
- **Valor retenido:** `H_retained = E[Y(π_persistente)] − E[Y(π_reset)]`.
- **Ratio de conversión (solo si H_PI>0):** `η = H_obs / H_PI`.

**Condición necesaria diagnóstica (derivada del corpus):** el mecanismo objetivo exige que **no exista** un cronograma open-loop independiente-de-tape que iguale a la retroalimentación en todas partes. Las invariantes físicas de la tesis **violan** esta condición para cada derecho de decisión examinado (§8/§11). Corolarios empíricos:
- **`H_PI` grande no implica valor desplegable** (DRA2b: H_PI=0.022, η→0).
- **η negativo** (atlas: media −0.79) = la política observable *destruye* valor frente al calendario estático fuerte pese a EVPI material.
- **Precisión de clasificación de acción-oráculo ≠ valor de política** (E: alta precisión de primera-acción, H_obs≈0).
- **Una política aprendida que reproduce un calendario fijo tiene valor adaptativo cero** (K3).

---

## 6. Registro de familias de enfoque (búsqueda independiente)

`REPOSITORY-EVIDENCED` (`results/paper2_search/candidate_registry.json`). Método: **49 agentes** (12 proponentes ciegos, una región-mecanismo distinta cada uno × panel adversarial de 3 lentes + síntesis; 0 errores, 2.9M tokens). Resultado: **10 FALSIFIED, 2 BLOCKED_PENDING_PI, 0 ACTIVE, 0 SURVIVES.**

| Región | Familia | Panel (S/F/B) | Flecha | Kit-de-muerte (contraejemplo/hecho) |
|---|---|---|---|---|
| R01 | cuadrilla-reparación condición | 0/3/0 | calendario dominante | Serie de un solo servidor ⇒ tiempo-de-restauración *invariante-al-orden*; reabre J por relabeling de época. |
| R02 | reasignación WIP finito bajo bloqueo | 0/3/0 | calendario dominante | Almacenamiento finito **contradice** §6.5 (storage ilimitado); sin señal aperiódica pre-acción. |
| **R03** | **flota persistente + desvío de ruta** | **0/1/2** | blocked_pending_pi | Buffers de 120h absorben un retraso R22 de 24h ⇒ H_PI≈0 salvo que R22 sea ruta-específica y autocorrelada. |
| R04 | inspección vs throughput | 0/3/0 | calendario dominante | R14 con detección total ⇒ palanca *ausente de acción*; lote 5000/48h lava transitorios. |
| R05 | timing de liberación/reserva | 0/3/0 | calendario dominante | Sin señal R24 anticipada ⇒ base-stock constante óptimo; hold no-conservativo dispara pérdidas (D1). |
| R06 | prepos. multi-echelon + R23 | 0/3/0 | calendario dominante | Topología niega canal a teatro en outage ⇒ stock varado; 24h≪120h ⇒ pooling central reactivo domina. |
| R07 | mix de producto / kit | 0/2/1 | calendario dominante | Shocks producto-simétricos ⇒ ELSP cíclico fijo iguala en toda tape (H_PI=H_obs=0). |
| R08 | sensado de demanda censurada | 0/3/0 | calendario dominante | Backorder registra cantidad *solicitada* ⇒ demanda **no censurada** ⇒ EVPI de sensado ≈ 0. |
| **R09** | **vencimiento-misión + admisión** | **0/1/2** | blocked_pending_pi | Sin deadlines, colapsa a D1; con deadlines, EDD+Moore-Hodgson miope lo captura. |
| R10 | cuadrilla móvil / cuello móvil | 0/3/0 | calendario dominante | Pinza de escalas de tiempo (24h reposición ≈ R22 / ≫ R11); reabre post-K3. |
| R11 | anticipación de régimen | 0/3/0 | calendario dominante | Storage ilimitado + ReT ciego-a-holding ⇒ "máxima prontitud" constante; reabre Track C. |
| R12 | secuenciación de recuperación | 0/3/0 | calendario dominante | Un servidor no-preemptivo ⇒ tiempo total = suma (invariante-al-orden); permutación reverso-topológica fija clairvoyant-óptima. |

**Las dos únicas familias con vida (BLOCKED_PENDING_PI)** son mecanismos *genuinamente nuevos* pero condicionados a un hecho de dominio ausente **y** con un *kill* de comparador fuerte que probablemente vencería la conversión incluso si se desbloquean. Ninguna es ACTIVE.

---

## 7. Matriz de candidatos rankeada

`REPOSITORY-EVIDENCED`. Orden por proximidad a un positivo desplegable (mejor primero):

| # | Familia | Estado | Techo H_PI condicional | Riesgo residual aun desbloqueada |
|---|---|---|---|---|
| 1 | **R09 vencimiento-misión/admisión** | BLOCKED_PENDING_PI | ~0.01–0.02 si hay deadlines duros | colapsa a regla miope EDD+Moore-Hodgson y a D1 en ReT canónica |
| 2 | **R03 desvío-de-ruta flota persistente** | BLOCKED_PENDING_PI | ~0.01–0.02 si R22 ruta-específica+autocorr. | "always-primary + buffer" domina a menor vehicle-hours |
| 3 | R07 kit-completeness | FALSIFIED | 0 bajo simetría; >0 sólo con shocks idiosincráticos (prohibidos Op1-2) | degenera a 21 copias de Track A |
| 4 | R06 prepos. lead-time | FALSIFIED | negativo (stranding) | reserva central constante domina |
| 5 | R05 release-timing | FALSIFIED | base-stock constante | guardarraíl de pérdidas (D1) |
| 6–12 | R01/R02/R04/R08/R10/R11/R12 | FALSIFIED | ≈0 o sub-umbral | invariante-al-orden / storage-inventado / info-relabeling / acción-ausente |

**Candidato más fuerte meramente-condicionado: R09.** **Candidato ACTIVE (sin nuevos hechos): NONE.**

---

## 8. El ambiente decisivo para el Artículo 2

Dado que ningún ambiente *positivo* sobrevive, **el "ambiente" decisivo del Artículo 2 es el propio certificado de frontera**: el objeto científico es *el techo de valor-de-información y el atlas de invariantes*, no una nueva física. `COMMITTEE-INFERENCE`.

**Por qué tiene mayor probabilidad de aporte que cualquier lane previa:** cada lane previa buscó *un* positivo y falló localmente. El Artículo 2 convierte el patrón en teoría: mapea cada **mecanismo de fallo** a la **invariante física de la tesis** que lo genera (§4/§11), de modo que un practicante pueda **leer la física de un derecho de decisión y predecir sin entrenar** si estará dominado por calendario/constante. La ancla empírica cuantitativa es la descomposición `H_PI`-vs-`η` (esta sesión: `H_PI` material en 69% de celdas, `η` medio −0.79), replicada en full-DES (DRA2b 0.022/≈0; K2; H) y en el adaptador estilizado.

**Detección de fallo *antes* del RL:** el mismo *gate* que cerró D–K (branching CRN exacto → `H_PI`; política observable no-neuronal fuerte → `H_obs`; matching de recursos; auditoría de traza-de-acción) es la herramienta reutilizable que el Artículo 2 formaliza y que garantiza detectar el fallo antes de gastar cómputo en entrenamiento.

---

## 9. Diseño experimental del Artículo 2

`COMMITTEE-INFERENCE`. No requiere datos nuevos; usa artefactos existentes + el atlas fresco de esta sesión.

- **Unidad de análisis:** familia-de-decisión × celda-de-parámetro. Estimando: `H_PI`, `H_obs`, `η`, guardarraíles (worst-CSSU, pérdidas, ReT cantidad, cola).
- **Ancla full-DES:** DRA2b (H_PI=0.0221), E (0/10), F (0/24), G (ABAB domina), H (belief no-pass), J (0/6), K2 (EVPI-dominado), K3 (confusión período-8).
- **Ancla estilizada (fresca, esta sesión):** `results/paper2_search/voi_ceiling_atlas.json` — LHS 64 celdas × 48 tapes frescas (seed0 7.200.001+), `headroom_at`. Reportar la distribución de `η` y la correlación `H_obs`~(surge×commonality) que localiza los positivos raros en la región de *starvation*.
- **CRN/causalidad:** RNG con clave de evento; mismo snapshot + misma tape + acción distinta ⇒ logs exógenos bit-idénticos.
- **Regla de promoción/stop (congelada ex-ante):** una familia es "RL-elegible" **solo** si `LCB95(H_obs) ≥ 0.01`, mejora co-direccional de servicio, recursos ≤, sin aumento de pérdidas, sin starvation, y la auditoría de traza-de-acción prueba retroalimentación genuina. **Ninguna familia pasa** ⇒ certificado de frontera.
- **Figuras/tablas mínimas:** (Fig1) escalera de arrows con la familia que rompe cada eslabón; (Fig2) atlas `H_PI` vs `η` con la línea η=1 y el umbral 0.01; (Fig3) traza-de-acción K3 (1 secuencia única) vs calendario período-8; (Tabla1) 17 familias × invariante-generadora; (Tabla2) descomposición EVPI/η full-DES vs estilizado.

---

## 10. Blueprint de manuscrito — Artículo 2 (inglés)

**Title (primary):** *Perfect-information headroom without deployable value: a structural characterization of when adaptive control fails in supply-chain resilience operations.*
**Title (alt):** *A physics-derived falsification atlas for reinforcement-learning eligibility in a military food supply-chain simulation.*

**Research question.** In a thesis-grounded military food supply-chain DES, which decision rights admit deployable, observable, resource-equal adaptive value over the strongest same-contract open-loop policy under the canonical resilience endpoint — and which physical invariants a-priori preclude it?

**Hypotheses.**
- H1 (EVPI ceiling): materially positive clairvoyant headroom `H_PI` co-exists with non-positive observable headroom `H_obs` across the connected admissible region (a value-of-information ceiling), so `η = H_obs/H_PI ≤ 0` on average.
- H2 (invariant map): each failure mechanism is generated by an identifiable thesis-native physical invariant (unlimited storage; stationarity; holding-cost-blind ReT; buffer damping of short memoryless disruptions; single-server order-invariance; absence of a pre-action forecast channel).
- H3 (open-loop confound): apparent learned adaptive wins in this class are reproducible by a tape-independent calendar once the static frontier is completed to the learner's full horizon (K3).

**Contribution statement (four).** (1) A formal `H_PI`/`H_obs`/`η` decomposition operationalized as a pre-learner eligibility screen with common-random-number exact branching and an action-trajectory certificate. (2) An empirical value-of-information ceiling: material EVPI without observable conversion, demonstrated across eight full-DES decision families and a 64-cell stylized atlas (`η̄ = −0.79`). (3) A physics-invariant → failure-mechanism atlas that predicts calendar/constant dominance a-priori from a decision right's physics. (4) A reproducible open-loop-confound audit (the period-8 case) showing how an incomplete static frontier manufactures a spurious neural win. `PI-VALIDATED` for (1)–(2)-(4); `COMMITTEE-INFERENCE` for the generality of (3).

**Manuscript-ready abstract (draft).**
> Reinforcement learning is increasingly proposed for adaptive supply-chain resilience, yet negative results are rarely characterized structurally. Using a thesis-grounded discrete-event simulation of a 13-operation military food supply chain, we develop a pre-training eligibility screen that decomposes the value of state feedback into clairvoyant headroom (`H_PI`, an EVPI analogue), deployable observable headroom (`H_obs`), and their ratio `η`, all measured with common random numbers and a mandatory action-trajectory certificate. Across eight full-DES decision families — finite-convoy dispatch, spatial CSSU allocation, mitigation-portfolio commitment, maintenance allocation, and budgeted replenishment — and a 64-cell stylized atlas, we find a robust *value-of-information ceiling*: materially positive `H_PI` (up to 0.022 canonical ReT; 69% of atlas cells ≥ 0.01) co-exists with non-positive `H_obs` (mean −0.008; `η̄ = −0.79`), because thesis-native physical invariants — unlimited storage, stationary demand, a holding-cost-blind resilience metric, proactive buffers that damp short memoryless disruptions, single-server order-invariant restoration, and the absence of a pre-action forecast channel — furnish, for every decision right examined, a tape-independent open-loop schedule that matches feedback everywhere. We further show that an apparent neural win reduces to a fixed period-8 calendar once the static frontier is completed to the learner's horizon. We map each failure mechanism to its generating invariant, yielding a physics-derived atlas that lets practitioners predict RL ineligibility without training. The study reframes a null as a positive methodological and structural contribution and delimits precisely which disclosed physical relaxations would reopen adaptive value.

**Manuscript-ready contribution paragraph (intro).**
> This paper contributes a physics-derived eligibility theory for adaptive control in resilience operations. Where prior work reports whether a learner beat a baseline, we characterize *why* — decomposing state-feedback value into clairvoyant and observable components, validating the decomposition with exact common-random-number branching and an action-trajectory certificate, and mapping the resulting failure mechanisms to the physical invariants that generate them. The military food supply chain serves as a fully worked negative certificate: a value-of-information ceiling in which perfect-information headroom is real but not deployable, and an open-loop confound in which an apparent neural advantage is a calendar in disguise.

**Claim ladder (publishable under every terminal outcome).**
- *No headroom:* report the invariant atlas + `H_PI` sub-bar map.
- *Clairvoyant-only headroom (current):* the EVPI ceiling is the headline (`H_PI`>0, `η`≤0).
- *Interpretable-policy win (if Garrido reopens R03/R09):* positive adaptive control, neural increment reported as absent.
- *RL parity / RL increment:* becomes the seed of a positive Artículo 2′; not claimed now.

---

## 11. Condiciones de elegibilidad del Artículo 3

`COMMITTEE-INFERENCE`. El Artículo 3 (retención positiva) exige primero `H_learned > 0` confirmado, hoy ausente. Se formalizan **ocho condiciones necesarias** para valor adaptativo desplegable, cada una con la invariante de la tesis que la bloquea y la perturbación física mínima que la satisfaría:

| # | Condición necesaria | Invariante que la bloquea | Perturbación mínima divulgada que la satisface |
|---|---|---|---|
| 1 | Recurso compartido escaso genuino | Storage ilimitado (§6.5) | Cap de almacenamiento finito y físico (validado por Garrido) |
| 2 | Reversión de ranking dependiente de estado | Demanda estacionaria | No-estacionariedad con régimen persistente |
| 3 | Señal pre-acción con lead ≥ lead físico | Sin canal de pronóstico; R24 onset=impacto | Indicador líder nombrado con lead cuantificado |
| 4 | Consecuencia intertemporal no amortiguada | Buffers 120h amortiguan disrupciones cortas | Disrupción ruta-específica autocorrelada (R03) |
| 5 | Acción no-invariante-al-orden | Serie/servidor-único ⇒ invariante | Recursos paralelos con contención real |
| 6 | Endpoint sensible a la reacción | ReT ciego-a-holding | Costo de holding/obsolescencia en el endpoint |
| 7 | Estimando que distingue reacción de pérdida | Backorder = pérdida en denominador | Deadline duro + abandono permanente (R09) |
| 8 | Autoridad de decisión no-conservativa | Órdenes siempre backorder | Admisión/triage con autoridad doctrinal (R09) |

**Regla:** el Artículo 3 sólo procede como *retención* si una perturbación divulgada satisface ≥1 condición **y** una política observable supera 0.01 ReT con servicio co-direccional y sin starvation. En caso contrario, el Artículo 3 es el *inverso constructivo + cierre-certificado*.

---

## 12. Diseño experimental del Artículo 3

`COMMITTEE-INFERENCE`. Dos modos, decididos por la respuesta de Garrido.

**Modo A — conversión (si R03/R09 se desbloquean favorablemente).** Pre-registrar el contrato con hecho-de-dominio congelado, cota clairvoyant restringida-por-recursos, conversión clásica observable (EDD+Moore-Hodgson para R09; MPC de ruta para R03), celda nula, guardarraíles y auditoría completa de traza/frontera-estática **antes** de cualquier PPO. Si `LCB95(H_obs) ≥ 0.01` en tapes vírgenes ⇒ Artículo 2/3 positivo de control adaptativo; sólo entonces evaluar incremento neuronal y, por encima de eso, retención (campañas recurrentes: persistente vs reset vs frozen vs scratch, mismo multiconjunto y cómputo, `H_retained = E[ReT_persist] − E[ReT_reset]`, IC95 > 0).

**Modo B — cierre-certificado (si los hechos no aterrizan).** Ejecutar el barrido de frontera bajo la instanciación más favorable *consistente con la tesis* y certificar `H_obs` sub-umbral o solo-clairvoyant, cerrando las dos últimas familias condicionadas.

**Arms de retención (Modo A, si aplica):** estático, heurístico, frozen, pesos persistentes, reset-local, persistente+optimizador, scratch mismo-cómputo, replay, belief/estimación estructurada, historias A/B/C con mismo multiconjunto en distinto orden. **Probes:** frío común, backward-transfer, novel same-support, OOD, final idénticos. Separar carry-over físico / normalización / replay del estado de pesos.

---

## 13. Blueprint de manuscrito — Artículo 3 (inglés)

**Title (primary, conditional):** *The constructive inverse of an eligibility boundary: which disclosed relaxations convert a calendar-dominated resilience decision into adaptively controllable — a preregistered domain-elicitation and conditional-design study.*
**Title (fallback, self-contained):** *Metric-induced policy ranking reversal and construct validity of resilience measurement in a supply-chain simulation.*

**Research question (primary).** What minimal, operationally disclosed relaxation of the thesis assumptions flips a resilience decision family from open-loop-dominated to adaptively convertible, and do the two nearest-real candidates (route diversion; mission-expiry admission) actually convert under equal resources and canonical ReT?

**Hypotheses.** (H1) The eight necessary conditions are jointly sufficient for `H_obs > 0` in a controlled toy realization. (H2) R09 converts iff hard deadlines tighter than recovery timescales exist; else it collapses to D1. (H3) R03 converts iff R22 is route-specific and persistently autocorrelated; else "always-primary + buffer" dominates.

**Contribution statement.** A necessary-condition theory for deployable adaptive value; a per-condition catalogue of minimal disclosed physics perturbations; a preregistered Garrido elicitation instrument; and a conditional confirmatory design that either establishes the project's first positive adaptive result or certify-closes the last two gated mechanisms. `SPECULATION` until Garrido facts land; the *design* is `COMMITTEE-INFERENCE`.

**Manuscript-ready abstract (draft, primary).**
> Negative results in adaptive supply-chain control are rarely turned into prospective theory. Building on a boundary certificate that shows a value-of-information ceiling across a thesis-grounded military food supply-chain simulation, we formalize eight necessary conditions for deployable adaptive value and derive, for each, the minimal operationally disclosed relaxation of the source model that would satisfy it. We instantiate the theory on the two nearest-real candidates — finite-fleet route diversion under route-specific disruption, and mission-expiry admission triage — as a preregistered domain-elicitation and conditional-design study whose falsifiable gates determine, in advance, whether each family converts to positive adaptive control or is certify-closed. The study converts a null into a forward research program and specifies exactly which validated domain facts would reopen learned adaptive value.

**Claim ladder.** convert-both / convert-one / close-both — publishable in every branch; the self-contained construct-validity fallback (Program G reversal) needs no Garrido facts.

---

## 14. Splits alternativos de artículos

`COMMITTEE-INFERENCE`. Evaluados y descartados/retenidos:

- **(Recomendado) 2 = frontera VoI+atlas; 3 = inverso constructivo + elicitación.** Máxima distinción de Artículo 1; ambos publicables bajo cualquier resultado; sin *salami*.
- **Alternativa fuerte: 3 = validez-de-constructo/reversión-de-métrica (Programa G), autosuficiente sin Garrido.** Buen *fallback* si la elicitación se retrasa; distinto de 1 y 2.
- **Descartado: 2 = "metodología de descubrimiento de derechos-de-decisión" sola.** Riesgo de solaparse con la elegibilidad del Artículo 1 (Reviewer 2 lo leería como continuación).
- **Descartado por ahora: 3 = aprendizaje retenido.** Requiere positivo adaptativo previo (ausente); permanece como *future work* del Artículo 1.
- **Rechazado: partir el atlas en 2 papers.** Salami.

---

## 15. Estrategia de revistas

`LITERATURE-BACKED` (ajustes de alcance) + `COMMITTEE-INFERENCE`.

| Artículo | Objetivo primario | Fallback | Ajuste de alcance | Riesgo principal de rechazo |
|---|---|---|---|---|
| **2 (frontera VoI+atlas)** | *Simulation Modelling Practice and Theory* o *Journal of Simulation* (metodología+DES, tolerante a nulos estructurados) | *Computers & Industrial Engineering* | metodología reutilizable + implicaciones gerenciales (cuándo NO desplegar RL) | "es un negativo" → mitigar con el atlas de invariantes como teoría positiva y el confound K3 como lección de método |
| **3-primario (inverso constructivo)** | *Computers & Industrial Engineering* o *IJPR* | *EJOR* (si el diseño formal domina) | pre-registro + hechos validados por Garrido | especulativo si Garrido no responde → mitigar con Modo-B cierre-certificado |
| **3-fallback (constructo/métrica)** | *International Journal of Production Economics* o *Journal of Simulation* | *SMPT* | validez de constructo + reversión de ranking | percepción de nicho → anclar en política real (worst-CSSU) |

**Orden de sumisión:** Artículo 2 primero (no depende de nadie). En paralelo, reunión Garrido para fijar el modo del Artículo 3. Horizonte realista 16 meses: Artículo 2 sometido en ~4 meses; Artículo 3 (Modo A o B) sometido en ~10–12 meses.

---

## 16. Estrategia de autoría y preguntas de validación a Garrido

`COMMITTEE-INFERENCE`.
- **Propiedad:** Thomas Chisica como primer autor de 2 y 3 (diseño, implementación, ejecución, redacción). Garrido como autor de dominio/validación y fuente de los hechos que gobiernan el Artículo 3. Considerar co-autor estadístico para el clustering de inferencia si el Modo A avanza.
- **Deadline de decisión:** la reunión Garrido debe ocurrir **antes** de comprometer el Modo del Artículo 3; el Artículo 2 no la necesita.

**Dos preguntas falsables exactas (determinan Artículo 3):**
1. **(R09)** ¿Los requerimientos de teatro —en especial las órdenes de demanda contingente R24— llevan un **plazo duro** tras el cual la cantidad no cubierta se **abandona permanentemente** (no se hace backorder), tiene la agencia logística **autoridad doctrinal** para triar/rechazar órdenes en el pipeline, y la distribución de plazos es **más apretada** que los tiempos de recuperación de 24–120h (R21/R23/R22)? *Si no hay plazos, son más laxos que la recuperación, o todo se hace backorder → la familia se falsa (colapsa a D1).*
2. **(R03)** Para las piernas afectadas por R22 (Op4/Op8/Op10/Op12), ¿existe una **ruta alterna físicamente distinta** (con tránsito extra y costo vehicle-hour cuantificados), es R22 **específica-de-ruta** (bloquea la primaria y deja usable la alterna) en vez de degradar toda la pierna O-D, y su recuperación está **persistentemente autocorrelada** y es larga respecto a las 24h de la pierna, de modo que el estado-de-ruta observado al despacho predice el estado-de-ruta a la llegada? *Si R22 es ~24h sin memoria, de una sola pierna, o agnóstica a ruta → la familia se falsa (buffer intercept + compra de recursos).*

Preguntas secundarias de face-validation (para el atlas, no para reabrir): existencia de un cap de almacenamiento físico finito; si algún costo de holding/obsolescencia entra en la definición operativa de ReT; si el conjunto real de 21 raciones tiene acoplamiento de kit con shocks producto-idiosincráticos.

---

## 17. Roadmap de ejecución 12–16 meses

`COMMITTEE-INFERENCE`.
- **Semana 1 (ahora):** publicar procedencia (tag+push K3, aprobación de usuario); congelar el certificado de frontera; primer commit (§18).
- **30 días:** Artículo 2 — figuras (escalera de arrows, atlas H_PI/η, traza K3), Tabla invariante×familia, congelar contrato de métrica y tests de fixture; borrador de métodos + resultados.
- **60 días:** Artículo 2 — introducción/discusión, implicaciones gerenciales, paquete de reproducibilidad (manifiestos, hashes, LFS para trazas crudas); reunión Garrido con §16.
- **90 días:** someter Artículo 2 (SMPT/JoS). Decidir Modo del Artículo 3 según Garrido.
- **6 meses:** Artículo 3 Modo A (pre-registro + contrato + gate pre-RL + barrido fresco) **o** Modo B (cierre-certificado) **o** fallback constructo/métrica; ejecutar en VPS con watcher/heartbeat/checksums.
- **12 meses:** someter Artículo 3. **16 meses:** revisiones del Artículo 2 y, si Modo A positivo, semilla del Artículo de retención.

---

## 18. Primer commit a hacer ahora

`REPOSITORY-EVIDENCED`. Ya en árbol de trabajo (esta sesión), listo para commitear en `codex/paper2-maintenance-headroom`:
- `docs/PAPER2_PAPER3_PROVENANCE_RECONCILIATION.md`
- `docs/PAPER2_PAPER3_BOUNDARY_CERTIFICATE_AND_STRATEGY_2026-07-13.md` (este archivo)
- `results/paper2_search/{failure_taxonomy,seed_burn_ledger,artifact_index,candidate_registry,voi_ceiling_atlas,boundary_certificate}.json`

Mensaje sugerido: `Paper 2/3: provenance reconciliation + BOUNDARY_CERTIFIED (12-family adversarial screen, VoI-ceiling atlas, K3 confirmed)`. **Push a remoto gated en aprobación del usuario.**

---

## 19. Ataque de Reviewer #2 (tabla)

| Ataque | Respuesta |
|---|---|
| "Un negativo no es publicable." | El objeto es positivo: una teoría de invariantes+techo VoI + gate reutilizable + auditoría de confound, no un 'no funcionó'. |
| "Se solapa con el Artículo 1." | Artículo 1 = elegibilidad/comparador sobre Track A/B. Artículo 2 = descomposición EVPI/η e invariantes sobre D–K; ancla en familias que el Artículo 1 nunca toca. |
| "El atlas es un adaptador estilizado, no full-DES." | Se etiqueta explícitamente; la firma η<0 se replica en 8 familias **full-DES** (DRA2b/E/F/G/H/J/K2/K3). El estilizado sólo extiende el mapa a una región conexa. |
| "Escogieron cómputo/horizonte para ganar." | Al revés: K3 muestra cómo una frontera estática incompleta *fabrica* un triunfo; lo auditamos y retractamos. Recursos iguales por construcción (`ordered_D0 Δ=0`). |
| "Faltan comparadores fuertes." | Se incluyen constantes, calendarios periódicos completos, (s,S)/base-stock optimizados, MPC, belief, mezclas convexas; brecha de optimalidad reportada. |
| "H_PI grande ⇒ debería ser aprendible." | Refutado empíricamente: η medio −0.79; la política observable *destruye* valor pese a EVPI material. |
| "Cambiaron la métrica." | ReT canónica (`ret_excel_visible_v1`) es primaria en todo; Cobb-Douglas sólo sensibilidad. Se documenta la reversión de ranking como hallazgo (Programa G), no como rescate. |
| "Starvation oculto." | worst-CSSU es guardarraíl simultáneo; los positivos raros del atlas se marcan como violadores de equidad (región-I). |

---

## 20. Decisión final Go/No-Go

`PI-VALIDATED`.
- **Artículo 2 (frontera VoI + atlas de invariantes): GO inmediato.** Datos existentes + atlas fresco; no requiere entrenamiento; distinto del Artículo 1; publicable bajo cualquier resultado.
- **Artículo 2 positivo (control adaptativo): NO-GO** bajo física de la tesis. Sólo reabrible por hechos de dominio (R03/R09).
- **Artículo 3 (inverso constructivo + elicitación): GO condicional** — ejecutar reunión Garrido; el resultado fija Modo A (conversión) o Modo B (cierre). Fallback autosuficiente disponible (constructo/métrica).
- **Artículo 3 (retención positiva): NO-GO** hasta que exista un control adaptativo confirmado.
- **RL/PPO adicional: NO** hasta que una política no-neuronal desplegable demuestre `H_obs > 0` fuera de muestra.

**Cadena objetivo, estado por eslabón:** autoridad física ✅ → reversión de ranking dependiente-de-estado ⚠️ (sólo trivial) → H_PI ✅ (material) → **H_obs ❌ (≈0/negativo) ← la cadena se rompe aquí** → valor aprendido ❌ → valor retenido ❌.

---

## 21. Próximos diez artefactos/commits exactos

1. `git tag program-k3-retraction-2026-07-13 ef6b53b7` + push branch (aprobación usuario) — respalda la evidencia decisiva.
2. Commit de los 6 JSON + 2 MD de esta sesión (mensaje en §18).
3. `docs/GARRIDO_ELICITATION_R03_R09_2026-07.md` — instrumento con las 2 preguntas falsables + criterios de conversión/cierre.
4. `results/paper2_search/voi_ceiling_atlas_figure.py` — genera Fig2 (H_PI vs η, umbral 0.01, línea η=1).
5. `results/paper2_search/invariant_map.json` — tabla invariante-física × mecanismo-de-fallo × familia (Tabla1).
6. `docs/manuscript_paper2/` — esqueleto Elsevier (secciones método/resultados/discusión reutilizando figuras del atlas).
7. `tests/test_voi_ceiling_atlas.py` — fixtures que fijan `η<0` y `H_PI≥0.01` en celdas semilla (regresión del hallazgo).
8. `results/paper2_search/full_des_evpi_decomposition.json` — consolidar H_PI/H_obs/η de DRA2b/E/F/G/H/J/K2 en una tabla (Tabla2).
9. `docs/PAPER3_PROTOCOL_SKELETON.md` — protocolo congelado Modo A/B con gate pre-RL y arms de retención (dependencia única: hechos de Garrido).
10. `docs/manuscript_paper3_fallback_construct_validity/` — esqueleto del fallback autosuficiente (Programa G) por si la elicitación se retrasa.

---

### Apéndice — cumplimiento del presupuesto mínimo de evidencia (§7 del encargo)

reconciliación de procedencia ✅ · taxonomía-máquina 17 familias ✅ · generación independiente 12 streams ciegos ✅ · ≥8 familias materialmente distintas ✅ (12) · ≥3 con cotas cuantitativas ✅ (pruebas de invariancia + atlas 64 celdas) · ≥2 familias top con pregunta-Garrido precisa ✅ (R03, R09) · ≥1 barrido pre-aprendiz fresco ✅ (atlas VoI) · auditoría adversarial (3 lentes/familia) ✅ · auditoría de comparadores ✅ (taxonomía + K3) · auditoría de traza-de-acción ✅ (K3 reproducido). **Retorno: `SEARCH_ENVELOPE_BOUNDARY_CERTIFIED` para el sobre declarado — no una imposibilidad universal.**
