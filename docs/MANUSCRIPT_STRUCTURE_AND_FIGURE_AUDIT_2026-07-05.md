# Auditoría de estructura, secciones y figuras para el paper Q1 — 2026-07-05

## Veredicto corto

El manuscrito ya tiene una estructura científica razonable y varias figuras fuertes de resultados, pero
está desbalanceado en un punto visible frente a papers competidores de RL/MARL: **la arquitectura del
agente está sub-explicada**.

En `docs/manuscript_current/submission/elsevier/sections/03_methodology.tex`, PPO aparece casi como una
oración técnica dentro de metodología, mientras que los papers comparables suelen dedicar una subsección
propia a:

1. formulación RL/POMDP,
2. entorno/simulador,
3. arquitectura del agente,
4. entrenamiento,
5. baselines,
6. métricas,
7. reproducibilidad/experimentos.

Nuestra metodología cubre bien DES, acción, observación, recompensa, métricas y evidencia; le falta una
subsección visual y textual de arquitectura.

## Fuentes revisadas

### Locales

Downloads:

- `garrido et al 2024 factory resilience.pdf`: paper IJPR de Garrido et al. sobre resiliencia de fábrica.
- `j_ijpe_2020_107655 -- ... .pdf`: Dixit et al. 2020, IJPE, CVaR y resiliencia estructural.
- `v.0_neuralNet-scres.pdf`: borrador propio / Garrido-David.
- `main.pdf`: borrador propio.
- `2510.04871v1.pdf`, `Yang_etal_EcoMod_FINAL(1).pdf`, `RALU_CAR_PAG.pdf`: no son Ding ni competidores directos del paper Q1.

Manuscrito actual:

- `docs/manuscript_current/submission/elsevier/sections/01_introduction.tex`
- `docs/manuscript_current/submission/elsevier/sections/02_related_work.tex`
- `docs/manuscript_current/submission/elsevier/sections/03_methodology.tex`
- `docs/manuscript_current/submission/elsevier/sections/04_results.tex`
- `docs/manuscript_current/submission/elsevier/sections/05_discussion.tex`

Bibliografía:

- `docs/manuscript_current/submission/elsevier/references.bib`

### Web / literatura externa

- Ding et al. 2026, IJPE: `Multi-agent reinforcement learning-based resilience reconfiguration approach of supply chain system-of-systems under disruption risks`.
- Hazenberg et al. 2025, arXiv: `Multi-Agent Reinforcement Learning for Dynamic Pricing in Supply Chains`.
- Boute et al. 2022, EJOR: `Deep reinforcement learning for inventory control: A roadmap`.
- Stranieri et al. 2025, arXiv: `Classical and Deep Reinforcement Learning Inventory Control Policies for Pharmaceutical Supply Chains...`.

## Qué hacen los papers comparables

### 1. Papers SCRES / simulación clásica

Garrido 2024 y Dixit 2020 no son papers RL, pero muestran un estándar importante:

- justifican el constructo de resiliencia antes de entrar al modelo;
- dedican figuras conceptuales al flujo de resiliencia;
- separan formulación/modelo, metodología, análisis y conclusiones;
- usan tablas de literatura para posicionar el gap;
- muestran métricas con figuras de sensibilidad, distribución o escenarios.

Lección para nosotros: mantener fuerte el linaje Garrido/ReT/DES y no convertir el paper en "solo PPO".

### 2. Ding et al. 2026 / IJPE

Ding es el competidor más cercano por tema: MARL + resiliencia + reconfiguración de supply-chain
system-of-systems. Lo que se puede verificar desde la ficha institucional y el abstract:

- modelan reconfiguración de suppliers/manufacturers/distributors/consumers;
- introducen estrategias de resiliencia: filling, repairing, recruiting;
- formulan el problema como POMDP;
- balancean resiliencia y costo en la recompensa;
- usan MAPPO;
- comparan contra otros baselines de MARL;
- discuten atributos que afectan la reconfiguración.

Lección para nosotros: necesitamos que el lector vea explícitamente nuestra formulación RL y nuestra
arquitectura, aunque nuestra contribución sea distinta: operational-control benchmark Garrido-grounded,
no reconfiguración estratégica de red.

### 3. Papers MARL/RL con diagramas de arquitectura

El paper arXiv 2507.02698 tiene una organización muy explícita:

- Related Work dividido por tópicos.
- Methodology con dataset/preprocesamiento/modelo predictivo/simulador/agentes/baselines/setup/reproducibilidad.
- Resultados por familia de agente.
- Discusión con comparación contra estado del arte, interpretación y limitaciones.
- Apéndice con diagramas de arquitectura MADDPG/MADQN/QMIX.

Lo más útil para nosotros: sus diagramas de agente son de flujo completo, no solo "caja de red":

`state/features -> actor/critic network -> action -> environment -> reward -> learning update`

con replay/target networks cuando aplica. En nuestro caso, el análogo PPO debería mostrar:

`DES observation -> shared MLP/KAN extractor -> actor/value heads -> 8D action -> DES step -> control_v1 reward -> GAE/PPO clipped update`

## Auditoría del manuscrito actual

### Lo que ya está bien

El manuscrito ya tiene una secuencia sólida:

1. `01_introduction`: framing Track A/Track B y Ding como competidor.
2. `02_related_work`: SCRES, Garrido DES, RL supply chain, DES+ML/AI-SCRES, gap de bottleneck authority.
3. `03_methodology`: DES, riesgos, POMDP/control, action contract, observation audit, reward/metrics, evidence map.
4. `04_results`: Track A negative, Track B positive, Pareto/cost, ablations, privileged observation defense, generalization.
5. `05_discussion`: interpretación, Ding, límites.

Las figuras actuales son fuertes en:

- framing bottleneck: `fig1_bottleneck_alignment`;
- topología DES: `fig2_mfsc_topology`;
- gap decomposition: `fig3_gap_decomposition`;
- Pareto/tail: `fig4_pareto_ret_tail_ctj`;
- generalización: `fig5_generalization_heatmap`;
- action-space ablation: `fig6_action_space_ablation`;
- lineage ReT: `fig7_ret_metric_lineage`;
- timeline ReT branch: `fig8_ret_branch_timeline`.

### Lo que falta

Falta una figura de arquitectura y una subsección de "Learning architecture".

Ahora mismo, la arquitectura de PPO+MLP está descrita demasiado breve para un paper que quiere defender una contribución con red neuronal. Real-KAN/CAM tampoco está incorporado al cuerpo de metodología, aunque ya existe como sidecar confirmado e interpretable.

Esto abre dos riesgos:

1. Garrido puede sentir que PPO+MLP no tiene suficiente "novedad neural".
2. Un reviewer de RL puede sentir que el agente se trata como caja negra sin describir actor, crítico, update y policy.

## Recomendación de estructura final

Mantendría las seis secciones principales, pero reorganizaría `03_methodology` así:

### 3 Methodology

3.1 MFSC discrete-event simulation  
Ya existe; mantener.

3.2 Control problem and decision contracts  
Fusionar/ordenar POMDP, Track A, Track B, acción 8D, no-forecast.

3.3 Observation design and information boundaries  
Explicar `v7`, no-forecast, no-regime+forecast audit. Esta sección debe decir explícitamente:
la versión más conservadora no usa forecast explícito.

3.4 Learning architectures  
**Nueva.** Debe describir:

- PPO+MLP spine:
  - observación normalizada;
  - extractor MLP 64x64;
  - actor gaussiano/tanh o action mapping;
  - value network;
  - GAE;
  - PPO clipped objective;
  - acción 8D decodificada en Op3/Op9/Op5/shift/Op10/Op12.
- Real-KAN/CAM sidecar:
  - reemplaza extractor MLP por `kan.KAN`;
  - splines/edge functions aprendibles;
  - mismo PPO, misma acción, misma métrica;
  - valor como arquitectura interpretable, no como spine operacional.

3.5 Training reward and reporting metrics  
Separar claramente `control_v1` de ReT Excel.

3.6 Baselines and common-random-number evaluation  
Explicar densa 147, heurísticas, CRN, seed clustering.

3.7 Experimental design and evidence map  
La tabla existente de evidencia puede quedar aquí.

## Figuras recomendadas

### Figura obligatoria nueva: `fig9_learning_architecture`

Una figura de dos paneles.

Panel A: PPO+MLP spine.

Flujo:

1. DES state / observation vector (`v7 no-forecast` recomendado)
2. normalization
3. shared MLP 64x64
4. actor head -> 8D Gaussian action -> action decoder
5. value head -> GAE
6. DES transition + `control_v1` reward
7. PPO clipped update

Panel B: Real-KAN/CAM sidecar.

Mismo flujo, pero reemplazando `MLP 64x64` por:

`KAN extractor: learnable spline edge functions -> latent features -> actor/value heads`

Agregar una mini-inset opcional:

`rations_theatre_norm -> spline threshold-ramp`

para conectar con la interpretabilidad real que ya tenemos.

### Figura recomendable: `fig10_real_kan_interpretability`

Puede ir en discusión o apéndice.

Contenido:

- barra top-10 variables atribuidas;
- spline de `rations_theatre_norm`;
- nota: "Real-KAN learns interpretable nonlinear thresholds but at higher resource cost."

Esta figura responde directamente a Garrido: "KAN/CAM aporta novedad e interpretabilidad".

### Figura opcional: `fig11_no_forecast_defense`

Si hay espacio, una figura pequeña de barras:

- full-v7 ReT Excel
- no-forecast ReT Excel
- no-regime+forecast ablation
- best dense static

Lectura: forecast no es necesario para la ventaja principal.

## Qué NO agregaría

- No agregaría DKANA como resultado principal: sigue siendo BC/offline o diferido, no una política RL end-to-end confirmada.
- No agregaría prevención causal como claim: los contrafactuales y oracle ceiling fueron negativos.
- No agregaría una figura grande de todos los sidecars; confunde el spine.
- No haría del KAN el headline salvo que se escale fixed-RNG a 10-15 seeds y se acepte explícitamente el costo alto.

## Cómo escribir el texto sobre PPO y Real-KAN

Texto sugerido para metodología:

> The canonical learned controller uses PPO with a shared two-layer MLP feature extractor, followed by separate policy and value heads. The policy head parameterizes a continuous eight-dimensional action, later decoded into order-quantity multipliers, reorder-point multipliers, shift selection, and downstream dispatch multipliers. PPO is used here as a stable continuous-control baseline rather than as the paper's novelty claim; the experimental claim is whether the learned closed-loop controller can exploit a bottleneck-aligned decision surface under dense static baselines.

Texto sugerido para Real-KAN/CAM:

> To address architectural interpretability, we also evaluate a Real-KAN sidecar that replaces the MLP feature extractor with an official pyKAN Kolmogorov-Arnold network. The PPO optimization loop, action contract, reward, and evaluation protocol remain unchanged. This sidecar is not used as the main efficient spine because it buys a small ReT improvement at materially higher resource use; however, it provides an interpretable neural architecture whose learned spline functions can be inspected post hoc.

## Prioridad de implementación

1. **Alta:** crear `fig9_learning_architecture` y añadir subsección 3.4.
2. **Alta:** mover no-forecast al texto de observación como spine conservador.
3. **Media:** añadir Real-KAN/CAM como arquitectura sidecar en metodología o discusión, con una figura pequeña de interpretabilidad.
4. **Media:** actualizar `05_discussion` para comparar más explícitamente contra Ding:
   - Ding: MAPPO, reconfiguración estratégica, network/SoS, filling/repairing/recruiting.
   - Nosotros: PPO/Real-KAN sidecar, control operacional, MFSC DES Garrido-grounded, dense static frontier, ReT Excel, no-forecast defense.
5. **Baja:** apéndice con pseudocódigo PPO / training loop si el journal permite espacio.

## Resumen para Garrido

> "El paper ya tiene buenos resultados, pero visualmente le falta una figura de arquitectura del agente. Los papers competidores sí muestran cómo fluye la información por la red y cómo se actualiza el agente. Yo añadiría una figura doble: PPO+MLP como spine eficiente y Real-KAN/CAM como sidecar interpretable. Así protegemos dos flancos: el resultado principal sigue siendo sólido y conservador, pero también mostramos una arquitectura novedosa e inspeccionable, que es justo la preocupación de novelty."

