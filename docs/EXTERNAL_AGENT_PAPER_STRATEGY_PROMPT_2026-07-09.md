# Prompt para agente externo: estrategia editorial y científica SCRES-DES-RL

Actúa como un comité combinado de tres perfiles senior:

1. editor asociado de una revista Q1 de operations research / production / supply-chain management;
2. reviewer metodológico especializado en discrete-event simulation, reinforcement learning y diseños computacionales;
3. asesor de estrategia editorial y autoría académica.

Tu trabajo es realizar una evaluación independiente desde cero. No debes asumir que el framing actual, la división en dos papers ni las revistas que nosotros imaginamos son correctas. Debes revisar los artefactos, investigar el mercado editorial vigente y recomendar la estrategia con mayor probabilidad de producir una publicación de alto impacto que esté aceptada, en revisión avanzada o muy cerca de publicación dentro de los próximos 16 meses.

Responde en español. Los títulos, research questions, hypotheses, contribution statements y borradores de texto destinados al manuscrito deben escribirse además en inglés académico listo para usar.

## Restricciones epistemológicas

- No aceptes números porque aparezcan en un documento narrativo. Contrástalos con CSV/JSON cuando sea posible.
- Distingue estrictamente: confirmatorio, screen preliminar, resultado secundario, boundary/null y claim retirado.
- La métrica primaria del proyecto es Garrido Excel ReT (`order_ret_excel_mean` o `ret_excel`, según el runner), no `order_level_ret_mean`.
- No confundas arquitectura (MLP, RNN, KAN), paradigma de aprendizaje (RL) y optimizador (backpropagation).
- No confundas adaptación closed-loop dentro de un episodio, retención entre campañas, predicción y prevención anticipatoria.
- No uses “first” o “novel” sin una búsqueda bibliográfica que lo sustente.
- No recomiendes experimentos por reflejo. Cada experimento nuevo debe cerrar una objeción concreta y tener una regla de promoción/stop.
- Si una recomendación depende de información ausente, identifícala y explica cómo cambia la decisión.

## Material que debes revisar

### Borrador y base conceptual de Garrido

- `/Users/thom/Downloads/v.0_neuralNet-scres.pdf`
- `/Users/thom/Downloads/v.0_neuralNet-scres.docx`
- `/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Supernote/Document/20_RESEARCH/PhD-Papers/garrido2024 scres+AI.pdf`
- La tesis y workbooks de Garrido disponibles o citados dentro del repositorio.

El draft de Garrido propone `R_t=f(S_t,D_t,L_{t-1})`, aprendizaje acumulativo, ANN/RNN/RL, H1–H4 y predictive accuracy. Debes decidir qué partes siguen siendo defendibles a la luz de la evidencia real y cuáles deben reformularse o trasladarse a future work.

### Manuscrito y evidencia del repositorio

Repositorio local:

- `/Users/thom/Projects/research/scres-ia`

Rama GitHub:

- `codex/garrido-replication-experiments`

Inspecciona como mínimo:

- `docs/manuscript_current/submission/elsevier/main.pdf`
- `docs/manuscript_current/submission/elsevier/sections/`
- `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`
- `docs/REVIEWER_DEFENSE_MATRIX_2026-07.md`
- `docs/PROMISING_LANES_REGISTRY.md`
- `docs/PAPER2_TRACK_BP_RESEARCH_PROGRAM_2026-07-09.md`
- `docs/TRACK_BP_GATE2_SCREEN_VERDICT_2026-07-09.md`
- `docs/TRACK_B_PREVENTION_HEADROOM_GENERALIZED_VERDICT_2026-07-08.md`
- `docs/PREVENTION_GATE_AUTOPSY_AND_CLOSURE_2026-07-07.md`
- `docs/TRACK_B_FINAL_AUDIT_PACKAGE_2026-07-06.md`
- los artefactos CSV/JSON vinculados desde esos documentos;
- `git log`, para entender la evolución, retractaciones y contribuciones reales.

También revisa visualmente el PDF, las tablas y todas las figuras. Una compilación exitosa no equivale a calidad de submission.

## Claims actuales que debes auditar, no asumir

### Paquete Paper 1

1. Track B mejora Garrido Excel ReT frente a una frontera estática downstream densa: PPO `0.005898` vs estática `0.005460`, delta `+0.000438`, CI95 seed-clustered `[+0.000421,+0.000458]`, 10/10 seeds.
2. La dirección favorable se reproduce en CVaR05, fill, backlog, service-loss AUC y colas CTj/RPj/DPj.
3. El mecanismo defendible es action-space alignment con el bottleneck downstream: Track A es boundary/null; Track B gana cuando el contrato alcanza dispatch downstream.
4. La victoria no depende de forecast privilegiado: no-forecast es equivalente a full-v7 en 15 seeds; el mask régimen+forecast conserva la mayoría de la ganancia.
5. Un lookup estático con acceso al régimen verdadero no reproduce PPO.
6. Generalización: positiva en current/increased; severe es boundary de service-floor.
7. PPO es Pareto no-dominado, pero no existe cost dominance incondicional; con cargos positivos de dispatch puede resultar más barato.
8. SAC/TD3 replican el patrón solo a escala screen; no debe presentarse como comparación algorítmica definitiva.
9. Bajo `track_b_v1`, la evidencia apoya recuperación adaptativa, no prevención causal ni anticipación.
10. Ruta B preventiva fue retractada; el gate splice era inválido.
11. Retained-vs-reset tiene un efecto pequeño, pero no sostiene todavía una teoría central de path dependency u organizational learning.

### Paquete Paper 2

1. Bajo R21 extremo (frecuencia x8, impacto x4) aparece headroom de reservas; a intensidades naturales/moderadas es esencialmente nulo.
2. Confirmatorio 5 seeds x 60k:
   - 8D sin buffers: `0.311676`;
   - 11D con buffers dinámicos: `0.340164`;
   - 8D con postura fija heterogénea: `0.340605`.
3. Postura fija menos 11D: `+0.000440`, CI95 `[−0.000799,+0.001680]`: equivalencia.
4. Postura fija menos 8D sin buffers: `+0.028928`, CI95 `[+0.016283,+0.041574]`, 5/5 seeds y 117/120 episodios positivos.
5. La postura fija calibrada es aproximadamente `(Op3=0.1531, Op5=0.2480, Op9=0.2068)` como fracción de I_1344.
6. Los controles within-checkpoint, cross-fitted y pre/post-evento muestran scheduling, anticipación y modulación semanal nulos.
7. La interpretación actual es de dos etapas: optimizar una postura estratégica lenta/fija y usar PPO 8D para recuperación adaptativa.
8. La postura fue destilada desde políticas 11D; todavía falta una frontera clásica per-operation optimizada en calibration-only para demostrar si RL es necesario como posture optimizer.
9. El top-up actual puede inyectar inventario después del lead aunque la ruta esté caída; falta una sensibilidad route-aware y costos basados en inventario realmente almacenado.
10. Real-KAN 11D iguala aproximadamente PPO 11D como sidecar, pero falta KAN 8D para una descomposición arquitectónica simétrica.

## Contexto de autoría y colaboración

Thomas ha realizado prácticamente todo el trabajo computacional y empírico del repositorio: reconstrucción del DES, auditoría de la tesis/workbooks, software, wrappers Gymnasium, contratos de acción, entrenamiento, experimentos, CRN, fronteras estáticas, análisis estadístico, figuras, documentación, retractaciones y redacción técnica.

Garrido aportó el draft conceptual, la tesis/caso de estudio, conocimiento de dominio, guía académica y discusión de la agenda de investigación. Debes proponer una distribución CRediT honesta y una estrategia de autoría. No asumas que proporcionar un draft o supervisión resuelve automáticamente el orden de autores; tampoco ignores la propiedad intelectual, procedencia de los materiales, validación de fidelidad y contribución conceptual de Garrido.

Analiza por separado:

- qué puede preparar y liderar Thomas por su cuenta;
- qué necesita validación, aprobación, información o contribución activa de Garrido;
- qué contribuciones adicionales debería asumir Garrido antes de submission;
- orden de autores razonable y alternativas según contribución futura;
- responsabilidades de corresponding author;
- CRediT preliminar para ambos;
- permisos, acknowledgements, data/code availability y conflictos potenciales que deben resolverse.

No des asesoría jurídica definitiva. Señala cuándo hace falta revisar políticas institucionales o editoriales.

## Investigación editorial obligatoria

Investiga información vigente, no datos recordados. Usa fuentes oficiales de las revistas/editoriales y, cuando corresponda, JCR, Scopus/CiteScore o SCImago, indicando fecha de consulta y diferencias entre métricas.

Evalúa revistas de, al menos, estas familias:

- supply-chain management / resilience;
- production and operations management;
- operations research / decision sciences;
- simulation / digital twins;
- AI/ML aplicado a industrial engineering.

Puedes considerar, sin quedar limitado a ellas: IJPR, IJPE, EJOR, Computers & Industrial Engineering, Decision Support Systems, Transportation Research Part E, International Journal of Production Economics, International Journal of Production Research, Journal of Simulation y Simulation Modelling Practice and Theory. No recomiendes una revista solo por su cuartil: evalúa encaje real con el contribution type.

Para cada candidata verifica:

- scope y tipos de contribución aceptados;
- Q1/Q2 y métricas actuales, especificando la fuente;
- audiencia y papers comparables recientes;
- tolerancia a estudios de benchmark profundo en una sola topología;
- expectativa de novelty algorítmica frente a novelty metodológica/empírica;
- límites de palabras/páginas, políticas de apéndices y datos/código;
- open access/APC y opciones sin APC;
- tiempos oficiales disponibles y evidencia prudente sobre first decision, revisiones y publication timeline;
- principales razones probables de desk rejection;
- probabilidad cualitativa de que el paper esté aceptado o en revisión avanzada dentro de 16 meses.

No inventes acceptance rates ni review times. Si no hay una fuente confiable, escribe “no verificado”.

## Preguntas que debes responder

### 1. Diagnóstico desde cero

- ¿Cuál es la contribución científica realmente demostrada?
- ¿Qué historia intenta contar el draft de Garrido y cuál cuenta la evidencia?
- ¿Son compatibles o hay que reformular la teoría?
- ¿La evidencia actual sostiene un paper Q1? ¿Uno o dos?
- ¿Dónde está la novelty: algoritmo, acción/contrato, benchmark, mecanismo, teoría o metodología de auditoría?

### 2. Uno versus dos papers

Compara formalmente tres estrategias:

A. Un solo paper largo que integre Track A, Track B, prevention boundary, Track B-P y path dependency.

B. Dos papers:
   - Paper 1: action-space alignment y adaptive recovery;
   - Paper 2: strategic reserve posture / contract-regime interaction / two-stage design.

C. Paper 1 ahora y Paper 2 únicamente después de completar la frontera fija y las sensibilidades físicas.

Para cada estrategia reporta:

- claridad de tesis;
- novelty;
- carga experimental restante;
- riesgo de contradicción interna;
- riesgo de salami slicing;
- revistas objetivo;
- probabilidad y plazo editorial;
- recomendación final inequívoca.

### 3. Arquitectura de cada paper recomendado

Para cada paper que recomiendes entrega:

- título provisional;
- one-sentence claim;
- research question;
- objetivo;
- hipótesis falsables;
- contribuciones numeradas;
- estructura de secciones;
- tablas/figuras imprescindibles;
- resultados que van en el cuerpo y en el apéndice;
- lenguaje que debe evitarse;
- tres posibles abstracts de 150–220 palabras: conservador, ambicioso y orientado a la revista objetivo.

### 4. Minimum publishable package

Define lo mínimo indispensable antes de submission, separado en:

- mandatory blockers;
- strongly recommended;
- optional/future work;
- experiments that should not be run.

Para cada blocker especifica artefacto, contraste, seeds/escala, métrica primaria y criterio de éxito o stop.

Evalúa expresamente si faltan:

- frontera clásica per-op para Paper 2;
- route-aware replenishment;
- actual inventory holding cost;
- generalización de horizonte/régimen;
- KAN-8D;
- SAC/TD3 confirmatorio;
- H4 retained/reset más fuerte;
- predictor separado para “predictive accuracy”.

No respondas que todo es necesario. Prioriza por valor editorial y por el plazo de 16 meses.

### 5. Estrategia de journals y submission ladder

Entrega una tabla con:

- paper;
- revista;
- nivel: aspiracional / principal realista / backup;
- fit temático;
- fit metodológico;
- principal fortaleza;
- principal riesgo de desk rejection;
- trabajo adicional requerido;
- plazo probable y nivel de confianza;
- estrategia de cascada si es rechazado.

Debes recomendar una revista primaria concreta para cada paper y una secuencia de backups. La estrategia debe balancear “apuntar muy alto” con tener el trabajo cerca de publicación dentro de 16 meses.

### 6. Autoría y plan de colaboración

Entrega:

- matriz CRediT Thomas/Garrido;
- recomendación de first author y corresponding author con justificación;
- lista exacta de decisiones que Thomas puede cerrar solo;
- lista exacta de inputs/aprobaciones que debe pedir a Garrido;
- agenda propuesta para una reunión con Garrido;
- correo breve en español presentándole la estrategia y solicitando sus contribuciones;
- plan de versionado y aprobación antes de submission.

### 7. Roadmap de 16 meses

Construye un calendario por meses con:

- cierre científico;
- escritura;
- auditoría visual/numerical;
- revisión de Garrido;
- preprint si lo recomiendas;
- submission inicial;
- ventanas de revisión y respuesta;
- fecha límite para abandonar experimentos secundarios;
- estrategia de resubmission rápida.

Incluye una versión “agresiva” y una “conservadora”.

### 8. Simulación de reviewers

Escribe:

- un probable desk-reject editorial;
- una revisión Reviewer #2 hostil pero competente;
- la mejor defensa posible basada solo en evidencia existente;
- qué objeciones no pueden responderse todavía.

## Formato obligatorio de salida

1. **Executive verdict**: máximo 500 palabras, con recomendación inequívoca.
2. **What is actually proven**: tabla de claims con supported / bounded / unsupported / retracted.
3. **One-paper vs two-paper decision matrix**.
4. **Recommended Paper 1 architecture**.
5. **Recommended Paper 2 architecture**, si procede.
6. **Journal strategy matrix**, con fuentes y fecha de consulta.
7. **Minimum publishable package**.
8. **Authorship and Garrido collaboration plan**.
9. **Sixteen-month roadmap**.
10. **Reviewer simulation**.
11. **Final action list**, ordenada por impacto/tiempo.

Al final incluye tres listas explícitas:

- `SUBMIT NOW AFTER THESE FIXES`
- `RUN ONLY IF IT CHANGES THE JOURNAL DECISION`
- `DO NOT SPEND TIME ON THIS`

La respuesta debe terminar con una recomendación decisiva, no con “depende”. Si existen dos rutas razonables, elige una como principal y presenta la segunda únicamente como contingencia.
