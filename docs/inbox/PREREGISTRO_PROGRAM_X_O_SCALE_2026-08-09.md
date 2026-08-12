# Program X / O-Scale — protocolo candidato de prima neural por amortización

**Estado:** `CANDIDATE_DESIGN_DIAGNOSTIC_NOT_EXECUTABLE_NO_SEEDS_AUTHORIZED`

**Contrato candidato:** `contracts/program_x_o_scale_amortized_control_v1.json`

**Validador:** `scripts/validate_program_x_o_scale_contract.py`

**Alcance del validador:** ocho comprobaciones deterministas de consistencia; no simula, no abre
seeds y no certifica paridad física, headroom ni prima neural. El artefacto de preflight debe
regenerarse después de versionar este contrato candidato.

**Learner:** no autorizado

## 1. Pregunta que sobrevive a los fallos

Program O-R y su réplica Q encontraron una ventaja state-dependent en ReT frente a los 65.536
calendarios open-loop. Frente al mejor controlador state-rich, las diferencias fueron −0,00159,
−0,00072 y −0,00041 y quedaron dentro de ±0,01. El veredicto compuesto de Q siguió siendo `STOP`
porque falló el guardarraíl de peor producto. Por tanto, esa evidencia demuestra valor medio de
feedback frente a open-loop, no prima de calidad ni desplegabilidad neural.

Program X pregunta prospectivamente:

> Si el mecanismo mejor respaldado de Program O se escala y un planner estructurado resulta
> operacionalmente costoso, ¿puede una política neuronal aproximarlo con calidad no inferior y una
> reducción material de latencia?

Que el espacio de calendarios sea grande no demuestra que belief-MPC sea costoso. El coste natural
del planner es un falsador previo al learner.

Se separan tres claims:

- **calidad:** la red supera al mejor controlador estructurado high-budget con la misma información;
- **amortización:** la red es no inferior a ese teacher, cumple un SLA absoluto y reduce al menos
  10× el p95 de latencia en N=4 y N=8 sobre hardware congelado;
- **generalización:** la red congelada conserva, sin reentrenamiento, la regla de calidad o la de
  amortización en familias OOD explícitamente preregistradas.

Calidad y generalización se adjudican por separado: generalizar no convierte una política no
superior en una prima de calidad, y esta versión no autoriza un claim sobre topologías no vistas.

Behavioral cloning y DAgger pueden demostrar fidelidad, generalización y amortización. No pueden,
por sí solos, demostrar superioridad sobre el teacher que genera sus etiquetas.

## 2. Mecanismo candidato y frontera de atribución

Program X conserva como candidato el cuello compartido entre productos no fungibles y escala el
número de productos. No afirma que sea la única física posible ni que represente una propiedad
publicada de Garrido.

| productos N | asignaciones semanales de tres lotes | calendarios de ocho semanas |
|---:|---:|---:|
| 2 | 4 | 65.536 — sólo misma cardinalidad que O/Q |
| 4 | 20 | 25.600.000.000 |
| 8 | 120 | 42.998.169.600.000.000 |

La acción es una composición débil entera: $a_i\ge0$ y

\[
\sum_{i=1}^{N}a_i=3.
\]

El decoder debe imponerla exactamente mediante tres asignaciones secuenciales o un decoder entero;
no se permite producir N valores independientes y arreglarlos mediante clipping o renormalización.

Los invariantes se aplican a derechos y capacidades: mismos tres lotes, capacidad productiva
cobrada, derechos de transporte, payload máximo y dotación de vehicle-hours. Las salidas cargadas,
payload, vehicle-hours y utilización **reales** son outcomes endógenos y no se fuerzan iguales.
Dentro de cada tape emparejado se conservan demanda agregada realizada, tiempos y cantidades.

De Garrido se preservan los totales agregados y ledgers aplicables. N>2, no fungibilidad, régimen
latente, warning y cualquier BOM/yield específico son extensiones investigadoras. Op2 permanece en
190.000 unidades de cada rm cada 672 h; modificarlo exigiría otra extensión declarada.

## 3. HMM, warning y orden causal

El estado latente $Z_t\in\{1,\ldots,N\}$ identifica el producto dominante. El prior es uniforme:

\[
P(Z_0=i)=1/N.
\]

$Z_0$ se sortea antes del primer warning; no existe una transición oculta previa a $t=0$.

La transición es simétrica:

\[
P(Z_{t+1}=j\mid Z_t=i)=
\begin{cases}
\rho,&j=i,\\
(1-\rho)/(N-1),&j\ne i.
\end{cases}
\]

Condicionado a $Z_t=i$, el producto dominante recibe share $s$; cada otro producto recibe
$(1-s)/(N-1)$. El factorial candidato usa
$\rho\in\{0,75;0,90\}$, $s\in\{0,75;0,90\}$ y
$q\in\{0,70;0,85\}$. La enmienda ejecutable deberá congelar si todas las combinaciones son
primarias y la familia de multiplicidad.

El warning $W_t$ usa el mismo kernel simétrico:

\[
P(W_t=j\mid Z_t=i)=
\begin{cases}
q,&j=i,\\
(1-q)/(N-1),&j\ne i.
\end{cases}
\]

Así, y sólo así, $q=1/N$ produce una señal independiente. Del mismo modo, el nulo IID de régimen
es $P(Z_{t+1}=j\mid Z_t=i)=1/N$, equivalente a $\rho=1/N$ bajo este kernel; $\rho=0$ no sería
IID.

El orden causal de cada decisión es:

1. clonar estado físico y RNG para todos los brazos emparejados;
2. en $t=0$ sortear $Z_0$; después, entrar con el $Z_t$ ya transicionado al cierre anterior;
3. emitir $W_t$ y construir observación estrictamente half-open, sin demanda contemporánea;
4. elegir y bloquear los tres lotes;
5. realizar demanda y eventos físicos;
6. actualizar el posterior y transicionar a $Z_{t+1}$ para la siguiente decisión.

En la rama primaria de amortización, planner y learner reciben el mismo ledger físico
decision-sufficient, fase pública y posterior explícito. Si se ocultan edades de backlog u otra
variable que cambie la transición futura, esa compresión define un POMDP distinto y debe tratarse
como brazo separado.

## 4. Anchor N=2 y nulos previos

Los $4^8$ calendarios sólo demuestran cardinalidad. G0 exige replay ejecutable contra Program O/Q:
paridad evento a evento, observación half-open, vector de métricas, frontier completo e identidad y
outcome de los controladores clásicos. En ese anchor el warning se fija a $q=1/N$ y el
controlador no lo consume.

Antes de entrenar deben pasar:

1. productos idénticos y fungibles, mismos costes/yields/lead times, cero setup y sustitución
   completa: `H_PI=H_obs=0` exacto en el endpoint físico agregado;
2. régimen IID: retained no supera reset fuera de la tolerancia Monte Carlo congelada;
3. warning simétrico con $q=1/N$: warning-aware no supera warning-blind;
4. toda acción conserva exactamente los tres lotes y la masa;
5. toda acción es live en las celdas primarias no fungibles.

El último requisito no se aplica al nulo fungible: allí el colapso de acciones es el resultado
correcto.

## 5. Identificación de H4

H4 interviene sólo el conocimiento. Al inicio de cada ciclo, retained, reset, delayed y shuffled
reciben copias byte-identical de inventario, backlog y sus edades, pipeline y estado RNG. Retained
conserva el posterior; reset lo reinicia; delayed y shuffled alteran sólo la historia causal
declarada. Sin esta clonación, un efecto puede provenir del inventario heredado y no de
(L_{t-1}).

## 6. Comparadores y roles del planner

La escalera estructurada es acumulativa:

1. constantes y frontier open-loop completo en el anchor N=2;
2. base-stock/(s,S), max-pressure con histéresis y asignación robusta;
3. min-cost flow y MILP rodante;
4. scenario-MPC y belief-DP cuando sea exacto;
5. belief-MPC high-budget;
6. planner estructurado budget-matched bajo el SLA operacional.

El **teacher high-budget** recibe la misma información y corre hasta su regla congelada de
convergencia/gap. No se lo trunca al presupuesto neural: suministra etiquetas y es la referencia de
calidad/no inferioridad. Su coste natural se mide.

El **planner budget-matched** es un comparador operacional adicional. La incertidumbre de búsqueda,
el gap certificado y la reselección de familias deben propagarse en los intervalos de `H_obs` y
neural−estructurado.

## 7. Arquitectura y gates ramificados

RNN y RL son conceptos ortogonales: recurrencia es representación de historia; RL es un paradigma
de aprendizaje secuencial. El candidato primario de amortización es:

- encoder DeepSets;
- policy **permutation-equivariant**: relabelar productos relabela sus asignaciones;
- value **permutation-invariant**;
- decoder entero con suma exacta de tres;
- MLP fijo por N, igualado en parámetros y presupuesto, como baseline.

La ruta de amortización es belief-MPC high-budget → behavioral cloning → DAgger. `H_ret` no es
requisito de esta ruta.

Los gates se ramifican después del frontier estructurado:

| rama | condiciones previas | qué autoriza |
|---|---|---|
| amortización | G0–G5 + `H_obs` + planner rompe SLA absoluto o break-even vinculante | BC/DAgger |
| recurrencia | historia tiene valor condicional sobre snapshot+belief; retained vence reset/delayed/shuffled OOS | un GRU pequeño |
| calidad residual | headroom observable same-information sobre el high-budget structured supera SESOI | RL residual acotado |

Con el HMM exacto conocido, el posterior es la estadística suficiente nula. Si historia cruda vence
al belief exacto, primero se audita misspecification, fuga o estado físico oculto; no se atribuye
automáticamente a RNN. LSTM queda como sensibilidad.

Toda acción residual usa el mismo decoder de tres lotes y se compara con residual cero, lineal y
árbol. Una GNN queda prohibida hasta existir topología/BOM variable real. Offline RL no es primario
mientras existan simulador y consultas al teacher; el soporte estrecho de logs no sustituye la
escalera. KAN queda como sidecar bajo otro contrato.

## 8. Métricas

El endpoint físico primario es:

\[
V_{service}=1-\frac{\sum_k \Delta t_k(1-SL_k)}{H},
\]

donde $SL_k\in[0,1]$, $\sum_k\Delta t_k=H$, el resultado está en [0,1] y mayor es mejor. La
implementación exacta de `service_level_k` deberá congelarse desde el ledger causal antes de abrir
tapes. Con esta convención todos los estimandos `V(A)-V(B)` tienen signo positivo=favorable.

ReT y Cobb-Douglas son secundarios prespecificados y se reportan siempre, pero no pueden promover,
rescatar ni bloquear el claim físico primario:

- ReT `ret_excel_request_snapshot_v2` y full-ledger, exactos, sin clipping y con flags fuera de
  [0,1];
- Cobb-Douglas fuente con mapeo MFSC, pisos logarítmicos y referencia de coste congelados;
- recalibración O-Scale separada y etiquetada como métrica nueva.

No se elige entre ambos después de observar resultados. Cualquier contraste inferencial secundario
usa una familia de multiplicidad congelada; la discrepancia entre constructos se reporta.

También se informan fill medio/peor producto —con convención congelada para productos sin demanda—,
pedidos perdidos/no resueltos, backlog/edad, TTR, varianza, costes, conservación, utilización,
latencia p50/p95, llamadas DES, memoria y coste offline.

## 9. Claims y umbrales todavía no autorizados

| claim | regla candidata |
|---|---|
| feedback observable | LCB95 simultáneo de `H_obs` > 0; no implica prima neural |
| calidad estadística | LCB95 simultáneo de neural−high-budget structured > 0 |
| calidad material | además supera un SESOI justificado y congelado antes de tapes |
| amortización | no inferioridad + SLA absoluto + p95 al menos 10× menor en N=4 y N=8 |
| desplegable | además pasa peor-producto, cola, costes y recursos |

El SESOI, margen de no inferioridad, SLA en segundos y query-count de break-even están pendientes de
justificación de dominio y power. No se conservan automáticamente los umbrales 0,01/0,005 de otras
métricas o programas.

Las llamadas DES son diagnóstico: no sustituyen el SLA ni el gate de p95. Se congelan hardware,
warm-up, caching, belief-update y protocolo de medición. Costes de entrenamiento, generación de
expert data y compilación entran al break-even.

Calidad y generalización por amortización se adjudican por separado. Esta versión no permite un
claim genérico sobre topologías no vistas.

## 10. Orden de ejecución

Gates compartidos:

1. G0: paridad ejecutable N=2; cardinalidad no basta;
2. G1: conservación y equivalencia de derechos/capacidades;
3. G2: liveness en primarias y nulos scoped;
4. G3: `H_PI` material;
5. G4: `H_obs` material con incertidumbre de optimización;
6. G5: frontier estructurado y replacements completos.

Después se abre únicamente la rama cuyos gates específicos pasen. La falta de `H_ret` bloquea GRU,
no amortización. La falta de residual de calidad bloquea RL residual, no BC/DAgger. Si el planner no
incumple SLA ni presenta break-even vinculante, también se cierra amortización aunque el espacio de
calendarios sea enorme.

## 11. Custodia y estado actual

El registro vigente declara
`BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`. No basta una decisión informal del PI y no
se asignó ningún rango a Program X.

Antes de abrir una sola tape se requiere:

1. completar el inventario de custodia y obtener autorización PI;
2. versionar y hashear el kernel de paridad N=2 y métricas exactas;
3. congelar factorial, multiplicidad, SESOI, power y márgenes;
4. congelar SLA, hardware y budget de cómputo;
5. registrar namespaces de desarrollo/evaluación/confirmación y dependencias.

Hasta entonces sólo se permiten documentación, cardinalidad combinatoria, construcción del runner y
tests sintéticos deterministas. Las ocho comprobaciones del validador son **consistency checks** de
este diseño; ninguna abre un gate científico.
