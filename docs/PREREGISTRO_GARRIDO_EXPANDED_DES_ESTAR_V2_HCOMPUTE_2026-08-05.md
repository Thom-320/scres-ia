# Preregistro de diseño — Garrido expanded DES / E* v2

**Estado:** `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`

**Contrato:** `contracts/garrido_expanded_des_e_star_v2_hcompute.json`

**Semillas nuevas:** ninguna abierta

**Learner:** no autorizado

## Decisión científica

E* no se construye para seguir modificando el entorno hasta encontrar una victoria neural. La
pregunta es qué derechos de decisión físicamente válidos crean valor contingente, cuánto captura la
frontera estructurada y si queda una prima legítima de calidad, eficiencia o generalización.

La prioridad es **H_compute**: una política neural sólo puede amortizar un planificador que sea caro
en el problema ampliado. Si el MPC sigue siendo barato, el carril de amortización termina sin abrir
un learner.

El endpoint principal recomendado para el estudio C&IE es `ret_excel_request_snapshot_v2`. La elección
formal queda pendiente de la firma PI/Garrido antes de datos frescos. Cobb–Douglas se reportará como
sensibilidad calibrada. CVaR (`ret_excel_cvar05`, `ret_excel_cvar10`) será diagnóstico secundario de
cola: no será primario, no promoverá y no bloqueará por sí solo.

## Kernel único

Se implementará un solo `E_star` con:

- procurement de fuentes aprobadas;
- nodos `wdc`, `al`, `sb`, `cssu_a` y `cssu_b`;
- buffers y capacidades finitas;
- lead times e inventario en tránsito;
- transporte y dispatch downstream;
- producción, replenishment, servicio y recursos conservados;
- decisiones continuas y selección sobre un conjunto fijo de proveedores/rutas;
- horizonte y cadencias nativas, sin alargar artificialmente el episodio.

Una acción no puede crear inventario. Toda transferencia cumple:

```text
q_move <= shortfall
q_move <= upstream_available
q_move <= transport_capacity
```

El excedente de un buffer lleno permanece upstream. Las ocho máscaras `M000`–`M111` cambian sólo
los derechos de decisión; no introducen física nueva. Las dimensiones apagadas se arrastran y las
claves desconocidas fallan cerrado.

## Bridge y custodia

Antes de cualquier benchmark científico deben pasar:

1. bridge flags-off de `M000` contra el DES Garrido existente;
2. conservación de masa, WIP, capacidad, procurement y recursos;
3. ausencia de futuro en la observación;
4. liveness de cada acción nueva;
5. mutantes que prueben que cada falsador puede fallar;
6. manifiesto de módulos, comando, rol, replay, hardware y hashes del payload.

El runner no podrá declarar `H_compute` si el bridge está ausente o no verificable. El estado será
`STOP_ESTAR_DES_BRIDGE_NOT_READY`.

## Preflight H_compute

Sólo se usarán tapes burned y fixtures deterministas. El horizonte se mantiene fijo. La complejidad
se varía mediante esta escalera congelada:

```text
S0 = M000
S1 = procurement
S2 = buffers upstream
S3 = procurement + buffers upstream
S4 = M111 con downstream y dispatch
```

Dentro del conjunto aprobado se probarán las cardinalidades activas y permutaciones predeclaradas.
Se compararán regla estática, lookup/order-up-to, threshold/hysteresis, DP/rollout y MPC directo.

Cada nivel registrará latencia p50/p95 fría y caliente, llamadas al DES o kernel, iteraciones del
solver, memoria y tiempo total por episodio. Se separarán calentamiento y medición; no se usarán
promedios que oculten un p95 operativo.

El gate pasa si el MPC supera el presupuesto firmado de latencia o llamadas y la misma cantidad
aumenta en dos niveles consecutivos. Como referencia inicial se propone p95 ≥ 10 % de la cadencia
nativa o ≥ 10× las llamadas de `M000`; la cifra final debe quedar congelada antes de leer resultados.

Si no pasa:

```text
STOP_ESTAR_PLANNER_NOT_BINDING
```

No se abren semillas frescas ni se entrena.

## Frontera estructurada y learners

Después del recibo de Submission A o supersession explícita, se podrá abrir un bloque virgen para
la frontera estructurada:

```text
constante
→ lookup/order-up-to
→ threshold/hysteresis
→ árbol/tabular
→ spline/GAM
→ DP/rollout
→ MPC directo
```

La prima de amortización tiene un gate distinto de la prima de calidad. Si H_compute pasa, se puede
desarrollar una política supervisada que sustituya al MPC caro, siempre que iguale calidad y
guardarraíles. No necesita superar al MPC en calidad; debe reducir materialmente latencia o llamadas
sin invocar al planner durante la inferencia.

MLP y KAN se comparan en desarrollo por parámetros, tiempo de entrenamiento, convergencia, calidad,
latencia y llamadas. Sólo una arquitectura queda congelada para confirmación. DeepSets/attention
sólo se autorizan si el conjunto activo y la invariancia de permutación están en el contrato.

La prima de calidad requiere:

```text
LCB95(Delta_obs) >= SESOI
```

La generalización se prueba únicamente en subconjuntos, permutaciones, regímenes y lead times
reservados antes de observar resultados.

## Cierre

Una candidata positiva de desarrollo congela kernel, observaciones, acción, endpoint, arquitectura,
hiperparámetros, comparador, presupuesto y análisis. Sólo una confirmación independiente en tapes y
seeds vírgenes permite declarar `NEURAL_PREMIUM_CONFIRMED`.

Program Q, Program O y `thesis_1to1` permanecen cerrados e inmutables durante todo el proceso.

## Recibo de ingeniería burned-only (2026-08-05)

El bridge source-conserving quedó implementado y pasó el smoke de las ocho
 máscaras sobre un tape burned. El recibo está en
 `results/estar_expanded_bridge_smoke_v1/result.json`; su digest observado es
 `feaef05c0f31e9f82091d063b004b45823694341b7dc6225d4f4341ff37fc206`.

El preflight de coste usa ahora el DES histórico a través de `EStarDESAdapter`
 y el backend explícito `DirectDESMPC`, no el kernel sintético aislado. El gate
 de llamadas pasa en el fixture burned y el resultado es
 `H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE`. Este estado no es autoridad
 científica: no abrió semillas, no entrenó learners y no autoriza la frontera
 estructurada ni una confirmación. La selección de endpoint, SESOI, márgenes y
 autoridad de Submission A/Garrido siguen pendientes.
