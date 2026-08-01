# Programa de prima neural — estado canónico 2026-08-01

Este documento reemplaza la narrativa anterior de “seguir encendiendo generadores hasta que una
red gane”. La pregunta vigente es:

> ¿Bajo qué mecanismos físicos predeclarados queda valor contingente observable después de
> saturar los controles estructurados, y existe un residual material para una política neural?

Una victoria de desarrollo no es una prima confirmada. La primera candidata positiva congela una
única confirmación independiente. Si falla, sólo se recorren las extensiones ya declaradas; no se
añaden física, métrica o arquitectura de rescate.

## Estado editorial y gobernanza

- El artículo activo es Garrido–WRAP/v0 para *Computers & Industrial Engineering*.
- Program Q y Program O permanecen cerrados, inmutables y separados del manuscrito WRAP.
- `thesis_1to1` permanece congelado.
- `Authority Ladder V1` sigue siendo la frontera operativa hasta el recibo de Submission A:
  no se abren seeds científicas nuevas ni se entrenan learners.
- El preregistro operativo de G3a es
  [`docs/PREREGISTRO_G3_ASIMETRIA_V2_2026-08-01.md`](docs/PREREGISTRO_G3_ASIMETRIA_V2_2026-08-01.md),
  con contrato en
  [`contracts/g3a_asymmetric_claimants_v2.json`](contracts/g3a_asymmetric_claimants_v2.json).
- La custodia central de semillas está en
  [`research/seed_custody_registry.json`](research/seed_custody_registry.json)
  y permanece fail-closed.

## Qué está medido y qué no puede afirmarse

### WRAP/Q1

El panel `rho → ReT` no muestra una prima neural material: el modelo lineal ya explica la
superficie dentro del protocolo autorizado y las redes no superan el SESOI correspondiente.
La identidad de la Figura 5 no se presenta como aprendizaje.

G1 sí encontró un mecanismo de curvatura bajo Cobb–Douglas: el máximo de buffer es estrictamente
interior en dos celdas y desaparece al quitar el coste de mantenimiento. Su estado correcto es
`CAUSAL_MECHANISM_SUPPORTED_DEVELOPMENT`, no causalidad ya confirmada para publicación. La
ablación usa un contraste pareado que debe declararse como tal antes de elevar el claim.

La corrida de predicción sobre la superficie curva concluyó `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED`
como resultado de desarrollo ampliado: el margen disponible estimado cruza el cero y el spline
gana a las redes. No es una cota matemática ni una confirmación preregistrada. El objetivo Cobb–
Douglas se calibra dentro de cada fold; entre folds no es una etiqueta fija.

### WRAP/G2

G2 ya está medido bajo su contrato. La discontinuidad de autotomía aparece en el brazo FDB, pero
la regla de umbral y las redes no superan al baseline lineal con interacciones; el margen de media
de celda es negativo y la variación intra-celda domina. El resultado es
`THRESHOLD_RULE_SUFFICES` dentro del screen de predicción, no una prima de control ni una prueba
universal sobre todos los umbrales.

### CSSU

El Gate A demuestra que la acción de reasignación en `split_v1` es computacionalmente viva,
respeta latencia y conserva masa. El Gate B de manejo físico finito de Op11 sigue en `HOLD`:
no se afirma una física de manejo que carezca de contrato.

La contención y la expedición no abren headroom bajo los endpoints sanos ejecutados. El resultado
queda acotado a esos contratos y no cierra por sí solo temporalidad, N=3 o toda la familia G3.

### Q2 y H3′

El artefacto DES-288 existe, pero permanece
`ARTIFACT_PRESENT_CANONICAL_CUSTODY_PENDING`. No se usa ninguna cifra hasta reconciliar contrato,
hash, seeds, falsadores y claim ledger.

Las dos rebanadas H3′ tienen artefactos y semillas disjuntas, pero el merge permanece
`ARTIFACTS_PRESENT_MERGE_PENDING`: falta cerrar el manifiesto VPS, la equivalencia del DES, los
falsadores y la custodia. No se interpretan ni combinan.

### Program Q

Program Q mostró valor state-dependent frente a la frontera completa open-loop y equivalencia
práctica con la mejor familia estructurada probada. No demostró una prima neural, seguridad por
peor producto ni superioridad sobre un belief-MPC específico. La formulación autorizada es
“mejor familia estructurada probada”. Q no es un banco de rescate para WRAP.

## G3a: primer screen prospectivo

G3a conserva dos reclamantes y la interfaz histórica. Su screen inicial usa únicamente
`allocation_a ∈ {0.25, 0.50, 0.75}`. El factorial es:

| factor | niveles |
|---|---|
| demanda | 50/50, 70/30, 30/70 |
| riesgo | neutral, localizado parcialmente en A, localizado parcialmente en B |
| recurso | no fungible; pooling verdaderamente action-invariant |

El riesgo no puede convertir la acción en moot destruyendo por completo una CSSU. Los regímenes
se mezclan dentro del tape y la política no recibe `cell_id`, régimen verdadero ni futuro.

`weights=None` conserva exactamente la ruta SHA histórica. Los brazos ponderados usan uniformes
event-keyed de 64 bits y un tape exógeno con onset, duración, impacto y target. `reallocate_unused`
no se llama pooling completo: sólo reasigna sobrantes y puede dejar la acción vinculante.

Los falsadores obligatorios incluyen:

- realización de pesos por pedidos y cantidad;
- liveness en ambos sentidos;
- equivarianza A↔B;
- invariancia por acción del pooling completo;
- identidad de tiempos e impactos entre brazos;
- placebos shuffled, delayed y wrong-claimant;
- detección mutacional de pesos ignorados, futuro expuesto y paso por ruta caída;
- hashes separados de payload científico y provenance.

## Estimandos y gates

Se separan:

```text
H_PI          valor con régimen verdadero; sólo diagnóstico privilegiado
H_obs         valor de política cross-fitted, no anticipativa y observable
H_structured  valor del mejor control estructurado
H_residual    techo observable menos H_structured
Delta_N       valor neural menos BestStructured en rollout closed-loop
```

El endpoint primario escalar es `worst_claimant_fill`. Fill agregado, lost, unresolved, backlog,
service-loss, recursos y ReT full-ledger son guardarraíles o diagnósticos. `service_first_v2` puede
seleccionar candidatos, pero no recibe un intervalo de confianza como si su tupla lexicográfica
fuera una variable cardinal.

Los estados terminales de G3a son:

```text
STOP_G3A_N2_NO_MATERIAL_PHYSICAL_HEADROOM
STOP_G3A_HEADROOM_NOT_DEPLOYABLY_OBSERVABLE
G3A_STRUCTURED_CONTROL_SUFFICES
AUTHORIZE_NEURAL_DEVELOPMENT_G3A
NEURAL_PREMIUM_CONFIRMED_IN_G3A_CONTRACT
BUDGET_EXHAUSTED_WITHOUT_PREMIUM
```

Sólo `AUTHORIZE_NEURAL_DEVELOPMENT_G3A` abre entrenamiento de desarrollo. Sólo una confirmación
independiente puede producir `NEURAL_PREMIUM_CONFIRMED_IN_G3A_CONTRACT`.

## Orden después de G3a

Si G3a no tiene headroom físico, se cierra sólo el contrato N=2 probado. Si tiene headroom
observable pero lo captura una regla, el resultado es saturación estructurada y no se entrena una
red para redescubrirla.

Si queda residual observable y la pregunta siguiente es secuencial, se mantiene N=2 y se añade un
solo mecanismo temporal: switching cost, minimum dwell, setup, ramp, posición persistente o
tiempo de retorno. N=3 sólo se abre con una hipótesis independiente de interacción triple,
matching, kit, coaliciones, capacidad indivisible o número variable de reclamantes.

O→Q+ y Q-direct se fusionan en una sola familia prospectiva nueva; no existe `O+0` como réplica
de Program O. La observabilidad parcial es una extensión posterior y debe demostrar estados
aliasados, historias que los distinguen, mejora out-of-sample por historial y un comparador
belief-DP/MPC bajo los mismos derechos.

## Ladder de control y aprendizaje

Los comparadores se prueban antes de cualquier learner:

```text
constante → lookup → threshold/hysteresis → árbol/tabular
→ spline/GAM → DP/rollout → MPC → belief-MPC si corresponde
```

MLP, KAN, DeepSets y atención son representaciones; PPO es un algoritmo; RecurrentPPO añade
memoria. No se presentan como peldaños de una misma escalera. `Delta R²` queda fuera del gate
principal de control y sólo puede formar un contrato predictivo separado.

Una eventual prima también puede definirse por eficiencia —misma calidad con menor latencia,
menos llamadas al DES o mejor generalización—, pero esa pregunta se preregistra por separado de
la prima de calidad.

## Conclusión permitida

Si se agotan G3a, temporalidad y la familia Q+ sin confirmación, el claim será únicamente:

> No se detectó una prima neural material bajo los contratos prospectivos, familias comparadoras,
> presupuestos y márgenes predeclarados.

No se afirmará que la prima neural es imposible ni que un fallo N=2 refuta N=3. La contribución
puede ser precisamente localizar cuándo el valor contingente es absorbido por una regla, un DP o
un controlador estructurado, y qué contrato físico faltaría para que una arquitectura neural
fuera una hipótesis razonable.
