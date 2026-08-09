# Respuesta a Garrido y ruta falsable hacia una prima neuronal

**Fecha:** 2026-08-08
**Estado:** `PROGRAM_V_MECHANISM_IDENTIFIED__NEURAL_PREMIUM_NOT_YET_CLAIMED`
**Resultado sellado:** `results/program_v/prelearner_gate_v1/result.json`

## Respuesta corta

Sí hay un entorno justificable donde el aprendizaje retenido importa: comprometer capacidad limitada
entre proveedores aguas arriba bajo degradaciones persistentes, información parcial, entregas
retardadas y demanda estacional. Program V identifica ese mecanismo antes de entrenar una red.

No obstante, la primera celda también falsifica una conclusión más ambiciosa: el filtro Bayesiano
absorbe prácticamente todo el headroom de calidad. El resultado
no es una «prima neural». Autoriza construir un planner combinatorio en el DES completo y preguntar
si MLP o KAN pueden amortizarlo sin perder calidad. Calidad y amortización serán estimandos
separados.

## Qué se toma de Garrido y qué se añade

Garrido-Rios (2017) modela la red militar y sus procesos de proveedores, procurement, producción,
WDC, SB y CSSU. También deja límites explícitos que impiden estudiar la decisión propuesta:

- disponibilidad de vehículos y planificación de rutas dadas, y capacidad de WDC/SB/CSSB ilimitada
  (PDF p. 98; página impresa 97);
- colocación instantánea y ausencia de entregas parciales o cambios de orden (PDF pp. 98–99);
- parámetros estacionarios, incluidos hasta veinte años de demanda (PDF p. 99);
- coste excluido y lead time recomendado como extensión (PDF pp. 147–149).

Program V conserva la lógica de red y de shocks recurrentes, pero elimina de forma declarada esas
simplificaciones: tres proveedores sustituibles, lead de compromiso, yields parciales, presupuesto
fijo, demanda estacional y estado latente persistente. Ninguno de esos valores numéricos se atribuye
a la tesis; deberán calibrarse o someterse a sensibilidad antes de una afirmación externa.

La reconstrucción DES más reciente de G3a (`results/g3a_boundary_v2/result.full34.json`) descarta
usar de nuevo la asignación A/B entre CSSU como vía a una prima: con 34 controladores, la celda
persistente/uniforme de cuota rígida da `H_obs=+0,002789` e IC95 que cruza cero, y el mejor placebo
no pierde. Program V no rescata esa celda cambiando el learner; introduce un derecho de decisión
distinto aguas arriba.

Garrido et al. (2024) formula dos preguntas: qué categoría de AI imita mejor Supply Chain Learning y
cómo integrarla dentro de DES (PDF p. 2; página impresa 81). Llama «Alzheimer effect» al reinicio de
la experiencia entre escenarios y propone backpropagation, KAN y simulation-optimization/RL como
alternativas exploratorias, no como ranking demostrado (PDF p. 12; página impresa 91).

## Q1 — qué familia de AI debe usarse

La respuesta empírica propuesta no es «KAN porque Garrido la menciona». La unidad funcional es un
**aproximador con estado** de una política o función de valor producida por un planner bajo creencia.
La familia ganadora se decide en un bake-off emparejado:

1. spline-GAM como control interpretable;
2. MLP con presupuesto equivalente;
3. KAN de baja dimensión;
4. filtro recurrente sólo si el posterior explícito deja residuo informacional;
5. belief/scenario-MPC como experto no neuronal.

La red debe vencer al mejor comparador estructurado en datos nuevos o ser no inferior en calidad y
reducir el coste online de decisión. Si KAN y MLP empatan, se elige el más parsimonioso. RL end-to-end
no es el punto de partida porque confundiría memoria, búsqueda y aproximación.

## Q2 — integración exacta DES–learner

En cada época semanal el DES hace cinco operaciones auditables:

```text
1. entrega órdenes comprometidas y revela únicamente yields ya realizados;
2. actualiza inventario, backlog, pipeline y el posterior L_(t-1);
3. expone al controlador observaciones permitidas, nunca el régimen verdadero;
4. recibe una acción factible de compra/asignación/expedite bajo presupuesto compartido;
5. avanza eventos, registra coste, servicio, recuperación y nueva experiencia.
```

El estado de aprendizaje se conserva entre shocks de una misma cadena, pero no entre réplicas. Los
brazos `reset`, `delayed` y `shuffled` mantienen la misma física y rompen sólo la historia pertinente.
Así, `L_(t-1)` deja de ser una etiqueta conceptual y se vuelve una variable causal falsable.

## Evidencia que ya existe

Program V usó 30 seeds quemadas para selección y 30 distintas para evaluación. Produjo 780 filas
crudas; todos los falsadores de masa, common random numbers, derechos de orden, información y
movimiento físico pasaron.

| Contraste en evaluación | Media | IC 95% |
|---|---:|---:|
| Privilegiado − mejor constante (`H_priv`) | +0,180130 | [+0,163761; +0,196499] |
| Mejor observable − mejor constante (`H_obs`) | +0,179366 | [+0,162989; +0,195743] |
| Bayes retenido − Bayes reset (`H_ret`) | +0,041320 | [+0,026572; +0,056068] |
| Bayes retenido − placebo retardado | +0,073894 | [+0,055374; +0,092414] |
| Bayes retenido − placebo barajado | +0,186329 | [+0,163182; +0,209476] |
| Privilegiado − Bayes retenido/seleccionado | +0,000764 | [−0,000798; +0,002326] |

Esto identifica headroom físico, observable y dependiente de historia. También muestra que el UCB95
del residuo sobre Bayes es sólo 0,002326: no queda una prima de calidad de un punto porcentual que una
red pueda reclamar honestamente en esta celda.

## Cómo evitar el techo y abrir una prima defendible

El sucesor no debe empeorar artificialmente los comparadores. Debe mover la decisión al lugar donde
la tesis simplificó una dificultad real y donde la optimización online sea costosa:

- 6–12 proveedores/materiales con sustitución y compatibilidades;
- lead times y mínimos de orden heterogéneos;
- capacidad finita en materia prima, producción y transporte;
- caducidad y coste de holding/expedite/shortage;
- presupuesto compartido y compromisos no reasignables;
- shocks R22/R23/R24 y degradación de proveedor con persistencia variable;
- demanda estacional con cambios de fase y amplitud fuera de distribución.

Un scenario/belief-MPC consulta el DES muchas veces para decidir. El learner aproxima la política o
el valor del MPC. Esto crea dos pruebas legítimas:

\[
\Delta_{quality}=V(learner)-V(best\ structured\ comparator)
\]

\[
\Delta_{amortization}=C_{online}(MPC)-C_{online}(learner),
\]

con no inferioridad preregistrada en servicio, backlog, coste y recuperación. La complejidad se añade
por realismo operativo y se valida por ablación; no se aumenta hasta que una red gane.

## Traducción a H1–H4

| Hipótesis | Estimando y prueba necesaria | Estado actual |
|---|---|---|
| H1 — recuperación | Diferencia pareada en tiempo hasta volver a servicio objetivo, learner vs estático y vs MPC | Mecanismo visible; learner pendiente |
| H2 — adaptación | Pendiente de desempeño por índice de exposición en cadenas persistentes; comparar retained vs reset | Diseño identificado; debe medirse por shock, no sólo al final |
| H3 — volatilidad | Diferencia de varianza/cola por severidad predeclarada y coste, con bootstrap emparejado | Pendiente en DES completo |
| H4 — dependencia de trayectoria | Retained − reset, más delayed/shuffled, mismas tapes y física | Evidencia de desarrollo positiva en Program V; falta confirmación fresca |

La comparación «learner vs estático» responde H1 pero no prueba prima neural por sí sola. La prima
exige además vencer al filtro, a la heurística y al planner no neuronal.

## Secuencia de ejecución sin gasto circular

1. Portar Program V al adaptador DES completo y validar eventos, masa, capacidad y costes.
2. Congelar un belief/scenario-MPC y medir calidad y tiempo online; si no es costoso o no mejora la
   heurística, cerrar antes de entrenar.
3. Generar dataset del planner con seeds de desarrollo; entrenar spline-GAM, MLP y KAN con presupuestos
   emparejados.
4. Congelar modelos y evaluar en combinaciones fuera de distribución de frecuencia, impacto y demanda.
5. Abrir seeds confirmatorias una sola vez y adjudicar H1–H4 y las dos deltas por separado.

## Frase de contribución que sí puede sostenerse

> We operationalize accumulated learning as a retained, causally testable DES state and distinguish
> neural decision-quality gains from neural amortization of simulation optimization under recurring,
> heterogeneous disruptions.

La novedad no debe formularse como «primer KAN en supply chains» sin una revisión sistemática actual.
La contribución más resistente es el protocolo causal: memoria retenida, controles de reset/barajado,
elegibilidad física antes del learner y separación entre calidad y coste computacional.
