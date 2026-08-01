# Prerregistro G3a — asimetría de dos reclamantes, v2

**Estado:** `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`  
**Fecha:** 2026-08-01  
**Artículo activo:** Garrido–WRAP/v0, destinado a *Computers & Industrial Engineering*  
**Contrato machine-readable:** [`contracts/g3a_asymmetric_claimants_v2.json`](/Users/thom/Projects/research/scres-ia/contracts/g3a_asymmetric_claimants_v2.json)  
**Registro de semillas:** [`research/seed_custody_registry.json`](/Users/thom/Projects/research/scres-ia/research/seed_custody_registry.json)

## 1. Frontera de autorización

Este documento congela el diseño permitido para un screen mecanístico futuro. No autoriza
semillas nuevas, tapes científicos, entrenamiento neuronal ni una confirmación. La frontera
operativa vigente es `Authority Ladder V1`: hasta recibir el comprobante de sumisión de
Submission A sólo están permitidos diseño, código con defaults inertes, pruebas sintéticas,
pruebas con datos burned y custodia.

Submission A, Program Q, Program O y `thesis_1to1` mantienen sus contratos y resultados
inmutables. G3a no transfiere claims de Program Q/O ni reabre sus STOP. El registro central de
semillas es deliberadamente conservador: que un bloque no aparezca en él no demuestra que sea
virgen.

## 2. Pregunta neutral

> ¿La heterogeneidad de demanda y un riesgo localizado parcial generan valor de asignación
> observable, seguro y no capturado por controles estructurados bajo el contrato de dos CSSU?

La pregunta no es qué modificación hace ganar una red. G3a primero identifica, en orden, si
existe valor físico privilegiado, si puede observarse, si puede convertirse con una regla o
controlador clásico y si queda un residual material. Sólo el último de esos pasos puede abrir
desarrollo neural.

El resultado será válido únicamente para este contrato, sus observaciones, recursos, tapes,
presupuesto, endpoints y comparadores.

## 3. Contrato físico

Se conservan dos reclamantes (`A`, `B`) y la interfaz histórica. El screen inicial usa sólo:

```text
allocation_a ∈ {0,25; 0,50; 0,75}
```

El código acepta el intervalo continuo, pero una rejilla más fina (`0,1…0,9`) no está
autorizada en este screen. Sólo podrá abrirse mediante una enmienda de desarrollo declarada
antes de observar el resultado del screen.

El factorial mínimo es:

| factor | niveles |
|---|---|
| demanda | 50/50, 70/30, 30/70 |
| riesgo | neutral, parcialmente localizado en A, parcialmente localizado en B |
| recurso | no fungible; pooling verdaderamente invariante a la acción |

El riesgo localizado no puede destruir completamente una CSSU como mecanismo principal. Ambos
reclamantes deben permanecer operativos en una fracción material de los estados vivos y deben
existir direcciones de estrés A y B. El objetivo es conservar un trade-off de asignación, no
convertir la acción en una decisión obvia porque una ruta quedó inutilizada.

`reallocate_unused=True` no es el nulo fungible. Sólo redistribuye capacidad que quedó libre;
cuando ambos reclamantes pueden absorber sus cuotas, la acción sigue cambiando el ledger. El
brazo de pooling completo debe demostrar por tape que todas las acciones producen la misma
trayectoria y los mismos endpoints. Si no lo hace, se reportará como control de reasignación de
sobrantes, no como pooling action-invariant.

## 4. Generación exógena y CRN

La rama histórica `weights=None` debe delegar exactamente en
`stable_cssu_destination(simulation_seed, order_id)`. No se modifica la cadena SHA-256
`dra1-cssu-v1` ni se consume el RNG del simulador.

Los brazos ponderados usarán `stable_cssu_destination_weighted`, con un uniforme de 64 bits
derivado de:

```text
namespace + simulation_seed + event_id
```

La transformación de 50/50, 70/30 o 30/70 se hará mediante CDF sobre el mismo uniforme. Para
riesgos localizados se congelará un tape exógeno con, como mínimo:

```text
event_id, risk_id, onset, duration, impact, u_target
```

Los brazos compartirán onset, duración e impacto. Cambiar los pesos sólo podrá cambiar el
target mediante la CDF; no podrá cambiar el calendario futuro ni el consumo del RNG global.

## 5. Información y régimen

El régimen debe variar dentro del tape. No se seleccionará una constante distinta por celda
cuando el nombre de la celda ya revela el reclamante afectado.

La política observable no recibirá el régimen verdadero, el target futuro, la duración futura,
un `cell_id` privilegiado ni eventos que aún no hayan ocurrido. Antes de ejecutar ciencia se
congelará un manifiesto de observación; podrá incluir estado actual de CSSU, backlog por
reclamante, inventario, tránsito, servicio reciente y alarmas causales disponibles antes de la
acción.

Se separarán cuatro cosas:

1. **`H_PI`**: valor usando el reclamante/régimen verdadero; diagnóstico privilegiado.
2. **`H_obs`**: valor de una política no anticipativa, cross-fitted, usando sólo observaciones
   permitidas.
3. **`H_structured`**: valor del mejor controlador estructurado predeclarado.
4. **`H_residual`**: valor observable restante después de ese controlador.

`Delta_N` sólo se define después de autorizar desarrollo neural y siempre significa valor de
rollout closed-loop de la política neural menos el mejor controlador estructurado.

## 6. Endpoint y guardarraíles

El endpoint primario escalar será:

```text
worst_claimant_fill
```

`service_first_v2` puede servir como regla de selección admisible, pero no se calculará un
intervalo de confianza sobre una tupla lexicográfica. Los guardarraíles serán:

- fill agregado;
- cantidad perdida;
- cantidad no resuelta;
- backorders;
- `service_loss`;
- uso de recursos;
- ReT full-ledger.

ReT y Cobb–Douglas quedan como diagnósticos o sensibilidades; `ret_excel` conserva únicamente
su papel histórico y no será objetivo único. Los outcomes estocásticos tendrán márgenes de
no-inferioridad y una potencia firmada antes de abrir seeds. Las identidades físicas —masa,
capacidad, tape y ausencia de futuro— sí deben cumplirse exactamente o con tolerancia numérica
congelada.

## 7. Falsadores y mutation tests

Los siguientes falsadores son vinculantes para cualquier ejecución futura:

- realización de los pesos por número de pedidos y por cantidad;
- acción viva en ambos sentidos y fracción mínima de estados vivos;
- pooling completo action-invariant por trayectoria, no sólo media cercana a cero;
- equivarianza al intercambiar A/B, pesos, riesgo y acción `a ↔ 1-a`;
- identidad de onset, duración e impacto entre brazos;
- ausencia de target verdadero, futuro o `cell_id` en observaciones;
- señales shuffled, delayed y wrong-claimant como placebos;
- hash del payload científico separado del hash de provenance.

Además habrá mutation tests que saboteen deliberadamente el software y exijan que el harness
los detecte:

- ignorar `weights`;
- exponer el target o duración futuros;
- permitir atravesar una CSSU caída;
- romper la sustracción de capacidad o la conservación de masa.

Un falsador que falla no se borra ni se interpreta como un resultado científico válido. El
artefacto se conserva como inválido, con su diagnóstico, y queda bloqueado para promoción.

## 8. Escalera de comparadores

Antes de cualquier learner se probarán, con iguales derechos de información y recursos:

```text
constante universal
→ lookup
→ threshold/hysteresis
→ árbol/tabular
→ spline/GAM si corresponde
→ DP/rollout
→ MPC
→ belief-MPC/belief-DP sólo si la observabilidad parcial está demostrada
```

Una red que imita a un teacher puede medir convertibilidad o amortización; no demuestra por sí
sola una prima de calidad. `Delta R²` queda fuera del gate principal de control y sólo puede
abrir un contrato predictivo separado.

Las representaciones no son peldaños de algoritmo:

- MLP para interacciones vectoriales fijas;
- DeepSets/attention sólo para conjuntos o permutaciones;
- KAN sólo si la estructura funcional lo justifica;
- GRU/LSTM sólo si la historia demuestra valor causal.

## 9. Gating y estados terminales

Los estados de lectura son:

```text
STOP_G3A_N2_NO_MATERIAL_PHYSICAL_HEADROOM
STOP_G3A_HEADROOM_NOT_DEPLOYABLY_OBSERVABLE
G3A_STRUCTURED_CONTROL_SUFFICES
AUTHORIZE_NEURAL_DEVELOPMENT_G3A
NEURAL_PREMIUM_CONFIRMED_IN_G3A_CONTRACT
BUDGET_EXHAUSTED_WITHOUT_PREMIUM
```

Interpretación:

1. Si `H_PI` no supera el SESOI, se cierra el contrato G3a y no se afirma nada sobre N=3 o
   temporalidad.
2. Si `H_PI` pasa pero `H_obs` no, el valor existe sólo privilegiadamente.
3. Si `H_obs` pasa pero el mejor estructurado captura el valor, el resultado es
   `G3A_STRUCTURED_CONTROL_SUFFICES` y no se entrena una red para redescubrirlo.
4. Sólo si queda un residual observable material se autoriza desarrollo neural.

Un positivo de desarrollo no es una prima confirmada. Congela una única confirmación
independiente con tapes, seeds, código, arquitectura, comparador, presupuesto y análisis
nuevos. Sólo esa confirmación puede producir `NEURAL_PREMIUM_CONFIRMED_IN_G3A_CONTRACT`.

Si falla, sólo se continúa dentro del portafolio ya preregistrado. No se agrega una cuarta
física, una métrica de rescate ni una arquitectura elegida post hoc.

## 10. Orden posterior

G3a no autoriza automáticamente N=3. Si muestra valor observable pero el control estático o un
MPC sencillo lo absorbe, el siguiente mecanismo será temporal con `N=2`: switching cost,
minimum dwell, setup, ramp, posición persistente o tiempo de retorno.

N=3 sólo se podrá abrir con una hipótesis independiente —interacción triple, matching, kit,
coaliciones, capacidad indivisible o número variable de reclamantes— que no pueda expresarse
con dos. O→Q+ y Q-direct se mantienen fusionados como una sola familia prospectiva nueva; no
existirá un `O+0` que replique Program O cerrado. Observabilidad parcial será la última extensión
y tendrá que demostrar aliasado, historia informativa y un comparador belief estructurado.

## 11. Semillas y próximos pasos autorizados

El bloque `7_700_001…7_700_120` está reservado en el registro central, pero no abierto. Los
bloques de selección, validación y confirmación aún no se asignan. Antes de abrir cualquier
semilla faltan:

1. recibo de sumisión de Submission A;
2. cierre de H3′ y resolución del `HOLD_SOURCE_AUDIT` VPS;
3. reconciliación canónica de WRAP-288;
4. manifiesto final de observaciones y tape exógeno;
5. SESOI y potencia;
6. auditoría de virginidad local, Git y remoto;
7. actualización explícita de la autoridad operativa.

Mientras tanto sí están autorizados tests sintéticos, pruebas de compatibilidad legacy,
validación de hashes, mutation tests y trabajo documental. No están autorizados G3a científico,
MLP, KAN, PPO, RecurrentPPO ni una confirmación.

## 12. No-claims

Este prerregistro no permite afirmar que:

- la asimetría N=2 refute N=3 o toda la familia G3;
- `reallocate_unused` sea pooling fungible completo;
- Program Q haya competido específicamente contra belief-MPC;
- un resultado G3a transfiera evidencia a Program Q/O;
- la ausencia de prima en un contrato haga imposible una prima neural en general.
