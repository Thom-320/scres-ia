# Preregistro de diseño — Garrido expanded DES / E* v1

**Estado:** `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`

**Contrato:** `contracts/garrido_expanded_des_e_star_v1.json`

**Autoridad superior:** `main:contracts/authority_ladder_v1.json`

**Semillas nuevas:** ninguna abierta

**Learner:** no autorizado

Este documento ejecuta la primera parte del camino acordado: fijar un único kernel ampliado y
la jerarquía de métricas antes de modificar la física o abrir una raíz científica. No reabre
Program Q, Program O ni `thesis_1to1`.

## 1. Pregunta

¿Qué derechos de decisión físicamente válidos —procurement, buffers upstream y buffers/dispatch
downstream— generan valor contingente observable; cuánto captura el mejor control estructurado;
y queda un residual para un controlador aprendido?

La ampliación debe seguir la instrucción de Garrido del 28 de julio: añadir nodos y variables de
decisión, no alargar artificialmente el episodio. El DES debe conservar producción, capacidad,
transporte, lead times, inventario en tránsito y backorders.

## 2. Kernel único y máscaras

Se construirá un solo `E_star`, con todas las transiciones y observaciones aprobadas fijadas antes
de la primera ejecución científica. Los ocho masks (`M000`–`M111`) activan derechos, pero no
introducen física nueva. Un mask superior debe contener la política incumbente del mask inferior.

La primera implementación no entrenará una red. Primero debe pasar:

1. bridge flags-off contra `E_Garrido`;
2. conservación de masa y capacidad;
3. ledger de procurement, transporte, WIP y stock;
4. acción viva y cadencia correcta;
5. ausencia de futuro en la observación;
6. custodia de tapes y payload científico.

No se autoriza rellenar un buffer desde el aire. Todo aumento debe tener origen upstream,
capacidad y coste/lead time identificables.

## 3. Jerarquía de métricas

Excel/ReT y Cobb–Douglas son las dos alternativas de endpoint principal. La elección exacta se
firmará antes de abrir datos frescos; no se podrá escoger después de observar qué métrica favorece
a una política.

Ambos se reportarán siempre, junto con:

- `flow_fill_rate` y `worst_claimant_fill`;
- `lost_orders`, `unresolved_orders` y `backorder_qty_final`;
- `service_loss_auc`;
- raciones entregadas y todos los recursos consumidos.

`service_first_v2` puede ordenar candidatos durante el desarrollo, pero su tupla lexicográfica no
es un estimando cardinal y no recibirá un intervalo de confianza único.

### CVaR

CVaR sí se utilizará, pero como métrica secundaria de cola:

- `ret_excel_cvar05` y `ret_excel_cvar10` se reportarán en todas las campañas autorizadas;
- puede aparecer en una recompensa de desarrollo o en un análisis de sensibilidad;
- no será el endpoint principal;
- no podrá promover ni bloquear por sí solo una política;
- si Garrido solicita una barrera CVaR, se redactará una enmienda con margen, potencia y unidad de
  inferencia propios.

Esto mantiene la decisión de Garrido del 2 de julio: CVaR no es la métrica principal para estos
riesgos operacionales frecuentes, pero sigue siendo información útil sobre la cola.

## 4. Escalera de comparación

Antes de cualquier MLP, KAN o PPO se ejecutarán, bajo el mismo kernel, tapes, derechos y presupuesto:

```text
constante
→ lookup / order-up-to
→ threshold / hysteresis
→ árbol / tabular
→ spline / GAM
→ DP / rollout
→ MPC directo
→ MPC robusto o por escenarios, si el contrato lo justifica
```

El learner sólo se abre si una política observable supera al mejor control estructurado por el
SESOI firmado, con intervalos agrupados por tape/celda y guardarraíles de servicio y recursos.

## 5. Tres tipos de prima, separados

El contrato permite evaluar tres preguntas distintas:

1. **Calidad:** `Delta_N = V(neural) - V(BestStructured)`.
2. **Amortización:** misma calidad y seguridad con menor latencia o menos llamadas al DES.
3. **Generalización:** mejor resultado en regímenes, tamaños o permutaciones no vistos.

No se podrá cambiar de “prima de calidad” a “prima de velocidad” después de ver los resultados.
`Delta R²` queda fuera del gate de control.

## 6. Ruta neural, si el gate abre

- MLP para un vector fijo de observaciones;
- KAN como comparación solicitada por Garrido, no como rescate post hoc;
- DeepSets/attention sólo si el conjunto de nodos varía;
- GRU/LSTM sólo si una prueba de aliasado demuestra que la historia causal mejora el valor fuera de
  muestra;
- una sola arquitectura primaria para la confirmación independiente.

Una red que iguala a MPC pero reduce materialmente el coste online produce un resultado de
amortización, no una prima de calidad. Una regla o DP que captura todo produce
`STOP_STRUCTURED_CONTROL_SUFFICES`.

## 7. Qué se puede hacer ahora

Permitido antes del recibo editorial:

- revisar y completar el manifiesto con Garrido;
- implementar el contrato y validadores con defaults inertes;
- construir el bridge flags-off;
- escribir tests de conservación, liveness, no-futuro y mutantes;
- auditar el registro de semillas y los artefactos H3′/DES-288.

No permitido:

- abrir raíces o tapes nuevos;
- hacer screens científicos de MPC/DDMRP;
- entrenar MLP, KAN, PPO o RecurrentPPO;
- seleccionar una arquitectura por resultados de desarrollo.

## 8. Regla de cierre

El programa termina, dentro del portafolio aprobado, con cualquiera de:

```text
STOP_NO_PHYSICAL_HEADROOM
STOP_HEADROOM_NOT_OBSERVABLE
STOP_STRUCTURED_CONTROL_SUFFICES
STOP_NEURAL_EQUIVALENT
NEURAL_AMORTIZATION_PREMIUM
NEURAL_PREMIUM_CONFIRMED
BUDGET_EXHAUSTED_WITHOUT_PREMIUM
```

Ningún resultado de desarrollo será llamado “prima neural confirmada”. Esa etiqueta requiere una
única confirmación independiente en tapes y seeds vírgenes.
