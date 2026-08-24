# Prerregistro — Vía B: headroom físico en DES híbrido de dos productos

**Contrato:** `contracts/program_b_full_des_gate_v2.json`

**Estado:** congelado para reanálisis de los bloques Program O ya ejecutados.
No se abren raíces nuevas en este documento.

## Por qué priorizamos B, sin perder Garrido

La Vía B es la mejor primera prueba para el v.0 porque convierte la afirmación vaga
«una red aprende resiliencia» en una tarea causal: dos productos no fungibles compiten
por una capacidad compartida y una política distinta puede mejorar el resultado sin
abandonar órdenes.

No es el modelo nativo de Garrido-Ríos.
Es una extensión explícita de investigación.

La conexión con Garrido 2024 es directa:

1. las variables de decisión son las entradas del bucle;
2. el DES produce una métrica SCRES;
3. el estado de la corrida informa la siguiente configuración;
4. el resultado permite comparar retención, búsqueda y control.

La Vía A seguirá siendo necesaria para responder de forma fiel las dos preguntas de
Garrido 2024 sobre aprendizaje entre configuraciones.

## Pregunta y gates

**Pregunta B:** ¿existe headroom operativo atribuible a la asignación de una capacidad
compartida no fungible, después de impedir que la métrica premie abandono o pérdida de
un producto?

Se evalúa primero el oráculo perfecto sólo como diagnóstico de techo alcanzable.
No es una política implementable y no prueba ventaja de RL.

Si el headroom no supera el SESOI, no se entrena PPO, MLP, RecurrentPPO ni KAN en
esta extensión.

## Física congelada

- Productos: `P_C` y `P_H`.
- BOM, masa por ración y tasa de producción idénticos.
- Sustitución completa desactivada en el brazo primario.
- Riesgos y tiempos de proceso estocásticos desactivados en el brazo estructural.
- Tres lotes semanales; ocho semanas; acción `0,1,2,3` = número de lotes asignados a `P_C`.
- Frontera abierta completa: `4^8 = 65.536` calendarios.
- Scheduler principal: `centered_minority_v1`.
- Control de ordenamiento: `blocked_left_v1` y `blocked_right_v1`.
- Placebo fungible: sustitución completa; debe producir `H_PI = 0` bit a bit.

## Métrica y guardrails

### Primaria

`ret_excel_full_ledger`.

La fórmula de Garrido se calcula sobre los 48 pedidos generados.
Los pedidos no resueltos reciben cero.

El estimando es la diferencia por tape entre el mejor calendario que satisface todos
los guardrails y el comparador correspondiente.

Si varios calendarios empatan en el endpoint primario, el desempate es fijo y no usa
`ret_thesis`: menor `unresolved_quantity`, mayor `worst_product_fill`, mayor
`actual_payload`, menor `ending_inventory_total` y, finalmente, orden lexicográfico.

### Secundarias obligatorias

- `ret_excel_clipped_0_1`.
- `ret_thesis`.
- `flow_fill_rate`.
- `delivered_rations`.
- `lost_orders`.
- `unresolved_orders`.
- `terminal_stock`.
- `worst_product_fill`.
- `service_loss_auc`.
- `max_backlog_age`.

El endpoint clipped es descriptivo y nunca puede seleccionar por sí solo una política.

### Guardrails

Un calendario sólo es elegible frente a su comparador si:

- no aumenta `lost_orders` ni `unresolved_orders`;
- no reduce `delivered_rations`;
- no aumenta la cantidad restante de ningún producto;
- no reduce el fill de `P_C`, `P_H` ni `worst_product_fill`;
- no aumenta `service_loss_auc` ni `max_backlog_age`;
- mantiene producción bruta y capacidad cobrada idénticas;
- mantiene los residuos de conservación por debajo de `1e-8`.

`terminal_stock` se reporta siempre.
Se marca una alerta si aumenta más de 5% de la demanda generada sin mejora de la
métrica primaria.

## Comparadores

### Incumbente congelado

Calendario de desarrollo seleccionado antes de la validación:

```text
[2, 1, 2, 2, 2, 1, 2, 2]
```

### Incumbente in-sample

Máximo estático de los 65.536 calendarios dentro de cada bloque y perfil.
Se reporta por separado.
Nunca sustituye al contraste con el incumbente congelado.

## SESOI y adjudicación

- SESOI primario: `0,01` puntos absolutos de `ret_excel_full_ledger`.
- Desarrollo: media de headroom seguro `> 0,015`.
- Validación: LCB95 unilateral simultáneo `> 0,01`.
- El resultado debe ser estable en los bloques de desarrollo y validación.
- Cualquier fallo de guardrail, null fungible, conservación, paridad o custodia detiene
  la promoción.
- Ningún resultado de esta etapa autoriza todavía un learner.

## Semillas

Se reutilizan únicamente bloques ya ejecutados y con custodia verificada:

- Desarrollo: `7400049–7400072`.
- Validación: `7400097–7400120`.

Este análisis no declara una nueva confirmación ni abre semillas adicionales.

## Relación con el v.0

- H1: comparar el headroom y recovery de una política adaptativa contra estáticos.
- H2: evaluar mejora bajo exposición sucesiva, sólo después de que exista headroom físico.
- H3: medir varianza entre composiciones de demanda/disrupción, no sólo media.
- H4: comparar memoria retenida frente a reset manteniendo el mismo DES y tapes.

## Relación con Garrido 2024

La Vía B no reemplaza la respuesta principal a Garrido.
La respuesta de Garrido se completará con un bucle externo de optimización de simulación
que conserva el aprendizaje entre configuraciones y compara OFAT, random, UCB1,
lookahead, Bayesian optimization y surrogates.
