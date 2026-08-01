# Preregistro Q2 — DES extendido de 288 configuraciones

**Escrito antes de la confirmación DES-288.**
**Estado:** `READY_NOT_STARTED_H3_BLOCKING`
**Paper lane:** Garrido–WRAP/v0 → *Computers & Industrial Engineering*

## Pregunta y estimando

Q2 de Garrido et al. (2024) se operacionaliza como un ciclo entre corridas:

```text
configuración -> DES WRAP -> observación de servicio/SCRES -> actualización theta_k
             -> siguiente configuración
```

El estimando primario es la diferencia pareada **`retained − reset` en eficiencia de
búsqueda**, definida como `runs_to_oracle(reset) − runs_to_oracle(retained)`. Un valor
positivo favorece la memoria entre campañas. La unidad de inferencia es la réplica/semilla,
que agrupa sus seis contextos de riesgo; el intervalo se obtiene por bootstrap de esos
bloques, no tratando cada contexto como independiente.

La comparación contra OFAT, aleatorio y no-update es secundaria. Ningún brazo puede
retener inventario, WIP, backorders, RNG, eventos, normalizadores ni información futura.
Sólo `theta`/`rho` del aprendiz puede cruzar el límite de contexto en `retained`.

## Superficie y brazos

- 288 configuraciones: `buffer_hours × shifts × op9_rop × op12_rop`.
- Seis contextos: `R1r`, `R2r`, `R1r+R2r` y sus tres escaladas.
- Una semilla DES común por contexto y réplica para todos los brazos.
- Bloque virgen reservado para confirmación: `7_100_001 … 7_100_012`.
- Presupuesto común por contexto: 24 corridas.
- Orden de contextos fijo y publicado en el artefacto; no se cambia después de observar el
  resultado.

Brazos:

1. `ofat`: barrido lazy de un factor a la vez desde la configuración por defecto; se llama
   `thesis_order` en la interpretación y no se presenta como OFAT universal.
2. `random`: índice aleatorio sin consultar outcomes antes del sorteo.
3. `no_update`: mismo selector lineal, pero nunca incorpora observaciones.
4. `retained`: selector lineal vectorial que actualiza `rho` con la clave observada y conserva
   `rho` entre contextos.
5. `reset`: código, superficie, orden, presupuesto y streams idénticos; reinicia `rho` en
   cada contexto.

El selector predice los cuatro componentes por separado y compara la predicción como tupla
lexicográfica. No se promedia ni se convierte la tupla en una suma ponderada.

## Endpoint

La clave observada es `service_first_resilience_v2`:

```text
(worst_claimant_fill,
 flow_fill_rate,
 -backorder_qty_final,
 ret_excel_visible_clipped_0_1)
```

La clave es un endpoint normativo estipulado, no evidencia independiente. Se conservarán
además los componentes de servicio, cola, pérdidas, AUC de servicio y ReT diagnóstico. El
resultado no puede ganar por abandonar un reclamante.

## Reglas de lectura

El resultado podrá llamarse `PASS_Q2_CLOSED_LOOP` sólo si:

- todas las falsificadores pasan;
- el runner ejecutó DES real en las 288 configuraciones declaradas;
- el bloque de semillas es disjunto de los bloques previos;
- `retained` y `reset` difieren únicamente en la persistencia de `rho`;
- el IC agrupado del estimando primario excluye cero en favor de `retained`;
- no hay deterioro del endpoint de servicio en el contraste retenido–reset;
- el artefacto queda sellado con contrato, referencia, runner, semillas y hash.

Si el efecto es nulo, el resultado se reportará como frontera de Q2, no como ausencia
universal de aprendizaje. Si el mejor control clásico alcanza al oráculo, tampoco se
autoriza una red.

## Falsadores

Cada falsador debe registrar por qué puede fallar:

1. variación real de la superficie y existencia de un oráculo;
2. OFAT mueve como máximo un factor por propuesta;
3. retained/reset comparten superficie, semillas, orden, presupuesto y forma de trazas;
4. presupuesto cero produce brazos retained/reset idénticos;
5. el brazo aleatorio no cambia su secuencia al permutar outcomes;
6. permutar drivers post-episodio no cambia la secuencia de búsqueda;
7. las claves almacenadas coinciden con una recomputación independiente de `service_first_v2`;
8. la masa/servicio de las simulaciones es finita y no se introduce un reclamante ficticio;
9. las semillas de confirmación no colisionan con los bloques reservados.

Un fallo detiene la interpretación. Ningún resultado de este contrato autoriza MLP/PPO:
esa decisión sigue bloqueada por el gate de headroom E1 y por el contrato neural separado.
