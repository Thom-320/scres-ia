# Enmienda E*-C v2 — replay de reparación instrumental

**Estado:** `REPLAY_REPAIR_ONLY_NOT_NEW_SCIENCE`

**Contrato padre:** `docs/PREREGISTRO_ESTAR_CAPACIDAD_BARRIDO_2026-08-03.md`

**Hash del contrato padre:** `2719ff7c79fbbbe979246d26d6b6548a6fd3787a8e91428d9a099701f7061aea`

Esta enmienda no modifica retrospectivamente el contrato padre ni convierte el resultado anterior
en evidencia confirmatoria. Corrige dos falsadores defectuosos, sella la fuente de ejecución y
repite únicamente el bloque ya consumido `5.200.001–5.200.016` como replay declarado.

No se abren raíces nuevas, no se cambia la física, no se cambia el SESOI, no se cambia el conjunto
de presupuestos, regímenes, acciones o semillas, y ningún resultado de este replay permite afirmar
`G1` ni una confirmación independiente.

## Correcciones obligatorias

### `f2_mass_and_demand_are_untouched`

Cada brazo con capacidad se compara, para la misma semilla y régimen, con un brazo sin
`cssu_storage_capacity`. Se comparan:

* demanda total;
* demanda por reclamante A/B;
* todas las acciones, presupuestos y regímenes.

Un solo desvío por encima de `1e-9` hace fallar `f2`. La comprobación de que un conjunto observado
no está vacío no es evidencia de invariancia.

El presupuesto se construye mediante `budgeted_ledger`; la suma de las capacidades CSSU A/B debe
coincidir con el presupuesto declarado. El alcance físico sigue siendo exclusivamente CSSU A/B:
la capacidad de WDC, AL y SB queda fuera de esta corrida y requiere otro contrato.

### `f6_no_gain_by_abandonment`

Para la acción seleccionada por media de `flow_fill_rate` en cada celda `(presupuesto, régimen)`,
se calculan diferencias pareadas contra el brazo sin capacidad usando las mismas semillas:

```text
harm_worst_fill = worst_fill_reference - worst_fill_candidate
harm_lost_orders = lost_candidate - lost_reference
```

Se obtiene un bootstrap porcentual de 5.000 remuestras sobre semillas. Debe cumplirse:

```text
UCB95(harm_worst_fill) <= 0,010
UCB95(harm_lost_orders) <= 0,50
```

El `flow_fill_rate` no se usa como guardarraíl de no inferioridad contra el brazo sin capacidad:
es el endpoint primario de una intervención cuyo efecto esperado es precisamente cambiar el
servicio dentro del horizonte. Su margen `0,005` permanece registrado como sensibilidad, no como
prueba de ausencia de abandono.

### Regla de `f4`

`f4_argmax_moves_with_regime` es un diagnóstico científico terminal, no un falsador de software.
Si falla, el estado correcto es `CAPACITY_CONSTRAINS_WITHOUT_DECIDING`, no
`HALTED_FALSIFIER_FAILED`.

## Estimando y lectura

El estimando se calcula explícitamente como:

```text
H_regime = mean_r[ max_a( mean_s(Y[r,a,s]) ) ]
           - max_a( mean_r( mean_s(Y[r,a,s]) ) )
```

El promedio sobre semillas ocurre antes del máximo por acción; permitir que la acción cambie por
semilla sería un techo clarividente y no es el estimando.

`flow_fill_rate` es `raciones servidas al final / demanda total`. Es un endpoint terminal sujeto a
censura del horizonte, no una medida pura de latencia. La interpretación temporal queda limitada
a lo que permanece sin servir al final; AUC de pérdida de servicio o panel semanal requerirían una
extensión futura.

## Custodia de ejecución

El runner exige:

```text
--contract docs/ENMIENDA_ESTAR_CAPACIDAD_BARRIDO_V2_REPLAY_2026-08-05.md
--replay-of contention_headroom
--run-role REPLAY
```

El resultado debe sellar `run_role`, `replay_of` y un `module_manifest` que incluya el entry script,
`supply_chain.py`, `node_capacity.py`, métricas, ledger y custodia. El resultado anterior permanece
como artefacto de desarrollo con falsadores inválidos; no se edita ni se promueve.

## Estados terminales

```text
HALTED_FALSIFIER_FAILED
CAPACITY_OPENS_REGIME_DEPENDENT_HEADROOM
ARGMAX_MOVES_WITHOUT_VALUE
CAPACITY_CONSTRAINS_WITHOUT_DECIDING
```

Incluso si todos los falsadores pasan, el resultado sigue siendo un replay sobre semillas
quemadas y no concede autorización para learners, nuevas semillas o confirmación.

