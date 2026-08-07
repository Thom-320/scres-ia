# Enmienda G3c — apertura científica acotada y preflight burned-only

**Estado:** `BURNED_PREFLIGHT_AUTHORIZED_NO_FRESH_SEEDS`

**Fecha:** 2026-08-05

Esta enmienda no reescribe el resultado de G3-obs ni convierte una red en ganadora.
Supersede únicamente la regla de portafolio que impedía construir G3c después de
`STRUCTURED_CONTROL_SUFFICES_G3_OBS`. La razón es científica y está fijada antes del
preflight: G3c estudia un mecanismo ortogonal —compromiso intertemporal por permanencia
mínima— que no fue probado por el `if` observable de G3-obs.

La enmienda autoriza ahora sólo:

* implementación y tests del runner;
* una ejecución `BURNED_PREFLIGHT` sobre el bloque ya consumido
  `contention_headroom` (`5.200.001–5.200.016`);
* cálculo de potencia y publicación de un estado terminal.

No autoriza raíces nuevas, learners, confirmación ni modificación de Program Q, Program O
o Submission A.

## Contratos que permanecen inmutables

La física de G3c sigue siendo la de
`contracts/g3c_temporal_coupling_v2.json`:

```text
cssu_min_dwell_days ∈ {1, 3, 7}
switch_cost_rations = 0
N = 2
split_v1
activation delay = 24 h
```

Los márgenes continúan viniendo de
`docs/ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02.md`. Esta enmienda no cambia
el SESOI (`+0,010`) ni ningún margen.

## Diseño congelado del preflight

El preflight usa dos regímenes ya utilizados en el carril de contención:

```text
R1r+R2r | base
R1r+R2r | freq3_imp2
```

Cada régimen se evalúa con los tres niveles de dwell. `1` es el nulo legacy; `3` y `7`
son los dos tratamientos. La unidad de resampling es la semilla, con comparación pareada
dentro de cada régimen y nivel.

El comparador observable queda congelado como una regla miope equivariante:

```text
unmet_A > unmet_B  → allocation_a = 0.9
unmet_B > unmet_A  → allocation_a = 0.1
empate             → allocation_a = 0.5
```

La regla sólo lee el ledger observado antes de cada decisión. El placebo conserva la misma
cadencia y soporte de acciones, pero elige sin leer el estado. El control de reclamante
equivocado invierte la dirección. Ningún brazo lee el target futuro del riesgo.

El endpoint primario es `worst_claimant_fill`. Los guardarraíles son `flow_fill_rate`,
`lost_orders`, `backorder_qty_final_relative` y las identidades algebraicas de masa,
capacidad y recursos.

## Potencia

El preflight estima la desviación estándar pareada en las 16 semillas burned y publica, para
cada una de las cuatro comparaciones tratamiento×régimen:

```text
MDE(N) = (z_0.90 + z_(1 - 0.05/4)) · SD / sqrt(N)
```

La corrección de Bonferroni cubre los cuatro contrastes primarios. El preflight publica también
la N requerida para detectar el SESOI y la N requerida para cada margen de no inferioridad.
El presupuesto máximo prospectivo queda fijado en **256 semillas por celda**; si cualquier
celda requiere más, el estado terminal es `STOP_G3C_UNDERPOWERED` y no se abre ningún bloque.

La estimación de potencia es diagnóstico burned-only. No selecciona un resultado positivo ni
autoriza por sí misma la ejecución confirmatoria.

## Autoridad de ejecución

La autorización está concedida por el PI en la instrucción de esta sesión y queda limitada a
este preflight burned-only. La ejecución debe sellar:

```text
run_role = BURNED_PREFLIGHT
replay_of = contention_headroom
contract = contracts/g3c_burned_preflight_v1.json
```

Un resultado `POWER_SUFFICIENT` sólo permite preparar una supersession de autoridad y reservar
un bloque virgen. No permite abrirlo automáticamente.

## Estados terminales

```text
STOP_G3C_UNDERPOWERED
STOP_G3C_INSTRUMENT_INVALID
G3C_PREFLIGHT_POWER_SUFFICIENT_NO_FRESH_SEEDS
```

Después de un `G3C_PREFLIGHT_POWER_SUFFICIENT_NO_FRESH_SEEDS` se necesita una segunda
autorización versionada para abrir el bloque virgen y ejecutar G3c.
