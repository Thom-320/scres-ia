# Preregistro — G3-obs v2, confirmación independiente

**Estado:** `PROSPECTIVE_CONFIRMATION_NOT_YET_EXECUTED`

**Escrito y fijado antes de abrir el bloque.** Esta confirmación no reetiqueta ni extiende
el bloque de desarrollo `7.800.001–7.800.140`.

## Autorización y custodia

La ejecución se autoriza explícitamente en la sesión del 2026-08-02 como respuesta a la
decisión de resolver el bloqueo de alcance. La autorización se limita a este contrato y a
este bloque:

```text
semillas:       7.900.001–7.900.140
partición:      70 desarrollo / 70 test
repeticiones:   140
replay_of:      ninguno
entrenamiento:  no autorizado
```

El bloque debe estar `RESERVED_NOT_OPENED` en
`research/seed_custody_registry.json` antes de lanzar. No se ejecuta smoke ni prueba
parcial con estas semillas: la primera apertura será la corrida completa.

## Pregunta confirmatoria

¿El valor observable de la señal operacional sobre la mejor constante, bajo el contrato
G3-obs, es material y seguro, y queda agotado por el umbral estructurado simple sin un
residual material?

Se conserva la física, las celdas, la señal, los brazos, el endpoint, el SESOI y los
márgenes de `PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md`. El bloque nuevo es la única
diferencia confirmatoria.

## Estimandos y lectura

* Primario: `H_obs = V(threshold_windowed) − V(best_constant)` sobre
  `worst_claimant_fill`.
* SESOI: `+0,010` de fill absoluto.
* El contraste es interpretable sólo si el MDE publicado es `≤ 0,010` en ambas celdas.
* El residual es `V(tabular_5bin) − V(threshold_windowed)` y se considera material sólo
  bajo la regla ya fijada: media `≥ SESOI` y `LCB95 > 0`.
* Los guardarraíles mantienen `flow_fill_rate=0,005`,
  `lost_orders=0,50` y `backorder_qty_final_rel=0,01`.

## Falsadores vinculantes

1. La señal se construye antes de la acción y no lee el futuro.
2. El orden completo de `f2` se verifica en cada celda:

   ```text
   threshold_windowed > threshold_delayed > uninformed_placebo > wrong_claimant
   ```

3. Umbrales y bins se ajustan sólo con las 70 semillas de desarrollo.
4. Todos los guardarraíles usan márgenes firmados.
5. El MDE se publica pase o falle.
6. El actuador está vivo y respeta la latencia.
7. No hay ganancia por abandono.
8. El bloque pasa la custodia central sin `replay_of`.
9. El artefacto sella este contrato exacto en tiempo de ejecución y conserva el manifiesto
   de módulos/entry script.

## Estados terminales

```text
STRUCTURED_CONTROL_SUFFICES_G3_OBS
G3_OBS_RESIDUAL_OVER_SIMPLE_RULE
OBSERVABLE_CONVERSION_FAILS
STOP_G3_OBS_UNDERPOWERED
STOP_G3_OBS_GUARDRAIL
HALTED_FALSIFIER_FAILED
```

Sólo el resultado de esta corrida puede resolver el alcance confirmatorio v2. Cualquier
resultado positivo de desarrollo anterior sigue siendo suplementario; ninguna red se
entrena en este contrato.
