# Estado actual Garrido–WRAP — 2026-08-01

## Decisión editorial

El artículo activo es Garrido–WRAP/v0 para *Computers & Industrial Engineering*. Program Q es un
carril separado y no aporta claims a este manuscrito.

## Estado de implementación

- HEAD de referencia: `94478e9` (custody manifest actualizado).
- Contrato: `garrido_wrap_scres_ai_v1`.
- Estado global: `HOLD_WRAP_BEHAVIORAL_FIDELITY` / `DEVELOPMENT_ONLY`.
- `thesis_1to1` permanece congelado.
- Q1: no-go de prima neural en el panel WRAP actual.
- CSSU Gate A: acción de reasignación viva.
- CSSU Gate B: manejo físico de Op11 no especificado; `HOLD`.
- Contención y expedición: no abren headroom bajo `service_first_v2`.
- E1 neural-headroom gate: `HOLD_E1_PLACEBO_NOT_OPENED`; `training_authorized=false`.
- Q2: replay de 90 válido como replay; contrato/runner DES-288 listo, ejecución bloqueada
  hasta cerrar H3′.
- Registro de entornos: `docs/GARRIDO_WRAP_ENVIRONMENT_REGISTRY_2026-08-01.md`.

## Corridas activas

La corrida H3′ usa el runner corregido y semillas separadas. Al momento de este estado, los
procesos local y VPS estaban activos y no habían escrito su `result.json`. No se duplican ni se
interpretan hasta comprobar el hash del script, la disjunción de semillas y los falsadores de
merge. La inspección previa al cierre detectó además que el snapshot VPS tiene un
`supply_chain/supply_chain.py` distinto al local y carece de `service_first_metric.py`; por eso
el merge queda `HOLD_SOURCE_DRIFT` salvo que el manifest de módulos demuestre identidad.

## Cifras retiradas

Los contrastes del meta-aprendiz v1 (`+6.31`, `+5.18`, `+12.31`) y la curva H2 antigua no son
evidencia. El motivo es la fuga de drivers al rankear candidatos no ejecutados.

## Próxima lectura válida

1. cerrar H3′ sin ampliar retrospectivamente su bloque exploratorio;
2. auditar el merge y las semillas de H3′;
3. ejecutar DES-288 con `7_100_001…7_100_012`;
4. adjudicar H1–H4 y custodiar el resultado;
5. sólo entonces abrir E2 si un nuevo contrato observacional/físico lo autoriza;
6. no entrenar MLP/PPO mientras E1 siga cerrado.
