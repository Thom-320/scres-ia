# Estado actual Garrido–WRAP — 2026-08-01

## Decisión editorial

El artículo activo es Garrido–WRAP/v0 para *Computers & Industrial Engineering*. Program Q es un
carril separado y no aporta claims a este manuscrito.

## Estado de implementación

- HEAD de referencia: `8526445cea46204fbae1ea4d8719df3af877f0b3`.
- Contrato: `garrido_wrap_scres_ai_v1`.
- Estado global: `HOLD_WRAP_BEHAVIORAL_FIDELITY` / `DEVELOPMENT_ONLY`.
- `thesis_1to1` permanece congelado.
- Q1: no-go de prima neural en el panel WRAP actual.
- CSSU Gate A: acción de reasignación viva.
- CSSU Gate B: manejo físico de Op11 no especificado; `HOLD`.
- Contención y expedición: no abren headroom bajo `service_first_v2`.
- E1 neural-headroom gate: `NO_GO`; `training_authorized=false`.
- Q2: replay de 90 válido como replay; DES-288 pendiente.

## Corridas activas

La corrida H3′ usa el runner corregido y semillas separadas. Al momento de este estado, los
procesos local y VPS estaban activos y no habían escrito su `result.json`. No se duplican ni se
interpretan hasta comprobar el hash del script, la disjunción de semillas y los falsadores de
merge.

## Cifras retiradas

Los contrastes del meta-aprendiz v1 (`+6.31`, `+5.18`, `+12.31`) y la curva H2 antigua no son
evidencia. El motivo es la fuga de drivers al rankear candidatos no ejecutados.

## Próxima lectura válida

1. cerrar H3′ sin ampliar retrospectivamente su bloque exploratorio;
2. sellar el gate CSSU A;
3. ejecutar/validar replay thesis90;
4. ejecutar DES-288 corregido;
5. adjudicar H1–H4;
6. sólo entonces abrir E2 si un nuevo contrato observacional/físico lo autoriza;
7. no entrenar MLP/PPO mientras E1 siga cerrado.
