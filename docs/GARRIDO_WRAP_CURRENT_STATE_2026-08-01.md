# Estado actual Garrido–WRAP — 2026-08-01

## Decisión editorial

El artículo activo es Garrido–WRAP/v0 para *Computers & Industrial Engineering*. Program Q es un
carril separado y no aporta claims a este manuscrito.

## Estado de implementación

- HEAD de referencia: el campo `repository_head` de
  `results/garrido_wrap_custody_manifest_v1.json` (se regenera con el audit runner).
- Contrato: `garrido_wrap_scres_ai_v1`.
- Estado global: `HOLD_WRAP_BEHAVIORAL_FIDELITY` / `DEVELOPMENT_ONLY`.
- `thesis_1to1` permanece congelado.
- Q1: no-go de prima neural en el panel WRAP actual.
- CSSU Gate A: acción de reasignación viva.
- CSSU Gate B: manejo físico de Op11 no especificado; `HOLD`.
- Contención y expedición: no abren headroom bajo `service_first_v2`.
- E1 neural-headroom gate: `HOLD_E1_PLACEBO_NOT_OPENED`; `training_authorized=false`.
- Q2: el artefacto DES-288 existe en
  `results/garrido_meta_learner_v2/result.json`, pero su adjudicación canónica y custodia
  frente al claim ledger siguen pendientes; no se usa como claim.
- G3a: diseño y código de compatibilidad solamente; el contrato está en
  `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT` y no abre semillas.
- Registro de entornos: `docs/GARRIDO_WRAP_ENVIRONMENT_REGISTRY_2026-08-01.md`.

## Corridas activas

Las dos rebanadas H3′ ya tienen artefactos separados y semillas disjuntas: local
`6000001…6000090` y VPS `6000091…6000120`. Eso no equivale a un resultado merged. No se
interpretan ni combinan hasta comprobar el hash exacto del runner, el manifiesto de módulos, la
identidad conductual del DES, los falsadores y la custodia de ambos artefactos. La rebanada VPS
continúa en `HOLD_SOURCE_AUDIT`; el hecho de que exista `result.json` no elimina ese gate.

El estado H3′ canónico es, por tanto, `ARTIFACTS_PRESENT_MERGE_PENDING`, no “120 réplicas
adjudicadas”. Los artefactos H3′ siguen fuera de custodia Git en este worktree; esa deuda forma
parte del gate y debe resolverse sin regenerar ni duplicar las corridas.

## Cifras retiradas

Los contrastes del meta-aprendiz v1 (`+6.31`, `+5.18`, `+12.31`) y la curva H2 antigua no son
evidencia. El motivo es la fuga de drivers al rankear candidatos no ejecutados.

## Próxima lectura válida

1. cerrar H3′ sin ampliar retrospectivamente su bloque exploratorio;
2. auditar el merge, el manifiesto VPS y las semillas de H3′;
3. reconciliar y sellar el artefacto DES-288 existente, sin duplicar la corrida;
4. adjudicar H1–H4 y custodiar el resultado;
5. mantener G3a en diseño hasta el recibo de Submission A y la actualización de autoridad;
6. no abrir semillas ni entrenar MLP/PPO mientras E1 y la gobernanza sigan cerrados.
