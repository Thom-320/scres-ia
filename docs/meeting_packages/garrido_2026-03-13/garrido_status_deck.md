---
title: "Estado del proyecto MFSC-IA"
subtitle: "Reunión con Garrido"
author: "Thomas"
date: "13 de marzo de 2026"
---

# Objetivo de la reunión

- Mostrar qué ya quedó firme en el repo.
- Separar con claridad el carril principal del paper del carril DKANA.
- Confirmar qué parte ya está lista para David y qué parte sigue siendo trabajo nuestro.

# Qué ya está firme

- El DES de la MFSC ya está reconstruido en Python/SimPy y auditado contra la tesis.
- El wrapper RL ya funciona con benchmark reproducible, reward audit y comparación contra baselines.
- La lectura actual del proyecto ya no depende de una arquitectura nueva.
- La historia principal del paper es: `DES + RL + reward audit + POMDP + benchmark`.

# Auditoría frente a Garrido (2017)

- Sí coincide en la topología base de 13 operaciones, riesgos `current/increased`, buffers y escenarios de shifts.
- Ya quedó implementada la cola de backorders estilo Garrido:
  - backlog explícito
  - cap de `60`
  - `Ut` como órdenes no atendidas
  - prioridad a demanda contingente
  - `SPT` por tamaño
- El gap que sigue abierto no es la cola, sino el output tipo SDM y el cálculo orden-a-orden de `ReT`.

# Estado actual del carril RL

- La reward de entrenamiento base quedó congelada en `control_v1`.
- `ReT_thesis` queda como métrica de reporte, no como reward principal.
- `PPO + MLP` ya sirvió para validar el carril experimental.
- La comparación disciplinada sigue este orden:
  - `PPO` baseline
  - `SAC`
  - `PPO + frame stack`
  - `RecurrentPPO`
- La decisión congelada del paper es no depender de `DKANA`, `KAN` o `GNN` para submission.

# Qué ya queda listo para David / DKANA

- `observation_version="v3"` ya existe y está estable.
- El entorno exporta:
  - `observations.npy` con `20` dims
  - `state_constraint_context.npy` con `45` dims
- El contexto DKANA ya incluye:
  - estado de cola (`pending_backorders_count`, `pending_backorder_qty`, `unattended_orders_total`)
  - vector de backorder por nodo
  - vector de disrupción por operación
- Ya existe evaluación de políticas custom con `run_episodes()`.

# Qué NO está implementado todavía de DKANA

- No hay un `train_dkana_*.py` listo para entrenamiento real.
- No hay benchmark formal `DKANA vs PPO/SAC` dentro del carril principal.
- La representación relacional actual sigue siendo un starter:
  - sí hay enumeración y ventanas causales
  - no está implementado de forma completa el esquema `{=,<,>}` prometido en el texto
- Por eso, hoy DKANA está desbloqueado a nivel de entorno y datos, pero no cerrado como arquitectura benchmarkeada.

# Decisión congelada del proyecto

- El paper principal no se centra en novedad arquitectónica.
- El benchmark RL principal sigue separado del carril DKANA.
- A David le dejamos el entorno *pristine*:
  - contrato estable
  - export reproducible
  - contexto suficiente
- Si DKANA luego supera al benchmark congelado, entra como comparación real; si no, queda como future work.

# Qué necesitamos de Garrido

- Validar el framing:
  - DES base alineado con la tesis
  - RL como capa de control adaptativo
  - DKANA como carril paralelo, no como bloqueo del paper
- Confirmar que esta separación metodológica le parece defendible para journal.
- Priorizar la ruta de submission:
  - `IJPR` como target principal
  - `IEEE TAI` como alternativa rápida

# Cierre

- El cuello de botella ya no es “hacer funcionar el entorno”.
- El entorno ya funciona para RL y ya quedó suficiente para que David implemente DKANA.
- Nuestro foco ahora debe ser:
  - resultados comparativos limpios
  - escritura del manuscrito
  - benchmark defendible
