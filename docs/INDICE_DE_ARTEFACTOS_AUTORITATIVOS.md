# Índice de artefactos autoritativos — 7 de agosto de 2026

Un año de campaña dejó **112 directorios en `results/`, ~70 ramas y 441 documentos**. Este índice
existe para que nadie tenga que recordar de memoria cuál es el vigente. **Regla del repositorio:
un resultado retirado se conserva y se etiqueta; no se borra.**

Para *qué se puede afirmar* con cada número, la autoridad es
`docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`. Este documento sólo dice **cuál fichero manda**.

---

## 1. Grado de evidencia — la distinción que decide todo

| grado | qué significa | cuántos hay |
|---|---|---|
| **CONFIRMACIÓN** | prospectivo, bloque de semillas virgen, contrato preregistrado | **2** |
| desarrollo | cintas quemadas o bloque ya abierto; sin adjudicación | ~30 citables |
| diagnóstico | propiedad del instrumento, sin contraste | ~15 |

### Las dos confirmaciones, y son sólo dos

| artefacto | qué confirma | bloque | veredicto |
|---|---|---|---|
| `results/grid_transfer_confirmation_v2/result.json` | transferencia 288→4.608 configuraciones; **sólo UCB1** bate al replay marginal *state-blind*; **la neurona pierde** (−0,01178 [−0,01849, −0,00484]) | 8200001–060, virgen | `GRID_TRANSFER_CONFIRMED__UCB1` |
| `results/garrido_h2_h3_confirmation_v1/result.json` | el DES reconstruido reproduce H2 (buffer) y H3 (turno) de Garrido en R1r/R2r/R3; **seis paneles**, Holm p 4,8e−17…1,9e−15 | 12 raíces vírgenes 96111336–97836128 | `CONFIRM_H2_H3_ALL_SIX_PANELS` |

Rescatada hoy desde `codex/paper-b-retained-v5` (`docs/RESCATE_CONFIRMACION_H2_H3_2026-08-07.md`).
Su frontera viaja siempre con ella: *"it does not establish learner, feedback, or architectural
value."*

---

## 2. Familias con miembro autoritativo

| familia | **AUTORITATIVO** | superados, que se conservan |
|---|---|---|
| escalera de búsqueda | **`search_ladder_v5/`** | `search_ladder`, `_v2`, `_ordered`, `_v3`, `_v4`, `_v2_ordered` (= `SUPERSEDED_..._BY_V5`) |
| transferencia de rejilla | **`grid_transfer_confirmation_v2/`** | `grid_transfer`, `_v2` (= `SUPERSEDED_ORDERING`), `_ordered_v1` |
| auditoría de normalizador | **`garrido_normaliser_audit_v3/`** | `_smoke`, base, `_v2` |
| meta-aprendiz / H3′ | **el par contratado** `garrido_meta_learner_h3power_h3_contract_local_v2/` + `..._vps_v2/` | otros 8; `garrido_meta_learner` v1 = `RETIRED_LEAKAGE` |
| transformación monótona | **`monotone_transform_family_v4/`** | `_ceiling`, `_v2`, `_v3` (su `f6` falló) |
| barrido ESTAR | **`headroom/estar_capacity_sweep_v2_replay_20260805/`** | `_INFLATED_H_REGIME` (**se conserva a propósito** para que la corrección sea auditable), base, `v1_1` |
| bake-off arquitectura | `architecture_bakeoff/` + `surrogate_architecture_bakeoff/` | `_200k` **no superado, sí sin sellar** al ejecutarse |
| H1/H3 del borrador | **`manuscript/h1_h3_originales_v3/`** (original) + `h1_h3_v2_1/` (H1′) | `h1_h3_v1` (halted), `h1_h3_v2` (pre-auditoría) |
| G3-obs | `headroom/g3_obs_conversion_v2_replay_20260803/` | `_smoke`, base, pre-replay |
| gates de superficie | `surface_gates/` (A7) + `surface_gates_extended_v2/` | `_v2`, `_extended` |
| G3c preflight | `headroom/g3c_preflight_grid_v2/` | `g3c_preflight_burned` |
| v0 recovery | `garrido_v0_recovery_gate_v2/`, `garrido_v0_surface_gates_holdout/` | `_gate_v1` (sin result), `_surface_gates_v1` |

---

## 3. Lo mejor que obtuvimos, ordenado por lo que permite afirmar

1. **Sólo la memoria estructurada transfiere.** `grid_transfer_confirmation_v2` — el único
   resultado confirmado sobre aprendizaje, y va en contra de la neurona.
2. **El DES reproduce a Garrido prospectivamente.** `garrido_h2_h3_confirmation_v1` — seis
   paneles, raíces vírgenes.
3. **La retención domina la arquitectura.** `search_ladder_v5` — los seis primeros de quince son
   exactamente los seis que conservan estado.
4. **La Fig. 5 tal como está dibujada es una identidad algebraica.** `garrido_fig5_surrogate` +
   `garrido_wrap_q1` — error 3,22e−15, tres de cinco columnas idénticamente cero.
5. **El efecto Alzheimer tiene precio, y sobrevive a un normalizador honesto.**
   `garrido_normaliser_audit_v3` — memoria − reinicio +0,06070 [+0,04556].
6. **H1 del borrador se sostiene, por absorción.** `manuscript/h1_h3_originales_v3` — +126,0 h
   [+98,4, +154,5]; absorbe 875/960 choques contra 755/960.
7. **NUEVO 7-ago: la lane GSA califica bajo el objetivo declarado.**
   `gsa_resilience_only/` — H_obs positivo con cero excluido en tres bloques, η 0,78–0,91, y
   **+0,069…+0,073 sobre un placebo desinformado** que la corrida histórica no tenía.
8. **Ajustar no es buscar.** `surrogate_architecture_bakeoff` — KAN ajusta mejor y busca peor.
9. **La frontera:** `PAPER2_EXHAUSTION_CERTIFICATE` + `headroom/cobb_douglas_v1` (`H_regime` = 0) +
   `track_b_nonneural` (el suelo constante).

---

## 4. Lo que NO se puede citar

| artefacto / cifra | por qué |
|---|---|
| `7,24 / 13,54 / 12,42` y `7,90 / 5,43` | normalizador con fuga; sustituidos por el panel de prefijo |
| `results/architecture_bakeoff*` como «sellado» | el runner nunca selló; sello externo sólo de contenido desde hoy |
| `results/k3/confirmation.json` | auto-retractado (`effective_verdict: RETRACT_...`) |
| `q_r1/successor_confirmation_v1` (otra rama) | STOP compuesto del Programa Q sobre `worst_product_fill` |
| prima neural de `track_b_v1` | desarrollo, sin intervalo, y dentro de ±2,4 de ruido del arnés |
| «Q1/Q2 cerradas», «aprendizaje organizacional» | exceden el alcance medido |

---

## 5. Custodia — el recurso más escaso del proyecto

`research/seed_custody_registry.json`: `NO_NEW_SEEDS_AUTHORIZED`, `new_seed_opening: false`.

> **Queda UN bloque virgen en todo el proyecto**: `g3a_v2_development`, **7700001–7700120**,
> `RESERVED_NOT_OPENED`, condicionado a `submission_a_receipt_required_before_g3a_open`.

Quemados por confirmación: 8200001–060, 7490001–256. En cuarentena (intento sin artefacto sellado,
no reutilizables como vírgenes): 8100001–060, 7900001–140.

---

## 6. Procedencia

Rama científica: **`codex/expanded-contract-comparators-v2`**. `main` está en `89acc81` (28-jul),
**790 commits por detrás**, y **no es fuente vigente**.

Hueco A2 (deriva de `supply_chain.py` desde el manifiesto del 14-jul) sigue abierto, pero se ha
estrechado con evidencia en dos familias, ambas reproduciendo **exactamente**:

| familia | comprobación | resultado |
|---|---|---|
| meta-aprendiz | `f4` re-evaluó 24 celdas selladas | `max_abs_delta = 0,0` |
| GSA | `f1` recalculó H_PI en el θ localizado | `0,014446048488184385` idéntico |
