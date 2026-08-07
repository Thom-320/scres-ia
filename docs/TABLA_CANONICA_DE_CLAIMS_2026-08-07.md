# Tabla canónica de claims — 7 de agosto de 2026

Fuente única de verdad para lo que se puede decir y con qué número. Nace de cinco auditorías
externas que revisaron `a13ae22` y encontraron el resumen mezclando generaciones de resultados.
**Si un número no está aquí, no se cita.**

Rama científica: `codex/expanded-contract-comparators-v2`. `main` está en `89acc81` (28-jul) y
**no es fuente vigente**: la rama va 786 commits por delante y 8 por detrás.

## 1. Métrica primaria y normalizador — congelados

**Primaria: `auc_regret_norm` bajo el normalizador de PREFIJO.** Razón registrada en el artefacto:
`runs_to_within_1pct` imputa `budget+1` y está censurada a tasas muy distintas por brazo, así que
su media no es comparable entre brazos.

**Panel canónico** (`results/garrido_normaliser_audit_v3/result.json`, 6 contextos × 12 réplicas):

| brazo | AUC regret (prefijo) ↓ | corridas al 1 % (secundaria, censurada) |
|---|---:|---:|
| `neuron_memory` | **0,05203** | 7,08 |
| `neuron_reset` | 0,11274 | 12,92 |
| `ofat` | 0,10024 | 12,42 |
| `random` | 0,13851 | 19,54 |

| contraste | ΔAUC | LCB95 |
|---|---:|---:|
| memoria − reinicio | **+0,06070** | +0,04556 |
| memoria − OFAT | **+0,04821** | +0,03325 |

**RETIRADAS y prohibidas: `7,24 / 13,54 / 12,42` y `7,90 / 5,43`.** Vienen del normalizador
oráculo, que ve la superficie no ejecutada. El panel oráculo (`6,99 / 14,89 / 12,42`) se cita
**sólo** como sensibilidad y **siempre** con esa etiqueta. *Mi propio resumen de esta sesión mezcló
las corridas del oráculo con el AUC de prefijo; queda corregido aquí.*

## 2. Escalera vigente: `v5`, no `v2`

`results/search_ladder_v5/result.json` — 15 métodos + oráculo. `search_ladder_v2_ordered` queda
**`SUPERSEDED_FOR_CURRENT_CLAIMS_BY_V5`**.

| # | método | AUC ↓ | ¿conserva estado? |
|---:|---|---:|---|
| 1 | `ucb1_transfer` | 0,04502 | sí |
| 2 | `neuron_memory` | 0,05203 | sí |
| 3 | `ofat_transfer` | 0,06274 | sí |
| 4 | `lookahead_kg_transfer` | 0,08018 | sí |
| 5 | `gp_ei_transfer` | 0,08390 | sí |
| 6 | `thompson_transfer` | 0,08908 | sí |
| 7+ | todos los demás | ≥ 0,09655 | **no** |

**Los seis primeros son exactamente los seis que conservan estado.** Ése es el resultado.

Contrastes contra `neuron_memory` (positivo = la neurona gana):

* `ucb1_transfer`: **−0,00701 [−0,02444, +0,01408]** → **empate**. No se dice que UCB1 gana ni que
  la neurona gana.
* `ofat_transfer`: **+0,01071 [+3,56e−05, +0,02171]** → excluye cero **por 3,6e−05**. Es una
  diferencia real y **demasiado pequeña para importar**; se reporta con las dos cosas dichas.
* `neuron_reset`: +0,06070 [+0,04556, +0,07997].

**Alcance del artefacto:** `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION`, `run_role =
CACHE_ANALYSIS`, semillas `5300001–012` reutilizadas. **No es confirmación prospectiva.**

## 3. La única confirmación prospectiva no es neuronal

`results/grid_transfer_confirmation_v2/result.json`, bloque virgen `8200001–060`, rejilla
288 → 4.608 configuraciones. `GRID_TRANSFER_CONFIRMED__UCB1`, `transfers = {ucb1: true, neuron:
false, gp: false, ofat: false}`.

| familia | vs arranque frío | vs replay marginal *state-blind* |
|---|---|---|
| **UCB1** | +0,05744 [+0,04989] | **+0,03073 [+0,01990]** ✅ |
| neurona | +0,05439 [+0,04290] | **−0,01178 [−0,01849, −0,00484]** ❌ |
| GP-EI | +0,01433 | −0,02160 ❌ |
| OFAT | +0,01422 | −0,02467 ❌ |

**La neurona bate al arranque en frío pero pierde contra el replay de sus propias marginales.**
El único portador que sobrevive al falsador exigente es un bandido. Esto va en el centro del paper.

## 4. Fig. 5: la red dibujada no tiene nada que aprender

`results/garrido_fig5_surrogate/result.json` y `results/garrido_wrap_q1/result.json`
(`status: IDENTITY_NOT_A_LEARNING_TASK`), cinco falsadores pasan:
`f1_task_A_is_an_identity` con `max_abs_identity_error = 3,22e−15`, coeficientes `Re_FRt ≈ 1`,
`Re_RPj ≈ 1`, y **tres de las cinco columnas de drivers idénticamente cero**
(`Re_APj`, `Re_DPj_RPj`, `not_in_his_ReT`).

Lectura para Garrido, con cuidado: **la agregación literal de su Fig. 5 ya está incorporada
algebraicamente en las entradas; no es una tarea de aprendizaje.** El aprendizaje real está en
decidir *qué configuración ejecutar después*.

## 5. Curvatura y ruido — trazabilidad completa

`results/headroom/buffer_prediction_premium/result.json`
(`NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE`):

* **curvatura = 0,076259** — campo almacenado `profile_curvature.mean_one_minus_linear_r2`;
* **ruido = 0,317394** — **derivado**, `1 − held_out_r2_mean.linear` con `linear = 0,682606`;
* R² retenido: lineal 0,6826 · KAN 0,7163 · **backprop 0,5548 (peor que una recta)** · constante −0,0034.

Se cita siempre con la derivación explícita. La objeción externa era de presentación, no de hecho.

## 6. Añadido hoy, posterior a las cinco auditorías

* **`H1` del borrador, redacción original: SOSTENIDA.** TTR restringido: híbrido 75,7 h · reinicio
  149,7 h · estático 201,7 h; ventaja +126,0 h [+98,4, +154,5], Holm p<0,0001, 960 celdas, n=120.
  Mecanismo: **absorción** (875/960 vs 755/960), no velocidad de restauración.
  `results/manuscript/h1_h3_originales_v3/result.json`.
* **`H3` original: NO SOSTENIDA**, con el signo del lado contrario y el 78,7 % de las celdas
  desplegando configuraciones distintas — ya no es «sin estimando», es «sin efecto».
* **Hueco A2 acotado:** `f4` re-evaluó 24 celdas selladas bajo la física de hoy con
  `max_abs_delta = 0,0` exacto. La deriva de `supply_chain.py` no alcanza al meta-aprendiz.
* **Defecto que ninguna de las cinco auditorías vio:** `run_architecture_bakeoff_v1.py`
  **nunca selló nada**. Varias de ellas construyeron tablas sobre sus cifras. Sellado externo de
  sólo-contenido en `results/*/sealed_record.json`; **la procedencia no es certificable hacia atrás**.

## 7. Prohibido decir

| frase | por qué |
|---|---|
| «Q1 y Q2 están cerradas» | están respondidas **dentro del contrato de optimización por simulación ensayado** |
| «la neurona gana» / «hay prima neural» | empate con UCB1; pierde el falsador de transferencia |
| «las redes son la familia que imita SCL» | refutado en los contratos medidos |
| «aprendizaje organizacional» / «la cadena aprende» | es un lazo **externo entre corridas**, no control dentro del episodio |
| «KAN es mejor política» | IC cruza cero y es 4,1× más lenta por decisión |
| «C1 está confirmado» | desarrollo sobre bloque usado, sin `worst_product_fill` |
| «está en `main`» | `main` está en `89acc81`, 28-jul |
| las cifras retiradas del §1 | normalizador con fuga |

## 8. Decisiones

```
MANUSCRITO_Q1_Q2            GO
Q1_Q2_CERRADAS              NO — acotadas al contrato medido
C1_BLOQUE_VIRGEN            NO-GO hasta A1 (worst_product_fill) y el resto del registro de huecos
MECANISMO                   estado de búsqueda retenido, no arquitectura
INTEGRACION                 outer-loop simulation optimization acoplado al DES
```

Acepto el NO-GO de C1: coincide con el hueco A1 de mi propio registro, y nunca lo autoricé —
pedí tu firma. El manuscrito no debe quedar rehén de C1: son dos estimandos distintos.
