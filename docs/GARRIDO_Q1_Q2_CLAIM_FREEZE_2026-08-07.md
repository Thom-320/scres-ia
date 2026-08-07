# Claim freeze — respuesta a las dos preguntas de Garrido (2024)

**Fecha:** 2026-08-07 · **Rama:** `codex/expanded-contract-comparators-v2` · **Base:** `a13ae22`
**Estado de `main`:** `89acc813` (2026-07-28) — **790 commits detrás, 8 delante. No es fuente vigente.**

Este documento congela **qué podemos afirmar hoy, con qué métrica, en qué contrato y qué no**.
Sustituye a cualquier resumen anterior. Toda cifra lleva ruta y SHA-256 (16 caracteres) del
artefacto. Una cifra sin fila en las tablas de abajo **no está congelada y no debe circular**.

---

## 0 · Artefactos citables

| # | ruta | sha256[:16] | `run_role` / `scope` |
|---|---|---|---|
| A1 | `results/search_ladder_v5/result.json` | `f648a1da5aefaf2f` | `CACHE_ANALYSIS` / `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION` |
| A2 | `results/grid_transfer_confirmation_v2/result.json` | `7bc33823ccd90b5e` | **`CONFIRMATION` / `CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER`** |
| A3 | `results/garrido_normaliser_audit_v3/result.json` | `fd617753949947e6` | `BURNED_REPLAY_AUDIT` / `DEVELOPMENT_REPLAY…NO_ADJUDICATION` |
| A4 | `results/garrido_fig5_surrogate/result.json` | `58d4c8a071cec86a` | `DEVELOPMENT_FIG5_SURROGATE` |
| A5 | `results/surrogate_architecture_bakeoff/result.json` | `f96e5b6ff0489932` | `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION` |
| A6 | `results/headroom/buffer_prediction_premium/result.json` | `54bf5fa2594262bd` | desarrollo, 6 falsadores, CV agrupada por semilla |
| A7 | `results/surface_gates/result.json` | `954ac48301ff1234` | referencia de A1 |
| A8 | `results/track_b_nonneural/result.json` | `12e5f2562684655a` | `DEVELOPMENT_NO_CUSTODY_SEEDS_NO_ADJUDICATION` |
| A9 | `results/architecture_bakeoff/result.json` | `d641ab0a54ecf2c8` | `DEVELOPMENT_ONLY_NO_VIRGIN_SEEDS_NO_LEARNER_AUTHORISATION` |

**Superseded:** `results/search_ladder_v2_ordered/result.json` (`2d900bc27649b4b4`) →
`SUPERSEDED_FOR_CURRENT_CLAIMS_BY_A1_AND_A3`. Sus rankings siguen siendo válidos; se sustituye
porque A1 añade Knowledge Gradient y Thompson y reordena la lectura.

---

## 1 · Q1 — ¿qué categoría de algoritmos imita mejor el atributo SCL?

### Claim congelado

> **En el contrato de optimización-por-simulación evaluado, el componente que reproduce el atributo
> *history-dependent* del SCL no es una familia de aproximadores, sino la conservación de estado de
> búsqueda entre contextos. Igualada la memoria, no se identifica ventaja neuronal sobre el mejor
> método clásico retenido. Bajo el control de replay marginal, el único carrier que transfiere
> estructura —y no meramente su patrón de visitas— es un bandit de nivel de factor.**

Etiqueta: `Q1_ANSWERED_WITHIN_TESTED_SIMULATION_OPTIMISATION_CONTRACT`
**No** `Q1_UNIVERSALLY_CLOSED`.

### Evidencia primaria — A1, AUC de regret normalizado (menor es mejor)

Presupuesto 24 evaluaciones, 6 contextos (`R1r`, `R2r`, `R1r+R2r` y sus versiones `|esc`),
12 semillas `5300001–5300012`, replay de `garrido_q2_des288`.

| # | brazo | AUC | ¿retiene estado? |
|---|---|---:|---|
| 1 | `ucb1_transfer` | 0,04502 | sí |
| 2 | `neuron_memory` | 0,05203 | sí |
| 3 | `ofat_transfer` | 0,06274 | sí |
| 4 | `lookahead_kg_transfer` | 0,08018 | sí |
| 5 | `gp_ei_transfer` | 0,08390 | sí |
| 6 | `thompson_transfer` | 0,08908 | sí |
| 7 | `ucb1` | 0,09655 | no |
| 8 | `ofat` | 0,10024 | no |
| 9 | `gp_ei` | 0,10661 | no |
| 10 | `thompson` | 0,10893 | no |
| 11 | `lhs_local` | 0,10949 | no |
| 12 | `neuron_reset` | 0,11274 | no |
| 13 | `lookahead_kg` | 0,11479 | no |
| 14 | `random` | 0,13979 | no |
| 15 | `annealing` | 0,17420 | no |

**Los seis brazos declarados `memory_arms` en A1 ocupan exactamente los seis primeros puestos.**
`neuron_reset` cae al puesto 12. Ese es el resultado.

### Contrastes pareados publicados (A1/`search_ladder_v2_ordered`, `vs_neuron_memory`, n=12)

Positivo = la neurona con memoria es mejor.

| comparador | media | LCB95 | UCB95 | lectura |
|---|---:|---:|---:|---|
| `ucb1_transfer` | **−0,00701** | −0,02434 | +0,01408 | **empate; punto a favor de UCB1** |
| `ofat_transfer` | **+0,01071** | −0,0000276 | +0,02130 | **empate; el intervalo incluye cero** |
| `gp_ei_transfer` | +0,03187 | +0,01963 | +0,04190 | la neurona gana |
| `neuron_reset` | +0,06070 | +0,04574 | +0,07986 | efecto de memoria propio |
| `ucb1` | +0,04452 | +0,03516 | +0,05442 | la neurona gana |
| `ofat` | +0,04821 | +0,03265 | +0,06341 | la neurona gana |
| `random` | +0,08776 | +0,07015 | +0,10408 | la neurona gana |

**Lectura obligatoria:** `ofat_transfer` con LCB95 `−2,76e-05` **incluye cero**. Es un empate, no
una superioridad marginal. La neurona **no supera a dos de las tres familias clásicas** cuando
éstas reciben memoria comparable.

### Evidencia confirmatoria — A2, transferencia 288 → 4.608 configuraciones

Único artefacto de esta línea con `run_role: CONFIRMATION` sobre bloque reservado. n=60.
Positivo = la versión con transferencia es mejor.

| familia | vs cold start (media, LCB95) | **vs replay marginal state-blind** (media, LCB95, UCB95) |
|---|---|---|
| **`ucb1`** | +0,05744 · +0,04989 | **+0,03073 · +0,01990 · +0,04256** ✅ |
| `neuron` | +0,05439 · +0,04290 | **−0,01178 · −0,01849 · −0,00484** ❌ |
| `gp` | +0,01433 · +0,00879 | −0,02160 · −0,03051 · −0,01227 ❌ |
| `ofat` | +0,01422 · +0,00800 | −0,02467 · −0,03258 · −0,01666 ❌ |

Veredicto congelado del artefacto: **`GRID_TRANSFER_CONFIRMED__UCB1`**.

Este es el falsador que separa *«transferí estructura de la superficie»* de *«revisité
configuraciones que ya me habían funcionado»*. **UCB1 es el único que lo pasa.** La neurona bate
al arranque en frío pero pierde contra su propio replay marginal, con el intervalo entero del lado
desfavorable.

### Restricción de alcance que debe acompañar siempre a Q1

`H_regime = 0,003802`, LCB95 `1,08e-16`, UCB95 `0,014413` (A7), contra un gate preregistrado
de `0,05`. **El gate falla**; A1 hereda `surface_gates_v1 → NON_SEPARABLE_BUT_CONTEXT_INVARIANT`.

Consecuencia, y es la distinción central del paper:

> **valor de transferencia de búsqueda ≠ valor de adaptación operacional.**

La memoria evita **re-descubrir** una postura casi común entre contextos. **No** está demostrado
que aprenda a adaptar materialmente la política a cada régimen.

---

## 2 · Q2 — ¿cómo se integra en la estructura interna de un DES?

### Claim congelado

> **Como un lazo externo persistente de optimización por simulación: cada ejecución del DES produce
> un resultado observado; ese resultado actualiza un estado retenido `L_k`; el estado selecciona la
> configuración `x_{k+1}` de la siguiente ejecución. La transferencia de ese estado a una rejilla
> 16× mayor está confirmada prospectivamente para un método no neuronal.**

```
x_k → DES(x_k) → y_k, d_1..d_4 → actualización de L_k → x_{k+1} → DES(x_{k+1}) → …
```

Etiqueta: `Q2_OUTER_LOOP_INTEGRATION_IMPLEMENTED_AND_GRID_TRANSFER_CONFIRMED_FOR_UCB1`.

Esto cierra los nodos ③↔⑧ de la Fig. 2 **entre corridas**, que es una lectura legítima del texto
de Garrido. Operacionaliza `R_t = f(S_t, D_t, L_{t-1})` con `L_{t-1}` = estado de búsqueda retenido.

### Vocabulario permitido / prohibido

| ✅ usar | ❌ no usar |
|---|---|
| «coupled to the DES» | «embedded inside the DES transition kernel» |
| «outer-loop sequential simulation optimization» | «closed-loop adaptive control within the episode» |
| «persistent search state across runs» | «the supply chain learns» |
| «`L_{k}` operacionalizado como estado de búsqueda» | «organizational learning demonstrated» |

El carrier vive en el loop experimental, no en el kernel de eventos. Un revisor abrirá el código.

### El diagnóstico de la Fig. 5 — A4

Falsador `f1_task_A_is_an_identity`: R² = 1,0; error máximo de identidad **3,22e-15**; coeficientes
identificados `Re_RPj` = 0,999999999999968 y `Re_FRt` = 0,999999999999999; columnas degeneradas
todo-cero `Re_APj`, `Re_DPj_RPj`, `not_in_his_ReT`. Status: **`IDENTITY_NOT_A_LEARNING_TASK`**.

Formulación para Garrido, con cuidado:

> La neurona **tal como está dibujada en la Fig. 5** no necesita descubrir la función de agregación,
> porque ReT ya es exactamente la suma de las contribuciones que se le entregan como dendritas. La
> tarea de aprendizaje genuina no es predecir ReT: es **decidir qué configuración ejecutar después**.

Esto no resta valor a su propuesta. Reubica dónde está el aprendizaje.

---

## 3 · Resultado de arquitectura — KAN

### Como surrogate de búsqueda (A5, parámetros emparejados: KAN 532 · MLP 529)

| brazo | AUC regret | % del techo |
|---|---:|---:|
| `neuron_5p` (5 parámetros) | **0,05203** | 99,499 |
| `mlp_matched` | 0,08853 | 99,630 |
| `kan` | 0,09889 | 98,539 |
| `spline_poly` | 0,09754 | 99,215 |
| `gbt` | 0,10832 | 97,750 |
| `gp_matern` | 0,11379 | 96,995 |
| `random` | 0,13979 | 93,381 |

`kan_minus_matched_mlp` = **+0,01037**, IC95 **[+0,00302, +0,01893]**, p = 0,0012. Menor es mejor,
así que el intervalo está **enteramente en contra de la KAN**. Status: `KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP`.

### Como política de control (A9)

`KAN_minus_MLP` = −0,475, IC95 [−1,548, +0,598] — **sin separación**. KAN ≈ 4,1× más lenta por
decisión en el host medido.

### Claim congelado

> **La calidad de ajuste de un surrogate no determina su calidad de decisión secuencial.** KAN
> ajusta mejor ciertas superficies y busca peor que un MLP de parámetros emparejados; una unidad de
> cinco parámetros es el mejor buscador del bake-off.

**Prohibido:** «KAN es superior», «KAN es interpretable» como claim establecido (una sola partición,
sin CV, sin estabilidad de formas entre folds; las curvas son cortes de respuesta, no funciones de
arista internas).

---

## 4 · Condición cuantitativa de la prima neural — A6

`results/headroom/buffer_prediction_premium/result.json`, sello `54bf5fa2594262bd`, 1.530 episodios,
17 niveles de buffer × 3 familias × 3 escalados × 10 semillas, CV agrupada por semilla, 6 falsadores.

| modelo | R² held-out | vs lineal | IC95 |
|---|---:|---:|---|
| constante | −0,0034 | | |
| **lineal** | **0,6826** | — | — |
| backprop MLP | 0,5548 | **−0,1278** | [−0,3157, +0,0601] |
| KAN | 0,7163 | +0,0337 | [−0,0787, +0,1462] |

Ninguno alcanza el SESOI preregistrado de 0,05 y **el MLP es peor que una recta**. Curvatura
recomputada in situ por `f1`: **0,0763**; varianza episódica inexplicada: **0,3174**.

> **Una prima neural requiere que la curvatura de la superficie supere el ruido que la oculta.**

Cierra la objeción «tu superficie era demasiado fácil»: ésta *es* la curvada.

---

## 5 · Cifras RETIRADAS — no deben reaparecer

| cifra retirada | por qué | qué usar en su lugar |
|---|---|---|
| **7,24 · 13,54 · 12,42** corridas al óptimo | procedían de un normalizador que veía la superficie no ejecutada | ver abajo |
| «memoria vale +0,0515 a UCB1 con LCB95>0» | conflación del contraste neurona-vs-`ucb1_transfer` con el contraste interno UCB1 transfer-vs-reset; **no existe artefacto con CI pareado para el efecto de memoria de UCB1** | diferencia de medias de brazo 0,09655 → 0,04502, declarada como tal, **sin LCB** |
| ranking por «% del techo» de A1 (99,70 `lookahead_kg_transfer`, etc.) | **A1 no contiene `percent_of_ceiling`**; esas cifras no son trazables a la escalera v5 | omitir, o citar A5 que sí lo tiene, para su propio conjunto de brazos |
| «Q1 y Q2 cerradas» | excede el alcance medido | las etiquetas de §1 y §2 |
| «el estado está en la rama y en `main`» | `main` = `89acc813`, 790 commits detrás | «vive en `codex/expanded-contract-comparators-v2@a13ae22`» |

### Cifras correctas de `runs_to_within_1pct` (A3), como **secundario censurado**

| normalizador | `neuron_memory` | `neuron_reset` | `ofat` | tasa de censura (mem/reset/ofat/random) |
|---|---:|---:|---:|---|
| oracle | 6,99 | 14,89 | 12,42 | 0,069 / 0,153 / 0,222 / 0,611 |
| **prefix (honesto)** | **7,08** | **12,92** | **12,42** | 0,056 / 0,153 / 0,222 / 0,611 |

`primary_rationale` de A3, literal: *`runs_to_within_1pct` imputa presupuesto+1 y está censurada a
tasas muy distintas por brazo, así que su media no es comparable entre brazos.*

**Métrica primaria = AUC de regret normalizado.** Panel prefix, n=12:

| contraste | media | LCB95 | UCB95 |
|---|---:|---:|---:|
| memoria vs reset | **+0,06070** | **+0,04556** | +0,08020 |
| memoria vs OFAT | **+0,04821** | **+0,03325** | +0,06320 |
| memoria vs random | +0,08647 | +0,06729 | +0,10521 |

Traducción secundaria admisible, siempre etiquetada como censurada: ~**5,83** corridas ahorradas
frente a reset y ~**5,33** frente a OFAT. Lo estable no es el decimal: es `memoria ≪ reset`.

Status de A3: `ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER`.

---

## 6 · Track B / C1 — residual, no claim

A8: `NEURAL_PREMIUM_LIKELY_IN_TRACK_B`, 4/4 falsadores, **sobre el bloque de evaluación que las
redes ya habían usado**; `scope: DEVELOPMENT_NO_CUSTODY_SEEDS_NO_ADJUDICATION`.

| brazo | score | vs mejor constante |
|---|---:|---:|
| `trained_mlp` | 98,743 | +2,176 |
| `trained_kan` | 98,516 | +1,949 |
| `trained_dmlpa` | 98,004 | +1,437 |
| `threshold_rule` | 97,142 | +0,575 |
| `constant_best` | 96,567 | — |

Las tres redes caben en 0,74 entre sí: **elegir arquitectura no compra nada; usar una compra poco.**

### `C1_VIRGIN_BLOCK = NO-GO`

El bloqueador no es estadístico. Es que **el estimando no está definido**: el preregistro nombra
`worst_product_fill` guardarraíl bloqueante y el runner sólo persistió `flow_fill_rate`, un agregado
que no ve un producto abandonado (hueco A1 de `REGISTRO_DE_HUECOS_2026-08-07.md`). Con
`ret_excel` aún de default —métrica que **premia el abandono**— una ganancia agregada de +2 no se
distingue de sacrificar un producto.

Precondiciones antes de firmar, en orden:

```
C1-A  worst_product_fill persistido y bloqueante
C1-B  endpoint obligatorio; eliminado el default silencioso ret_excel
C1-C  constante y umbral congelados en bloque disjunto
C1-D  física congelada y hash único (hueco A2: supply_chain.py derivó 12+ commits)
C1-E  hiperparámetros idénticos entre arquitecturas (hueco B2)
C1-F  suite verde o waiver auditado por fallo (hueco B3: 20 en rojo)
C1-G  registro de semillas reconciliado
C1-H  endpoint, incumbente, SESOI y stop rule congelados (hueco A3)
C1-I  una sola apertura
```

Si tras C1-A el residual desaparece, se habrá ahorrado un bloque confirmatorio. Si sobrevive,
C1 merece confirmación **como estudio prospectivo separado** — estimando distinto del de este
documento: aquí se mide retención entre búsquedas de configuración; allí, prima neural de una
política dentro del episodio. No comparten abstract.

---

## 7 · Escalera de claims

### Soportado, confirmación prospectiva
1. `ucb1_transfer` transfiere de 288 a 4.608 configuraciones batiendo cold start **y** su replay marginal state-blind (A2).

### Soportado, desarrollo sobre tapes quemados
2. Los seis brazos con memoria dominan la escalera de 15 métodos (A1).
3. Retener estado mejora a la propia neurona: +0,06070, LCB95 +0,04556 (A3).
4. La neurona no supera de forma identificable a `ucb1_transfer` ni a `ofat_transfer` (A1).
5. KAN ajusta mejor y busca peor que un MLP emparejado (A5).
6. ReT es una identidad de sus drivers; la Fig. 5 literal no es tarea de aprendizaje (A4).
7. Una prima neural exige curvatura por encima del ruido; medido 0,0763 vs 0,3174 (A6).
8. El efecto sobrevive a un normalizador que no ve la superficie no ejecutada (A3).

### No soportado — prohibido afirmar
9. Que las redes sean la familia que mejor imita SCL.
10. Que exista prima neural confirmada.
11. Que se haya demostrado aprendizaje organizacional.
12. Que la cadena física aprenda dentro de una campaña.
13. Que la memoria mejore materialmente la adaptación por régimen (`H_regime` falla el gate).
14. Que C1 esté confirmado.
15. Que Q1/Q2 estén cerradas en sentido confirmatorio o universal.
16. Que estos resultados estén en `main`.

---

## 8 · Regla de custodia de este documento

Este fichero es **datado y no se edita en sitio**. Una corrección se emite como
`GARRIDO_Q1_Q2_CLAIM_FREEZE_<fecha>.md` sucesor, declarando qué fila supersede y por qué.

Una cifra que no aparezca aquí y no traiga ruta + SHA no entra en el manuscrito, en una diapositiva,
ni en una reunión con Garrido.
