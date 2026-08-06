# Preregistro — Paso 3 de Garrido: MPC y DDMRP sobre el contrato expandido de buffers

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_expanded_contract_comparators_v2.py`
(instrumento correctivo, ya construido y con preflight en verde).
Kernels: `scripts/build_kaggle_kernels_step3.py`. Semillas: **bloque nuevo `1.420.001+`**, libre en
custodia, declarado aquí.

## 1. Por qué éste y no otro entorno

De todo lo que tenemos, **éste es el único sitio donde el residual neural está sin definir** en vez
de medido en cero:

| entorno | headroom medido | estado |
|---|---|---|
| Program O (contención no fungible) | `H_PI` **0,1515** [LCB 0,1156], nulo fungible = 0 | **CERRADO**; reabrir exige física nueva de Garrido |
| **contrato expandido de buffers** | los derechos mueven ReT **+11 % a +25 %** (H2/H3) | **residual SIN DEFINIR** |
| `track_b_v1` | todo empata en ~97/104, recompensa saturada | sin prima, medido |
| superficie 288 / 4.608 | `H_regime` 0 / 0,0195 | cerrado |
| envolvente nativa de la tesis | ~1e-4 | certificado de agotamiento |

Y es **el diseño que pidió Garrido** el 28 de julio —baseline → MPC original → **MPC expandido** →
KAN— con DDMRP nombrado por él. Su propio docstring fija por qué el paso 3 va antes que cualquier
red: **el residual se define contra el mejor controlador estructurado.**

## 2. Lo que este instrumento repara, defecto por defecto

La v1 quedó reclasificada (`docs/EXPANDED_CONTRACT_COMPARATORS_RECLASSIFICATION_2026-07-29.md`) con
`must_not_be_claimed: ["garrido step 3 expanded mpc", "ddmrp defeated", "no dynamic value",
"neural residual closed"]`. La v2 repara los seis:

| defecto v1 | reparación v2 |
|---|---|
| el MPC nunca leía `sim` y planificaba sobre semillas ajenas | **replica el prefijo realizado y debe casar un hash de estado congelado** |
| sólo 6 de 216 posturas | **enumera las 216** |
| dominios de acción distintos | **DDMRP proyectado al mismo dominio 6³** |
| DDMRP estilizado | ADU con ventana, posición de flujo neto, zonas |
| «la planitud de las medias acota el valor adaptativo» *(falacia mía)* | se mide el valor condicionado al estado; no se argumenta |
| sólo medias por brazo | **filas por tape, trazas, valores de candidatos** |

Preflight verificado: `all_prefix_state_hashes_match: true`.

## 3. La métrica, y por qué NO es la que trae por defecto el runner

El runner arranca con `--metric ret_excel`. **Aquí se corre con `ret_excel_full_ledger`**, y no es
una preferencia: está **medido** que `ret_excel` premia el abandono —el reparto que lo maximiza
entrega 50 % de las raciones y el que lo minimiza entrega 80 %—, así que un controlador podría ganar
este experimento dejando de servir. `ret_excel_full_ledger` es **la misma fórmula de Garrido**
puntuando *todos* los pedidos generados, los no servidos a 0.

Se reportan además, como panel obligatorio y sin poder decidir: `ret_excel`, `cobb_douglas_index`,
`flow_fill_rate`, `worst_product_fill`.

## 4. Diseño y paralelismo declarado

`--phase full`: **12 tapes × 5 escenarios × 2 familias (R1r, R2r) × 216 posturas**, horizonte 52
semanas, época 4 semanas.

Se reparte en **cuatro shards** que difieren **sólo en la identidad de los tapes**:

| shard | familia | tapes | semillas |
|---|---|---|---|
| `s1` | R1r | 6 | 1.420.001+ |
| `s2` | R1r | 6 | 1.421.001+ |
| `s3` | R2r | 6 | 1.422.001+ |
| `s4` | R2r | 6 | 1.423.001+ |

**Dentro de cada shard los brazos comparten tape**, que es lo que hace válido el contraste pareado;
el análisis agrupado concatena las filas por tape. **Sharding no cambia el estimando** — y `f5` lo
comprueba comparando el resultado agrupado contra una corrida no fragmentada de un shard.

GPU **apagada a propósito**: el cuello es el DES en Python puro y los kernels GPU de Kaggle dan
menos vCPU.

## 5. Reglas de lectura, fijadas antes de mirar

Contraste pareado por tape contra **el mejor de las 216 posturas estáticas**, IC95 por bootstrap
sobre tapes.

* `replay_mpc_v2` **o** `ddmrp_projected_v2` con `LCB95 > 0` sobre el mejor estático →
  **`A_STRUCTURED_CONTROLLER_CONVERTS_THE_EXPANDED_RIGHTS`**. **Ése pasa a ser el incumbente que
  cualquier red tiene que batir**, y el residual neural queda definido como la distancia entre él y
  `greedy_pi_best_found_v2`.
* ninguno con `LCB95 > 0` → **`NO_STRUCTURED_CONTROLLER_CONVERTS`**. El incumbente sigue siendo la
  mejor postura estática, y el residual se define contra ella.
* **En ambos casos** se reporta `greedy_pi_best_found_v2` como **diagnóstico de mejor-encontrado,
  nunca como techo dinámico exacto**, porque no lo es.

**Y la regla que impide la trampa obvia:** ningún brazo se declara ganador si empeora
`worst_product_fill` más allá de su margen. Un controlador que gana abandonando no gana.

## 6. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_mpc_reads_the_state` | los hashes de prefijo deben casar en **todas** las ramas. **Es el defecto 1 de la v1**; falla si el MPC vuelve a planificar sobre futuros ajenos |
| `f2_all_216_postures_are_searched` | el incumbente estático debe salir de las 216, no de 6 |
| `f3_the_arms_share_the_action_domain` | DDMRP debe emitir objetivos **dentro** de 6³; falla si vuelve a inyectar material fuera del dominio |
| `f4_the_metric_is_not_ret_excel` | el endpoint decisor debe ser `ret_excel_full_ledger`. **Falla si alguien deja el default**, que es el que premia el abandono |
| `f5_sharding_does_not_change_the_estimand` | un shard corrido entero debe reproducir su parte del agrupado a 1e-9 |
| `f6_the_static_incumbent_is_inside_every_candidate_set` | si un controlador no puede ni empatar una acción que tenía disponible, es **fallo de búsqueda** y se reporta como tal, no como evidencia sobre el derecho |
| `f7_the_guardrail_can_veto` | debe existir al menos una configuración vetada por `worst_product_fill`; si nunca veta, es decoración |
| `f8_seeds_are_virgin_and_declared` | bloque `1.420.001+`, custodia central |

**Alcance:** desarrollo sobre semillas vírgenes declaradas. **No adjudica el paso 4** y no autoriza
ninguna red: fija el incumbente contra el que se medirá.
