# Preregistro — **H1 y H3 en su redacción ORIGINAL**, tercer intento

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_h1_h3_originales_v3.py`.
Contrato de lectura fijado aquí; nada de lo que sigue se decide después de ver un número.

## 1. Por qué existe, y qué NO repite

Ya están medidas y no se re-abren:

| | estado | artefacto |
|---|---|---|
| `H1′` servicio perdido acumulado | **SOSTENIDA** +61,3 M ración-hora, `LCB95` +14,4 M | `results/manuscript/h1_h3_v2_1/result.json` |
| `H3′` varianza del coste de búsqueda | **SOSTENIDA a n=120** mem−OFAT +16,22 `[+9,61, +22,74]` | `results/garrido_h3_merge_adjudication/result.json` |
| `H2` curva de aprendizaje | medida | `garrido_meta_learner_*` |
| `H4` dependencia de `L_{t−1}` | medida, +7,90 corridas | idem |

Las dos primadas son **reformulaciones declaradas**. `H1` y `H3` **en la redacción del borrador**
quedaron `NO EVALUABLES` el 2026-08-01 (`docs/RESULTADO_H1_H3_2026-08-01.md`) por dos bloqueos
concretos. Este preregistro existe porque **los dos bloqueos cambiaron de estado, y ninguno de los
dos cambió por relajar un criterio**:

| bloqueo del 1-ago | por qué ya no aplica igual |
|---|---|
| `f1` — híbrido y estático despliegan la **misma** configuración | se medía sobre la config **modal** de un bloque de **12** réplicas. En el bloque de **120** (`6.000.001–120`) los brazos despliegan **21 / 43 / 33 / 87** configuraciones distintas (memoria/reinicio/OFAT/azar). El estimando existe **por celda**, no por moda |
| `f3` — `system_ttr` censurado al **100 %** | el 2026-08-06 se construyó `supply_chain/garrido_v0_recovery.py` con `restricted_ttr = min(TTR, τ)` y un **placebo pareado** para decidir «absorbido». Se construyó para OTRA vía (v0) y **antes** de este preregistro; no es un instrumento ablandado a posteriori |

**La honestidad del punto 2 hay que decirla entera:** el documento del 1-ago puso «arreglar
`system_ttr`» como la **última** de tres opciones, precisamente porque cambiar un instrumento tras
verlo dar 1,000 es sospechoso. Lo que se usa aquí **no es** `system_ttr` arreglado: es un estimando
**acotado y distinto**, con un placebo que decide impacto por comparación incremental, escrito para
un lane ajeno. Aun así, **es una redefinición del endpoint y el manuscrito la presenta como tal.**

## 2. Los brazos, y de dónde salen

De la fusión de las dos rebanadas del bloque de potencia, ya abiertas y selladas contra
`docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md`:

* `results/garrido_meta_learner_h3power_h3_contract_local_v2/` — 90 réplicas, `6.000.001–090`
* `results/garrido_meta_learner_h3power_h3_contract_vps_v2/` — 30 réplicas, `6.000.091–120`

| brazo | estrategia | qué es |
|---|---|---|
| `hybrid` | `neuron_memory` | la neurona de la Fig. 5 que **conserva** `ρ` |
| `static` | `ofat` | **el diseño de la tesis de Garrido**, el comparador que importa |
| `reset` | `neuron_reset` | la ablación de memoria (el efecto Alzheimer) |

**Se usa la configuración desplegada CELDA A CELDA (`chosen_config`), no la modal.** Colapsar a la
moda es exactamente lo que hizo vacío el intento v1.

## 3. `H1` — «tiempos de recuperación más cortos»

**Estimando:** `restricted_ttr_hours = min(tiempo hasta restauración, τ)`, con `τ = 8` semanas,
restauración = servicio ≥ 95 % del basal pre-evento durante 7 días consecutivos y backlog
≤ 1,05× basal. Si el choque **no degrada** frente a su placebo pareado, vale **0** (absorbido); si
degrada y no restaura antes de `τ`, vale **τ**. **Ninguna celda queda sin valor y la censura no
puede fabricar un brazo rápido.**

**Por qué en régimen aislado y no en el de la meta-búsqueda:** medido hoy sobre el propio bloque,
los riesgos recurrentes `R11–R24` a 52 semanas se agregan en **un solo clúster** que nunca termina,
así que *no existe* un «tiempo de vuelta a la normalidad» — la cadena nunca deja de estar
perturbada. Eso no es un defecto del instrumento, es una propiedad del régimen, y **va escrito en
el resultado gane quien gane**. `H1` sólo tiene estimando bajo **choque aislado**, que es la rejilla
de `RECOVERY_CONTEXTS` (R11…R24, uno a uno, medianas redondeadas de las cintas del paso 3 ya
abiertas, inicio en la semana 8).

**Mapeo familia→configuración, fijado ahora:** `R11–R14` usan la configuración que el brazo
desplegó en el contexto `R1r`; `R21–R24`, la de `R2r`. Ningún otro emparejamiento se prueba.

**Celda:** `(semilla, contexto de choque)`. `n = 120 × 8 = 960` por brazo.
**Contraste primario:** `static − hybrid` en horas, pareado por celda. **Positivo = el híbrido se
recupera antes.** Bootstrap **sobre celdas**, 5.000 remuestreos (no sobre observaciones: esa fue
la corrección de la auditoría externa del 1-ago).

## 4. `H3` — «menor varianza de desempeño entre intensidades heterogéneas»

**Escalera congelada, la misma de v1 y ya validada por su `f2`:** multiplicadores de frecuencia
`×1, ×2, ×3, ×4` sobre los ocho riesgos del contexto.

**Endpoint primario: SERVICIO** — `service_loss_auc_ration_hours`. **No `ret_excel`**, porque está
medido que `ret_excel` premia el abandono; se reporta al lado, con la advertencia, y **nunca decide**.

**Celda:** `(semilla, contexto base ∈ {R1r, R2r, R1r+R2r})`; para cada celda, la varianza
(`ddof=1`) del endpoint **entre los cuatro peldaños**. `n = 120 × 3 = 360` por brazo.
**Contraste primario:** `static − hybrid` de esa varianza. **Positivo = el híbrido es menos volátil.**

## 5. Multiplicidad y regla de lectura, fijadas ahora

Familia declarada, `K = 4`: {`H1` mem−OFAT, `H1` mem−reinicio, `H3` mem−OFAT, `H3` mem−reinicio}.
**Holm-Bonferroni.** Los contrastes contra `random` y los subconjuntos son **descriptivos** y no
entran en la familia.

* **`H1` SOSTENIDA** sólo si `LCB95 > 0` en `static − hybrid` **y** sobrevive a Holm.
* **`H3` SOSTENIDA** con el mismo criterio sobre la varianza.
* Cualquier otra cosa → **NO SOSTENIDA**, y se reporta el nivel y el orden entre brazos sin
  titularlo como tendencia.

**Compromiso:** la tabla completa entra al manuscrito gane quien gane. Si `H1` sale a favor del
estático —y puede: el híbrido despliega `turnos 2` donde el reinicio despliega `turnos 3`, y más
capacidad debería restaurar antes— **se publica así**, junto al hecho de que el híbrido gana en
`H1′` y en velocidad de búsqueda.

## 6. Falsadores, con por qué cada uno **puede** fallar

| falsador | por qué puede fallar |
|---|---|
| `f1_the_arms_deploy_different_configurations` | si `< 30 %` de las 960 celdas tienen `hybrid ≠ static`, ambas hipótesis vuelven a ser **vacías por construcción**. **Falló en v1**; por eso existe |
| `f2_the_recovery_endpoint_has_range` | si la fracción absorbida `> 0,999` o la censurada en `τ` `> 0,999`, el endpoint no separa nada. Es **el mismo criterio que mató a `system_ttr`**, aplicado al sucesor |
| `f3_the_placebo_is_really_shock_free` | el placebo debe tener **cero** eventos con física (sólo el marcador de ventana) y compartir configuración y semilla. Falla si algún placebo trae riesgo real, o si «impactado» no se decide por la comparación incremental |
| `f4_the_sealed_surface_still_reproduces` | **el ancla externa.** Re-evalúa una muestra de `(config, contexto, semilla)` bajo la física de hoy y exige el `chosen_value` sellado a 12 decimales. Si `supply_chain.py` derivó desde que se selló el bloque —hueco A2 del registro— **falla y se detiene todo** |
| `f5_the_intensity_ladder_escalates` | el número medio de eventos realizados debe **crecer estrictamente** de `×1` a `×4`. Falla si la escalera no escala y entonces `H3` no tiene eje |
| `f6_variance_is_across_intensities_not_within` | recalcula la varianza **dentro** de un peldaño y exige que sea un número distinto. Falla si el runner promedió por el eje equivocado |
| `f7_no_new_seeds_are_opened` | `6.000.001–120` es un bloque **ya abierto**; esto es re-análisis + evaluación de las configuraciones que produjo. Falla si el registro central marca alguna semilla como virgen o si aparece una fuera del bloque |

**Si `f4` falla, no hay resultado**: se reporta la deriva de física como el hallazgo y se detiene.
**Si `f1` falla**, se reporta «vacío por construcción, otra vez» y no se publican intervalos.
**Si `f2` falla**, `H1` vuelve a `NO EVALUABLE` y `H3` se reporta igual.

## 7. Alcance

Desarrollo sobre un bloque ya abierto. **No abre semillas vírgenes, no adjudica el manuscrito** y
no toca ningún artefacto fechado. `H1′` y `H3′` quedan como están; esto se añade al lado.
