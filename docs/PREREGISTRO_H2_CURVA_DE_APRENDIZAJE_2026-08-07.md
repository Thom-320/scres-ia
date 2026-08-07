# Preregistro — `H2`, la única de las cuatro hipótesis del borrador sin veredicto

**Escrito y commiteado ANTES de mirar los datos.** Runner: `scripts/run_h2_learning_curve_v1.py`.
Re-análisis de artefactos ya sellados: **cero cómputo de simulación, cero semillas**.

## 1. Qué dice el borrador y qué le falta

> **H2 – Adaptation Hypothesis.** *Neural-enabled SCRES models demonstrate improved performance
> under successive disruptions (learning curve effect).*

Estado actual: **medida, sin adjudicar**. La forma que circulaba —«ventaja +0,00 → +10,00 entre
contextos»— quedó **retirada** con el runner que tenía fuga de normalizador
(`docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md`). Desde entonces H1, H3 y H4 tienen veredicto y
H2 no. Es la hipótesis de la **curva de aprendizaje**, sobre la que descansa el encuadre entero.

## 2. El estimando, fijado antes de mirar

Fuente: el par contratado de 120 réplicas,
`results/garrido_meta_learner_h3power_h3_contract_{local,vps}_v2/`, semillas `6.000.001–120`,
bloque ya abierto. Normalizador **de prefijo**, métrica primaria `auc_regret_norm`, exactamente
como `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md`.

Los seis contextos se recorren **en orden fijo**: `R1r`, `R2r`, `R1r+R2r`, `R1r|esc`, `R2r|esc`,
`R1r+R2r|esc`. Índice ordinal `k = 1…6`.

Para cada réplica `r` y contexto `k`:

    ventaja(r,k) = AUC_regret(reinicio) − AUC_regret(memoria)      (positivo = la memoria gana)

**Estimando primario: la pendiente OLS de `ventaja(r,·)` contra `k`, por réplica.** Bootstrap sobre
las 120 réplicas.

> **`H2` SOSTENIDA ⟺ `LCB95` de la pendiente media `> 0`.**

Es decir: la ventaja de conservar `ρ` **crece** con el número de disrupciones sucesivas ya vistas.
Una ventaja grande pero **plana** no sostiene H2 — sostiene H4, que ya está medida.

## 3. El control que absorbe la dificultad del orden

Los tres últimos contextos son **escalados** (×3 de frecuencia). Cualquier brazo puede parecer
mejor o peor por la posición, no por aprender. Por eso se calcula **la misma pendiente** sobre un
par que **no conserva estado**:

    ventaja_nula(r,k) = AUC_regret(aleatorio) − AUC_regret(OFAT)

Ninguno de los dos retiene nada entre contextos, así que su pendiente estima la deriva por
dificultad. **`f3` falla si la pendiente nula es positiva con `LCB95 > 0`**: entonces la tendencia
es del orden de los contextos y no del aprendizaje, y H2 no se puede leer.

## 4. Reglas de lectura, fijadas ahora

| resultado | veredicto |
|---|---|
| pendiente `LCB95 > 0` **y** control nulo no positivo | **`H2_SUPPORTED_LEARNING_CURVE`** |
| pendiente con IC que cruza cero | **`H2_NOT_SUPPORTED_ADVANTAGE_IS_FLAT`** — y se dice que la ventaja existe pero **no crece**, que es H4 y no H2 |
| pendiente `UCB95 < 0` | **`H2_REFUTED_ADVANTAGE_SHRINKS`** |
| control nulo positivo | **`H2_NOT_READABLE_ORDER_CONFOUND`** |

**Compromiso:** el número entra al manuscrito gane quien pierda, y si sale plana **no se
reescribe como «la memoria ayuda en todos los contextos»** — eso ya lo dice H4.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_source_is_the_contracted_pair_at_n120` | exige las 120 réplicas y las dos rebanadas selladas contra `PREREGISTRO_H3_POTENCIA`. Falla si se cuela otra fuente |
| `f2_the_normaliser_is_prefix_not_oracle` | la ventaja se recalcula desde las curvas de arrepentimiento con el normalizador de prefijo. Falla si reproduce el panel oráculo, que es el que tenía fuga |
| `f3_the_order_confound_is_absorbed` | la pendiente nula OFAT-vs-aleatorio **no** puede ser positiva con LCB95>0. **Es el falsador que puede impedir leer H2** |
| `f4_the_slope_can_be_negative` | comprueba que el estimador admite pendiente negativa y la reporta; un estimador que sólo puede salir ≥0 no mide nada |
| `f5_no_new_seeds` | `6.000.001–120`, ya abiertas |

## 6. Alcance

Desarrollo sobre un bloque ya abierto. **No adjudica el manuscrito**, no autoriza entrenamiento, y
no toca `H1′`, `H3′` ni `H4`.
