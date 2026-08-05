# Enmienda — la escalera de comparadores de búsqueda

**Escrita ANTES de correr.** Runner: `scripts/run_search_comparator_ladder_v1.py`. Opera sobre
`results/surface_cache/wrap288_v1` (bloque quemado `5.300.001–012`, réplica declarada). **Sin
simulación nueva, sin semillas, sin adjudicación.**

## 1. Qué la habilita

`results/surface_gates/result.json` → `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. El gate `g2` pasa con
ΔCV-R² fuera de muestra de 0,072 a 0,159 (LCB95 hasta +0,149) en cinco de seis contextos: **la
superficie no es separable**, así que OFAT no es óptimo por construcción y **existe un problema de
búsqueda real**. Sin ese gate esta escalera no tendría objeto.

`g1` falla —`H_regime` +0,0038 [LCB95 +0,0000] contra un umbral de 0,05—, así que **el óptimo es
común a los seis contextos**. Eso no cierra el carril: lo redefine. Lo que la memoria puede comprar
aquí **no es adaptación al régimen, es no re-derivar una constante**.

## 2. Los brazos, en el orden en que un revisor los pedirá

| # | brazo | qué es |
|---|---|---|
| 0 | `oracle` | techo de referencia, nunca una política |
| 1 | `random` | el nulo, sin reemplazo |
| 2 | `ofat` | el diseño de la tesis, una coordenada por movimiento |
| 3 | `lhs_local` | arranque space-filling + escalada greedy sobre vecinos Hamming-1 |
| 4 | `gp_ei` | **optimización bayesiana**, el comparador que el revisor nombrará |
| 5 | `ucb1` | bandido factor-independiente sobre niveles |
| 6 | `annealing` | recocido simulado sobre la retícula |
| 7 | `neuron_memory` | la neurona de la Fig. 5 con `ρ` cruzando contextos |
| 8 | `neuron_reset` | su control: `ρ` reiniciado en cada contexto |

**Primario:** AUC de regret normalizado (`Σ regret / (B · |best|)`), sin censura. Presupuesto
`B = 24`, idéntico para todos. Unidad de resampling: la réplica. Bootstrap de bloques, 5.000
remuestreos, LCB95.

## 3. Desviación declarada respecto del plan

El plan decía reusar `supply_chain/gsa.py:68 gp_locate`. **No se reusa literalmente**, y la razón
es técnica: `gp_locate` propone 2.048 candidatos **continuos** y no devuelve historial de visitas,
así que sobre una rejilla de 288 exigiría una regla de anclaje al punto más cercano más una
política arbitraria de duplicados — y ambas decisiones podrían inclinar la comparación a nuestro
favor. **Enumerar las 288 candidatas es exacto y más barato.** El kernel (Matérn ν=2.5 ×
constante + ruido blanco), la fórmula de EI y `normalize_y` se siguen de `gsa.py:75-86`.

`n_init = 8` de un presupuesto de 24. El defecto de `gp_locate` era `n_init = 16`, que gastaría
**dos tercios** del presupuesto en muestreo latino antes del primer paso de EI y convertiría a BO
en un hombre de paja.

## 4. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_budgets_are_matched` | los consumos se cuentan **del log de accesos**, no se afirman; la inicialización de `gp_ei` es exactamente donde esto se rompe |
| `f2_no_arm_reads_an_unrun_configuration` | **no puede fallar en silencio**: `Surface.value_of_visited` lanza `LookupError` ante cualquier configuración no seleccionada, así que un brazo que espiara abortaría la corrida en vez de ganarla. Enforzado estructuralmente |
| `f3_the_oracle_is_a_ceiling_not_a_competitor` | si algún brazo igualara al oráculo, el techo estaría mal especificado y todos los contrastes serían vacíos |
| `f4_no_fresh_seeds` | custodia central, réplica declarada |

## 5. Reglas de lectura

* **la neurona bate a los siete comparadores con LCB95 > 0** → `NEURON_BEATS_THE_FULL_CLASSICAL_LADDER`.
* **bate a `gp_ei` pero no a todos** → `NEURON_BEATS_BAYESIAN_OPTIMISATION_NOT_ALL`.
* **algún clásico la bate** → `CLASSICAL_SEARCH_WINS__<brazo>`. **Es el desenlace que el marco ya
  eligió**: la contribución es la frontera, y «la familia que imita el SCL entre corridas es la
  búsqueda basada en surrogate» sigue respondiendo la Q1 de Garrido —y nombra uno de sus tres
  candidatos— gane quien gane.

**Nada de esto autoriza entrenar una red ni abrir una semilla.**
