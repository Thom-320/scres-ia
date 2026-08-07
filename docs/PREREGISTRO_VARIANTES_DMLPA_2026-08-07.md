# Preregistro — variantes del DMLPA de David, a parámetros igualados

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_dmlpa_variants_v1.py`.
Se lanza **cuando termine** `results/architecture_bakeoff_200k/` — no en paralelo, porque
compartirían los mismos ocho entornos y se degradarían las dos.
Semillas `9491–9495`, las mismas del bake-off, **declaradas como desarrollo, no vírgenes**.

## 1. Por qué

A 60.000 pasos y 200.000 parámetros igualados, el DMLPA de David es **nominalmente el mejor** de
las tres arquitecturas (97,66 contra 97,52 del MLP y 97,05 de la KAN) y **ninguna separa**. La
pregunta no es si otra arquitectura gana, sino **si alguna configuración de la suya sí separa**.

David itera su arquitectura por su lado. Esto es lo que corremos nosotros, sin depender de él.

## 2. Las variantes, y el mecanismo de cada una

Base: `DMLPA(hidden_dim=84, features_dim=84, nhead=12, num_layers=2, ff_mult=4)` — `d_model=84`,
`head_dim=7`, `ff=336`, 187.404 parámetros.

| variante | cambio | mecanismo por el que podría ganar |
|---|---|---|
| `dmlpa_base` | — | referencia |
| **`dmlpa_meanpool`** | `forward` devuelve **media sobre tokens** en vez de `[:, -1, :]` | **No es un hiperparámetro.** El encoder no lleva máscara causal, así que quedarse con el token del último frame descarta el resumen que la atención repartió entre los 16. Es la que más probabilidad le doy |
| `dmlpa_1layer` | `num_layers=1`, `d_model` reajustado al presupuesto | 16 tokens y señal semanal suave: dos capas probablemente sobran, y lo liberado compra anchura |
| `dmlpa_ff2` | `ff_mult=2`, `d_model` reajustado | mismo argumento, movido al feed-forward |
| `dmlpa_nhead4` | `nhead=4` | con `d_model=84`, doce cabezas dan `head_dim=7`; menos cabezas y más anchas suele ayudar |
| **`dmlpa_untrained`** | los mismos pesos **sin entrenar** | **control**, ver §4 |

**Todas se dimensionan al mismo presupuesto de 200.000 y el runner aborta si alguna queda a más
del 10 %.** Sin eso mediríamos capacidad, no arquitectura — la objeción de David.

## 3. Reglas de lectura, fijadas antes de mirar

Primaria: **ReT medio** del episodio, 24 episodios de evaluación. Contraste pareado por semilla
contra `dmlpa_base`, bootstrap sobre semillas, **Holm sobre las cuatro variantes**.

* alguna variante mejor con IC95 que excluye el cero tras Holm →
  **`A_DMLPA_VARIANT_SEPARATES`**. Es un hallazgo sobre **su** arquitectura y se le atribuye.
* ninguna separa → **`ARCHITECTURE_IS_NOT_THE_LEVER_WITHIN_DMLPA`**. Junto con el bake-off —donde
  tampoco separan KAN, MLP ni DMLPA entre sí— el titular pasa a ser que **la familia de
  arquitecturas no es la palanca en este entorno**, y eso es publicable.

## 4. El control que Garrido pidió y nunca corrimos

> *«Medir aprendizaje comparando algoritmo entrenado vs. sin entrenar.»* (reunión del 22 de julio)

`dmlpa_untrained` es exactamente eso: **misma arquitectura, mismos pesos iniciales, cero pasos de
entrenamiento**, evaluada igual.

**Es el falsador que hace legible todo lo demás.** Si el DMLPA entrenado **no** bate al no
entrenado con IC que excluye el cero, entonces en este entorno **el entrenamiento no está
comprando nada**, y ninguna comparación entre arquitecturas significa algo — estaríamos ordenando
ruido. Ese resultado sería más importante que cualquier variante.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_all_variants_are_parameter_matched` | todas a menos del 10 % de 200.000. **Es la objeción de David** |
| `f2_the_variants_are_actually_different` | los conteos de parámetros **o** las salidas del extractor deben diferir entre variantes. Falla si dos son la misma red con otro nombre |
| `f3_training_beats_not_training` | `dmlpa_base` debe batir a `dmlpa_untrained` con IC95 que excluye el cero. **Si falla, no hay aprendizaje que comparar y el resto de la tabla no se interpreta** |
| `f4_budgets_are_matched` | todas entrenan exactamente los mismos pasos |
| `f5_no_fresh_seeds` | 9491–9495, desarrollo declarado, no vírgenes |

**Alcance:** desarrollo. No abre semillas, no adjudica, y **no autoriza reportar una variante
ganadora sin confirmación en bloque virgen**.
