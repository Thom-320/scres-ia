# Preregistro — la familia completa de derivaciones de ReT-Excel y Cobb-Douglas

**Escrito y commiteado ANTES de medir ninguna variante.** Runner:
`scripts/run_metric_derivation_family_v1.py`. Semillas: bloque quemado `5.300.001+`, réplica
declarada. **Ninguna nueva.**

## 1. Qué se hace y qué NO se hace

El PI pidió *«deriva la métrica de resiliencia que nos sirva»* y *«revisa con cuál llegamos al
umbral»*.

**Lo que NO se hace: probar variantes hasta que una cruce y reportar sólo ésa.** Eso es p-hacking
con grados de libertad métricos, y con nuestro propio atlas publicado un revisor lo vería en una
tarde.

**Lo que sí se hace, y responde a lo mismo:** enumerar aquí **el espacio completo** de derivaciones
defendibles, medirlas **todas**, reportarlas **todas**, y **pagar la multiplicidad**. Si alguna
cruza el umbral con la corrección aplicada, es un hallazgo real. Si ninguna cruza, la pregunta
métrica queda **cerrada** — y eso también es contribución, porque nadie podrá decir que no miramos.

**El compromiso vinculante: la tabla completa entra al manuscrito, gane quien gane.**

## 2. Por qué estas dos familias son defendibles

Ambas son **de Garrido**. Ninguna derivación inventa una métrica: todas mueven **un parámetro que
él mismo eligió** o **reparan un defecto medido y documentado**. Cada variante lleva abajo su
justificación y su fuente.

## 3. Familia A — derivaciones de ReT-Excel

La fórmula del Excel es
`IF(AVG(risk)>0, IF(APj>0, APj/LT, 0.5·(1/RPj)), 1−((Bt+Ut)/j))`, y la Eq. 5.5 de la tesis es su
versión a cuatro ramas con pesos `Re^max = 1,0`, `Re^mean = 0,5`, `Re^min = 0,0`.

| eje | niveles | por qué es defendible |
|---|---|---|
| **población** | `visible` · `full_ledger` · `risk_conditional` | ya en el panel; `full_ledger` **repara la censura medida** |
| **guarda de rama** | `thesis` (`CT ≤ LT`) · `excel` (sin guarda) | **son sus dos versiones**: el Excel no lleva la guarda y la tesis sí |
| **normalización de recuperación** | `raw` `0,5/RP` · `adimensional` `0,5·LT/RP` · `acotada` `0,5/(1+RP/LT)` | `RP` está en horas y `0,5/RP` **supera 1 cuando `RP < 0,5 h`** — un pedido puntuó **73,9**. Es un **error dimensional documentado**, no una preferencia |
| **`Re^min`** | `0,0` (suyo) · `0,25` | con `Re^min = 0` la rama *non_recovery* puntúa **siempre 0** y `(DP−RP)/CT` no se usa nunca. Liberarlo es explorar **su propio parámetro** |
| **recorte** | `ninguno` · `[0,1]` | el Excel no recorta; la tesis declara el rango [0,1] |
| **agregación** | media por pedido · **ponderada por cantidad** | un pedido de 10 raciones y otro de 10.000 pesan igual en la media; ponderar por cantidad es la lectura de servicio de la misma fórmula |

`3 × 2 × 3 × 2 × 2 × 2 = 144` variantes.

## 4. Familia B — derivaciones de Cobb-Douglas

`R = σ(Σ signo·aₓ·ln x)` con `aₓ = 0,20/ln(x_max)`.

| eje | niveles | por qué es defendible |
|---|---|---|
| **exponentes** | `publicados` (los suyos) · `re-derivados en nuestros máximos` · `re-derivados por contexto` | **su propia regla** dice derivarlos del conjunto; aplicarla a nuestro conjunto es seguirla, no desviarse |
| **conjunto de variables** | `sus cinco` · `sin τ` · `+ servicio` | τ está **muerta en el 18 % de las celdas** y su exponente está **mal condicionado** (amplificación 3,39); y **su §6.2 pide explícitamente añadir drivers que su índice no consideró** |
| **conjunto de κ̇** | `dentro de (contexto, semilla)` · `global` | κ̇ es relativo al conjunto y él no declara cuál; ambos son lecturas suyas |

`3 × 3 × 2 = 18` variantes.

## 5. Multiplicidad — el precio, fijado aquí

**`K = 162` variantes.** Corrección **Holm-Bonferroni** sobre las 162, no Bonferroni simple: Holm es
uniformemente más potente y sigue controlando el error familiar.

**Umbral por variante: `LCB95_corregido ≥ 0,05`**, el mismo de todos los gates del proyecto.

**Y una regla que hace esto honesto o no lo hace:** si ninguna cruza, se reporta *«ninguna de 162
derivaciones defendibles alcanza el umbral»*. Si alguna cruza, se reporta **con su posición en la
familia y su corrección**, nunca como si hubiera sido la única declarada.

## 6. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_family_was_declared_before_any_variant_was_measured` | este documento y su commit deben preceder al `created_at` del artefacto. **Falla si el resultado ya existía** |
| `f2_all_variants_are_reported` | el artefacto debe contener las 162, no un subconjunto. Falla si alguna se cae en silencio |
| `f3_the_family_separates` | las variantes no pueden dar todas el mismo `H_regime`; si lo dieran, mediríamos el estimador y no la métrica |
| `f4_the_baseline_reproduces` | las tres variantes que ya existen en el panel —`visible`, `full_ledger`, `risk_conditional` con guarda excel, raw, sin recorte, media por pedido— deben reproducir el atlas: `+0,00050`, `+0,00028`, `+0,00380`. **Falla si la reimplementación no coincide con la sellada** |
| `f5_multiplicity_is_applied` | ningún `LCB95` sin corregir puede compararse contra 0,05 |
| `f6_no_fresh_seeds` | custodia central, réplica declarada |

`f4` es el ancla: sin ella, 162 números nuevos no son comparables con nada.

## 7. Reglas de lectura, fijadas de antemano

* **alguna cruza con Holm** → `DEFENSIBLE_DERIVATION_REACHES_THE_BAR`. Se declara **esa** variante
  como candidata a primaria, se documenta su justificación mecánica, y **se reporta la familia
  entera al lado**.
* **ninguna cruza** → `NO_DEFENSIBLE_DERIVATION_REACHES_THE_BAR`. La pregunta métrica queda cerrada
  sobre 162 variantes de las dos métricas nativas de Garrido, y el manuscrito lo dice así.
* **En ambos casos** se reporta el máximo alcanzado y a qué distancia quedó del umbral.

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica, no autoriza aprendices.
