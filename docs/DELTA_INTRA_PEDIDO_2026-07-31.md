# Qué genera `δ` intra-pedido: es aleatoriedad exógena acotada por el turno

**Status:** `DEVELOPMENT_MECHANISM_CHARACTERISED`. 21.667 filas, nueve hojas R1r.
Nada implementado.

## 1. `δ` es una identidad de fase, y la manda la llegada

    δ = (fase(OATj) − fase(OPTj)) mod 24

**Se cumple en el 100% de las filas con error exactamente cero** — es una identidad, ya que
`CTj = OATj − OPTj`. Lo informativo es de cuál de las dos fases depende:

| | corr con `δ` |
|---|---:|
| fase de **llegada** (`OATj mod 24`) | **0,8965** |
| fase de **colocación** (`OPTj mod 24`) | 0,0047 |

La colocación no aporta nada. **`δ` es dónde cae la llegada dentro del día**, medido desde
la colocación de esa misma orden.

## 2. No lo predice nada del pedido

Usando las columnas de la hoja que no había explotado —`OP9` (inventario en Op9), `∑Bt`,
`∑Ut` y los indicadores de riesgo por tipo:

| candidato | corr con `δ` |
|---|---:|
| `OP9` (stock al pedir) | 0,024 |
| `∑Bt` | 0,016 |
| `Q` | 0,077 |
| `k` (días esperados) | 0,089 |
| `R14` | 0,116 |
| `R11_1+R11_2` | 0,060 |
| `R12+R13` | 0,041 |

Y **diez variantes de acumulación de producción** —`(Q−OP9)/λ`, `OP9/λ`, `cumsum(Q)/λ mod 8`
y otras siete— dan todas `|corr| < 0,03`.

## 3. El techo es el turno, no el ensamblaje del pedido

Las dos hipótesis daban ajustes casi idénticos, porque `Q/λ ∈ [7,49; 8,10]` y
`HOURS_PER_SHIFT = 8` son casi el mismo número. **El test por deciles de `Q` las separa:**

| decil de `Q` | `Q` medio | `Q/λ` | `δ` p99 | `δ` máx |
|---|---:|---:|---:|---:|
| 1 | 2.408 | **7,514** | 7,985 | 8,158 |
| 5 | 2.472 | **7,714** | 8,043 | 8,402 |
| 9 | 2.537 | **7,915** | 8,042 | 8,370 |
| 10 | 2.558 | **7,983** | 8,309 | 22,856 |

`Q/λ` sube de 7,51 a 7,98 entre deciles; **el `p99` de `δ` se queda plano en ~8,0**, y el
máximo **excede** `Q/λ` en todos los deciles. **El techo no se mueve con `Q`**, luego es la
constante fija de 8 h y no el tiempo de ensamblaje del propio pedido.

Normalizado, `δ/8` da media 0,4955 y SD 0,2863 contra 0,5 y 0,2887 teóricos, con 98,5%
dentro.

## 4. La conclusión, y corrige mi propia cautela

**`δ` es un sorteo uniforme sobre el turno de 8 h, independiente de todo lo medible en sus
datos.** En su modelo la orden se sirve en un punto uniformemente aleatorio dentro de la
jornada, y eso **es** el generador: no hay cola que lo produzca —ya lo refutamos por conteo,
con un pedido por día— ni atributo del pedido que lo prediga.

En `docs/RESULTADO_TURNO_Y_CAPACIDAD_2026-07-31.md` §5 escribí que implementar `δ` como
sorteo `U(0,8)` sería «ajustar la forma observada» y volvería tautológico el falsador. **Eso
era correcto entonces y ya no lo es**, y la diferencia es el test de §3: el techo fijo en 8 h
mientras `Q/λ` varía es evidencia **independiente de la forma** a favor del turno. Un sorteo
`U(0, HOURS_PER_SHIFT)` es ahora una afirmación de mecanismo con apoyo, no una curva
ajustada.

Sigue siendo **un supuesto estocástico**, y eso hay que declararlo: su prueba no puede ser
que reproduzca `δ` —lo hará por construcción— sino **qué arrastra en los otros cinco
momentos**. Es exactamente la forma que dejé escrita en aquel §5.

## 5. Lo que queda sin explicar

* **La cola de `δ > 8`** (1,5% de las filas, hasta 22,86 h). No es `Q/λ`, que topa en 8,10.
* **`k`**, los días adicionales. Refutado el encolamiento por capacidad; `corr(δ, k) = 0,089`
  dice que los dos términos son casi independientes, así que `k` necesita su propia
  explicación.
* Su fase de colocación **deriva** a lo largo de la corrida (`corr = 0,9999`, de 0,85 h a
  5,53 h) y aparecen huecos de ~47 h entre pedidos consecutivos, compatibles con
  `DAYS_PER_WEEK = 6`. Ninguna de las dos cosas afecta a `δ`, pero las dos son estructura de
  su calendario que nuestro modelo no reproduce.

## 6. Estado

Nada implementado. El sucesor tendría que declarar `δ ~ U(0, HOURS_PER_SHIFT)` **como
supuesto**, con la evidencia de §3 citada, y puntuarlo **solo** por los otros momentos —
nunca por reproducir `δ`.
