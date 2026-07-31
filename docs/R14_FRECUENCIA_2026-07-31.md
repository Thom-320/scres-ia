# La frecuencia de R14 — está bien, y mi «0,01×» era un error de unidades

**Status:** `DEVELOPMENT_HYPOTHESIS_REFUTED`. Nada implementado.

## 1. La corrección primero

En `docs/K_GENERADOR_2026-07-31.md` §4 escribí que `R14` es «donde más lejos estamos —
258 eventos/año contra 22.153, un factor 0,01×». **Es falso, y es un error de unidades.**

La Tabla 6.11 mide `R14` en **«Defective products»**, no en eventos. Nuestros 258 son
**eventos de agregación diaria**, cada uno con su magnitud en unidades defectuosas.

| | por año |
|---|---:|
| tesis, Tabla 6.11 | **22.153** defectuosos |
| nuestro (nivel `current`), magnitud sumada | **18.770** defectuosos |
| **razón** | **0,85×** |

**La frecuencia de R14 es esencialmente correcta.** Y el déficit del 15% tiene causa
física: disparamos `R14` en 258 de 336 días, porque el sorteo se hace sobre
`_today_produced` y los días sin producción no generan defectos. Eso es correcto, no un
defecto.

El parámetro además cuadra con la aritmética de la tesis: `n·p = 2564 × 0,03 = 76,9`
defectuosos por sorteo, y `76,9 × 336 = 25.845/año` si todos los días produjeran.

## 2. La exposición al riesgo, orden a orden

| riesgo | Garrido | nuestro |
|---|---:|---:|
| `R11` averías | 68,1% | **75,4%** |
| `R12` contratos | 10,9% | **3,7%** |
| `R13` faltantes | 25,3% | **37,6%** |
| `R14` defectuosos | 98,1% | **81,1%** |
| **algún riesgo** | **100,0%** | — |

Estamos en el mismo orden de magnitud en los cuatro. `R12` es el único claramente bajo
(3,7% contra 10,9%), consistente con lo que ya medimos en frecuencia de eventos (0,43×).

## 3. Lo que esto refuta

**La brecha de fracción demorada (36,5% contra 83,5%) NO se explica por exposición al
riesgo.** Tocamos aproximadamente las mismas órdenes que él. Lo que difiere es que **sus
órdenes tocadas se demoran y las nuestras no**: él tiene 100% tocadas y 83,5% demoradas;
nosotros ~81% tocadas por `R14` solo y 36,5% demoradas.

Es decir, la pregunta abierta se desplaza otra vez: no es *cuántas órdenes ve el riesgo*,
sino *cuánto retrasa el riesgo a la orden que toca*. Su `R14` por orden tiene mediana 4 y
máximo 20 unidades defectuosas — cantidades pequeñas — y aun así el 83,5% de sus órdenes
pierde al menos un día completo.

## 4. Estado, y lo que NO voy a hacer

**No propongo un quinto mecanismo hoy.** El patrón de la sesión es claro: cada hipótesis que
propuse sobre este bloque —duración de R12/R13, multiplicador serial, recurrencia y
solapamiento, el clamp, la cadencia como causa de la dispersión, la cola por capacidad, y
ahora la frecuencia de `R14`— resultó refutada o mal medida por mí. Las refutaciones son el
producto, pero proponer la octava sin un observable nuevo sería adivinar.

**Lo que sí queda firme y acotado**, y es bastante:

* `CTj = 48 + k·24 + δ`, con `48` y la **rejilla** `k·24` **reproducidos exactos**;
* `δ` caracterizado: sorteo uniforme sobre el turno de 8 h, techo verificado contra `Q/λ`;
* `k` es demora por riesgo (`corr` 0,40 con `R14`), no cola de capacidad ni stock;
* la exposición al riesgo es correcta; la **conversión de toque en demora** no lo es.

`ret_mean` bajo los defaults embarcados sigue sin verse afectado por nada de esto.
