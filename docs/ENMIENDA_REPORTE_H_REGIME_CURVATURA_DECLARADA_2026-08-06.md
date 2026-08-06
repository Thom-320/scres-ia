# Enmienda — todo `H_regime` se reporta con su curvatura declarada

**Vinculante desde ahora.** Origen: `results/monotone_transform_family_v4/result.json`
(`A_MONOTONE_RESCALING_SURVIVES_ALL_THREE`, 9/9 falsadores), preregistro
`docs/PREREGISTRO_TECHO_MONOTONO_V4_BORDE_Y_PISO_2026-08-06.md`.

**No edita ningún contrato ni artefacto en sitio.** Cambia cómo se *reporta*, no lo medido.

## 1. El hecho que la obliga

`H_regime = 1 − max_a mean_r Ṽ(r,a)` **no es invariante a reparametrizaciones monótonas de la
métrica**. Una `f` creciente no puede cambiar el argmax de ningún régimen, pero sí cambia qué
configuración fija gana *en promedio* — y eso es todo lo que `H_regime` mide.

Medido sobre las dos rejillas, familia declarada `K = 661`, Holm sobre las 661, LCB por bootstrap
exacto sobre semillas:

| rejilla | curvatura de Garrido (`γ=1`) | techo con señal ≥ 99 % | techo con señal ≥ 80 % |
|---|---:|---:|---:|
| `wrap288_v1` (288) | **0,0000** | **0,0000** | **0,0000** |
| `wrap288_compat_extended_v1` (4.608) | **0,0195** | **+0,1311** (LCB +0,1269) | **+0,3815** (LCB +0,3536) |

## 2. Las tres reglas

**R1 — La rejilla de 288 es a prueba de curvatura y se reporta sin salvedad.** Una sola
configuración (la 240) es óptima en los seis regímenes, así que **toda** `f` creciente da `H = 0`.
Cero en las 661. Ese nulo no se puede atacar por elección de métrica.

**R2 — Todo `H_regime` de rejilla extendida se reporta con su curvatura al lado.** Nunca un número
suelto. La forma canónica es *«H = x bajo la curvatura de Garrido (γ=1)»*, y si se cita cualquier
otra, **la curva `H*(piso)` completa va al lado**, no su máximo.

**R3 — La curvatura sólo se cambia por mecanismo declarado, nunca por el `H` que produce.** La única
curvatura declarada en todo este expediente es **la suya**: su índice publicado es
`σ(Σ signo·aₓ·ln x)`, que es exactamente nuestra identidad. **Bajo ella no hay headroom en ninguna
de las dos rejillas.** Adoptar otra exige preregistro propio que diga **por qué** esa curvatura
—antes de ver su `H`— y confirmación en bloque virgen.

## 3. Lo que esto NO autoriza

No autoriza reabrir ningún nulo cerrado sobre la rejilla de 288. No autoriza entrenar contra una
métrica recurvada. Y **no convierte `0,38` en un resultado**: el número depende de un piso de señal
que elegimos nosotros —se mueve **2,9×** entre 0,99 y 0,80— y el orden de configuraciones es
idéntico por construcción, así que lo que se mueve es **la actitud ante el riesgo, no la cadena**.

## 4. La frase para el manuscrito

> `H_regime` no es invariante a la curvatura de la métrica. Bajo la parametrización que Garrido
> publicó, la rejilla de 288 da exactamente 0 y la extendida 0,0195, ambas por debajo del umbral.
> El headroom aparece sólo al imponer curvatura adicional que él no declaró, y su magnitud es
> función de un piso de señal antes que de la cadena de suministro.

## 5. Lo que queda abierto, y es la siguiente pregunta bien planteada

La curvatura **es** una actitud ante el riesgo. Si alguna actitud **citable** de la literatura de
cadenas de suministro —una utilidad de potencia con coeficiente publicado, un CVaR con nivel
estándar— cae dentro del conjunto que califica, entonces la curvatura estaría **declarada por la
literatura** y la adopción sería posible sin que la elijamos nosotros. **Puede fallar**: si todos
los coeficientes citables dan `γ < 3`, ninguna llega al umbral y la vía se cierra por número.
