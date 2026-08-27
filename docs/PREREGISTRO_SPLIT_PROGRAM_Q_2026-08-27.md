# PREREGISTRO — Separación selección/evaluación sobre los paneles de Program Q (`q_split_bias_v1`)

| Campo | Valor |
|---|---|
| Fecha | 2026-08-27 |
| Sucede a | `control_ceiling_v1` (veredicto `NOT_CLOSED`), prereg sha256 `28aab3df…` |
| Datos | Paneles sellados de Program Q: 256 tapas × 65.536 calendarios × 3 celdas, ya consumidas |
| Cómputo | numpy sobre `.npz` en disco. **Cero simulación, cero semillas nuevas, nada entrenado** |
| Salida | `results/program_n/q_split_bias_v1/result.json` |

## La pregunta exacta

`control_ceiling_v1` midió un techo clarividente de **+0,0529 a +0,0791** sobre el
mejor clásico, y no pudo cerrarlo porque el estimador es un máximo dentro de
muestra sobre 65.536 calendarios. Su falsador F2 dio la razón para dudar:
**244, 250 y 204 calendarios distintos sobre 256 tapas** — casi cada cinta quiere
el suyo, que es la firma del sesgo del ganador.

Gate-0 midió ese sesgo en esta misma física y le salió `Δ_bias` de **+0,119 a
+0,176**, es decir **más grande que todo el techo**. Pero lo midió en otro bloque
de semillas y con 64 tapas por celda. Aquí hay 256, cuatro veces más, ya en disco.

**Estimando primario: `Δ_bias` sobre este bloque.** Si el sesgo explica el techo,
el techo no es margen y la lane de control queda cerrada para la selección de
calendario. Si no lo explica, queda un residuo que sí habría que perseguir.

## Estimandos

Partición determinista de las 256 tapas de cada celda en A y B de 128, por orden
de `sha256(str(seed))` — fijada por la semilla, no por ningún resultado.

```
k*(A)    = argmax_k mean_A[ X_ol[t,k] ]
G_naive  = mean_A[ max_k X_ol[t,k] ]      − max_c mean_A[ X_cl[t,c] ]
G_split  = mean_B[ X_ol[t, k*(A)] ]       − max_c mean_B[ X_cl[t,c] ]
Δ_bias   = G_naive − G_split
```

`max_c` se **reselecciona dentro de cada remuestreo** y por separado en A y en B:
fijarlo en A metería la selección del comparador por la puerta de atrás.

Además, **cross-fitting** con K=8 pliegues sobre tapas —seleccionar en 7/8,
evaluar en 1/8, promediar— para que `G_split` no dependa de una partición
afortunada. Bootstrap sobre tapas, 10.000 remuestreos, CI percentil.

## Regla de decisión, fijada antes de mirar

SESOI = 0,01, el de Program Q, sobre `ret_visible`.

1. **`CEILING_IS_BIAS`** si `Δ_bias ≥ techo_clarividente` de `control_ceiling_v1`
   **y** `UCB95(G_split) < 0,01`. El techo se explica por selección y no queda
   margen: la lane de selección de calendario cierra.
2. **`RESIDUAL_ROOM`** si `LCB95(G_split) > 0,01`. Queda margen real fuera de
   muestra y hay caso para entrenar.
3. **`UNDETERMINED`** en cualquier otro caso.

## Alcance — lo que este estimando NO cubre

`G_split` acota la **selección de un calendario fijo** a partir de datos de otras
tapas. **No acota una política condicionada al estado dentro del episodio**, que
es lo que hace RecurrentPPO: el aprendiz ya bate al mejor calendario fijo por
+0,0795 / +0,0725 / +0,1172 (`H_OL`, LCB muy sobre cero), algo que ninguna
selección de calendario logra — `brecha_fijo` es −0,081 / −0,073 / −0,118.

Por tanto un `CEILING_IS_BIAS` cierra la pregunta «¿hay un calendario mejor que
encontrar?», y **no** la pregunta «¿puede una política observacional mejor sacar
más?». Esa segunda exige datos de observación que estos paneles no contienen.
Se declara aquí para que nadie lea de más en el resultado.

## Falsadores

**F1 — reproducción del ancla.** Recomputar `mean[L] − max_c mean[X_cl]` sobre las
256 tapas debe reproducir el `Delta_N` sellado a 1e-9.
*Puede fallar* si leo un panel distinto del que produjo el veredicto.
*Puede pasar*: ya dio error exactamente 0 en `control_ceiling_v1`.

**F2 — la partición no está desbalanceada.** `max_c mean_A[X_cl]` y
`max_c mean_B[X_cl]` deben diferir en menos de 0,02, y el `argmax_c` debe
coincidir en A y B. *Puede fallar*: con 128 tapas por mitad, una partición
desafortunada movería el ancla y `G_split` mediría la partición, no el sesgo.

**F3 — el cross-fitting concuerda con el split simple.** `G_split` por partición
única y por K=8 pliegues deben diferir en menos de 0,02.
*Puede fallar* si el resultado depende de qué mitad se usó para seleccionar.

**F4 — `k*(A)` no es el `argmax` global.** Si la selección en A devolviera el
mismo calendario que el máximo sobre las 256, no habría separación real.
*Puede pasar y puede fallar*, y es informativo en ambos sentidos.

## Lo que no autoriza

No abre semillas, no entrena, no reabre Program Q —cuyo contrato prohíbe
reentrenar—, no cambia ningún estimando sellado. Un `RESIDUAL_ROOM` **no**
autoriza una campaña de tuneo: autoriza escribir el contrato de esa campaña.
