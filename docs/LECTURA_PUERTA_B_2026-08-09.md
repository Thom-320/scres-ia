# Lectura de la Puerta B — la primera prima neural capturada del proyecto

**Artefacto:** `results/program_n/gate_b_cd_surface/result.json` ·
**Veredicto:** `SURFACE_PREMIUM_CAPTURED` · **7 de 7 falsadores, cero fallidos.**

## 1. Los números, y el control que los hace legibles

```
recurrent                   +0.8981   (conjunto de informacion MAS RICO, ver §4)
kan_tuned                   +0.7446   (antes +0.6019)
mlp_tuned                   +0.6952   (antes +0.5841)
train_cell_mean_comparator  +0.6931   (antes +0.6931)
linear_lagged               +0.6927
spline_buffer               +0.6365   (antes +0.6365)
linear_interactions         +0.6306   (antes +0.6306)   <- baseline primario
tree                        +0.6225   (antes +0.6225)
linear_additive             +0.6062   (antes +0.6062)
constant                    -0.0167   (antes -0.0167)
```

**Los brazos clásicos reproducen con una desviación máxima de 4,9e-05.** Ése es `f2` y es lo que
hace que todo lo demás signifique algo: mismo objetivo, mismos folds, misma calibración por fold,
mismas ocho semillas. **Lo único que cambió fue el ajuste de las redes.**

| contraste contra `linear_interactions` | media | IC95 | ¿pasa? |
|---|---|---|---|
| **`kan_tuned`** | **+0,1140** | **[+0,0614, +0,1665]** | **sí** |
| `mlp_tuned` | +0,0646 | [−0,0883, +0,2174] | no |
| `spline_buffer` | +0,0059 | [−0,0465, +0,0584] | no |
| `tree` | −0,0081 | [−0,1631, +0,1469] | no |

Con `SESOI = 0,05` y el intervalo excluyendo cero, **sólo el KAN pasa**.

## 2. H_B era cierta: el problema éramos nosotros

La hipótesis congelada era *«las redes perdieron por cómo se ajustaron, no por lo que el entorno
ofrece»*. **Confirmada.** Con el mismo dato y el mismo objetivo:

* **KAN pasa de 0,6019 a 0,7446** — de estar por debajo del lineal a estar por encima del techo;
* **MLP pasa de 0,5841 a 0,6952**.

Lo que cambió: estandarización sobre train, validación interna con parada temprana, una rejilla
declarada de ocho puntos idéntica para ambas, y cinco semillas de inicialización promediadas. El
predecesor tenía **600 pasos fijos, una semilla, sin validación** contra **OLS en forma cerrada**.

**No fue el entorno.** Era un ajuste sin sintonizar contra un óptimo analítico.

## 3. Y el KAN gana donde Garrido dijo que estaba el problema

Esto importa más que el número. En la tarea de **búsqueda** ya habíamos medido
`KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP`. Aquí, en **predicción**, el KAN gana y el MLP no llega:

> **KAN pierde como controlador de búsqueda y gana como reconocedor de patrones.**

Es exactamente la distinción que la Fig. 3 de Garrido plantea al situar el problema en el
**nivel 3, *pattern recognition***. La respuesta a su Q1 deja de ser «KAN sí» o «KAN no» y pasa a ser
**«KAN sí, para la tarea que usted identificó; no para la otra»** — con las dos mitades medidas.

El KAN es además **estable entre folds** (0,711 a 0,774) mientras el MLP oscila (0,459 a 0,790), y
por eso el intervalo del MLP cruza cero pese a una media positiva. La ventaja del KAN es tanto de
nivel como de varianza.

## 4. El brazo recurrente ve más, y por eso no se compara con los demás

`recurrent` alcanza **0,8981**, pero **ve la resiliencia de la configuración anterior**, que ningún
otro brazo ve. Es deliberado: la activación de la Fig. 5 de Garrido es literalmente *«¿es la medida
en la configuración x mayor que en la x−1?»*, así que un surrogate de secuencia **necesita** ese
valor.

Por eso se juzga contra `linear_lagged`, un clásico con **la misma entrada**:

> **recurrente − lineal-con-lag = +0,2053 [+0,1051, +0,3056]**

**Comparar el 0,8981 contra el 0,6306 del lineal sería medir el conjunto de información, no la
arquitectura.** No se hace, y el artefacto lo declara.

## 5. Lo que esto NO autoriza

**Es predicción, no control.** El endpoint es R² fuera de fold sobre `R_cobb_douglas`. Que una red
prediga mejor la superficie **no** implica que un controlador neuronal mejore el servicio, y este
resultado **no autoriza ningún aprendiz de control**. La Puerta C sigue siendo la que decide si hay
algo que amortizar.

**El «techo» no era un techo.** `train_cell_mean_comparator` es la media por celda calculada en
train, y el KAN la supera (0,7446 > 0,6931) porque generaliza entre celdas donde la media no puede.
Sigue siendo una referencia útil, pero no es una cota superior y no se citará como tal.

**Grado de desarrollo.** Ocho semillas, cinco folds, sin bloque virgen, sin adjudicación. Una
confirmación necesitaría tapas frescas y su propio preregistro.

## 6. Lo que sí se puede decir

> Con el mismo dato, el mismo objetivo y los mismos comparadores clásicos reproduciendo a 5e-05, un
> KAN correctamente ajustado captura **+0,1140 [+0,0614, +0,1665]** de R² sobre el mejor baseline
> clásico declarado de antemano, en la tarea de reconocimiento de patrones que Garrido identifica
> como el nivel donde vive el hueco. Es la primera prima neural del proyecto que sobrevive a su
> propio protocolo.
