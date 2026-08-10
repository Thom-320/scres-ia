# Corrección — `train_cell_mean_comparator` no es un techo, y «+0,0625 de margen» sale del borrador

**Fecha:** 2026-08-09 · **Origen:** re-adjudicación `results/program_n/gate_b_readjudication/result.json`
**No edita ningún artefacto ni documento fechado.** Los corrige por sucesión, que es la única vía.

## Lo que se dijo

El artefacto `results/headroom/cd_surface_prediction_premium` (`PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED`)
introdujo `train_cell_mean_comparator = 0,693` como **techo disponible**, con el baseline primario
en 0,631, y de ahí salió la frase que circuló en el plan y en las lecturas de la Puerta B:

> margen disponible **+0,0625**, y backprop y KAN quedaron por debajo del lineal

## Lo que la medición dice

`train_cell_mean_comparator` predice la media de entrenamiento de la celda
`(familia, escalación, buffer)`. Es un predictor más, no una cota. Y es superado en **las cuatro**
corridas de la Puerta B, incluida la de desarrollo donde se acuñó el término:

| corrida | endpoint | brazos que lo superan |
|---|---|---|
| `gate_b_cd_surface` | Cobb-Douglas | `kan_tuned`, `mlp_tuned`, `recurrent` |
| `gate_b_confirmation_v2` | Cobb-Douglas | `kan_tuned`, `linear_lagged`, `recurrent` |
| `gate_b_confirmation_v3` | Cobb-Douglas | `kan_tuned`, `mlp_tuned`, `recurrent` |
| `gate_b_sensitivity_ret_excel` | `ret_excel` | `linear_lagged`, `recurrent` |

En `ret_excel` lo supera incluso un **modelo lineal** con la entrada retardada.

## Lo que se retira

* La palabra **techo** aplicada a `train_cell_mean_comparator`. Acota a su propia familia —
  predictores sin la identidad de celda — y nada más.
* La cifra **+0,0625 de margen disponible**, y cualquier construcción de la forma «quedaba X por
  capturar». No había una cota de la que restar.
* `f1_ceiling_still_above_the_primary` de la Puerta B **sigue siendo un falsador válido** —
  comprueba que el comparador de medias de celda supera al primario, y eso es cierto y puede
  fallar— pero su **nombre** afirma más de lo que mide. Se lee como «el comparador de medias de
  celda sigue por encima del primario», nunca como «el techo sigue ahí».

## Lo que NO se retira

El hallazgo central del artefacto predecesor: **había margen sobre el baseline lineal y sus redes
no lo cogieron**. Eso se sostiene y además está ahora capturado — `gate_b_confirmation_v3`, bloque
virgen, MLP `+0,1081 [+0,0601, +0,1561]` contra el mejor clásico de su clase de información.

Lo que cambia es que ese margen **no tiene una cota superior conocida**, así que no puede
expresarse como fracción de nada.

## Por qué pasó

El comparador se nombró «techo» por su papel esperado, no por una propiedad medida. Nadie
comprobó que acotara. Es el mismo patrón que el `passed: True` codificado a mano y que la
cláusula de orden de `f2`: **un nombre que afirma, sin una medición que lo respalde**.

La regla que deja: un artefacto no puede llamar techo, cota, óptimo ni límite a una cantidad sin
un falsador que compruebe que ningún brazo la supera. Si algún brazo la supera, es un comparador.
