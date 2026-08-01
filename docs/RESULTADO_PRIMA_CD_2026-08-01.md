# Resultado — la prima **estaba disponible y las redes no la capturaron**

**Artefacto:** `results/headroom/cd_surface_prediction_premium/result.json` (sello
`527227bd96eb8d8d…`, `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED`) · falsadores vinculantes **PASAN**
· 1.224 episodios · preregistro `docs/PREREGISTRO_PRIMA_CD_2026-08-01.md`.

## 1. Los números

`R²` held-out, CV agrupada por semilla, intervalos **t(4) = 2,776**:

| modelo | `R²` |
|---|---:|
| `train_cell_mean_comparator` | **0,6931** |
| spline en el buffer | 0,6365 |
| **lineal + interacciones** *(baseline primario)* | **0,6306** |
| árbol CART | 0,6225 |
| **KAN** | 0,6019 |
| **backprop** | 0,5841 |

| contraste | diferencia | IC95 (t) |
|---|---:|---|
| **margen disponible** (comparador de celda − primario) | **+0,0625** | [−0,0606, +0,1856] |
| KAN − primario | **−0,0287** | [−0,1048, +0,0473] |
| backprop − primario | **−0,0465** | [−0,1388, +0,0459] |

**Había margen por encima del SESOI (+0,0625 > 0,05) y las dos redes quedaron POR DEBAJO del
baseline clásico.** Es el tercer desenlace del preregistro: **un resultado sobre las
arquitecturas, no sobre el entorno**.

## 2. Cuatro límites que impongo al resultado

**(a) El comparador de celda NO es un techo matemático.** Es la media **empírica** de cada celda
sobre las filas de **train**. Su `R²` **estima** cuánto alcanzaría un modelo perfectamente
celda-a-celda con estos datos; **no acota** lo que cualquier función podría lograr. Por eso ya no
se llama «oráculo».

**(b) El objetivo VARÍA entre folds.** Cerrar la fuga de `κ̇` obliga a que cada fold calibre sus
propios exponentes y normalizador sobre train. El estimando es **«predecir un Cobb-Douglas
calibrado en train»**, no un índice CD único. Dentro de un fold todos los modelos ven el mismo
objetivo —lo que hace válidos los contrastes—; **entre folds la etiqueta no es la misma
cantidad**, y el paper debe decirlo así.

**(c) `best_classical` se elige mirando los folds de test.** Defecto conocido y declarado. Por eso
el contraste citable es contra el **primario pre-declarado por principio** (`linear_interactions`,
el clásico más expresivo, elegido **sin mirar resultados**), y el contraste contra el mejor
post-hoc va al lado, etiquetado.

**(d) El margen disponible cruza el cero** ([−0,061, +0,186]). *«Había prima»* es un estimado
puntual, **no está establecido**. Lo honesto: **no se puede descartar que el margen real fuese
nulo**, y con cinco folds no hay potencia para separarlo.

## 3. Lo que sí queda dicho, y es lo interesante

**Ninguna de las dos redes alcanza siquiera al clásico**, en una superficie donde:

* el máximo es **estrictamente interior** (`f1`),
* el clásico incluye **interacciones, cuadrático, spline y árbol**,
* y el **spline gana a ambas redes**.

Esa última observación es la que más dice: **la estructura es una curva suave en UNA variable**,
que es exactamente lo que un spline captura y donde una red con 1.224 filas y 5 folds generaliza
peor. **La no linealidad de este entorno es del tipo que los métodos clásicos ya resuelven.**

Eso responde Q1 con más precisión que «no hay prima»:

> **No basta con que la superficie sea no lineal. La prima neural exige una no linealidad que los
> métodos clásicos no capturen** — y la curvatura de una variable, suave y monótona a trozos, no
> lo es.

## 4. Procedencia y enmiendas

* **`f7` es una enmienda posterior al preregistro** y **no vinculante**. Cazó la fuga de `κ̇`
  —que era real— pero se añadió durante la implementación, y un check post hoc no puede
  convertirse retroactivamente en puerta científica.
* **Supersede** `results/headroom/buffer_prediction_premium/result.json`, que midió la misma
  pregunta sobre `ret_excel` —**monótona**— con baseline aditivo e intervalos a 1,96.
* La afirmación *«la curvatura está por debajo del ruido»* **sigue retirada**: comparaba falta de
  ajuste sobre medias contra error predictivo sobre episodios.
