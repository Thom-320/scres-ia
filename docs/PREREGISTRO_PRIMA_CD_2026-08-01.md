# Preregistro — prima de predicción sobre la superficie de Cobb-Douglas, con tres reparaciones

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_cd_surface_prediction_premium.py`.
Sucede a `results/headroom/buffer_prediction_premium/result.json`, que midió **la superficie
equivocada** y con dos defectos de método.

## Por qué se rehace

1. **La superficie.** El G1 corregido mostró que `ret_excel` y `flow_fill_rate` son **monótonas**
   y que la única con **máximo estrictamente interior** es **Cobb-Douglas**. La corrida anterior
   midió `ret_excel` presentándola como «la superficie curva». Se mide la que curva.
2. **El baseline era aditivo.** Sin productos `buffer × familia` ni `buffer × escalada`, una red
   podía «ganar» representando interacciones que **un modelo clásico mejor también representa**.
   Ahora el comparador es **el mejor de** {lineal aditivo, lineal con interacciones y cuadrático}.
3. **La inferencia usaba 1,96 con cinco folds.** Con 4 grados de libertad el multiplicador es
   **t(0,975, 4) = 2,776**. El signo no cambia; la aritmética estaba mal igual.

## Y la afirmación que retiro

Escribí que *«la curvatura está por debajo del ruido»* comparando **0,0763** —falta de ajuste
lineal sobre **medias de perfil**— contra **0,3174** —error predictivo sobre **episodios
individuales**. **Son escalas distintas y la comparación no sostiene la afirmación. Queda
retirada.**

Se sustituye por la cantidad que **sí** es comparable: el **oráculo de media por celda**, un
modelo que conoce la media verdadera de cada `(familia, escalada, buffer)`. Su `R²` es el **techo**
que cualquier función de esos rasgos puede alcanzar, de modo que

    prima DISPONIBLE = R²(oráculo) − R²(mejor clásico)

es el máximo que **cualquier** modelo podría ganar. Si esa brecha está bajo el SESOI, **ninguna
arquitectura puede cobrarla**, y eso sí es una afirmación medida.

## Diseño

* **Objetivo**: `R_cobb_douglas`. 17 niveles × 3 familias × 3 escaladas × 8 semillas = **1.224
  episodios**, cadencia diaria (la que exige `CobbDouglasRecorder`).
* **Modelos**: constante · lineal aditivo · **lineal con interacciones y cuadrático** ·
  **oráculo de media por celda** · backprop · KAN. Las redes se importan del subrogado de Q1.
* **CV agrupada por semilla**, 5 folds, intervalos **t**.
* **Semillas** `6 900 001…` vírgenes por escaneo real.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_target_is_the_curved_surface` | si CD tampoco tiene máximo estrictamente interior aquí, esta corrida repite el error de la anterior |
| `f2_baseline_includes_interactions` | un baseline aditivo dejaría ganar a la red por algo que un clásico mejor también captura |
| `f3_inference_uses_t_not_normal` | 1,96 con cinco folds estrecha el intervalo indebidamente |
| `f4_available_premium_is_measured_not_asserted` | sustituye la comparación de escalas incompatibles que retiro arriba |
| `f5_folds_grouped_by_seed` | una semilla compartida infla todos los `R²` |
| `f6_seeds_are_virgin` | escaneo real de artefactos sellados |

## Regla de lectura

* **Alguna red ≥ SESOI con IC95 (t) > 0** → `NEURAL_PREMIUM_ON_CD_SURFACE`.
* **Prima disponible < SESOI** → `NO_PREMIUM_AND_NONE_WAS_AVAILABLE`. **Es el desenlace más
  fuerte**: no es que las redes fallaran, es que **no había nada que ganar** y está medido.
* **Prima disponible ≥ SESOI pero ninguna red la alcanza** → `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED`,
  que sería un resultado sobre **las arquitecturas**, no sobre el entorno, y el más interesante
  de los tres.

**No autoriza entrenar control.** Sigue exigiendo el gate de headroom completo.
