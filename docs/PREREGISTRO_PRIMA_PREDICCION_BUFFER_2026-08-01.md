# Preregistro — prima de predicción sobre el perfil de buffer

**Escrito y commiteado ANTES de correr.** Runner:
`scripts/run_buffer_profile_prediction_premium.py`.

## Por qué esta superficie y no otra

Q1 se respondió sobre el panel `ρ → ReT`, donde un lineal ya logra `R² = 0,9697` y las redes
añaden `+0,0166` y `+0,0216` — **significativo, despreciable** contra el SESOI preregistrado de
`0,05`. La réplica evidente era: *la superficie era lineal, así que no había nada que hacer*.

G1 midió una superficie que **no** lo es: el perfil de buffer tiene `1 − R²` lineal de **0,0790**
sobre `ret_excel`, con óptimo **interior** en dos regímenes. Y esa curvatura es **física, no de la
métrica** — `flow_fill_rate`, que no tiene término de coste, pone el óptimo en el mismo nivel.

> **Ésta es la primera superficie del proyecto donde la premisa de la pregunta se cumple.**

## Hipótesis

> **P1.** Sobre el perfil de buffer, una red (backprop o KAN) supera al lineal en `R²` held-out
> por al menos el **SESOI de 0,05**, con IC95 pareado excluyendo el cero.

## Diseño

* **Objetivo**: `ret_excel_risk_conditional`. **No** Cobb-Douglas: `f8` de G1 mostró que su
  regla de exponentes degenera aquí (τ con el 79 % del presupuesto).
* **Rejilla**: **17** niveles de buffer × 3 familias × 3 escaladas (`×1, ×3, ×5`) × 10 semillas
  = **1.530 episodios**. Más niveles que en G1 porque un ajuste no lineal necesita resolución.
* **Rasgos**: `ρ` (buffer escalado) + diseño de riesgo. **Sin drivers** — un driver es
  post-simulación y los cuatro suman ReT exactamente.
* **Modelos**: constante, lineal, backprop (MLP 16-16 tanh), KAN — **importados del subrogado de
  Q1**, no reimplementados, para que la comparación sea contra las mismas redes bajo el mismo
  protocolo.
* **Validación cruzada agrupada por semilla**, 5 folds.
* **Cadencia**: `sim.run()`. Se declara porque `ret_excel` depende de la cadencia y estos valores
  **no** son comparables con los de G1, que usó paso diario.
* **Semillas**: `6 800 001…` vírgenes, por escaneo real.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_surface_actually_has_curvature` | **la premisa**, y se **recalcula aquí** en vez de confiar en el artefacto de G1. Preguntar otra vez sobre una superficie lineal no probaría nada |
| `f2_no_driver_leakage` | un driver entre los rasgos entregaría la respuesta |
| `f3_folds_are_grouped_by_seed` | una semilla en train y test dejaría memorizar el ruido de ese episodio e inflaría **todos** los `R²`, incluido el lineal |
| `f4_linear_baseline_is_not_a_straw_man` | una prima sobre un lineal mal ajustado mediría nuestra incompetencia, no la no linealidad |
| `f5_sesoi_was_fixed_in_advance` | elegir el umbral tras ver la diferencia es cómo una ganancia despreciable se convierte en titular |
| `f6_seeds_are_virgin` | escaneo real de artefactos sellados |

## Regla de lectura, fijada de antemano

* **Alguna red supera el SESOI con IC95 > 0** → `NEURAL_PREMIUM_ON_CURVED_SURFACE`. Sería la
  **primera prima neural del proyecto**, y respondería Q1 con una condición: *las redes aportan
  cuando la superficie tiene curvatura, y la del panel de Garrido no la tiene*.
* **Ninguna la supera** → `NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE`. **Y ése es el
  resultado más fuerte disponible para el paper**: ya no se puede decir «la superficie era
  demasiado fácil». Habría curvatura medida y aun así ninguna prima que valga la pena.

**Lo que no autoriza en ningún caso:** entrenar control. Esto es predicción; el control exige
además el gate de headroom completo, que sigue sin abrirse.
