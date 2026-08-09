# Preregistro — Puerta B: la prima estaba disponible y nuestras redes no la cogieron

**Fecha:** 2026-08-09. **Congelado antes de escribir el runner.**
Contrato marco: `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`.
**Rol:** `DEVELOPMENT_REANALYSIS_NO_NEW_SEEDS`.

## 1. Lo que dice el artefacto y lo que no

`results/headroom/cd_surface_prediction_premium/result.json` =
`PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED`:

```
train_cell_mean_comparator 0.693   <- techo alcanzable con la informacion de train
spline_buffer              0.637
linear_interactions        0.631   <- baseline primario, declarado de antemano
tree                       0.622
kan                        0.602
backprop                   0.584
```

Margen disponible sobre el baseline primario: **+0,0625**. Y las dos redes quedaron **por debajo
del lineal**. La conclusión que el artefacto NO puede sostener es «el entorno no tiene prima»; lo
que muestra es que **nuestras redes no llegaron ni al lineal**.

## 2. La hipótesis a falsar, y es sobre nosotros

> **H_B: las redes perdieron por cómo se ajustaron, no por lo que el entorno ofrece.**

No es especulación. Los brazos neuronales de esa corrida son, leídos del código
(`scripts/build_garrido_fig5_surrogate.py`):

* **MLP** de 16-16 tanh, **600 pasos de Adam fijos** a `lr=0.01`;
* **KAN** de ancho `[d, 4, 1]`, `grid=3`, los mismos 600 pasos;
* **sin división de validación, sin early stopping, sin regularización, sin selección de
  hiperparámetros, y una sola semilla de inicialización por fold**.

Enfrente, los brazos clásicos son **OLS en forma cerrada** —óptimo para su base por construcción— y
un árbol. **Es un ajuste sin sintonizar contra un óptimo analítico.** Que el lineal ganara no dice
nada del entorno.

## 3. Qué cambia, y qué está prohibido cambiar

**Cambia sólo el ajuste de los brazos neuronales**, y de forma simétrica para MLP y KAN:

1. **estandarización del objetivo y de las variables**, calculada **sólo en train** de cada fold;
2. **partición interna de validación** dentro de train, con **early stopping** sobre ella;
3. **rejilla pequeña de hiperparámetros declarada aquí y cerrada** — ancho `{16, 64}`, `lr`
   `{3e-3, 1e-2}`, `weight_decay` `{0, 1e-4}`, presupuesto máximo 5.000 pasos con parada temprana —
   seleccionada **en la validación interna, jamás en test**;
4. **cinco semillas de inicialización por fold**, promediadas, para que la comparación no sea contra
   el ruido de una inicialización;
5. **brazo nuevo: surrogate recurrente** sobre la **secuencia de configuraciones** —la Fig. 5 de
   Garrido implementada como predictor y no como scorer— con el mismo presupuesto.

**Prohibido y declarado:** no se toca el objetivo, ni los folds, ni la calibración por fold (el
leak de κ̇ ya está cerrado), ni el baseline primario, ni el SESOI de `0,05`. Los brazos clásicos
reciben **el mismo tratamiento de estandarización** para que la simetría sea real.

**El baseline que decide sigue siendo `linear_interactions`**, elegido por principio antes de ver
resultados. `spline_buffer` fue elegido *sobre test* en la corrida original y ese defecto está
declarado allí; aquí se reporta pero no decide.

## 4. Falsadores, y por qué cada uno puede fallar

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_ceiling_still_above_the_primary` | `train_cell_mean_comparator > linear_interactions` | si el techo desaparece al estandarizar, no había margen que capturar |
| `f2_classical_arms_reproduce` | los R² clásicos se mueven menos de 0,02 respecto al artefacto original | si se mueven, cambié algo más que el ajuste neuronal |
| `f3_tuning_used_only_inner_validation` | la selección no ve el fold de test | es exactamente el pecado que le achacamos al `spline_buffer` original |
| `f4_networks_now_reach_the_linear` | MLP y KAN ≥ `linear_interactions` | **puede fallar**, y entonces H_B es falsa: no era el ajuste |
| `f5_neural_premium_over_the_primary` | `mean ≥ 0,05` y `CI95 low > 0` contra `linear_interactions` | puede fallar aunque `f4` pase: llegar al lineal no es superarlo |
| `f6_recurrent_arm_is_reported` | el brazo recurrente aparece con su presupuesto | sin él no se responde la Fig. 5 como predictor |
| `f7_budgets_are_matched` | parámetros y pasos comparables entre MLP, KAN y recurrente | un presupuesto desigual mide capacidad |

## 5. Reglas de lectura, en orden

1. Si `f2` falla → `BLOCKED_INSTRUMENT`: cambié más que el ajuste y nada es comparable.
2. Si `f4` falla → `NETWORKS_WERE_NOT_THE_PROBLEM`, y la lectura correcta pasa a ser que la
   superficie **no** admite un aproximador mejor que el lineal en esta parametrización. **Es un
   resultado**, y además refuerza el negativo del portafolio.
3. Si `f4` pasa y `f5` no → `NETWORKS_REACH_THE_LINEAR_BUT_DO_NOT_BEAT_IT`: el margen existía y
   sigue sin capturarse, ahora con las redes bien ajustadas. También es publicable, y es la
   respuesta honesta al nivel 3 de Garrido.
4. Con `f4` y `f5` → `SURFACE_PREMIUM_CAPTURED`, que es la primera prima neural de **predicción**
   del proyecto.

**No hay rama que diga «casi»**, y el SESOI de `0,05` es el de la corrida original: no se mueve
después de ver el intervalo.

## 6. Custodia

**No se abren semillas.** Se reusan las ocho del artefacto original (`6900001–6900008`) porque el
objeto de estudio es **el ajuste**, no el entorno: cambiar las tapas a la vez que el ajuste haría
inseparable qué causó la diferencia. Grado de desarrollo, sin posibilidad de confirmación.
