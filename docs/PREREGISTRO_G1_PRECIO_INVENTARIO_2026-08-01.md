# Preregistro — G1: ¿el precio del inventario genera curvatura?

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_g1_buffer_price_cobb_douglas.py`.
Primer generador del programa `docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md`.

## La hipótesis, y por qué es la más *faithful* de las cuatro

Una prima neural exige que la superficie de respuesta sea **no lineal** en lo que el aprendiz ve.
Medimos `R² = 0,9697` para un lineal sobre `ρ → ReT`, y las redes ganan sólo `+0,0166` y `+0,0216`
— significativo, despreciable. **No es que las redes fallen: no hay no linealidad que aprender.**

Una causa concreta: **`ret_excel` no cobra el inventario**. Más buffer es débilmente mejor, así
que la superficie es **monótona** y su óptimo está en un extremo. Una monotonía la representa un
lineal exactamente.

**El índice Cobb-Douglas del propio Garrido sí lo cobra.** IJPR 2024 Eq. (5):
`R = 0,024·Lnζ − 0,026·Lnε + 0,04·Lnφ − 0,06·Lnτ − 0,1771·Lnκ̇`, donde `ζ` (inventario medio) entra
**positivo** pero `κ` incluye el coste de mantener `c_i`. **Ésa es exactamente la estructura a dos
lados que falta.**

> **G1.** Bajo Cobb-Douglas el óptimo de buffer es **interior** —ni 0 ni el máximo— **y se mueve
> con la frecuencia de riesgo**. Bajo `ret_excel` no.

Si se cumple, hay **curvatura** *y* **dependencia del régimen** — las dos condiciones del
programa — **sin salir de la métrica de Garrido y sin inventar física**.

## Diseño

* **Palanca**: buffer del batallón (`op9_rations`), **9 niveles** equiespaciados entre 0 y 1.344 h
  de cobertura de demanda.
* **Regímenes (6)**: `{R1r, R2r, R1r+R2r}` × `{base, frecuencia ×3}`.
* **Métricas, en una sola corrida y una sola cadencia**: `R_cobb_douglas` (**primaria**),
  `ret_excel_risk_conditional` (**control de contraste**), `flow_fill_rate`, y la clave
  `service_first_v2`.
* **Semillas**: `6 700 001…` vírgenes, verificadas por **escaneo real** de todos los artefactos
  sellados — no `passed: True`.
* **Cadencia diaria**, que es la que exige `CobbDouglasRecorder`. Se declara porque `ret_excel`
  **depende de la cadencia** y comparar contra artefactos de `run()` sería inválido.
* `κ̇` es **relativa al conjunto**, así que el conjunto de comparación son las **54 celdas**
  (9 buffers × 6 regímenes) — que es el uso para el que el índice está definido.

## Falsadores

Con la regla que me ha costado cinco correcciones esta semana: **si la afirmación es sobre un
mecanismo del código, se prueba el código; si es sobre una cantidad, se recalcula de forma
independiente; y siempre que se pueda, se inyecta el defecto y se exige que el falsador lo cace.**

| falsador | por qué puede fallar |
|---|---|
| `f1_kappa_actually_charges_inventory` | **control positivo + defecto inyectado**: subir el buffer debe subir `κ`, y con `c_i = 0` esa dependencia debe **desaparecer**. Si `κ` no responde al inventario, Cobb-Douglas no lo cobra y **G1 no tiene premisa** |
| `f2_the_buffer_lever_moves_the_system` | ya está medido inerte bajo `ret_excel` (`S_T ≈ 0,006`); si tampoco mueve el ledger físico, el barrido es vacuo |
| `f3_optimum_is_interior_not_at_a_bound` | **es la hipótesis misma**: si el óptimo CD está en 0 o en 1.344, no hay curvatura y G1 falla |
| `f4_ret_excel_stays_monotone` | **control de contraste**: si `ret_excel` también curva, la diferencia no es atribuible a la métrica |
| `f5_H_regime_is_non_negative` | `mean[max] ≥ max[mean]` por construcción; un negativo sería bug de agregación |
| `f6_cadence_is_disclosed` | comparar contra artefactos de otra cadencia sería inválido |
| `f7_seeds_are_virgin` | escaneo real de `results/**/result.json` |

## Regla de lectura, fijada de antemano

* **Óptimo interior Y `argmax` que se mueve entre regímenes** → `G1_GENERATES_CURVATURE`. Autoriza
  medir la **prima de predicción** sobre esta superficie (MLP/KAN vs lineal, SESOI 0,05 ya
  preregistrado). **No** autoriza entrenar control: eso exige además el gate de headroom.
* **Óptimo interior pero `argmax` fijo** → `CURVATURE_WITHOUT_STATE_DEPENDENCE`. Una red podría
  **predecir** mejor, pero **no habría política que aprender**. Es un resultado con contenido y
  hay que reportarlo tal cual.
* **Óptimo en un extremo** → `G1_DOES_NOT_GENERATE_CURVATURE`, y se pasa a G2 (umbral de
  autotomía). Dos de los cuatro generadores quedarían descartados **constructivamente**, que es
  precisamente el argumento del paper.

**Lo que NO autoriza en ningún caso:** entrenar un MLP o PPO. Eso requiere curvatura **más** el
criterio de apertura de headroom completo, y ese criterio se mantiene tal como está escrito.

**Y una advertencia que me impongo:** el resultado esperado por mí es que el óptimo sea interior.
Si sale en un extremo, lo reporto igual de rápido y con el mismo detalle. La predicción declarada
existe para poder fallar.
