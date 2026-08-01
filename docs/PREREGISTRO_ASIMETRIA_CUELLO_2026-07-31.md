# Preregistro — Fase 1A′: contención **EN el cuello** con reclamantes **asimétricos**

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_bottleneck_asymmetry_v1.py`.
Prueba la hipótesis que dejó la Fase 1A, con el instrumento que ya la midió una vez.

## La hipótesis, y de dónde sale exactamente

La Fase 1A puso contención **aguas abajo** del cuello, con reclamantes **simétricos**, y midió
`H_regime = 1,5e-04` pese a renunciar al 17 % del flujo. El diagnóstico `v1_1` refinó el
diagnóstico y refutó mi primera explicación: **el óptimo SÍ se mueve** (0,1 en `R2r`, 0,9 en
`R1r+R2r`) y **la superficie es una U profunda** (78–92 % de dispersión), pero el reparto 0,1
está cerca del máximo en **ambas** familias, así que saber el régimen cuesta ~3 % de una cifra
pequeña.

Program O midió `H_PI = 0,1515` con dos productos **no fungibles** compitiendo por **Op5–Op7**,
que es el cuello real (margen 2,6 %). Dos diferencias contra la Fase 1A, y ambas son variables
del instrumento, no supuestos:

| | Fase 1A | Program O |
|---|---|---|
| dónde | **aguas abajo** del cuello | **EN** el cuello (Op5–Op7) |
| reclamantes | **simétricos** (hash 50/50) | **asimétricos** (`dominant_share`) |

> **H1.** `H_PI` **crece con la asimetría**: es monótona (o al menos creciente) en
> `dominant_share`, y en `0,5` —el caso simétrico, que es la condición de la Fase 1A— cae al
> mínimo.
>
> **H2 (mecanismo).** Con `complete_substitution=True` (los productos se sustituyen ⇒ el recurso
> es fungible) `H_PI` colapsa. Es el mismo control que dio **exactamente 0** dos veces hoy.
>
> **H3 (escala).** En la asimetría alta, `H_PI ≥ 0,01`.

## Lo que este experimento NO es

**No es un rescate de Program O.** Aquel contrato prohíbe cambiar controlador, celdas, métrica,
física, comparador, umbral o guardarraíl **para rescatar su veredicto**, y su veredicto era sobre
`H_obs` —conversión observable segura— tras una validación sellada. Aquí:

* el estimando es **otro**: `H_PI` **como función de la asimetría**, no `H_obs`;
* las semillas son **otras y vírgenes** (`7 600 001…`; los bloques quemados `7 420 049–96` y
  `7 430 001–48` no se tocan);
* **no se entrena nada** y no se afirma nada sobre un aprendiz;
* tiene **su propio preregistro** y su propia regla de lectura.

Usar un instrumento validado para responder una pregunta distinta no es reabrir la anterior. Si
sale positivo, **no** resucita la conversión de Program O: sólo dice dónde hay headroom.

## Diseño

* **Palanca (el clarividente)**: calendario de 8 semanas, una acción por semana en `{0, 3}` =
  `(P_C,P_C,P_C)` o `(P_H,P_H,P_H)` sobre las tres franjas de ensamblaje. **256 calendarios,
  enumerados exactamente** — sin optimizador, sin ruido de búsqueda.
* **Asimetría**: `dominant_share ∈ {0,5 · 0,6 · 0,7 · 0,8 · 0,9}`. **0,5 es el caso simétrico**,
  es decir la condición bajo la que la Fase 1A midió casi cero.
* **Persistencia del régimen**: `regime_persistence ∈ {0,5 · 0,9}`.
* **Control de fungibilidad**: `complete_substitution ∈ {False, True}`.
* **Riesgos**: `R2r` activo (`R21–R24`), que es donde vive R23.
* **Semillas**: `7 600 001…` vírgenes, CRN entre celdas.
* **Métricas**: `ret_excel_risk_conditional` **primaria** (la de toda la campaña, para poder
  comparar con la Fase 1A en la misma escala); `ret_excel_visible_clipped_0_1` acotada y
  **`worst_product_fill`** al lado. El fill por producto importa aquí más que en ninguna fase:
  la U de la Fase 1A sugiere que abandonar un destino **mejora** `ret_excel`, y con dos productos
  eso se puede ver directamente.

`H_PI = mean_seed[max_calendario] − max_calendario[mean_seed]`, con LCB95 por bootstrap sobre
semillas.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_asymmetry_is_actually_asymmetric` | la demanda realizada de `P_C`/`P_H` debe separarse al subir `dominant_share`; si no, la palanca no crea asimetría y H1 es vacua |
| `f2_substitution_control_binds` | con sustitución completa `H_PI` debe caer; si no cae, el control no controla y H2 no se ha probado |
| `f3_the_calendar_changes_the_outcome` | si los 256 calendarios puntúan igual, no hay decisión y `H_PI` mide ruido |
| `f4_H_PI_is_non_negative` | `mean[max] ≥ max[mean]` por construcción; un negativo sería un bug de agregación |
| `f5_seeds_are_virgin_and_disjoint_from_program_o` | tocar `7 420 049–96` o `7 430 001–48` contaminaría una validación quemada |
| `f6_this_is_not_a_program_o_rescue` | declaración estructural verificable: estimando, semillas, ausencia de aprendiz y preregistro propio |

## Regla de lectura, fijada de antemano

* **`H_PI` crece con `dominant_share`, alcanza ≥ 0,01 con `LCB95 > 0`, y colapsa con sustitución**
  → `ASYMMETRY_AT_THE_BOTTLENECK_IS_THE_MISSING_INGREDIENT`. Es la receta de headroom del
  proyecto, y **autoriza la Fase 3** sobre esta palanca.
* **`H_PI` alto pero PLANO en `dominant_share`** → lo que importa es **dónde** (el cuello), no la
  asimetría. Corrige mi hipótesis y redirige la Fase 1B.
* **`H_PI < 0,01` en todas las asimetrías** → ni en el cuello ni con asimetría. Junto con la
  Fase 1A cerraría las dos formas de contención construibles aquí, y el paper pasaría de «no
  encontramos» a **«medimos que no lo hay, y por qué»** — que sigue siendo una respuesta a
  Garrido, no un vacío.

**Y una advertencia que me impongo:** si `worst_product_fill` **cae** mientras `ret_excel` sube,
el headroom es la métrica premiando el abandono de un producto. En ese caso **no lo reporto como
headroom** — lo reporto como defecto de métrica, que es lo que sería.
