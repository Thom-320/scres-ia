# Preregistro — el delay de cumplimiento: constante contra distribución

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Ninguna constante cambiada. Toda cifra congelada permanece como fue reportada.

Sucede a `docs/RESULTADO_AUTOTOMIA_2026-07-30.md` (`3175110`), que estableció que ningún
**valor** de la constante puede funcionar.

---

## 1. El defecto, y por qué no es un número

`GARRIDO_FULFILLMENT_DELAY_HOURS = 54,0` es el `CTj` de toda orden servida desde stock. Es
una **masa puntual**: el 69,2% de nuestras órdenes termina en exactamente el mismo instante.

| | Garrido (21.667 filas) | nuestro |
|---|---|---|
| `min(CTj)` | 48,0074 | = la constante |
| p1 / p5 / p25 / p50 | 48,41 / 50,42 / 75,00 / 101,45 | constante en los cuatro |
| filas en [48,007, 48,06] | 98 = **0,45%** | **69,2%** |
| `autotomy_share` | 0,004 | 0,000 (con 54) / 0,659 (con 48,0074) |

Medido en `3175110`: con el delay en 54 la masa cae por encima de `LT = 48` y la autotomía
**nunca** dispara; en 48,0074 cae dentro de la banda y dispara el **69%**. Las tres
tolerancias probadas dieron resultados **bit-idénticos**, porque una masa puntual no tiene
cola. **Ningún valor constante produce 0,44%.**

## 2. Lo que se propone

Reemplazar la constante por una **distribución** `demand_on_hand_fulfillment_delay`, con
soporte en `[48,0074, ∞)`. La forma se declara **antes** de ver su ajuste.

| brazo | forma | parámetros |
|---|---|---|
| **A** (statu quo) | constante | 54,0 |
| **D1** | `48,0074 + Exp(β)` | β del p50 observado |
| **D2** | `48,0074 + Lognormal(μ, σ)` | de p25 y p50 observados |
| **D3** | `48,0074 + Weibull(k, λ)` | de p25 y p50 observados |

**Los parámetros se derivan por momentos de cuantiles observables suyos, no por búsqueda.**
Con `min = 48,0074`, `p25 = 75,00` y `p50 = 101,45`:

* **D1** tiene un solo parámetro: `β = (101,45 − 48,0074) / ln 2 = 77,1`. Queda **fijado por
  el p50 y nada más**, y su p25 predicho es una **predicción falsable** (§4.2).
* **D2** y **D3** tienen dos y se fijan con p25 y p50 exactamente. **Su p5 y p95 predichos
  son falsables.**

**Ningún parámetro se barre.** Si la forma declarada no reproduce los cuantiles que no se
usaron para fijarla, eso se reporta como el fallo de esa forma, no se re-parametriza.

## 3. Lo que esto NO es

No es un ajuste libre. Es la misma disciplina de la §7 del preregistro de autotomía: se
reproduce una **distribución publicada suya**, juzgada por los cuantiles que **no**
entraron en la estimación.

**Prohibido explícitamente:** añadir una cuarta forma después de ver los resultados;
ajustar `β`, `σ` o `k` fuera de las fórmulas de arriba; o usar cuantiles fuera de
`{min, p25, p50}` para estimar.

## 4. Predicción, en `d_k`, la misma escala que la regla

Declaro dirección **y** el riesgo, tras haber fallado una dirección ayer (`3175110` §3).

1. **`autotomy_share` mejora en ambas familias** en al menos una de D1/D2/D3. Es la razón de
   ser del cambio: una cola inferior delgada produce una fracción pequeña de órdenes en la
   banda del piso. **Si ninguna forma la mejora, el mecanismo está mal** y esta línea se
   cierra.
2. **Predicción falsable de forma, independiente del criterio:** el p25 predicho por D1 es
   `48,0074 + 77,1·ln(4/3) = 70,2` contra un p25 observado de **75,00**, un error del 6,4%.
   Si D1 sale peor que D2 y D3 en los momentos, eso es **consistente** con que una
   exponencial sea demasiado ligera en el cuerpo. Lo digo ahora para que no sea una
   racionalización después.
3. **`ret_mean`: sin dirección declarada, y esto es deliberado.** El cambio tiene dos
   efectos opuestos — baja el piso (sube `0,5/RPj`) pero **alarga** el cuerpo hacia el p50
   de 101 (baja `0,5/RPj`). No sé cuál domina y no voy a fingir que sí. Conserva veto igual.
4. **`rpj_mean` y `rpj_p95` deberían mejorar**, porque `RPj ≈ CTj` y nuestro `CTj` p50 pasa
   de 54 a ~101, que es su valor. **No predigo nada sobre la saturación**, que sigue sin
   mecanismo tras cuatro refutaciones.

## 5. Falsadores del instrumento

1. **Brazo A reproduce lo congelado.** Sobre las raíces 2.500.001–12 debe dar las cifras de
   `results/metric_audit/autotomy_arms_v1/result.json`. Si no, no se reporta nada.
2. **En D1–D3, `min(CTj) ≥ 48,0074`** y **ninguna orden con `CTj < LT`**. Sus 21.667 filas
   lo cumplen sin excepción.
3. **En D1–D3, `CTj` tiene más de 500 valores distintos** en una corrida. Es la prueba
   directa de que dejó de ser masa puntual; hoy son 46 con 69,2% en uno solo.
4. **El `p50(CTj)` realizado de cada brazo cae dentro del ±10% del p50 objetivo (101,45)**
   para órdenes no bloqueadas. Verifica que la parametrización se implementó como se declaró
   y no que el resultado sea bueno.

## 6. Criterio de aceptación

Dominancia sobre los seis momentos, referencia `fidelity_reference_v3`, `EPSILON = 0,5`,
**ambas familias objetivo**.

**Una forma se adopta si y solo si:**

* `d_k(autotomy_share)` **mejora en ambas familias**; **y**
* `d_k(ret_mean)` **no empeora más de `EPSILON` en ninguna de las dos**; **y**
* ningún otro momento empeora más allá de `EPSILON` en ninguna familia; **y**
* los cuatro falsadores de §5 pasan.

**Si más de una forma califica**, se adopta la de menor suma de `d_k` sobre las dos
familias, y **se reportan las tres** con sus cuantiles predichos contra los observados.

**Si ninguna califica**, se reporta el negativo con la forma que más se acercó, y la
constante se conserva **con su defecto documentado en `config.py`**, no en silencio.

**Prohibido** elegir forma o parámetros por el `H_PI` que produzcan, por el signo de un
contraste MPC-contra-estático, por que una familia cruce un umbral de servicio, o por que el
resultado sea publicable.

## 7. Declarado por adelantado

| ítem | valor |
|---|---|
| constante **no** tocada | `LEAD_TIME_PROMISE = 48` (tesis §6.8.2 p.111) |
| piso del soporte | 48,0074 = `min(CTj)` de sus nueve hojas R1r |
| formas | constante (A), Exp (D1), Lognormal (D2), Weibull (D3) |
| estimación | por cuantiles `{min, p25, p50}`; **sin barrido** |
| cuantiles reservados para falsar | p1, p5, p95 |
| raíces | **2.600.001–2.600.012**, disjuntas de todo bloque previo |
| raíces de regresión | 2.500.001–12, solo para el falsador 1 |
| familias | R1r y R2r, ambas objetivo |
| configuración | `S = 1`, buffers 0, nivel «+» de la Tabla 6.12 |
| defaults del resto | `elapsed`, `serial`, `clamped`, `autotomy_predicate = le` |
| criterio | dominancia seis-momentos, `EPSILON = 0,5`, `ret_mean` con veto |
| predicción | §4, en `d_k`, con §4.3 declarado como **sin dirección** |

## 8. Alcance

**Nada se reetiqueta.** Program Q, la confirmación H2/H3, el buffer gate, las 90
configuraciones y la frontera conjunta conservan sus cifras. Si una forma se adopta, **abre
un cuerpo de resultados nuevo**.

**Fuera de alcance:** la saturación de `RPj` (cuatro mecanismos refutados hoy, ninguno
propuesto), el predicado de autotomía en banda (medido, no adoptado, `3175110`), y el clamp
(medido, no adoptado, `a0912bd`).

**Nota sobre el predicado.** Si la distribución sola lleva `autotomy_share` cerca de 0,004
con el predicado `<=` embarcado, la banda **no hace falta**, y eso sería la confirmación más
limpia posible del diagnóstico de §1. Si hace falta la banda además, se reporta como
resultado conjunto y requiere su propio preregistro; **no se añade aquí a mitad de camino.**

## 9. Firma

Requiere aprobación del PI antes de ejecutar.

La decisión que no me corresponde: si sustituir una constante ajustada por una distribución
estimada de sus datos es un cambio de modelo aceptable para el paper, o si la línea de
reproducción debe conservar el 54 y la distribución abrirse en paralelo. Mi lectura: la
constante ya está documentada como fallida en tres frentes en `config.py:124`, y el tercero
—ser constante donde la fuente es distribución— no lo arregla ningún número. Pero es un
cambio de modelo, no de código, y por eso es tuyo.
