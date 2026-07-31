# Preregistro — alinear la población puntuada y el modo de RPj

**Estado:** `CHANGE_A_WITHDRAWN__CHANGE_B_APPLIED_2026-07-30`.
> El **cambio A queda retirado**: el ledger canónico ya filtraba a servidas, así que
> era un no-op sobre una perilla muerta. Ver
> `docs/RETRACTACION_POBLACION_PUNTUADA_2026-07-30.md`.
> El **cambio B (modo `elapsed`) está aplicado** en `config.py:141`.

**Estado original:** `PREREGISTRATION_DRAFT_NOTHING_APPLIED`.
Ninguna constante cambiada. Toda cifra congelada permanece como fue reportada.

> **CORRECCIÓN 2026-07-31 — `scored_orders_per_year` estaba inflado un 8,33% y el titular
> «0,7 SD» es falso.** Los cuatro runners contaban órdenes sobre **8.736 h** (52 semanas) y
> dividían entre **`1,0 año`**, mientras la referencia usa el año de tesis de **8.064 h**
> (`fidelity_moments.py:78`). Además el numerador filtraba por warm-up y el denominador no.
> **Y mi primera corrección fue peor que el error.** Reescalé solo NUESTRO lado a 8.064 h y
> publiqué «5,33 / 7,14». Pero la referencia tiene la MISMA inconsistencia: sus hojas
> empiezan en `min(OPTj)` = 823-1.225 h (excluyen el warm-up, §6.8.2), y v3 divide por
> `max(OPTj)`, que sí lo incluye. Con la ventana puntuada consistente en **ambos** lados
> —`n / ((max OPTj − min OPTj) / 8064)`— el resultado es:
>
> | definición | R1r | R2r |
> |---|---:|---:|
> | publicada (dos errores que se cancelan en parte) | 0,98 | 2,52 |
> | mi corrección parcial (solo nuestro lado) | 4,07 | 7,14 |
> | **consistente en ambos lados** | **1,72** | **1,61** |
>
> **El valor correcto es ~1,7 SD**, no 0,7 y no 5,3. Tres capas de corrección sobre la misma
> cantidad, y las dos primeras eran mías. Enmienda:
> `contracts/paper_b_v2_amendment_2026-07-31.json`.

## 1. Qué se cambia, y por qué no es un ajuste

Dos correcciones **de definición**. Ninguna tiene un parámetro libre, ninguna se
elige por el resultado que produce, y las dos son verificables contra fuentes
externas a nosotros.

### Cambio A — puntuar solo órdenes servidas

**Evidencia:** las hojas canónicas de Garrido **no contienen ninguna orden sin
`OATj`**. Medido sobre CF1, CF3 y CF11: 4.241 / 2.151 / 2.165 filas, **0,0% sin
`OATj`** en las tres. Su población puntuada son, por construcción, las servidas.

La nuestra incluye **22,5% que nunca se sirven**, y verificamos que **no** es
truncamiento de horizonte: su distribución de posición coincide con la de las
servidas (p50 en 0,55 del horizonte contra 0,56; solo 22,7% en el último 20%, que
es el azar).

Es decir: hemos comparado dos poblaciones distintas y llamado a la diferencia un
defecto del modelo.

### Cambio B — `ret_recovery_period_mode = "elapsed"`

El default embarcado (`disruption`) no es la fórmula de la tesis. El Algoritmo 2
(p.69) define `RPj = OATj − primer R⁰`, que es `elapsed`. Ya establecido en
`RPJ_MODE_FINDING_2026-07-30.md`.

## 2. Efecto declarado por adelantado

Medido antes de firmar, sobre raíces 2.200.001–3, referencia `fidelity_reference_v3`:

| familia | momento | actual | con cambio A | referencia |
|---|---|---:|---:|---:|
| R1r | `scored_orders_per_year` | 274,7 (**19,9 SD**) | **213,0 (0,7 SD)** | 215,1 |
| R2r | `scored_orders_per_year` | 278,7 (**22,5 SD**) | **204,7 (4,6 SD)** | 217,3 |

El cambio A **no toca `rpj_mean`**: las no servidas llevan `RPj = 0` y ya estaban
fuera de la media sobre positivos. El cambio B lo mueve ~3,5 SD.

**Residual declarado y no resuelto:** `rpj_mean` en R1r queda en **19,2 SD**. Es la
cola de `CTj`, nuestras servidas tardan ~45% más que las suyas, y ni el barrido de
emparejamiento ni la hipótesis de volumen la movieron. R2r ya está en 4,4 SD.

## 3. Prohibiciones

Queda prohibido elegir cualquiera de los dos cambios por el `H_PI` que produzca,
por el signo de un contraste MPC-contra-estático, por que una familia cruce un
umbral de servicio, o por que el resultado sea publicable.

## 4. Qué se re-corre y qué no

**No se reetiqueta nada.** Program Q, la confirmación H2/H3, el buffer gate, la
reproducción de 90 configuraciones y la frontera conjunta de 648 conservan sus
cifras bajo la definición con la que se calcularon. Si esto se adopta, **abre un
cuerpo de resultados nuevo, no reescribe el viejo**, y ambos se reportan con su
definición declarada y su sello de calibración.

**Falsador del instrumento:** si tras el cambio A alguna orden puntuada carece de
`OATj`, la implementación está mal y se detiene.

## 5. Lo que autoriza y lo que no

**Autoriza:** reportar la fidelidad multi-momento bajo la definición de población de
Garrido, declarando en qué momentos mejora (población) y en cuáles no (la cola de
`CTj`, sin cambio).

**No autoriza:** entrenar nada, reemplazar cifras congeladas, ni afirmar que el
modelo es «más fiel» sin declarar el residual de 19,2 SD que sigue abierto.

## 6. Firma

Requiere aprobación del PI. La decisión que no me corresponde: si el cuerpo de
resultados se migra a la definición nueva, o si la actual se conserva como línea de
reproducción y la nueva se abre en paralelo.
