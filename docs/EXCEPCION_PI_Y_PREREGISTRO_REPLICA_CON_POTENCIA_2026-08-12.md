# Excepción PI y preregistro — réplica con potencia de la conversión observable de O

**Fecha:** 2026-08-12 · **Concedida por:** el PI, en sesión («pide la excepción PI y corre la
réplica con potencia… coge un bloque virgen»).
**Bloque virgen solicitado y verificado libre:** `7500001–7500288` (288 semillas).
**Escrito ANTES de abrir una sola semilla.**

---

## 1. Qué se pide, y qué NO se pide

Se pide **una sola cosa**: **más tapas**. Ni un umbral, ni un guardarraíl, ni un comparador, ni una
métrica, ni la política, ni la física cambian.

El contrato correctivo de O lleva `no_second_rescue: true` y prohíbe cambiar tras el fallo
*controlador, hiperparámetros, celdas, comparador, física, métrica, placebos, umbrales y
guardarraíles*. **El tamaño de muestra no está en esa lista**, y ése es el único grado de libertad
que este preregistro usa.

**Esto NO es un rescate de Program O.** O está cerrado (`program_o_closed: true`,
`second_rescue_forbidden: true`) y es inmutable en el registro. **Ningún resultado de aquí puede
promover a O.** Es un **programa nuevo que hereda su física** y responde una pregunta propia.

## 2. Por qué la respuesta es potencia, y está medido

R1 (`results/program_o/r1_tail_in_the_objective_v1/result.json`) refutó la hipótesis de que la
selección ignoraba la cola. Enumerando **exhaustivamente las 16 configuraciones** de la clase
declarada, `S_mean` y `S_cvar` eligen **la misma** en las cuatro celdas: la política óptima en media
**ya era factible en cola**.

Y los puntos estimados **no cambian de signo** entre ajuste y validación:

| celda | ajuste, punto [LCB] | validación, punto [LCB **simultáneo**] |
|---|---|---|
| `rho75_share90` | +0,023428 [+0,012287] | **+0,035017** [−0,008578] |
| `rho90_share75` | +0,024263 [+0,016090] | **+0,019535** [−0,015507] |
| `rho90_share90` | +0,125967 [+0,100733] | **+0,122369** [+0,056883] |

Lo que voltea el veredicto es **la inferencia**, no el efecto: el ajuste usa una t(47) unilateral de
**1,6779**; la validación aplica un crítico **simultáneo de 2,8357** sobre 69 estimandos. La ventaja
sobrevive a su propio intervalo y se ahoga en el de la familia.

**Es un problema de potencia.** Y se resuelve con muestra, no aflojando el listón.

## 3. El dimensionado, con el listón intacto

Con el crítico simultáneo **original de 2,8357**, la celda vinculante es `rho90_share75` y necesita
**n ≥ 154,4**. Se declara **n = 288 por celda** (6 × 48):

```
rho75_share90   n=288   LCB simultaneo proyectado  +0.017219   CRUZA
rho90_share75   n=288   LCB simultaneo proyectado  +0.005230   CRUZA
rho90_share90   n=288   LCB simultaneo proyectado  +0.095635   CRUZA
```

Margen sobre la n mínima: **1,87×**, que tolera que el efecto verdadero sea el **73 %** del
observado. Ése es el colchón contra la maldición del ganador — los puntos vienen de un bloque ya
usado y **pueden ser optimistas**. Si el efecto verdadero es menor del 73 % del observado, esto
falla, y **debe** fallar.

**No se relaja ninguna de las tres laxitudes disponibles.** Ni se reduce la familia de multiplicidad
—que habría bajado la n necesaria de 154 a 93—, ni se introduce un margen de no-inferioridad
tolerante, ni se toca el SESOI. Con 288 tapas el listón **original** se cruza en las tres celdas, y
esa es una posición más fuerte que cualquier relajación.

## 4. Cómo se ejecuta, y por qué así

El runner congelado `scripts/screen_program_o_fixed_clock_hobs_validation.py` **rechaza por diseño**
cualquier bloque que no sea de 48 tapas. Es un guard correcto y **no se toca**.

Por tanto: **se ejecuta el runner congelado, sin modificar una sola línea, seis veces**, sobre seis
sub-bloques vírgenes **disjuntos** de 48 tapas. Sólo se añade código nuevo para **agrupar** los
deltas por tapa ya producidos — inferencia sobre resultados, no ciencia nueva.

Cada uno de los seis contratos se genera desde el correctivo y **se verifica programáticamente que
difiere en exactamente un campo**: `validation_tapes.range`. Esa verificación es un falsador.

## 5. Falsadores

* **w1_los_contratos_difieren_en_un_solo_campo** — cada contrato hijo contra el correctivo, sólo
  `validation_tapes.range` (más las claves de procedencia declaradas). *Puede fallar* si arrastro
  cualquier otro cambio, y entonces esto **sí** sería un segundo rescate.
* **w2_los_sub_bloques_son_disjuntos_y_virgenes** — 288 semillas, sin colisión conocida, sin
  solapamiento entre sub-bloques. *Puede fallar.*
* **w3_el_runner_es_byte_identico** — el `sha256` del runner ejecutado debe coincidir con el del
  árbol antes y después. *Puede fallar* si algo lo toca.
* **w4_el_efecto_replica_en_punto** — el punto estimado agrupado debe conservar el signo en las tres
  celdas. *Puede fallar*, y sería la señal de que el efecto del bloque anterior era ruido.
* **w5_el_LCB_simultaneo_cruza_cero_en_las_tres_celdas** — **el titular**, con el crítico original.
  *Puede fallar*, y falla si el efecto verdadero es menor del 73 % del observado.
* **w6_los_guardarrailes_restantes_siguen_no_inferiores** — `worst_product_fill` y el resto del
  vector. *Puede fallar*: arreglar la cola no autoriza romper la equidad.
* **custody** — bloque virgen, apertura única, registrada antes de correr.

## 6. Reglas de decisión

| resultado | veredicto |
|---|---|
| w1, w2 o w3 fallan | `BLOCKED_INSTRUMENT_OR_CUSTODY` |
| w4 falla | `THE_EFFECT_DID_NOT_REPLICATE_IN_POINT` |
| w5 falla | `POWERED_AND_STILL_NOT_SIMULTANEOUSLY_SIGNIFICANT` |
| w5 pasa, w6 falla | `TAIL_CLEARED_BUT_ANOTHER_GUARDRAIL_BROKE` |
| w5 y w6 pasan | `OBSERVABLE_CONVERSION_SURVIVES_AT_ADEQUATE_POWER` |

**Apertura única.** El bloque se abre una vez. No hay segundo intento sobre él, gane o pierda, y no
se pedirá otro bloque para la misma hipótesis: **si falla, la conversión observable segura queda
cerrada por potencia adecuada**, que es una frase terminal y mucho más fuerte que la actual.

## 7. Lo que un `PASS` significaría, y lo que no

**Significaría:** que la conversión observable de la contención no fungible es *segura* a potencia
adecuada, bajo un contrato nuevo, con la política congelada de O y sus umbrales intactos.

**No significaría:** que Program O se reabre —no puede—, ni que existe una prima **neural**: la
política es un belief-MPC **clásico**. Sería una conversión observable de un controlador
estructurado, no una victoria de una red.
