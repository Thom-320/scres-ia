# Resultado — **`H1` sostenida en su redacción original**; `H3` no, ni siquiera con brazos distintos

**Artefacto:** `results/manuscript/h1_h3_originales_v3/result.json` (sello `dc46ce6069755a28…`,
`H1_SUPPORTED__H3_NOT_SUPPORTED`) · **los siete falsadores PASAN** · preregistro
`docs/PREREGISTRO_H1_H3_ORIGINALES_V3_2026-08-07.md`, commiteado antes de correr (`189c2c8`) ·
120 réplicas, bloque `6.000.001–120` ya abierto, **cero semillas vírgenes**.

## 1. `H1` — sostenida, y el mecanismo es la absorción

Endpoint: `restricted_ttr = min(tiempo hasta restauración, τ=8 semanas)`, choque aislado `R11…R24`,
placebo pareado por configuración y semilla. **960 celdas por brazo.**

| brazo | TTR restringido (horas) |
|---|---:|
| **híbrido** (neurona con memoria) | **75,7** |
| reinicio | 149,7 |
| **estático** (OFAT, el diseño de la tesis) | **201,7** |

| contraste | ventaja del híbrido | IC95 | Holm |
|---|---:|---|---:|
| **primario — 960 celdas** | **+126,0 h** | **[+98,4, +154,5]** | `p < 0,0001` |
| secundario — 756 celdas con configuraciones distintas | +160,0 h | [+124,0, +195,6] | — |
| híbrido vs reinicio | +74,1 h | [+51,5, +97,5] | `p < 0,0001` |

**El mecanismo no es que se recupere más rápido: es que absorbe más veces.**

| brazo | choques absorbidos | restaurados dentro de τ | censurados en τ |
|---|---:|---:|---:|
| híbrido | **875 / 960 (91,1 %)** | 923 | **37 (3,9 %)** |
| reinicio | 823 (85,7 %) | 885 | 75 (7,8 %) |
| estático | **755 / 960 (78,6 %)** | 865 | **95 (9,9 %)** |

La configuración que el aprendiz despliega **encaja el choque sin degradar el servicio** en nueve
de cada diez casos, contra ocho de cada diez del diseño de la tesis. Eso es lo que produce las
126 horas, y hay que escribirlo así en el manuscrito: **`H1` se sostiene por absorción, no por
velocidad de restauración.**

## 2. `H3` — NO sostenida, y esta vez no es por falta de potencia

Varianza del servicio perdido (`service_loss_auc_ration_hours`) entre los cuatro peldaños de
intensidad `×1…×4`, por celda `(semilla, contexto base)`, 360 celdas por brazo:

| brazo | varianza |
|---|---:|
| reinicio | 1,50 e16 |
| estático | 1,60 e16 |
| híbrido | 1,61 e16 |

| contraste | diferencia | IC95 |
|---|---:|---|
| híbrido vs estático | −1,05 e14 | [−3,30 e15, +2,94 e15] |
| híbrido vs reinicio | −1,11 e15 | [−3,65 e15, +1,29 e15] |

**El signo está del lado contrario y el intervalo cruza el cero.** A `n = 120` con el 78,7 % de
las celdas desplegando configuraciones **distintas**, la explicación de 2026-08-01 —«no hay
estimando»— ya no aplica: **el estimando existe y el efecto no está.** `H3` en su redacción
original queda **no sostenida**, no *no evaluable*.

`ret_excel` apunta igual (híbrido 4,01 e−6 · reinicio 4,17 e−6 · estático 4,35 e−6) y **no decide**:
está medido que premia el abandono.

## 3. `f4` — el hueco A2, medido en vez de temido

`f4` re-evaluó 24 celdas selladas bajo la física instalada hoy y exigió su valor almacenado.

> **24 / 24, con `max_abs_delta = 0,0` exacto.**

El bloque `6.000.001–120` fue sellado el 2026-08-02 y `supply_chain.py` ha recibido commits desde
entonces. **Ninguno mueve esta superficie.** Eso no cierra el hueco A2 —otras familias de
artefactos siguen sin comprobar— pero **acota**: la deriva no alcanza al meta-aprendiz, y el
procedimiento para comprobarlo en las demás ya está escrito.

## 4. Un hecho sobre el régimen de riesgo de Garrido que sale de aquí

Bajo el régimen recurrente `R11–R24` a 52 semanas, los eventos **se agregan en un único clúster
que nunca termina**: la censura de `system_ttr` era 1,000 en los tres brazos porque **no existe un
regreso a la normalidad que cronometrar**. No es un defecto del instrumento.

> **En la cadena de Garrido con sus riesgos recurrentes activos, «tiempo de recuperación» no está
> definido.** Sólo lo está bajo choque aislado, que es como se midió aquí.

Esto es una **limitación del constructo `ReT` heredado**, y es publicable: la mitad de la
literatura de SCRES opera `time-to-recovery` sobre regímenes donde la perturbación nunca cesa.

## 5. Estado de las cuatro hipótesis del borrador

| | redacción original | reformulación declarada |
|---|---|---|
| **H1** tiempos de recuperación | **SOSTENIDA** +126 h `[+98,4, +154,5]` | `H1′` servicio perdido: SOSTENIDA +61,3 M ración-hora |
| **H2** curva de aprendizaje | **medida** — ventaja +0,00 → +10,00 entre seis contextos | — |
| **H3** varianza entre intensidades | **NO SOSTENIDA** — signo contrario, IC cruza cero | `H3′` varianza del coste de búsqueda: SOSTENIDA a `n=120`, +16,22 `[+9,61, +22,74]` |
| **H4** dependencia de `L_{t−1}` | **medida** — memoria vs reinicio, +7,90 corridas `[+6,88, +8,93]` | — |

**Tres de las cuatro se sostienen; `H3` no, en ninguna lectura del desempeño desplegado.** La
única varianza que el aprendizaje reduce es la del **coste de buscar**, no la del **desempeño de
lo desplegado**. El borrador tiene que decir eso y no lo contrario.

## 6. Alcance

Desarrollo. Bloque ya abierto, sin semillas vírgenes, sin adjudicación. `H1′` y `H3′` quedan
intactas. La confirmación de `H1` en bloque virgen **no está hecha** y hace falta antes de
reclamarla en el manuscrito.
