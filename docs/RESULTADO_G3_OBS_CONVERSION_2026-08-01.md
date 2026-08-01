# Resultado — G3-obs: **SIN POTENCIA** en el primario, y el número que hace falta

**Artefacto:** `results/headroom/g3_obs_conversion/result.json` (sello `1d434485cec99bc7…`) ·
preregistro `docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md` (commiteado antes) ·
bloque **quemado** `5.200.001–16`, **ninguna semilla nueva** · **los ocho falsadores PASAN** ·
**los tres guardarraíles PASAN**.

**Veredicto: `STOP_G3_OBS_UNDERPOWERED`** — un estado terminal **legal**, fijado en el
preregistro precisamente para que un nulo sin potencia no pueda leerse como afirmación.

## 1. Por qué se detiene, con el número encima de la mesa

| celda | MDE (90 %, `n = 8`) | SESOI | `N_test` requerido | total |
|---|---:|---:|---:|---:|
| `R1r+R2r \| base` | **0,0256** | 0,010 | **53** | ~106 |
| `R1r+R2r \| freq3_imp2` | **0,0286** | 0,010 | **66** | ~132 |

Con 8 semillas de test **no se puede** detectar el SESOI de +0,010. Harían falta **~106–132
semillas**, y sólo hay **16 quemadas**. Conseguirlas exige abrir raíces frescas, que
`authority_ladder_v1` prohíbe hasta el recibo de Submission A.

**Lo que no hago:** aflojar el SESOI, tomar prestadas semillas de otro bloque, ni promover el
estimado puntual. El preregistro dice que el MDE se publica **pase o falle**, y éste falla.

## 2. Lo que el primario dice, sin promoverlo

| brazo (vs. mejor constante) | base | freq3_imp2 |
|---|---:|---:|
| `threshold_cumulative` | — | **+0,0169** [LCB95 **+0,0035**] |
| `threshold_windowed` (`H_obs`) | — | +0,0090 [−0,0081] |
| `threshold_delayed` | — | +0,0094 [−0,0018] |
| `threshold_noisy` | — | +0,0074 [−0,0075] |
| `uninformed_placebo` | **−0,0194** | **−0,0226** |
| `wrong_claimant` | −0,1571 | −0,2187 |

**El placebo no informado PIERDE en las dos celdas.** El smoke de 4 semillas lo mostraba
*ganando*, y no lo reporté como hallazgo — con 2 semillas de test era ruido, y lo era.

**La dirección importa**: apuntar al reclamante equivocado cuesta −0,16 a −0,22.

## 3. El coste del realismo, que es el hallazgo más sólido

| degradación vs. acumulado desde el día 1 | base | freq3_imp2 |
|---|---:|---:|
| ventana de 14 días | **+0,0121** [+0,0024] | +0,0079 [+0,0010] |
| retardo de 3 días | +0,0113 [+0,0005] | +0,0075 [−0,0003] |
| ruido `σ = 0,30` | +0,0121 [+0,0025] | +0,0095 [+0,0012] |

**Los tres intervalos excluyen el cero.** Y la ventaja total del acumulado era +0,0169, así que
**cada limitación realista se lleva entre la mitad y dos tercios de ella, por separado**.

> Ésa es la lectura útil: el valor no vive en «conocer el desbalance», vive en **una contabilidad
> perfecta desde el primer día**. Degrádala como la degrada cualquier operación real —ventana
> finita, retardo, ruido— y se queda en nada distinguible de cero.

## 4. El único contraste que no depende de la potencia del mismo modo

`residual_over_simple` = política tabular de 5 bins **menos** umbral simple:

| celda | media | IC95 |
|---|---:|---|
| `base` | **−0,0044** | [−0,0121, +0,0022] |
| `freq3_imp2` | **0,0000** | [0,0000, 0,0000] |

En `freq3_imp2` el residual es **exactamente cero** porque el desarrollo, **pudiendo elegir entre
tres familias de bins, eligió `(0,1 · 0,1 · 0,5 · 0,9 · 0,9)`** — que **es** el umbral de dos
ramas. En `base` eligió los bins graduados y el resultado es **peor**.

**Dada la opción de ser más rica, la familia rica eligió ser el `if`.** Con tres candidatos y
estos intervalos no es un hecho establecido, pero las dos celdas apuntan al mismo sitio, y ese
sitio es `STRUCTURED_CONTROL_SUFFICES`.

## 5. Qué queda, y qué NO se afirma

**No se afirma** `H_obs`, ni conversión observable, ni residual, ni nada sobre prima neural. La
etiqueta canónica es:

> `STOP_G3_OBS_UNDERPOWERED` — instrumento limpio, guardarraíles respetados, **potencia
> insuficiente y cuantificada**.

**Lo que esto le cuesta al programa:** G3-obs era el paso previo a G3c, y **no lo despeja**.
Ninguno de los dos avanza sin semillas frescas, es decir **sin el recibo de Submission A**. La
consecuencia práctica es que el bloqueo del programa **no es científico ni computacional: es
editorial y humano**, exactamente donde el execution board ya lo situaba.

**Lo que sí queda dicho, y es citable en el manuscrito:**

1. bajo observación **realista** —ventana, retardo, ruido— la ventaja adaptativa en este
   actuador **se cae a un margen que 16 semillas no distinguen de cero**, y cada limitación por
   separado se lleva la mitad de ella con IC que excluye el cero;
2. una familia de políticas más rica, **pudiendo separarse del umbral, no lo hace**;
3. y el instrumento aguantó: los ocho falsadores pasan, incluidos los tres reparados hoy
   —márgenes firmados, ajuste sólo en desarrollo, y potencia publicada pase o falle—, que son
   precisamente los que detuvieron o mancharon las dos corridas anteriores.
