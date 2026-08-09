# Lectura de la Puerta A — el veredicto es el congelado, y el falsador que lo produjo estaba mal

**Artefacto:** `results/program_n/gate_a_track_b/result.json` ·
**Veredicto:** `NO_VALID_NONNEURAL_COMPARATOR` ·
**Regla que lo decide:** `docs/PREREGISTRO_PUERTA_A_TRACK_B_CUSTODIA_2026-08-09.md` §6.2, escrita
antes de correr: *«Si `f3` falla, la regla no es comparador válido y el veredicto es
`NO_VALID_NONNEURAL_COMPARATOR`, **aunque la red gane**.»*

**La red ganó.** Y el veredicto es el que dice la regla.

## 1. Los números

```
mlp                   98.4641
mlp_frozen_history    98.2060
mlp_shuffled_history  98.0975
threshold_rule        97.9942
constant_best         97.9696
untrained_net         72.0266
random_action         68.4338
```

| contraste | media | IC95 | tapes |
|---|---|---|---|
| **mlp − regla** | **+0,4699** | **[+0,2372, +0,7024]** | 36/48 |
| mlp − constante | +0,4945 | [+0,2646, +0,7273] | 36/48 |
| mlp − placebo congelado | +0,2118 | [+0,0035, +0,4199] | 33/48 |
| mlp − placebo barajado | +0,1033 | [−0,1670, +0,3718] | 26/48 |
| **regla − constante** | **+0,0246** | **[−0,0302, +0,0754]** | 32/48 |

Seis de siete falsadores pasan, incluidos presupuesto (215.889 parámetros, error 7,9 %), bloques
disjuntos y el control obligado a diferir. **Falla `f3`**: la regla de umbral no bate a la mejor
constante — su intervalo cruza cero.

## 2. Por qué falló `f3`, y es lo contrario de lo que `f3` vigilaba

`f3` existía para impedir un **comparador de paja**: *«si la regla ajustada no puede batir a la
mejor constante, es un hombre de paja y batirla no probaría nada»*.

Aquí ocurrió lo opuesto. La constante **no es débil: está saturada**. En el pre-vuelo la constante
ajustaba en **59,4** y la regla en **87,3**; tras el refuerzo del comparador —200 candidatos más 100
pasos de refinamiento local, un arreglo que introduje yo tras ese mismo pre-vuelo— la constante
ajusta en **98,21** y la regla en **98,30**. **La regla ya no tiene nada que mejorar.**

Y contra la corrida sin custodia el cambio es enorme: allí la mejor constante era **96,567** y la
regla **97,142**. La constante de esta corrida, bien optimizada, **bate a la regla de aquélla**.

**`f3` no distingue dos situaciones que no son la misma:** una regla débil frente a una constante
saturada. Escribí un falsador que confunde «el comparador es de paja» con «el comparador es tan
bueno que su versión adaptativa no añade nada». Eso es un defecto de mi diseño, no del entorno.

## 3. Lo que el experimento sí midió, dicho sin reinterpretar el veredicto

Bajo el mejor brazo **no neuronal realmente disponible** —que aquí resulta ser la constante, porque
la regla no la mejora— la red gana **+0,4945 [+0,2646, +0,7273]** en 36 de 48 tapes frescas, con el
presupuesto emparejado y los bloques disjuntos.

**Y la memoria explica parte, no todo.** La red bate a su placebo **congelado** por
+0,2118 [+0,0035], pero contra el **barajado** el intervalo **cruza cero** (+0,1033 [−0,1670]). Es
decir: tener historia ayuda; **el orden** de esa historia no está demostrado que ayude. Eso apunta a
`PREMIUM_IS_CAPACITY_NOT_MEMORY` más que a memoria, y es exactamente la distinción que los dos
placebos existían para hacer.

**El `+1,60` original no se reproduce.** Con un comparador bien ajustado la ventaja es **un tercio
de eso**, y sigue siendo positiva con intervalo. La diferencia no la causó la red: la causó **lo mal
ajustado que estaba el comparador** en la corrida sin custodia.

## 4. Qué NO se hace aquí

**No se reetiqueta el artefacto.** El veredicto congelado es `NO_VALID_NONNEURAL_COMPARATOR` y así
se queda. Reescribir la regla después de ver el resultado —para leer un `+0,47` que me gusta— es
exactamente el mecanismo que este proyecto lleva un día entero desmontando en otros.

**Y este bloque está quemado.** `9200001–9200120` queda `BURNED_CONFIRMATION_COMPLETE`. Un sucesor
necesita bloque nuevo.

## 5. El sucesor, con el falsador arreglado

`f3` se sustituye por dos condiciones que sí distinguen los dos casos:

1. **el mejor no-neuronal es `max(constante, regla, …)`**, no la regla por decreto;
2. la validez del comparador se comprueba contra un **suelo absoluto** —que bata a `random` y al
   no entrenado por margen— y **no** contra otro brazo del mismo lado.

Y el sucesor añade lo que aquí faltó: **una familia estructurada de verdad** —umbral sobre un
estadístico más rico, o control por creencia— porque el hallazgo real de esta puerta es que **en
`track_b_v1` una constante bien buscada ya casi agota la clase no neuronal que sabemos escribir**.
