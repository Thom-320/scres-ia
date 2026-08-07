# Corrección — dos de mis seis falsadores en la confirmación GSA estaban mal especificados

`results/gsa_confirmation/result.json` (sello `f1181706cf591edb…`) devolvió
`GSA_CONFIRMED_ON_VIRGIN_BLOCK` **con `f4` y `f6` FALLANDO**. El artefacto **no se edita**; esta
corrección lo sucede.

## 1. Los dos errores, y son míos

**`f4_the_placebo_is_uninformed`** exigía que **ninguna** cinta recibiera su propia secuencia como
placebo. La política de creencia emite **sólo dos secuencias distintas en 120 cintas** —
`(A,B,A,B)` en 89 y `(A,A,A,A)` en 31—, así que esa propiedad es **insatisfacible por
construcción**. Era **un falsador que no podía pasar**: la imagen especular del pecado que llevo
todo el día cazando.

**`f6_the_result_can_be_negative`** exigía que alguna cinta mostrara diferencia no positiva, para
«demostrar que el estimador puede devolver negativos». Eso comprueba **el dato, no el estimador**.
La capacidad de un estimador se demuestra sobre un **control de signo conocido**, no esperando que
los datos reales salgan mezclados.

Ninguno de los dos fallos es sobre el dato. Los dos son especificación mía.

## 2. Lo que el defecto destapó, que importa más que el defecto

Las dos secuencias que la política emite **están las dos en el conjunto de calendarios periódicos
del comparador**, y **el mejor calendario estático es `(A,B,A,B)`** — una de ellas.

> **La «política observable» es una elección de UN BIT por cinta entre dos calendarios fijos.**

Eso reencuadra la lane entera, y a mejor: no es «una política adaptativa bate a la estática», es
**«elegir cuál de dos calendarios fijos correr, cinta a cinta, según el estado observable, captura
la mayor parte del techo de información perfecta»**. Es más nítido, más falsable y más defendible.
Y explica por qué η salía tan alta.

## 3. El placebo corregido

Se permuta **qué cinta recibe qué secuencia**, conservando el marginal 89/31 **exacto**. Con un
tratamiento de dos valores, un nulo por permutación deja ~62 % de unidades sin cambio, y esas
aportan cero: **los empates son esperados y correctos**, no dilución. Es el nulo estándar de
permutación de asignación.

## 4. `f6`, hecho bien

Se ejecuta el mismo estimador sobre un control cuyo signo se conoce **a priori**: la política
observable **no puede** batir al oráculo de información perfecta, así que `obs − oráculo` debe ser
≤ 0. Si el estimador no devuelve negativo ahí, no puede dejar de confirmar y no confirma nada.

## 5. Lo que NO cambia

θ, las semillas, el estimando primario y el bloque. **El bloque sigue quemado** — esto es
re-análisis correctivo del mismo dato, no una segunda apertura, y así se etiqueta
(`run_role: CORRECTIVE_REANALYSIS_OF_A_CONFIRMATION`).

## 6. Lo que hay que retirar de lo ya reportado

La frase *«el margen contra el placebo es cinco veces mayor que el H_obs»*, que reporté sobre el
desarrollo (`results/gsa_resilience_only/`), **descansaba sobre el mismo placebo de desplazamiento
cíclico**. Sigue siendo un nulo por permutación válido, pero **no era el que `f4` decía que era**, y
el artefacto de desarrollo hereda esta corrección. Se cita con el placebo permutado, o no se cita.
