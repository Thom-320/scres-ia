# Enmienda — el endpoint pasa a cobrar por lo que los mecanismos gastan

**Fecha:** 2026-08-08. Enmienda a `docs/PREREGISTRO_PRESUPUESTO_COMPARTIDO_Y_CADUCIDAD_2026-08-08.md`.
El artefacto anterior (`results/budget_expiry_boundary/result.json`,
`NO_SEQUENTIAL_HEADROOM_UNDER_BUDGET_AND_EXPIRY`) **se conserva** y no se reescribe.

## 1. El defecto de diseño, que es mío

Congelé `L*` como endpoint. `L*` mide **retraso** y no ve coste. Bajo un endpoint ciego al coste,
más buffer nunca perjudica, el óptimo es «lo máximo asequible» **por construcción**, y no puede
existir decisión secuencial que medir. El resultado lo confirmó de la forma más limpia posible: las
cuatro celdas devolvieron la **misma** postura `[0, 0, 0.5]` y el **mismo** `L_test = 0,3051`,
gastando 313.002 unidades en unas y 667.386 en otras.

Es decir: el contrato anterior no podía responder del todo a su propia pregunta. El negativo que
produjo es real pero **acotado** — dice que no hay headroom *cuando el endpoint no cobra*, que es
casi una tautología.

## 2. Qué cambia

**Sólo el endpoint**, a la misma forma con precio que ya usó el gate conservativo:

```
J(lambda) = L* + lambda * (unidades repuestas / máximo de la celda)
```

**El coste son las unidades repuestas, y NO se suma lo caducado.** Caducar no es un gasto adicional:
es gasto que no compró nada, y su efecto ya entra por donde debe — obliga a **reponer otra vez**, así
que sube `repuestas`. Sumar ambas cosas contaría dos veces la misma unidad y fabricaría un
trade-off, que es exactamente el error que retracté esta mañana.

**Todo lo demás queda igual:** las cuatro celdas, la vida útil inerte de 156 semanas como control
fiel, las 27 posturas, la barra de 0,01, el control fijo elegido **sólo en train**, las semillas
quemadas, y la exigencia de que `f6` y `f7` pasen **juntos**.

**Y el veredicto se lee sobre el frente completo de λ, no sobre un λ elegido.** Ningún resultado
puede depender de escoger el precio que más favorece.

## 3. Lo que espero, dicho antes de correr

**Espero que siga sin haber headroom secuencial.** El gate conservativo ya midió esta familia con
precio y dio **+0,000403** contra una barra de 0,01. Lo escribo aquí para que no pueda contarse
después como si lo hubiera predicho al revés.

Lo que esta corrida añade es que el precio actúe **sobre un entorno donde la caducidad obliga a
recomprar y el presupuesto ata**, que es el único sitio donde quedaba una razón física para que el
óptimo se moviera con la tape.

## 4. Regla de cierre, fijada aquí

Si ninguna celda pasa `f6` **y** `f7`, **la familia del buffer estratégico queda cerrada**: con
física conservativa, coste atribuible, contención por presupuesto, caducidad y endpoint con precio,
no hay valor secuencial que un aprendiz pueda capturar. No habrá una cuarta variante sobre este
actuador; un sucesor exigiría un mecanismo distinto y su propia autorización del PI.
