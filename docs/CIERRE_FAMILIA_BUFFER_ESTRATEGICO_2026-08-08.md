# Cierre de la familia del buffer estratégico

**Fecha:** 2026-08-08. **Artefacto que cierra:** `results/budget_expiry_priced/result.json`,
`STRATEGIC_BUFFER_FAMILY_CLOSED__NO_PRICED_SEQUENTIAL_HEADROOM`. La regla de cierre estaba fijada
en `docs/ENMIENDA_ENDPOINT_CON_PRECIO_2026-08-08.md` §4 **antes** de correr.

## 1. Lo que se cierra, y en qué condiciones

La familia entera del actuador de buffer estratégico, bajo **todas** estas condiciones a la vez:

* **física conservativa** — bajar el objetivo detiene la reposición y no evapora stock; cero
  unidades destruidas en cada corrida;
* **coste exacto y atribuible** — unidades kit-equivalentes realmente repuestas;
* **contención por presupuesto compartido** — bolsa por periodo entre op3/op5/op9, sin arrastre, y
  se activa de verdad;
* **caducidad por lotes** — 8 semanas retira 352.352 unidades, con 156 semanas (fiel a la tesis)
  como control inerte que no retira nada;
* **endpoint con precio** — `J(λ) = L* + λ·(repuestas/máximo)`, leído sobre **todo** el frente de λ.

## 2. Y el resultado es el mismo en las 24 combinaciones

**Cuatro celdas × seis λ: hueco clarividente exactamente +0,000000, `p` del nulo 1,0000, y UN solo
óptimo distinto sobre las doce tapes de test.** No es que el hueco sea pequeño: es que **ninguna tape
quiere una postura distinta de la que quiere cualquier otra**.

**El instrumento está vivo, y eso es lo que hace fuerte al negativo.** Los falsadores que exigían
que las cosas se movieran **pasan**:

* `f1` el coste separa la clase;
* `f2` **el precio mueve la decisión** — la postura elegida pasa de `[0, 0, 0.5]` a `[0, 0, 0]` en
  λ = 0,5, en las cuatro celdas;
* `f3` el control fiel se mantiene plano.

Sólo fallan `f4` y `f5`, que son los que piden headroom. Un entorno donde el precio mueve el óptimo
**global** pero ninguna tape pide algo distinto de las demás es la definición de una decisión de
**diseño**, no de **operación**.

## 3. Qué queda dicho sobre el mecanismo

Con lead time de 336 h, la razón por la que no hay decisión secuencial no era la ausencia de coste
ni la ausencia de escasez. Añadimos ambas y no apareció. Lo que hay es que **op3 y op5 son
irrelevantes para el retraso** —óptimo cero en ambos en toda la rejilla, coherente con que la
materia prima mueva 4,56M unidades por exactamente cero ReT— y que **op9 se decide una vez**: medio
si el inventario es gratis, nada si se cobra.

## 4. Lo que este cierre NO dice

No dice que la contención no genere headroom en general: Program O lo midió en `0,1515` con el nulo
fungible en exactamente 0, y la validación positiva del audit lo reprodujo sobre verdad conocida en
`contention_v1`. Dice que **este actuador** no lo tiene, con estas cinco condiciones satisfechas a
la vez.

Tampoco es un veredicto sobre `L*` como constructo: es un veredicto sobre la familia de decisión.

## 5. Regla de sucesión

**No hay cuarta variante sobre este actuador.** Un sucesor exige un **mecanismo distinto** —no otro
precio, otro λ ni otra rejilla— con su propia preregistración y su propia autorización del PI. Las
tres variantes gastadas quedan registradas:

| variante | artefacto | veredicto |
|---|---|---|
| liberación + `inventory_hours` | `results/priced_buffer_gate/` | **RETRACTADO**: liberaba cero o destruía stock operativo |
| conservativa con precio | `results/conservative_buffer_gate/` | trade-off estático real, hueco +0,000403 |
| presupuesto + caducidad, `L*` | `results/budget_expiry_boundary/` | mecanismos vivos, endpoint ciego al coste |
| presupuesto + caducidad, con precio | `results/budget_expiry_priced/` | **CIERRE** |

Todos se conservan y se etiquetan. Ninguno se borra.
