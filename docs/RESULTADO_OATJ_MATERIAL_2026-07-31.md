# Result — enlace del `OATj`: HALTED, la predicción genuina falla, y hay dos defectos míos

**Status:** `HALTED_FALSIFIER_FAILED`. Artefacto
`results/metric_audit/oatj_material_arms_v1/result.json`, sellado. Raíces 2.900.001–12.
**Ningún momento se reporta.**

| brazo | min | p50 | dist/corrida | demoradas | `δ` p50 | `δ>0` |
|---|---:|---:|---:|---:|---:|---:|
| **Garrido** | 48,01 | **101,45** | — | **83,5%** | **4,02** | **98,5%** |
| A legacy+const | 54,00 | 54,00 | 34 | 39,1% | 6,00 | 60,9% |
| W legacy+olas | 48,00 | 48,00 | 34 | 39,1% | 0,00 | 0,0% |
| L linked+const | 54,00 | **78,00** | 32 | **53,4%** | 6,00 | **99,9%** |
| LW linked+olas | 54,00 | **78,00** | 32 | **53,4%** | 6,00 | **99,9%** |

## 1. La predicción genuina falla

**§3.1 — `δ` no emerge como distribución.** `LW` da **32 valores distintos por corrida**
contra los **500** exigidos. `δ` sí pasa a ser positivo en el 99,9% de las órdenes, pero es
una **masa puntual en 6,00**, no `U(0,8)`.

Enlazar el `OATj` a la disponibilidad de material **no vuelve continuo el `CTj`**. La
hipótesis de que los instantes de disponibilidad continuos generarían `δ` está **refutada**:
la producción se consume en lotes, no en un flujo que devuelva instantes arbitrarios.

## 2. Defecto de código: `op9_linked` ignora el modo de tránsito

**`L` y `LW` salen idénticos en cada cifra.** Bajo `order_fulfillment_mode = "op9_linked"`,
`fulfillment_transit_mode` **no tiene efecto** — el `min` sigue en 54,00 mientras `W` sí baja
a 48,00. La ruta enlazada no pasa por la rama de olas de
`_finalize_order_after_fulfillment_delay`.

Eso es un defecto real: dos opciones que el contrato trataba como ejes independientes **no
lo son**, y el factorial 2×2 es en realidad 3 celdas. El falsador `R2` lo capturó.

## 3. Defecto de contrato: mi tolerancia estaba mal especificada

`R1` falla porque medí la línea base con **3 semillas** (2.900.001–3) y luego declaré una
tolerancia de **±2 puntos** contra una corrida de **12 semillas nuevas**. Medido: `A` da
39,1% contra el 33,5% declarado y `L` da 53,4% contra 49,3%.

**La discrepancia es variación muestral, no fallo del instrumento**, y la tolerancia debió
derivarse del error estándar entre semillas en vez de fijarse a ojo. Es mi error de
especificación, el tercero de este tipo en la sesión.

## 4. Lo que sí se mueve, y es lo primero que se mueve hoy

`op9_linked` mueve el `CTj` **p50 de 54,00 a 78,00** —hacia su 101,45— y la fracción con
`δ > 0` **de 60,9% a 99,9%**, además de subir las demoradas de 39,1% a 53,4%.

Es movimiento real en la dirección correcta sobre tres cantidades a la vez. Pero **no basta
para el criterio**, y el contrato no permite adoptarlo por eso: la aceptación exigía §3.1, y
§3.1 falló.

## 5. El balance de la sesión sobre este bloque

Ocho mecanismos propuestos, ocho detenidos o refutados. Lo que queda **establecido** y no
depende de ninguno de ellos:

* `CTj = 48 + k·24 + δ`, con `48` y la **rejilla** reproducidos exactos;
* `δ` caracterizado: uniforme sobre el turno de 8 h, techo verificado contra `Q/λ`,
  i.i.d. y **no** derivable de ningún atributo del pedido;
* `k` es demora por riesgo, no cola ni stock;
* la frecuencia y exposición de riesgo son correctas; **la conversión toque→demora no**, y
  `op9_linked` cierra parte de ella;
* **enlazar al material no genera `δ`** — medido aquí.

`δ` queda entonces donde `docs/DELTA_INTRA_PEDIDO_2026-07-31.md` §4 lo dejó: un **supuesto
estocástico declarado**, `U(0, HOURS_PER_SHIFT)`, que solo puede juzgarse por lo que arrastre
en los otros momentos. Esta corrida es la evidencia de que no hay un mecanismo endógeno que
lo produzca — que era exactamente la pregunta que faltaba responder para poder declararlo.

## 6. Estado

Nada implementado, ningún default movido. `ret_mean` intacto.
