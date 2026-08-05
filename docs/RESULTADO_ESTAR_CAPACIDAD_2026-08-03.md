# Resultado — E\*-C: la capacidad restringe, y casi no decide

**Contrato:** `docs/PREREGISTRO_ESTAR_CAPACIDAD_BARRIDO_2026-08-03.md` (`2719ff7c…`, commiteado
antes) · screen de **desarrollo** sobre el bloque **quemado** `5.200.001–16`, réplica declarada ·
**ninguna semilla nueva** · **seis falsadores en orden** (`f5` NO APLICA por réplica declarada).

**Veredicto: `ARGMAX_MOVES_WITHOUT_VALUE`.**

## 1. Los números, con la corrección de fórmula ya aplicada

| presupuesto | `H_regime` | `argmax` base | `argmax` freq3_imp2 | dispersión | binding |
|---|---:|---:|---:|---:|---:|
| **600** | **+0,00025** | 0,6 | **0,5** | 0,104 | 0,77 |
| **1.200** | **+0,00000** | 0,6 | 0,6 | 0,043 | 0,59 |

**SESOI = 0,010.** A presupuesto 600 el `H_regime` está **40× por debajo**; a 1.200 es
**exactamente cero**.

Y la forma es exactamente la de contención: **la palanca tiene autoridad real** —la dispersión
entre repartos llega a **0,104**, dos órdenes de magnitud por encima del `H_regime`— pero **saber
el régimen no compra casi nada**, porque el reparto 0,6 está cerca del óptimo en ambos.

**El `argmax` sólo se mueve a presupuesto 600.** A 1.200 es el mismo en los dos regímenes. Los dos
presupuestos se declararon precisamente para distinguir «no hay headroom» de «elegí el nivel
equivocado», y lo que muestran es que **el propio movimiento del `argmax` depende del nivel**, lo
cual es una razón más para no leerlo como dependencia del estado.

## 2. Un defecto de mi fórmula, que inflaba 10×

La primera corrida reportó `H_regime` **+0,00253** y **+0,00228**. Estaban **inflados**: mi
`stat()` tomaba el máximo sobre acciones **por régimen Y por semilla**, lo que deja que el óptimo
varíe con la semilla. Eso no es `H_regime` — es clarividencia por semilla, y mide un objeto
distinto y mayor.

Comprobado con las medias almacenadas: el valor correcto es **+0,00025** y **+0,00000**, es decir
**un orden de magnitud menos**.

**El veredicto no cambia** —seguía por debajo del SESOI— pero **las cifras publicadas eran
erróneas**, así que el artefacto se **regeneró** en vez de editarse, y el inflado se conserva como
`results/headroom/estar_capacity_sweep_INFLATED_H_REGIME/` para que la corrección sea auditable.

## 3. La sonda que decidió el endpoint, y que es el hallazgo físico

Antes de escribir el contrato:

> Bloquear **1.306.164 raciones** entrega **exactamente lo mismo** (292.308) y deja
> `worst_claimant_fill` en **0,6791** en todos los presupuestos. `lost_orders = 0` siempre.

**La capacidad retrasa; no destruye.** Un cociente acumulado `entregado/demandado` no puede verlo,
así que `worst_claimant_fill` es **estructuralmente ciego** a este mecanismo y pasó a guardarraíl.
El primario es `flow_fill_rate`, elegido **por responder al mecanismo** —condición previa para
medir— y no por responder a la hipótesis.

## 4. Qué cierra esto

**Tercer eje de E\* cerrado constructivamente**, y cada uno por una razón distinta:

| eje | por qué se cierra |
|---|---|
| enrutado (§6.5.5, rutas) | Program L: `ROUTE_FAMILY_OPEN`, sin residual sobre los comparadores probados |
| **capacidad (§6.5.5, almacenamiento)** | **restringe con autoridad, pero el óptimo casi no depende del régimen** |
| contención CSSU | `H_regime` 0 bajo endpoint sano |

**Lo que NO afirma.** `G1` **sigue sin concederse**: esto es desarrollo sobre tapes quemados, con
16 semillas y sin potencia declarada. Un `H_regime` de +0,00025 con LCB95 > 0 es *positivo y
despreciable*, no *nulo* — y la distinción importa: no cierra la clase, cierra **este contrato**.
Tampoco dice nada sobre buffers **aguas arriba** (WDC, AL), que están en el módulo pero no
cableados: sólo se midió la capacidad de los CSSU.

**Y no autoriza entrenar nada.**
