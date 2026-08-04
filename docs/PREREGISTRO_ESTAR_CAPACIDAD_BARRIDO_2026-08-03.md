# Preregistro — barrido del presupuesto de almacenamiento (E\*-C, gate G1)

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_estar_capacity_sweep_v1.py`.
**Semillas:** bloque **quemado** `5.200.001–16`, réplica declarada. **Ninguna nueva** — esto es un
**screen de desarrollo**, no una confirmación, y no requiere autorización de raíces frescas.

## 1. La sonda de liveness que hice ANTES, y que cambia el endpoint

Antes de escribir este documento probé si la palanca mueve algo, porque un barrido sobre topes que
nunca atan no mide nada. El resultado obliga a declarar el endpoint aquí y no después:

| presupuesto | bloqueado | entregado | `flow_fill_rate` | `worst_claimant_fill` | `lost_orders` |
|---:|---:|---:|---:|---:|---:|
| sin tope | 0 | 292.308 | 0,6415 | **0,6791** | 0 |
| 200 | **1.306.164** | **292.308** | 0,5228 | **0,6791** | 0 |
| 600 | 310.123 | 292.308 | 0,6258 | **0,6791** | 0 |
| 2.000 | 30.753 | 292.308 | 0,6415 | **0,6791** | 0 |
| 3.000 | 0 | 292.308 | 0,6415 | **0,6791** | 0 |

**Bloquear 1,3 millones de raciones entrega exactamente lo mismo.** La capacidad **retrasa**; no
destruye. Y de ahí lo importante:

> **`worst_claimant_fill` es estructuralmente CIEGO a este mecanismo.** Es un cociente acumulado
> `entregado/demandado`, y como todo acaba entregándose, no puede moverse. No es que la capacidad
> no haga nada — es que este endpoint no lo ve.

## 2. El endpoint, elegido por el mecanismo y no por el resultado

**Primario: `flow_fill_rate`.** La restricción es **temporal**, así que el endpoint debe ser
sensible al tiempo. `flow_fill_rate` se mueve 0,6415 → 0,5228 en la sonda; `worst_claimant_fill`
no se mueve en absoluto.

**Y digo cómo se eligió, porque importa:** *no* miré qué endpoint daba el resultado que quería.
Miré cuál puede **responder al actuador**, que es la condición previa a cualquier medición — la
misma comprobación de liveness que en G3-obs costó una corrida entera al actuador muerto. Elegir
un endpoint por su respuesta al *mecanismo* es legítimo; elegirlo por su respuesta a la *hipótesis*
no lo es, y esa línea no se cruza aquí.

**`worst_claimant_fill` pasa a guardarraíl**, junto con `lost_orders` δ = 0,50 ·
`flow_fill_rate` agregado δ = 0,005 · masa y demanda **δ = 0 exacto**. `ret_excel` es diagnóstico,
nunca guardarraíl: está medido premiando el abandono.

## 3. La hipótesis

> **E\*-C.** Con un presupuesto **total fijo** de almacenamiento repartido entre CSSU A y B, el
> reparto óptimo **se mueve con el régimen de riesgo**, y `LCB95(H_regime) > 0` sobre
> `flow_fill_rate`.

El presupuesto es lo que crea la decisión: topes independientes sólo empeoran la cadena. Un total
fijo convierte el almacenamiento en **recurso escaso no fungible** — el único mecanismo del que
este proyecto ha medido headroom material (Program O, `H_PI` 0,1515; nulo fungible exactamente 0).

## 4. Diseño

* **Palanca**: `share_A ∈ {0,1 … 0,9}`, continua por construcción (Garrido pidió continuas el
  2 de julio); el resto va a B. Total conservado por `budgeted_ledger`, que **se niega a existir**
  si no suma.
* **Presupuestos**: `{600, 1.200}` — la sonda muestra que 600 ata con fuerza y 3.000 no ata nada,
  así que ambos caen en la región viva. Se declaran los dos porque un solo presupuesto no
  distingue «no hay headroom» de «elegí el nivel equivocado».
* **Regímenes**: `R1r+R2r|base` y `R1r+R2r|freq3_imp2`.
* **Semillas**: 16 quemadas, réplica declarada (`f5`).
* **Inferencia**: `H_regime = media_r[max_a] − max_a[media_r]`, bootstrap agrupado por semilla,
  5.000 remuestreos, LCB95.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_capacity_actually_binds` | la sonda dice que 600 ata, pero debe verificarse **en cada celda**: `binding_fraction > 0`. Un tope que no ata hace vacuo todo lo demás |
| `f2_mass_and_demand_are_untouched` | entregado + bloqueado conservado, y **demanda idéntica** al modelo sin tope. Una capacidad que redujera la DEMANDA borraría el problema en vez de restringir la solución |
| `f3_the_endpoint_responds_to_the_lever` | `flow_fill_rate` debe variar entre repartos. Si no varía, el endpoint tampoco ve el mecanismo y el barrido no mide nada |
| `f4_argmax_moves_with_regime` | **la hipótesis**: si el óptimo es el mismo reparto en todos los regímenes, hay restricción sin dependencia del estado |
| `f5_no_fresh_seeds` | réplica declarada del bloque quemado, verificada contra el registro central |
| `f6_no_gain_by_abandonment` | `worst_claimant_fill` y `lost_orders` con **margen firmado**, `UCB95(daño) ≤ δ` |

## 6. Reglas de lectura, fijadas de antemano

* **`LCB95(H_regime) ≥ 0,01` y `argmax` móvil** → `CAPACITY_OPENS_REGIME_DEPENDENT_HEADROOM`.
  **Sería el primer headroom del carril E\***, y abriría la escalera de comparadores. **No
  autoriza entrenar.**
* **`argmax` móvil pero `H_regime < 0,01`** → `ARGMAX_MOVES_WITHOUT_VALUE`, la U de contención
  otra vez. Resultado con contenido.
* **`argmax` fijo** → `CAPACITY_CONSTRAINS_WITHOUT_DECIDING`. Cierra el eje de capacidad como
  fuente de headroom, y con G1/G2 ya cerrados sería el **tercer** eje de E\* cerrado
  constructivamente.

**Alcance:** desarrollo sobre tapes quemados. No autoriza nada, no abre semillas, y `G1` sigue sin
concederse hasta que un resultado con potencia lo respalde.
