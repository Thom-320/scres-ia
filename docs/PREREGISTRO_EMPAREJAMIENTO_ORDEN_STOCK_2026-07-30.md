# Preregistro — emparejamiento de órdenes contra stock

**Estado:** `PREREGISTRATION_DRAFT_AWAITING_PI_SIGNATURE`. Nada aplicado. Ninguna cifra
congelada se mueve.

Este es el cambio que `RPJ_RESIDUAL_DIAGNOSIS_2026-07-30.md` identificó como el de mayor
apalancamiento disponible, y **es más grande que cualquier cosa intentada hoy**. Va por
preregistro por eso, no por cautela ritual.

## 1. El mecanismo actual, exacto

`supply_chain.py:4794` decide el destino de cada orden con un branch binario:

```python
if (mode != "op9_linked" and not pending_backorders and available >= order.quantity):
    # ruta instantánea: OATj = OPTj + delay, exacto
else:
    # cola BLOQUEANTE
```

Y la cola está documentada como bloqueante en `_serve_pending_backorders`: *«if the
highest-priority delayed order cannot be fully served from on-hand theatre inventory,
lower-priority orders wait behind it»*.

**No hay estado intermedio.** O la orden se sirve completa al instante, o espera detrás de
una cabeza que puede no poder servirse durante mucho tiempo.

## 2. Lo que eso produce, medido

| | nuestro (R1r) | Garrido (R1r) |
|---|---:|---:|
| masa exactamente en el mínimo | **64,1%** en `CTj = 54,0` | — |
| masa en el medio (54 < `CTj` ≤ 500) | 18,1% | — |
| masa en la cola (> 500) | 17,8% | — |
| `CTj` p50 | **54,0** | **101,4** |
| `CTj` p95 | **3.246** | **2.239** |

Somos **bimodales**: un pico en el mínimo y una cola larga. Su masa está en el medio.
Mediana 0,53× la suya, p95 1,45×.

Y por eso tres ítems de la lista de prioridades son **el mismo defecto**: el residuo de
`RPj` (que hereda su distribución de `CTj` con ratio 1,0000 exacto), la cuota de autotomía,
y `scored_orders_per_year` a 2× (servimos demasiado al instante).

## 3. Los tres ejes, barridos y nunca elegidos

Ninguno es un parámetro continuo que se pueda ajustar hasta acertar. Los tres son
**estructurales y discretos**, lo que reduce el riesgo de fitting que hundió a
`op11_handling_hours`.

| eje | valores | qué mueve |
|---|---|---|
| `partial_fulfilment` | `off` (actual) / `on` | sirve lo disponible y encola el resto → tiempos de ciclo intermedios |
| `queue_blocking` | `blocking` (actual) / `skip_head` | permite servir órdenes que sí caben cuando la cabeza no → acorta la cola |
| `order_fulfillment_mode` | actual / `op9_linked` | `op9_linked` elimina la ruta instantánea por completo |

`op9_linked` ya existe y hace que **toda** orden encole. Es el extremo opuesto al actual y
por eso pertenece al barrido: acota el efecto por los dos lados sin inventar nada.

## 4. Criterio

**ε-dominancia sobre los seis momentos**, con el conjunto corregido de
`fidelity_reference_v2` (`scored_orders_per_year`, no el conteo crudo). La regla de no
selección del contrato v2 se aplica sin cambios: **se reportan todas las celdas y ninguna
se elige por el resultado que produce.**

**Criterio de aceptación declarado por adelantado**, y es de *forma*, no de un solo momento:

- masa en el mínimo por debajo del 30% (hoy 64,1%);
- `CTj` p50 dentro de un factor 1,5 de 101,4;
- `CTj` p95 dentro de un factor 1,5 de 2.239;
- **y `rpj_mean` no peor que hoy** — porque `RPj` hereda de `CTj` y el cambio lo mueve.

**Falsador:** si mover masa al medio empeora la cola en la misma proporción en que mejora
la mediana, el cambio está redistribuyendo y no corrigiendo. La dominancia lo detecta; un
criterio de un solo momento no.

## 5. Prohibiciones

Ninguna celda puede elegirse por `H_PI`, por un contraste MPC-vs-estático, por si una
familia pasa servicio, ni por publicabilidad. Y ningún eje puede convertirse en continuo
para ajustarlo — **`partial_fulfilment` y `queue_blocking` son binarios y así quedan**. Si
alguien propone parametrizar la fracción servida, eso es `op11_handling_hours` otra vez y
está prohibido por este documento.

## 6. Qué se re-corre y qué no

**Raíces nuevas:** 2.200.001–2.200.012 por familia, disjuntas de todo bloque previo.

**No se re-corre ni se reetiqueta nada.** Program Q, la confirmación H2/H3, el buffer gate,
la reproducción de 90 configuraciones, el comparador v2, la confirmación prospectiva de ReT
y la frontera de 648 conservan sus cifras bajo la calibración con la que se calcularon. Si
el barrido cambia el emparejamiento, **abre un cuerpo de resultados nuevo, no reescribe el
viejo** — y el sello de `calibration_provenance` es lo que permite distinguirlos, que es
para lo que se construyó.

**Resultado esperado: no declarado.** A diferencia del preregistro de la cola, aquí el
signo no se conoce.

## 7. Advertencia de proporción

Esto toca el corazón del DES, no una constante. Va a mover **todos** los momentos a la vez,
incluidos los dos que hoy reproducimos razonablemente. La comparación por dominancia existe
para que ese intercambio se vea completo en vez de celebrarse por la mitad que mejora — que
es exactamente el error que produjo `delay = 54`, el modo de `RPj`, la congestión del flete
diario, y casi `op11_handling_hours`. Cuatro veces el mismo patrón en un día.

## 8. Firma

Requiere aprobación del PI. La decisión que no me corresponde: si el proyecto migra al
emparejamiento ganador, o si el actual queda como línea de reproducción y el nuevo se abre
en paralelo. Recuerdo que hoy esa pregunta quedó respondida para el delay —no se migra, se
construye hacia adelante— y que la respuesta debería ser la misma salvo argumento nuevo.
