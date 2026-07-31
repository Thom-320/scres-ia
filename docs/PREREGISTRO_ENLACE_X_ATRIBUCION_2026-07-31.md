# Preregistro — el cruce `op9_linked` × `causal_exposure`

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Se ejecuta con `supply_chain/arm_runner.py`.

Los dos ejes son **opciones ya implementadas**. **Cero parámetros libres, cero física nueva.**

---

## 1. Por qué cruzarlos

Cada eje arregla un momento distinto y **estropea el del otro**:

| eje | qué arregla | qué estropea |
|---|---|---|
| `op9_linked` | **`ret_mean` `d_k` 1,67 → 0,23** | `rpj` sube (2.180 en p95) |
| `causal_exposure` | **`rpj_p95` 2.545 → 672** (3,8×) | `ret_mean` 0,23 → 0,36 |

Ninguno se ha medido junto al otro. La regla de dominancia existe exactamente para arbitrar
esto, y `sum_dk` —que el contrato maestro prohíbe— lo habría escondido.

## 2. Brazos

| | `des_events` (statu quo) | `causal_exposure` |
|---|---|---|
| `legacy_theatre_stock` | **A** | **C** |
| `op9_linked` | **L** | **LC** |

**Defecto conocido y ya corregido:** hasta `6c3fe2a`, `op9_linked` no aplicaba `δ` ni el modo
de tránsito porque la ruta física llama directo a `_finalize_pending_backorder`. Está
arreglado; **`δ` queda en `off` en los cuatro brazos** de este contrato, porque ya se midió y
se descartó como palanca (`1833243`).

## 3. La regla nueva, y es la razón de ser de este contrato

Con `causal_exposure`, `rpj_p95` **mejora de nivel 3,8×** y su **`d_k` empeora 9×**, porque la
varianza entre semillas colapsa y el denominador se encoge más rápido que el numerador.

**Declarado por adelantado:**

1. **`d_k` gobierna la adopción.** Es la regla del contrato maestro y no la cambio a mitad de
   sesión porque me convenga un nivel.
2. **Se reporta además un `d_k` de SE apareada**, diagnóstico: el mismo numerador contra el
   **error estándar del brazo `A`**. Aísla si el empeoramiento viene del numerador (la
   estimación se alejó) o del denominador (el instrumento se volvió más preciso).
3. **El diagnóstico NO puede adoptar nada.** Solo etiqueta el residuo.
4. **Si `d_k` empeora mientras el `d_k` de SE apareada mejora**, el veredicto es
   `RESIDUO_MAS_CIERTO_NO_MAS_GRANDE`, y se reporta así: es un hallazgo sobre el instrumento,
   no un deterioro del modelo.

## 4. Predicción, en `d_k`

**Replicaciones** (ya medidas, **no cuentan como confirmación**; con tolerancia derivada del
**SE entre semillas**, no fijada a ojo — el defecto de `5c09437`):

* `L` da `ret_mean` `d_k` ≈ 0,23; `C` da `rpj_p95` nivel ≈ 672.

**Predicciones genuinas:**

1. **`LC` mejora `rpj_p95` de nivel contra `L`** (de ~2.180 hacia ~700). *Puede fallar:* los
   dos ejes podrían interactuar y el enlace re-abrir la ventana de atribución.
2. **`LC` NO alcanza el 456,5 de referencia.** `C` solo ya da 672; predigo `LC` **por encima
   de 550**. *Si bajara de 500, habría que auditarlo.*
3. **`ret_mean` en `LC`: sin dirección declarada.** `L` lo mejora y `C` lo empeora; no sé
   cuál domina.
4. **`autotomy_share` sigue en 11,20** en los cuatro. Es inalcanzable mientras el piso de
   `CTj` supere `LT`, y ningún eje lo toca. *Puede fallar:* si `causal_exposure` cambia la
   rama, se reporta como anomalía.

**Predigo que ningún brazo califica**, por la misma tensión de §1. Lo declaro para que
calificar sea informativo.

## 5. Falsadores

| # | qué | puede fallar porque |
|---|---|---|
| f1 | `A` reproduce la línea base en los cinco momentos puntuados | cualquier perturbación del default |
| f2 | ninguna orden con `CTj < LT = 48` | — |
| f3 | en `C` y `LC`, **todo `RPj` atribuido tiene un bloqueo físico** de la orden | es la condición que `causal_exposure` debe imponer; si atribuye por ventana temporal, falla |
| f4 | `L` y `LC` **difieren** en `rpj_p95` | si salen idénticos, `causal_exposure` no llega a la ruta enlazada — el mismo defecto que este contrato ya sufrió dos veces |
| f5 | `epsilon` barrido; conjunto que se mueve se reporta inestable | — |

**`f4` está aquí porque el mismo defecto apareció dos veces**: `fulfillment_transit_mode` y
después `δ` fueron ignorados en silencio bajo `op9_linked`. No lo doy por corregido para un
tercer eje sin comprobarlo.

## 6. Aceptación

**Conjunto no dominado** sobre los **cinco** momentos puntuados (`scored_orders_per_year`
excluido hasta la referencia v4), `EPSILON = 0,5` **barrido**, ambas familias, `sum_dk` vetado
para rankear.

Un brazo entra si y solo si: `d_k(ret_mean)` no empeora más de `EPSILON` en ninguna familia;
ningún otro momento puntuado empeora más allá de `EPSILON`; los cinco falsadores pasan; y el
conjunto es `epsilon`-estable.

**Si ningún brazo califica pero `LC` domina a `A` en nivel sobre `rpj` y en `d_k` sobre
`ret_mean`**, se reporta como **frontera de compensación medida**, con los dos `d_k` de §3
lado a lado. Ese es un resultado publicable: diría exactamente qué cuesta cada arreglo.

**Prohibido** elegir por `H_PI`, por contrastes MPC-contra-estático, por umbrales de
servicio, o por publicabilidad.

## 7. Declarado por adelantado

| ítem | valor |
|---|---|
| ejes | `order_fulfillment_mode` × `risk_attribution_source` |
| parámetros libres | **ninguno** |
| `δ` | **`off`** en los cuatro (medido y descartado, `1833243`) |
| raíces | **3.200.001–3.200.012**, disjuntas de todo bloque previo |
| momentos puntuados | 5 |
| tolerancia de replicación | derivada del **SE entre semillas**, no fijada a ojo |
| predicción | §4, con §4.3 **sin dirección** y **no-adopción predicha** |

## 8. Alcance

**Nada se reetiqueta.** Fuera de alcance: `δ` (medido, descartado), el predicado de banda, el
clamp, el multiplicador serial, y la referencia v4.

## 9. Firma

Requiere aprobación del PI.

La decisión que no me corresponde: si adoptar un cambio que **mejora el nivel de un momento
3,8× mientras su `d_k` empeora** es aceptable. La regla que propongo —`d_k` gobierna, el
diagnóstico de SE apareada etiqueta— es una decisión de método, y prefiero que la firmes
antes de ver los números que después.
