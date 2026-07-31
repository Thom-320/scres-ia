# Preregistro — `δ` como supuesto estocástico declarado

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Se ejecuta con `supply_chain/arm_runner.py`.

**Este contrato es distinto de los ocho anteriores: no propone un mecanismo.** Declara un
**supuesto** y lo somete a la única prueba que un supuesto admite.

---

## 1. Por qué un supuesto y no un mecanismo

`δ` —el offset intradía de `CTj = 48 + k·24 + δ`— **no tiene generador endógeno**. Esta
sesión lo probó por eliminación, cada una con su medición:

| candidato | resultado | evidencia |
|---|---|---|
| cola entre pedidos | **refutado por conteo** | 1 pedido/día; sin competencia posible |
| capacidad de flete diaria | **refutado** | pedido de 2.480 cabe en 2.600 |
| ventana de turno (encolamiento) | **refutado** | el turno nunca se llena, `δ ≡ 0` |
| atributos del pedido | **refutado** | `OP9` 0,024, `∑Bt` 0,016, `Q` 0,077, `R14` 0,116 |
| ensamblaje `Q/λ` | **refutado** | el techo no se mueve con `Q` entre deciles |
| deriva de calendario | **refutado** | `corr(δ, OPTj) = −0,01`, `δ` es i.i.d. |
| disponibilidad de material | **refutado** | 32 valores distintos/corrida, `δ` masa puntual en 6,00 |

Y lo que **sí** está establecido sobre él: `δ ~ U(0, 8)` con p25/p50/p75 = 2,005 / 4,020 /
6,000 contra 2,000 / 4,000 / 6,000 teóricos, 98,5% dentro de `[0,8]`, y **techo fijo en 8 h
verificado independientemente de la forma** — el `p99` se queda plano en ~8,0 mientras `Q/λ`
sube de 7,514 a 7,983 entre deciles de `Q`.

Ocho horas es `HOURS_PER_SHIFT` con `S = 1`.

## 2. El supuesto, declarado

    delta ~ Uniform(0, HOURS_PER_SHIFT)   por orden, i.i.d.

**Sin parámetro libre**: el soporte es la constante de turno de la tesis. Se sortea del flujo
`fulfillment_rng`, aislado de los demás.

## 3. La regla que hace esto honesto

**Está PROHIBIDO puntuar este brazo por `δ`, o por cualquier estadístico derivado de `δ`.**
Reproducirá `U(0,8)` por construcción; eso no es evidencia de nada.

**Se puntúa exclusivamente por los otros cinco momentos**: `autotomy_share`, `ret_mean`,
`ret_above_one_share`, `rpj_mean`, `rpj_p95`. (`scored_orders_per_year` sigue excluido hasta
la referencia v4, per enmienda §2.)

La comprobación de que `δ` sale `U(0,8)` se registra como **`CONSTRUCTION_CHECK`**, no como
falsador. Llamarla falsador sería exactamente la vacuidad que esta sesión pasó un día
eliminando.

## 4. Brazos

| | `δ` apagado | `δ ~ U(0,8)` |
|---|---|---|
| `legacy_theatre_stock` | **A** (statu quo) | **D** |
| `op9_linked` | **L** | **LD** |

`op9_linked` entra porque es la única palanca medida que mueve algo (p50 54→78, `δ>0`
60,9%→99,9%, demoradas 39,1%→53,4%), y porque `δ` solo no puede bastar — ver §5.

**Nota de defecto conocido:** `fulfillment_transit_mode` **no tiene efecto** bajo
`op9_linked` (medido, `5c09437`), así que ese eje queda fuera de este factorial en vez de
fingir que son cuatro celdas independientes.

## 5. Predicción, en `d_k`, y es pesimista a propósito

1. **`δ` NO cierra el `p50` de `CTj`.** Su 101,45 exige `k` p50 = 2; el nuestro es 0–1.
   Añadir un offset acotado por 8 h no puede aportar los ~24 h que faltan. **Predicho:
   `CTj` p50 en `LD` se queda por debajo de 90.** *Si cerrara, habría que auditarlo, no
   celebrarlo.*
2. **`ret_mean`: sin dirección declarada.** `δ` alarga `RPj`, lo que baja `0,5/RPj`; pero
   también cambia qué órdenes cruzan `LT`. No lo sé.
3. **`autotomy_share` sube desde 0.** Con `CTj = 48 + δ` y `δ ~ U(0,8)`, bajo el predicado de
   banda `tol = 0,05` la fracción esperada es `0,05/8 = 0,625%` contra su **0,443%**. Pero el
   predicado embarcado es `le`, así que con `δ > 0` la autotomía **sigue en 0**; se reporta
   como diagnóstico, **no** como criterio.
4. **`rpj_mean` y `rpj_p95` empeoran levemente** en los brazos con `δ`, porque `RPj ≈ CTj`
   crece sin que `k` cambie.

**Predigo que este brazo NO se adopta.** Lo declaro por adelantado para que adoptarlo, si
ocurre, sea informativo.

## 6. Falsadores — y uno es una comprobación de construcción, marcada como tal

| # | qué | puede fallar porque |
|---|---|---|
| **C1** | `δ` sale `U(0,8)` — **CONSTRUCTION_CHECK, no falsador** | es cierto por construcción; se registra, no puntúa |
| f2 | `A` reproduce la línea base en los seis momentos | cualquier perturbación del default lo rompe |
| f3 | ninguna orden con `CTj < LT = 48` | el sorteo podría restar en vez de sumar |
| f4 | `A ≡ L` en `δ` bit a bit cuando `δ` está apagado | el flujo `fulfillment_rng` podría filtrarse a otros |
| f5 | los flujos RNG no-`fulfillment` bit-idénticos entre `A` y `D` | el sorteo debe estar aislado |
| f6 | `epsilon` barrido; conjunto inestable se reporta inestable | — |

## 7. Aceptación

**Conjunto no dominado** sobre los **cinco** momentos puntuados, `EPSILON = 0,5` barrido,
ambas familias, `sum_dk` vetado para rankear.

Un brazo entra si y solo si: `d_k(ret_mean)` no empeora más de `EPSILON` en ninguna familia;
ningún otro momento puntuado empeora más allá de `EPSILON`; los falsadores f2–f6 pasan; y el
conjunto es `epsilon`-estable.

**Si ningún brazo califica**, el resultado es que **reproducir la mecánica temporal de
Garrido no mejora su `ReT`** — y eso es un hallazgo publicable, no un fracaso. Sería la
forma cuantitativa de decir que su métrica y su cronología están menos acopladas de lo que
el modelo asume.

**Prohibido** elegir por `H_PI`, por el signo de un contraste MPC-contra-estático, por
umbrales de servicio, o por publicabilidad.

## 8. Declarado por adelantado

| ítem | valor |
|---|---|
| supuesto | `δ ~ U(0, HOURS_PER_SHIFT)`, i.i.d. por orden |
| parámetros libres | **ninguno** |
| brazos | A, D, L, LD |
| raíces | **3.000.001–3.000.012**, disjuntas de todo bloque previo |
| momentos puntuados | 5 (sin `scored_orders_per_year`) |
| momentos prohibidos como criterio | `δ` y todo derivado |
| predicción | §5, y **predigo no-adopción** |

## 9. Firma

Requiere aprobación del PI.

La decisión que no me corresponde: si un supuesto estocástico sin mecanismo es aceptable en
el modelo del paper. Mi lectura: la alternativa es dejar `δ ≡ 0`, que es también un supuesto
—y uno que los datos refutan— así que la elección no es entre supuesto y física, sino entre
un supuesto medido y otro que sabemos falso. Pero eso es criterio editorial.
