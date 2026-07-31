# Preregistro — enlazar el `OATj` a la disponibilidad de material

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Se ejecuta con `supply_chain/arm_runner.py`. Enmienda vigente:
`contracts/paper_b_v2_amendment_2026-07-31.json`.

Sucede a `docs/CONVERSION_TOQUE_DEMORA_2026-07-31.md` (`f218cc0`).

---

## 1. El defecto, y es uno solo con tres síntomas

La finalización de la orden está **desacoplada de la disponibilidad de material**: el `OATj`
se estampa con una constante desde `OPTj`, así que un déficit de producción no llega al
`CTj` mientras quede stock de teatro. Eso explica los tres síntomas medidos esta sesión:

| síntoma | medido |
|---|---|
| `CTj` es masa puntual | 60,5% de las órdenes en un solo valor |
| `δ ≡ 0` | contra `U(0,8)` suyo |
| la conversión toque→demora falla | `R14` sola: `P(k>0)` **66,4%** suyo contra **0,0%** nuestro |

Y **la frecuencia de riesgo NO es la causa**: `R14` está a 0,85× de la Tabla 6.11 y la
exposición por orden está en rango en los cuatro riesgos.

## 2. Brazos — factorial 2×2

| | tránsito constante 54 | tránsito por ola (`freight_waves`) |
|---|---|---|
| `legacy_theatre_stock` | **A** (statu quo) | **W** |
| **`op9_linked`** | **L** | **LW** |

Los dos ejes ya están implementados y probados; ninguno introduce parámetro libre.

## 3. Predicciones, y una de ellas NO cuenta como confirmación

**Declarado como REPLICACIÓN, no como predicción** (esto ya lo medí, así que confirmarlo no
es evidencia nueva y no puede usarse como tal):

* `L` sube `P(k>0)` de **33,5% a 49,3%**;
* `W` baja el piso de `CTj` de **54,0 a 48,0**.

Si esas dos no se replican, el instrumento está mal y la corrida se detiene.

**Predicción genuina, y es la que importa:**

1. **`LW` produce `δ > 0`.** Es la única pregunta abierta del contrato. Si el `OATj` se
   condiciona a que el material exista y la producción es continua a `λ = 320,5/h`, los
   instantes de disponibilidad son continuos y `δ` debería **emerger**. Criterio: en `LW`,
   `CTj` con **más de 500 valores distintos por corrida** y `δ` p50 en `[1, 7]`.
   *Puede fallar, y con facilidad:* si el material siempre está disponible al llegar la ola,
   `δ` se queda en 0 exactamente como en los brazos de ayer.
2. **`LW` sube `P(k>0)` por encima de `L`.** *Puede fallar:* los dos efectos podrían no
   componerse.
3. **`ret_mean`: sin dirección declarada.** Alargar `CTj` sube `RPj` y baja `0,5/RPj`, pero
   también cambia la mezcla de ramas. No lo sé. Conserva veto.
4. **Nada sobre `k` como rejilla** — ya está reproducida exacta (`mod 24 = 0` en el 100% de
   las demoradas), así que no es objeto de este contrato.

## 4. Falsadores

1. **Las dos replicaciones de §3** dentro de ±2 puntos y ±0,1 h.
2. **`A` reproduce la línea base bajo `arm_runner.py`** en los seis momentos.
3. **Ninguna orden con `CTj < LT = 48`** en ningún brazo. *Puede fallar:* enlazar al material
   podría permitir entregas antes del lead time si el gating se implementa mal.
4. **En `L` y `LW`, ninguna orden se sirve con material inexistente**: el inventario de
   origen nunca cruza a negativo. *Puede fallar:* es la condición que el enlace debe imponer.
5. **`epsilon` barrido**; conjunto que se mueve con `epsilon` se reporta inestable.

## 5. Criterio de aceptación

**Dominancia sobre los seis momentos**, `EPSILON = 0,5` barrido, ambas familias, referencia
`fidelity_reference_v3`. **La salida es el conjunto no dominado, nunca un ganador**;
`sum_dk` no puede rankear.

Un brazo entra en el conjunto adoptable si y solo si:

* **la predicción 3.1 se cumple** (`δ > 0` con el criterio declarado); **y**
* `d_k(ret_mean)` no empeora más de `EPSILON` en ninguna familia; **y**
* ningún otro momento puntuado empeora más allá de `EPSILON`; **y**
* los cinco falsadores pasan.

**Momento excluido de la puntuación:** `scored_orders_per_year`, hasta la referencia v4.

**Si la conversión mejora pero `δ` sigue en 0**, se reporta así: sería evidencia de que el
enlace arregla *un* síntoma de los tres y que `δ` necesita el sorteo del turno como supuesto
declarado — que es la ruta que `docs/DELTA_INTRA_PEDIDO_2026-07-31.md` §4 dejó abierta.

**Prohibido** elegir brazo por el `H_PI`, por el signo de un contraste MPC-contra-estático,
por que una familia cruce un umbral de servicio, o por que el resultado sea publicable.

## 6. Declarado por adelantado

| ítem | valor |
|---|---|
| constantes **no** tocadas | `LEAD_TIME_PROMISE = 48`, `λ = 320,5`, ROP = 24 |
| parámetros libres | **ninguno** — los dos ejes son opciones ya implementadas |
| brazos | A, W, L, LW |
| raíces | **2.900.001–2.900.012**, disjuntas de todo bloque previo |
| familias | R1r y R2r, ambas objetivo |
| instrumento | `supply_chain/arm_runner.py` (obligatorio) |
| defaults del resto | `elapsed`, `serial`, `clamped`, `le`, `union` |
| predicción | §3, con §3.1 como la única genuina y §3.3 **sin dirección** |

## 7. Alcance

**Nada se reetiqueta.** Adoptar un brazo abre un cuerpo de resultados nuevo.

**Fuera de alcance:** `δ` como sorteo declarado (su propio contrato si 3.1 falla), la
referencia v4, el predicado de banda, el clamp, y el multiplicador serial.

## 8. Nota de honestidad sobre la sesión

Este es el octavo mecanismo que propongo sobre este bloque en un día. Los siete anteriores
—duraciones de R12/R13, multiplicador serial, recurrencia y solapamiento, el clamp, la
cadencia como causa de la dispersión, la cola por capacidad, y la frecuencia de `R14`— se
refutaron o los medí mal. **La diferencia de este es que ya tiene una medición a favor**
antes de correrse: los 15,8 puntos que `op9_linked` gana en conversión. Por eso esa parte va
declarada como replicación y **no puede contar como confirmación**; lo único que este
contrato pone a prueba de verdad es §3.1.

## 9. Firma

Requiere aprobación del PI.

La decisión que no me corresponde: si enlazar el `OATj` a la disponibilidad de material es un
cambio de modelo aceptable para el paper. Mi lectura: el desacoplamiento actual es lo que
hace que el `CTj` sea una constante, y una constante no puede reproducir una distribución —
eso ya está establecido. Pero mueve el `CTj` de toda orden.
