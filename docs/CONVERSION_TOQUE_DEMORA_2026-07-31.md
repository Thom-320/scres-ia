# La conversión toque→demora: localizada en `R14`, y hay una palanca ya implementada

**Status:** `DEVELOPMENT_DEFECT_LOCALISED`. Nada implementado. 21.667 filas suyas.

## 1. El caso limpio es la prueba

`P(demorada | riesgo presente)`:

| riesgo | Garrido n | `P(k>0)` | nuestro n | `P(k>0)` |
|---|---:|---:|---:|---:|
| `R11` | 14.762 | 90,3% | 625 | **34,7%** |
| `R12` | 2.361 | 87,9% | 31 | 90,3% |
| `R13` | 5.482 | 93,9% | 312 | 59,3% |
| `R14` | 21.253 | 83,7% | 648 | **33,5%** |
| **todas** | 21.667 | **83,5%** | 648 | **33,5%** |

Y aislando `R14` **sin ningún otro riesgo**:

| | n | `P(k>0)` | `k` p50 |
|---|---:|---:|---:|
| **Garrido** | 5.050 | **66,4%** | 1 |
| **nuestro** | 17 | **0,0%** | 0 |

**En su modelo `R14` sola demora dos de cada tres órdenes. En el nuestro, ninguna.**

## 2. Y en sus datos hay dosis-respuesta

| defectuosos en la orden | n | `P(k>0)` |
|---|---:|---:|
| 1–2 | 4.103 | 49,4% |
| 3–5 | 9.013 | 84,7% |
| **6–10** | 2.656 | **100,0%** |
| **11–20** | 5.481 | **100,0%** |

**Seis unidades defectuosas garantizan al menos un día entero de demora.** Y seis unidades a
`λ = 320,5/h` son 0,019 h de reproceso — así que no es el *tiempo* de reproceso: es que
cualquier reproceso pierde la ola diaria.

## 3. Dónde se pierde, en el código

`_risk_R14` crea `RiskEvent("R14", now, now, 0, …)` — **duración cero** — retira los
defectuosos de `_pending_batch` y los manda a `rework_op6`. El mecanismo **existe** y la
frecuencia es correcta (0,85× la Tabla 6.11, ya verificado).

Lo que falta es que **esa merma llegue al `OATj` de la orden**. Nuestras órdenes se sirven
desde el stock de teatro con un delay constante, así que un déficit de producción no
propaga al `CTj` mientras quede stock. Es la **misma brecha arquitectónica** que ya explicó
`δ ≡ 0` y la masa puntual: **la finalización de la orden está desacoplada de la
disponibilidad de material.**

Un solo defecto explica los tres síntomas.

## 4. La palanca, y ya está implementada

| modo de cumplimiento | `P(k>0)` | `P(k>0 | R14)` | `CTj` p50 |
|---|---:|---:|---:|
| Garrido | **83,5%** | **83,7%** | **101,45** |
| `legacy_theatre_stock` (default) | 33,5% | 33,5% | 54,00 |
| **`op9_linked`** | **49,3%** | **49,3%** | 54,00 |

**`order_fulfillment_mode = "op9_linked"` cierra ~un tercio de la brecha de conversión**
(33,5% → 49,3%) sin tocar el piso de `CTj`, que sigue clavado en 54 por la constante.

Es la primera palanca de esta sesión que mueve un momento en la dirección correcta **sin
requerir física nueva**: la opción existe, está probada y hoy no es el default.

## 5. Lo que queda, y su tamaño

* **conversión**: 49,3% con `op9_linked` contra 83,5% — queda la mitad de la brecha;
* **`δ`**: sigue en 0, y el piso constante lo impide estructuralmente;
* **`k` (rejilla)**: ya reproducido exacto.

Las dos primeras son el **mismo** desacoplamiento. Un contrato que enlace el `OATj` a la
disponibilidad real de material las atacaría juntas — y a diferencia de los siete mecanismos
que propuse y se refutaron hoy, este **ya tiene una medición a favor**: los 15,8 puntos que
gana `op9_linked`.

## 6. Estado

Nada implementado. `op9_linked` cambia el `CTj` de toda orden, así que su adopción necesita
preregistro. `ret_mean` bajo los defaults embarcados sigue intacto.
