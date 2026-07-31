# Qué genera `k` — y ya lo reproducimos, lo que invierte el diagnóstico

**Status:** `DEVELOPMENT_MECHANISM_IDENTIFIED_GAP_NARROWED`. 21.667 filas suyas contra 641
nuestras. Nada implementado.

## 1. `k` es demora por riesgo, no cola de capacidad

| predictor | corr con `k` |
|---|---:|
| **`R14`** (productos defectuosos) | **0,4028** |
| riesgos totales | 0,3956 |
| `Q` | 0,2992 |
| `R11_1` / `R11_2` (averías) | 0,246 / 0,250 |
| `R13` (faltantes) | 0,1464 |
| `∑Bt` | 0,0630 |
| `OP9` (stock al pedir) | −0,0775 |

Y condicionalmente:

* **con `R12`/`R13`**: `k` media **39,34**, p50 **14** — **sin** ellos: media 14,18, p50 **2**;
* **solo 10 de 21.667 órdenes no tienen riesgo alguno**, y esas tienen `k` medio 0,70.

El stock (`OP9`) y la cola (`∑Bt`) no lo explican. `k` es el tiempo de recuperación de los
riesgos, redondeado a días. Eso ya refuerza lo que medimos ayer: la cola por capacidad de
flete está refutada por conteo, con un pedido al día.

## 2. Y aquí está la vuelta: la rejilla ya la reproducimos

| | n demoradas | `(CTj−48) mod 24` p50 | en banda `[0,8]` |
|---|---:|---:|---:|
| **Garrido** | 18.098 | **4,017** | 98,4% |
| **nuestro** | 234 | **0,000** | **100%** |

**Nuestras órdenes demoradas caen en `(CTj−48) mod 24 = 0,000` exactamente**, en 72, 96,
120, 144, 168, 192, 216… — **la misma rejilla de 24 h que la suya**. Las suyas están en esa
rejilla **más** `δ`.

Es decir: `CTj_nuestro = 48 + k·24` y `CTj_suyo = 48 + k·24 + δ`.

**El mecanismo de `k` ya está en el modelo.** Mi informe de ayer —«la cadencia de flete no
produce la distribución»— medía lo que no era: el brazo F movía la constante base, mientras
las órdenes **demoradas** llevaban todo el tiempo la cadencia diaria correcta.

## 3. La brecha real, término a término

| término | estado |
|---|---|
| `48` (`LT`) | **reproducido exacto** |
| `k · 24` (la rejilla) | **reproducido exacto** — `mod 24 = 0` en el 100% de las demoradas |
| `k` (la **distribución**) | **no** — `k > 0` en 36,5% contra 83,5% suyo |
| `δ` (offset intradía) | **no** — `δ ≡ 0` contra `U(0,8)` |

| | n | `k>0` | media | p50 | p90 | p99 | máx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Garrido | 21.667 | **83,5%** | 22,83 | 2 | 61 | 307 | 2.170 |
| nuestro | 641 | **36,5%** | 13,89 | 0 | 47 | 210 | 317 |

Las colas son del mismo orden (p90 47 contra 61, p99 210 contra 307). Lo que difiere es la
**base**: él demora el 83,5% de las órdenes y nosotros el 36,5%.

Y eso conecta con la §1: **casi todas sus órdenes están tocadas por riesgo** (21.657 de
21.667). Si en nuestro modelo solo el 36,5% se demora, es que nuestros riesgos tocan menos
órdenes o se recuperan antes — no que nos falte un mecanismo de cola.

## 4. Lo que esto simplifica

La brecha de `CTj` se reduce a **dos cantidades**, no a un mecanismo desconocido:

1. **`δ`** — sorteo uniforme sobre el turno de 8 h, ya caracterizado
   (`docs/DELTA_INTRA_PEDIDO_2026-07-31.md`), con el techo fijo verificado contra `Q/λ`.
2. **la fracción demorada** — 36,5% contra 83,5%, que es una pregunta sobre **exposición al
   riesgo**, no sobre el último tramo.

La segunda es medible con lo que ya tenemos: la Tabla 6.11 da sus frecuencias por año, y
nosotros ya las comparamos una vez (`R11` 1,00×, `R12` 0,43×, `R13` 0,57×, `R14` 0,01×). **`R14`
es el predictor más fuerte de `k` y es donde más lejos estamos** — 258 eventos/año contra
22.153. Ahí hay una hipótesis concreta que no requiere física nueva.

## 5. Correcciones que llevo

* Ayer escribí que «la cadencia de flete no produce la distribución». **Es falso para las
  órdenes demoradas**: la produce exactamente. Solo era cierto para el término constante.
* Y escribí que la dispersión «viene de otro sitio: disponibilidad de stock, lote, o cola».
  **Los tres están medidos y descartados** aquí: `corr(k, OP9) = −0,08`,
  `corr(k, ∑Bt) = 0,06`, y la cola por capacidad ya estaba refutada por conteo.

## 6. Estado

Nada implementado. `ret_mean` bajo los defaults embarcados no lo toca nada de esto.
