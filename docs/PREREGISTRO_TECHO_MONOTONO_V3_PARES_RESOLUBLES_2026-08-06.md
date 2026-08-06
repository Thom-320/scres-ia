# Preregistro v3 — pares resolubles, rejilla sin borde, y el mismo techo bajo LCB y multiplicidad

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_monotone_transform_family_v3.py`.
Predecesores: `results/monotone_transform_ceiling/result.json` (v1) y
`results/monotone_transform_family_v2/result_after_verdict_fix.json` (v2,
`SURVIVES_LCB_AND_MULTIPLICITY__SIGNAL_CRITERION_VOID`).
Semillas: bloque quemado `garrido_q2_des288`, réplica declarada. **Ninguna nueva.**

## 1. Los dos defectos abiertos que esto cierra

**El criterio de señal sigue sin instrumento.** Van dos proxies fallidos, y por la misma razón de
fondo: *no podían caer*.

* **v1 — orden por pares.** Ninguna función estrictamente creciente desordena un par, salvo por
  empates numéricos. Pasó sin poder fallar.
* **v2 — razón señal/ruido.** `SD_configs / media(SD_semillas)`. Cayó a 0,30× en la rejilla de 288
  pero sólo a 0,65× en la extendida, contra una barra de 0,5×. El motivo es estructural, no de
  calibración: **la saturación mata el denominador a la vez que el numerador**. Con 4.608
  configuraciones, un escalón deja constante a la mayoría, su SD entre semillas se va a cero, y el
  cociente se sostiene justo donde debía colapsar.

**Y la rejilla estaba truncada.** El óptimo de v2 fue `power(γ=10)`, **en el borde** del intervalo
declarado. Un máximo en el borde no es un máximo.

## 2. El proxy nuevo: pares resolubles

```
resolubles(f) = media sobre contextos de la fracción de pares (i,j) con
                | m_i − m_j |  >  sqrt( s_i² + s_j² )
```

donde `m` es la media entre semillas de `f(R)` y `s` su SD entre semillas.

**Por qué éste sí puede caer, por construcción:** bajo un escalón las configuraciones saturadas
tienen `m_i = m_j` **y** `s_i = s_j = 0`, así que la desigualdad es `0 > 0` — falsa. El par deja de
ser resoluble. No hay forma de que la saturación lo salve, que es exactamente lo que rescataba al
proxy de v2.

Es además la cantidad que un aprendiz necesita de verdad: **qué fracción de pares de
configuraciones se pueden distinguir por encima del ruido de replicación**. La identidad **no** da
1 —hay pares genuinamente indistinguibles— y eso es correcto, no un defecto.

**Muestreo declarado:** 200.000 pares por contexto, sorteados con `numpy.random.default_rng(20260806)`
**fijado aquí**. 10,6 millones de pares por contexto × 631 transformaciones es inviable; el error
estándar de una fracción con n = 200.000 es ≤ 0,12 %, dos órdenes por debajo de cualquier margen
que decida algo.

## 3. La familia, ampliada para que el óptimo sea interior

| subfamilia | parámetros | n |
|---|---|---:|
| identidad | — | 1 |
| logística | 25 umbrales × 20 nitideces (`β` 0,05–500) | 500 |
| potencia | `γ` de **0,01 a 100** (antes 0,1–10, y el óptimo salió en el borde) | 31 |
| escalón | 99 umbrales por cuantil | 99 |

**`K = 631`.** Holm sobre las 631.

## 4. Reglas de lectura, fijadas antes de mirar

`GATE = 0,05`. Una transformación **califica** si cumple **las tres**:

```
LCB95 >= 0,05    y    Holm sobre K=631 < 0,05    y    resolubles >= 0,90 x resolubles(identidad)
```

* potencia insuficiente → **`UNDERPOWERED_NO_VERDICT`**
* el proxy no cae en el escalón → **`SIGNAL_CRITERION_VOID`** (por tercera vez; y entonces se
  abandona la vía del proxy y se declara así en el manuscrito)
* alguna califica → **`A_MONOTONE_RESCALING_SURVIVES_LCB_MULTIPLICITY_AND_SIGNAL`**
* ninguna califica pero alguna pasa LCB+Holm → **`SURVIVES_LCB_AND_MULTIPLICITY_BUT_COSTS_SIGNAL`**
* ninguna pasa LCB+Holm → **`NO_MONOTONE_RESCALING_SURVIVES`**

**Lo que no cambia con el resultado:** el orden de configuraciones es idéntico por construcción, así
que cualquier headroom obtenido así es **curvatura de la métrica —una actitud ante el riesgo no
declarada— y no física de la cadena**. Se publica; **no se adopta** sin mecanismo declarado y
confirmación en bloque virgen.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_identity_reproduces_the_sealed_scalar` | `H(identidad)` = `scalar_h_regime` sellado a 1e-9. **Ancla externa** |
| `f2_the_signal_proxy_can_actually_fall` | un escalón debe dar `resolubles < 0,5 ×` la identidad **en las dos rejillas**. **Es el falsador que ya tumbó al proxy de v2**; si vuelve a fallar, la vía del proxy se abandona |
| `f3_the_instrument_has_power` | óptimo plantado a `H = 0,10` debe dar `LCB95 ≥ 0,05`. Si no, no hay veredicto |
| `f4_multiplicity_over_the_declared_family` | `K` debe ser exactamente 631 |
| `f5_the_base_grid_stays_at_zero` | **control negativo**: la rejilla de 288 debe dar 0 en las 631 |
| `f6_the_optimum_is_interior` | la mejor por LCB **no** puede estar en un borde de la rejilla de parámetros. **Falla si la familia sigue truncada**, como en v2 |
| `f7_no_fresh_seeds` | custodia central, réplica declarada |

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica, **no adopta ninguna
transformación** y no cambia la primaria del contrato.
