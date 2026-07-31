# Where his `CTj` dispersion comes from — resolved, and it decomposes exactly

**Status:** `DEVELOPMENT_MECHANISM_IDENTIFIED_NOTHING_CHANGED`. Measured on 21,667 rows
across his nine R1r sheets.

## The answer

    CTj = 48  +  k · 24  +  δ

| término | qué es | evidencia |
|---|---|---|
| **48** | `LT` = Op10 PT + Op11 PT + Op12 PT = 24 + 0 + 24 | tesis §6.3, confirmado en §6.8.2 |
| **k · 24** | días adicionales esperados, un flete por día | «at a daily freight rate (ROP = 24 hours)», §6.3 |
| **δ** | posición dentro del turno de **8 h** | `HOURS_PER_SHIFT = 8`, `S = 1` |

Reconstruido, da `p25 = 75,00` y `p50 = 101,45` — **exactamente** los observados.

## Cómo se ve en sus datos

**El histograma tiene bandas discretas con huecos vacíos entre ellas**, que es lo que puso
el mecanismo a la vista:

| banda de `CTj` | n | % |
|---|---:|---:|
| [48, 54) | 2.692 | 12,4% |
| [54, 60) | 877 | 4,1% |
| **[60, 72)** | **0** | **0,0%** |
| [72, 84) | 5.023 | 23,2% |
| **[84, 96)** | **0** | **0,0%** |
| [96, 108) | 3.117 | 14,4% |
| **[108, 120)** | **1** | **0,0%** |

La masa vive en `48 + k·24`. Los huecos no son ruido: son la cadencia diaria.

**`k` decae como una cola de espera:** 16,5% / 23,2% / 14,4% / 7,6% / 2,8% para k = 0..4.
Y `corr(q, CTj) = 0,30`, con mediana 77,8 h para pedidos pequeños contra 343,5 h para
grandes — los pedidos grandes esperan más olas, que es lo que hace el lote de 2.400–2.600
raciones.

**`δ` es `U(0, 8)` con precisión notable:**

| cuantil | δ observado | `U(0,8)` | error |
|---|---:|---:|---:|
| p5 | 0,425 | 0,400 | 0,025 |
| p25 | 2,005 | 2,000 | **0,005** |
| p50 | 4,020 | 4,000 | 0,020 |
| p75 | 6,000 | 6,000 | **0,000** |
| p95 | 7,707 | 7,600 | 0,107 |

98,5% de las órdenes caen dentro de `[0, 8]`; de esas, media 3,964 contra 4,000 teórica y
SD 2,290 contra 2,309. **Ocho horas es la jornada de un turno**, y la tesis corre `S = 1`.

## Por qué nuestro modelo da masa puntual

Nos faltan **los dos términos variables**:

* **`k = 0` siempre** — nuestra orden se sirve en la primera ola disponible, porque el stock
  siempre la cubre. No hay competencia por capacidad de flete.
* **`δ = 0` siempre** — tratamos el último tramo como continuo 24/7. `HOURS_PER_SHIFT = 8` y
  `DAYS_PER_WEEK = 6` **ya están en `config.py`**, pero no se aplican a la pierna de
  cumplimiento.

Por eso el brazo de olas del factorial bajó el piso de 54 a 48 —el término constante, que
sí acertó— y no produjo dispersión alguna: sin turno y sin cola, `CTj ≡ 48`.

## Lo que esto corrige de mi diagnóstico anterior

En `docs/RESULTADO_DELAY_FISICO_2026-07-31.md` escribí que la dispersión **«no viene de la
cadencia»**. Es **medio falso**: la cadencia es exactamente el término `k · 24`, y su
estructura de bandas está a la vista. Lo que fallaba era mi implementación, que sincronizaba
la finalización con la ola y por tanto forzaba `k = 0`.

Lo que sí sostengo de aquel documento: la cadencia **sola** no basta. Hacen falta las tres
piezas, y `δ` —el turno de 8 h— no la tenía identificada en absoluto.

## Estatus

**Nada implementado.** Los dos términos faltantes mueven `CTj` de toda orden, y por tanto
`RPj`, `APj` y `ReT`. Necesitan preregistro propio, con la ventaja poco habitual de que
**los tres términos son observables y ninguno tiene parámetro libre**: 48 sale de los PT de
la tesis, 24 de su ROP declarado, y 8 de `HOURS_PER_SHIFT` con `S = 1`.

La predicción falsable, declarable antes de correr: si se añaden el turno y la espera por
capacidad, `δ` debe salir `U(0,8)` y las bandas `48 + k·24` deben aparecer con huecos
vacíos entre ellas. Eso es mucho más fuerte que ajustar cuantiles.
