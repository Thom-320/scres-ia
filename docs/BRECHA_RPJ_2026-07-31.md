# La brecha de `RPj`: es la ventana de atribución, y hay una opción que la mueve 3,8×

**Status:** `DEVELOPMENT_MECHANISM_IDENTIFIED`. Nada implementado.

## 1. Su `RPj` es función del CONTEO DE RIESGOS, no del tiempo de ciclo

| correlación con `RPj` | |
|---|---:|
| **suma de indicadores de riesgo** | **0,8804** |
| `R14` | 0,7872 |
| `R11_2` / `R11_1` | 0,688 / 0,681 |
| `R12` / `R13` | 0,385 / 0,343 |
| **`CTj`** | **0,3670** |

El conteo de riesgos lo explica **más del doble** que el tiempo de ciclo. **Por eso satura:**
el conteo de riesgos que toca una orden está acotado, mientras `CTj` puede crecer sin límite
por espera en cola.

## 2. Y la transición está en ~500 h

| tramo de `CTj` | n | `RPj` p50 | `CTj − RPj` p50 |
|---|---:|---:|---:|
| [48, 100) | 10.192 | 71,7 | **2,4** |
| [100, 200) | 4.361 | 120,9 | **2,1** |
| [200, 500) | 1.443 | 313,8 | **4,6** |
| **[500, 1.000)** | 1.972 | 413,3 | **253,0** |
| **[1.000, 5.000)** | 3.289 | **404,3** | **1.224,8** |
| **[5.000, ∞)** | 304 | **400,2** | **9.482,7** |

`RPj` **sigue** a `CTj` hasta ~500 h con una diferencia de 2–5 h, y a partir de ahí se
**despega y se congela** cerca de 400.

## 3. La lectura, y encaja con el Algoritmo 2

Una orden que espera en cola **no está en proceso**. Los riesgos que ocurren durante esa
espera no la tocan físicamente, así que no se le atribuyen — y el Algoritmo 2 mide `RPj`
desde el primer `R⁰` **atribuido**, no desde el primer riesgo del mundo.

Nosotros atribuimos por **solapamiento temporal sobre toda la vida `[OPTj, OATj]`**
(`raw_start = max(event.start, OPTj)`, `raw_end = min(event.end, OATj)`), cola incluida. Por
eso nuestro `RPj ≈ CTj` sin cota.

## 4. Y la opción ya existe: `risk_attribution_source = "causal_exposure"`

Atribuye solo cuando un bloqueo físico específico de la orden identifica la operación y el
intervalo. Medido, raíces 3.100.001–3:

| atribución | modo | `rpj_mean` | `d_k` | `rpj_p95` | `d_k` | `ret_mean` `d_k` |
|---|---|---:|---:|---:|---:|---:|
| **Garrido** | | **193,7** | — | **456,5** | — | — |
| `des_events` | A | 401,1 | 3,78 | **2.545,2** | 4,31 | 1,67 |
| `des_events` | L | 361,8 | 3,30 | **2.180,4** | 3,57 | **0,23** |
| **`causal_exposure`** | A | **166,5** | 3,33 | **672,0** | 38,09 | 1,83 |
| **`causal_exposure`** | L | **163,1** | 3,43 | **678,0** | 39,15 | 0,36 |

**En el momento crudo la mejora es enorme**: `rpj_p95` pasa de 2.545 a **672** contra su
456,5 — una reducción de **3,8×**. Y `rpj_mean` pasa de 401 a **166** contra su 193,7,
cruzando al otro lado.

**Pero el `d_k` de `rpj_p95` explota de 4,31 a 38,09.** No es contradicción: `causal_exposure`
también colapsa la varianza entre semillas, así que el denominador se encoge más rápido que
el numerador. **Acerca la estimación y a la vez vuelve el residuo mucho más cierto** — el
mismo patrón exacto que ya vimos con el clamp.

## 5. Qué significa para la adopción

`op9_linked` sigue siendo lo mejor para `ret_mean` (**0,23**), y `causal_exposure` lo degrada
a 0,36 mientras arregla los niveles de `RPj`. **Los dos ejes tiran en direcciones distintas
sobre momentos distintos**, que es exactamente la situación que la regla de dominancia existe
para arbitrar — y que `sum_dk` habría escondido.

Un contrato sucesor tendría que cruzar los dos y, sobre todo, **declarar de antemano cómo se
trata el caso «el nivel mejora 3,8× y el `d_k` empeora 9×»**. Mi lectura: `d_k` es la regla
del contrato maestro y manda, pero un residuo que se vuelve más cierto **porque el
instrumento mejoró** merece reportarse como tal y no como deterioro.

## 6. Estado

Nada implementado. `causal_exposure` es una opción existente, no física nueva, así que su
adopción es preregistrable sin construir nada.
