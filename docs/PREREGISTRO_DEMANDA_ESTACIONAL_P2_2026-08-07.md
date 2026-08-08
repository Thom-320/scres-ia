# Preregistro — demanda estacional pseudo-estocástica (Paper 2, motor)

**Fecha:** 2026-08-07, **escrito antes de implementar**
**Origen:** petición de Garrido, reunión 2026-08-07. Fuente normativa: Garrido, Pongutá &
García-Reyes (2024), *IJPR* 10.1080/00207543.2024.2425771, **§3.2 «Demand variability (input
variable)»**, Ec. (1) y Figura 3.
**Clase:** implementación + caracterización. Sin aprendiz. Sin semillas de custodia.

---

## Por qué, con el argumento corregido

Garrido pide sustituir la demanda por sendas pseudo-estocásticas estacionales porque «la actual es
uniforme discreta, variación mínima, y por eso el modelo aprende fácil».

**Su premisa cuantitativa es falsa** y está medida hoy (`results/demand_process/result.json`,
sello `9711cb366b74f4d8…`): el CV semanal realizado es **7,1 %**, no mínimo, y **24,8 % de las
semanas ya exceden la capacidad de un turno**.

**Su inferencia tampoco se sostiene tal cual**: el empate entre arquitecturas tiene causa medida
—curvatura 0,076 contra ruido 0,317, con el MLP peor que una recta— así que subir varianza sin
subir curvatura **refuerza** el empate.

**Pero la prescripción es correcta, por una razón distinta a la suya.** La memoria que la demanda
tiene hoy es `acf1 = −0,228`: **negativa, débil y sin dirección**, compatible con anti-agrupamiento
de surges. Una estructura estacional da memoria **positiva, persistente y con fase** — el único tipo
de estado que una política condicionada puede explotar y una constante no. Es el candidato más
fuerte a producir `H_regime > 0` que queda tras cerrar Program L.

---

## La especificación de Garrido, literal

**Ec. (1), §3.2:**

```
GR_{t+v} = α·D_t + (1−α)(F_t + δ_t) + δ_{t+1} + γ(F_{t+1} − F_t) + (1−γ)·δ_t
```

- `α` = parámetro de nivel, `γ` = parámetro de tendencia.
- **`α, γ ~ U[0,1]`**, sorteados por Monte Carlo por corrida («assume random values in the range
  [0,1]»). Ésta es la fuente de la familia de sendas.
- Semilla del generador: **«a seasonal time series of 36 demand values»** (Makridakis, Wheelwright
  & Hyndman 1998), §3.3.1.
- Horizonte: **36 periodos semanales**.
- Medida de variabilidad declarada: `cv = σ_GR / mean(GR)`.

**Momentos objetivo, Figura 3** (360.000 puntos = 10.000 corridas × 36 periodos):

| momento | valor |
|---|---:|
| media | 819,13 |
| sd | 174,51 |
| **CV implicado** | **21,3 %** |
| mínimo | 0 |
| máximo | 1.335 |

La figura muestra **periodo ≈ 12** con caídas profundas a ~0 en t ≈ 8, 20 y 32.

### Una inconsistencia en la fuente, declarada y no corregida en silencio

El paper reporta **«kurtosis and skewness indices of −1,88 and 5,19»**. Esa pareja es
**matemáticamente imposible**: para cualquier distribución, curtosis ≥ asimetría² + 1, y
5,19² + 1 = 27,9 ≫ −1,88. La lectura compatible con la Figura 3 —cola inferior larga hacia 0— es
que los rótulos están **intercambiados**: asimetría −1,88, curtosis 5,19.

**No calibramos contra esos dos momentos.** Calibramos contra media, sd, mín. y máx., que son
internamente consistentes, y se le pregunta a Garrido cuál es cuál. Se registra como
`blocked_domain_fact` menor, no bloqueante.

---

## Los tres hechos que decidimos nosotros, y su precio

Bajo la constante permanente del PI, lo que la fuente no fija lo decidimos, lo declaramos y lo
pagamos.

### D1 · La serie semilla de 36 valores

No tenemos el libro de Makridakis y su serie no se transcribe en el paper. **Decidimos:**
reconstruimos un perfil estacional de **periodo 12** con la forma de su Figura 3 (meseta alta con
un valle profundo por ciclo), y lo **normalizamos a nuestra escala**, no a la suya.

**Precio:** nuestras sendas no son sus sendas. Reproducimos su *estructura* (Ec. 1, α/γ ~ U[0,1],
periodo 12, CV objetivo), no sus valores. Se declara en el manuscrito.

### D2 · La escala. **Nunca se copian sus números**

Su media es 819,13 y la nuestra 2.500/día. Copiar 819 rompería la calibración contra la tesis de
2017, que es lo que valida nuestro DES. **Decidimos:** preservar nuestra media y trasplantar su
**CV (21,3 %)** y su forma. Es la misma regla que ya rige el índice Cobb-Douglas.

**Precio:** la comparación con su Figura 3 es de forma y de CV, nunca de nivel.

### D3 · `GR` es un pronóstico, no la demanda realizada

Ec. (1) genera **gross requirements forecast**, no `D_t`. En su modelo la demanda es la única
incertidumbre y `GR` es el input de planificación. **Decidimos** las dos cosas, porque juntas son
más informativas que cualquiera sola:

- **la demanda realizada `D_t`** pasa a seguir el proceso estacional (con ruido), y
- **`GR_{t+v}` de la Ec. (1) se expone como observación** al agente.

**Por qué importa:** eso crea a la vez el *estado* (estacionalidad en `D_t`) y una *señal
observable imperfecta* del mismo (`GR`, cuya calidad depende de α y γ). Es exactamente la
estructura donde una política condicionada puede batir a una constante, y donde un placebo
desinformado **puede** perder — que es la prueba que en op12 no superamos.

**Precio:** dos mecanismos nuevos a la vez. El diseño lo compensa con el arm `forecast_shuffled`
(abajo), que aísla cuál de los dos aporta.

---

## Qué se implementa

Una **opción declarada nueva**, `demand_process='garrido_seasonal_v1'`, junto a la actual
(`'thesis_uniform'`, por defecto). **La ruta nativa de la tesis no se toca.**

Arms para la caracterización:

| arm | qué es |
|---|---|
| `thesis_uniform` | el proceso actual, control |
| `garrido_seasonal_v1` | demanda estacional + `GR` observable |
| `garrido_seasonal_no_forecast` | misma demanda, `GR` **oculto** — aísla el valor del estado |
| `forecast_shuffled` | misma demanda, `GR` **permutado entre periodos** — placebo desinformado que conserva la distribución marginal del pronóstico y destruye su alineación temporal |

---

## Falsadores — cada uno puede fallar **y** puede pasar (R6)

| # | falsador | por qué puede fallar | por qué puede pasar |
|---|---|---|---|
| **g1** | Con `demand_process='thesis_uniform'`, la serie realizada es **byte-idéntica** a la de `result.json` de hoy con las mismas semillas. | Cualquier fuga del código nuevo hacia la ruta nativa la rompe. Es la que protege la validación contra la tesis de 2017. | Si el interruptor está bien aislado, coincide exactamente. |
| **g2** | El CV semanal de `garrido_seasonal_v1` cae en **[0,15, 0,28]**, banda declarada alrededor del 0,213 de la Figura 3. | Una implementación mal escalada aterriza fuera; también fallaría si la agregación semanal amortigua la estacionalidad. | Una implementación correcta cae dentro. |
| **g3** | La autocorrelación a lag = periodo estacional es **positiva y fuera** de la banda iid ±2/√n. | Si la Ec. (1) con α, γ ~ U[0,1] destruye la fase, la ACF estacional no aparece y **el motor no sirve para lo que se construyó**. Es el falsador central. | Una estacionalidad real la produce. |
| **g4** | `α` y `γ` realizados cubren [0,1] sin agruparse: media dentro de [0,40, 0,60] y sd > 0,20. | Un sorteo mal cableado (constante, o sesgado) lo delata. | `U[0,1]` lo pasa. |
| **g5** | El pronóstico `GR` es **informativo pero imperfecto**: correlación con la demanda futura realizada en **(0, 1)**, estrictamente. | Correlación ≈ 0 → el pronóstico no informa y `forecast_shuffled` no puede perder. Correlación ≈ 1 → es clarividente y el resultado sería un oráculo disfrazado. **Puede fallar por los dos lados.** | Un pronóstico realista cae en medio. |

`g2`, `g3` y `g5` se juzgan por **magnitud contra banda**, nunca por signo — regla R6, y con la
cláusula añadida hoy: se ha verificado que el estadístico de cada falsador **puede diferir** de
aquel contra el que se compara.

---

## Regla de decisión, congelada antes de implementar

```
si g1 falla                      -> HALT. La ruta nativa está contaminada; nada más se lee.
si g3 falla                      -> SEASONAL_ENGINE_DOES_NOT_PRODUCE_PHASE
                                    el motor no crea el estado que motivó construirlo; se
                                    re-especifica antes de medir headroom
si g1,g2,g3,g4,g5 pasan          -> ENGINE_READY_FOR_HEADROOM_GATE
                                    y SOLO entonces se preregistra el gate de H_regime
en cualquier otro caso           -> ENGINE_PARTIAL: se reporta celda a celda, sin gate
```

Rama `else` explícita — regla **R7**, escrita hoy tras la brecha de ramas de L-0.

---

## Lo que este preregistro NO autoriza

- **No mide headroom.** `H_regime` necesita su propio preregistro, **después** de `ENGINE_READY`.
- **No entrena nada.** La escalera constante → umbral → MLP → PPO no se toca hasta que exista
  headroom medido, per la jerarquía de decisión.
- **No abre semillas de custodia.**
- **No toca `ret_excel`** como endpoint: rige la Decisión 1 del PI (resiliencia, `full_ledger` /
  Cobb-Douglas).

## Salida

`results/demand_seasonal_engine/result.json`, sellado con `seal_and_write`.
Instrumento: extensión de `scripts/measure_demand_process_v1.py`, que ya mide ACF contra banda iid.
