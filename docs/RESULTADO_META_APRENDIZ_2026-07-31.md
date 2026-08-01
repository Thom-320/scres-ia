# Resultado — el **efecto Alzheimer tiene precio**, y es de 6,3 corridas

**Artefacto:** `results/garrido_meta_learner/result.json` (sello `230a0074a10f12ee…`,
`ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE`) · **los seis falsadores PASAN** · 12 réplicas,
presupuesto 24 corridas por contexto, **20.736 episodios** de superficie · preregistro
`docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md`, commiteado antes de correr.

## 1. El número

Corridas de simulación necesarias para llegar al **1 % del óptimo** de un contexto de riesgo,
sobre 288 configuraciones y seis contextos sucesivos:

| estrategia | corridas | regret final |
|---|---:|---:|
| **neurona con memoria** (`ρ` cruza contextos) | **7,24** | **0,000090** |
| OFAT — **el diseño de su propia tesis** | 12,42 | 0,000297 |
| neurona **reiniciada** en cada contexto | 13,54 | 0,000404 |
| búsqueda aleatoria — el nulo | 19,54 | 0,000821 |

| contraste | corridas ahorradas | IC95 |
|---|---:|---|
| **efecto Alzheimer** (reinicio − memoria) | **+6,31** | **[+5,18, +7,49]** |
| memoria vs **OFAT** (la tesis) | +5,18 | [+3,53, +6,64] |
| memoria vs aleatorio (el nulo) | +12,31 | [+10,56, +14,15] |

Los tres intervalos excluyen el cero con holgura. **Recordar entre corridas casi divide por dos
el coste de encontrar la mejor configuración.**

## 2. La curva de aprendizaje — su `H2`, con una comprobación incorporada

La ventaja **por contexto**, en el orden en que se recorren:

| contexto | 1 `R1r` | 2 `R2r` | 3 `R1r+R2r` | 4 `R1r|esc` | 5 `R2r|esc` | 6 `R1r+R2r|esc` |
|---|---:|---:|---:|---:|---:|---:|
| ventaja de la memoria | **+0,00** | +2,67 | +8,08 | +8,75 | +8,33 | **+10,00** |

**Exactamente 0,00 en el primer contexto** — como tiene que ser: los dos brazos empiezan en
blanco y el reinicio todavía no ha borrado nada. Y crece hasta **+10 corridas** en el sexto.

Ese cero no lo puse yo: **es una comprobación de cordura que el diseño produce solo**, y sale
donde debe. La curva es la forma que su `H2` predice, medida.

## 3. Qué preguntas de Garrido contesta esto, y cuáles no

| pregunta | respuesta |
|---|---|
| **Q2** — *¿cómo se integra un algoritmo de IA en la estructura interna del DES?* | **Entre sus nodos ③ y ⑧, como buscador sobre configuraciones que conserva `ρ`.** El precio de no hacerlo es 6,31 corridas |
| **H2** — curva de aprendizaje entre disrupciones sucesivas | **confirmada**: +0,00 → +10,00 |
| **H4** — `R_t` depende de `L_{t−1}` | **confirmada**: es el estimando principal, y los brazos difieren **sólo** en si `ρ` sobrevive |
| **Q1** — *¿qué familia de algoritmos?* | **parcial**. Que una neurona logística sobre sus cuatro drivers baste dice que el problema **no** necesita capacidad no lineal. Sigue siendo compatible con lo medido el 29 de julio: un lineal explica 0,970–0,982 de su ReT |
| **H1** — tiempos de recuperación | **no medido todavía** |
| **H3** — reducción de varianza | **no medido todavía** |

## 4. Lo que NO afirma

* **Nada sobre control dentro del episodio ni sobre RL.** Esto es aprendizaje **entre corridas**,
  que es lo que su Fig. 2 pide. Las dos cosas son distintas y no las mezclo.
* **Nada sobre que haga falta una red.** El aprendiz es una neurona logística de nueve entradas;
  que gane no demuestra que se necesite profundidad — demuestra que se necesita **memoria**.
* La superficie se recorre por **enumeración medida** (288 configuraciones × 6 contextos × 12
  semillas, todas simuladas), así que la «corrida de simulación» que se cuenta es real.

## 5. Lo que falta, y es barato

`H1` y `H3` son **dos lecturas más de estas mismas corridas**: reevaluar la configuración final
elegida por cada estrategia con el panel temporal encendido (`system_ttr_*`, declarando su
censura por la derecha) y su **varianza** entre intensidades. Con eso, las cuatro hipótesis del
borrador quedan cerradas.
