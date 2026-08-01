# Resultado — `backlog` **no se confirma**, y el placebo explica por qué

**Artefacto:** `results/sensitivity/backlog_confirmation_v1/result.json` (sello
`894bea94e51b8a6e…`, `REFUTED_BACKLOG_SENSOR_WAS_SELECTION_NOISE`) · **cinco falsadores PASA** ·
preregistro `docs/PREREGISTRO_CONFIRMACION_BACKLOG_2026-07-31.md`, commiteado antes de correr
(`4d7a173`).

## El veredicto

Regla congelada, leída del JSON sellado (`low = 21 h`, `high = 30 h`, `threshold = 116 361,6`),
12 semillas vírgenes, diferencia pareada por (régimen, semilla), bootstrap agrupado por semilla:

| brazo − mejor constante | media | IC95 |
|---|---:|---|
| **reactiva sobre `backlog`** | +5,02e-05 | **[−6,99e-05, +1,49e-04]** |
| **placebo** (traza de otro episodio) | +6,91e-05 | **[+1,97e-05, +1,34e-04]** |

`LCB95 ≤ 0` **y** no bate al placebo. Las dos mitades de la regla de lectura fallan. El +43,2%
del barrido **era ruido de selección**: un positivo de seis, elegido después de mirar.

## Pero el placebo no es un cero — y ahí está el hallazgo

**El placebo tiene `LCB95 = +1,97e-05 > 0`.** Está movido por la traza de `backlog` de **otro
episodio**: misma distribución, **cero información sobre este**. Y aun así **bate a la mejor
constante de forma significativa**, y **más que la señal real**.

La lectura es directa y separa dos cosas que suelen ir juntas:

> **El valor está en que el periodo VARÍE, no en QUÉ lo hace variar.** Una programación
> *open-loop* que alterna 21 h y 30 h supera a cualquier constante del grid. Condicionarla al
> estado real del sistema **no añade nada** — de hecho aquí resta.

Eso es exactamente lo contrario de lo que compra un lazo cerrado. Un controlador —MPC o RL—
paga por *información*; aquí la información vale cero y la variación vale 6,9e-05.

Y confirma desde otro ángulo lo que dio la campaña anterior: el mejor observable **no** era el
conteo de eventos de riesgo, y ningún observable lo era. `hours_since_last` ya lo insinuaba en el
barrido con su placebo en **+300%**.

## Lo que esto cierra

La puerta declarada decía: *si la regla mínima no captura nada, no se entrena nada más caro.* Su
premisa —que el umbral no ve el acoplamiento **sobre ese observable**— dejaba **el sensor sin
probar**. Ya está probado, con la clase de política congelada y siete sensores:

| | estado |
|---|---|
| clase de política | **cerrado por segunda vía**: la puerta ya no se puede atribuir a falta de expresividad… |
| observable único | **cerrado**: 6 sensores utilizables + 1 inutilizable; ninguno sobrevive a semillas vírgenes |

…**con una salvedad honesta**: lo que queda refutado es que *más sensor* abra la puerta. Una
clase de política más rica sobre el mismo sensor sigue sin probarse, y ahora tiene **menos**
motivo que antes: si la información no paga en la clase mínima, el argumento para pagarla en una
mayor se debilita, no se refuerza.

## Y la escala, que no se puede perder de vista

La brecha completa del oráculo es **3,5e-05**. La ganancia *no informacional* del placebo es
**6,9e-05** — mayor que la brecha, porque conmutar dentro del episodio es otra clase de política
que el oráculo de constantes no acota (esa corrección está en `4d7a173`). Aun así, ambas cifras
están **~145× bajo la barra de 0,01**.

**Nada de esto es headroom material.** Es la medición de que en el punto más favorable que el
propio sistema ofrece —`op12_rop` × `impact_R1r`, `S_ij = 0,219`— el lazo cerrado no paga, y el
poco valor que hay es *open-loop*.
