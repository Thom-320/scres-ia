# Resultado — mezclar familias **sí** crea headroom medible. Y es de 1,8e-4

**Artefacto:** `results/sensitivity/mixed_risk_downstream_v1/result.json` (sello
`a17ae3a2aa8d1462…`) · **Contrato:** `docs/PREREGISTRO_MEZCLA_RIESGOS_2026-07-31.md` ·
4.375 corridas, 7 regímenes × rejilla 5³ aguas abajo × 5 semillas CRN · **los cinco falsadores
pasan**.

## La respuesta

`H_regime` = el valor de **conocer el régimen**: el mejor ajuste sabiéndolo, menos el mejor
ajuste único que debe servir a todos. Es un **techo** para cualquier política que condicione en
el régimen.

| conjunto de regímenes | `H_regime` | LCB95 | ¿supera su ruido? |
|---|---:|---:|---|
| **3 puros** (el diseño de Garrido) | 0,000121 | 0,000069 | sí |
| **4 mezclas** (nuevos) | **0,000173** | **0,000164** | **sí** |
| **los 7** | **0,000182** | 0,000169 | sí |

**Mezclar sube el headroom un ~50%** (1,21e-4 → 1,82e-4) **y lo hace superar su propio ruido.**
Mi predicción 4 —que seguiría sin ser significativo— **queda refutada**, esta vez en la
dirección buena.

## Pero hay que decir la magnitud, no solo el signo

**1,8e-4 es exactamente el orden de todos los nulos que este proyecto ya cerró:**

| lane | `H_PI` medido | veredicto entonces |
|---|---:|---|
| A1 aprovisionamiento | 9,8e-05 | **CERRADO** |
| puerta de buffers | 1,16e-04 | cerrado (óptimo interior) |
| sensibilidad de riesgo (45 perfiles) | 6,9e-05 | `NO_DOOR`, 144× bajo la barra de 0,01 |
| **mezcla de familias, aguas abajo** | **1,82e-04** | **≈55× bajo esa misma barra** |

Es **el mayor de todos** —y por eso es una señal— pero sigue en el mismo régimen de magnitud. Si
se aplicara la barra de 0,01 que este proyecto usó antes, **no pasa**.

Así que el resultado honesto es de **dirección, no de tamaño**: mezclar familias es la primera
intervención de esta línea que produce valor dependiente del estado **estadísticamente real**, y
sigue siendo dos órdenes de magnitud pequeña para desplegar.

## Un mecanismo que sale de paso, y explica por qué

| régimen | ReT medio |
|---|---:|
| R1r | 0,006729 |
| R2r | 0,324806 |
| R3 | 0,490067 |
| **R1r+R2r** | **0,006253** |
| **R1r+R3** | **0,006712** |
| R2r+R3 | 0,321615 |
| **R1r+R2r+R3** | **0,006236** |

**R1r domina toda mezcla en la que entra.** Añadirle R2r o R3 no sube nada: la mezcla cae al
nivel de R1r. Encaja exactamente con la descomposición de drivers: bajo R1r **el 100% de los
pedidos cae en la rama de recuperación**, así que `ReT ≡ 0,5/RPj` y el resto del régimen no puede
expresarse. Mezclar con R1r **no enriquece el problema: lo absorbe.**

Las mezclas que sí cambian el nivel son las que **no** llevan R1r (`R2r+R3`, 0,3216 contra 0,3248
y 0,4901). Ahí es donde la mezcla tiene margen de maniobra.

## Qué hacer con esto

1. **La mezcla útil es sin R1r.** `R2r+R3` es la única que combina dos regímenes que se expresan
   ambos. Cualquier diseño futuro de mezcla debería empezar ahí, no por «todos los riesgos».
2. **La barra sigue sin superarse.** Antes de añadir nodos, el permiso de Garrido para **editar
   frecuencia e impacto** puede aplicarse **sobre las mezclas** — el mapa dijo que escalar
   riesgos aporta poco por sí solo (`S_T` 0,021 y 0,006), pero nunca se ha probado escalarlos
   **dentro de un régimen mixto**, que es donde el acoplamiento acaba de aparecer.
3. **Y solo entonces, nodos.** El mapa ya dijo dónde: aguas abajo, nunca arriba.

## Límites

* Horizonte 52 semanas; los niveles **no** se comparan con los suyos.
* La rejilla es de **tres palancas aguas abajo**; un headroom mayor podría vivir en una
  combinación que esta rejilla no cubre.
* `H_regime` es un **techo con régimen observado**. Una política real tendría que **inferir** el
  régimen, así que su valor alcanzable es **menor** que estos 1,8e-4, nunca mayor.
