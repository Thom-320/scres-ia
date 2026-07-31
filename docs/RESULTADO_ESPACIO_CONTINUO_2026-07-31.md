# Resultado — liberar el espacio NO creó la no linealidad. Mi predicción principal falló

**Artefacto:** `results/garrido_continuous_space/result.json` (sello `8afd916177379bba…`) ·
**Contrato:** `docs/PREREGISTRO_ESPACIO_CONTINUO_2026-07-31.md` · 384 configuraciones Sobol,
raíces vírgenes, **los cinco falsadores pasan**, una sola corrida.

## La respuesta

| | su rejilla 6×3 | **espacio continuo** |
|---|---:|---:|
| **B1** lineal | 0,9697 | **0,9823** |
| **B1** backprop | 0,9863 | 0,9898 |
| **B1** KAN | 0,9913 | 0,9887 |
| **B2** logística | 0,7111 | **0,6326** |
| **B2** backprop | 0,7178 | **0,5802** |
| **B2** KAN | 0,7711 | **0,5853** |

**Predije que el lineal caería a 0,80–0,95 al desacoplar las variables. Subió a 0,9823.**
Predicción 1 **refutada**; se cumplió la 4, que es la que escribí porque prefería que fuera
falsa: **la adecuación lineal es de su métrica, no de su rejilla.**

Liberar el periodo de reposición y desacoplar las tres cantidades de stock —dentro de sus
propios límites de la Tabla 6.16, sin física inventada— **no volvió el problema no lineal**. Al
contrario: la superficie continua es *más* suave que su rejilla acoplada.

## Sobre «con que le ganemos de forma mínima»

Hay que decirlo con precisión, en las dos direcciones:

* **B1: sí, ganan, y por la regla declarada.** Backprop 0,9898 contra 0,9823 lineal, con SD del
  lineal 0,0035 — supera el listón. **Pero el listón mide un sliver:** la red captura el **42%
  del 1,77% de varianza que le quedaba al lineal**. Es una victoria real y **diminuta**, y es la
  misma que ya teníamos en su rejilla.
* **B2 — su pregunta de activación, la que su Fig. 5 formula: pierden, y pierden más que
  antes.** 0,580 y 0,585 contra 0,633 de la logística. En el espacio rico, las redes están
  **por debajo** de la línea base.

Así que el espacio continuo **no** es la palanca. Reportarlo como avance sería vender el 0,0075
de B1 y callar el −0,05 de B2.

## Por qué esto no cierra la puerta que Garrido pidió

En la reunión del 28 de julio pidió dos cosas distintas, y aquí solo se probó una:

1. **«Preferir variables continuas»** — probado. **No aporta.**
2. **«Añadir nodos y variables de decisión, aguas arriba y aguas abajo»** — **NO probado.**
   Esto no es hacer continuas las que ya hay: es **buffers en puntos que su modelo no
   considera**, lo que cambia la *topología*, no la resolución.

El resultado dice exactamente esto: **más resolución sobre los mismos nodos no genera
headroom; hay que probar nodos nuevos.** Es una dirección más precisa que antes de correr, y
sale de una predicción fallida, no de una corazonada.

## Un límite que hay que declarar antes de que alguien sobrelea esto

Esto mide un **surrogado estático**: `ρ → ReT` con `ρ` fijo por corrida. **No dice nada
directamente sobre headroom de control dinámico**, que es una política dependiente del estado y
otra pregunta. Que la superficie estática sea casi lineal no implica que un controlador reactivo
no tenga margen — ni lo contrario.

Lo que sí acota es la **Fig. 5 de Garrido**, que es exactamente un mapeo estático de
configuración a SCRES. Para esa figura, la respuesta medida en dos espacios distintos es la
misma.

## Estado

`DEVELOPMENT_CONTINUOUS_SPACE`. Nada adoptado. Una corrida, la declarada; sin re-muestreo, sin
ampliar `n`, sin tocar arquitectura ni la barra — las cuatro cosas que el contrato prohibía.
