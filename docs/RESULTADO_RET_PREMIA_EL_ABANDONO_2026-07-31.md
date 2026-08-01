# La métrica canónica premia **abandonar una unidad** — medido, no argumentado

**Artefacto:** `results/sensitivity/contention_headroom_v1_2/result.json` · mismo código y mismas
semillas que `v1` (sello `b54ee322…`, seis falsadores PASA); `v1_2` sólo añade **todas** las
métricas por reparto. 5.184 episodios.

## El cuadro, en una tabla

`FIFO_PARTIAL`, capacidad **no fungible**, régimen `R1r+R2r` base. Nueve repartos de la misma
capacidad de transporte entre las dos CSSU:

| reparto A/B | 0,1 | 0,2 | 0,3 | 0,4 | **0,5** | 0,6 | 0,7 | 0,8 | 0,9 |
|---|---|---|---|---|---|---|---|---|---|
| **`ret_excel`** (resiliencia) | **0,0047** | 0,0037 | 0,0025 | 0,0009 | **0,0004** | 0,0012 | 0,0030 | 0,0041 | **0,0051** |
| **`flow_fill_rate`** (servicio) | **0,507** | 0,637 | 0,743 | 0,793 | **0,795** | 0,788 | 0,735 | 0,620 | **0,497** |

**Están exactamente invertidas.** El reparto que **maximiza la resiliencia** entrega el
**50 % de las raciones**. El reparto que **minimiza la resiliencia** entrega el **80 %**.

Un **12×** de «mejora en resiliencia» comprado con **30 puntos de servicio**.

Y no es una celda: **se repite en las seis** (dos familias × tres escaladas), con la misma forma
y casi los mismos números.

## Por qué pasa — el mecanismo, no la anécdota

`ret_excel` puntúa sobre la población **visible**: los pedidos que nunca se sirven **salen** del
denominador. Al estrangular un destino:

1. sus pedidos dejan de completarse y **abandonan la población puntuada**;
2. los que quedan van todos al destino bien alimentado, que ya no compite por capacidad;
3. esos se recuperan rápido, y la rama `0,5/RPj` les da nota alta;
4. la media de la población superviviente **sube**.

**La censura no es un sesgo pequeño: es una política ganadora.** Y es exactamente la que una
cadena militar no puede ejecutar — dejar una unidad de combate sin raciones.

## Lo que esto obliga a cambiar, hoy

**No se puede entrenar un agente sobre `ret_excel`.** Un RL competente encontraría esta política
antes que cualquier solución real, y la reportaría como un éxito. Esto deja de ser una
preocupación teórica y pasa a ser una demostración:

> El paper no puede afirmar «RL mejora la resiliencia» usando `ret_excel` como objetivo. Puede —
> y ahora **debe**— afirmar que **la operacionalización de SCRES más usada de esta literatura es
> explotable**, y mostrar la política que la explota.

Esa es, además, una **respuesta a Garrido**, no un desvío: su paper de 2024 dice que las métricas
de SCRES basadas en DES son *«inadequate or incomplete»* y pide mejorar su **credibilidad y
validez**. Esto es una demostración cuantitativa de en qué sentido lo son.

## Cómo encaja con lo ya medido

Explica hacia atrás tres cosas que quedaron sin mecanismo:

* la **U profunda** de la Fase 1A (`v1_1`): no era estructura, era censura;
* que `ret_excel` **pierda discriminación bajo riesgo** — ya registrado el 28 de julio, ahora con
  el mecanismo señalado;
* por qué `H_regime ≈ 1e-4` en toda la campaña: la métrica premia una dirección que el sistema no
  puede tomar, así que las palancas legítimas apenas la mueven.

## Lo que NO afirma

* **No** dice que las otras métricas estén bien. Cobb-Douglas es **ciega al servicio** por otra
  vía (no cobra los pedidos perdidos) y `ret_thesis` colapsa de casos. Las tres fallan; **por
  tres mecanismos distintos**.
* **No** propone todavía el endpoint sustituto. Eso es la Fase 2, y ahora tiene un requisito
  duro que antes era una preferencia: **el endpoint del paper debe hacer perder a esta política.**
  El candidato mínimo es `ret_excel_visible_clipped_0_1` con `flow_fill_rate` y
  `worst_product_fill` como guardarraíles simultáneos — que es a donde el propio repo ya había
  convergido, y ahora se sabe **por qué** hacía falta.
* **No** es un defecto nuestro de implementación. La censura está en la fórmula del Excel de la
  tesis; nuestra reproducción está verificada contra sus 47.546 filas sin discrepancia de
  fórmula.
