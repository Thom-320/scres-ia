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

## AUDITORÍA (misma tarde) — el hallazgo se sostiene, **mi explicación no**

El PI pidió auditar el porqué en vez de deducirlo. Se midió, y **corrige la sección siguiente**.
Artefacto: `results/sensitivity/contention_headroom_v1_3/result.json`.

**La censura existe y es grande** — eso se confirma:

| reparto | 0,1 | 0,5 | 0,9 |
|---|---|---|---|
| pedidos **omitidos** de la puntuación | 136 | **58** | 138 |
| pedidos **perdidos** | 76 | **1,6** | 78 |

Los extremos omiten **2,4×** más pedidos y pierden **~50×** más. La forma de la censura es la
misma U que la de la métrica.

**Pero la censura NO basta para explicar la U, y la reparación que propuse no funciona.** Se
midieron las cuatro variantes sobre el mismo barrido:

| variante | ¿qué repara? | ¿desaparece la U? |
|---|---|---|
| `ret_excel_risk_conditional` | — | **no** |
| `ret_excel_full_ledger` | **quita la censura** (los no servidos puntúan 0) | **NO** — 0,0024 / 0,0003 / 0,0025 |
| `ret_excel_visible_clipped_0_1` | **acota** la cola `0,5/RPj` | **NO, y es la peor**: en `R2r` va 0,367 → **0,027** → 0,353, un **14×** |
| `ret_thesis` | acota **y** puntúa todo | **parcial**: plana en `R2r`, con U en `R1r+R2r` |

**Retiro dos afirmaciones que hice antes de medir:**

1. «la censura **es** el mecanismo» → la censura es **un** mecanismo, y **no el suficiente**;
2. «la reparación ya está en el panel» → **falso**. `full_ledger` no la quita y el recorte
   tampoco. **El endpoint que yo mismo propuse para el paper
   (`ret_excel_visible_clipped_0_1`) falla su propia prueba.**

**Y el titular sale reforzado, no debilitado:** que la U sobreviva a quitar la censura **y** a
acotar la cola significa que la preferencia por el abandono **está en el constructo ReT**, no en
un defecto aislado de una de sus variantes. Eso es peor para la métrica, no mejor.

**Lo que queda por medir, y está corriendo:** la **mezcla de ramas**
(`excel_case_pct_*`) por reparto. La hipótesis viva es que el reparto extremo cambia **en qué
rama caen los pedidos** —de la rama de fill rate, que castiga las pérdidas, a la rama de riesgo
`0,5/RPj`, que no las ve— y que ése es el mecanismo dominante. **No lo afirmo hasta verlo.**

## Por qué pasa — el mecanismo, no la anécdota  *(SUPERSEDIDO por la auditoría de arriba)*

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
  Mi candidato anterior —`ret_excel_visible_clipped_0_1` con guardarraíles— **queda descartado
  por la auditoría de arriba**: acotar no quita la U, la **agrava** (14× en `R2r`). Ninguna
  variante de ReT medida hasta ahora pasa la prueba, así que el endpoint tendrá que **cargar el
  servicio dentro**, no al lado.
* **No** es un defecto nuestro de implementación. La censura está en la fórmula del Excel de la
  tesis; nuestra reproducción está verificada contra sus 47.546 filas sin discrepancia de
  fórmula.
