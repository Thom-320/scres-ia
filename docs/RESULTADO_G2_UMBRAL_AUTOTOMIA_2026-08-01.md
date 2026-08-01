# Resultado — G2: la discontinuidad existe, y **nadie** supera al lineal con interacciones

**Artefacto:** `results/headroom/g2_autotomy_threshold/result.json` (sello `3ed41860e829b83c…`) ·
preregistro `docs/PREREGISTRO_G2_UMBRAL_AUTOTOMIA_2026-08-01.md` (sha
`bd0cbb5b…`, commiteado **antes**) · **siete falsadores PASAN** · 249 s.

## 1. La premisa se cumple: el umbral está encendido

| | |
|---|---:|
| autotomía, brazo embarcado (`shipped_dead_autotomy`) | **0,0000 %** |
| autotomía, brazo `FDB_live_autotomy` | **0,1749 %** |
| referencia Garrido | 0,44 % |
| episodios con **algún** cruce `CTj ≤ 48` | **34,2 %** |

`f1` y `f2` pasan: la rama muerta se enciende, y el cruce no es anecdótico a nivel de episodio.

## 2. Los números — `R²` held-out sobre `ret_excel_risk_conditional`, CV agrupada por semilla

| modelo | `R²` |
|---|---:|
| **lineal + interacciones** *(primario, declarado en el preregistro)* | **0,1553** |
| regla de umbral explícita | 0,1432 |
| lineal aditivo | 0,1393 |
| backprop | 0,1336 |
| KAN | 0,1248 |
| comparador de media de celda (train) | 0,0808 |
| constante | −0,0179 |

| contraste | diferencia | IC95 (t) |
|---|---:|---|
| margen disponible (comparador de celda − primario) | **−0,0745** | [−0,1751, +0,0262] |
| backprop − primario | −0,0217 | [−0,0655, +0,0220] |
| KAN − primario | −0,0305 | [−0,0789, +0,0179] |
| regla de umbral − primario | −0,0121 | [−0,1731, +0,1489] |

## 3. La lectura, y una ambigüedad de mi propio contrato que declaro

La etiqueta del runner es `THRESHOLD_RULE_SUFFICES`. Es **correcta pero incompleta**: en este
resultado se cumplen **a la vez** dos cláusulas del preregistro —«la regla de umbral iguala o
supera a las redes» y «nadie supera al primario»— y el código resuelve el empate a favor de la
primera. **La regla de lectura no era mutuamente excluyente; lo declaro en vez de taparlo.**

El enunciado defendible es el conjunto de ambas:

> **Ningún modelo supera al lineal con interacciones.** Entre los que quedan por debajo, la
> **regla de umbral** queda por encima de las dos redes. **Ninguna diferencia excluye el cero.**

## 4. Y el dato que más dice: **no había margen que capturar**

El margen disponible es **negativo** (−0,0745): el comparador de media de celda —construido sólo
con train— **rinde peor** que el lineal. No es que las redes fallaran en cosechar un margen; es
que **el margen no está ahí**.

La razón está en los propios diagnósticos exigidos por la revisión externa:

* fracción media de autotomía **0,00175**,
* **ruido de etiqueta dentro de celda, SD = 0,00203**.

**El ruido supera a la señal.** El umbral se cruza —34 % de los episodios— pero *cuánto* se cruza
está dominado por la variación intra-celda. Una discontinuidad cuya posición es casi aleatoria
dado `ρ` no es una función aprendible, y `f2` reportó ese número **pase o falle**, como se
preregistró.

## 5. Qué queda del programa de prima neural

| generador | veredicto |
|---|---|
| G1 — precio del inventario (Cobb-Douglas) | curvatura **real**, pero **el spline gana a las dos redes** |
| **G2 — umbral duro de autotomía** | **discontinuidad real, margen ausente, redes por debajo del clásico** |
| G4 — observabilidad parcial | respondido por **Program Q**: `Δ_N` **TOST-equivalente a cero** dentro de ±0,01, con `H_OL` de +0,07…+0,12 |
| G3 — tres reclamantes asimétricos | **único sin construir** |

**Tres de los cuatro generadores están cerrados constructivamente**, y cada uno por una razón
distinta: G1 por *tipo* de no linealidad (suave y de una variable), G2 por *relación
señal-ruido*, Q por *equivalencia certificada*. Eso es un argumento mucho más fuerte para C&IE
que un «no encontramos prima».

**Precio de fidelidad declarado (`f6`):** `FDB` empeora `ret_mean` en 0,95 SE
(`docs/RESULTADO_CIERRE_AUTOTOMIA_2026-07-31.md`); la diferencia medida aquí en `ret_excel` es
+0,0023.

**Alcance, como se impuso antes de correr:** esto es **predicción**, no control. `H_regime` en
este carril sigue en ~1e-4 y G2 no autoriza entrenar política.
