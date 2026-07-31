# Resultado — la métrica escondía el headroom. `ret_excel` es de las peores, por 65×

**Artefacto:** `results/sensitivity/multi_metric_headroom_v1/result.json` (sello
`863c21c1fa9ecc4f…`) · 4.375 corridas, **16 métricas + Cobb-Douglas sobre las MISMAS corridas** ·
**los cinco falsadores pasan**.

## 1. Primero, la respuesta honesta a «¿están completos los análisis?»

**No.** Tres huecos, nombrados:

| análisis | cubierto | **NO cubierto** |
|---|---|---|
| **nodos** | 13 variables de decisión + 3 buffers, `S1` y `S_T` | **nodos nuevos** (topología); y **nunca se calcularon los índices de segundo orden `S_ij`** — sabemos *cuánta* interacción tiene cada factor, no **con quién** |
| **riesgos** | qué familia activa, escalas **globales** de frecuencia e impacto | **frecuencia e impacto POR RIESGO** (9 × 2 = 18 factores). Su permiso es por riesgo; nosotros usamos solo el escalar global |
| **combinado** | la mezcla de familias × 3 palancas aguas abajo | el cruce **nodo × riesgo específico**, que es precisamente donde `S_ij` respondería |
| **métricas** | ahora sí: 16 + Cobb-Douglas | — (era el hueco declarado; ya cerrado) |

**Y una corrección de coste:** aplacé Cobb-Douglas estimando **~300×**. Medido: **1,0×**
(0,054 s contra 0,053 s con muestreo semanal). El aplazamiento se basó en una estimación falsa.

## 2. La métrica cambia el resultado por 65×

`H_regime` normalizado por la dispersión propia de cada métrica — sin normalizar, una métrica de
rango ancho «gana» gratis:

| métrica | `H` | LCB95 | **`H/SD`** |
|---|---:|---:|---:|
| **`ret_excel_risk_conditional`** | 0,000307 | 0,000190 | **0,131** |
| **`ret_thesis`** | 0,000135 | 0,000075 | **0,069** |
| `ret_excel_rolling_4w_min` | 0,000005 | 0,000002 | 0,030 |
| **`flow_fill_rate`** | **0,004635** | 0,003459 | 0,028 |
| **`cobb_douglas_index`** | 0,000239 | 0,000160 | **0,021** |
| `ret_continuous` | 0,001091 | 0,000964 | 0,004 |
| **`ret_excel`** (la canónica) | 0,000345 | 0,000238 | **0,002** |
| `ret_excel_cvar05` / `cvar10` / `p05` | ~0,00005 | ~0,00004 | **0,001** |

**La canónica está entre las peores.** El headroom es **65× mayor** medido sobre ReT
**condicional a riesgo** que sobre `ret_excel`.

## 3. Las tres lecturas

**La ganadora tiene mecanismo, no es un accidente.** `ret_excel_risk_conditional` puntúa **solo
los pedidos tocados por riesgo**. `ret_excel` mezcla esos con la rama de fill-rate sin riesgo, y
esa rama —que no depende de la decisión aguas abajo— **diluye** la señal. Es la misma patología
que ya medimos: bajo R1r el 100% de los pedidos cae en la rama de recuperación, y bajo R3 el
99% en la de fill-rate; promediarlas tapa la parte que la decisión mueve.

**Cobb-Douglas confirma tu recomendación:** 0,021, **10× por encima** de `ret_excel`. Es una
segunda familia legítima, con la ventaja de que su índice **es sensible al coste** —lo que
`ret_excel` ignora por completo—.

**CVaR NO gana aquí, y hay que decirlo.** `cvar05`, `cvar10` y `p05` quedan **al fondo**
(0,001). La hipótesis de que «las colas mostrarían mayor magnitud» **queda refutada sobre estos
datos**: la cola inferior de ReT está dominada por pedidos que ninguna decisión aguas abajo
alcanza, así que condicionar en la cola **quita** señal en vez de añadirla. Lo que sí funciona es
condicionar en **riesgo**, no en cuantil.

**Y el mayor headroom bruto no es de resiliencia:** `flow_fill_rate`, con `H = 0,0046` — **25×**
el de `ret_excel`. El servicio responde a la decisión aguas abajo mucho más que cualquier índice
de resiliencia. Es incómodo para el encuadre del paper y por eso va aquí.

## 4. Qué implica para la estrategia

1. **La métrica primaria para buscar headroom debería ser ReT condicional a riesgo**, con
   `ret_excel` reportada al lado por fidelidad con la tesis. Eso no cambia la métrica del paper:
   cambia **con qué medimos si hay algo que aprender**.
2. **Cobb-Douglas entra como segunda métrica**, ya sin la excusa del coste.
3. **CVaR queda como métrica secundaria de seguridad**, no como buscador de headroom — que es
   exactamente el papel que la reunión del 22 de julio le asignó.
4. Los huecos de §1 —`S_ij`, riesgos por-riesgo, nodos nuevos— siguen abiertos, **en ese orden**:
   `S_ij` es barato y dice *dónde* poner el nodo antes de construirlo.

## 5. Límites

* Cadencia **semanal** en todas las celdas; `ret_excel` es dependiente de la cadencia, así que
  los **niveles** no se comparan con corridas de otra cadencia. La diferencia está medida en `f1`.
* Exponentes de Cobb-Douglas **de la calibración congelada**, no de este barrido: `derive_exponents`
  rechaza un máximo ≤ 1 y aquí `tau` llega a 0,28. Re-derivarlos por experimento haría el índice
  incomparable entre experimentos.
* `H_regime` sigue suponiendo el **régimen observado**: es un techo, no un alcanzable.
