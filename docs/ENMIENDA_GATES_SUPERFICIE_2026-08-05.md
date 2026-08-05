# Enmienda — los dos gates de superficie, con estimadores que se pueden citar

**Escrita ANTES de correr.** Sustituye el §5 de
`docs/PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md`, que declaró los dos gates pero **no** su
estimador. Runner: `scripts/run_surface_gates_v1.py`. **Sin simulación nueva**: opera sobre
`results/surface_cache/wrap288_v1/` (72 rebanadas, bloque quemado `5.300.001–012`).

## 1. Por qué hace falta la enmienda

La corrida de ayer reportó ganancia por interacciones de **0,122–0,164** y «el argmax se mueve en
4 de 6 contextos». **Ninguna de las dos cifras es citable**:

* la ganancia de R² se calculó **en muestra**, con 6 parámetros extra sobre 288 puntos. Es
  exactamente el defecto que el propio plan advertía sobre `garrido_continuous_space` y que luego
  repetí;
* «el argmax se mueve» es casi gratis con ruido, y **no mide lo que importa**: si moverse **vale**
  algo.

## 2. `g2` — separabilidad, con validación cruzada agrupada por semilla

Sobre las filas `(config, seed)` de cada contexto:

* **modelo aditivo**: one-hot de cada nivel de los cuatro factores, sin interacciones;
* **modelo con interacciones**: lo anterior más los productos de las seis parejas de factores;
* **CV dejando una semilla fuera** (12 pliegues, uno por semilla). La semilla es la unidad de
  resampling declarada, así que la partición se hace por semilla y no por fila;
* estadístico: `ΔCV-R² = R²_interacciones − R²_aditivo`, un valor por semilla retenida;
* **LCB95 por bootstrap sobre las 12 semillas**, 5.000 remuestreos.

**Umbral: `LCB95(ΔCV-R²) ≥ 0,05` en al menos un contexto**, y se reporta contexto a contexto.

*Por qué puede fallar:* si la superficie es separable, el modelo aditivo predice igual de bien
fuera de muestra y `ΔCV-R²` cae a cero o se vuelve negativo por el coste de los parámetros extra.
Y si falla, **OFAT es casi óptimo por construcción** y el carril de búsqueda no tiene mecanismo.

## 3. `g1` — que el óptimo **valga** moverse, no sólo que se mueva

El estimando es `H_regime`, con la media sobre semillas **antes** del máximo sobre acciones —el
error que ya infló un resultado de E\*-C diez veces:

```text
H_regime = media_ctx[ max_cfg( media_semillas(V) ) ] − max_cfg( media_ctx( media_semillas(V) ) )
```

`V` es el valor **normalizado dentro de cada contexto** por su propio rango
`(v − min_ctx) / (max_ctx − min_ctx)`, porque `ret_excel_risk_conditional` vale ~0,009 en R1r y
~0,8 en R2r y sin normalizar el promedio entre contextos sería aritmética sin sentido. Así
`H_regime` se lee como **fracción del rango alcanzable que compra conocer el régimen**.

**LCB95 por bootstrap sobre semillas, umbral `≥ 0,05`.**

Se reporta además, como diagnóstico y no como gate, el argmax por contexto y el punto común.

## 4. Reglas de lectura

* **ambos gates pasan** → `SURFACE_SUPPORTS_A_SEARCH_LANE`: la superficie es no separable **y**
  adaptar al contexto vale algo. Abre la escalera de comparadores. **No autoriza entrenar.**
* **`g2` pasa y `g1` no** → `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`: la búsqueda es difícil pero la
  respuesta es la misma en todos los contextos. El carril sigue vivo para comparar buscadores, y
  **la transferencia deja de tener objeto**: se escribe que el valor de la memoria es el de no
  re-derivar una constante.
* **`g2` no pasa** → `STOP_SEPARABLE_SURFACE_OFAT_IS_NEAR_OPTIMAL`. No se añade física para
  rescatarlo dentro de este contrato.

**Alcance:** desarrollo sobre tapes quemados, sin semillas nuevas, sin adjudicación.
