# Preregistro — propagación de riesgos, y por qué NO recomiendo ejecutarlo ya

**Estado:** `PREREGISTRATION_DRAFT_NOT_RECOMMENDED_FOR_EXECUTION`. Redactado como se pidió.
La recomendación de no ejecutarlo todavía es mía y va con su evidencia; la decisión es del PI.

## 1. El cambio propuesto

En el modelo de Garrido, un riesgo puede tocar una orden sin retrasarla: sus 110 filas de
autotomía absorben entre 0,45 y 48,04 h de disrupción con sobrepasos de 0,007 a 0,048 h.
Es el efecto de autotomía de cola de su Figura 5.2 — la cadena se desprende de la parte
afectada y el resto sigue.

En el nuestro, un riesgo que toca una orden **siempre** consume tiempo real en su ruta. Por
eso la cuota de autotomía no puede emerger; solo puede imponerse por parámetros, como
demostró `TRANSIT_REPLACEMENT_RESULT_2026-07-30.md`.

El cambio sería dar a los riesgos una probabilidad de no propagarse a una orden ya
comprometida — que la orden fluya mientras la operación está caída, si hay stock aguas
abajo que la cubra.

## 2. Por qué recomiendo NO ejecutarlo todavía

**El momento que cerraría vale 0,488% del endpoint primario.**

| | órdenes por configuración | de |
|---|---:|---:|
| R1r | **10,4** | 2.381 |
| R2r | **1,4** | 2.169 |

Ignorar la autotomía por completo mueve la media global de ReT de 0,10472 a 0,10421.

**Y hay momentos con 10 a 20 veces más apalancamiento, sin tocar:**

| momento | referencia R1r | nuestro | desvío |
|---|---:|---:|---|
| `rpj_mean` | 193,70 | ~440 | **2,3×** (≈10 SD de referencia) |
| `rpj_p95` | 456,50 | ~2.363 | **5,2×** |
| `ret_mean` | 0,00633 | ~0,0034 | 1,9× |

`RPj` lleva **toda** la señal de ReT: con la rama de autotomía muerta, cada orden puntuada
cae en `0.5/RPj`. Estamos entre 2 y 5 veces desviados en el único insumo que la métrica
consume, y ese desvío afecta al 100% de las órdenes, no al 0,44%.

**La secuencia correcta es `RPj` primero.** Si se corrige `RPj` y la distribución de tiempos
de ciclo se acerca a la suya, es posible que la cuota de autotomía se mueva sola — porque
ambas salen de la misma física de disrupción. Ejecutar la propagación antes arriesga
ajustar dos cosas a la vez sin saber cuál movió qué.

## 3. Si aun así se ejecuta: el diseño

**Se barre**, nunca se ajusta: la probabilidad de no-propagación `p ∈ {0, 0.1, 0.25, 0.5}`,
declarada por adelantado, con la regla de no-selección del contrato v2 intacta — se reportan
todas las filas y ninguna se elige por el resultado que da.

**Condición física, no libre:** una orden solo puede evitar la propagación si hay stock
aguas abajo que la cubra en el momento de la disrupción. Sin esa condición `p` es otro
parámetro puro, y volveríamos al problema de `op11_handling_hours`.

**Criterio:** ε-dominancia sobre los momentos, con el conjunto de momentos **corregido** —
ver §4.

**Falsador:** si la cuota de autotomía resulta ser una función monótona de `p` sola,
independiente del stock aguas abajo, entonces la condición física no está mordiendo y el
mecanismo es cosmético. Se detiene ahí.

**Raíces:** 2.100.001–2.100.012, disjuntas de todo bloque previo.

## 4. Un defecto del conjunto de momentos, a corregir antes de cualquier barrido

`scored_rows` **no debe estar en el ajuste**. Referencia 2.381, nuestro ~215 — un factor 11
que no es infidelidad sino que sus corridas son de 20 años y las nuestras de 52 semanas. Es
un artefacto de horizonte y lo incluí sin pensarlo, así que hoy contamina toda distancia
agregada que lo promedia.

Debe **excluirse**, o normalizarse por unidad de tiempo si se quiere conservar como control
de población. Esto afecta a los barridos ya corridos: sus distancias agregadas incluyen un
término espurio de ~11 SD idéntico en todas las celdas. **No cambia los rankings** —es una
constante aditiva por celda dentro de cada familia— pero infla toda la escala y debe
declararse.

## 5. Qué autoriza y qué no

**Autorizaría:** medir si dar a los riesgos una vía de no-propagación acerca la cuota de
autotomía a 0,44%/0,064% sin empeorar `RPj`, `ReT` y la cola.

**No autorizaría:** ajustar `p` hasta que la cuota coincida. Ese es el procedimiento que
produjo `delay = 54` y que `op11_handling_hours` estuvo a punto de repetir.

## 6. Firma

Requiere aprobación del PI. Mi recomendación es posponerlo detrás de `RPj`, y está aquí
por escrito para que la decisión sea explícita y no por omisión.
