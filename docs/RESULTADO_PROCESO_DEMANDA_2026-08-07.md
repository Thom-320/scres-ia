# Resultado — el proceso de demanda, medido

**Fecha:** 2026-08-07 · **Preregistro:** `docs/PREREGISTRO_PROCESO_DEMANDA_2026-08-07.md`
**Artefactos:** `results/demand_process/result.json` (sello `9711cb366b74f4d8…`) y `result_v2.json`
(añade la atribución `f5`). 12 semillas 8600001–8600012 × 2 entornos, acción neutra.

---

## Lo medido

| | thesis-native | track_b |
|---|---:|---:|
| media semanal | 15.627 | 16.385 |
| **CV semanal** | **0,0713** | **0,1090** |
| `acf1` | **−0,2283** (SE 0,0038) | −0,1761 (SE 0,0089) |
| banda iid (±2/√954) | ±0,0648 | — |
| pedidos/semana | **6,0000, sd 0,0000** | 6,0000, sd 0,0000 |
| % contingente (cantidad) | 8,0 % | 12,2 % |
| **semanas sobre capacidad S=1** | **24,8 %** | **60,0 %** |

`f1` **PASA**: 65.835 pedidos regulares nativos, mín. exactamente 2400, máx. exactamente 2600,
media 2500,21. `f2` **PASA**. `f4` **PASA** con su mecanismo visible: en track_b los pedidos
«regulares» van de **2280 a 2912** = 2400×0,95 y 2600×1,12, exactamente el rango de `demand_scale`
por régimen; los contingentes llegan a 11.053 contra 5.200 (`surge_scale`).

---

## `f3` FALLA — la demanda semanal NO es iid

`acf1 = −0,2283` contra una banda iid de ±0,0648, con SE 0,0038 sobre 12 episodios de 954 semanas.
Decae ordenadamente: `acf2` −0,170, `acf4` −0,015. Es reversión a la media, y es real.

## `f5` FALLA — y refuta MI hipótesis, no el dato

Propuse que la ACF negativa era **aliasing de calendario**: un ciclo de pedidos de 6 días no tesela
una semana de 168 h, las semanas alternarían entre 5 y 6 pedidos, y esa alternancia sola produciría
autocorrelación negativa sin ninguna memoria.

**Medido: 6,0000 pedidos por semana, sd 0,0000.** No hay alternancia. No hay aliasing.

Mi aritmética estaba mal y así se retracta: leí «media semanal regular 14.376 / 2500 = 5,75
pedidos» y concluí que faltaba un pedido en algunas semanas. Los 5,75 son los pedidos **regulares**;
el 0,25 restante son los **contingentes**, que en esas semanas ocupan el sexto hueco con ~5.000
unidades en vez de ~2.500. El conteo total nunca se mueve de 6.

### Un segundo defecto de falsador, el mismo día que escribí la regla contra él

`f5` comparaba `acf1(tamaño medio de pedido)` contra `acf1(conteo)`. Con el conteo constante, el
tamaño medio es la cantidad dividida por 6 — y **la correlación es invariante a escala**, así que
`acf1(tamaño) = acf1(cantidad) = −0,2283` **por construcción**. La comparación no tenía poder
discriminante una vez el conteo resultó constante.

No es tautológico *a priori* —si el conteo hubiera variado, el contraste habría sido informativo— y
refutó la hipótesis de aliasing, pero **por vía de mostrar que su premisa era falsa**, no por el
contraste que pretendía. Escribí R6 esta misma tarde tras el fallo de `F1` en L-0, y volví a
tropezar en la misma clase de error unas horas después.

> **Añadido a R6:** comprobar también que el estadístico del falsador **puede diferir** del que se
> compara. Un cociente por una constante nunca cambia una correlación.

---

## De dónde viene la ACF negativa — hipótesis, NO medida

Con 6 pedidos por semana y sorteos regulares iid, la única fuente posible de memoria son los surges
R24. Su inter-arribo es `uniform(1, 672)` h, con CV = (671/√12)/336,5 ≈ **0,58 < 1**: es un proceso
de renovación **sub-disperso** frente a Poisson, y esos anti-agrupan — un surge hace *menos*
probable otro la semana siguiente, lo que produce autocorrelación negativa en los totales semanales.

**Esto es aritmética del contrato, no una medición.** Hoy ya me equivoqué una vez derivando a mano
(el CV semanal de 0,94 % que resultó ser 7,1 %). Se declara como hipótesis principal y **no entra en
ningún claim** hasta medirse: el test sería la ACF de la serie de conteo de surges por semana.

---

## Lo que esto corrige de lo que yo mismo escribí hoy

| afirmación mía | estado |
|---|---|
| «CV semanal 0,94 %» | **falsa** — medido 7,1 % |
| «la demanda es iid, no hay estado que condicionar» | **falsa** — `acf1` −0,228, fuera de banda |
| «cuarta razón estructural del `H_regime` = 0» | **retirada** — sí hay estado de demanda |
| «5,75 pedidos/semana, hay aliasing» | **falsa** — 6,0000 exactos, sd 0 |

Lo que **no** cambia: que exista estado de demanda **no** implica que valga algo. El screen de
sensibilidad al riesgo ya midió que la postura constante óptima es invariante en 45 perfiles
(`H_profile_safe` máx. 6,9e-05 contra barra 0,01). Son dos claims distintos y el segundo sigue en
pie con su propia evidencia.

---

## Consecuencia para la petición de Garrido

Garrido pide sustituir la demanda por suavización exponencial con estacionalidad, porque «la actual
es uniforme discreta, variación mínima, y eso hace que el modelo aprenda fácil».

- Su **premisa cuantitativa es falsa**: CV semanal 7,1 %, y 24,8 % de las semanas ya exceden la
  capacidad de un turno.
- Su **inferencia** tampoco se sostiene tal cual: el empate entre arquitecturas tiene causa medida
  —curvatura 0,076 contra ruido 0,317, con el MLP peor que una recta—, así que subir varianza sin
  subir curvatura refuerza el empate.
- Pero su **prescripción sigue siendo la palanca correcta**, por una razón distinta a la suya: la
  memoria que hoy existe es **negativa, débil y no direccional** (anti-agrupamiento de surges).
  Estacionalidad daría memoria **positiva, persistente y con fase** — el tipo de estado que una
  política condicionada puede explotar y una constante no.

Es decir: hacerlo sí, con el argumento corregido.

---

## Custodia

Datado, no se edita en sitio. `result.json` y `result_v2.json` conviven; v2 añade `f5`,
`acf1_order_count`, `acf1_mean_order_size` y `orders_per_week_*`. Ningún falsador se retira tras ver
su resultado.
