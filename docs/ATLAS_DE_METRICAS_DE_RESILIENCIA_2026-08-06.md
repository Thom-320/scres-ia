# Atlas de métricas de resiliencia — dónde hay headroom, y qué endpoint podría cambiar la historia

**Pregunta:** ¿alguna de las métricas que hemos construido cambia el veredicto de «no hay headroom
dependiente del régimen»? Y en particular, ¿Cobb-Douglas o el ReT del Excel?

**Respuesta corta: ninguna lo cambia — y eso es ahora una medición, no un argumento.** Pero
difieren brutalmente en **qué premian**, y ahí sí hay un hallazgo publicable.

---

## 1. El atlas nuevo: mismo estimador, mismas semillas, sólo cambia el endpoint

`results/endpoint_headroom_atlas/result.json` · `NO_ENDPOINT_CARRIES_REGIME_HEADROOM`

| endpoint | H_regime (288) | **H_regime (4.608)** | spread bruto |
|---|---:|---:|---:|
| `ret_excel_risk_conditional` | **+0,00380** | **+0,02829** | 0,0053 |
| `ret_excel_full_ledger` | +0,00028 | **+0,00978** | 0,101 |
| `ret_excel` (visible) | +0,00050 | +0,00045 | 0,172 |
| `delivered_rations` | +0,00015 | 0,00000 | 4,8e5 |
| `flow_fill_rate` | +0,00008 | **0,00000** | 0,539 |
| `demanded_rations` | 0,00000 | 0,00000 | 8,8e4 |
| `lost_orders` | 0,00000 | 0,00000 | 108 órdenes |

**Ninguno llega al umbral de 0,05 en ninguna de las dos rejillas.** La objeción «lo mediste con una
métrica rota» queda cerrada **empíricamente sobre siete endpoints y dos espacios de diseño**.

Tres lecturas que van al manuscrito:

**El endpoint que veníamos usando es el más favorable de los siete.** Todos nuestros nulos se
midieron donde el headroom era más probable, no donde era más cómodo.

**El full-ledger es el más sensible a añadir variables de decisión**: ×35 al pasar de 288 a 4.608,
contra ×7,4 del risk-conditional. Es lo que cabe esperar de la única variante que puntúa los
pedidos no servidos.

**`flow_fill_rate` y `lost_orders` tienen autoridad real de palanca —0,54 y 108 órdenes de
recorrido— y headroom exactamente cero.** Es la misma forma que encontró el barrido de capacidad:
la palanca mueve mucho, y el óptimo no se mueve con el régimen.

## 2. Lo que ya estaba medido, y concuerda

`results/sensitivity/multi_metric_headroom_v1/result.json` — 4.375 corridas, **16 métricas más
Cobb-Douglas sobre las mismas corridas**:

| métrica | H_regime | H/SD |
|---|---:|---:|
| `ret_excel_risk_conditional` | 0,000307 | **0,131** ← el mejor normalizado |
| `flow_fill_rate` | **0,004635** ← el mayor en bruto | 0,028 |
| `ret_thesis` | 0,000135 | 0,069 |
| `cobb_douglas_index` | 0,000238 | 0,021 |
| `ret_excel` | 0,000345 | **0,002** ← de los peores |
| `ret_excel_cvar05/10` | 3,5e-5 / 5,2e-5 | 0,001 |

**Dos hipótesis quedan refutadas por medición**: que las colas mostrarían más magnitud —**CVaR está
en el fondo**— y que la métrica canónica sería la informativa —**es de las peores**, 65× por debajo
del risk-conditional en headroom normalizado.

Y **mi atlas y este banco coinciden de forma independiente** en que `ret_excel_risk_conditional` es
el endpoint con más headroom relativo. Dos diseños distintos, misma conclusión.

## 3. El único headroom grande que existe, y por qué no nos salva

`H_PI = 0,15151` (LCB95 0,11562) en **Program O**, endpoint `ret_visible` /
`ret_excel_request_snapshot_v2`. Es el único techo material de toda la búsqueda.

Pero: es **otra física** (dos productos no fungibles compartiendo Op5–Op7), su **`Δ_N` es negativo
en las tres celdas**, y el estatus del convenio de ledger sigue siendo
`PROVISIONAL_PENDING_GARRIDO_SIMULINK_CONFIRMATION`. **No es una métrica que nos rescate: es un
mecanismo distinto que ya cerró en negativo.**

## 4. Lo que las métricas SÍ deciden — y esto es el hallazgo

El veredicto de headroom no cambia. **Lo que cambia radicalmente es qué configuración recomienda
cada métrica.**

**El patrón estable en 18 celdas** (`docs/RESULTADO_SERVICE_FIRST_V2_CONTENCION_2026-08-01.md`):
`ret_excel` **siempre** elige un extremo —0,1 o 0,9, nunca 0,5— y **todo endpoint sano siempre
elige 0,5**. El *lado* del extremo es inestable entre bloques de semillas; lo estable es que elige
un extremo.

**`ret_excel` premia el abandono, medido**: el reparto que lo maximiza entrega **50 %** de las
raciones; el que lo minimiza entrega **80 %**. Doce veces más «resiliencia» comprada con 30 puntos
de servicio, en las seis celdas.

**Cobb-Douglas es ciego al servicio por otra vía**: un pedido que nunca se sirve **sale de la cola
de backorders y deja de costar**. No hay término de fill-rate entre sus cinco variables.

**La rama de autotomía está muerta y eso mueve el ReT 221,6×**: con `GARRIDO_FULFILLMENT_DELAY_HOURS
= 54` contra `LT = 48`, ReT vale 0,004424; con 48, **0,980513**. Seis horas de diferencia.

**El ReT depende de la cadencia de paso**: 37 % de dispersión entre cadencias sobre trayectorias
idénticas, y **la cadencia invierte la política recomendada** (0,9 con `sim.run()`, 0,1 con paso
diario). Reparado prospectivamente a dispersión exactamente 1,0.

**Y el hallazgo más incómodo de todos**: bajo recurso fungible, el `H_regime` de
**`ret_excel_omitted_n` es 0,375** y el de `ret_excel_visible_n` es **0,43** — contra 1e-5 del score
en sí. **Lo que varía con el régimen no es la resiliencia: es cuántos pedidos quedan excluidos de la
puntuación.**

## 5. Qué significa esto para C&IE

Garrido cita a **Bruckler et al. 2024, C&IE 192, 110176** — una revisión de métricas de SCRES. Este
atlas le habla directamente:

> Cuatro métricas de resiliencia construidas de forma independiente —el ReT del Excel de Garrido,
> su variante de ledger completo, su índice Cobb-Douglas de 2024, y un endpoint service-first
> propio— **coinciden en no encontrar headroom dependiente del régimen** y **discrepan por completo
> en qué configuración recomiendan**. Dos de ellas premian dejar de servir, por mecanismos
> distintos: censura en una, ausencia de precio del pedido perdido en la otra.

Eso es una contribución metodológica, no un resultado negativo. Y es **exactamente el hueco que
Garrido señala en su §6.2**: *«otros drivers importantes de la resiliencia de fábrica no fueron
tenidos en cuenta… futuros estudios deberían incluir eventos de riesgo recurrentes»*. Nuestra cadena
**los tiene**, y es al aplicarle su índice a una cadena con riesgo cuando aparece la ceguera.

## 6. Cobb-Douglas: la regla de exponentes se rompe en nuestra cadena

Aplicamos **su propia regla** —`exponente = 0,20/ln(x_max)`, sobre NUESTROS máximos, como manda—
y sale esto:

| var | su max (implícito) | nuestro max | su exp | **nuestro exp** | ratio vs ζ |
|---|---:|---:|---:|---:|---:|
| ζ inventario | 4.160 | 1.216.000 | 0,024 | 0,01427 | 1,0× |
| ε backorders | 2.191 | 136.700 | 0,026 | 0,01691 | 1,2× |
| φ capacidad ociosa | 148 | 2.779 | 0,040 | 0,02522 | 1,8× |
| **τ tiempo** | **28** | **1,343** | 0,060 | **0,67810** | **47,5×** |
| **κ̇ coste** | **3** | **1,565** | 0,1771 | **0,44633** | **31,3×** |

*(su max implícito = `exp(0,20/exponente_publicado)`; el de ζ da 4.160 contra el 3.612 que él
reporta, diferencia de redondeo en el tercer decimal del exponente)*

**En su ajuste el peso máximo es κ̇/ζ = 7,4×. En el nuestro es τ/ζ = 47,5×.**

El mecanismo es aritmético: la regla divide por `ln(x_max)`, así que **cuanto MENOR es el recorrido
observado de una variable, MAYOR es su exponente**. Está pensada para variables cuyo máximo son
miles —los suyos lo son—. Nuestro τ recorre `[1 · 1,343]`, así que `0,20/ln(1,343) = 0,678`.

Y la variable que se queda con el 68 % del peso **es la que está muerta**: nuestro τ_max es 1,343
contra los 28 suyos, porque el punto de operación de la tesis lleva stock suficiente para que los
requerimientos netos casi nunca se vuelvan positivos.

La amplificación de error `1/ln(x_max)` lo confirma: **τ 3,39 y κ̇ 2,23, ambas mal condicionadas**
—un error relativo en el máximo observado se amplifica más que proporcionalmente en el exponente—
frente a 0,07 de ζ. Nuestro puerto ya lo declaraba como aviso; ahora está medido.

> **Su índice no es transportable a una cadena con riesgo tal cual.** No porque la fórmula esté
> mal, sino porque **la regla de normalización presupone que las cinco variables tienen recorridos
> comparables en órdenes de magnitud**, y en una cadena militar con eventos de riesgo no los
> tienen. Es una limitación de alcance de su Ec. (5), y es exactamente el tipo de hallazgo que su
> §6.2 invita a producir.

## 7. Lo que falta medir, y corre ahora

`H_regime` **por componente** del Cobb-Douglas (ζ, ε, φ, τ, κ̇) en vez de sobre el escalar. La
hipótesis es concreta: el cero del escalar podría ser un **artefacto de agregación**, porque los
pesos son muy desiguales —κ̇ pesa ~7× ζ en el ajuste de Garrido— y **τ está muerta en nuestra
cadena** (exactamente 0 en 88 de 108 episodios de calibración, porque el punto de operación de la
tesis lleva stock suficiente para que los requerimientos netos nunca se vuelvan positivos).

Si algún componente lleva headroom que el escalar esconde, **el cero es del índice**. Si ninguno lo
lleva, **el cero es de la cadena** — y eso es mucho más fuerte de lo que hemos podido afirmar hasta
ahora.
