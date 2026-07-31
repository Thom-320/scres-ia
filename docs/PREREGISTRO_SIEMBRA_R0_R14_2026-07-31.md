# Preregistro — cómo `R14` siembra `R⁰`

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Se ejecuta con `supply_chain/arm_runner.py`.

Sucede a `docs/POR_QUE_LA_VENTANA_NO_CAMBIO_NADA_2026-07-31.md` (`8482047`).

---

## 1. El defecto, medido y con su causa retractada

`RPj ≈ CTj` en nuestro modelo porque **el origen `R⁰` cae en `OPTj`**: sobre las órdenes con
`RPj > 0`, la distancia de `OPTj` al primer ref de `R14` en ventana tiene **p10 = p50 = 0,00 h**
y el 69,6% está por debajo de 24 h.

`R14` se modela como **compuerta persistente** —su bucket no se agota, a propósito— y esa
compuerta se manifiesta al inicio de cada orden. Con `RPj = OATj − R⁰`, eso da `RPj = CTj`.

**Retractado:** no es una fuga de mínimo global (`04654aa`). La lista de refs es una cola que
se drena; medido, 25 refs vivos al final de una corrida de 52 semanas.

**Y la comparación que lo hace un defecto y no una elección:** su `R14` toca el **98,1%** de
las órdenes, igual que el nuestro, y aun así **su `RPj` satura cerca de 400** mientras el
nuestro sigue a `CTj` sin cota. Su `RPj` correlaciona **0,88 con el conteo de riesgos** y solo
**0,37 con `CTj`**; el nuestro es `CTj` por construcción.

## 2. La tensión con el Algoritmo 2

El Algoritmo 2 (p.69) exige que el impacto **se manifieste dentro** de `[OPTj, OATj]`. Una
compuerta persistente que se manifiesta al **inicio de toda orden** cumple la letra y **vacía
la condición de contenido**: si siempre se cumple al instante inicial, no filtra nada.

## 3. Brazos

| brazo | cómo `R14` siembra `R⁰` |
|---|---|
| **A** (statu quo) | el mínimo de los refs pendientes, que cae en/junto a `OPTj` |
| **N** | `R14` marca el indicador pero **no siembra `R⁰`**; solo los riesgos de duración lo hacen |
| **E** | `R⁰` = el **instante real del evento `R14`** que aportó los defectuosos consumidos por esta orden |

**Los otros ejes quedan en su default** (`des_events`, `legacy_theatre_stock`, `elapsed`,
`clamped`, `le`, `union`, `δ` off). La lección de `413c9a9` y `6192460` es que estos ejes **no
se componen**, así que aislar es la única forma de atribuir un efecto.

**Cero parámetros libres** en los tres brazos.

## 4. Predicciones

**La fuerte es de FORMA, no de momento** — y es la firma que distingue su `RPj` del nuestro:

1. **`E` produce saturación.** La razón `RPj/CTj` debe **caer al crecer `CTj`**: en sus datos
   el p95 de esa razón es **0,19–0,21** y `RPj` se congela cerca de 400 para `CTj ≥ 1.000`.
   Criterio: en `E`, `RPj/CTj` mediana **por debajo de 0,60** para órdenes con `CTj > 500`,
   contra ~1,00 hoy. *Puede fallar:* si los defectos consumidos por una orden larga son
   antiguos, `R⁰` sigue siendo temprano y no hay saturación.
2. **`N` reduce la población `RPj > 0`.** Las órdenes tocadas **solo** por `R14` perderían
   `RPj`. *Riesgo declarado:* en sus datos esas órdenes **sí** tienen `RPj` y `P(k>0) = 66,4%`,
   así que **predigo que `N` poda de más** y sale peor que `E`.

**En `d_k`:**

3. **`E` mejora `rpj_mean` y `rpj_p95`.** Dirección declarada, magnitud no.
4. **`ret_mean`: sin dirección.** `RPj` baja, luego `0,5/RPj` sube; pero cambia la mezcla de
   ramas. No lo sé.
5. **`autotomy_share` sigue en 11,20.** El piso de `CTj` lo bloquea y ningún brazo lo toca.

**Regla de nivel contra `d_k`, ya declarada y usada en `6192460`:** `d_k` gobierna; se reporta
además el **`d_k` de SE apareada** (mismo numerador, error estándar del brazo `A`) como
**diagnóstico que no adopta nada**. Si `d_k` empeora mientras el de SE apareada mejora, el
veredicto es `RESIDUO_MAS_CIERTO_NO_MAS_GRANDE`.

## 5. Falsadores

| # | qué | puede fallar porque |
|---|---|---|
| f1 | `A` reproduce la línea base en los cinco momentos puntuados | cualquier perturbación del default |
| f2 | ninguna orden con `CTj < LT`, y `RPj ≤ CTj` siempre | `E` podría tomar un origen anterior a `OPTj` |
| f3 | en `E`, **todo `R⁰` de `R14` coincide con un evento `R14` real** del `risk_events` de esa corrida | si se sintetiza un instante, falla |
| f4 | **`A`, `N` y `E` difieren entre sí** en `rpj_mean` | tres veces esta sesión un eje fue ignorado en silencio; no lo doy por hecho |
| f5 | `epsilon` barrido; conjunto inestable se reporta inestable | — |

## 6. Aceptación

**Conjunto no dominado** sobre los cinco momentos puntuados (`scored_orders_per_year`
excluido hasta la referencia v4), `EPSILON = 0,5` **barrido**, ambas familias, `sum_dk` vetado
para rankear.

Un brazo entra si y solo si: **la predicción de forma §4.1 se cumple** para ese brazo;
`d_k(ret_mean)` no empeora más de `EPSILON` en ninguna familia; ningún otro momento puntuado
empeora más allá de `EPSILON`; los cinco falsadores pasan; y el conjunto es `epsilon`-estable.

**Si `E` produce la saturación pero los momentos no mejoran**, se reporta así: sería la
evidencia de que reproducimos la *forma* de su `RPj` sin reproducir su *nivel*, que es un
resultado y no un fracaso.

**Prohibido** elegir por `H_PI`, por contrastes MPC-contra-estático, por umbrales de servicio,
o por publicabilidad.

## 7. Declarado por adelantado

| ítem | valor |
|---|---|
| eje | cómo `R14` siembra `R⁰` |
| parámetros libres | **ninguno** |
| brazos | A, N, E |
| otros ejes | **todos en su default**, para aislar |
| raíces | **3.400.001–3.400.012**, disjuntas de todo bloque previo |
| momentos puntuados | 5 |
| predicción | §4, con §4.4 **sin dirección** y §4.2 prediciendo que `N` poda de más |

## 8. Alcance

**Nada se reetiqueta.** Fuera de alcance: `causal_exposure` y `op9_linked` (medidos, no
adoptados, y **no se componen** con nada), `δ`, el clamp, el predicado de banda, la referencia
v4, y la fuga de `f3` —que resultó ser en parte mi falsador y en parte este mismo defecto.

## 9. Firma

Requiere aprobación del PI.

La decisión que no me corresponde: si cambiar cuándo una compuerta persistente siembra `R⁰`
es un cambio de modelo aceptable. Mi lectura: la compuerta actual satisface el Algoritmo 2 de
forma vacía —se cumple siempre, al instante inicial— y una condición que nunca filtra no está
haciendo el trabajo que el algoritmo le pide. Pero mueve el `RPj` de toda orden con `R14`,
que son el 81% de las nuestras.
