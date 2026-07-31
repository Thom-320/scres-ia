# Result — `δ` como supuesto: predije no-adopción y se cumple, pero `ret_mean` se mueve mucho

**Status:** `PREREGISTERED_NO_ARM_ADOPTABLE`. Artefacto
`results/metric_audit/delta_assumption_arms_v1/result.json`, sellado. Raíces 3.000.001–12.

> **RE-CORRIDO 2026-07-31 tras arreglar el defecto de `op9_linked`.** `δ` no se aplicaba en
> la ruta física (`_deliver_order_from_op9` llama directo a `_finalize_pending_backorder`),
> así que `L` y `LD` salían idénticos y el supuesto **nunca se probó bajo el enlace**.
> Corregido: `δ` se aplica ahora en **ambas** rutas, y el brazo `LD` pasa de 47 a **107
> valores distintos sobre 107 órdenes**.
>
> **El resultado corregido refuta el supuesto como beneficioso.** Con `δ` correctamente
> aplicado bajo el enlace:
>
> | momento (R1r) | A | D | **L** | **LD** |
> |---|---:|---:|---:|---:|
> | `ret_mean` | 1,86 | 0,80 | **0,23** | **0,75** |
> | `rpj_mean` | 7,28 | 7,39 | 7,95 | **8,16** |
> | `rpj_p95` | 11,32 | 11,32 | 12,11 | **12,15** |
>
> `LD` empeora `ret_mean` contra `L` (0,23 → 0,75) y empeora los dos `rpj`. **`LD` sale del
> conjunto no dominado**, que pasa de `[D, LD, L]` a **`[D, L]`**. Sigue sin haber brazo
> adoptable y `f6` sigue fallando.
>
> **Conclusión: añadir `δ` no ayuda.** La mejora de `ret_mean` viene enteramente de
> `op9_linked`, y `δ` encima la degrada. El supuesto queda medido y descartado como palanca,
> que es exactamente lo que un supuesto declarado debe poder recibir.

| brazo | min | `CTj` p50 | demoradas | `δ` p25 | `δ` p50 | `δ` p75 |
|---|---:|---:|---:|---:|---:|---:|
| **Garrido** | 48,01 | **101,45** | **83,5%** | **2,00** | **4,02** | **6,00** |
| A legacy, sin `δ` | 54,00 | 54,00 | 38,0% | 0,00 | 6,00 | 6,00 |
| D legacy + `δ` | 54,00 | **60,35** | 38,0% | 0,00 | 7,40 | 10,52 |
| L linked, sin `δ` | 54,00 | **78,00** | **54,0%** | 6,00 | 6,00 | 6,00 |
| LD linked + `δ` | 54,00 | **78,00** | **54,0%** | 6,00 | 6,00 | 6,00 |

## 1. La predicción se cumple: **ningún brazo es adoptable**

Lo declaré por adelantado (§5 del contrato) precisamente para que adoptarlo hubiera sido
informativo. No lo fue.

## 2. Pero `ret_mean` se mueve, y mucho

`d_k` sobre los cinco momentos puntuados, R1r:

| momento | A | D | L | LD |
|---|---:|---:|---:|---:|
| **`ret_mean`** | **1,86** | **0,80** | **0,23** | **0,23** |
| `rpj_mean` | 7,28 | 7,39 | **7,95** | **7,95** |
| `rpj_p95` | 11,32 | 11,32 | **12,11** | **12,11** |
| `autotomy_share` | 11,20 | 11,20 | 11,20 | 11,20 |
| `ret_above_one_share` | 3,90 | 3,90 | 3,90 | 3,90 |

**`ret_mean` —el endpoint del manuscrito— pasa de 1,86 a 0,23**, un factor 8. Y el conjunto
no dominado de R1r es **`[D, LD, L]`**: el statu quo `A` queda **dominado** por los tres.

**Y aun así no se adopta**, porque `rpj_mean` y `rpj_p95` empeoran más allá de `EPSILON` y
**el falsador `f6` falla**: el conjunto no dominado **se mueve con `epsilon`**, así que por
la regla del contrato maestro se reporta **inestable** en vez de mostrarse como resultado.

Esa es la regla funcionando: sin el barrido de `epsilon` —que ninguno de los cuatro runners
del 2026-07-30 implementaba— este habría sido el primer «ganador» de la sesión.

## 3. El defecto de código se repite y ahora muerde aquí

**`L` y `LD` son idénticos en cada cifra.** Bajo `op9_linked`, el sorteo de `δ` **nunca
ocurre** — la ruta enlazada no pasa por el punto donde se añade. Es el mismo defecto que
`5c09437` encontró con `fulfillment_transit_mode`, y ahora afecta a un segundo eje.

**Consecuencia:** el factorial 2×2 de este contrato es en realidad **tres celdas**, y el
supuesto `δ` **solo se probó bajo `legacy_theatre_stock`**. La combinación que el contrato
quería medir —enlace más `δ`— no se ha medido.

## 4. `δ` no sale `U(0,8)` en el brazo D, y es esperable

`δ` p25 = 0,00 y p75 = **10,52** en `D`, contra `U(0,8)`. No es un fallo del sorteo: la
métrica `(CTj − 48) mod 24` mezcla el sorteo con el término base, así que no aísla `δ`. La
comprobación de construcción quedó registrada como tal —**no** como falsador— exactamente
para no leer esto como evidencia en ninguna dirección.

## 5. Falsadores

`f3` (ninguna orden bajo `LT`), `f4` (con `δ` apagado su flujo queda intacto) y `f5` (los
flujos de riesgo y demanda no se perturban) **pasan**. `f6` falla, como arriba.

El aislamiento del sorteo funciona: `δ` no toca ningún otro flujo.

## 6. Lo que esto deja

* **`ret_mean` tiene una ruta medida a 0,23 `d_k`**, y viene de `op9_linked`, no de `δ`
  (`L` sin `δ` ya da 0,23). Es el hallazgo con más valor de la sesión y **no se puede adoptar
  todavía** por la inestabilidad de `epsilon` y por `rpj`.
* **El supuesto `δ` no está descartado**: nunca se probó bajo el enlace, por el defecto de §3.
* **La brecha de `rpj`** es ahora el obstáculo, no `ret_mean`.

## 7. Estado

Nada implementado, ningún default movido. El artefacto está sellado con los momentos
reportados —los falsadores de instrumento pasaron— y con el veredicto marcado inestable.
