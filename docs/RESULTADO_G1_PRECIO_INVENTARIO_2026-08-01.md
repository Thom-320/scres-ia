# Resultado G1 — RETRACTADO Y REVERTIDO: **G1 se sostiene**

> ## ⚠️ RETRACCIÓN 2026-08-01 — la conclusión de abajo está INVERTIDA
>
> Una revisión externa señaló que `f4` no probaba monotonicidad: `max()` devuelve **el primero de
> un empate**, así que un perfil que sube y luego queda **plano** reportaba un falso «óptimo
> interior». Lo comprobé y **tenía razón**.
>
> Con el test correcto —**diferencias sucesivas** y **conjuntos óptimos con tolerancia**—:
>
> | | resultado |
> |---|---|
> | `ret_excel` | **monótona en las seis celdas** |
> | `flow_fill_rate` | **monótona en las seis**; `R1r\|base` tiene conjunto óptimo **`[1008, 1176, 1344]`** — empate a tres |
> | **Cobb-Douglas** | **estrictamente interior** en `R1r\|base` y `R2r\|base`, sin empates, con diferencias negativas después |
>
> **Por tanto G1 se SOSTIENE.** Cobb-Douglas curva **porque cobra el inventario**; ReT y fill
> **saturan** porque no lo cobran. Es exactamente la hipótesis de G1, y la rechacé por un
> artefacto de mi propio falsador.
>
> **Se retiran de este documento:** «el óptimo interior es físico, no de la métrica», «todo
> curva», y «la primera curvatura del proyecto» (el barrido de contención ya tenía U — que
> resultaron ser censura de `ret_excel`). Lo correcto: **la primera auditoría directa del perfil
> de `op9_rations`**.
>
> **Y dos errores más, ambos señalados y ambos ciertos:**
>
> * **`τ = 0,79092` es del SMOKE de 2 semillas**, no de la corrida de 6, que da **0,8829**.
>   Transcribí el número equivocado.
> * **«τ disfrazado» es una mala interpretación.** Un exponente grande **no** es una cuota grande
>   del índice: por construcción `exponente × ln(x_max) ≈ 0,20` en su propio máximo. Indica **mal
>   condicionamiento**, no dominancia. `f8` queda además **degradado a diagnóstico no vinculante**
>   porque se añadió a mitad de análisis, y **un check post hoc no puede detener uno
>   preregistrado**.
>
> Artefacto corregido: `results/headroom/g1_buffer_price/result.json`, re-sellado con `f3` por
> conjuntos óptimos, `f4` por diferencias sucesivas y los perfiles completos almacenados — que el
> artefacto original **no guardaba**, y por eso el empate no era auditable desde él.
>
> **Consecuencia para la prima de predicción:** su resultado **no cambia de signo** (se midió
> sobre `ret_excel`, monótona, curvatura 0,076 contra ruido 0,317), pero **su encuadre sí**: la
> superficie medida no era la curva de Cobb-Douglas. La condición *«curvatura > ruido»* se
> mantiene; **falta medir la prima sobre la superficie de CD**, que es la que tiene el máximo
> estricto.

---

## Texto original, conservado como registro de la lectura equivocada

**Artefacto:** `results/headroom/g1_buffer_price/result.json` (sello `b3cc51cc0e819bcf…`,
`HALTED_FALSIFIER_FAILED`) · preregistro `docs/PREREGISTRO_G1_PRECIO_INVENTARIO_2026-08-01.md`,
commiteado antes de correr · 6 semillas, 54 celdas, cadencia diaria.

**No reporto el veredicto** (`G1_GENERATES_CURVATURE`) porque dos falsadores fallan. Pero los dos
fallos son informativos y cambian el programa.

## 1. Mi premisa era falsa: **todo curva**, no sólo Cobb-Douglas

`f4` era el control de contraste: *si `ret_excel` también curva, la diferencia no es atribuible a
cobrar el inventario*. **Falla.**

| régimen | argmax **CD** | argmax **ReT** | argmax **fill** |
|---|---:|---:|---:|
| `R1r\|base` | **1008** | **1008** | **1008** |
| `R2r\|base` | **1008** | **1008** | **1008** |
| los otros cuatro | 1344 | 1344 | 1344 |

**Las tres métricas coinciden en las seis celdas**, incluido `flow_fill_rate`, **que no tiene
término de coste alguno**. Así que el óptimo interior en 1.008 h **no lo produce el precio del
inventario** — es **físico**. Más buffer del necesario **estorba** por sí solo.

**G1 queda refutada como mecanismo.** Pero fíjate en lo que sí queda.

## 2. Y sin embargo: **hay curvatura, y es la primera que medimos**

    no linealidad (1 − R² de un ajuste lineal al perfil):
        Cobb-Douglas   0,1515
        ret_excel      0,0790

**El perfil del buffer NO es lineal**, y su óptimo es **interior en dos regímenes**. Contra el
`R² = 0,9697` del panel `ρ → ReT`, esto es la primera superficie del proyecto con curvatura
apreciable.

**Y la lectura correcta es incómoda para mi propia hipótesis:** buscaba curvatura *creada por la
métrica* y encontré curvatura **que ya estaba en la física** y que nuestro barrido de sensibilidad
anterior no vio porque **barría otra cosa** (`S_T ≈ 0,006` era sobre el buffer como factor Morris
global, no sobre su perfil a 9 niveles).

`H_regime` sobre CD sigue siendo **0,000252** — dos órdenes bajo la barra. Hay **curvatura sin
dependencia del régimen**: exactamente el segundo desenlace que el preregistro contempló.

## 3. `f8` — la regla de exponentes de Garrido **se rompe aquí**

Añadí `f8` a mitad de camino, y falla:

    exponentes derivados con SU regla 0,20/ln(x_max):
        tau        0,79092   <-- captura el 79 % del presupuesto
        kappa_dot  0,05014
        phi        0,02947
        epsilon    0,01686
        zeta       0,01551

La regla `0,20/ln(x_max)` **presupone `x_max ≫ 1`**. Cuando el máximo de una componente se acerca
a 1, `ln(x) → 0` y **su exponente explota**. Aquí `τ` se lleva **79 %** del índice.

> **El «Cobb-Douglas» de este barrido es τ disfrazado.** Cualquier conclusión sacada de él es
> sobre τ, no sobre un índice de cinco componentes.

Esto **ya estaba anotado** como ill-conditioning de τ en la auditoría de métricas del 29 de julio
(«sensibilidad relativa 5,4× frente a 0,07× de ζ»), pero **no como una condición bajo la cual la
regla de Eq. (5) deja de ser aplicable**. Ahora lo está, y es un resultado sobre **su** índice:
publicable, concreto y falsable.

**No lo «arreglo» poniendo un suelo a los máximos.** Ajustar el instrumento después de ver el
resultado es exactamente lo que el preregistro existe para impedir. Se declara y las conclusiones
CD de este barrido quedan **no utilizables**.

## 4. Corrección de un falsador mío — sexta de la semana

`f1` predecía que **`κ` sube con el buffer**. **Baja**: 427,0k → 400,6k. `κ` tiene siete términos
de coste y **el ahorro en backorders supera al coste de mantener**.

La pregunta correcta no es el signo del total sino **si el término de mantener hace trabajo**, y
el defecto inyectado lo contesta: con `c_i = 0` la respuesta al buffer **se triplica**
(26,4k → 83,2k), porque desaparece el único término que empuja en contra. **Cobb-Douglas SÍ cobra
el inventario** — sólo que no lo bastante como para invertir el signo.

## 5. Qué queda, y qué haría ahora

| | |
|---|---|
| **G1 como mecanismo** | **refutado** — la curvatura no viene de la métrica |
| **curvatura en el perfil de buffer** | **confirmada** (0,079 en ReT, sin depender de τ) |
| **dependencia del régimen** | **ausente** — `H_regime = 2,5e-04` |
| **índice CD en este barrido** | **no utilizable** — `f8`, τ con el 79 % |

**El siguiente paso ya no es G2.** Hay una superficie con curvatura medida y **sin** dependencia
del régimen, y eso es exactamente el caso donde el preregistro dice: *una red podría **predecir**
mejor aunque no haya política que aprender*. Es la pregunta Q1 de Garrido sobre una superficie
nueva, y es barata: **medir la prima de predicción (MLP/KAN vs lineal, SESOI 0,05) sobre el perfil
de buffer**, que es donde por primera vez hay algo no lineal que predecir.

Si ahí tampoco hay prima, el argumento del paper se vuelve mucho más fuerte: **ni siquiera con
curvatura medida aparece la prima neural.**
