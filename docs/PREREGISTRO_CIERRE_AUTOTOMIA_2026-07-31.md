# Preregistro — cerrar `Re(APj)` con el brazo de olas de flete

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Se ejecuta con `supply_chain/arm_runner.py`
contra `fidelity_reference_v4`.

> **Corrección al propio contrato, antes de correr nada (`46d16ba` → esta versión).** Escribí
> «la tolerancia es la que ya está embarcada (0,05 h)» y **es falso**: el default de
> `autotomy_tolerance_hours` es **0,0** (`supply_chain.py:243`), con lo cual `band` con el
> default sería idéntico a `le` y el brazo `FDB` no existiría como tal. La banda de 0,05 h se
> **lee de sus datos** —sus 96 filas de autotomía caen en `CTj − LT ∈ [0,0074, 0,048]`— y por
> tanto **es un parámetro, uno**, calibrado contra su ledger y **no** contra nuestra salida.
> La línea «parámetros libres: ninguno» de §6 queda **retractada**: es **uno, declarado y
> trazable a su banda observada**. Ninguna corrida se ha ejecutado contra la versión anterior.

## 1. Por qué, y qué ya sabemos antes de correr

`Re(APj)` es el único driver de la Fig. 4 de Garrido que **no existe** en nuestra tabla de 90
configuraciones (`results/garrido_drivers_per_configuration/`, sello `491694175a3975a7…`):
la constante de cumplimiento de 54 h contra `LT = 48` hace inalcanzable la rama de autotomía.

**El brazo de olas de flete ya está medido, y por sí solo no la cierra.**
`results/metric_audit/delay_physical_arms_v1/` (detenido por su falsador de dispersión) mide:

| | `autotomy_share` R1r | R2r | `ret_mean` R1r |
|---|---:|---:|---:|
| **A** constante 54 (statu quo) | 0,0000 | 0,0000 | 0,0069 |
| **F** olas de flete | **0,6370** | **0,1322** | **0,5020** |
| **Garrido** (v4) | 0,004334 | 0,000637 | 0,006282 |

No es un cierre: es pasar de un suelo del 0% a un techo del **64%**, **147× por encima** de él,
arrastrando `ret_mean` de 0,0069 a 0,5020 contra una referencia de 0,0063.

**El mecanismo está identificado.** Nuestra rejilla de olas deja el **60,7%** de las órdenes en
`CTj = 48,0` exactamente, y el predicado embarcado es `CTj ≤ LTj`, así que todas entran. En sus
libros el suelo es **raro**: 98 filas en la banda del suelo, de las cuales 96 con autotomía,
sobre ~26.000 filas de R1r — y su mínimo es **48,0074**, estrictamente por encima de `LT`.

Es decir: **el problema no es el desfase del suelo, es su incidencia.**

## 2. Brazos

| | predicado `le` | predicado `band` (tol 0,05 h) |
|---|---|---|
| constante 54 | **A** (statu quo) | — |
| olas de flete | **F** | — |
| olas + `δ ~ U(0, 8)` | **FD** | **FDB** |

`δ` entra porque es lo único que puede volver **raro** el suelo: con `CTj = 48 + δ` y
`δ ~ U(0,8)`, la fracción dentro de una banda de 0,05 h es `0,05/8 = 0,625%`. Ese cálculo es
aritmética declarada: el soporte de `δ` es `HOURS_PER_SHIFT` con `S = 1` y la tolerancia sale
de la banda en la que caen **sus** filas de autotomía.

## 3. Predicción, declarada antes de correr

1. **`A`**: `autotomy_share` = 0 exactamente, `min CTj` = 54,0. Es el bloque congelado.
2. **`F`**: ≈ 0,637 / 0,132 — ya medido, se re-mide solo para que esté en el mismo artefacto.
3. **`FD`**: **vuelve a 0**. Con `δ > 0` casi seguro, `CTj = 48 + δ > LT`, y el predicado `le`
   no dispara. Añadir `δ` sin cambiar el predicado **no cierra nada**.
4. **`FDB`**: **≈ 0,625%** contra su **0,443%** — el único brazo que puede acercarse.
5. **`ret_mean` en `FDB`**: **sin dirección declarada**. La autotomía aporta `APj/LT ≈ 1` a un
   0,6% de las órdenes (+0,006) pero cambia qué órdenes cruzan `LT`. No lo sé.
6. **`rpj_p95` empeora** en todos los brazos con `δ`: `RPj ≈ CTj` crece sin que `k` cambie.

**Predigo que `FDB` cierra `autotomy_share` y NO se adopta igualmente**, porque `ret_mean` o
`rpj_p95` se degradarán más de `EPSILON`. Lo declaro por adelantado para que adoptarlo, si
ocurre, sea informativo.

## 4. Falsadores — cada uno con su modo de fallo

| # | qué | puede fallar porque |
|---|---|---|
| f1 | `A` da `autotomy_share` = 0 y `min CTj` = 54,0 | si el default se movió, todo lo demás se compara contra otra cosa |
| f2 | **el suelo es modal en `F` y raro en él** — medido en AMBOS lados con la misma regla | si su suelo también fuera modal, mi diagnóstico del mecanismo es falso |
| f3 | ninguna orden con `CTj < LT` en ningún brazo | el sorteo de `δ` podría restar en vez de sumar |
| f4 | en `FDB`, toda orden con autotomía cumple `CTj − LT ≤ tol` | el predicado de banda podría estar leyendo otra cosa |
| f5 | el conjunto no dominado es `epsilon`-estable en la banda enmendada | — |

## 5. Aceptación

**Conjunto no dominado** sobre los **seis** momentos contra `fidelity_reference_v4`,
`EPSILON = 0,5`, barrido en la banda enmendada, ambas familias, `sum_dk` **vetado** para
rankear.

`scored_orders_per_year` **entra** al scoring: la exclusión de la enmienda §2 era «hasta que
una referencia v4 arregle el denominador», y v4 existe y `arm_runner` ya usa su misma
convención de ventana desde hoy.

Un brazo se adopta si y solo si: mejora `d_k(autotomy_share)`; `d_k(ret_mean)` no empeora más
de `EPSILON` en ninguna familia; ningún otro momento empeora más allá de `EPSILON`; f1–f5 pasan;
y el conjunto es estable.

**Si ningún brazo califica, el resultado es que la autotomía de Garrido no se puede reproducir
sin degradar su propia métrica** — y eso es un hallazgo, no un fracaso.

## 6. Declarado por adelantado

| ítem | valor |
|---|---|
| brazos | A, F, FD, FDB |
| raíces | **3.700.001–3.700.012**, vírgenes |
| referencia | `fidelity_reference_v4`, sello `32e23a79b43f76a7…` |
| momentos puntuados | **6** |
| tolerancia de banda | **0,05 h, leída de su banda observada** `[0,0074, 0,048]` (el default embarcado es 0,0) |
| parámetros libres | **uno**: la tolerancia de banda, calibrada contra SU ledger, nunca contra nuestra salida |
| predicción | §3, incluida la de **no-adopción** |
