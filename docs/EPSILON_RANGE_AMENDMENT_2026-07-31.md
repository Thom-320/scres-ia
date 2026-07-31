# Enmienda — el rango barrido de `epsilon`

**Estado:** `AMENDMENT_PENDING_PI_SIGNATURE`. Artefacto
`contracts/epsilon_range_amendment_2026-07-31.json`, sellado.

## 0. Conflicto de interés, declarado primero

**Estoy enmendando una regla después de que bloqueara tres corridas que habría preferido
reportar.** Ése es exactamente el conflicto que este proceso existe para vigilar.

Por eso: la justificación de §2 se hace sobre la **escala del estadístico** y no sobre ningún
veredicto bloqueado, y la §5 fija **por adelantado** qué significa que pase y qué que falle —
de modo que pasar no sea automáticamente una victoria.

## 1. Qué NO se enmienda

* `EPSILON` declarado sigue en **0,5**.
* La regla de que **un conjunto que se mueve se reporta inestable** sigue.
* El **conjunto no dominado** sigue siendo la salida, y `sum_dk` sigue vetado para rankear.

Se enmienda **solo** el rango barrido, la definición de estricticidad y la forma de reportar.

## 2. El rango, justificado por escala y no por resultado

| | rango | envergadura |
|---|---|---|
| antes | `0,25 · 0,5 · 1,0 · 2,0` | **8×** (−50%/+300%) |
| **ahora** | `0,25 · 0,375 · 0,5 · 0,625 · 0,75` | **3×** (±50%) |

`EPSILON` se mide en **errores estándar combinados**. Un `epsilon` de 2,0 declara
indiferencia ante una diferencia de **dos SE combinados** — más de lo que separa a cualquier
par de brazos que este proyecto compare. En ese extremo `no_worse` se cumple trivialmente y
`strictly` es insatisfacible, así que **el chequeo degenera con independencia de los datos**.

La banda queda en el `EPSILON` declarado **±50%**, que abarca de medio a tres cuartos de un
SE combinado y **se mantiene dentro del régimen donde ambos términos pueden morder**.

## 3. Los momentos de brecha cero salen de `strictly`

`strictly` exige `a[k] < b[k] − epsilon`. Un momento **idéntico** entre dos brazos no puede
satisfacerlo con ningún `epsilon` positivo, pero se cuenta entre los candidatos — así que
**solo puede quitar estricticidad, nunca aportarla**. Medido: un volteo mandado por una
brecha de `autotomy_share` de **exactamente +0,00**.

Se excluyen **solo de `strictly`**. `no_worse` debe seguir viendo todos los momentos vivos.

## 4. Se reporta la tabla de volteos, no un booleano

Además del booleano, para cada par cuya dominancia cambia dentro de la banda se registra: el
par, el `epsilon` donde voltea, el **momento crítico** y la **brecha `d_k`**.

El booleano dice «no mires». La tabla dice **qué comparación es frágil y por cuánto**, que es
la información que la regla pretende proteger.

## 5. Regla de adjudicación, declarada antes de re-correr

1. **Esta enmienda NO re-adjudica las tres corridas bloqueadas.** Re-correrlas bajo la banda
   nueva es un acto aparte, y cada una debe re-correrse **completa** — prohibido escoger la
   familia o el momento que ahora pasa.
2. **Si siguen inestables**, el veredicto original se mantiene y la inestabilidad es un
   hallazgo real.
3. **Si ahora son estables**, el conjunto **puede reportarse**, pero adoptar sigue exigiendo
   todos los demás conjuntos de la regla de aceptación. **La estabilidad es una compuerta,
   nunca una razón para adoptar.**
4. **Prohibido** ensanchar la banda otra vez si la nueva también bloquea algo. Cualquier
   cambio posterior necesita su propia enmienda y un motivo que **no sea el veredicto que
   produce**.

## 6. Alcance

**Nada se reetiqueta.** Ninguna cifra congelada se modifica. Los conjuntos no dominados ya
reportados en `EPSILON = 0,5` siguen siendo los que son; lo único que cambia es con qué banda
se juzga su robustez.

## 7. Firma

Requiere aprobación del PI **antes** de volver a mirar ningún veredicto bajo la banda nueva.
Ése es el punto entero de la §0.
