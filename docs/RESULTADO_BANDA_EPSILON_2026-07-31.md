# Re-corrida bajo la banda enmendada: nada se rescata, y eso es lo importante

**Status:** `RE_ADJUDICATED_NO_VERDICT_CHANGES`. Las tres corridas re-ejecutadas **completas**,
sin escoger familia ni momento, per §5.1 de la enmienda.

## 1. Resultado por corrida

| corrida | `epsilon` estable antes | **ahora** | ¿cambia el veredicto? |
|---|---|---|---|
| `oatj_material_arms` | FALLA | **PASA** (0 volteos en ambas familias) | **no** — sigue detenida por `R1`, `R2` y `P3.1` |
| `link_x_attribution` | FALLA | **R1r PASA** (0 volteos), R2r FALLA | **no** |
| `delta_assumption_arms` | FALLA | FALLA | **no** |

**Ningún brazo pasa a ser adoptable. Ningún veredicto cambia.**

## 2. La tabla de volteos, que es lo que la enmienda añadió

**`delta_assumption` — volteos a `eps = 0,625`:**

| par | momento crítico | brecha `d_k` |
|---|---|---:|
| `D` empieza a dominar a `L` (R1r) | `ret_mean` | **+0,574** |
| `LD` empieza a dominar a `A` (R2r) | `ret_mean` | **+0,583** |

**`link_x_attribution` R2r — volteos a `eps = 0,5`:**

| par | momento crítico | brecha `d_k` |
|---|---|---:|
| `A` deja de dominar a `C` | `ret_mean` | **+0,059** |
| `L` empieza a dominar a `A` | `ret_mean` | **+0,381** |
| `L` empieza a dominar a `C` | `ret_mean` | **+0,440** |

**Los cinco volteos los manda `ret_mean`**, con brechas de **0,059 a 0,583** — todas dentro
de la banda ±50%. Eso es información que el booleano no daba: la fragilidad **no está
repartida**, está concentrada en un solo momento y en diferencias de medio error estándar
combinado.

## 3. Lo que esto dice de la enmienda

**La enmienda no rescató nada**, y eso es la mejor evidencia de que no fue interesada. La
propuse después de que la regla bloqueara tres corridas; si hubiera estado ajustando la banda
para que pasaran, habrían pasado. Pasó una, por 0 volteos, y esa sigue detenida por otros
tres falsadores.

**Y las que siguen inestables lo están por una razón sustantiva**, no por el rango: con la
banda vieja los volteos se disparaban a `eps` 1,0–2,0, fuera de todo régimen razonable; con la
nueva ocurren a **0,5–0,625**, es decir **justo alrededor del `EPSILON` declarado**. Eso ya no
es un artefacto de barrer demasiado ancho: es que `ret_mean` separa a esos brazos por menos de
lo que el `EPSILON` declarado considera indiferente.

Per §5.2 de la enmienda, **el veredicto original de las dos inestables se mantiene y la
inestabilidad es un hallazgo real**.

## 4. Limitación conocida del reporte

El «momento crítico» que registra `dominance_flips` es el **argmax de la brecha**, que no
siempre es el momento que *causa* el volteo — cuando un brazo es débilmente mejor en todo, el
argmax sale con brecha `+0,000` aunque lo que falle sea el término `strictly`. Aparece así en
dos filas de `delta_assumption`. **Se anota; no se arregla ahora**, porque cambiar el reporte
en la misma corrida en que se estrena la enmienda mezclaría dos cosas.

## 5. Estado

Nada adoptado, ningún default movido, los tres sellos de contrato verifican. La exclusión de
momentos con brecha cero de `strictly` está implementada y **no cambió ningún conjunto** en
estas tres corridas — su efecto es evitar volteos artificiales, y aquí no había ninguno que
dependiera solo de eso.
