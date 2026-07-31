# Preregistro — mezclar familias de riesgo y medir el headroom aguas abajo

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Una sola corrida.

## 1. Por qué exactamente esto

El mapa de sensibilidad (`results/sensitivity/headroom_map_v1/`, sello `6a16e3263e1ccaf6…`)
dejó dos hechos que apuntan al mismo experimento:

* **cuál familia de riesgo está activa es el 75,5% de la varianza** — más que los otros
  diecinueve factores juntos — y **Garrido corre una familia por vez**. Mezclarlas no existe ni
  en su diseño ni en el nuestro, y está dentro de los permisos que nos dio.
* **toda la interacción que existe está aguas abajo**: `op12_q_max` (0,061), `op10_q_max`
  (0,042), `op9_rop` (0,039). Aguas arriba y los buffers son inertes (`S_T ≈ 0,006`).

La hipótesis, entonces: **mezclar familias crea el acoplamiento decisión × riesgo que una sola
familia no tiene, y ese acoplamiento aparece en el despacho aguas abajo.**

## 2. Qué se mide: `H_regime`, y por qué es la cantidad correcta

No se mide «cuánto sube ReT». Se mide **cuánto vale conocer el régimen**:

    H_regime = media_r [ max_a ReT(a, r) ]  −  max_a media_r [ ReT(a, r) ]

es decir, **el mejor ajuste sabiendo el régimen, menos el mejor ajuste único que debe servir a
todos**. Es un **techo** para cualquier política que condicione en el régimen: ninguna política
adaptativa puede superarlo, y si es cero, no hay nada que aprender por mucha red que se ponga.

Es la versión barata y honesta del headroom de información perfecta que este proyecto ya usa, y
no requiere entrenar nada.

**Regla:** `H_regime` cuenta solo si su **LCB95 por bootstrap sobre semillas es > 0**. Un
headroom que no supera su propio ruido no es headroom.

## 3. Diseño

* **Regímenes (7):** `R1r`, `R2r`, `R3` solos, y las mezclas **`R1r+R2r`**, **`R1r+R3`**,
  **`R2r+R3`**, **`R1r+R2r+R3`**.
* **Rejilla aguas abajo (5³ = 125):** `op9_rop` ∈ [12, 48], `op10_q_max` ∈ [1.200, 5.200],
  `op12_q_max` ∈ [1.200, 5.200], 5 niveles cada uno. Los tres factores con interacción del mapa.
* **5 semillas** por celda, **CRN**: la misma semilla en todos los ajustes y regímenes, para que
  la comparación sea pareada.
* **4.375 corridas**, 52 semanas, semillas **4.300.001+**, vírgenes. ~6 min medidos.
* Métrica `ret_excel`. Todo lo demás en su valor por defecto.

`H_regime` se calcula **dos veces**: sobre los **3 regímenes puros** y sobre los **7 con
mezclas**. La diferencia entre ambos es la respuesta.

## 4. Predicción, antes de mirar

1. **Mezclar sube la exposición** — ReT medio de las mezclas por debajo del de las familias
   puras. Trivial, se declara para descartar que la mezcla no haga nada.
2. **`H_regime` puro será pequeño**, `< 0,01`, replicando la invariancia ya medida.
3. **`H_regime` con mezclas será mayor que el puro** — la predicción principal, y la que
   quiero que sea cierta.
4. **Pero probablemente siga sin superar su LCB95**, porque el mapa dijo que el riesgo mueve el
   **suelo** y no la **pendiente** (interacción de `risk_family_selector`: 0,019). Lo escribo
   para que si sí lo supera, sea informativo.

Si 3 y 4 se cumplen a la vez —mayor pero no significativo— el resultado es que **la mezcla
apunta en la dirección correcta y hace falta más señal**, no que no haya nada.

## 5. Falsadores

| # | qué | puede fallar porque |
|---|---|---|
| f1 | las mezclas se distinguen de las puras: ReT medio de al menos una mezcla fuera del rango de las tres puras | si mezclar no cambia nada medible, el experimento no tiene objeto |
| f2 | el óptimo **no** está en el borde de la rejilla en al menos un régimen | un óptimo pegado al borde haría de `H_regime` un artefacto del rango |
| f3 | `H_regime ≥ −1e-12` (identidad max-mean ≥ mean-max al revés) | un signo invertido delataría un error en la agregación |
| f4 | CRN real: la misma semilla da el mismo ReT para el mismo ajuste y régimen | sin pareo, `H_regime` es ruido entre semillas |
| f5 | las siete listas de riesgos coinciden con las familias de la tesis | mezclar mal invalidaría la interpretación |

## 6. Prohibido

Ampliar la rejilla, añadir semillas o cambiar la regla del LCB95 después de ver el resultado.
