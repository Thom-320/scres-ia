# Resultado — la Fig. 5 de Garrido, con backprop contra KAN

**Artefacto:** `results/garrido_fig5_surrogate/result.json` (sello `40bf64852c6e6f44…`) ·
**Runner:** `scripts/build_garrido_fig5_surrogate.py` · **Datos:** la tabla de drivers de las 90
configuraciones (sello `491694175a3975a7…`) · **Los cinco falsadores pasan.**

## 1. La figura, tal como está dibujada, no tiene nada que aprender

Su Fig. 5 pone los cuatro drivers como dendritas y SCRES como axón. En nuestra descomposición
**ReT es exactamente la suma de las contribuciones de los drivers**, así que ese mapeo es una
**identidad**:

    R² = 1,000000000000     error máximo de la identidad < 1e-12
    coeficientes identificados: Re(RPj) = 1,0000   Re(FRt) = 1,0000

Los otros tres —`Re(APj)`, `Re(DPj)` y nuestro quinto término— son columnas **idénticamente
cero**, así que no tienen coeficiente que estimar.

`IDENTITY_NOT_A_LEARNING_TASK`. Un R² perfecto aquí **no es evidencia de nada**, y lo
registramos precisamente para que nadie lo presente como si lo fuera. Es un hallazgo sobre la
figura: **con los drivers como entradas, la neurona ya tiene la respuesta**.

La versión que sí se puede aprender es la otra: **predecir SCRES desde `ρ` y el diseño de
riesgo**, sin los drivers. Eso es lo que un lazo cerrado tendría que resolver de verdad.

## 2. Los dos experimentos que sí lo son

CV **agrupada por semilla** —su diseño reutiliza una semilla en cada tripleta
`Cf_b / Cf_b+30 / Cf_b+60`, y una partición al azar pondría la misma trayectoria a ambos lados—
y contra **líneas base**, porque con 90 filas y 9 variables «la red ajusta» no es evidencia:
solo lo es «la red supera a la regla lineal».

### B1 — regresión `(ρ, familia, patrón) → ReT`, R²

| modelo | R² | ± |
|---|---:|---:|
| constante | 0,0000 | 0,0000 |
| **lineal** | **0,9697** | 0,0165 |
| backprop | 0,9863 | 0,0082 |
| **KAN** | **0,9913** | 0,0074 |

Ambas redes superan al lineal por más de una SD entre pliegues. **Pero el titular es la línea
base:** las variables de decisión de Garrido explican el **97% de su ReT con un modelo lineal**.
Las redes se llevan la mitad del 3% restante.

### B2 — su pregunta de activación: *«¿ReT en `x` > ReT en `x−1`?»*, exactitud

| modelo | acierto | ± |
|---|---:|---:|
| mayoritaria | 0,3333 | 0,1039 |
| **lineal (logística)** | **0,7111** | 0,0824 |
| backprop | 0,7178 | 0,1353 |
| KAN | 0,7711 | 0,0925 |

**Ninguna red supera a la logística** por más de una SD de la propia línea base. En la pregunta
que su figura formula, **la neurona no aporta sobre una regla lineal**.

## 3. La respuesta a su pregunta 2, medida

Su paper pregunta *«¿cómo puede integrarse esta familia de IA en la estructura interna de un
modelo DES para evaluar SCRES?»*. Sobre su propio diseño de 90 celdas, la respuesta medida es:

* **como está dibujada, no se integra** — el mapeo drivers → SCRES es una identidad;
* **reformulada como `ρ → SCRES`, se integra pero apenas paga**: el lineal ya da 0,97, y la
  ganancia neural es de centésimas;
* **en su pregunta de activación, no paga en absoluto**.

Eso **no refuta** su tesis del «efecto Alzheimer». Dice algo más preciso: en el régimen de su
propio experimento —90 configuraciones, dos variables de decisión, tres familias de riesgo— la
relación entre `ρ` y SCRES es **suficientemente lineal como para que el reconocimiento de
patrones no tenga qué reconocer**. Si el aprendizaje entre corridas ha de aportar, tendrá que
ser en un espacio de decisión más rico que el que su diseño explora.

## 4. KAN contra backprop

KAN queda por delante en ambas tareas (0,9913 vs 0,9863; 0,7711 vs 0,7178). **No lo declaro
como diferencia**: con cinco pliegues y SDs de 0,07–0,14 en B2, las dos redes no se separan. Lo
que sí se separa —y es lo que importa— es **red contra lineal**, y ahí el veredicto es el de §2.

## 5. Lo que un falsador me corrigió, otra vez

`f1` exigía que **los cinco** coeficientes de la identidad valieran 1, y falló. La identidad
estaba bien: fallaban tres columnas **idénticamente cero** —`Re(APj)` inalcanzable, `Re(DPj)`
cero por su Eq. 5.3, y nuestro quinto término nunca disparado— que **no tienen coeficiente que
estimar**. Ahora la comprobación mira la identidad directamente
(`max |Σ contribuciones − ReT| < 1e-12`) y los coeficientes **solo donde están identificados**.

Es la tercera vez hoy que un falsador atrapa mi propia especificación en vez de la ciencia. Va
en el artefacto, no en una nota al pie.

## 6. Estado y límites

`DEVELOPMENT_FIG5_SURROGATE`. Nada adoptado, ninguna política entrenada, cero PPO.

* **90 filas, 30 grupos.** Es su diseño completo, y sigue siendo poco. Todo intervalo aquí es
  ancho y lo digo antes de que alguien cite una media.
* **`Re(APj) ≡ 0` y `Re(DPj) ≡ 0`** en la tabla de entrada: hoy la neurona ve **dos drivers
  vivos**, y en R1r **uno solo**. Cerrar `Re(APj)` es posible pero degrada `ret_mean`
  (`docs/RESULTADO_CIERRE_AUTOTOMIA_2026-07-31.md`), así que este límite es una **frontera
  medida**, no un descuido.
* La comparación es **red contra lineal**, no red contra red.
