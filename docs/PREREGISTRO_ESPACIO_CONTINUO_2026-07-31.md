# Preregistro — ¿la adecuación lineal es de su métrica o de su rejilla?

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Se corre **una vez**.

## 1. La pregunta, y por qué ahora

Sobre las 90 configuraciones de Garrido medimos que un modelo **lineal** explica el **0,970** de
su propia ReT, y que en su pregunta de activación **ninguna red supera a la logística**
(`docs/RESULTADO_FIG5_BACKPROP_VS_KAN_2026-07-31.md`). Concluimos que «el reconocimiento de
patrones no tiene qué reconocer».

**Esa conclusión está confinada a su rejilla**: dos variables de decisión, seis niveles de
buffer y tres de turno, todos monótonos. Es exactamente el espacio que Garrido pidió ampliar en
la reunión del 28 de julio: *añadir nodos y variables de decisión, aguas arriba y aguas abajo*,
y **preferir variables continuas** —«más gradiente, más fácil hallar el óptimo»—, con la métrica
ReT de 2017 como recompensa.

Así que la pregunta es separable y se puede contestar con la tesis y el simulador:

> **¿El 0,970 lineal es una propiedad de su MÉTRICA, o un artefacto de su REJILLA?**

Si es de la rejilla, al liberar el espacio la superficie se vuelve no lineal y una red gana. Si
es de la métrica, la linealidad sobrevive al espacio continuo — y **ése es un hallazgo más
fuerte que el actual**, porque ya no se puede atribuir al diseño.

## 2. El espacio, ampliado dentro de sus propios límites

| variable | Garrido | aquí |
|---|---|---|
| periodo de reposición | 6 niveles `{0,168,336,504,672,1344}` h | **continuo** en `[0, 1344]` h |
| `op3_rm` | atado al nivel (Tabla 6.16) | **continuo** en `[0, 122.880]` |
| `op5_rm` | atado al nivel | **continuo** en `[0, 122.880]` |
| `op9_rations` | atado al nivel | **continuo** en `[0, 126.000]` |
| turnos | 3 niveles | 3 niveles (**sin cambio**: es entero en la física) |

**Nada sale de sus rangos.** Los topes son los máximos de su Tabla 6.16 y su periodo máximo. Lo
que se rompe es el **acoplamiento**: él ata las tres cantidades a un único índice, y aquí varían
por separado. Eso es literalmente «añadir variables de decisión» sin inventar física nueva.

Familia de riesgo y patrón se muestrean también, para que la comparación cubra los tres
regímenes y no un corte.

## 3. Diseño

* **384 configuraciones**, muestreo Sobol sobre `[0,1]^4` mapeado a los rangos de arriba, más
  familia (3) y patrón (flags) uniformes, más turnos uniformes en `{1,2,3}`.
* **Una raíz por configuración**, `4.100.001–4.100.384`, vírgenes.
* **Horizonte 52 semanas**, declarado: es más corto que sus 10–20 años, y por eso este
  resultado **no** se compara con sus valores absolutos, solo consigo mismo.
* Métrica: `ret_excel`, la canónica.

## 4. Modelos y evaluación — idénticos a la corrida sobre su rejilla

Lineal / logística contra **backprop** y **KAN**, misma arquitectura, mismo optimizador, misma
CV **agrupada por semilla**, mismas dos tareas:

* **B1** regresión `(ρ, familia, patrón) → ReT`, R²
* **B2** su pregunta de activación, `¿ReT(x) > ReT(x−1)?`, exactitud

Nada se re-sintoniza. Si algo cambia respecto a la corrida anterior, es el **espacio**, no el
método.

## 5. Regla de aceptación, declarada antes de correr

Una red «gana» si y solo si **supera al lineal por más de una SD entre pliegues del lineal**, la
misma barra que usamos sobre su rejilla. Sin excepciones y sin barra alternativa.

**Prohibido**: re-muestrear si el resultado no gusta, ampliar `n` después de ver el resultado,
cambiar arquitectura, o cambiar la regla. Si esta corrida no da una ganancia, **el resultado es
que no la hay** y se reporta así.

## 6. Predicción

1. **B1 lineal baja** del 0,970 medido en su rejilla: al desacoplar las tres cantidades hay
   interacciones que una recta no captura. **Predigo `R²` lineal entre 0,80 y 0,95.**
2. **Las redes ganan en B1** por más de una SD. Es la predicción principal.
3. **B2 es dudoso.** La pregunta de activación depende de diferencias pequeñas entre
   configuraciones vecinas; con muestreo aleatorio los pares son menos informativos que en su
   diseño ordenado. **No declaro dirección.**
4. Si el lineal **se mantiene por encima de 0,95**, la conclusión anterior se refuerza: la
   linealidad es de la métrica, no de la rejilla.

Las cuatro se registran antes de mirar. La 4 es la que preferiría que fuera falsa, y por eso va
escrita.

## 7. Falsadores

| # | qué | puede fallar porque |
|---|---|---|
| f1 | el muestreo cubre el espacio: cada variable continua con ≥ 100 valores distintos | un mapeo roto colapsaría el diseño a la rejilla original |
| f2 | ninguna configuración excede los rangos de su Tabla 6.16 | inventar física fuera de sus límites invalidaría la comparación |
| f3 | sin fuga entre pliegues (agrupado por semilla) | — |
| f4 | las dos redes entrenan (pérdida final < inicial en cada pliegue) | una red muerta puntuaría como la línea base en silencio |
| f5 | la línea base no es degenerada: `sd(ReT) > 0` y la mayoritaria de B2 < 0,95 | — |
