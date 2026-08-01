# Resultado — **sin prima neural ni con curvatura medida**, y la razón es cuantitativa

**Artefacto:** `results/headroom/buffer_prediction_premium/result.json` (sello
`ae1a545bb4274e75…`, `NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE`) · **los seis falsadores
PASAN** · 1.530 episodios, 17 niveles × 3 familias × 3 escaladas × 10 semillas · preregistro
`docs/PREREGISTRO_PRIMA_PREDICCION_BUFFER_2026-08-01.md`, commiteado antes de correr.

## 1. El resultado

`R²` held-out, validación cruzada **agrupada por semilla**:

| modelo | `R²` | vs lineal | IC95 | ¿supera el SESOI 0,05? |
|---|---:|---:|---|---|
| constante | −0,0034 | | | |
| **lineal** | **0,6826** | — | — | — |
| backprop (MLP 16-16) | 0,5548 | **−0,1278** | [−0,3157, +0,0601] | **no — es PEOR** |
| KAN | 0,7163 | +0,0337 | [−0,0787, +0,1462] | **no** |

**Y la premisa se cumplía:** `f1` recalculó la curvatura **aquí**, sin fiarse de G1, y da
**0,0763** (por celda: 0,059 a 0,111). La superficie **no es lineal**, y aun así el lineal gana.

**El MLP es peor que el lineal.** Con capacidad de sobra y 1.530 filas, la red **pierde 0,128 de
`R²`** frente a una recta. KAN saca +0,034, por debajo del umbral y con el intervalo cruzando el
cero.

## 2. Por qué — y esto es lo publicable

La respuesta no es «la superficie era fácil». Mira el otro número:

    R² del lineal        0,6826
    varianza SIN explicar 0,3174     <-- ruido a nivel de episodio
    curvatura del perfil  0,0763     <-- la no linealidad disponible

> **La no linealidad existe pero está POR DEBAJO DEL SUELO DE RUIDO.** Hay un 7,6 % de estructura
> curva que capturar y un 31,7 % de varianza estocástica que la tapa. Una red con capacidad extra
> ajusta ruido antes que curvatura — que es exactamente lo que le pasa al MLP, cuyos folds van de
> 0,26 a 0,74 mientras el lineal se mantiene entre 0,62 y 0,73.

Eso convierte «no hay prima» en **una condición cuantitativa**:

> **Una prima neural exige que la curvatura de la superficie supere el ruido que la oculta.**
> Aquí `0,076 ≪ 0,317`, y por eso la capacidad extra no se paga.

Es una afirmación falsable, general, y **le dice a Garrido qué haría falta** para que su propuesta
aplique: no «más red», sino **más señal curva o menos ruido por episodio**.

## 3. Lo que esto cierra

La réplica evidente a Q1 era: *«el lineal ganó porque la superficie `ρ → ReT` era casi lineal
(`R² = 0,9697`); en una curva la red ganaría»*. **Probado sobre una superficie curva. No gana.**

| superficie | curvatura | lineal | mejor red | Δ |
|---|---:|---:|---:|---:|
| panel `ρ → ReT` (Q1) | ~0 | 0,9697 | KAN 0,9913 | +0,0216 |
| **perfil de buffer** (ésta) | **0,0763** | 0,6826 | KAN 0,7163 | **+0,0337** |

**En ninguna de las dos se alcanza el SESOI de 0,05**, y en la curva **el MLP directamente
empeora**. La respuesta a Q1 deja de ser provisional.

## 4. Lo que NO afirma

* **No dice que las redes no sirvan para SCRES.** Dice que **en este entorno**, con este nivel de
  ruido por episodio, no se pagan. La condición cuantitativa de §2 es precisamente lo que habría
  que romper.
* **No es un resultado sobre control.** Esto es predicción. El control exige además el gate de
  headroom, que sigue cerrado.
* **El ruido puede reducirse promediando semillas**, y eso subiría el `R²` de todos los modelos.
  Lo que la prima necesita no es un `R²` alto, sino que **la brecha** entre curvo y recto supere
  el SESOI. Promediar sube ambos.

## 5. Consecuencia para el programa de generadores

G1 se propuso crear curvatura y **la encontró donde no la buscaba** (física, no de la métrica).
Ahora sabemos que **la curvatura por sí sola no basta**: hace falta curvatura **por encima del
ruido**.

Eso reordena los generadores que quedan. **G2 (umbral de autotomía)** pasa a ser el más
prometedor de los tres restantes, porque un **umbral duro** en `CTj = 48` es una discontinuidad
—no una curvatura suave— y una discontinuidad produce señal que **no se promedia con el ruido**:
o el pedido cruza el umbral o no. **G3** (tres reclamantes) y **G4** (observabilidad parcial)
quedan detrás.
