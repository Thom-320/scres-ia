# Preregistro — el techo de `H_regime` bajo reescalada monótona del índice Cobb-Douglas

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_monotone_transform_ceiling_v1.py`.
Semillas: bloque quemado `5.300.001–03`, réplica declarada. **Ninguna nueva.**

## 1. La pregunta, y de quién es

El PI propone **afinar la Cobb-Douglas de modo que crezca junto con la Cobb-Douglas normalizada
—es decir, una transformación monótona— pero que dé más señal de entrenamiento.** Le respondí con
un argumento. Un argumento no es un resultado, y este proyecto no cierra lanes con argumentos.
Esto lo convierte en número.

## 2. Lo que el argumento predice, escrito antes de medirlo

Nuestro estimador ([`h_regime`](../scripts/run_cobb_douglas_component_headroom_v1.py#L107))
normaliza cada contexto por min-max **después** de calcular la métrica, así que el máximo
normalizado vale 1 en cada contexto y

```
H_regime  =  1 − max_a  mean_r  Ṽ(r, a)
```

De ahí salen tres predicciones **falsables**:

1. **En la rejilla de 288, `H = 0` para toda `f` creciente.** Ahí `scalar_h_regime = 0,0` exacto,
   lo que sólo puede ocurrir si una misma configuración es el argmax en los seis regímenes. Si
   `a*` maximiza `V` en todo régimen, maximiza `f(V)` en todo régimen. **Falla si los seis argmax
   no coinciden**, en cuyo caso mi lectura del cero era errónea.
2. **El supremo sobre `f` crecientes se alcanza en una función escalón.** `H` es `1 −` un máximo
   sobre configuraciones de una media en `f`; el máximo es convexo en `f`, así que el ínfimo del
   segundo término se toma en un punto extremo del conjunto de funciones crecientes acotadas, y
   esos puntos extremos son los indicadores `1[V ≥ t]`. **Falla si alguna `f` muestreada al azar
   supera el mejor escalón.**
3. **La `f` que maximiza `H` es la que destruye más resolución.** Un escalón parte las
   configuraciones en dos clases y no puede retener más del ~50 % de los pares ordenados.
   **Falla si en el óptimo la resolución se conserva.**

## 3. Qué se mide

Sobre **las dos rejillas** (288 y 4.608), con el índice canónico del proyecto —sus cinco
variables, exponentes derivados con **su** regla sobre **nuestros** máximos, `kappa_dot` dentro de
`(contexto, semilla)`—:

| cantidad | definición |
|---|---|
| `H_identity` | `H_regime` sin transformar. **Debe reproducir el artefacto sellado** |
| `argmax_per_context` | la configuración óptima en cada régimen. ¿Coinciden? |
| `ceiling_step` | `max_t H_regime(1[V ≥ t])` sobre todos los umbrales distintos |
| `ceiling_sampled` | el mejor `H` sobre una familia logística `σ(β(V−t)/s)` y 2.000 monótonas aleatorias |
| `resolution(f)` | fracción de pares de configuraciones que `f` sigue ordenando estrictamente, media sobre contextos |
| **`H_at_90pct_resolution`** | el mejor `H` alcanzable **conservando ≥ 90 % de la resolución** |

`H_at_90pct_resolution` es la cifra decisiva: es exactamente lo que el PI pidió —más señal, mismo
orden— y no la que maximiza `H`.

## 4. Reglas de lectura, fijadas antes de mirar

Umbral `GATE = 0,05`, el de todos los gates del proyecto.

* `ceiling < 0,05` → **`MONOTONE_RESCALING_CANNOT_REACH_THE_BAR`**. La vía queda cerrada por
  número. No se vuelve a abrir sin física nueva.
* `ceiling ≥ 0,05` **y** `H_at_90pct_resolution < 0,05` →
  **`THE_BAR_IS_ONLY_REACHED_BY_DESTROYING_THE_SIGNAL`**. La propuesta queda refutada *en sus
  propios términos*: pedía las dos cosas a la vez y son incompatibles.
* ambas `≥ 0,05` → **`A_MONOTONE_RESCALING_REACHES_THE_BAR_WITH_SIGNAL_INTACT`**. Es un hallazgo
  real. **No autoriza adoptarla**: la `f` se elegiría entonces por **mecanismo declarado**, en un
  preregistro propio, nunca por ser la que más `H` da.

**Y la regla que hace esto honesto:** si el techo cruza, se reporta **junto con la advertencia de
que cruzarlo es una propiedad de la curvatura de la métrica y no de la física de la cadena**. Un
`H_regime` que se puede subir reescalando no mide headroom físico. Esa frase entra al manuscrito
gane quien gane.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_identity_reproduces_the_sealed_scalar` | `H_identity` debe igualar `scalar_h_regime` de `results/cobb_douglas_component_headroom{,_extended}/result.json` a 1e-9. **Ancla externa**: falla si la reimplementación del índice se desvía |
| `f2_steps_attain_the_supremum` | ninguna de las 2.000 monótonas aleatorias ni de la familia logística puede superar `ceiling_step`. Falla si alguna lo hace, y entonces la búsqueda estaba incompleta y el techo reportado es un suelo |
| `f3_the_base_grid_ceiling_is_zero` | en la rejilla de 288 el techo debe ser exactamente 0 y los seis argmax deben coincidir. **Falla si mi predicción 1 es falsa** |
| `f4_headroom_and_resolution_trade_off` | la resolución en el óptimo debe ser menor que la de la identidad. Falla si se conservan, y entonces la tensión que afirmo no existe |
| `f5_no_fresh_seeds` | custodia central, réplica declarada |

`f1` es el ancla: sin ella, un techo nuevo no es comparable con nada de lo publicado.

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica, no autoriza
aprendices, y **no cambia la primaria del contrato**.
