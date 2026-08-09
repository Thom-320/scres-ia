# Preregistro — validación positiva del audit sobre verdad conocida

**Fecha:** 2026-08-08. **Congelado antes de escribir el runner.**

## 1. Qué falta y por qué esto lo resuelve

El audit de `papers/before_learning/` demuestra hoy que puede **detener** un aprendiz. No demuestra
que pueda **autorizar** uno correctamente. Un método que sólo dice «no» no es un método: es un
escéptico. Sin la otra mitad, el comentario del revisor está escrito de antemano — *«esto es
depuración elaborada de un caso»*.

**No se hace sobre la MFSC, y es deliberado.** Sobre la MFSC no hay verdad conocida: sólo estimados
que se comparan con otros estimados, que es exactamente cómo un techo de doce tapes sobrevivió tres
semanas. Aquí la verdad se fija **por construcción** y el audit se corre a ciegas contra ella.

**Nada de lo que salga de aquí es un claim sobre la MFSC**, ni sobre Garrido-Ríos (2017), ni
reabre ninguna lane. Es un banco de pruebas del **instrumento**.

## 2. El banco: `contention_v1`

La única física en la que hemos medido headroom material es **contención por un recurso escaso y no
fungible** — `H_PI = 0,1515` con el nulo fungible en exactamente 0. El banco reproduce ese mecanismo
en pequeño y con un mando que lo apaga.

Por periodo `t = 1..T`:

* régimen `z_t ∈ {A_alto, B_alto}`, con **permanencia mínima `k`** y luego cambio con probabilidad
  `1-ρ`;
* demandas `d_A, d_B` Poisson, con media alta o baja según el régimen;
* **acción** `a_t ∈ [0,1]`: reparto de la capacidad `C` entre las dos clases;
* **fungibilidad `α`**: la capacidad sobrante de una clase sirve a la otra con eficiencia `α`;
* **observación** `s_t`: señal ruidosa de `z_t` con exactitud `q`, disponible **antes** de decidir;
* **endpoint** `V`: fill rate agregado sobre el horizonte. Mayor es mejor.

### La verdad, que es lo que hace que esto sea una validación y no otra medición

* **`α = 1` → `H_PI = 0` exactamente, por álgebra.** Con capacidad plenamente fungible el servicio
  es `min(d_A + d_B, C)` sea cual sea el reparto: la acción no puede cambiar el resultado. La celda
  nula **no es «no encontramos nada»; es «no hay nada que encontrar»**, y está demostrado, no
  medido.
* **`α = 0`, `ρ` alto → `H_PI > 0`.** El óptimo depende del régimen y un clarividente lo explota.

### La permanencia mínima `k`, que es el punto fino

Con `k > 1` el régimen es **semi-Markov**: el estado verdadero incluye el tiempo desde el cambio.
Un filtro bayesiano de primer orden sobre dos estados —el model-based que un practicante escribe—
está **mal especificado**. Ese es el hueco donde un aprendiz puede pagar, y decirlo así es el
contenido del resultado, no una excusa por él.

## 3. Las celdas, fijadas aquí

| celda | `α` | `ρ` | `k` | `q` | qué debe pasar |
|---|---|---|---|---|---|
| **NULA** | 1,0 | 0,90 | 4 | 0,85 | el audit **DETIENE** en el gate 2 |
| **POSITIVA** | 0,0 | 0,90 | 4 | 0,85 | el audit **AUTORIZA** y el aprendiz convierte |
| **CONTROL-SIN-MEMORIA** | 0,0 | 0,50 | 1 | 0,85 | headroom sí, valor de memoria **no** |

La tercera celda existe para que «autoriza» no se confunda con «autoriza siempre que haya
contención»: con régimen sin persistencia la historia no informa, y el gate de retención debe
quedarse corto aunque el de headroom pase.

## 4. La frontera estructurada, enumerada antes de correr

```
mejor acción fija            (rejilla de 21)
umbral sobre la señal s_t    (rejilla de acciones por señal)
MPC de creencia 1er orden    filtro bayesiano de dos estados, reparto miope óptimo
MPC de modelo-oráculo        [DIVULGACIÓN] conoce k y ρ verdaderos
```

El cuarto **no es un comparador que el aprendiz deba batir**: es la referencia superior que impide
vender como «prima neural» lo que en realidad es una prima sobre mala especificación. **Se reportan
las dos diferencias**, siempre juntas.

## 5. Los presupuestos

Idénticos entre brazos: mismas tapes, mismo número de episodios de evaluación, mismos derechos de
información (todos ven `s_t` y la historia; ninguno ve `z_t` salvo el clarividente y el oráculo,
ambos etiquetados). El aprendiz es un MLP sobre una ventana de historia, ajustado por CEM sobre las
semillas de **entrenamiento** y evaluado **sin reajuste** sobre las de test.

## 6. Semillas

Espacio propio y disjunto del registro de la MFSC: `9100001–9100120`. **60 de desarrollo
(9100001–9100060) y 60 de held-out fresco (9100061–9100120).** El bloque de held-out no se toca
hasta que la frontera estructurada y el aprendiz están congelados. Son semillas de un banco
sintético, no de la MFSC, y se registran como tales para que nadie las confunda con custodia
científica.

## 7. Los falsadores, y por qué cada uno puede fallar

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_null_is_algebraically_flat` | en la celda NULA, todas las acciones dan el mismo `V` bit a bit | si el simulador filtra valor por el reparto aun con `α=1`, la física está mal y el nulo no es nulo |
| `f2_audit_stops_on_the_null` | veredicto STOP en la celda NULA | si el audit autoriza donde la verdad es cero, produce **falsos positivos** y no sirve |
| `f3_positive_cell_has_real_headroom` | `LCB95(H_PI) ≥ 0,02` en la POSITIVA | la construcción podría no generar headroom suficiente; entonces no hay instancia positiva |
| `f4_placebo_loses` | la señal barajada por el mismo mapa pierde | si el placebo empata, lo medido es cadencia, no información |
| `f5_structured_frontier_was_searched` | ≥ 4 familias, mejor de cada una | una frontera pobre convierte cualquier cosa en prima |
| `f6_learner_converts_on_fresh_seeds` | `LCB95(aprendiz − mejor estructurado) ≥ +0,01` en held-out | puede perder; el MPC de creencia puede absorber el residual |
| `f7_memory_control_falls_short` | en CONTROL-SIN-MEMORIA el aprendiz **no** supera el SESOI | si gana también ahí, gana sin información y el banco mide otra cosa |
| `f8_oracle_gap_is_reported` | se reporta la diferencia contra el MPC-oráculo | sin ella, una prima sobre mala especificación se vende como prima sobre optimalidad |

## 8. Reglas de lectura, en orden

1. **Primero el nulo.** Si `f1` o `f2` fallan, el banco está roto y **nada más se lee**:
   `BENCH_INVALID`.
2. Si el nulo se comporta y la celda positiva pasa el gate de headroom, corre la escalera completa.
3. Veredicto `AUDIT_VALIDATED_IN_BOTH_DIRECTIONS` **sólo si**: STOP correcto en el nulo **y**
   autorización con conversión en held-out en la positiva **y** `f7` corto en el control.
4. Si el aprendiz no convierte, el veredicto es
   `AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED`. **Eso no es un fallo del
   experimento**: es el mismo resultado que llevamos publicando, y se reporta igual.

**No hay rama que diga «casi».** El SESOI es `+0,01` y está tomado de
`papers/before_learning/NEURAL_PREMIUM_PREREGISTRATION.md`, congelado antes que esto.
