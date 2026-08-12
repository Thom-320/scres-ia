# Preregistro — Fase 3: surrogate orientado a decisión con optimizador congelado

**Fecha:** 2026-08-12 · **Escrito ANTES de correr nada.**
**Autoriza:** PI («ataca la fase 3», trabajo autónomo)
**Semillas:** ninguna nueva. Replay declarado de `program_n_gate_b_confirmation_v3`
(9600001–9600008), tapas ya consumidas.

---

## 1. La pregunta, y por qué es la única que queda

Hoy cerraron las otras: la prima de **predicción** no sobrevive a la clase comparadora completa
(`mlp − gaussian_process = +0,0342 [−0,1030, +0,1715]`; `recurrent − gbdt_lagged = −0,0300`), el
**portador neural** del bucle externo empata con `ucb1_transfer` por delante, y **control** y
**amortización** están cerradas.

Queda un hueco medido y nunca explorado: **ajustar mejor no es elegir mejor**, y nadie ha
comprobado si la red **elige** mejor configuración. `grep` de pérdida orientada a decisión en el
árbol: **0 resultados**.

> **¿Un surrogate entrenado sobre el REGRET DE LA DECISIÓN elige mejores configuraciones que el
> mejor surrogate clásico, con el optimizador congelado y las mismas features?**

## 2. El problema de decisión, que ya existe y no hay que inventar

El diseño de Gate B es **9 celdas** de contexto —`familia ∈ {R1r, R2r, R1r+R2r}` ×
`escalación ∈ {base, freq_x3, freq_x5}`— y **17 niveles de buffer** por celda. La decisión es:
*dado el contexto, ¿qué buffer se despliega?*

Para cada celda `c`, con `R̄_test(c,b)` la resiliencia media del buffer `b` sobre las semillas de
test:

```
regret(c) = max_b R̄_test(c,b)  -  R̄_test(c, b_elegido(c))
```

**Estimando primario:**

```
Δ_decision = regret(mejor clásico) - regret(neural)      positivo = la red elige mejor
```

## 3. El optimizador se congela; sólo cambia el surrogate

Todos los brazos usan **exactamente el mismo `argmax` exhaustivo** sobre los 17 niveles. Ninguno
puede ganar por tener mejor optimizador, porque es el mismo objeto. Lo que varía es únicamente el
modelo que puntúa.

**Brazos clásicos** (los que ganaron hoy): `gbdt`, `random_forest`, `gaussian_process`,
`kernel_ridge`, `linear_interactions`, `spline_buffer`.

**Brazos neuronales:**

* `mlp_mse` — la misma arquitectura de Gate B, entrenada sobre **MSE**. Es el control que aísla si
  el mérito es de la **pérdida** o de la **arquitectura**.
* `mlp_decision` — misma arquitectura, entrenada sobre **regret esperado**:

```
L = Σ_c  Σ_b  softmax_τ(ŷ(c,·))_b · ( max_b' y(c,b') - y(c,b) )
```

una política *soft* sobre las 17 acciones de cada celda, ponderando el regret verdadero. Es
diferenciable, no mira el test, y su `τ` se declara aquí: **τ = 0,02** sobre `y` estandarizado.

**Suelo obligatorio:** `random_surrogate`, un modelo sin ajustar. Si no pierde claramente, la
comparación no tiene resolución.

## 4. Falsadores, y por qué cada uno puede fallar

* **k1_la_decision_es_viva** — el buffer óptimo debe **variar entre celdas**. *Puede fallar:* si el
  mismo buffer gana en las nueve, el `argmax` es trivial, todo brazo elige igual y no hay nada que
  medir. Es el fallo que este proyecto ya sufrió como «el óptimo no se mueve».
* **k2_el_optimizador_es_identico** — un único `argmax` compartido; se cuenta cuántas veces lo
  invoca cada brazo y debe ser el mismo número. *Puede fallar* si alguien añade búsqueda por un lado.
* **k3_la_perdida_de_decision_aporta_sobre_MSE** — `mlp_decision` contra `mlp_mse`, misma
  arquitectura y presupuesto. *Puede fallar:* la pérdida orientada a decisión puede no aportar nada.
* **k4_la_red_bate_al_mejor_clasico** — el titular. `Δ_decision` con LCB95 > 0. *Puede fallar, y hoy
  lo esperable es que falle:* el `gbdt_lagged` bate al recurrente en predicción.
* **k5_un_control_debe_ser_peor** — `random_surrogate` debe tener regret claramente mayor.
* **custody** — `NOT_APPLICABLE`: replay declarado, cero semillas nuevas.

## 5. Reglas de decisión, escritas antes

| resultado | veredicto |
|---|---|
| k1 falla | `DECISION_PROBLEM_IS_DEGENERATE_NOTHING_TO_MEASURE` |
| k5 falla | `BLOCKED_NO_RESOLUTION` |
| k4 pasa | `DECISION_PREMIUM_FOR_THE_NEURAL_SURROGATE` |
| k4 falla y k3 pasa | `DECISION_LOSS_HELPS_BUT_NOT_ENOUGH_TO_BEAT_THE_CLASSICAL` |
| k4 y k3 fallan | `NO_DECISION_PREMIUM` |

**Ningún veredicto autoriza abrir semillas.** La Fase 2 sigue condicionada a una decisión del PI.

## 6. Lo que este preregistro NO permite

No permite cambiar `τ`, la arquitectura, la clase clásica ni el criterio después de ver resultados.
No permite entrenar sobre `ret_excel`. No permite añadir un brazo neuronal nuevo si los dos
declarados pierden: eso sería un tercer intento con el mismo bloque.

Y no permite presentar un resultado positivo como prima de **control**: elegir una configuración
entre corridas es el bucle externo de Garrido, no una decisión dentro del episodio.
