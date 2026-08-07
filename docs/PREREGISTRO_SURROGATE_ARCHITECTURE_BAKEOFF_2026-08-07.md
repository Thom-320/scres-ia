# Preregistro — arquitecturas como surrogate DENTRO de la búsqueda

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_surrogate_architecture_bakeoff_v1.py`.
Caché sellada `results/surface_cache/wrap288_v1`, bloque quemado `5.300.001–012`, réplica declarada.
**Ninguna semilla nueva** — el registro central sigue en `NO_NEW_SEEDS_AUTHORIZED`.

## 1. Una corrección antes de nada

Escribí que el paso 3 *«acota el premio del paso 4 antes de entrenar»*. **Eso era demasiado
fuerte.** La auditoría lo señala y tiene razón: el guardarraíl preregistrado `worst_product_fill`
**no se aplicó** —el runner sólo persiste `flow_fill_rate`, un agregado— y un controlador que mejora
la media abandonando un producto pasaría el filtro actual. Con el guardarraíl incompleto,
`NO_STRUCTURED_CONTROLLER_CONVERTS` **es un diagnóstico de desarrollo válido y no define el residual
neural**.

Además el signo del lane MPC **no es estable**: cambia con el endpoint, con el incumbente y con el
bloque de tapes. Esa inestabilidad es un hallazgo, no ruido de entrenamiento.

**Por eso este experimento no vive en ese lane.** Vive donde la medición sí es estable: la búsqueda
sobre la superficie cacheada, con oráculo exacto y sin guardarraíl de servicio pendiente, porque no
se despliega ninguna política — se elige una configuración.

## 2. La pregunta

La KAN tiene **una** ventaja medida y no es como política: como **surrogate supervisado** de la
superficie de diseño gana a un MLP a 532 vs 529 parámetros, retenido, en los seis contextos
(0,9777 vs 0,7424 en R2r). Y como surrogate dentro de la búsqueda **empata** con la neurona de 5
parámetros.

> **¿Esa ventaja en precisión de ajuste se convierte en mejor búsqueda?**

Es la posición de la Fig. 5 de Garrido: el aproximador entre el nodo ③ y el ⑧.

## 3. Los brazos, y por qué cada uno puede ganar

Todos son **surrogates dentro del mismo bucle**: ajustan sobre lo visitado, puntúan lo no visitado,
evalúan, repiten. Mismo presupuesto 24, misma cinta, mismos contextos.

| brazo | por qué está | parámetros |
|---|---|---:|
| `neuron_5p` | la Fig. 5 de Garrido; **la referencia** | 5 |
| `kan` | la única ventaja de ajuste medida | ~532 |
| `mlp_matched` | control a parámetros igualados (±10 %) | ~529 |
| `gp_matern` | el surrogate clásico de la optimización bayesiana | — |
| **`gbt`** | **árboles con boosting**: en tablas de 288 filas y 4 variables es donde los árboles suelen ganar a las redes | — |
| **`spline_poly`** | **polinomio con interacciones**: en nuestras dos sondas previas de superficie, el clásico batió a ambas redes | — |

Los dos últimos son los que yo apostaría que ganan. **Incluirlos es lo que hace que una victoria de
la KAN signifique algo**; sin ellos, ganarle sólo a un MLP es un hombre de paja.

**GPU: irrelevante aquí y no se usa.** Son 288 puntos y cuatro variables; el ajuste es instantáneo
en CPU y el paralelismo útil es por semilla, no por tensor. Decirlo para no vender aceleración que
no existe.

## 4. Reglas de lectura, fijadas antes de mirar

Primaria `auc_regret_norm` (**menor es mejor**, coste de búsqueda). Secundaria **% del techo**, que
es la métrica que pidió Garrido. Contraste pareado por semilla contra `neuron_5p`, bootstrap sobre
semillas, **Holm sobre los cinco brazos no-referencia**.

* `kan` mejor que `mlp_matched` con IC95 que excluye el cero → **la arquitectura importa para
  buscar**, y la ventaja de ajuste se convierte. Primer positivo neural del proyecto; se preregistra
  confirmación aparte antes de reclamar nada.
* `kan` y `mlp_matched` indistinguibles → **paridad**: la precisión del surrogate no se convierte.
* un clásico (`gbt`, `spline_poly`, `gp_matern`) mejor que ambas redes → el titular honesto es
  **«los surrogates clásicos ganan»**, y se dice así.

**El compromiso:** la tabla de los seis brazos entra al manuscrito completa, gane quien gane.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_kan_and_mlp_are_parameter_matched` | los conteos deben quedar a menos del 10 %. **Es la objeción de David**; falla si volvemos a comparar 532 contra 31 |
| `f2_no_arm_can_read_an_unrun_cell` | `Surface` debe **lanzar**; se comprueba provocándolo |
| `f3_budgets_are_matched` | los seis gastan exactamente 24 evaluaciones por contexto |
| `f4_the_arms_are_not_the_same_policy` | las secuencias de visitas deben diferir entre brazos. Falla si dos son alias |
| `f5_the_harness_can_detect_skill` | el mejor brazo debe batir a `random` con IC que excluye el cero. **Si no, el arnés no distingue nada y ningún empate significa algo** |
| `f6_no_fresh_seeds` | custodia central, réplica declarada |

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica y no autoriza
confirmación.
