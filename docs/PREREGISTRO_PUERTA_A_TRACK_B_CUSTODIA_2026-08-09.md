# Preregistro — Puerta A: cobrar la prima de Track B con custodia

**Fecha:** 2026-08-09. **Congelado antes de escribir el runner.**
Contrato marco: `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`.
**Rol:** `DEVELOPMENT_PAIRED_CONFIRMATION_NO_LEARNER_BEYOND_THE_ARMS_DECLARED_HERE`.

## 1. Qué hay y qué le falta

`results/track_b_nonneural/result.json` (2026-08-07) es el único sitio del repositorio donde una
red bate a un comparador **no neuronal**:

```
constant_best   96.567
threshold_rule  97.142   (+0.575 [LCB95 +0.330] sobre la constante)
trained_mlp     98.743   (+2.176 sobre la constante, +1.60 sobre la regla)
trained_kan     98.516 · trained_dmlpa 98.004 · untrained_net 72.202
```

Su propio alcance dice `DEVELOPMENT_NO_CUSTODY_SEEDS_NO_ADJUDICATION`, y le faltan **cuatro cosas**:

1. **No hay emparejamiento.** El bake-off guardó sólo la media por (arquitectura, semilla): 5
   números, ningún episodio. El `+1,60` es una diferencia de medias **sin intervalo**.
2. **Los modelos no existen.** `run_architecture_bakeoff_v1.py` **nunca llamó a `model.save`**, así
   que no se pueden reevaluar. Verificado: cero ocurrencias de `model.save` en el fichero.
3. **El bloque de evaluación no es fresco.** `777_000` ya se usó para elegir arquitectura.
4. **No hay placebo de historia**, así que memoria y cadencia no están separadas.

**Consecuencia: hay que reentrenar.** No es una preferencia; es la única forma de tener episodios
emparejados y modelos auditables.

## 2. Semillas

Excepción del PI, bloque **`9200001–9200120`**, verificado libre: **cero colisiones** con el
registro y **cero** con las semillas de cualquier artefacto sellado.

* **entrenamiento** `9200001–9200005` — cinco réplicas de optimizador, como el bake-off;
* **ajuste de comparadores** `9200011–9200034` — donde se busca la mejor constante y los umbrales;
* **evaluación fresca** `9200051–9200098` — 48 tapes, **nunca usadas** para entrenar ni para elegir.

Los tres bloques son disjuntos y `f1` falla si se tocan.

## 3. Los brazos, enumerados aquí y cerrados

| brazo | qué es | entrena |
|---|---|---|
| `random_action` | acción uniforme en `[-1,1]^8` | no |
| `constant_best` | mejor constante de 8 dimensiones, buscada en el bloque de ajuste | ajusta |
| `threshold_rule` | constante base desplazada por señales del último frame, ajustada en el mismo bloque | ajusta |
| `mlp` | PPO MlpPolicy con historia de 16, presupuesto ~200k parámetros | sí |
| `mlp_shuffled_history` | **placebo**: la misma red, con la pila de historia barajada dentro del episodio | no (reusa pesos) |
| `mlp_frozen_history` | **placebo**: la misma red, con la historia congelada en el primer frame | no (reusa pesos) |

Los dos placebos **conservan la red y destruyen la información temporal**, que es lo que separa
memoria de cadencia. Reusan los pesos entrenados: no se reentrena para el placebo.

## 4. El estimando primario

```
Delta_calidad = mean_tape( mlp - threshold_rule )
```

emparejado **por tape** sobre las 48 de evaluación, con bootstrap pareado de 20.000 réplicas.
**Contra la regla, no contra la constante.** El contraste contra la constante se reporta al lado,
como diagnóstico.

Endpoint: el mismo `ret_mean` del entorno `track_b_v1` que usaron el bake-off y el comparador
no-neuronal, para que el número sea comparable con el `+1,60` que se intenta cobrar. Su escala es
la del entorno y no se cambia aquí; el contrato marco pide Cobb-Douglas como primario del programa
y **esta puerta declara su desviación**: cambiar el endpoint a la vez que el diseño haría
inseparable qué causó la diferencia.

## 5. Falsadores, y por qué cada uno puede fallar

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_blocks_are_disjoint` | los tres bloques no se tocan | elegir el comparador donde se mide es la fuga que infla todo |
| `f2_training_actually_moved_the_policy` | la red entrenada bate a la no entrenada | 200k pasos pueden no bastar |
| `f3_rule_beats_the_constant` | la regla mejora sobre la mejor constante | si no, el comparador es de paja y la prima no significaría nada |
| `f4_quality_premium_over_the_rule` | `LCB95(mlp − regla) >= +0,01` | **puede fallar**: es exactamente lo que la corrida sin custodia no midió |
| `f5_beats_both_history_placebos` | bate a barajado y congelado | si empata, lo medido es capacidad, no memoria |
| `f6_budget_is_matched` | parámetros dentro del 10 % del objetivo | un presupuesto desigual mide capacidad |
| `f7_a_control_must_differ` | random pierde contra la constante por margen | un arnés que no distingue nada acuerda con todo |

## 6. Reglas de lectura, en orden

1. Si `f1` falla → `BLOCKED_INSTRUMENT`, nada más se lee.
2. Si `f3` falla, la regla no es comparador válido y el veredicto es
   `NO_VALID_NONNEURAL_COMPARATOR`, **aunque la red gane**.
3. `f4` **y** `f5` deciden juntos. Con ambos:
   `TRACK_B_QUALITY_PREMIUM_CONFIRMED_UNDER_CUSTODY`.
4. Con `f4` y sin `f5`: `PREMIUM_IS_CAPACITY_NOT_MEMORY`, que también es publicable y es una
   respuesta distinta a la Q1 de Garrido.
5. Sin `f4`: `TRACK_B_QUALITY_PREMIUM_DID_NOT_SURVIVE_CUSTODY`. **Es un resultado, no un
   contratiempo**, y cierra la puerta A.

**No hay rama que diga «casi».** El SESOI es `+0,01` del contrato marco y no se mueve después de
ver el intervalo.
