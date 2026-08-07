# Preregistro — el comparador no-neuronal que falta en `track_b_v1`

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_track_b_nonneural_v1.py`.
Entorno idéntico al de todas nuestras corridas de arquitectura: `track_b_v1`, `obs v10`,
`HISTORY_LEN=16`, `MAX_STEPS=104`, acción `Box(-1, 1, (8,))`.

## 1. Por qué existe, y qué admite

Hemos medido, en este entorno, que **entrenar compra +20 a +26** y que **elegir entre KAN, MLP y
DMLPA no compra nada**. Lo que **no** hemos medido nunca aquí es lo que decide si hay prima neural:

> **¿Le gana una red a una política que no es una red?**

En `track_b_v1` **no existe ni un comparador estático ni una regla heurística**. Los tres brazos
que hemos comparado son redes, y el único no-entrenado es *la misma red* con pesos aleatorios —
que mide el efecto de entrenar, no la necesidad de la arquitectura.

**El orden fue el equivocado y queda dicho:** este experimento debía preceder a las comparaciones
entre arquitecturas, no seguirlas.

## 2. Los brazos

| brazo | qué es | entrena |
|---|---|---|
| `random_action` | acción uniforme en `[-1,1]^8` cada paso | no |
| `constant_best` | **la mejor acción constante de 8 dimensiones**, buscada por muestreo aleatorio | no (se ajusta) |
| `threshold_rule` | regla sobre el estado: constante base, desplazada por señales observadas del último frame | no (se ajusta) |
| `untrained_net` | DMLPA con pesos iniciales, 0 pasos | no |
| `trained_mlp`, `trained_dmlpa` | de `results/architecture_bakeoff_200k/` | sí, 200k |

## 3. La fuga que hay que evitar, y cómo se evita

Buscar la mejor constante **sobre los episodios de evaluación sería un oráculo**, y el resultado
no valdría nada. Por eso:

* la búsqueda de `constant_best` y de los umbrales se hace **sobre un bloque de ajuste
  disjunto**, `seed0 = 888_000`, con 8 episodios por candidato;
* la evaluación final usa **exactamente el mismo protocolo que las redes**: `seed0 = 777_000`,
  24 episodios, que es el que ya usaron el bake-off y el control sin entrenar.

**`f1` falla si los dos bloques se tocan.** Es el falsador central.

## 4. Reglas de lectura, fijadas antes de mirar

Primaria: ReT medio del episodio. Contraste contra `constant_best`, bootstrap sobre los episodios
de evaluación.

* **redes > `constant_best`, IC95 excluyendo el cero** →
  **`NEURAL_PREMIUM_ESTABLISHED_IN_TRACK_B`**. Es el primer positivo neural del proyecto y exige
  confirmación en bloque virgen antes de reclamarse.
* **empate** → **`A_CONSTANT_ACTION_SUFFICES`**. La red no es necesaria en este entorno, y la
  contribución del paper pasa a ser dónde sí y dónde no hace falta aprender.
* **`constant_best` > redes** → **`THE_CONSTANT_BEATS_THE_NETWORKS`**, el negativo más fuerte
  posible y también publicable.

**El compromiso:** la tabla completa entra al manuscrito gane quien gane, y si sale empate **no se
reescribe como «las redes igualan a la heurística con menos coste»** sin medir el coste.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_constant_is_not_fitted_on_the_evaluation_block` | ajuste en `888_000`, evaluación en `777_000`. **Falla si se solapan**, y entonces la constante es un oráculo y todo el experimento se cae |
| `f2_the_arms_share_the_action_space_and_protocol` | los seis usan `Box(-1,1,(8,))`, 24 episodios y el mismo `seed0`. Falla si algún brazo evalúa distinto |
| `f3_the_harness_can_detect_skill` | las redes entrenadas deben batir a `random_action` con IC que excluye el cero. **Si no, el arnés no separa nada y ningún empate significa algo** |
| `f4_the_constant_search_actually_searched` | la mejor constante no puede ser la primera muestreada ni quedar en un vértice del cubo por defecto. Falla si la búsqueda fue decorativa |
| `f5_no_fresh_seeds` | bloques de episodios, no semillas de custodia; se declara |

**Alcance:** desarrollo. No abre semillas de custodia, no adjudica y **no autoriza reclamar prima
neural sin confirmación posterior**.
