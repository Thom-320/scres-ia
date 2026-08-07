# Análisis del export de David — 7 de agosto de 2026

**Fuente:** `david_export.json`, `schema_version: david_kan_lab_export_v1`, creado 2026-08-07T20:26:33Z
**Repo que corrió:** `08c14ff` (rama `david/kan-lab`, 2026-08-06)
**Máquina:** Kaggle, Tesla T4, `cuda` elegido (26,73 vs 9,90 pasos/s en CPU), `n_envs=3`, torch 2.10.0+cu128
**Config:** `RUN_PROFILE=final`, 100.000 pasos × 5 semillas × 2 brazos, `HISTORY_LEN=16`, `OBS_VERSION=v10`,
`MAX_STEPS=104`, `N_STEPS=512`, `EVAL_EPISODES=24`, semillas 9491–9495
**Coste:** ~10.777 s de entrenamiento agregado (~3 h)

Este documento analiza lo entregado. **No adjudica nada**: la corrida es de desarrollo, sin semillas
de custodia. Se escribe para devolvérselo a David con lo que hay que corregir antes de que estos
números entren a ningún sitio.

---

## Resumen en una línea

> **El efecto de memoria reportado (+0,99) no se distingue de cero, su celda de control —que por
> construcción debe dar 0— da +0,68, y el brazo etiquetado `DMLPA` corrió en realidad con la KAN
> encendida. La corrida no está desperdiciada: es la primera ejecución de `DMLPA_KAN` y da el primer
> suelo de ruido medido para `track_b_v1`.**

---

## 1 · El contraste principal cruza cero

`verdict_preliminary` reporta:

```
persistent_mean                    98,19468
independent_mean                   97,20432
persistent_minus_independent_mean  +0,99036
persistent_minus_independent_sd     1,46181
n_seeds                             5
```

De ahí, SE = 1,46181 / √5 = **0,65374**:

| criterio | IC95 | lectura |
|---|---|---|
| normal (z=1,96) | **[−0,2910, +2,2717]** | cruza cero |
| **t(df=4) = 2,776**, que es el correcto con n=5 | **[−0,8244, +2,8052]** | cruza cero |

La celda 8 del cuaderno imprime esta lectura en pantalla («el intervalo cruza cero: no se distingue
de no retener nada»), pero `verdict_preliminary` sólo exporta media y sd. **El JSON no lleva la
conclusión que el cuaderno sí calculó.** Es la primera corrección al instrumento: exportar el IC y
el veredicto, no las piezas para reconstruirlo.

---

## 2 · La celda nula: el ruido es el 68 % del efecto

En `train_arm` (celda 9):

```python
model = build_model(env, seed, device=BEST_DEVICE)
if arm == 'persistent' and carried is not None:
    model.policy.load_state_dict(carried)
```

En `order 0`, `carried is None`. **El brazo persistente no hereda nada**: misma semilla, misma
arquitectura, mismos 100k pasos, misma evaluación que su gemelo independiente. Es una celda nula por
construcción y debe dar 0.

| order | semilla | persistent | independent | Δ | sd(ind) |
|---|---|---:|---:|---:|---:|
| **0** | 9491 | 98,4148 | 97,7396 | **+0,6753** | 2,411 |
| 1 | 9492 | 97,6247 | 94,0691 | +3,5557 | **4,401** |
| 2 | 9493 | 98,4775 | 97,8953 | +0,5822 | 2,805 |
| 3 | 9494 | 97,6784 | 97,5631 | +0,1153 | 2,629 |
| 4 | 9495 | 98,7779 | 98,7546 | +0,0233 | 1,222 |

**Suelo de ruido medido: +0,6753 sobre un efecto declarado de +0,9904. El 68 %.**

### Causa mecánica, en el código

```python
def make_vec(seed=None):
    if n_envs == 1:
        return DummyVecEnv([lambda: make_env(seed)])
    ...
    return SubprocVecEnv([lambda: make_env(None) for _ in range(n_envs)], start_method='fork')
```

Con `n_envs=3` se toma la segunda rama y **la semilla se descarta**: los entornos de entrenamiento
van sin sembrar. Sumado a no determinismo de CUDA, dos corridas nominalmente idénticas divergen.

Esto no es culpa de David — el defecto está en el cuaderno que le entregamos, y conecta directamente
con el trabajo de determinismo en curso (`3473a31`).

### Sensibilidad: el efecto vive en una semilla

| subconjunto | Δ medio |
|---|---:|
| las cinco | +0,9904 |
| sin `order 0` (la celda nula) | +1,0691 |
| **sin `order 1`** (el atípico) | **+0,3490** |
| **sólo `order 2,3,4`** — con herencia real y sin atípico | **+0,2403** |

`order 1` aporta +3,5557 de los +4,9518 acumulados: **el 72 % del efecto viene de una sola celda**,
en la que el brazo independiente se hundió a 94,07 con la sd más alta de las diez corridas. Es una
corrida fallida del control, no una ganancia del tratamiento.

Y en los tres órdenes limpios el efecto **decrece** con el orden: +0,58 → +0,12 → +0,02. Lo contrario
de una curva de aprendizaje acumulativo.

---

## 3 · La pendiente de tendencia no es interpretable

`persistent_trend_slope = +0,07798` — verificado, el cálculo es correcto.

Pero el brazo independiente **no puede mejorar con el orden**: cada semilla arranca de pesos nuevos.
Su pendiente debe ser 0 por construcción. Medida:

```
slope persistent    +0,0780
slope independent   +0,5524     ← siete veces más empinada
```

La variable `order` está confundida con algo que no es la herencia de pesos. Si la pendiente del
persistente se leyera como curva de aprendizaje, habría que concluir que **resetear enseña siete
veces más que recordar**, que es absurdo. El estadístico se retira hasta que el control dé plano.

---

## 4 · El brazo etiquetado `DMLPA` corrió con la KAN encendida

**Este es el hallazgo que cambia qué es esta corrida.**

Tres hechos, verificables en el propio export:

1. El notebook del repo (`david/kan-lab`, `notebooks/scresia_david_kan_lab.ipynb` línea 405) define
   la firma con **`use_kan=False`**.
2. La celda 5 que David ejecutó (`cells_executed[4]`, sha `441e4f50a3d3fcf5`) la tiene en
   **`use_kan=True`**.
3. `dmlpa_factory` (sha `74374eac116bab41`) **no pasa `use_kan`** → cae en el default, que ahora es
   `True`.

Confirmación independiente dentro del JSON, sin leer una línea de código —
`parameter_matching`:

```json
"DMLPA":     {"width": 69, "params": 225410, "error": 0.12705},
"DMLPA_KAN": {"width": 69, "params": 225410, "error": 0.12705}
```

**Idénticos hasta el último parámetro.** Dos fábricas distintas no coinciden así salvo que
construyan el mismo objeto.

### Consecuencias

- **La DMLPA lisa sigue sin ejecutarse.** Nunca se ha corrido.
- **`DMLPA_KAN` sí se ejecutó.** El propio cuaderno la describe como «la única pregunta de
  arquitectura que sigue abierta». Esta corrida es su primera ejecución, mal etiquetada.
- Todo lo que diga `DMLPA` en este export debe releerse como `DMLPA_KAN` antes de entrar a un
  artefacto, una tabla o una diapositiva.

`edited_objects.DMLPA.sha256` viene `null` porque `inspect.getsource` falla con clases definidas en
notebook. **Segunda corrección al instrumento:** cuando `source_of` devuelva `None`, caer a
`type(obj).__mro__` o al texto de la celda, para que la arquitectura efectivamente ejecutada quede
hasheada y no dependa de la etiqueta.

---

## 5 · Sin suelo — pero esta vez sí es comparable

La corrida compara dos brazos de la misma red. Es el hueco E1 del registro de huecos, repetido: sin
comparador no neuronal, ni un empate ni una victoria dicen si hace falta una red.

La buena noticia: David evalúa con `seed0=777000` y 24 episodios, que es **exactamente el bloque de
evaluación** de `results/track_b_nonneural/result.json` (`eval_block_seed0: 777000`,
`eval_episodes: 24`). Los números son directamente comparables.

| | score | vs mejor constante |
|---|---:|---:|
| `constant_best` (nuestro suelo) | 96,567 | — |
| `threshold_rule` (nuestro) | 97,142 | +0,575 · LCB95 +0,330 |
| **David, `independent`** | **97,204** | **+0,637** |
| `trained_dmlpa` (nuestro, sellado) | 98,004 | +1,437 |
| **David, `persistent`** | **98,195** | **+1,628** |

Dos lecturas:

- El brazo independiente de David, con 225.410 parámetros y una KAN dentro, **apenas empata con
  nuestra regla de umbral**, que no tiene ninguna red.
- El brazo persistente llega donde ya estaba nuestra `trained_dmlpa` sellada. La memoria no lo lleva
  más allá de lo que el entrenamiento normal ya daba.

### Nota sobre igualdad de parámetros

```
MLP        199.215   desviación 0,4 %
KAN        204.816   desviación 2,4 %
DMLPA      225.410   desviación 12,7 %
```

Pasa la tolerancia declarada de 30 %, así que el cuaderno no aborta — correctamente, según su propio
contrato. Pero la DMLPA lleva **13 % más capacidad que el MLP**. Cuando se compare arquitectura
contra arquitectura, esa holgura hay que apretarla o declararla, porque es exactamente la objeción
que David levantó contra el preprint anti-KAN.

---

## 6 · Custodia

- Semillas de entrenamiento **9491–9495**: no figuran en `research/seed_custody_registry.json`.
- Semillas de evaluación **777000–777023**: tampoco, aunque coinciden con el bloque que ya usa
  `track_b_nonneural`.

No es un problema para una corrida de desarrollo, y el registro global está en
`BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`. Se anota para que nadie confunda después
esta corrida con una confirmación.

Clasificación: **`DEVELOPMENT_NO_CUSTODY_NO_ADJUDICATION_ARCH_LABEL_DEFECT`**.

---

## 7 · Qué nos indica

**Es la tercera medición independiente del mismo patrón.**

| instrumento | el brazo con memoria gana a… | …y falla contra |
|---|---|---|
| `search_ladder_v5` (desarrollo) | su gemelo reseteado | — (no hay contrafactual fuerte) |
| `grid_transfer_confirmation_v2` (confirmación) | arranque en frío | **su replay marginal state-blind** |
| **export de David** (desarrollo) | su gemelo independiente | **su propia celda nula** |

Tres instrumentos distintos, tres contrafactuales distintos, la misma forma: **la retención produce
un punto estimado positivo que no sobrevive a su propio control.**

Eso **refuerza** la enmienda 1 al claim freeze en vez de contradecirla. E1 congeló que el estado
retenido es necesario pero no suficiente, y que su estructura decide si la transferencia sobrevive un
contrafactual más exigente. La corrida de David es el tercer contrafactual y sale igual.

**Lo que sí aporta, y no lo teníamos:**

1. **La primera ejecución de `DMLPA_KAN`** — la única pregunta de arquitectura declarada abierta.
2. **Un suelo de ruido medido para `track_b_v1`: ≈ 0,68 puntos.** Cualquier contraste en ese entorno
   por debajo de ese valor está dentro del ruido de arranque. Hasta hoy lo suponíamos.
3. **Confirmación de que la ventaja de C1 no crece con memoria.** Refuerza el `NO-GO`: si el bloque
   virgen se abriera para C1, mediría un efecto del mismo orden que su propio ruido de arranque.

---

## 8 · Qué le devolvemos a David

En este orden. **Nada de esto requiere semillas nuevas.**

### D1 · Corregir la etiqueta

La corrida es `DMLPA_KAN`, no `DMLPA`. Re-exportar con `ARCH` correcto, o anotarlo en el fichero.
Es su resultado y es más interesante con el nombre correcto: nadie había corrido esa variante.

### D2 · Sembrar los entornos y acotar el nulo

En `make_vec`, pasar la semilla también en la rama `SubprocVecEnv`:

```python
return SubprocVecEnv([lambda i=i: make_env(None if seed is None else seed + i)
                      for i in range(n_envs)], start_method='fork')
```

Después, repetir **sólo `order 0`** tres o cuatro veces. Son ~35 min por repetición, no una campaña.
Eso convierte el +0,675 de anécdota en suelo de ruido con intervalo. Es el número más útil que puede
producir hoy, porque **calibra todos los contrastes futuros de ese entorno**, no sólo éste.

### D3 · Correr la DMLPA lisa

`use_kan=False`. Sigue sin existir, y es su arquitectura. Con el nulo de D2 acotado, el contraste
`DMLPA` vs `DMLPA_KAN` a parámetros emparejados sí sería una respuesta de arquitectura.

### D4 · Poner el suelo, no otra red

Antes de leer cualquier comparación de arquitecturas en `track_b_v1`, correr la escalera
constante → umbral en ese mismo entorno y bloque. Nuestro `track_b_nonneural` ya la tiene:
`constant_best` 96,567 y `threshold_rule` 97,142. **No hay que recalcularla, sólo citarla.**

### D5 · Qué NO hace falta

- Más semillas del mismo brazo persistente: sin acotar el nulo, más semillas dan un intervalo más
  estrecho alrededor de una cantidad confundida.
- Ajustar hiperparámetros de DMLPA: el cuaderno ya dice que converge, y el problema no es la
  convergencia.
- `nhead4` / `1layer`: optimizar arquitectura antes de que exista una prima neural estable.

---

## Custodia de este documento

Datado, no se edita en sitio. Los datos analizados están en `david_export.json` tal como llegó; este
documento no lo modifica ni lo importa a `results/`. Si David re-exporta, se emite un sucesor.
