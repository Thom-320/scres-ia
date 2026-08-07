# Preregistro — ¿qué capa hace no determinista el arnés de Track B?

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_determinism_diagnostic_v1.py`.
Diagnóstico de instrumento. No abre semillas: 9492 es de desarrollo y ya está gastada.

## 1. El hecho

`results/repro_probe/{A,B}/result.json` y el bake-off sellado, misma semilla 9492, misma
arquitectura `DMLPA` (187.404 parámetros), mismos 200.000 pasos, mismos argumentos de `PPO`:

| corrida | ReT |
|---|---:|
| sellado | 96,746719517 |
| A | 94,383394755 |
| B | 96,023296013 |

**Dispersión a semilla fija: 2,363.** Dispersión del bake-off entre **cinco semillas distintas**:
**2,102**. La semilla no explica nada.

## 2. Mi sospecha preregistrada, y por qué la pongo en duda yo mismo

El preregistro de la sonda declaró como sospecha que `make_vec` **descarta su argumento `seed`**
en la rama multi-worker:

```python
return SubprocVecEnv([lambda: make_env(None) for _ in range(n_envs)], start_method="fork")
```

Es un defecto real. Pero leyendo SB3 2.9 **después** de ver el resultado:
`PPO(seed=seed)` → `BaseAlgorithm.set_random_seed` → `self.env.seed(seed)`, y `VecEnv.seed`
reparte `seed + idx` a cada worker para el siguiente reset. **Los workers sí quedan sembrados,
por otra vía.** Así que mi sospecha probablemente **no** es la causa.

**Arreglar una causa que no he demostrado produce una reparación que no verifica nada.** Por eso
esto mide antes de tocar.

## 3. Las tres configuraciones

Todo idéntico salvo la capa aislada: semilla 9492, `DMLPA`, `n_steps=512`, `ent_coef=0,01`,
`learning_rate=3e-4`, `device=cpu`. Dos réplicas por configuración, horizonte corto (20.000 pasos)
porque el determinismo se rompe o no se rompe, no necesita convergencia.

| id | `n_envs` | hilos de torch | qué aísla |
|---|---:|---|---|
| **A** | 8 | por defecto | la configuración que usaron **todos** los artefactos de Track B |
| **B** | 8 | 1 | quita la variación de orden de reducción en punto flotante intra-op |
| **C** | 1 | 1 | quita los subprocesos por completo |

## 4. Regla de lectura, fijada ahora

| condición | veredicto | qué implica |
|---|---|---|
| A difiere, **B coincide** | `TORCH_THREADING` | el arreglo es fijar los hilos |
| A difiere, B difiere, **C coincide** | `SUBPROCESS_WORKERS` | el arreglo está en la vec env |
| las tres difieren | `DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT` | **no se reclama arreglo**; se reporta como límite del entorno y todo Track B queda restringido a contrastes dentro de un barrido |
| **A coincide** | `INCONCLUSIVE_HORIZON_TOO_SHORT_TO_EXPOSE_IT` | el horizonte de 20k no lo expone; esta corrida **no decidió nada** y lo dice |

Tolerancia **exacta**: `1e-9`. Una tolerancia laxa llamaría determinista a un pipeline que deriva.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_diagnostic_can_come_back_empty` | si A reproduce a este horizonte, la corrida no aisló nada, y la regla la manda a `INCONCLUSIVE` en vez de dejar que una rama posterior reclame una causa. **Un diagnóstico que no puede volver vacío no es un diagnóstico** |
| `f2_all_three_share_seed_arch_and_hyperparameters` | una configuración que difiera en más que la capa aislada no aísla esa capa |
| `f3_the_tolerance_is_exact_not_approximate` | falla si alguien afloja `1e-9` |

## 6. Lo que decide

Si sale `TORCH_THREADING` o `SUBPROCESS_WORKERS`, hay arreglo y **la verificación es re-correr la
sonda de reproducibilidad completa**: pasa sólo si `A = B = 96,746719517` a `1e-9`.

Si sale `DEEPER_THAN_BOTH`, **no se arregla nada** y la consecuencia se escribe en el manuscrito:
en este arnés la semilla no es unidad de réplica, la prima neural de `track_b_v1` (+1,44 a +2,18)
**cae dentro de una banda de ruido de ±2,4** y el instrumento no puede resolverla. Eso refuerza el
NO-GO de C1 por una razón anterior al hueco A1.

**Alcance:** diagnóstico de instrumento. No adjudica ningún resultado científico.
