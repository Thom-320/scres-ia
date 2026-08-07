# Preregistro — sonda de reproducibilidad del bake-off (celda `DMLPA` / semilla 9492)

**Escrito y commiteado ANTES de correr.** Diagnóstico, no reclamo científico. Lo que fija es la
**regla de lectura de tres ramas**, porque sin ella cualquier resultado se puede contar como se
quiera después.

## 1. El hecho que la motiva

Misma semilla, misma arquitectura (`DMLPA`, 187.404 parámetros), mismos 200.000 pasos, y —
verificado línea a línea en los dos runners — **los mismos argumentos de `PPO`**
(`learning_rate 3e-4`, `n_steps 512`, `batch_size 64`, `gamma 0.99`, `gae_lambda 0.95`,
`clip_range 0.2`, `ent_coef 0.01`, `device cpu`, `n_envs 8`):

| semilla | bake-off sellado `results/architecture_bakeoff_200k` | reconfirmación `dmlpa_base` |
|---|---:|---:|
| 9491 | 98,059 | 98,946 |
| 9492 | **96,747** | **99,122** |

Ya quedó retirado el «igualar hiperparámetros restaura la comparabilidad entre corridas». Esta
sonda decide **cuál de tres cosas** está pasando.

## 2. El diseño

**Dos ejecuciones independientes de `scripts/run_architecture_bakeoff_v1.py`**, idénticas entre
sí y al artefacto sellado en todo lo declarable:

    --arch DMLPA --seeds 9492 --total-steps 200000 --eval-episodes 24 --n-envs 8

`--n-envs 8` **no se baja** aunque la máquina esté cargada: el valor sellado se produjo con 8, y
bajarlo cambiaría el entrenamiento y haría la comparación inválida. Se paga en tiempo.

Sean `A` y `B` los dos ReT medios obtenidos, y `S = 96,74672` el valor sellado.

## 3. Regla de lectura, fijada ahora

| rama | condición | lo que significa |
|---|---|---|
| **R1** | `A = B = S` (a 1e−9) | el bake-off **sí** reproduce. La discrepancia vive en `run_dmlpa_variants_v1.py` y hay que encontrarla ahí |
| **R2** | `A = B ≠ S` | el pipeline es determinista **hoy**, pero algo cambió desde las 07:15 UTC. La comparabilidad **en el tiempo** está rota y todo artefacto de entrenamiento anterior queda sin poder cruzarse con los nuevos |
| **R3** | `A ≠ B` | **el pipeline no es determinista en absoluto.** Ninguna comparación entre corridas es válida nunca; sólo sobreviven los contrastes **dentro** de un mismo barrido |

**Sospecha declarada antes de ver el número, para que pueda fallar:** `make_vec(n_envs, seed)`
construye `SubprocVecEnv([lambda: make_env(None) …])` y **descarta el argumento `seed`**. Si el
entorno de cada worker toma aleatoriedad no sembrada por worker, `fork` la vuelve dependiente del
arranque y la rama sería **R3**. Si sale `R1` o `R2`, **la sospecha queda refutada y se dice**.

## 4. Qué NO decide

No adjudica `nhead4` ni `1layer`: esos contrastes viven **dentro** del barrido de variantes y son
válidos en las tres ramas. No toca ningún artefacto sellado. No abre semillas: 9491–9495 son de
desarrollo y ya están abiertas.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_two_replicas_are_configured_identically` | compara los `design` de los dos artefactos clave a clave. Falla si difieren en semilla, pasos, episodios o `n_envs`, y entonces `A ≠ B` no probaría nada |
| `f2_the_sealed_cell_exists_and_is_the_one_named` | localiza la fila `(DMLPA, 9492)` en el artefacto sellado y compara su sello. Falla si el artefacto se movió o no contiene esa fila |
| `f3_n_envs_matches_the_sealed_run` | falla si alguna réplica corrió con `n_envs ≠ 8`, porque entonces la comparación con `S` es de otra cosa |

**Alcance:** diagnóstico de instrumento. Desarrollo.
