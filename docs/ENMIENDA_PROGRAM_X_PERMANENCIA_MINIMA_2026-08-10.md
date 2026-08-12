# Enmienda a Program X — permanencia mínima, antes de que exista una sola semilla

**Fecha:** 2026-08-10 · **Autoriza:** PI
**Contrato v1:** `contracts/program_x_o_scale_amortized_control_v1.json` (`bbb35be`) — **se conserva**
**Contrato v2:** `contracts/program_x_o_scale_amortized_control_v2.json`
**Preregistro base:** `docs/PREREGISTRO_PROGRAM_X_O_SCALE_2026-08-09.md`

Se aplica la propia regla de versionado de v1: *«este diseño candidato puede cambiar sólo antes de
abrir semillas; todo cambio material exige una versión nueva y hashes»*. **Ninguna semilla existía
cuando se hizo este cambio**, y v1 no se edita en sitio.

## Por qué v1, tal como está, no puede dar una prima de calidad

El §3 del preregistro define la transición como Markov de primer orden con kernel simétrico:

```
P(Z_{t+1}=j | Z_t=i) = rho          si j = i
                     = (1-rho)/(N-1) si j != i
```

Permanencia geométrica, sin memoria de cuánto lleva el régimen activo. Bajo esa física **el
posterior exacto es estadística suficiente**, y el propio documento lo escribe en su §7:

> *Con el HMM exacto conocido, el posterior es la estadística suficiente nula.*

Es exactamente la física que cerró todo lo demás en este repositorio:

| lane | estimando | resultado |
|---|---|---|
| Program Q | neural − mejor clásico state-rich | −0,00159 / −0,00072 / −0,00041, equivalentes ±0,01 |
| Program V abstracto | privilegiado − Bayes | +0,000764 [−0,000798, +0,002326] |
| `headroom/g3_obs_conversion_v2` | — | `STRUCTURED_CONTROL_SUFFICES_G3_OBS` |
| `headroom/g2_autotomy_threshold` | — | `THRESHOLD_RULE_SUFFICES` |

Con la creencia exacta calculable en forma cerrada, una red **sólo puede empatar**. Por eso v1 ruta
todo a amortización, y es coherente — pero significa que v1 puede producir un claim de **coste** y
es **estructuralmente incapaz** de producir uno de **calidad**. Y su rama de amortización depende de
que el planificador incumpla un SLA, que es el gate que el E* falló:
`results/program_n/gate_c_prereq_mpc_quality/result.json` →
`PLANNER_OBJECTIVE_IS_FLAT_NO_QUALITY_TO_MEASURE`.

## El cambio

El estado latente pasa a ser **`(Z_t, D_t)`** — régimen y periodos transcurridos desde el último
cambio, truncado en `d_min`:

```
P(Z_{t+1}=i | Z_t=i, D_t <  d_min) = 1                 el regimen NO puede cambiar
P(Z_{t+1}=i | Z_t=i, D_t >= d_min) = rho
P(Z_{t+1}=j | Z_t=i, D_t >= d_min) = (1-rho)/(N-1)
```

**Factorial `d_min ∈ {1, 4}`.** Con `d_min = 1` se recupera v1 **exactamente**, y ésa es la razón de
que esté ahí: es el **control negativo** de toda la enmienda, y es primario, no opcional.

Con `d_min > 1` el tiempo desde el cambio forma parte de la verdad, y un filtro de primer orden
—el controlador que un practicante escribe a partir de la dinámica declarada— **no puede
representarlo**. Está mal especificado. Ese hueco, y sólo ese, es donde un aprendiz puede cobrar.

## La evidencia que lo respalda, y es nuestra

`supply_chain/contention_bench_v1.py`, celda `positive` (`min_dwell = 4`): el aprendiz batió al
belief-MPC por **+0,0136 [LCB95 +0,0124]**. Es el **único** sitio de todo el proyecto donde eso ha
ocurrido, y la única diferencia estructural con las celdas donde no ocurre es la permanencia mínima.
El banco lo dice en su propio encabezado:

> *THE DWELL IS THE FINE POINT. With `min_dwell > 1` the regime is semi-Markov: the true state
> includes time-since-switch. A first-order two-state Bayes filter — the model-based controller a
> practitioner actually writes — is MISSPECIFIED.*

`grep -c dwell` sobre el contrato v1 daba **0**. Sobre `contention_bench_v1.py` da **8**.

## Lo que la enmienda añade además del kernel, y por qué

**Brazo de divulgación obligatorio — filtro semi-Markov exacto sobre `(Z, D)`.** No es el listón que
el aprendiz debe superar; existe para que una ventaja sobre el filtro mal especificado **no pueda
presentarse como superioridad sobre la optimalidad decisoria**. Se reporta en toda celda con
`d_min > 1`. Sin esto, la enmienda fabricaría una victoria haciendo tonto al comparador — que es
justamente lo que el propio audit reprocha a K3 y a Q2.

**Gate `G4b`, antes de cualquier learner.** Con `d_min > 1`, el planificador de primer orden **debe
rendir mediblemente peor** que el filtro exacto. Si no lo hace, la mala especificación en la que se
apoya todo esto no es material y **la rama de calidad se cierra antes de entrenar nada**.

**Celda nula `d_min = 1`.** Con el posterior suficiente, el aprendiz **no debe** batir al planificador
exacto más allá de la tolerancia Monte Carlo congelada. Un aprendiz que gane **ahí** es evidencia de
fuga o de comparador mal entrenado, nunca de prima.

**Confusión con la persistencia, declarada.** Subir `d_min` sube la permanencia media, así que un
efecto de `d_min` podría ser un efecto de persistencia. La enmienda ejecutable **debe** incluir
celdas que igualen la permanencia media entre `d_min = 1` y `d_min = 4` bajando `rho`. Sin eso, la
comparación mide persistencia y no mala especificación.

**Alcance del nulo IID.** Queda definido **sólo** en `d_min = 1`. Con `d_min > 1` el proceso no puede
ser IID por construcción, así que correr ese nulo allí sería un falsador incapaz de fallar — la
regla que este repositorio ya pagó dos veces con bloques de semillas quemados.

## Lo que NO cambia

Toda la disciplina de v1 se conserva sin tocar: escalera acumulativa de comparadores, decoder entero
con suma exacta de tres lotes, `q = 1/N` como nulo de señal, clonación byte-idéntica para H4, los
gates G0–G5, la separación de calidad / amortización / generalización, ReT y Cobb-Douglas como
secundarios congelados y reportados por separado, el endpoint físico primario en [0,1], y
`Op2 = 190.000 unidades de cada rm cada 672 h`.

**Y no cambia la custodia.** El registro sigue en
`BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`. Program X sigue sin rangos asignados y sin
autorización científica. Esta enmienda **no abre ningún gate**: cambia la física candidata para que,
si algún día se autoriza, la pregunta que responda pueda tener una respuesta positiva.

## El precio, dicho por adelantado

Con `d_min > 1` el belief-MPC de primer orden es **deliberadamente subóptimo**. Eso es legítimo
—es el controlador que un practicante escribe— pero hay que decirlo en cada tabla, y por eso el
brazo de divulgación es obligatorio y no opcional. La afirmación defendible sería *«una red bate al
controlador model-based que un practicante escribiría, en un régimen donde el modelo declarado está
mal especificado»*, jamás *«una red bate al control óptimo»*.
