# Preregistro — Fase 4: el efecto Alzheimer, medido

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_meta_learner_over_configs_v1.py`.
Contesta **directamente** las dos preguntas de Garrido 2024 y **no depende de que aparezca
headroom**.

## La pregunta, tal como él la formula

Su Fig. 2 marca los nodos ③ (variables de decisión) y ⑧ (métrica de SCRES) como **los dos
extremos de una cadena en lazo abierto**. Un algoritmo de IA colocado entre ellos la cierra. La
ausencia es lo que llama el **efecto Alzheimer**: *«the modeled SC network fails to retain
information from previous experiences—or simulation scenarios»*.

Su Fig. 5 dice **exactamente** qué poner ahí: una neurona cuyas dendritas son los cuatro drivers
`d_i`, ponderadas por `ρ`, con una función de activación del tipo *«¿es la medida de SCRES en la
configuración x mayor que en la (x−1)?»*.

Y la tesis de 2017 es el caso: su diseño experimental es **un factor a la vez** (5 niveles de
inventario × 3 de turno), repetido **desde cero** para cada familia de riesgo. Eso *es* el lazo
abierto, y el coste de no recordar es medible.

> **La pregunta que este experimento contesta, sin metáforas:**
> ¿cuántas corridas de simulación necesita cada estrategia para encontrar la mejor configuración
> en un contexto de riesgo **nuevo**, y **cuánto de esa diferencia se debe a recordar**?

## Diseño

**Espacio de configuraciones (288)** — sus variables de decisión, más las dos que la campaña de
sensibilidad identificó como las únicas con autoridad:

| variable | niveles |
|---|---|
| `buffer_hours` | 0, 168, 336, 504, 672, 1344 *(sus seis)* |
| `shifts` | 1, 2, 3 *(sus tres)* |
| `op9_rop` | 12, 24, 36, 48 |
| `op12_rop` | 12, 24, 36, 48 |

**Contextos (6)** — las «experiencias sucesivas» de sus H2/H3: {R1r, R2r, R1r+R2r} × {base,
escalado ×3 en frecuencia}. Se recorren **en orden**, como corridas sucesivas de un estudio.

**Estrategias, todas con el mismo presupuesto de corridas:**

1. **OFAT** — el diseño de la tesis: partir del defecto, barrer un factor, fijar el mejor, pasar
   al siguiente. **Es el lazo abierto.**
2. **Búsqueda aleatoria** — el nulo honesto. Cualquier afirmación tiene que batirlo.
3. **Neurona de la Fig. 5, con memoria** — tras cada corrida ajusta `ReT ≈ σ(Σ ρ_i d_i)` sobre
   los drivers y elige la siguiente configuración por valor predicho. **Los pesos `ρ` cruzan de
   un contexto al siguiente**: eso es el atributo SCL.
4. **La misma neurona, reiniciada en cada contexto** — la **ablación del recuerdo**. Idéntico
   código, idénticas semillas; **lo único que cambia es si `ρ` se conserva**.

**(3) contra (4) es el efecto Alzheimer medido.** Es el contraste que da el número de §4.2.

**Métricas**: *regret* contra el mejor real del contexto en función del presupuesto, y
**corridas hasta quedar dentro del 1 % del mejor**. Semillas vírgenes `5 300 001…`, CRN entre
estrategias: **todas ven exactamente la misma superficie**.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_surface_has_a_real_optimum` | si la dispersión entre configuraciones no supera al ruido entre semillas, «encontrar la mejor» no significa nada y todas las estrategias empatan por construcción |
| `f2_ofat_is_really_one_factor_at_a_time` | cada paso de OFAT debe cambiar **exactamente una** coordenada; si cambia más, no es el diseño de la tesis y la comparación es contra un hombre de paja |
| `f3_memory_is_the_only_difference` | (3) y (4) deben compartir semillas, orden de contextos y código; si difieren en algo más, el contraste no aísla el recuerdo |
| `f4_random_search_is_uninformed` | si el aleatorio consulta la tabla antes de correr, deja de ser un nulo |
| `f5_no_context_leakage` | el aprendiz no puede haber evaluado una configuración del contexto actual antes de elegir en él |
| `f6_seeds_are_virgin` | reutilizar semillas invalidaría la confirmación |

## Regla de lectura, fijada de antemano

* **(3) alcanza el 1 % del óptimo en menos corridas que (1) y que (2), con `LCB95 > 0` en la
  diferencia** → la integración funciona, y el número es la respuesta a su Q2.
* **(3) bate a (4)** → **el efecto Alzheimer tiene un precio medido**, y ése es el resultado
  central del paper. Si (3) ≈ (4), recordar no aporta **en este espacio**, y hay que decirlo
  igual de claro: sería un negativo sobre su hipótesis central.
* **(2) bate a (3)** → la neurona no aprende nada útil; la familia de algoritmos correcta no es
  ésta, y su Q1 se responde por descarte, no por confirmación.

**Lo que este experimento NO afirma:** nada sobre control dentro del episodio, ni sobre RL. Es
aprendizaje **entre corridas**, que es lo que su Fig. 2 pide. Las dos cosas son distintas y no
las voy a mezclar al reportar.
