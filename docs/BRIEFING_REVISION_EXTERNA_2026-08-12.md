# Briefing — estado real tras la noche del 11 al 12 de agosto

**Fecha:** 2026-08-12 · **Rama:** `codex/expanded-contract-comparators-v2` · **HEAD:** `67d81b71`
**Supersede:** `docs/BRIEFING_REVISION_EXTERNA_2026-08-10.md`, cuyas tres afirmaciones falsas están
retractadas aquí y en `docs/RETRACTACION_CONTENTION_V1_Y_ENMIENDA_X_2026-08-12.md`.
**Suite:** 2364 passed, 2 skipped, 2 xfailed.

---

## 0. Lo que cambió en una noche, en una línea

El proyecto tiene, **por primera vez, un positivo con custodia completa** — y **no es neuronal**.
Al mismo tiempo, la prima neural que el briefing anterior presentaba como su mejor resultado **se
cayó** contra una clase comparadora completa.

---

## 1. EL POSITIVO — conversión observable segura, a potencia adecuada

`results/program_o/powered_replication_v1/result.json` →
**`OBSERVABLE_CONVERSION_SURVIVES_AT_ADEQUATE_POWER`**, 7/7 falsadores.

**288 tapas vírgenes por celda** (bloque `7500001–7500288`, excepción PI documentada), seis
sub-bloques disjuntos de 48, con el runner congelado ejecutado **byte-idéntico** en cada uno.

```
celda              cola punto   cola LCB sim   primario punto   primario LCB
rho75_share90       +0.038786      +0.021591        +0.091016      +0.077131
rho90_share75       +0.024439      +0.010936        +0.076211      +0.064356
rho90_share90       +0.127313      +0.103989        +0.117207      +0.099978

critico simultaneo usado: 2.8770   (el que mato a O fue 2.8357)
```

`ret_visible_cvar10` —la restricción de cola que cerró Program O con LCB simultáneos de
**−0,008578** y **−0,015507**— **cruza cero en las tres celdas**.

**Ninguna laxitud se usó.** No se estrechó la familia de multiplicidad (habría bajado la n necesaria
de 154 a 93), no se introdujo margen tolerante de no-inferioridad, no se tocó el SESOI. El único
grado de libertad fue **el tamaño de muestra**, que es lo único que la lista
`no_post_failure_changes` del contrato correctivo **no** prohíbe.

**La prueba de que era potencia está en los sub-bloques:** `[STOP, STOP, STOP, PASS, STOP, STOP]`.
Cinco de seis paran solos con 48 tapas y uno pasa — la firma de un efecto real bajo un intervalo
demasiado ancho.

**Límites, y son duros:**

* **la política es un belief-MPC CLÁSICO** (`belief_mpc__3`). Esto **no es una prima neural**;
* **no promueve a Program O**, que sigue cerrado (`second_rescue_forbidden: true`) e inmutable. Es un
  **programa nuevo que hereda su física**;
* apertura única: no habrá segundo bloque para esta hipótesis.

## 2. El headroom que lo sostiene, y su corrección

`results/program_o/hpi_jensen_null_v1/result.json` → **`H_PI_SURVIVES_ITS_JENSEN_NULL`**.

```
safe_h_pi observado   +0.151514
nulo de Jensen         media +0.114431   p95 +0.120352   p=0.0000
```

Sobrevive de forma contundente. **Pero el 75,5 % del titular era sesgo de selección** —el máximo
sobre 65.536 calendarios— y nadie lo había medido. El headroom **corregido de sesgo es +0,037083**,
que aún supera la barra de 0,01 por **3,7×**.

El nulo fungible de O (`0,0` exacto) **no podía** controlar esto: la varianza intra-tapa entre
calendarios bajo fungibilidad es `0.000e+00`, así que era un nulo de **física**, no de **estimador**.

**Toda cita futura de `H_PI` debe decir 0,0371, no 0,1515.**

## 3. El bucle externo — vive, pero el portador es clásico

* **retención**: `AUC(reset) − AUC(retenida)` = **+0,0607**, 6/6 familias bajo inferencia
  simultánea. Sólido.
* **portador neural**: `results/program_n/outer_loop_carrier/result.json` →
  **`RETENTION_YES_NEURAL_CARRIER_NO`**. La neurona menos el mejor portador clásico es
  **−0,007010 [−0,024399, +0,013955]**, con `ucb1_transfer` (0,045023) por delante de
  `neuron_memory` (0,052033). Estimando que **nadie había calculado**.
* en *simple regret* final la familia que sobrevive es **`lookahead_kg`, NO la neurona**
  (`neuron` simultaneous_lcb95 −0,0035).

**La respuesta a Garrido es que retener estado de búsqueda reduce el coste de redescubrimiento, y
que el portador que mejor lo hace es UCB1.** No hace falta una red, y hay que decirlo así.

---

## 4. LO QUE SE CAYÓ

### 4.1 La prima de predicción no sobrevive a la clase completa

`results/program_n/gate_b_readjudication/result.json` →
**`SURFACE_PREMIUM_SURVIVES_THE_NARROW_CLASS_ONLY_NOT_THE_WIDENED_ONE`**

```
mlp_tuned  vs gaussian_process  +0.0342 [-0.1030, +0.1715]   no
kan_tuned  vs gaussian_process  +0.0172 [-0.1783, +0.2128]   no
recurrent  vs gbdt_lagged       -0.0300 [-0.1113, +0.0513]   no
```

Contra `linear_interactions` el MLP daba +0,1081 y el recurrente +0,1487. Mismas features, mismos
folds, mismas semillas, mismo criterio — **sólo se ensanchó la clase comparadora**, y `gbdt_lagged`
(0,9306) es sencillamente mejor que el recurrente (0,9007). **El número era real; su interpretación
no**: medía que `linear_interactions` no es el mejor clásico de esa superficie.

### 4.2 No hay prima de decisión, y no podía haberla

`phase3_decision_surrogate` → **`NO_DECISION_PREMIUM`**: con el `argmax` congelado y compartido, un
random forest elige mejor (regret 0,000022) que las dos redes.

`phase3_decision_headroom` → **`DECISION_HEADROOM_IS_JENSEN_BIAS`**: un oráculo al que se le da el
mejor buffer de cada contexto compra **+0,000065** contra una barra de 0,01, y contra un nulo cuya
**media es +0,003978** — sesenta veces el observado. **Todos los brazos competían por un premio menor
que el ruido.** El óptimo sí se mueve; moverse con él no compra nada.

### 4.3 Control y amortización, cerradas

Sin cambios: `linear_feedback` (99,127) bate al MLP (98,567) por −0,559 [−0,748, −0,386];
`NO_QUALIFYING_EXPERT`; `PLANNER_OBJECTIVE_IS_FLAT`.

---

## 5. LO QUE SE RETRACTA DEL BRIEFING ANTERIOR

| afirmación anterior | realidad |
|---|---|
| `contention_v1`: aprendiz − belief-MPC **+0,0136 [+0,0124]** | **no existe en ningún artefacto**. Real: +0,011477 [+0,009135] contra SESOI 0,010 — no cruza. `AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED` |
| «el único sitio donde un aprendiz batió a un planificador» | el aprendiz también «gana» con `min_dwell=1`, con **más** tapas favorables (58/60 vs 51/60); `min_dwell` y `ρ` están **confundidos**; y los dos brazos «MPC» son filtro + reparto **miope** |
| enmienda `d_min` a Program X | **retractada**. `v1` vuelve a ser el contrato vigente |
| «bloque virgen» en Gate B | el registro se declara incompleto: lo demostrado es `NO_KNOWN_COLLISION`. Grado corregido: `PROSPECTIVE` |
| «1/6 en simple regret final» sin nombrar familia | la familia es **`lookahead_kg`**, no la neurona |
| Q «ganó en media y murió en la cola» | **no ganó en media** contra el clásico: `Delta_N` −0,00159/−0,00072/−0,00041. Sólo batió al lazo abierto |
| `H_PI = 0,1515` | **0,0371** corregido de sesgo de Jensen |

Y dos defectos de instrumento reparados: `F.summarise` ignoraba la custodia
(`gate_b_cd_surface` sellaba `all_passed: true` junto a `custody.passed: false`), y `run_role`/`scope`
eran cadenas fijas.

---

## 6. El claim más fuerte que hoy se puede escribir

> **En un banco de contención sobre un recurso escaso no fungible, un controlador belief-MPC
> clásico convierte headroom privilegiado en ventaja observable que satisface simultáneamente sus
> guardarraíles de cola y de equidad, sobre 288 tapas vírgenes por celda y con inferencia simultánea
> sobre 69 estimandos. El headroom que explota es 0,0371 una vez descontado el sesgo de selección.
> Separadamente, retener estado de búsqueda entre corridas reduce el coste de redescubrimiento en
> 6/6 familias, y el portador que mejor lo hace es UCB1. Ninguna arquitectura neuronal supera a su
> mejor comparador clásico en predicción, en decisión ni en control.**

Es un paper de **operacionalización con control estructurado**, con un negativo neuronal fuerte y
completamente instrumentado. No es el paper de la prima neural.

## 7. Dónde apuntar ahora

1. **La pregunta 1 de Garrido tiene respuesta empírica** y es incómoda: la categoría que mejor
   operacionaliza el aprendizaje de la cadena **no es una red**; es una regla de búsqueda con estado
   retenido. Eso es publicable y es honesto.
2. **La pregunta 2 tiene una respuesta constructiva**: el DES debe exponer un ledger causal, decidir
   antes del shock y devolver el resultado al ciclo siguiente — y ahora hay un caso donde eso
   funciona con guardarraíles satisfechos.
3. **Lo que falta para una prima neural** sigue sin candidato: no existe en el árbol ningún
   comparador que planifique de verdad, y los dos que se llaman «MPC» en `contention_bench_v1.py`
   llaman ambos a `_myopic_split`.

## 8. Autoridades

```
67d81b71  la replica con potencia
bbe035fb  el nulo de Jensen sobre H_PI
47f0303e  R1, que refuto mi propia hipotesis
6811ac9f  el estado gobernado por la clase mas ancha
099b9402  las dos reglas de nombres
3c67b881  la retractacion

results/program_o/powered_replication_v1/result.json     OBSERVABLE_CONVERSION_SURVIVES_AT_ADEQUATE_POWER
results/program_o/hpi_jensen_null_v1/result.json         H_PI_SURVIVES_ITS_JENSEN_NULL
results/program_o/r1_tail_in_the_objective_v1/result.json  mi hipotesis, refutada
results/program_n/gate_b_readjudication/result.json      la prima cae con la clase ancha
results/program_n/outer_loop_carrier/result.json         RETENTION_YES_NEURAL_CARRIER_NO
results/program_n/phase3_decision_headroom/result.json   DECISION_HEADROOM_IS_JENSEN_BIAS
```
