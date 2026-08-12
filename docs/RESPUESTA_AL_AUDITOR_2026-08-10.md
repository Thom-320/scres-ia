# Respuesta al auditor — el Programa N ya respondió tu siguiente punto de decisión

**Fecha:** 2026-08-10 · **Rama:** `codex/expanded-contract-comparators-v2`
**Tu trabajo:** `bbb35be`, fusionado sin editar en `e6959857`
**Lo que no podías ver:** 21 commits posteriores a `c09cd2d2`, con el **Programa N** entero.

---

## 0. Lo primero, porque es lo justo

Tu auditoría es mejor que mi briefing en cobertura y en precisión de autoridad. Dos puntos donde
llegaste antes y por razonamiento puro:

**El gate del E\*.** Escribiste que pasó *por número de llamadas (>60), no por latencia (presupuesto
60.480 s)*, y que la regla de llamadas sola **no prueba cuello operacional**. Es exacto, lo dedujiste
del artefacto sin correr nada, y yo estuve citando
`H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE` sin cuestionarlo hasta que lo medí.

**El ReT sin acotar.** 38 celdas con ReT>1 y máximo **160,2564** por la rama `0,5/RP`. Más duro y más
citable que nuestro «premia el abandono».

Tu tabla de las cuatro afirmaciones que no deben mezclarse —*el feedback vale · retener historia
vale · una red aporta calidad · una red amortiza cómputo*— es la mejor pieza conceptual del
material, y **H5 es indispensable** es correcto: sin ella, H1–H4 las satisface Bayes, MPC o UCB1.

---

## 1. Tu siguiente punto de decisión, ya medido — y en dirección negativa

Pediste, antes de congelar destilación:

> *4. perfil operacional del planner.*

Se hizo. Dos etapas, ninguna semilla abierta, ninguna red entrenada.

### C0 — auditoría de expertos candidatos

`results/program_n/gate_c0_expert_audit/result.json` → **`NO_QUALIFYING_EXPERT`**

Un experto merece amortizarse sólo si es **caro** *y* **mejor**. Los dos candidatos del árbol fallan
mitades distintas:

| experto | caro | mejor | qué se midió |
|---|---|---|---|
| `k3_strong_mpc` | **no**, 0,051× | sí, `ret_order +0,01242 [+0,00546, +0,01928]` | instrumentado sobre 320 decisiones: **0 evaluaciones de candidato, 0 llamadas al simulador** |
| `estar_direct_des_mpc` | sí, 44.359× | **no** | nunca medido — hasta ahora |

**El brazo que K3 llama `strong_mpc` no planifica.** Es `paced_policy(α, β, γ)`, la misma regla en
forma cerrada que su brazo `inventory_paced` con α liberado; su propio docstring dice *«no latent
state»*. Es **20× más barata** que la red que la imitaría, así que `Δ_amortización` es negativo por
construcción. Tu inventario clasificaba K3 como retractado por el confound de comparador de período
ocho; esto es un defecto **distinto y adicional** en el mismo lane.

### C0-prerequisito — la calidad del E\*, medida

`results/program_n/gate_c_prereq_mpc_quality/result.json` →
**`PLANNER_OBJECTIVE_IS_FLAT_NO_QUALITY_TO_MEASURE`**

Un `DirectDESMPC` real: 52 épocas, 24 tapas de desarrollo step-3, **9.984 evaluaciones de candidato,
254.592 pasos de replay, 7,47 s por episodio planificado**.

```
objetivo por constante, 8 niveles:  -3100,0  en TODOS, en las 24 tapas
n_lost por constante:  0,0->251,0   0,125->200,5   0,25->225,2  ...  1,0->242,9
```

**El objetivo que el planificador maximiza es exactamente constante.** Sus 9.984 candidatos empatan
siempre, comete `GRID[0]` en cada época, y su plan es una política constante a la que llegó por el
camino más caro posible. Como `frac = 0,0` es el peor nivel físico, aterriza ahí:

```
n_lost vs mejor constante   -50,46 [-52,62, -48,30]   0/24 tapas favorables
n_lost vs secuencia aleatoria  -15,92 [-18,50, -13,33]   0/24
```

Pierde contra el azar. Y el ledger físico **sí** responde a la acción —óptimo interior en 0,125—, así
que la acción hace algo y **la recompensa no lo ve**.

**Consecuencia para tu §«escala amortizable».** Tu diagnóstico era «el gate por número de llamadas es
demasiado débil para establecer un cuello operacional». Es correcto y ahora hay algo peor que
añadirle: en ese sustrato **no hay nada que planificar**. Tu gate A2 —SLA absoluto o break-even
vinculante— sigue siendo el gate correcto, y ahora se sabe que el E\* no sólo no lo pasaría: no
califica siquiera como experto.

---

## 2. Dos filas de tu inventario que cambian

### Track B — la prima murió contra una clase de comparador ensanchada

`results/program_n/gate_a2_track_b/result.json` → `NO_QUALITY_PREMIUM_AGAINST_THE_WIDENED_CLASS`

```
linear_feedback  99,127     <- comparador nuevo, no neuronal
mlp              98,567
threshold_rule   98,095
constant_best    98,016

mlp vs mejor no-neuronal  -0,559 [-0,748, -0,386]   7/48 tapas
mlp vs threshold_rule     +0,472 [+0,275, +0,658]   37/48
```

El MLP **sí** bate a la regla de umbral y a los dos placebos de historia —barajada y congelada—, así
que la memoria hace algo medible. Pierde contra cuatro líneas de realimentación lineal. Registrado
como `SUPERSEDED_BY_A_WIDENED_COMPARATOR_CLASS` en `research/supersession_registry.json`, con la
regla de lectura de que su número se cita **contra la clase estrecha que batió**, nunca como prima.

### «RNN no es el ingrediente ausente» — cierto para control, falso para predicción

Es la única frase tuya que pediría matizar. Como **controlador**, tienes razón y está triplemente
medido. Como **predictor**, el brazo recurrente bate a `linear_lagged` —su comparador clásico con
**exactamente** la misma información— por **+0,1487 [+0,1069, +0,1905]**, en bloque virgen
9600001–9600008, 7/7 falsadores:
`results/program_n/gate_b_confirmation_v3/result.json` → `SURFACE_PREMIUM_CAPTURED`, re-adjudicado
contra el mejor no neuronal de cada clase de información en
`results/program_n/gate_b_readjudication/result.json`.

Distinto estimando, no contradicción. Pero tal como está escrita, esa frase se llevaría por delante
el resultado más limpio del repositorio — y es literalmente la Fig. 5 de Garrido **como predictor**.

Sus límites, ya medidos y que impongo yo mismo:

* **la arquitectura no replica** — KAN gana en dos corridas, MLP en una, ninguno en la sensibilidad;
  la afirmación defendible es de **familia**;
* **es específico del endpoint Cobb-Douglas** — en `ret_excel` ningún brazo neuronal bate al mejor
  clásico (`kan − tree = −0,0029 [−0,0839, +0,0782]`, empate; `linear_lagged` encabeza todo);
* es **predicción**, no control.

---

## 3. La enmienda a Program X, y es lo único que te pido que discutas

`contracts/program_x_o_scale_amortized_control_v2.json` ·
`docs/ENMIENDA_PROGRAM_X_PERMANENCIA_MINIMA_2026-08-10.md`
v1 se conserva; las ocho comprobaciones del validador pasan sobre v2
(`results/program_x/o_scale_design_preflight_v2/result.json`).

**Tu diseño es serio** —escalera acumulativa, decoder entero con suma exacta, `q = 1/N` como nulo de
señal, ver que `ρ = 0` **no** sería IID, clonación byte-idéntica para H4, G0–G5 antes del learner—.
Es la disciplina que a nosotros nos habría ahorrado meses.

**Pero tal como está no puede producir una prima de calidad, y lo dice tu propio §7.** Tu §3 define
una transición Markov de primer orden con permanencia geométrica; bajo esa física el posterior exacto
es estadística suficiente, y lo escribes: *«con el HMM exacto conocido, el posterior es la estadística
suficiente nula»*. Es la física que cerró Q (−0,00159/−0,00072/−0,00041), V (+0,000764
[−0,000798, +0,002326]), G3 y G2. Por eso rutas todo a amortización, y es coherente — pero deja a X
capaz de un claim de **coste** y **estructuralmente incapaz** de uno de **calidad**. Y su rama de
amortización depende de que el planificador incumpla un SLA, que es justo lo que acabamos de medir
en negativo.

`grep -c dwell` sobre tu contrato v1: **0**. Sobre `supply_chain/contention_bench_v1.py`: **8**.

**El cambio.** El estado latente pasa a `(Z_t, D_t)` con permanencia mínima `d_min`:

```
P(Z_{t+1}=i | Z_t=i, D_t <  d_min) = 1                el regimen NO puede cambiar
P(Z_{t+1}=i | Z_t=i, D_t >= d_min) = rho
P(Z_{t+1}=j | Z_t=i, D_t >= d_min) = (1-rho)/(N-1)
```

Factorial `d_min ∈ {1, 4}`, con `d_min = 1` recuperando v1 **exactamente** como control negativo
primario.

**La evidencia es nuestra y es la única de su tipo.** En `contention_v1`, celda con `min_dwell = 4`,
el aprendiz batió al belief-MPC por **+0,0136 [LCB95 +0,0124]**. Es el **único** sitio de todo el
proyecto donde eso ha ocurrido, y la única diferencia estructural con las celdas donde no ocurre es
la permanencia mínima.

**Cuatro salvaguardas, porque esta enmienda podría fabricar una victoria haciendo tonto al
comparador** —que es exactamente lo que tú reprochas a K3 y a Q2:

1. **Brazo de divulgación obligatorio:** filtro semi-Markov exacto sobre `(Z, D)`. No es el listón que
   el aprendiz debe superar; existe para que una ventaja sobre el filtro mal especificado **no pueda**
   presentarse como superioridad sobre la optimalidad decisoria. Se reporta en toda celda `d_min > 1`.
2. **Gate `G4b`, antes de cualquier learner:** con `d_min > 1`, el planificador de primer orden debe
   rendir mediblemente peor que el filtro exacto. Si no, la mala especificación no es material y **la
   rama de calidad se cierra sin entrenar nada**.
3. **Celda nula `d_min = 1`:** el aprendiz **no debe** ganar ahí. Si gana, es fuga o comparador mal
   entrenado, nunca prima.
4. **Confusión con la persistencia, declarada:** subir `d_min` sube la permanencia media, así que la
   enmienda ejecutable **debe** incluir celdas que igualen la permanencia media bajando `ρ`. Sin eso,
   la comparación mide persistencia y no mala especificación.

**El precio, dicho por adelantado:** con `d_min > 1` el belief-MPC de primer orden es
deliberadamente subóptimo. La afirmación defendible sería *«una red bate al controlador model-based
que un practicante escribiría, en un régimen donde el modelo declarado está mal especificado»*, jamás
*«una red bate al control óptimo»*.

Nada de la custodia cambia: `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, X sin rangos y
sin autorización científica. **La enmienda no abre ningún gate**; cambia la física candidata para
que, si algún día se autoriza, la pregunta pueda tener una respuesta positiva.

---

## 4. Un patrón que te pido que sigas cazando

Tú encontraste que `H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE` certificaba menos de lo que su
nombre afirma. El mismo día, midiendo, cayeron otros dos:

* **`train_cell_mean_comparator`, llamado «techo»** — lo superan brazos neuronales en las **cuatro**
  corridas de la Puerta B, incluida la de desarrollo que acuñó el término. La cifra «+0,0625 de
  margen disponible» está retirada (`docs/CORRECCION_TECHO_SUPERFICIE_CD_2026-08-09.md`).
* **`strong_mpc`** — no planifica.

Los tres son el mismo defecto: **un nombre puesto por el papel esperado, sin una medición que lo
respalde**, y los tres cayeron con el primer falsador que los midió. La regla que dejó: *un artefacto
no puede llamar techo, cota, óptimo, planificador ni elegible a una cantidad sin un falsador que lo
compruebe.*

No tengo motivo para creer que sean los únicos tres. Es lo más valioso que puedes seguir haciendo.

---

## 5. Preguntas concretas para ti

1. **¿Aceptas la enmienda `d_min`?** Si no, ¿cuál es tu ruta a un claim de **calidad** —no de
   amortización— dado que tu propio §7 declara el posterior suficiente?
2. **¿Es `G4b` el gate correcto**, o hay una forma más barata de falsar «la mala especificación es
   material» antes de gastar cómputo?
3. **Con el E\* descalificado como experto**, ¿queda algún planificador en el árbol que sea a la vez
   caro y mejor? Si no lo hay, la rama de amortización de X no tiene teacher y habría que decirlo en
   el contrato.
4. **`retention_simultaneous`: 6/6 en AUC pero 1/6 en simple regret final.** Es la debilidad del
   único resultado fuerte que tenemos. ¿Cierras esa asimetría antes que abrir X, o después?
5. **¿Más nombres del §4?**

---

## 6. Autoridades

```
e6959857  merge de bbb35be sin editar
21553715  los tres artefactos sellados del Programa N
results/program_n/gate_a2_track_b/result.json            NO_QUALITY_PREMIUM_AGAINST_THE_WIDENED_CLASS
results/program_n/gate_b_confirmation_v3/result.json     SURFACE_PREMIUM_CAPTURED (7/7, bloque virgen)
results/program_n/gate_b_readjudication/result.json      re-adjudicacion vs mejor no-neuronal
results/program_n/gate_b_sensitivity_ret_excel/result.json  sensibilidad legada, empate
results/program_n/gate_c0_expert_audit/result.json       NO_QUALIFYING_EXPERT
results/program_n/gate_c_prereq_mpc_quality/result.json  PLANNER_OBJECTIVE_IS_FLAT
results/program_x/o_scale_design_preflight_v2/result.json  8/8 sobre el contrato enmendado
docs/BRIEFING_REVISION_EXTERNA_2026-08-10.md             el estado completo
```

Suite: `pytest tests/ -q` → 2350 passed, 2 skipped, 2 xfailed antes del merge; los tests que trajiste
(`test_program_x_o_scale_contract.py`, `test_procurement_overorder_source_v2.py`) pasan 12/12.
