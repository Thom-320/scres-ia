# Preregistro — G3c: acoplamiento temporal, y el margen de no inferioridad que faltaba

**Estado:** `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`
**Runner (a escribir, no a ejecutar):** `scripts/run_g3c_temporal_coupling.py`
**Autoridad superior:** `contracts/authority_ladder_v1.json` en `main`
(`scientific_execution_authorized: false`, `fresh_roots_opened: false`). Este documento **no
autoriza ninguna ejecución**; define qué se ejecutará y bajo qué reglas cuando el recibo de
Submission A exista.

## 1. Por qué G3c va ahora ANTES de G3a, y con qué evidencia

Mi orden anterior era G3a → G3c. **Lo invierto, y no por argumento sino por medición.**

`results/headroom/contention_policy_class/result.json` (`HALTED_FALSIFIER_FAILED`, tapes quemados
`5.200.001–16`, ninguna semilla nueva) mide que **la restricción vinculante era la CLASE DE
POLÍTICA, no la simetría de la demanda**:

* el valor incremental del estado es **+0,036…+0,041** (LCB95 +0,021…+0,027) en las cuatro celdas;
* el **placebo no informado pierde** contra la mejor constante en las cuatro (−0,011…−0,013);
* apuntar al reclamante **equivocado** cuesta **−0,62**;
* mecanismo: el reclamante estresado **alterna de forma persistente** dentro del episodio, y el
  barrido sellado probó **sólo constantes**.

**Corolario que me obliga:** *la asimetría no era el ingrediente que faltaba.* G3a se diseñó para
romper una simetría que resultó no ser la barrera. **Su prioridad baja; no se cancela** — sigue
siendo el contrato que puede expresar una equivarianza A↔B real, hoy `NOT_EXPRESSIBLE_IN_split_v1`.

## 2. La pregunta de G3c, y por qué es la siguiente que importa

Una regla **miope equivariante** —«manda hacia quien va peor hoy»— ya captura el valor medido.
Con eso, un aprendiz no tiene nada que ganar: un `if` de dos ramas basta, que es exactamente el
desenlace de G2.

**El acoplamiento temporal es lo que rompe la regla miope.** Si cambiar de destino cuesta, o si
hay una permanencia mínima, entonces la acción de hoy restringe la de mañana y **la respuesta
óptima deja de ser función del estado actual**. Ésa es la única estructura del catálogo que un
comparador constante **no puede representar por su forma**, y donde un planificador puede superar
a la regla miope.

> **G3c.** Bajo permanencia mínima y coste de cambio, existe un residual **observable** que la
> mejor **regla miope equivariante** no captura y **un planificador (DP/MPC) sí**, con
> `LCB95 > SESOI` sobre `worst_claimant_fill` y **sin violar ningún margen de no inferioridad**.

**Nótese quién es el titular.** El comparador a batir **ya no es la mejor constante**: es la
**regla miope equivariante** medida en la reauditoría. Subir el listón es la consecuencia directa
de ese resultado, y evita reclamar por segunda vez un valor ya explicado.

## 3. Física nueva — una sola, no un bufé

**Sólo `N = 2`.** Se añade **una** familia de acoplamiento, con su parámetro barrido:

| ingrediente | parámetro | niveles |
|---|---|---|
| **permanencia mínima** (`min_dwell_days`) | días que una asignación debe mantenerse | 1 (nulo), 3, 7 |
| **coste de cambio** (`switch_cost_rations`) | raciones perdidas al reasignar | 0 (nulo), y dos niveles calibrados |

`min_dwell = 1` y `switch_cost = 0` **reproducen el modelo actual bit a bit** y son el control de
regresión, no una celda científica.

**Declarado como nuestra asunción, con su precio:** la tesis no especifica ni permanencia ni coste
de cambio de destino. **Lo decidimos nosotros**, se declara aquí, y su precio de fidelidad se mide
contra el brazo nulo — nunca se presenta como si viniera de Garrido.

## 4. EL MARGEN DE NO INFERIORIDAD — el instrumento que faltaba

La reauditoría se detuvo porque `f7` exigía **no deterioro a margen cero sobre estimados
puntuales**, y falló por **0,0625 pedidos: UN pedido en UNA de 16 semillas**. Las revisiones
externas lo habían nombrado de forma prospectiva —*«exigir no deterioro a margen cero es una
prueba de superioridad disfrazada»*— y envié el instrumento igualmente. **Esto lo repara.**

### Endpoint primario

`worst_claimant_fill`, **escalar**. `service_first_v2` se conserva como **regla de selección**,
jamás como estimando: un `LCB95` sobre una clave lexicográfica no significa nada.

**SESOI primario: `+0,010` de fill absoluto**, justificado en operación —un punto porcentual del
peor reclamante— y **no** heredado del `±0,01` de Program Q, que estaba en `ReT` canónico y no en
fill. Escalas distintas, margen distinto: se firma aquí.

### Márgenes de no inferioridad, uno por guardarraíl

Un candidato **falla** si el **UCB95** del daño `(titular − candidato)` **excede `δ`**. No basta
con que el estimado puntual sea negativo, y no basta con que el intervalo cruce el cero.

| guardarraíl | `δ` | justificación |
|---|---:|---|
| `flow_fill_rate` (agregado) | **0,005** | medio punto porcentual de servicio agregado. La reauditoría midió el agregado **plano** (0,796 en todo), así que un candidato que lo mueva medio punto está haciendo algo distinto de redistribuir |
| `lost_orders` | **0,25** órdenes/episodio | una orden cada cuatro episodios. **Cuatro veces** la granularidad Monte Carlo que produjo el halt espurio (0,0625 = 1/16), y muy por debajo de relevancia operativa |
| `backorder_qty_final` | **1,0 %** del backlog del titular | relativo, porque su escala varía por celda |
| recursos programados, masa, capacidad creada | **0,0 exacto** | son identidades **algebraicas**, no cantidades estocásticas: aquí el margen cero sí es legítimo, y es el único sitio donde lo es |

`ret_excel` en cualquiera de sus variantes es **diagnóstico**, nunca guardarraíl: está **medido**
premiando el abandono de un reclamante.

### Potencia, firmada antes de abrir nada

**Preflight obligatorio sobre tapes QUEMADOS**, sin abrir una sola semilla: estimar la SD por
semilla de cada guardarraíl y de `worst_claimant_fill`, y **congelar `N`** tal que

> potencia ≥ **0,90** para declarar no inferioridad frente a `δ`, y ≥ **0,90** para detectar el
> SESOI de +0,010 en el primario, con **corrección simultánea** sobre las celdas.

**Si el preflight dice que la `N` requerida excede el presupuesto, G3c no se ejecuta.** Se registra
`STOP_G3C_UNDERPOWERED` y se dice el número. No se corre con potencia desconocida ni se ajusta
`δ` para que el presupuesto alcance — que es la versión educada de elegir el resultado.

## 5. Escalera de comparadores — el titular es la regla miope

```
mejor constante                      (el nulo viejo, sólo como contexto)
regla miope equivariante             <- EL TITULAR A BATIR
umbral con histéresis                (la respuesta barata al coste de cambio)
política tabular sobre estado discretizado
DP / rollout con horizonte finito    <- si esto basta, no hay sitio para una red
MPC
```

**Si la histéresis o el DP igualan al planificador, el veredicto es
`STRUCTURED_CONTROL_SUFFICES_G3C`** y **no se entrena nada**. Es un desenlace exitoso del
contrato, no una decepción a rescatar.

## 6. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_null_arm_reproduces_current_model` | `min_dwell=1`, `switch_cost=0` deben dar el **payload científico idéntico** al modelo actual. **No** el mismo `self_sha256`: `seal_and_write` mete `created_at` y `calibration_provenance` dentro del cuerpo hasheado, así que exigirlo sería imposible por construcción |
| `f2_coupling_actually_binds` | la permanencia debe reducir de forma medible la frecuencia de conmutación de la regla miope. Si no ata, no hay acoplamiento y G3c no tiene premisa |
| `f3_myopic_rule_is_the_incumbent` | el contraste primario debe ser contra la regla miope, no contra la constante. Reclamar contra la constante sería cobrar dos veces un valor ya medido |
| `f4_uninformed_placebo_fails` | misma cadencia, mismo soporte, sin leer el estado |
| `f5_wrong_claimant_control_destroys_the_gain` | la dirección debe importar, no sólo la cadencia |
| `f6_every_guardrail_has_a_signed_margin` | **el falsador de este documento**: ningún guardarraíl puede evaluarse a margen cero salvo las identidades algebraicas. Si el runner compara a cero, falla |
| `f7_power_was_frozen_before_execution` | `N` y `δ` deben venir del preflight sobre tapes quemados, con su fecha y su artefacto |
| `f8_no_fresh_seeds_without_receipt` | **gobernanza**: sin recibo de Submission A, cualquier semilla fuera de bloques quemados viola `authority_ladder_v1` |
| `f9_no_gain_by_abandonment` | ahora **con margen**: `UCB95(daño) ≤ δ`, no «diferencia puntual no negativa» |

**Mutantes exigidos** (un falsador que nunca se ve fallar no prueba nada): ignorar `min_dwell` →
`f2` debe fallar · comparar contra la constante → `f3` debe fallar · poner `δ = 0` → `f6` debe
fallar.

## 7. Reglas terminales

* **residual sobre la regla miope ≥ SESOI, `LCB95 > 0`, todos los márgenes respetados** →
  `G3C_RESIDUAL_OVER_MYOPIC_RULE`. Autoriza **desarrollo** neuronal — una sola arquitectura,
  elegida por el mecanismo — **nunca una afirmación de prima**, que exige confirmación única en
  universo virgen con multiplicidad controlada.
* **el planificador iguala a la regla miope** → `MYOPIC_RULE_SUFFICES_UNDER_COUPLING`.
* **la histéresis o el DP capturan el residual** → `STRUCTURED_CONTROL_SUFFICES_G3C`.
* **algún margen se viola** → `STOP_G3C_GUARDRAIL`. **Sin segundo rescate**, igual que Program O.

## 8. Lo que este documento NO afirma

No reabre Program O ni Program Q. No dice nada sobre `N ≥ 3`: un fallo de G3c cierra
**el acoplamiento temporal con dos reclamantes bajo este contrato**, y nada más. No hereda el
`±0,01` de Q. Y no convierte la reauditoría en un resultado: **sigue `HALTED`**, es un techo
clarividente sobre tapes quemados, y **no autoriza entrenar nada**.
