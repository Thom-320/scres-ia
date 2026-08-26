# Abstract — final (2026-08-25)

**Manuscript:** *"Measuring what there is to learn: a falsification-grade evaluation protocol for
learning-based supply-chain-resilience control, and what it finds in a validated
military-food-supply DES"*
**Primary endpoint:** `ret_excel_request_snapshot_v2` (frozen; PI decision
`docs/ENDPOINT_PRIMARY_DECISION_2026-08-25.md`). **Evidence status:** Program Q executed and
adjudicated (`docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md`). Every number below carries
an HTML comment naming its custodied source.

---

## English

Reinforcement learning is routinely proposed to close the learning loop in simulation-based
supply-chain resilience assessment, yet reported gains are usually measured against weak
baselines, without knowing how much value there is to capture. We present a protocol that makes
such claims falsifiable, built on four elements: a perfect-information ceiling measured before any
training, paired with a mechanism placebo that must return exactly zero; a comparator defined as
the maximum over all 65,536 enumerated open-loop production calendars, reselected inside every
bootstrap resample; equivalence declared as an estimand, with a preregistered ±0.01 indifference
zone and stated power; and a per-product equity guardrail permitted to fail. Applied to a
validated full discrete-event model of a military food supply chain, extended with two non-fungible
ration classes contending for shared assembly capacity, the protocol yields a decomposition rather
than a headline. The clairvoyant headroom is 0.15151 (simultaneous LCB95 0.11562) and collapses to
exactly zero under product fungibility. Recurrent policies beat the complete open-loop frontier in
all three operating cells, and a preregistered replication on 256 virgin seeds per cell reproduced
that superiority independently. Against the strongest structured comparators, however,
**no material neural residual is detected within the preregistered ±0.01 zone against the best of
the ten classical configurations evaluated**, and the per-product guardrail failed in every cell:
the learner buys mean fill by unbalancing the weakest product — a substitution invisible to every
scalarised resilience index we examine. Finally, admissible members of the same resilience
construct can return opposite signs on identical tapes, so endpoint, incumbent and tape block must
be declared together for any such comparison to be well posed. Closing the loop is worth much;
making the loop neural is worth nothing measurable here.

## Español

El aprendizaje por refuerzo se propone rutinariamente para cerrar el lazo de aprendizaje en la
evaluación de resiliencia de cadenas de suministro basada en simulación, pero las mejoras
publicadas suelen medirse contra líneas base débiles, sin saber cuánto valor hay por capturar.
Presentamos un protocolo que hace falsables tales afirmaciones, con cuatro elementos: un techo de
información perfecta medido antes de entrenar, junto con un placebo mecanístico que debe devolver
exactamente cero; un comparador definido como el máximo sobre los 65.536 calendarios open-loop
enumerados, re-seleccionado dentro de cada remuestreo bootstrap; la equivalencia declarada como
estimando, con una zona de indiferencia preregistrada de ±0.01 y potencia explícita; y un
guardarraíl de equidad por producto que tiene permitido fallar. Aplicado a un modelo completo de
eventos discretos validado de una cadena de suministro militar de alimentos, extendido con dos
clases de raciones no fungibles que compiten por capacidad compartida de ensamble, el protocolo
produce una descomposición y no un titular. El headroom clarividente es 0.15151 (LCB95 simultáneo
0.11562) y colapsa a exactamente cero bajo fungibilidad de producto. Las políticas recurrentes
superan la frontera open-loop completa en las tres celdas operativas, y una réplica preregistrada
con 256 semillas vírgenes por celda reprodujo esa superioridad de forma independiente. Frente a
los comparadores estructurados más fuertes, en cambio, **no se detecta ningún residuo neural
material dentro de la zona de ±0.01 preregistrada frente a la mejor de las diez configuraciones
clásicas evaluadas**, y el guardarraíl por producto falló en todas las celdas: el aprendiz compra
llenado medio desbalanceando el producto más débil —una sustitución invisible para todo índice de
resiliencia escalarizado que examinamos. Finalmente, miembros admisibles de un mismo constructo de
resiliencia pueden devolver signos opuestos sobre las mismas cintas, de modo que endpoint,
incumbente y bloque de cintas deben declararse juntos para que cualquier comparación de este tipo
esté bien planteada. Cerrar el lazo vale mucho; hacer neural el lazo no vale nada medible aquí.

---

### Fuentes de cada número (inventario de trazabilidad)

| Claim del abstract | Fuente custodiada |
|---|---|
| Techo H_PI = 0.15151, LCB95 0.11562, placebo fungible = 0 exacto | `results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json` vía `papers/paper2/results_table.md` (fila L1) |
<!-- results_table.md@917a217, regenerada hoy -->
| Superioridad recurrente vs. frontera completa, replicada en 3/3 células con 256 semillas vírgenes | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json` (commit congelado `031d0af`; SHA-256 `62f6fd39…`) — `inference.estimates.<cell>::H_OL.lcb95` = +0.0661/+0.0623/+0.1061; `::N` = 256 |
<!-- status canónico: docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md -->
| Claim corregido (±0.01, «ten classical configurations *evaluated*») | `adjudication.json::cell_gates.<cell>.equivalence = true`, `neural_premium = false` (SHA-256 `e13e17f0…`); wording fijado por `docs/ENDPOINT_PRIMARY_DECISION_2026-08-25.md` |
| Guardarraíl por producto fallido en todas las celdas | `evaluation/result.json::guardrail_inference.estimates.<cell>::worst_product_fill::vs_classical.lcb95` = −0.0227/−0.0257/−0.0263; `adjudication.json::integrity_gates.worst_product_fill_noninferior = false` |
| Signos opuestos entre miembros admisibles del constructo sobre las mismas cintas | `results/paper_prep/endpoint_block_inversion_v1/endpoint_block_inversion_v1.md` (familia R2r, inversión dentro de bloque) |
| Brazo primario aprendidor = RecurrentPPO; belief-MPC dentro de la familia clásica | `contracts/program_o_ret_only_learner_v1.json`; ganadores por célula `min_cost_flow__2 / min_cost_flow__2 / max_pressure__0` (`evaluation/result.json::cell_summaries.<cell>.best_classical_config`) |
