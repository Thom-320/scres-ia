# PRERREGISTRO V1 — Brazo secundario: comparador desplegable (MPC/baseline con información del learner)

| Campo | Valor |
|---|---|
| ID | `deployable_comparator_v1` |
| Versión | V1, 2026-08-25 |
| Estado | **LISTO_PARA_FIRMAR** — sin ejecutar |
| Autor | OpenCode (agente), por delegación del PI |
| Firma del PI | ______________________ fecha ________ |
| SHA-256 del fichero al firmar | _(calcular en el momento de la firma; congela este texto)_ |
| Origen del mandato | Decisión del PI 2026-08-25 adoptando el dictamen de ChatGPT Pro (Decisión 4): añadir brazo secundario preregistrado con comparador desplegable |

---

## 0. Rol de este brazo

El flanco más atacable del claim es «empataste con un oráculo, ¿y qué?». Los
comparadores clásicos ganadores por celda en Program Q (`min_cost_flow__2`,
`min_cost_flow__2`, `max_pressure__0`) son miembros de la familia clásica; la
objeción desplegable exige mostrar que el learner también es equivalente —o
superior— a un **baseline implementable con la misma información restringida que
el learner**. Este brazo es **secundario y declarado**: se preregistra ANTES de
correr y **no toca el primario**.

## 1. Regla dura anti p-hacking

**El primario está congelado y no se degrada post-hoc.** El primario del manuscrito
sigue siendo el programa Q ya adjudicado bajo `ret_excel_request_snapshot_v2`
(`docs/ENDPOINT_PRIMARY_DECISION_2026-08-25.md`). Este brazo añade un estimando
secundario declarado; ningún resultado aquí puede redefinir el primario, cambiar su
endpoint, ni relajar sus márgenes. Cambiar el primario a la luz de estos datos sería
exactamente la clase de margin-shopping que la gobernanza prohíbe.

## 2. Diseño

### 2.1 Brazo comparator (dos variantes preregistradas, ambas se corren)

- **V-a MPC de modelo estimado:** belief-MPC cuyo modelo interno se **estima de los
  datos observados** (sin parámetros verdaderos del generador). Nada de oracle
  model, nada de parámetros verdaderos inyectados.
- **V-b misma observación parcial:** baseline greedy/heurístico que recibe
  **exactamente el mismo estado parcial no anticipativo de 21 campos** que el
  learner (contrato Q: `unchanged_contract.observation`; prohibido `tape_id`,
  `seed`, `latent_regime`, `true_rho`, `true_share`, `future_demand`,
  `oracle_calendar`). Sin pre-event offsets negativos (RT3): solo historia
  realizada y estado operacional.

### 2.2 Celdas, tapes y CRN

- Mismas **3 celdas**: `rho75_share90`, `rho90_share75`, `rho90_share90`.
- Evaluación sobre tapes B vírgenes del contrato hermano
  `gate0_split_tape_v1` (bloque `7550193–7550384`), **misma barra CRN**: mismos
  tapes para learner congelado, comparadores clásicos y las dos variantes V-a/V-b.
  Si el Gate-0 no llegara a ejecutarse o su bloque quedara comprometido, este
  contrato abre bloque propio `7550385–7550512` (mismas verificaciones de virginidad).
- Learner: **RecurrentPPO checkpoints históricos congelados** (10 seeds 8101–8110,
  SHA fijado en `program_q_historical_recurrentppo_fallback_freeze_20260717.json`);
  reentrenamiento prohibido.
- Comparadores clásicos: los mismos 10 del contrato Q, intactos.

### 2.3 Estimando secundario declarado

Para cada celda y cada variante v ∈ {V-a, V-b}:

- `D_deploy(v) = mean(ReT_learner) − mean(ReT_v)` sobre tapes B, con IC95 bootstrap
  pareado CRN (unidad: tape; 10.000 remuestras).
- Lectura de equivalencia por ambos lados: el claim desplegable queda acotado si
  `D_deploy(V-a)` y `D_deploy(V-b)` tienen IC95 dentro de ±0,01 (el learner es
  equivalente al desplegable) o positivos (lo supera).

### 2.4 Presupuesto online declarado

- Cómputo: **20–40 CPU-h** total (estimación del paquete, Decisión 4); sin GPUs,
  sin entrenamiento. Si el presupuesto se agota antes de completar las 3 celdas ×
  64 tapes × 2 variantes, se reporta lo completado con N efectivo declarado y se
  marca el resto como no ejecutado — prohibido rellenar con submuestras elegidas a
  posteriori.
- Tiempo de decisión dentro del entorno: idéntico al del learner (misma cadencia
  semanal Discrete(4), 8 decisiones); ninguna variante puede actuar más rápido que
  el learner. La cuestión de cadencia real (dwell/autoridad) permanece en RT2 y
  fuera del alcance de este brazo.

## 3. Falsadores

- **F1 — Fuga de información:** si cualquier variante recibe un campo del conjunto
  prohibido (auditado por diff del vector de observación contra el learner), el run
  completo se invalida.
- **F2 — Comparator straw-man:** si V-a usa parámetros verdaderos o V-b usa
  señales no disponibles para el learner, el brazo se invalida; un comparador débil
  por construcción no prueba nada (regla del repo: un straw-man no responde a
  Gijsbrechts 2022).
- **F3 — Asimetría de tapes:** si algún brazo corre sobre tapes distintos a otro
  (CRN roto), la celda afectada se invalida.
- **F4 — Replay bit-exact:** ≥ 10 % de tapes reproducidos con error > 1e-12 →
  invalidación completa (mismo criterio de custodia que Q).

## 4. Criterio de parada

1. **Parada técnica:** presupuesto agotado (§2.4) o invalidación por falsador →
   se detiene y se reporta el estado exacto, gane o pierda.
2. **Parada interpretativa:** tras completar las 3 celdas:
   - Equivalente-oráculo Y equivalente-o-superior al desplegable (IC95 de
     `Δ_N` dentro de ±0,01 ya adjudicado en Q + IC95 de `D_deploy(v)` dentro de
     ±0,01 o > 0) → el claim de cierre de loop se escribe como «feedback real con
     política aprendida, sin prima neural frente a decision theory estructurada NI
     frente al baseline desplegable».
   - `LCB95(D_deploy(v)) < −0,01` en alguna celda → el learner pierde contra el
     desplegable: se publica tal cual; el manuscrito lo reporta como limitación del
     artefacto aprendido y el valor práctico migra al comparador. **No** se elimina
     del paper.
3. Este brazo **no autoriza ni desautoriza Paper 3**, no toca guardrails de cola
   adjudicados, y no abre ninguna lane de entrenamiento nuevo.

## 5. Trazabilidad de números citados

- Ganadores clásicos por celda (`min_cost_flow__2` / `min_cost_flow__2` /
  `max_pressure__0`) y RecurrentPPO como brazo primario learner:
  `docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md` §Brazos;
  `docs/ENDPOINT_PRIMARY_DECISION_2026-08-25.md` §Correcciones narrativas.
- Observación parcial de 21 campos y lista de información prohibida:
  `contracts/program_q_frozen_policy_replication_v1.json`
  (`unchanged_contract.observation`, `forbidden_information`).
- Coste 20–40 CPU-h y motivación Gijsbrechts 2022 («exige comparador desplegable»):
  `CHATGPT_PRO_PACKAGE/DECISIONES_SOLICITADAS.md` (Decisión 4);
  `CHATGPT_PRO_PACKAGE/RESUMEN_EJECUTIVO.md` (bloqueantes 3–4).
- SESOI ±0,01 y regla de no degradar el primario:
  `CHATGPT_PRO_PACKAGE/DECISIONES_SOLICITADAS.md` (Decisión 4, «Regla dura»);
  `reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md` §1 («Anti p-hacking», línea roja Claude).
