# Enmienda 3 — el censo de confirmaciones, y por qué cinco auditorías lo contaron mal

**Predecesores:** `GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` (`550a253`), `…_ENMIENDA_1.md`
(`d7a205b`), `…_ENMIENDA_2.md` (generada por `scripts/build_claim_freeze_amendment_v1.py`).
**Motivo:** cuatro auditorías externas afirmaron «hay dos confirmaciones, no tres». Verificado
contra los artefactos: **el número que proponen es utilizable, su enumeración no lo es**, y la causa
raíz es un defecto de metadatos que hará fallar también al próximo censo.

**Filas superseded:** ninguna. Esta enmienda **añade** el censo y dos reglas.

---

## E1 · El censo, definitivo

Tres artefactos tienen grado de confirmación. **No dos.**

| # | artefacto | grado | `sha256[:16]` fichero | frontera |
|---|---|---|---|---|
| **C1** | `results/grid_transfer_confirmation_v2/result.json` | `run_role: CONFIRMATION`, `scope: CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK` | `7bc33823ccd90b5e` | `GRID_TRANSFER_CONFIRMED__UCB1` — sólo UCB1 supera el replay marginal |
| **C2** | `results/garrido_h2_h3_confirmation_v1/result.json` | `global_confirmation_pass: true`, recibo `COMPLETE_VALID_CONFIRMATION_AGGREGATE` | `bc375d3021b64d10` (recibo `d4305bcf6bf5209d`) | *«does not establish learner, feedback, or architectural value»* — **literal del artefacto** |
| **C3** | `results/gsa_confirmation/result.json` | `run_role: CONFIRMATION`, `scope: CONFIRMATION_ON_REPURPOSED_VIRGIN_BLOCK` | `1f487d91900e2ea4` | **superseded** ↓ |
| | `results/gsa_confirmation_corrective/result.json` | `CORRECTIVE_REANALYSIS_OF_A_CONFIRMATION` | `5e393b64b8ab950a` | `GSA_CONFIRMED_ON_VIRGIN_BLOCK_AS_A_ONE_BIT_CALENDAR_CHOICE` |

### La formulación autorizada

> **Dos confirmaciones prospectivas utilizables por el manuscrito de retención, más una tercera
> (GSA) que existe, corrió sobre un bloque virgen *reutilizado*, y quedó degradada por corrección
> propia a una conclusión de un bit.**

**Prohibido** decir «tres confirmaciones» (infla el grado) y **prohibido** decir «sólo hay dos»
(omite un artefacto sellado que existe). Las cuatro auditorías llegaron al número operativo correcto
por un camino equivocado: ninguna mencionó C3.

### Por qué C3 se degradó, en sus propias palabras

```
why: "f4 demanded an unsatisfiable property and f6 tested the data instead of the
      estimator; both were my specification errors"
```

Un falsador que exigía una propiedad insatisfacible **no podía pasar** — es la regla **R6** de la
Enmienda 1, encontrada de forma independiente y antes de que se escribiera. Y el corrective
mantiene `θ` y las semillas idénticas (`f1`), por lo que no es un experimento nuevo sobre un bloque
quemado.

---

## E2 · La causa raíz: el grado de C2 no es descubrible por máquina

`results/garrido_h2_h3_confirmation_v1/result.json` **no tiene** `run_role`, ni `scope`, ni
`claim_status`, ni `self_sha256`. Sus claves top-level son:

```
claim_boundary, code_commit, confirmation_roots_opened, confirmation_tape_roots,
contract_sha256, created_at, development_roots_opened, freeze_receipt_sha256,
global_confirmation_pass, holm, neutral_shift_checks, panel_gates, results,
row_count, schema_version, status
```

Su condición de confirmación vive en `completion_receipt.json`, un fichero aparte.

**Consecuencia:** cualquier censo que enumere por `grep '"run_role": *"CONFIRMATION"'` —que es lo que
hace el instinto— **encuentra C1 y C3 y pierde C2**. Un censo que enumere por memoria encuentra C1 y
C2 y pierde C3. Ningún método ingenuo produce las tres. Eso explica exactamente el patrón de error
de las cuatro auditorías, y explica por qué el conteo ha cambiado tres veces esta semana sin que
nadie mintiera.

### R9 · El grado de evidencia debe ser descubrible en el propio artefacto

> Todo `result.json` lleva `run_role`, `scope`, `claim_status` y `self_sha256` **en su nivel
> superior**. Un grado que sólo existe en un fichero hermano no es citable como grado: es una nota.
> El registro de evidencia enumera por esos cuatro campos, y un artefacto sin ellos se reporta como
> `GRADE_NOT_MACHINE_DISCOVERABLE` en lugar de omitirse en silencio.

C2 no se re-ejecuta para arreglar esto —está sellado y su recibo cierra la cadena— pero **se cita
siempre con su recibo al lado**, y la próxima confirmación nace con los cuatro campos.

---

## E3 · R8 · No vinculante no es no reportado

La Decisión 1 del PI (`DECISION_PI_ENDPOINT_Y_APERTURA_PROGRAM_L_2026-08-07.md`) fija la resiliencia
media como endpoint primario, sin piso vinculante por peor producto. Tres auditorías señalaron el
riesgo de que eso contradiga al manuscrito de instrumentos.

El registro de decisión **ya pagó la mitad del precio**: prohibió `ret_excel` como endpoint, porque
el piso y una métrica que censura pedidos omitidos no pueden caer los dos. Esta enmienda paga la
otra mitad:

```
PRIMARIO                    resiliencia media
GUARDARRAÍLES OBLIGATORIOS  lost_orders · peor producto/CSSU · cola (CVaR)
                            REPORTE OBLIGATORIO, decisión NO vinculante
```

> **R8 — un guardarraíl que deja de ser vinculante no deja de ser reportado.** El PI decide qué
> adjudica; nadie decide qué se oculta. Un manuscrito que demuestra que los promedios esconden
> resultados no puede publicar promedios sin su cola.

Esto **no** revierte la Decisión 1: la media sigue decidiendo. Y no rescata nada pasado — R4 intacta.

---

## E4 · Lo que esta enmienda deja congelado para el manuscrito de retención

| elemento | veredicto |
|---|---|
| conteo de confirmaciones | **la formulación de E1, literal** |
| C2 en el manuscrito | siempre con su frontera *«does not establish learner, feedback, or architectural value»* |
| C3 (GSA) | **no entra** en la espina de ningún manuscrito; se declara en el censo |
| H1–H4 originales | **fuera de la espina**, a apéndice de reconciliación |
| claim central | *State retention ranked above memoryless search during development, but prospective transfer was carrier-specific: factor-level UCB1 outperformed both cold start and state-blind marginal replay, whereas the neural carrier did not.* |
| alcance de demanda | declarado: `within the thesis-inherited U(2400,2600) demand process`, con el **CV semanal medido de 7,1 %** (`results/demand_process/`, `cb4f88398c4f93a4`) — no «variación mínima» |
| distancias entre ramas | **prohibidas en prosa**; se fija SHA |
| bloqueantes de escritura | ninguno: ni KAN-latente, ni AIC/Ramsey, ni R2 aleatorio, ni el motor estacional |

## Custodia

Datado, no se edita en sitio. Sucesor: `…_ENMIENDA_4.md`. Reglas vigentes acumuladas: **R1–R7**
(Enmienda 1), **R8** y **R9** (esta).
