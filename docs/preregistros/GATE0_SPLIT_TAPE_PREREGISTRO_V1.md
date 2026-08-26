# PRERREGISTRO V1 — Gate-0 corregido split-tape (selección y evaluación en tapes disjuntos)

| Campo | Valor |
|---|---|
| ID | `gate0_split_tape_v1` |
| Versión | V1, 2026-08-25 |
| Estado | **LISTO_PARA_FIRMAR** — sin ejecutar; ningún byte de este diseño ha corridro |
| Autor | OpenCode (agente), por delegación del PI |
| Firma del PI | ______________________ fecha ________ |
| SHA-256 del fichero al firmar | _(calcular en el momento de la firma; congela este texto)_ |
| Origen del mandato | Decisión del PI 2026-08-25 adoptando el dictamen de ChatGPT Pro (Decisión 3): ejecutar Gate-0 corregido split-tape antes del manuscrito |

---

## 0. Naturaleza de este preregistro

Este experimento es **prospectivo y nuevo**: no re-adjudica, re-margea ni re-siembra
Program O, Program O-R ni Program Q (veredictos históricos inmutables; ver
`scres-ia-expanded-v2/docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md`).
Mide una cantidad distinta: cuánto headroom físico sobre la frontera open-loop
completa sobrevive cuando **la selección del ganador y su evaluación ocurren en
cintas disjuntas**.

## 1. Antecedente y motivo

La propuesta original (`reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md`) definía
`G_PI = mean_t[max_calendar] − max_c[mean ReT]` sobre la frontera completa de
4^8 = **65.536 calendarios**, con 128 tapes nuevos por celda. La auditoría
`pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md` (2026-08-24)
verificó que ese `G_PI` **nunca fue ejecutado** y que, si el máximo se elige
mirando los mismos tapes con los que luego se promedia, el estimando implícito es
`E_t[max_k X_{t,k}]`, no `E_B[X_{B,k*(A)}]`. Por la desigualdad `E[max] ≥ max E`,
el diseño sin split puede inflar el headroom por selección; el tamaño del sesgo no
está medido. Este preregistro implementa el diseño corregido y convierte el propio
sesgo en resultado citable.

## 2. Diseño experimental

### 2.1 Celdas y entorno

- Mismas **3 celdas** del contrato Q/O-R: `rho75_share90`, `rho90_share75`,
  `rho90_share90` (fuente: `contracts/program_q_frozen_policy_replication_v1.json`,
  campo `unchanged_contract.cells`).
- Física full-DES sin cambios; medición exclusivamente a través del pipeline
  (`arm_runner.py`); prohibido cualquier script ad-hoc de medición.
- Endpoint de evaluación: `ret_excel_request_snapshot_v2` (primario congelado,
  `docs/ENDPOINT_PRIMARY_DECISION_2026-08-25.md`). Sin cambios de endpoint.

### 2.2 Tapes A (selección)

- **64 tapes vírgenes por celda** (192 en total).
- Se evalúan los **65.536 calendarios** sobre cada tape A.
- Selección por celda: `k*(A) = argmax_k mean_A(X_{A,k})`.
- `k*(A)` se **congela por SHA-256 en un artefacto fechado** antes de tocar un solo
  tape B. Prohibido re-elegir después de ver B.

### 2.3 Tapes B (evaluación congelada)

- **64 tapes vírgenes por celda**, disjuntos de A (192 en total).
- Se evalúa únicamente `X_{B,k*(A)}` más los **10 comparadores clásicos**
  (misma barra: CRN, mismos tapes para todos los brazos).
- Ningún miembro del equipo ve resultados de B hasta que el artefacto de congelación
  de `k*(A)` esté firmado y hasheado.

### 2.4 Estimandos

| Estimando | Definición | Rol |
|---|---|---|
| `G_PI_naive` | `mean_t[max_k X_{t,k}] − max_c[mean ReT_c]` sobre la muestra completa (diseño original, selección y evaluación sobre las mismas cintas) | **solo diagnóstico**, nunca gate |
| `G_PI_split` | `mean_B(X_{B,k*(A)}) − max_c mean_B(X_{B,c})` | **estimando de gate** |
| `Δ_bias` | `G_PI_naive − G_PI_split`, por celda | resultado metodológico citable: cuantifica cuánto habría inflado la oportunidad un gate sin split |

### 2.5 Inferencia

- **Unidad de bootstrap: el tape** (cinta semanal completa), estratificado por celda;
  remuestreo con reemplazo, **10.000 remuestras**, intervalo percentual.
- El comparador clásico se **reselecciona dentro de cada remuestra**
  (`reselect_classical_10_inside_every_resample=true`, misma regla que Q); esta
  reselección forma parte del estimando y **no se altera tras ver datos**.
- `UCB95(·)` = límite superior unilateral del intervalo bootstrap al 95 %.

### 2.6 Regla de decisión (SESOI = 0,01)

1. Si `UCB95(G_PI_split) < 0,01` en **alguna** celda → **cerrar la lane** sin
   entrenar nada. El claim de manuscrito será «sin headroom físico detectable bajo
   este contrato».
2. Si `Δ_bias ≥ 0,01` en alguna celda → documentar explícitamente que el diseño sin
   split habría sobreestimado la oportunidad en al menos ese margen.
3. Si `G_PI_split` supera el SESOI con su incertidumbre en todas las celdas → eso
   **solo autoriza estudiar aprendizaje** (smoke bajo contrato nuevo con su propio
   preregistro). **No** autoriza afirmar prima neural ni promocionar Paper 3.

## 3. Falsadores (escritos antes de correr, con su porqué)

- **F1 — Inestabilidad de ranking:** si `k*(A)` cae fuera del decil superior del
  ranking en B, la «oportunidad» era ruido de selección. Puede fallar porque la
  varianza entre tapes (σ_seed ≈ 0,032 documentada en el paquete) domina diferencias
  pequeñas entre calendarios.
- **F2 — Sesgo domina:** si `G_PI_naive` es grande pero el LCB95 de `G_PI_split`
  es < 0, la oportunidad aparente era íntegramente sesgo de selección. La lane se
  cierra igualmente por la regla 2.6.1.
- **F3 — Placebo de instrumento:** un calendario elegido **al azar uniforme** entre
  los 65.536 debe dar `G ≈ 0`; si no, el instrumento mide algo distinto de headroom y
  todo el run se invalida antes de mirar `k*`.
- **F4 — Replay bit-exact:** reproducir ≥ 10 % de los tapes (muestreados antes de
  correr) contra el pipeline; error > 1e-12 invalida el run completo. Motivo: los
  veredictos previos del repositorio exigen replay físico como condición de custodia.

## 4. Semillas vírgenes propuestas

Verificadas hoy contra `research/paper2_exhaustive_search/program_q_s_seed_registry_v1.json`
y `program_s_seed_manifest_v1_1.json` (semántica de intervalos enteros cerrados):

| Bloque | Estado en registros | Este preregistro |
|---|---|---|
| `7480101–7480148` | SEALED_FOREVER (Program O-R confirmation) | **NO TOCAR** |
| `7490001–7490256` | CONSUMIDO por Program Q el 2026-07-18 | **NO TOCAR** |
| `7510001–7510012` (S1 Morris) | RESERVED_UNOPENED | **NO TOCAR** |
| `7510101–7510148` (S2 observable) | RESERVED_UNOPENED | **NO TOCAR** |
| `7510201–7510248` (S3 full DES) | RESERVED_UNOPENED | **NO TOCAR** |
| `7510301–7510348` (S4 calibration) | RESERVED_UNOPENED | **NO TOCAR** |
| `751100001–751350000` (S4 training) | RESERVED_UNOPENED | **NO TOCAR** |
| `7520001–7520256` (S4 confirmation) | RESERVED_UNOPENED | **NO TOCAR** |
| `7530001–7539999` (Paper3) | SEALED_UNAUTHORIZED | **NO TOCAR** |
| `949100001+` / `950100001+` (sandbox / blind qualification) | desarrollo / sellado fuera de Q | **NO TOCAR** |

**Bloque solicitado: `7550001–7550512`** (512 enteros consecutivos), asignación:

- `7550001–7550192`: tapes A (64 × 3 celdas, orden de celdas fijado en el mapa de
  asignación que se congela junto a este fichero);
- `7550193–7550384`: tapes B (64 × 3 celdas);
- `7550385–7550412`: placebos, replays F4 e instrumento;

Justificación de virginidad: el registro canónico no contiene ninguna reserva ni
consumo estructurado en `7.500.001–7.551.999` (los únicos rangos estructurados son
los tabulados arriba; los valores sueltos tipo `75xxxxx` hallados en cachés de
resultados son subcadenas de dígitos de métricas en punto flotante, no semillas).
El bloque continúa la convención del repositorio (siguiente bloque 75x libre tras
Paper3) y exige **scan de colisiones repository-wide antes de abrir** (regla
`collision_scan_required_before_each_stage` del manifest de Program S).

## 5. Presupuesto y calendario

- Coste estimado (heredado del paquete, Decisión 3): **~6 CPU-h**, ~3 días.
- Sin entrenamiento de ningún tipo. Sin GPUs. Sin reentrenar checkpoints.

## 6. Compromiso de publicación

Se publica **gane o pierda**: tanto `G_PI_split` como `G_PI_naive` y `Δ_bias` entran
al manuscrito (o a su suplemento) pase lo que pase. Un resultado negativo aquí es
parte del aparato de medición con capacidad de falsación que es la contribución
declarada del paper. Prohibido: cambiar el estimando de gate tras ver B, relajar el
SESOI, eliminar falsadores, o abrir semillas fuera del bloque asignado.

## 7. Trazabilidad de números citados

- 65.536 calendarios (4^8), definición de `G_PI`, 128 tapes/celda originales:
  `reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md` (§2).
- Diseño split A/B, estimandos `G_PI_naive`/`G_PI_split`, regla UCB95<0,01,
  Δ_bias≥0,01, 64/64 opcional: `pdfs_frontier/context_reports/AUDITORIA_GATE0_SPLIT_TAPES.md`.
- 3 celdas: `contracts/program_q_frozen_policy_replication_v1.json`.
- Endpoint primario: `docs/ENDPOINT_PRIMARY_DECISION_2026-08-25.md`.
- σ_seed ≈ 0,032: `reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md` (§1, análisis de potencia).
- ~6 CPU-h: `CHATGPT_PRO_PACKAGE/DECISIONES_SOLICITADAS.md` (Decisión 3).
- Rangos de semillas ocupados/reservados: `program_q_s_seed_registry_v1.json`,
  `program_s_seed_manifest_v1.json`, `program_s_seed_manifest_v1_1.json`,
  `docs/SUITE_FAILURE_TRIAGE_2026-07-31.md` §D.
