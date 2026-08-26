# PROGRAM_Q_CANONICAL_EVIDENCE_STATUS — 2026-08-25

**Decisión del PI (2026-08-25):** adoptar el dictamen ejecutivo de ChatGPT Pro.
Este documento es el **único artefacto canónico** que resuelve la contradicción
«Q cerrado vs Q sin abrir». Cualquier manuscrito, resumen o informe debe citar
este fichero para el estado evidencial de Program Q.

## Veredicto canónico

Program Q **SÍ fue ejecutado y adjudicado**, con N=256 por celda, semillas
vírgenes `7490001–7490256`, el 2026-07-18. La evidencia física existe, está
custodiada y verifica bit a bit:

| Artefacto | SHA-256 verificado hoy |
|---|---|
| `evaluation/result.json` | `62f6fd390471624f7c301b8baa96d31871db99e22dd5a22d6bb8cf7bba8088b2` ✓ |
| `adjudication.json` | `e13e17f001a1d24f86f00257e145c26f9c09def68ef7b2ee2f90fcb23148b0e9` ✓ |
| direct audit | `3da52ca129707e883be0179f82be8058d29ddf454c27a4f578918c26c7ec82eb` (registrado) |

Ubicación canónica: commit científico congelado
**`031d0af9479fcf73e95f34cece9a0ea76a218c97`** (rama
`codex/submission-a-program-q`), bajo
`results/program_q/confirmation_v1_20260718/artifacts/confirmation/`.
Custodia pesada (829 MiB) inmutable en `ovh-agent-lab`; copia slim reviewable
en el mismo árbol.

La confusión «bloque virgen» venía de mezclar dos registros: el
*registry* de reserva (`program_q_s_seed_registry_v1.json`, escrito ANTES de la
ejecución con estado `RESERVED_UNOPENED`) y la *ejecución* posterior que consumió
exactamente ese rango. La reserva precede al consumo; no lo contradice.

## Veredicto adjudicado (texto exacto del artefacto)

```text
STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION   paper3_authorized=false
```

Ese rótulo compuesto NO significa que los endpoints fallaran. Significa que el
gate compuesto preregistrado falló en su guardrail. Descomposición verificada:

| Gate | Resultado |
|---|---|
| H_OL (learner > 65.536 open-loop) | **PASS 3/3** (+0.0795/+0.0726/+0.1172; LCB95 simultáneos +0.0661/+0.0623/+0.1061) |
| Equivalencia Δ_N ⊂ ±0.01 vs mejor clásico | **PASS 3/3** (puntos −0.00159/−0.00072/−0.00041) |
| Prima neural ≥ +0.01 | **FAIL 3/3** |
| Semillas learner positivas | 10/10 en las tres celdas |
| Tapes favorables | 84.77% / 89.84% / 95.70% |
| Guardrail worst_product_fill ≥ −0.02 | **FAIL 3/3** (LCB95 −0.02266/−0.02566/−0.02632) |

Integridad: 768/768 shards verificados, 789/789 ficheros del manifest,
21,696 replays full-DES independientes con 0 fallos, error máximo de replay
`7.77e-16`.

## Lenguaje PERMITIDO en el manuscrito

- «In the evaluated contract, recurrent feedback control independently replicated
  state-dependent ReT superiority over every one of the 65,536 open-loop
  calendars (N=256 virgin seeds per cell, joint power 0.876).»
- «The learner was statistically equivalent to the best of ten classical
  feedback controllers within the preregistered ±0.01 indifference zone.»
- «No material neural premium was detected against the ten classical
  configurations evaluated.»
- «The per-product equity guardrail failed in all three cells.»

## Lenguaje PROHIBIDO

- «Program Q no fue ejecutado» / «solo existe el artefacto de potencia».
- «El bloque sigue sellado / sin abrir» (la ejecución lo consumió).
- Cualquier cuantificador universal sobre controladores no evaluados:
  prohibido «no hay residuo neural frente a cualquier controlador posible».
- «Tail-safe», «deployment-safe», «Paper 3 autorizado».
- Re-adjudicar, re-margear o re-sembrar Q.

## Clasificación editorial

Los números H_OL, Δ_N y el fallo de cola son **confirmación preregistrada
independiente** dentro del contrato Q, con su potencia declarada. H_PI=0.15151
(LCB95 0.11562, placebo fungible = 0 exacto) permanece como **medición de techo
de calibración** sobre cintas distintas: demuestra que el entorno tiene
oportunidad material y que el instrumento detecta el mecanismo; no sustituye el
análisis de potencia de Δ_N.

## Estado del endpoint

Primario de este manuscrito: **`ret_excel_request_snapshot_v2`** (congelado;
todos los niveles 1–4 fueron calculados bajo él). `ret_excel_full_ledger`,
`ret_excel_clipped_0_1` y demás miembros de la familia: sensibilidad declarada,
incluyendo la tabla completa de inversión de signo endpoint×bloque. Full ledger
queda candidato prospectivo para un contrato futuro. Ver
`ENDPOINT_PRIMARY_DECISION_2026-08-25.md`.

## Brazos

Brazo primario learner: **RecurrentPPO** (10 checkpoints históricos congelados).
belief-MPC pertenece a la **familia clásica** (los ganadores por celda fueron
`min_cost_flow__2`, `min_cost_flow__2`, `max_pressure__0`). Documentos que digan
lo contrario quedan corregidos por este fichero.

---
*Firmado como decisión fechada del PI, 2026-08-25. Commit de referencia: este.*
