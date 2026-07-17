# Guardrail taxonomy audit — does the ReT-only construct reopen any program? (2026-07-17)

**PI directive:** Garrido's construct is canonical ReT; CVaR was our added standard. Question:
which terminal closures fail ONLY on added standards (and therefore admit a ReT-centred rereading),
and which fail on the construct itself or on causal-identification guards?

## The three guardrail classes (binding rule going forward)

- **Class A — canonical construct failure.** The headroom/advantage on canonical ReT itself was
  immaterial or absent. Nothing to renegotiate: the construct is Garrido's.
- **Class B — identification guards.** Resource equality, placebos, anti-shed/spatial fairness,
  action-trajectory audits. These do not express preferences; they protect the CAUSAL meaning of
  "adaptive value" (without them, "improving ReT" can be bought with extra transport or by
  starving a theatre — the documented root causes of every retraction in this program).
  **Non-negotiable under any construct reading.**
- **Class C — deployability preferences.** Standards beyond the source construct (tail-risk /
  CVaR non-inferiority). Legitimate to include, but their membership in the ACCEPTANCE rule is a
  domain-owner decision → negotiable only with a written, dated Garrido sign-off (question M2),
  and only prospectively (new contract), never retroactively.

## Audit table (every row traceable to a populated verdict artifact)

| Programa | Razón terminal (artefacto) | Clase | ¿Reabre bajo ReT-only? |
|---|---|---|---|
| **Program O — corrective validation** | Mean canonical ReT **PASS** all 3 cells (LCB95 +0.043/+0.059/+0.066; 42/44/46 de 48; 27/27 placebos; 1,451 replays 0 fallos) — falla SOLO `ret_visible_cvar10` joint LCB95 −0.0086/−0.0155 con puntos +0.035/+0.020 (`results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json`) | **C** | **SÍ — la única reapertura.** Vía: M2 sign-off → adjudicación del auditor → contrato learner nuevo. El STOP del contrato congelado NO se reescribe. |
| Program O — label-only controller | Ventaja coincidente con +388 vehicle-hours / ~20k raciones extra (recursos no conservados) | B | No — confound causal, no preferencia |
| Program I GP region (≡ ancla op11) | `qualifies_new_lane:false` por worst-CSSU fill −0.13 vs −0.02 y attended −0.09..−0.28 (`results/headroom_gsa/oos_guardrail_check.json`) | B | No — reabrirlo legitima shed-to-win (el "win" agregado mata de hambre un teatro). Solo un M2 que NIEGUE pisos de servicio por teatro lo movería a clase C — improbable y no recomendado |
| D1 rationing (§6.5.4 cap-60) | Valor estado-dependiente ≈ +0.0011 EN ReT canónico (`results/program_d/d1_branching/verdict.json`: `STOP_NO_STATE_DEPENDENT_RATIONING_HEADROOM`; frontier `d1_v3_visible_frontier`) | A | No. Nota de paper: el hallazgo ESTÁTICO `spt_flat` (+0.0105) falló además `service_loss_5pct_ci_positive` — criterio clase-C menor; citable como limitación, no reabre nada adaptativo |
| Program G | `G_RETEXCEL_NO_WIN` (`results/program_g/retexcel/verdict.json`): el learner G5 ganó OOS en la métrica estilizada (`g5/verdict.json`) pero NO en ReT canónico; `terminal_metric_audit`: `STOP_PROGRAM_G_NO_ROBUST_ADAPTIVE_VALUE_UNDER_STYLIZED_CONTRACT` | A | No — falló exactamente en el constructo de Garrido |
| DRA1 | `STOP_NO_DYNAMIC_ORACLE_HEADROOM` (`results/program_d/dra1_exact_branching/verdict.json`) | A | No |
| DRA2b | `STOP_DRA2B_PRE_TREE_GATE` (`results/program_d/dra2b_long_horizon_smoke/verdict.json`) + H_obs≈0 en canónico | A | No |
| L(e-1) | `STOP_NO_DEPLOYABLE_ADAPTIVE_HEADROOM` (`results/headroom/l_program_gate2_*/verdict.json`) | A | No |
| Paper2 maintenance | `STOP_NO_OBSERVABLE_MAINTENANCE_HEADROOM` (screen_period4/6) | A | No |
| Paper2 bottleneck | `STOP_NO_ADAPTIVE_BOTTLENECK_VALUE` (`results/paper2_bottleneck/locked_confirmation/verdict.json`) | A | No |
| Program F | `STOP_PROGRAM_F_SCREEN` | A | No |
| Garrido risk screen | max H_profile_safe 6.9e-05 vs 0.01 EN canónico (result e4a3d4a0) | A | No |
| Track C / Track B-P / H / M / K→K2 | Cierres de magnitud/conversión en el canónico (certificado de agotamiento, `quantitative_ceilings`) | A | No |
| op11 fair probe | `OP11_FAIR_CONVERSION_DEVELOPMENT_NO_GO`: candidatos fair NEGATIVOS en el propio ret_order | A (la familia fair pierde en la métrica) | No |
| Wartime atlas + GSA | `STOP_COMPUTE_INFEASIBLE` — no es guardrail | — | Sin cambio |

**Grep completo de `cvar` en contracts/ y results/:** aparece como guardrail en muchos contratos,
pero como RAZÓN DE CIERRE únicamente en la validación correctiva de Program O. **Conclusión:
exactamente una reapertura candidata — Program O.**

## Optional paper addition

Retrospective CVaR panels are computable on burned tapes (Program O cells, Program G, D1) as a
SECONDARY cross-program tail-risk section. If informative, good addition; if not, nothing is lost
— the acceptance rule under the source construct is canonical ReT (pending M2).

## What this audit does NOT do

It does not reweaken any Class-B guard, does not alter `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`
(whose `no_post_failure_changes` covers thresholds and guardrails), and does not authorize any
learner. The gate to a learner runs: M2 written sign-off → CVaR instrument audit
(`cvar_gate_instrument_audit_v1`) → independent-auditor adjudication → new frozen contract on
fresh sealed tapes.
