# Explore — Repo: /Users/thom/Projects/research/scres-ia.

- **agent_id**: `ac6ccc4993569a1b1`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-07T17:14:44.257Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-ac6ccc4993569a1b1.jsonl`
- **turns**: 166 · **assistant messages**: 3

## Task given

```text
Repo: /Users/thom/Projects/research/scres-ia.

GOAL: map what is stale, superseded, retracted or duplicated, so the repo can be organised without losing anything.

Report with exact paths:

1. `docs/` — list every file, grouped by kind: PREREGISTRO, RESULTADO, ENMIENDA, RETRACTION/RETRACTACION, CERTIFICATE, STANDARD, plan/strategy, other. Flag any document that RETRACTS or SUPERSEDES another (grep for "RETRACT", "SUPERSEDE", "retirad", "obsolet", "no citar", "prohibid"). Produce an explicit list of retracted/prohibited claims and figures.
2. `results/` — find duplicated or superseded artifact families: directories whose names differ only by a suffix (_v1/_v2/_v3, _smoke, _ordered, _replay, _INFLATED, _preflight, _crosscheck). For each family say which one is the current/authoritative artifact (most recent `created_at`, or the one referenced by a RESULTADO doc) and which are superseded.
3. Tests: run `python -m pytest --collect-only -q 2>&1 | tail -5` to count tests, then `python -m pytest -x -q 2>&1 | tail -40` is TOO SLOW — instead just run `python -m pytest -q --timeout=120 2>&1 | tail -30` if a timeout plugin exists, otherwise run `python -m pytest -q -p no:cacheprovider --co -q | tail -3` and separately grep `docs/TRIAJE_SUITE_21_FALLOS_2026-08-06.md` for the failure breakdown. Report the count of failing tests and their names/categories from that triage doc.
4. Scripts: list `scripts/` and flag which runners lack a required `--contract` argument (grep each for `add_argument("--contract"` and `required=True`), and which lack `seal_and_write`. This identifies runners that can produce unsealed artifacts.
5. Repo size: report `du -sh .git results docs scripts supply_chain` and the 10 largest files under `results/`.

OUTPUT: four concrete lists — (a) retracted/prohibited claims and figures, (b) superseded artifact families with the authoritative member named, (c) runners that can emit unsealed or contract-less artifacts, (d) size/bloat hotspots.

Read-only, do not modify or delete anything. Exact paths.
```

## Final answer

# Repo triage map — `/Users/thom/Projects/research/scres-ia`

Read-only. Nothing modified. All paths absolute-relative to the repo root above.

---

## 1 · `docs/` — inventory by kind

`docs/` holds **441 top-level files** + 21 subdirectories (1131 files total, 45 MB). The naming convention is genuinely load-bearing: kind is encoded in the filename prefix.

### Governing / current source-of-truth (read these first)
| path | role |
|---|---|
| `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` | **the authority.** §0 lists the 9 citable artifacts A1–A9 with sha256[:16]; §5 lists retired figures; §7 the claim ladder. Self-declares: *"Sustituye a cualquier resumen anterior"*, and a figure without a row here "no está congelada y no debe circular" |
| `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` | canonical claims table, same date, born of five external audits — *"Si un número no está aquí, no se cita"* |
| `docs/REGISTRO_DE_HUECOS_2026-08-07.md` | open-gaps register (A1–A4, B1–B4, C1) |
| `docs/ESTADO_PARA_REUNION_GARRIDO_2026-08-07.md` | meeting state, 2026-08-07 |
| `docs/REPOSITORY_SOURCE_OF_TRUTH.md`, `docs/PROMISING_LANES_REGISTRY.md`, `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`, `docs/PAPER_FINDINGS_REGISTRY.md` | older registries, still referenced |

### PREREGISTRO (Spanish) — 65 files
`docs/PREREGISTRO_ACTITUDES_DE_RIESGO_CITABLES_2026-08-06.md`, `PREREGISTRO_ASIMETRIA_CUELLO_2026-07-31.md`, `PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md`, `PREREGISTRO_AUTOTOMIA_2026-07-30.md`, `PREREGISTRO_CIERRE_AUTOTOMIA_2026-07-31.md`, `PREREGISTRO_CLAMP_RPJ_2026-07-30.md`, `PREREGISTRO_COMPARADOR_NO_NEURONAL_TRACK_B_2026-08-07.md`, `PREREGISTRO_CONFIRMACION_BACKLOG_2026-07-31.md`, `PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md`, `PREREGISTRO_CONTENCION_HEADROOM_2026-07-31.md`, `PREREGISTRO_CSSU_LIVENESS_2026-08-01.md`, `PREREGISTRO_DELAY_DISTRIBUCION_2026-07-30.md`, `PREREGISTRO_DELAY_FISICO_2026-07-31.md`, `PREREGISTRO_DELTA_SUPUESTO_2026-07-31.md`, `PREREGISTRO_DURACION_R12_R13_2026-07-30.md`, `PREREGISTRO_EMPAREJAMIENTO_ORDEN_STOCK_2026-07-30.md`, `PREREGISTRO_ENLACE_X_ATRIBUCION_2026-07-31.md`, `PREREGISTRO_ESCALERA_V5_MPC_Y_THOMPSON_2026-08-06.md`, `PREREGISTRO_ESPACIO_CONTINUO_2026-07-31.md`, `PREREGISTRO_ESTAR_CAPACIDAD_BARRIDO_2026-08-03.md`, `PREREGISTRO_EXPEDICION_HEADROOM_2026-07-31.md`, `PREREGISTRO_EXPEDICION_HEADROOM_V2_2026-08-01.md`, `PREREGISTRO_FAMILIA_DERIVACIONES_METRICA_2026-08-06.md`, `PREREGISTRO_G1_PRECIO_INVENTARIO_2026-08-01.md`, `PREREGISTRO_G2_UMBRAL_AUTOTOMIA_2026-08-01.md`, `PREREGISTRO_G3_ASIMETRIA_V2_2026-08-01.md`, `PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md`, `PREREGISTRO_G3_OBS_V2_CONFIRMACION_2026-08-02.md`, `PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md`, `PREREGISTRO_G3C_ACOPLAMIENTO_TEMPORAL_2026-08-01.md`, `PREREGISTRO_G3C_ACOPLAMIENTO_TEMPORAL_V2_2026-08-02.md`, `PREREGISTRO_GARRIDO_EXPANDED_DES_ESTAR_V1_2026-08-01.md`, `PREREGISTRO_GARRIDO_EXPANDED_DES_ESTAR_V2_HCOMPUTE_2026-08-05.md`, `PREREGISTRO_GARRIDO_Q2_DES288_V1_2026-08-01.md`, `PREREGISTRO_GARRIDO_V0_RECOVERY_GATE_2026-08-06.md`, `PREREGISTRO_GARRIDO_V0_RECOVERY_SURFACE_V1_2026-08-06.md`, `PREREGISTRO_H1_H3_2026-07-31.md`, `PREREGISTRO_H1_H3_V2_2026-08-01.md`, `PREREGISTRO_H1_H3_ORIGINALES_V3_2026-08-07.md`, `PREREGISTRO_H3_POTENCIA_2026-08-01.md`, `PREREGISTRO_HEADROOM_COBB_DOUGLAS_2026-07-31.md`, `PREREGISTRO_LEAD_TIME_2026-07-30.md`, `PREREGISTRO_META_APRENDIZ_2026-07-31.md`, `PREREGISTRO_META_APRENDIZ_V2_2026-08-01.md`, `PREREGISTRO_METRICA_SERVICE_FIRST_2026-08-01.md`, `PREREGISTRO_METRICA_SERVICE_FIRST_V2_2026-08-01.md`, `PREREGISTRO_MEZCLA_RIESGOS_2026-07-31.md`, `PREREGISTRO_NEURAL_HEADROOM_ENV_V1_2026-08-01.md`, `PREREGISTRO_OATJ_MATERIAL_2026-07-31.md`, `PREREGISTRO_PASO3_GARRIDO_MPC_EXPANDIDO_2026-08-06.md`, `PREREGISTRO_POBLACION_PUNTUADA_2026-07-30.md`, `PREREGISTRO_PRIMA_CD_2026-08-01.md`, `PREREGISTRO_PRIMA_PREDICCION_BUFFER_2026-08-01.md`, `PREREGISTRO_PROPAGACION_RIESGOS_2026-07-30.md`, `PREREGISTRO_REAUDITORIA_CLASE_DE_POLITICA_2026-08-01.md`, `PREREGISTRO_SENSIBILIDAD_HEADROOM_2026-07-31.md`, `PREREGISTRO_SIEMBRA_R0_R14_2026-07-31.md`, `PREREGISTRO_SONDA_REPRODUCIBILIDAD_BAKEOFF_2026-08-07.md`, `PREREGISTRO_SURROGATE_ARCHITECTURE_BAKEOFF_2026-08-07.md`, `PREREGISTRO_TECHO_MONOTONO_COBB_DOUGLAS_2026-08-06.md`, `PREREGISTRO_TECHO_MONOTONO_SUCESOR_LCB_2026-08-06.md`, `PREREGISTRO_TECHO_MONOTONO_V3_PARES_RESOLUBLES_2026-08-06.md`, `PREREGISTRO_TECHO_MONOTONO_V4_BORDE_Y_PISO_2026-08-06.md`, `PREREGISTRO_TURNO_Y_CAPACIDAD_2026-07-31.md`, `PREREGISTRO_VARIANTES_DMLPA_2026-08-07.md`

### PREREGISTRATION (English, older cohort) — 36 files
`docs/L_PROGRAM_PREREGISTRATION_2026-07-10.md`, `PAPER2_BOTTLENECK_MIGRATION_PREREGISTRATION_2026-07-13.md`, `PAPER2_H4_PREREGISTRATION_2026-07-02.md`, `PREVENTIVE_RESERVE_V2_PREREGISTRATION_2026-07-10.md`, `PREVENTIVE_RESERVE_V3_PREREGISTRATION_2026-07-10.md`, `PROGRAM_D_D1_RET_ONLY_TREE_PREREGISTRATION_2026-07-11.md`, `PROGRAM_D_D1_V2_PREREGISTRATION_2026-07-11.md`, `PROGRAM_D_DRA1_CSSU_ALLOCATION_PREREG_2026-07-11.md`, `PROGRAM_D_DRA1_V3_PREREGISTRATION_2026-07-11.md`, `PROGRAM_D_DRA2_FINITE_CONVOY_PREREGISTRATION_2026-07-11.md`, `PROGRAM_D_DRA2B_LONG_HORIZON_PREREGISTRATION_2026-07-12.md`, `PROGRAM_D_LEVER_DISCOVERY_PREREG_2026-07-11.md`, `PROGRAM_E_ORACLE_TO_POLICY_PREREGISTRATION_2026-07-12.md`, `PROGRAM_F_PHASE_DIAGRAM_PREREGISTRATION_2026-07-12.md`, `PROGRAM_G_TERMINAL_METRIC_AUDIT_PREREGISTRATION_2026-07-12.md`, `PROGRAM_H_INFORMATION_SUFFICIENCY_PREREGISTRATION_2026-07-12.md`, `PROGRAM_H_INFORMATION_SUFFICIENCY_PREREGISTRATION_2026-07-13.md` *(dated pair — the 07-13 is the successor)*, `PROGRAM_I_BRANCHING_PREREGISTRATION_2026-07-12.md`, `PROGRAM_I_DECISION_RIGHT_DISCOVERY_PREREGISTRATION_2026-07-12.md`, `PROGRAM_I_HEADROOM_SENSITIVITY_PREREGISTRATION_2026-07-12.md`, `PROGRAM_K2_HOLDING_COST_PREREGISTRATION_2026-07-12.md`, `PROGRAM_K3_RET_BUDGETED_REPLENISHMENT_PREREGISTRATION_2026-07-12.md`, `PROGRAM_O_R_RET_ONLY_LEARNER_PREREGISTRATION_2026-07-17.md`, `PROGRAM_O_STATE_RICH_HOBS_PREREGISTRATION_2026-07-15.md`, `REAL_KAN_10SEED_EXTENSION_PREREGISTRATION_2026-07-03.md`, `REAL_KAN_SIDECAR_PREREGISTRATION_2026-07-03.md`, `RET_REPAIR_PREREGISTRATION_2026-07-29.md`, `TRACK_A_LAST_CHANCE_PREREGISTRATION_2026-07-03.md`, `TRACK_A_LAST_CHANCE_2_GAE_LAMBDA_PREREGISTRATION_2026-07-03.md`, `TRACK_B_ARCHITECTURE_BAKEOFF_PREREGISTRATION_2026-07-03.md`, `TRACK_B_CASE_C_PER_RISK_HEADROOM_PREREG_2026-07-06.md`, `TRACK_B_ENV_HPARAM_TUNING_PREREG_2026-07-05.md`, `TRACK_B_RECURRENT_PPO_HISTORY_SMOKE_PREREGISTRATION_2026-07-03.md`, `TRACK_BP_PREREGISTRATION_2026-07-08.md`, `TRACK_C_PREREGISTRATION_2026-07-10.md`, plus `APERTURA_HOLDOUT_V0_CONTRA_EL_PREREGISTRO_2026-08-07.md`

### RESULTADO — 55 files
`docs/RESULTADO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md`, `RESULTADO_ASIMETRIA_CUELLO_2026-07-31.md`, `RESULTADO_AUDITORIA_FUENTE_H3_2026-08-01.md`, `RESULTADO_AUDITORIA_G3_OBS_F2_2026-08-02.md`, `RESULTADO_AUTOTOMIA_2026-07-30.md`, `RESULTADO_BANDA_EPSILON_2026-07-31.md`, `RESULTADO_CIERRE_AUTOTOMIA_2026-07-31.md`, `RESULTADO_CLAMP_RPJ_2026-07-30.md`, `RESULTADO_CONFIRMACION_BACKLOG_2026-07-31.md`, `RESULTADO_CONTENCION_HEADROOM_2026-07-31.md`, `RESULTADO_DELAY_DISTRIBUCION_2026-07-30.md`, `RESULTADO_DELAY_FISICO_2026-07-31.md`, `RESULTADO_DELTA_SUPUESTO_2026-07-31.md`, `RESULTADO_DURACION_R12_R13_2026-07-30.md`, `RESULTADO_ENLACE_X_ATRIBUCION_2026-07-31.md`, `RESULTADO_ESPACIO_CONTINUO_2026-07-31.md`, **`RESULTADO_ESTAR_CAPACIDAD_2026-08-03.md`**, **`RESULTADO_ESTAR_CAPACIDAD_2026-08-05.md`**, **`RESULTADO_ESTAR_CAPACIDAD_2026-08-05_SUPERSEDED.md`**, **`RESULTADO_ESTAR_CAPACIDAD_V2_REPLAY_2026-08-05.md`**, `RESULTADO_ESTAR_HCOMPUTE_PREFLIGHT_2026-08-05.md`, `RESULTADO_EXPEDICION_HEADROOM_V2_2026-08-01.md`, `RESULTADO_FIG5_BACKPROP_VS_KAN_2026-07-31.md`, `RESULTADO_G1_PRECIO_INVENTARIO_2026-08-01.md`, `RESULTADO_G2_UMBRAL_AUTOTOMIA_2026-08-01.md`, `RESULTADO_G3_OBS_CONVERSION_2026-08-01.md`, `RESULTADO_G3_OBS_V2_2026-08-02.md`, `RESULTADO_G3C_PREFLIGHT_BURNED_2026-08-05.md`, `RESULTADO_G3C_PREFLIGHT_BURNED_V2_2026-08-05.md`, `RESULTADO_G3C_PREFLIGHT_REJILLA_VIVA_2026-08-05.md`, `RESULTADO_H1_H3_2026-08-01.md`, `RESULTADO_H1_H3_V2_2026-08-01.md`, `RESULTADO_H1_H3_ORIGINALES_V3_2026-08-07.md`, `RESULTADO_H3_ADJUDICACION_FUSION_2026-08-02.md`, `RESULTADO_H3_AMBAS_REBANADAS_CONTRATADAS_2026-08-02.md`, `RESULTADO_HEADROOM_COBB_DOUGLAS_2026-07-31.md`, `RESULTADO_META_APRENDIZ_2026-07-31.md`, `RESULTADO_METRICAS_HEADROOM_2026-07-31.md`, `RESULTADO_MEZCLA_RIESGOS_2026-07-31.md`, `RESULTADO_OATJ_MATERIAL_2026-07-31.md`, `RESULTADO_POLITICA_OP12_2026-07-31.md`, `RESULTADO_PORRIESGO_CRN_2026-07-31.md`, `RESULTADO_PORRIESGO_Y_NODO_NUEVO_2026-07-31.md`, `RESULTADO_PRIMA_CD_2026-08-01.md`, `RESULTADO_PRIMA_PREDICCION_BUFFER_2026-08-01.md`, `RESULTADO_REAUDITORIA_CLASE_DE_POLITICA_2026-08-01.md`, `RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md`, `RESULTADO_SENSIBILIDAD_HEADROOM_2026-07-31.md`, `RESULTADO_SERVICE_FIRST_V2_CONTENCION_2026-08-01.md`, `RESULTADO_SIEMBRA_R0_R14_2026-07-31.md`, `RESULTADO_SIJ_Y_NODO_2026-07-31.md`, `RESULTADO_SOLAPAMIENTOS_2026-07-31.md`, `RESULTADO_TURNO_Y_CAPACIDAD_2026-07-31.md`, `RESULTADO_WRAP288_RECONCILIACION_F3_F4_2026-08-02.md`, `RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md`

### ENMIENDA / AMENDMENT / CORRECTION / ERRATA — 38 files
`docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md`, `ENMIENDA_DDMRP_FUERA_DEL_DOMINIO_COMPARTIDO_2026-08-06.md`, `ENMIENDA_ESCALERA_COMPARADORES_2026-08-05.md`, `ENMIENDA_ESCALERA_TRANSFERENCIA_2026-08-05.md`, `ENMIENDA_ESTAR_CAPACIDAD_BARRIDO_V2_REPLAY_2026-08-05.md`, `ENMIENDA_G3C_APERTURA_POST_G3OBS_Y_PREFLIGHT_BURNED_2026-08-05.md`, `ENMIENDA_G3C_BLOQUEADORES_RESUELTOS_2026-08-02.md`, `ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02.md`, `ENMIENDA_G3C_PREFLIGHT_BURNED_2026-08-05.md`, `ENMIENDA_G3C_PREFLIGHT_BURNED_V2_2026-08-05.md`, `ENMIENDA_G3C_RECIBO_MUTACION_HASH_2026-08-02.md`, `ENMIENDA_G3C_REJILLA_DERIVADA_2026-08-05.md`, `ENMIENDA_G3C_TRES_BLOQUEADORES_2026-08-01.md`, `ENMIENDA_G3C_Y_CORRECCIONES_REAUDITORIA_2026-08-01.md`, `ENMIENDA_GARRIDO_V0_RECOVERY_GATE_V2_2026-08-06.md`, `ENMIENDA_GATES_SUPERFICIE_2026-08-05.md`, `ENMIENDA_METRICA_COPRIMARIA_2026-08-06.md`, `ENMIENDA_METRICA_PRIMARIA_ESTAR_2026-08-06.md`, `ENMIENDA_POTENCIA_CONFIRMACION_ORDEN_2026-08-05.md`, `ENMIENDA_REJILLA_EXTENDIDA_4608_2026-08-05.md`, `ENMIENDA_REPORTE_H_REGIME_CURVATURA_DECLARADA_2026-08-06.md`, `ENMIENDA_SELLADO_RETROACTIVO_BAKEOFF_2026-08-07.md`, `ENMIENDA_SURROGATES_Y_EFICIENCIA_2026-08-05.md`, `ENMIENDA_TRANSFERENCIA_REJILLA_2026-08-05.md`, `ENMIENDA_TRANSFERENCIA_REJILLA_ORDEN_CONTRACTUAL_2026-08-05.md`, `AUTOTOMY_PROXY_CORRECTION_2026-07-30.md`, `CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md`, `CVAR_GATE_INSTRUMENT_AUDIT_INTERPRETATION_CORRECTION_2026-07-17.md`, `EPSILON_RANGE_AMENDMENT_2026-07-31.md`, `ERRATA_FIDELITY_REFERENCE_V4_2026-07-31.md`, `MOMENT_SET_CORRECTION_2026-07-30.md`, `PAPER_B_V2_AMENDMENT_2026-07-31.md`, `PROGRAM_D_DRA1_METRIC_AND_FIDELITY_ADDENDUM_2026-07-11.md`, `PROGRAM_G_CHARTER_PREBUILD_AMENDMENT_V1_1.md`, `PROGRAM_G_CHARTER_V1_2_ADDENDUM_2026-07-12.md`, `PROGRAM_H_BOUND_METHOD_AMENDMENT_2026-07-12.md`, `PROGRAM_H_PREFLIGHT_AMENDMENT_2026-07-13.md`, `PROGRAM_K3_STRONG_MPC_AMENDMENT_2026-07-12.md`

### RETRACTION / SUPERSESSION (explicitly named) — 3 files
- `docs/RETRACTACION_POBLACION_PUNTUADA_2026-07-30.md`
- `docs/SUPERSESION_CIENTIFICA_G3C_2026-08-05.md`
- `docs/RESULTADO_ESTAR_CAPACIDAD_2026-08-05_SUPERSEDED.md`

### CERTIFICATE / CONTRACT / FREEZE / CHARTER — 31 files
`docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md`, `PAPER2_PAPER3_BOUNDARY_CERTIFICATE_AND_STRATEGY_2026-07-13.md`, `PROGRAM_O_TERMINAL_OUTCOME_CERTIFICATE_2026-07-15.md`, `PROGRAM_H_CERTIFIED_BOUND_2026-07-12.md`, `GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md`, `PAPER_CLAIM_FREEZE_2026-07-12.md`, `GARRIDO_DES_FREEZE_STATUS_2026-06-26.md`, `GARRIDO_TRACK_A_FRONTIER_FREEZE_2026-06-26.md`, `THESIS_FAITHFUL_ENV_FREEZE_2026-06-26.md`, `PPO_REWARD_FREEZE_2026-06-26.md`, `PROGRAM_F_IMPLEMENTATION_FREEZE_2026-07-12.md`, `PROGRAM_K3_CONFIRMATION_FREEZE_2026-07-12.md`, `PROGRAM_K3_PPO_FREEZE_2026-07-12.md`, `EXPERIMENT_CONTRACT_V2_2026-06-26.md`, `PAPER_CONTRACT_2026-06-24.md`, `GARRIDO_CAUSAL_EXPOSURE_CONTRACT_2026-07-10.md`, `GARRIDO_WRAP_SCRES_AI_CONTRACT_V1.md`, `RET_EXCEL_REQUEST_SNAPSHOT_V2_CONTRACT_2026-07-14.md`, `RET_EXCEL_VISIBLE_V1_CONTRACT_2026-07-11.md`, `CONTRATO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md`, `EXPANDED_CONTRACT_STRUCTURED_COMPARATORS_2026-07-29.md`, `EXPANDED_CONTRACT_COMPARATORS_RECLASSIFICATION_2026-07-29.md`, `SHIFT_ACTION_CONTRACT_AUDIT_2026-06-29.md`, `TRACK_B_CONTRACT_FACTORIAL_VERDICT_2026-07-10.md`, `TRACK_B_SAME_CONTRACT_CHALLENGE_{PROTOCOL,VERDICT}_2026-07-10.md`, `PAPER2_MAINTENANCE_CONTROL_CHARTER_2026-07-12.md`, `PROGRAM_F_RISK_MITIGATION_PORTFOLIO_CHARTER_2026-07-12.md`, `PROGRAM_G_STRUCTURED_SPATIAL_HEADROOM_CHARTER_2026-07-12.md`, `PROGRAM_J_PAPER2_MAINTENANCE_CHARTER_2026-07-12.md`, `PROGRAM_G_CHARTER_{PREBUILD_AMENDMENT_V1_1,V1_2_ADDENDUM_2026-07-12}.md`

### VERDICT / DICTAMEN — 87 files
Dominated by the July cohort (`TRACK_B_*_VERDICT_2026-07-0[2-8].md` ×34, `PROGRAM_[DEFGHIJKL]_*_VERDICT_2026-07-1[1-3].md` ×20, `E1–E4_*_VERDICT_2026-07-02.md`, `REAL_KAN_*_VERDICT`, `TRACK_A_*_VERDICT`, `TRACK_BP_*_VERDICT`, `TRACK_C_ORACLE_PHASE_VERDICT_2026-07-10.md`). **Two same-name dated pairs**: `E3_H104_DENSE_FRONTIER_VERDICT_2026-07-02.md` / `..._2026-07-03.md`, and `PROGRAM_H_NON_RESCUE_BOUNDARY_2026-07-12.md` / `..._2026-07-13.md`.

### AUDIT / TRIAGE — 49 files
Including `docs/TRIAJE_SUITE_21_FALLOS_2026-08-06.md`, `SUITE_FAILURE_TRIAGE_2026-07-31.md`, `PROVENANCE_REPAIR_AUDIT_2026-07-31.md`, `AUDITORIA_SERVICE_FIRST_METRIC_2026-08-01.md`, `NOTA_AUDITORIA_FUENTE_H3_LECTURA_PREVIA_2026-08-01.md`, plus the `docs/audits/` subdir.

### Plan / strategy / registry — ~40 files
`docs/ESTRATEGIA_CIE_2026-08-06.md`, `STRATEGIC_DECISION_CLOSE_CIE_2026-07-10.md`, `TRACK_B_FINAL_CONSOLIDATED_PLAN_2026-07-01.md`, `TRACK_A_EXHAUSTION_PLAN_2026-06-16.md`, `PREVENTIVE_LEARNING_ROADMAP_2026-07-03.md`, `PROMISING_LANES_REGISTRY.md`, `RESEARCH_PROPOSALS_REGISTRY_2026-06-28.md`, `GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md`, `PAPER2_MAINTENANCE_INTERVENTION_LEDGER_2026-07-12.md`, `PROGRAM_G_INTERVENTION_LEDGER_2026-07-12.md`, `REVIEWER_DEFENSE_MATRIX_2026-07.md`, `GARRIDO_REPLICATION_MATRIX_2026-06-15.md`, `LITERATURE_POSITIONING_MATRIX_2026-03-30.md`, `OVERNIGHT_HANDOFF_2026-06-2[67].md`, `OVERNIGHT_STATUS_2026-07-02.md`, `DECISIONES_SIN_GARRIDO_2026-07-30.md`, `PAPER_B_INDEPENDENT_CALIBRATION_DECISIONS_2026-07-30.md`, `THESIS_INTERPRETATION_DECISIONS_2026-06-24.md`, `POST_TRIPLE_STOP_RESEARCH_OPTIONS_2026-07-10.md`, `DONDE_PODEMOS_SER_LAXOS_2026-08-02.md`, `PPO_MLP_RET_SEQ_Q1_Q2_STRATEGY_2026-03-27.md`, `SCRES_DES_RL_PUBLICATION_STRATEGY_INDEPENDENT_ASSESSMENT_2026-07-09.md`, `EXTERNAL_AGENT_PAPER_STRATEGY_PROMPT{,_EN}_2026-07-09.md`.

### Reference / spec / model docs (stable, not experiment records)
`docs/DES_MODEL_DOCUMENTATION.md` (56 KB), `MODEL_SPECIFICATION.md`, `RISK_MODEL.md`, `REWARD_DESIGN.md`, `RESILIENCE_METRICS.md`, `RL_EXTENSION.md`, `FUNCTION_REFERENCE.md`, `PROJECT_DOCUMENTATION.md`, `REPRODUCIBILITY.md`, `environment_spec.md`, `GARRIDO_EXACT_BASELINES.md`, `RET_GARRIDO2024_IMPLEMENTATION.md`, `TRACK_B_MINIMAL_SPEC.md`, `DES_VALIDATION_DOSSIER.md`, `VALIDATION_REPORT.md`, `PAPER_EXPERIMENTAL_CHECKLIST.md`, `DKANA_INTEGRATION_GUIDE.md`, `DKANA_CONTRIBUTOR_HANDOFF.md`, `DAVID_DKANA_THESIS_FAITHFUL_TUTORIAL.md`, `SCRES_BIBLIOGRAPHY_2026-06-28.md`, `CITATION`-adjacent JSON receipts (`PROGRAM_D_DRA2_AUTONOMY_AUTHORIZATION_2026-07-12.json`, `PROGRAM_F_AUTHORITY_RECORD_2026-07-12.json`, `PROGRAM_G_AUTONOMY_AUTHORIZATION_2026-07-12.json`, `SEED_MANIFEST_*_2026-08-01.json`).

### Doc subdirectories (21) — three are a versioned family
`docs/track_b_q1_stats_2026-07-01/` (252 K) → `docs/track_b_q1_stats_2026-07-02/` (248 K) → `docs/track_b_q1_stats_2026-07-02_final/` (4.6 M) → `docs/track_b_q1_stats_2026-07-02_final_10seed/` (272 K). `CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` C1 cites `_final_10seed` and C11 cites `_final`; the two bare-dated ones are superseded. Also: `docs/artifacts/` (20 M — the largest), `docs/garrido_track_b_alternatives_2026-07-03/` (5.2 M), `docs/manuscript_current/` (4.2 M) vs `docs/manuscript_draft/` (20 K) vs `docs/manuscript_notes/` (88 K).

---

## (a) Retracted / prohibited claims and figures — explicit list

Each row is stated in a document; I quote the source path so you can verify.

| # | claim / figure | status | source of the retraction | replacement |
|---|---|---|---|---|
| 1 | **`7,24` / `12,42` / `13,54`** runs-to-optimum | **RETIRADA, prohibited** — came from a normaliser that saw the un-executed surface (leakage) | `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` §5; `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §1; `docs/ESTADO_PARA_REUNION_GARRIDO_2026-08-07.md`; `docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md:196` | prefix panel `7,08 / 12,92 / 12,42` labelled as censored secondary |
| 2 | **`7,90` / `5,43`** (runs saved) | **RETIRADA y prohibida** — oracle normaliser | `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §1 | `5,83` / `5,33`, always labelled censored |
| 3 | **`+6,31`** | prohibited alongside 7,24/12,42/13,54 | `docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md:196` | — |
| 4 | "memory is worth **+0,0515** to UCB1 with LCB95>0" | **RETIRADA** — conflated neuron-vs-`ucb1_transfer` with UCB1's internal transfer-vs-reset; no artifact has a paired CI for it | `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` §5 | arm-mean difference 0,09655 → 0,04502, **without LCB** |
| 5 | Ranking by **"% of ceiling"** of A1 (99,70 `lookahead_kg_transfer`, etc.) | **RETIRADA** — `results/search_ladder_v5/result.json` contains no `percent_of_ceiling` field | ibid. §5 | omit, or cite A5 (`results/surrogate_architecture_bakeoff/`) |
| 6 | **"Q1 and Q2 are closed"** | **prohibited** — exceeds measured scope | ibid. §5 + §7 item 15 | labels `Q1_ANSWERED_WITHIN_TESTED_SIMULATION_OPTIMISATION_CONTRACT` |
| 7 | "the state is on the branch **and on `main`**" | **prohibited** — `main` = `89acc813`, 790 commits behind | ibid. §5 + §7 item 16 | "lives in `codex/expanded-contract-comparators-v2@a13ae22`" |
| 8 | **"KAN is superior" / "KAN is interpretable"** as established claim | **Prohibido** — one partition, no CV, no shape stability across folds | ibid. §3 | `KAN_minus_MLP = −0,475 [−1,548, +0,598]`, no separation |
| 9 | Networks are the family that best imitates SCL | **prohibited to assert** | ibid. §7 item 9 | — |
| 10 | A confirmed neural premium exists | **prohibited** | ibid. §7 item 10; `docs/REGISTRO_DE_HUECOS_2026-08-07.md` C1 | `C1_VIRGIN_BLOCK = NO-GO` |
| 11 | Organizational learning has been demonstrated | **prohibited** | ibid. §7 item 11 | — |
| 12 | The physical chain learns within a campaign | **prohibited** | ibid. §7 item 12 | — |
| 13 | Memory materially improves per-regime adaptation (`H_regime`) | **prohibited** — fails the gate | ibid. §7 item 13 | `H_regime` must always be reported *with its declared curvature* — binding rule in `docs/ENMIENDA_REPORTE_H_REGIME_CURVATURA_DECLARADA_2026-08-06.md` |
| 14 | C1 is confirmed | **prohibited** | ibid. §7 item 14 | — |
| 15 | **Scored-population mismatch / `scored_orders_per_year` collapse 19.9 SD → 0.7 SD** | **RETRACTION** — the canonical ledger already filtered to served orders; the 274.7 figure came from an ad-hoc script. **Change A withdrawn and reverted** | `docs/RETRACTACION_POBLACION_PUNTUADA_2026-07-30.md` (retracts `docs/RPJ_MEAN_DECOMPOSITION_2026-07-30.md` central claim + Change A of `docs/PREREGISTRO_POBLACION_PUNTUADA_2026-07-30.md`) | Change B (`RET_RECOVERY_PERIOD_MODE` disruption→elapsed) survives |
| 16 | Inside #15: the **"0,7 SD"** headline **and** the author's own first correction **"5,33 / 7,14"** | both false; three correction layers | ibid., inline `CORRECCIÓN 2026-07-31` | **~1,7 SD** (1,72 R1r / 1,61 R2r), amendment `contracts/paper_b_v2_amendment_2026-07-31.json` |
| 17 | Inside #15: table figures **`rpj_p95` 249,8** and **`rpj_mean` 19,3** | inflated (ad-hoc script omitting own SE) | ibid., inline `CORRECCIÓN 2026-07-30 (8a6aa16)` | `rpj_p95` **14,6**, `rpj_mean` **11,0** |
| 18 | **Autotomy is unreachable at any delay / "strict partition"** | **RETRACTADO — "el hallazgo central de este documento es falso"** — wrong proxy (`RPj>0` instead of `APj>0`); measured the absence of a set that cannot exist | banner at top of **`docs/PARTITION_HYPOTHESES_2026-07-30.md`** and **`docs/FIDELITY_DELAY_SWEEP_2026-07-30.md`**; superseded by `docs/AUTOTOMY_PROXY_CORRECTION_2026-07-30.md` | autotomy fires at ~115×/236× reference below delay 48, zero above 49 |
| 19 | **memoria vs OFAT = +5,18** and **memoria vs aleatorio = +12,31** | **RETIRADO** — OFAT never received that information; unfair comparison | `docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md:37-38` | rerun v2 (`docs/PREREGISTRO_META_APRENDIZ_V2_2026-08-01.md`) |
| 20 | **`H2` curve +0,00 → +10,00** | **forma retirada como medida** (the zero of the first context survives as a structural check only) | ibid.:39; also `docs/GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md:21` → `RETIRED_LEAKAGE` | — |
| 21 | Lead time ≠ 48 h assumption | **"incorrecta y queda retirada"** — the thesis fixes 48 h explicitly | `docs/DECISIONES_SIN_GARRIDO_2026-07-30.md:4` | — |
| 22 | `docs/PREREGISTRO_LEAD_TIME_2026-07-30.md` — three pieces | **Retirado 2026-07-30**, whole doc `SUPERSEDED_BY_contracts/paper_b_independent_calibration_v2.json` | banner lines 3, 5 | — |
| 23 | **"G3-obs es indecidible para siempre"** | **retirado** | `docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md` §2 | — |
| 24 | **`SEARCH_ENVELOPE_BOUNDARY_CERTIFIED`** (Paper 2) | **queda retractado**; `PAPER3_NOT_AUTHORIZED__DEPENDENCY_PAPER2_LEARNED_VALUE_UNMET`; **K3 remains retracted** for the period-8 confound | `docs/PAPER2_PAPER3_BOUNDARY_CERTIFICATE_AND_STRATEGY_2026-07-13.md:5,18` | — |
| 25 | **Ruta B preventiva** / the preventive gate splice | **retractada; el gate splice era inválido**. Do not cite the 74.1% positive-pair result | `docs/EXTERNAL_AGENT_PAPER_STRATEGY_PROMPT_2026-07-09.md:75`; `docs/TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md:12` | — |
| 26 | **Figura 5 literal** (drivers → ReT as planning task) | **invalidada** — identity / leakage | `docs/GARRIDO_WRAP_ENVIRONMENT_REGISTRY_2026-08-01.md:9` | `results/garrido_fig5_surrogate/` (A4) |
| 27 | **`ReT_tail_v2` improves tail risk over `control_v1`** | **NOT supported at n=15 — "do NOT cite in any form"**; textbook winner's-curse | `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` C23 | keep as documented negative |
| 28 | Superseded positive **severe/h104** and **inflated h52** deltas (+0.000244/+0.000623) | "**never cite**" | ibid. C11 | +0.000209 / +0.000552 (stress-screen), +0.000209 / +0.000542 (dense) |
| 29 | **legacy_validated r3 fidelity numbers** (9/10, 7/10) | "**Do not resurrect**" | ibid. C22 | corrected 8/10, 6/10 |
| 30 | Coarse frontier for C1 | "do not cite coarse frontier" | ibid. C1 | dense CRN table |
| 31 | **PROGRAM_G G5 win headline** | **⚠ SUPERSEDED HEADLINE** — service-loss-proxy only | `docs/PROGRAM_G_G5_FINAL_VERDICT_2026-07-12.md:3` | — |
| 32 | **"la curvatura está por debajo del ruido"** | **sigue retirada** — compared incompatible scales | `docs/RESULTADO_PRIMA_CD_2026-08-01.md:89`; `docs/PREREGISTRO_PRIMA_CD_2026-08-01.md:23` | per-cell mean oracle |
| 33 | **"el presupuesto compra 5,4 puntos"** | **ya retirado** — the variant sweep ran with different hyperparameters (`n_steps` 2048 vs 512, `ent_coef` 0,0 vs 0,01) | `docs/REGISTRO_DE_HUECOS_2026-08-07.md` B2 | 200k re-confirmation |
| 34 | **"igualar hiperparámetros restaura la comparabilidad entre corridas"** | **quedó retirado** | `docs/PREREGISTRO_SONDA_REPRODUCIBILIDAD_BAKEOFF_2026-08-07.md:19` | — |
| 35 | **`PREREGISTRO_CIERRE_AUTOTOMIA` §6 "parámetros libres: ninguno"** | **retractada** — it is one, declared | `docs/PREREGISTRO_CIERRE_AUTOTOMIA_2026-07-31.md:12` | — |
| 36 | DDMRP diagnosis **"corre sobre un número mágico"** | **era falso y queda retractado** | `docs/ENMIENDA_DDMRP_FUERA_DEL_DOMINIO_COMPARTIDO_2026-08-06.md:11` | scale incompatibility |
| 37 | **`R0/R14` seeding "fuga de mínimo global"** | **Retractado** (`04654aa`) | `docs/PREREGISTRO_SIEMBRA_R0_R14_2026-07-31.md:19` | — |
| 38 | **`ret_excel`** as training objective or primary metric | **"sigue prohibido"** — it rewards abandonment | `docs/ENMIENDA_METRICA_COPRIMARIA_2026-08-06.md:78`; `docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md`; still the **silent default** of the step-3 runner (`docs/REGISTRO_DE_HUECOS_2026-08-07.md` B4) | — |
| 39 | **`sum_dk`** as a scalarization | **"escalarización prohibida"** | `docs/PAPER_B_V2_AMENDMENT_2026-07-31.md:63` | — |
| 40 | **Paper 2's exact bound**, as valid over the current simulator | must **not be cited as vigente** until the 29 unclassified simulator fields are classified | `docs/SUITE_FAILURE_TRIAGE_2026-07-31.md` §A | — |
| 41 | **Program S / S1 transducer** | **INVALIDADO 2026-07-18**; the error figure documenting it is itself now stale (1.36e-05 → 4.6e-02) | `docs/SUITE_FAILURE_TRIAGE_2026-07-31.md` §F, :130, :343 | — |
| 42 | **`results/architecture_bakeoff*` as "sealed artifacts"** | **self-retracted**: *"Los he citado repetidamente como «artefactos sellados», y no lo son"* — the field `network_means_from_sealed_artifacts` "afirma algo falso" | `docs/ENMIENDA_SELLADO_RETROACTIVO_BAKEOFF_2026-08-07.md` §1 | development, uncertified provenance |
| 43 | Whole-doc **HISTORICAL, do not treat as active preregistration** | 3 docs carry this banner | `docs/PROGRAM_D_DRA1_CSSU_ALLOCATION_PREREG_2026-07-11.md`, `docs/PROGRAM_D_DRA1_V2_RET_PRIMARY_ADJUSTMENT_2026-07-11.md`, `docs/PROGRAM_D_DRA1_METRIC_AND_FIDELITY_ADDENDUM_2026-07-11.md` | `PROGRAM_D_DRA1_V3_PREREGISTRATION_2026-07-11.md` |
| 44 | Whole-doc **SUPERSEDED** banners | `docs/PPO_EXPERIMENT_PROTOCOL_2026-06-26.md`, `docs/PPO_REWARD_FREEZE_2026-06-26.md` (→ `EXPERIMENT_CONTRACT_V2_2026-06-26.md`); `docs/PAPER_CONTRACT_2026-06-24.md` (partially); `docs/PREVENTIVE_PARETO_RESULTS_2026-06-28.md`; `docs/GARRIDO_RESULTS_STATUS_2026-06-15.md` §"Superseded Results"; `docs/FAMILY_A_DECISION_NOTE.md` | — |
| 45 | **G3-obs V2 §5 terminal rule** (that `STRUCTURED_CONTROL_SUFFICES_G3_OBS` does **not** open G3c) | **superseded** → `G3C_REOPENED_AS_ORTHOGONAL_EXTENSION_PREFLIGHT_ONLY` (preflight on burned tapes only; does not reopen G3-obs, does not authorize training or any virgin seed) | `docs/SUPERSESION_CIENTIFICA_G3C_2026-08-05.md` | — |

---

## (b) Superseded artifact families in `results/` — authoritative member named

Authority basis: the artifact table §0 of `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` (A1–A9), `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`, and `created_at` from each `result.json`.

| family | members (with `created_at`) | **AUTHORITATIVE** | superseded |
|---|---|---|---|
| **search ladder** | `results/search_ladder` 08-05T19:07 · `search_ladder_v2` 19:11 · `search_ladder_ordered` 23:55 · `search_ladder_v3` 23:55 · `search_ladder_v4` 23:56 · `search_ladder_v2_ordered` 23:58 · `search_ladder_v5` 08-06T23:01 | **`results/search_ladder_v5/result.json`** (= A1, `f648a1da5aefaf2f`) | all six others. `search_ladder_v2_ordered` is explicitly `SUPERSEDED_FOR_CURRENT_CLAIMS_BY_A1_AND_A3` / `…BY_V5`. `results/search_ladder` v1 headline `NEURON_BEATS_THE_FULL_CLASSICAL_LADDER` is declared "no defendible" in `docs/ENMIENDA_ESCALERA_TRANSFERENCIA_2026-08-05.md` |
| **grid transfer** | `results/grid_transfer` 08-05T21:14 · `grid_transfer_v2` 23:29 · `grid_transfer_ordered_v1` 08-06T00:11 · `grid_transfer_confirmation_v2` 08-06T09:39 | **`results/grid_transfer_confirmation_v2/result.json`** (= A2, `7bc33823ccd90b5e`, the only `CONFIRMATION` role in the repo) | `grid_transfer_v2` = **`SUPERSEDED_ORDERING`** (wrong `rglob` context order; `docs/ENMIENDA_TRANSFERENCIA_REJILLA_ORDEN_CONTRACTUAL_2026-08-05.md`) — "no se borra y no se cita como confirmación". `grid_transfer`, `grid_transfer_ordered_v1` superseded |
| **normaliser audit** | `results/garrido_normaliser_audit_smoke` 18:32 · `garrido_normaliser_audit` 18:58 · `_v2` 19:31 · `_v3` 08-05T23:18 | **`results/garrido_normaliser_audit_v3/result.json`** (= A3, `fd617753949947e6`) | smoke, base, v2 |
| **surface gates** | `results/surface_gates` 08-05T19:04 · `surface_gates_v2` 19:31 · `surface_gates_extended` 21:01 · `surface_gates_extended_v2` 23:18 | **two live**: `results/surface_gates/result.json` (= A7, `954ac48301ff1234`, reference for A1) and `results/surface_gates_extended_v2/result.json` (4.608-config grid, cited in `docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md:202`) | `surface_gates_v2`, `surface_gates_extended` |
| **twin surface** | `results/twin_surface` 19:29 · `twin_surface_v2` 23:18 | **`results/twin_surface_v2/`** (cited `RESULTADOS_CIE_CONSOLIDADOS:49`) | `twin_surface` |
| **search surrogates** | `results/search_surrogates_smoke` 19:14 · `search_surrogates` 19:16 | **`results/search_surrogates/`** | `_smoke` |
| **monotone transform** | `results/monotone_transform_ceiling` 08-06T14:55 · `_family_v2` 15:50 · `_family_v3` 20:50 · `_family_v4` 20:55 | **`results/monotone_transform_family_v4/result.json`** — origin of the binding rule in `docs/ENMIENDA_REPORTE_H_REGIME_CURVATURA_DECLARADA_2026-08-06.md`, 9/9 falsifiers | ceiling, v2, v3 (v3's `f6` FAILED — see `docs/PREREGISTRO_TECHO_MONOTONO_V4_BORDE_Y_PISO_2026-08-06.md` §1) |
| **architecture bake-off** | `results/architecture_bakeoff` 08-06T04:55 · `architecture_bakeoff_200k` 08-07T07:15 · `surrogate_architecture_bakeoff` 08-07T04:20 | **`results/architecture_bakeoff/`** = A9 (`d641ab0a54ecf2c8`); **`results/surrogate_architecture_bakeoff/`** = A5 (`f96e5b6ff0489932`). `architecture_bakeoff_200k` is the *newest* and feeds the only neural positive but is **not superseded — it is unsealed** (see (c)) | none formally superseded; all three are `DEVELOPMENT_ONLY_NO_VIRGIN_SEEDS_NO_LEARNER_AUTHORISATION` |
| **estar capacity sweep** | `results/headroom/estar_capacity_sweep_INFLATED_H_REGIME` 08-04T01:17 · `estar_capacity_sweep` 08-05T16:51 · `estar_capacity_sweep_v1_1` 17:02 · `estar_capacity_sweep_v2_replay_20260805` 17:05 | **`results/headroom/estar_capacity_sweep_v2_replay_20260805/result.json`** — per `docs/RESULTADO_ESTAR_CAPACIDAD_2026-08-05_SUPERSEDED.md`, the vigente report is `docs/RESULTADO_ESTAR_CAPACIDAD_V2_REPLAY_2026-08-05.md` | `_INFLATED_H_REGIME` (kept deliberately so the correction is auditable — `docs/RESULTADO_ESTAR_CAPACIDAD_2026-08-03.md:40`); `estar_capacity_sweep`; `estar_capacity_sweep_v1_1` (its entry script is **not committed** — that is why 08-05 was superseded) |
| **G3-obs conversion** | `results/headroom/g3_obs_conversion_smoke` · `g3_obs_conversion` 08-01T23:21 · `g3_obs_conversion_v2` 08-02T16:22 · `g3_obs_v2_confirmation_20260802` · `g3_obs_conversion_v2_replay_20260803` 08-03T21:19 | **`results/headroom/g3_obs_conversion_v2_replay_20260803/`** (latest); adjudication in `docs/RESULTADO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md` | smoke, base, and the pre-replay v2 |
| **G3c preflight** | `results/headroom/g3c_preflight_burned` 08-05T17:27 · `g3c_preflight_grid_v2` 17:45 · `g3c_dwell_inertia` (no `created_at`, **unsealed**) | **`results/headroom/g3c_preflight_grid_v2/`** — `docs/ENMIENDA_G3C_PREFLIGHT_BURNED_V2_2026-08-05.md`: "supersede únicamente la grilla y el contrato del preflight v1" | `g3c_preflight_burned` |
| **meta-learner / H3 power** | `results/garrido_meta_learner` 08-01T03:19 · `_smoke_v2` 05:22 · `_thesis90_v2` 05:28 (6.7 M) · `_v2` 05:29 · `_v2_vps_crosscheck` 05:33 · `_h3power_vps` 08-01T20:59 · `_h3power_local` 21:33 · `_h3power_vps_local_replay` 08-02T03:11 · `_h3power_h3_contract_vps_v2` 04:47 · `_h3power_h3_contract_local_v2` 10:10 (2.4 M) | **the contracted pair**: `results/garrido_meta_learner_h3power_h3_contract_local_v2/result.json` (90 replicas, seal `e768e0f69f187272…`) **+** `…_h3power_h3_contract_vps_v2/result.json` (30 replicas) — `docs/RESULTADO_H3_AMBAS_REBANADAS_CONTRATADAS_2026-08-02.md`, re-cited in `docs/PREREGISTRO_H1_H3_ORIGINALES_V3_2026-08-07.md:38-39` | `_h3power_local`, `_h3power_vps` (were `ARTIFACTS_PRESENT_MERGE_PENDING`, `docs/GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md:22`), `_h3power_vps_local_replay` (audit only), `_v2_vps_crosscheck`, `_smoke_v2`. `garrido_meta_learner` v1 = `RETIRED_LEAKAGE` |
| **Q2 DES-288** | `results/garrido_q2_des288_v1` (**no `result.json`**) · `garrido_q2_des288_reconciled_v2` 08-02T18:44 | **`results/garrido_q2_des288_reconciled_v2/`** | `garrido_q2_des288_v1` (empty of a result) |
| **expanded contract comparators** | `results/expanded_contract_comparators` 07-29T16:53 · `_v2_preflight_b840256` · `_v2_preflight_r2_b840256` · `_v2_preflight_1dc40c1` (**all three preflights have no `result.json` and no doc references anywhere in `docs/`**) | `results/expanded_contract_comparators/result.json` | the three `_preflight_*` dirs are **orphans** — 524 K of commit-hash-suffixed scratch |
| **cobb-douglas headroom** | `results/cobb_douglas_component_headroom` 08-06T03:01 (4.6 M) · `_extended` 05:46 (**36 M**) | `results/cobb_douglas_component_headroom_extended/` | base version |
| **contention policy class** | `results/headroom/contention_policy_class_smoke` 22:44 · `contention_policy_class` 22:45 | `results/headroom/contention_policy_class/` | `_smoke` |
| **L-program gates** | `results/headroom/l_program_gate1_2026-07-10` · `_crn_v2` · `_crn_v3`; `l_program_gate2_2026-07-10` · `_crn_v2` · `_crn_v3` | `…_crn_v3_2026-07-10` for both gates | base + `_crn_v2` (the two `_crn_v2`/`_crn_v3` `manifest.json` files are **11.9 MB each**) |
| **garrido v0 recovery gate** | `results/garrido_v0_recovery_gate_v1` (**no `result.json`**) · `_v2` 08-07T01:55 | `results/garrido_v0_recovery_gate_v2/` — `docs/ENMIENDA_GARRIDO_V0_RECOVERY_GATE_V2_2026-08-06.md` | `_v1` |
| **garrido v0 surface gates** | `results/garrido_v0_surface_gates_v1` 08-07T03:34 · `_holdout` 03:47 | `results/garrido_v0_surface_gates_holdout/` — `docs/APERTURA_HOLDOUT_V0_CONTRA_EL_PREREGISTRO_2026-08-07.md` | `_v1` |
| **estar bridge / hcompute** | `results/estar_expanded_bridge_smoke_v1` · `estar_hcompute_preflight_v1` (both 08-05T18:32) | `results/estar_hcompute_preflight_v1/` | `_smoke_v1` |
| **step3 shards** | `results/step3_s1_r1r_a`, `_s2_r1r_b`, `_s3_r2r_a`, `_s4_r2r_b` (4×~105 M, **no `result.json`**) + `results/step3_pooled` 08-07T02:38 | `results/step3_pooled/result.json` is the analysable artifact | the four shard dirs are the raw tapes — 418 MB and **all git-tracked** |
| **`results/repro_probe/`** | untracked, 0 B, contains empty `A/` and `B/` | in-flight reproducibility probe (`docs/PREREGISTRO_SONDA_REPRODUCIBILIDAD_BAKEOFF_2026-08-07.md`) | — |

---

## 3 · Tests

I could not use `--timeout` (no `pytest-timeout` installed; `pytest 9.1.1`), so I ran the full suite scoped to `tests/`. Result:

```
21 failed, 2277 passed, 2 skipped, 2 xfailed, 7 warnings in 666.45s (11:06)
```

This **exactly matches** `docs/TRIAJE_SUITE_21_FALLOS_2026-08-06.md` ("21 fallos, 2.260 pasados"), so the triage doc is current.

### The 21 failing tests, by name and triage category

**Markov-completeness guard / hash-drift on the exact transducer (11)** — triage §3 "pendientes", root cause §A of `docs/SUITE_FAILURE_TRIAGE_2026-07-31.md` (29 unclassified live simulator fields):
- `tests/test_paper2_bottleneck_exact_transducer.py::test_immutable_sim_contract_state_is_bound_into_every_key`
- `…::test_sequential_run_prefix_calls_do_not_create_natural_binding_drift`
- `…::test_full_completed_order_and_risk_histories_are_conservatively_serialized`
- `…::test_small_horizon_primary_transducer_matches_every_brute_calendar`
- `…::test_real_collision_certificate_binds_finite_markov_theorem`
- `…::test_two_fresh_isolated_w2_processes_emit_distinct_bound_receipts`
- `…::test_signed_harness_receipt_normalizes_to_exact_chain_w1`
- `…::test_every_live_simulator_attribute_is_classified_fast`
- `tests/test_paper2_bottleneck_full_frontier.py::test_w6_real_canonical_frontier_matches_unaccelerated_brute_and_replay`
- `tests/test_paper2_bottleneck_full_frontier.py::test_checkpoint_resume_invalidates_on_dependency_hash_drift`
- `tests/test_k3_frontloading_dominance.py::test_direct_cli_needs_no_pythonpath_and_checked_json_is_current`

**Bound-execution harness (6)** — triage §3: `ca84f39` (*"Sweep 37 runners: --contract required everywhere it can be"*) left these out **on purpose**, because they are invoked from tests that don't pass `--contract`:
- `tests/test_paper2_bound_execution_harness.py::test_signed_reduced_w2_session_launches_only_after_retained_ack`
- `…::test_one_tape_w4_smoke_has_heartbeat_partial_status_and_checksums`
- `…::test_checksum_tampering_fails_retrieval_verification`
- `…::test_checksum_manifest_is_anchored_and_covers_status_receipts`
- `…::test_checksum_manifest_structure_and_confinement_fail_closed`
- `…::test_seed_status_tampering_is_checksum_protected`

**Reproducibility-manifest hash drift — flagged in the triage as a PI decision, not a fix (2)**:
- `tests/test_paper2_exhaustive_search_registry.py::test_reproducibility_manifest_hashes_every_listed_artifact_and_source` — `scripts/bound_program_o_affected_orders.py` no longer matches its frozen hash: frozen by `9eb2433`, then edited by `ca84f39` (3 Aug), three lines, `default=DEFAULT_CONTRACT` → `required=True`. The triage is explicit: **"No se arregla actualizando el hash esperado."** The historical invocation without `--contract` is no longer executable.
- `tests/test_paper2_exhaustive_search_registry.py::test_canonical_v2_metric_governance_quarantines_every_visible_v1_claim` (`validation["passed"] is False`)

**File-hash drift (1)**: `tests/test_program_j_request_snapshot_v2_structure_audit.py::test_program_j_structure_audit_is_content_and_source_addressed`

**Seed-custody scanner false positive (1)**: `tests/test_program_q_contract.py::test_live_program_q_seed_custody_declarations_do_not_fake_a_collision` — the scanner cannot tell *declaring* a reserved range from *consuming* a seed (§D of the July triage). This is the same family of defect that let a real custody failure hide in the noise.

### A finding the triage does not cover — bare `pytest` is unusable

`python -m pytest --collect-only -q` from the repo root gives:
```
1784 tests collected, 616 errors in 187.33s
Interrupted: 616 errors during collection
```
`python -m pytest tests/ --collect-only -q` gives `2302 tests collected in 12.06s` — clean.

Cause: there is no `testpaths` setting, so pytest walks `outputs/`, which contains **14 full nested copies of the repo** (e.g. `outputs/experiments/track_b_adaptive_confirm_v9_2026-07-01/fetched_manual_complete/scres-ia/`). 442 of the errors are basename collisions inside `outputs/`; the other 174 are `ModuleNotFoundError: No module named 'scripts.…'` in real `tests/` files, caused by the nested copies' `scripts` package shadowing the real one in `sys.modules`. Anyone running bare `pytest` sees an interrupted, meaningless run.

---

## (c) Runners that can emit unsealed or contract-less artifacts

`scripts/` holds **567 `.py` files**, of which **229 are `run_*.py`**. Aggregate: **171 runners have no `--contract` argument at all**; **161 have no sealing call** (`seal_and_write` / `seal_run` / `seal_slice`); **141 have neither**. `seal_and_write` is defined once, at `supply_chain/arm_runner.py:159`.

The old bulk is mostly frozen/terminal work. The list below is what actually matters — runners touched in August 2026, i.e. the active campaign:

**Tier 1 — no `--contract` AND no sealing (highest risk):**
- **`scripts/run_architecture_bakeoff_v1.py`** (mtime 2026-08-06). This is the one that matters. It produces `results/architecture_bakeoff/result.json` **and** `results/architecture_bakeoff_200k/result.json`, and `scripts/run_track_b_nonneural_v1.py` reads the latter to build a field literally named `network_means_from_sealed_artifacts` — feeding the project's **only neural positive**. `docs/ENMIENDA_SELLADO_RETROACTIVO_BAKEOFF_2026-08-07.md` §1 admits the artifacts have no `self_sha256`, no `contract_path`, no `contract_sha256`, no `calibration_provenance`, and that the field name "afirma algo falso". The retroactive `sealed_record.json` fixes content-addressing from 2026-08-07 onward but **cannot certify provenance**. A commitment to add `--contract` (required, no default) + `seal_and_write` is recorded in §4, pending the reproducibility probe closing.

**Tier 2 — `--contract` present but NOT `required=True` (a default silently applies):**
- `scripts/run_garrido_v0_surface_gates_v1.py` (2026-08-06)
- `scripts/run_track_bp_gate2_screen.py` (2026-08-03) — *also* lacks sealing
- `scripts/run_dra2_static_frontier.py`
- `scripts/run_expanded_contract_comparators_v2.py`
- `scripts/run_family_lane_rl.py`
- `scripts/run_preventive_reserve_v2_gate.py`

**Tier 3 — `--contract` required but no sealing call (August-touched):**
`scripts/run_estar_hcompute_preflight.py`, `scripts/run_estar_bridge_smoke.py` (both 08-05); `scripts/run_rpj_onset_admission_arms.py`, `scripts/run_program_i_branching.py`, `scripts/run_procurement_delay_reading_arms.py`, `scripts/run_garrido_risk_headroom_sensitivity.py`, `scripts/run_fulfillment_delay_distribution_arms.py`, `scripts/run_fidelity_delay_sweep.py`, `scripts/run_autotomy_arms.py` (all 08-03).

**Tier 4 — sealing present but no `--contract` at all (08-01 cohort, pre-`ca84f39` sweep):**
`scripts/run_service_first_v2_audit.py`, `run_h1_h3_v2.py`, `run_h1_h3_recovery_volatility_v1.py`, `run_g2_autotomy_threshold.py`, `run_g1_buffer_price_cobb_douglas.py`, `run_expedite_headroom_v2.py`, `run_des_change_differential_probe.py`, `run_cssu_liveness_scope.py`, `run_cssu_liveness_gate.py`, `run_contention_service_first_v2.py`, `run_cd_surface_prediction_premium.py`, `run_buffer_profile_prediction_premium.py`.

**Verification — artifacts on disk actually missing `self_sha256` (6 total):**
```
results/architecture_bakeoff/result.json
results/architecture_bakeoff_200k/result.json
results/garrido_risk_headroom_sensitivity_v1/result.json
results/garrido_wrap_q2_smoke_2016h/result.json
results/garrido_wrap_source_audit/result.json
results/headroom/g3c_dwell_inertia/result.json
```
Every other `result.json` under `results/*/` and `results/headroom/*/` carries a seal. So the leak is narrow and the four Tier-3/Tier-4 runners above mostly seal via a different path — but the two bake-off artifacts are load-bearing.

**Related, from `docs/REGISTRO_DE_HUECOS_2026-08-07.md` B4:** `ret_excel` is still the **silent default endpoint** of the step-3 runner, and `ret_excel` is a metric that rewards abandoning a product. The register calls it "una trampa esperando al siguiente que lo invoque" and proposes making it mandatory "como se hizo con `--contract`". `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` §6 lists this as precondition **C1-B**.

---

## (d) Size / bloat hotspots

```
du -sh .git results docs scripts supply_chain
696M    .git
2.4G    results
 45M    docs
 21M    scripts
3.6M    supply_chain
```

**But the largest thing in the repo is not in that list:**
```
8.6G    outputs        ← gitignored, but on disk, and it breaks bare pytest
2.4G    results
154M    tmp            ← gitignored
 99M    kaggle
 45M    docs
 21M    scripts
7.4M    deliverables
4.2M    research
3.6M    supply_chain
2.9M    archive_untracked
1.5M    scres_ia_manuscript_overleaf.zip  (a zip, at repo root)
436K    thesis.txt
164K    notebookffc7b2c5ff.ipynb + Untitled90.ipynb  (two stray notebooks at root)
```

### 10 largest files under `results/`
```
50,9 MB  results/step3_s1_r1r_a/full/traces.json
50,9 MB  results/step3_s2_r1r_b/full/traces.json
50,7 MB  results/step3_s4_r2r_b/full/traces.json
50,7 MB  results/step3_s3_r2r_a/full/traces.json
48,2 MB  results/paper2_bound_harness/w16_179f7c2/transport/source.bundle
48,2 MB  results/paper2_bound_harness/w12_179f7c2/transport/source.bundle
48,2 MB  results/paper2_bound_harness/w24_0a1e3be/transport/source.bundle
48,2 MB  results/paper2_bound_harness/w16_0a1e3be/transport/source.bundle
48,2 MB  results/paper2_bound_harness/w12_0a1e3be/transport/source.bundle
36,1 MB  results/cobb_douglas_component_headroom_extended/aggregates.json
```

### Largest `results/` directories
```
931M  results/surface_cache/           (913 files git-tracked)
623M  results/program_o/               (1306 files git-tracked; Program O is CLOSED with terminal verdict)
294M  results/paper2_bound_harness/    (44 files tracked)
105M  results/step3_s2_r1r_b/   105M  results/step3_s1_r1r_a/
104M  results/step3_s4_r2r_b/   104M  results/step3_s3_r2r_a/
 56M  results/program_d/
 43M  results/headroom/
 36M  results/cobb_douglas_component_headroom_extended/
```

### The hotspots, ranked

1. **`outputs/` — 8.6 GB, 71% of the working tree.** Gitignored (only 18 files tracked), so it is not in `.git`, but it contains **14 nested full copies of `scres-ia`** and is the sole reason bare `pytest` reports 616 collection errors. Prime candidate for archival off-repo.
2. **`results/surface_cache/` — 931 MB, 913 files *git-tracked*.** The single biggest contributor to the 696 MB `.git`. Cache data under version control.
3. **`results/program_o/` — 623 MB, 1306 files tracked**, for a program that is closed with a terminal verdict (`docs/PROGRAM_O_TERMINAL_OUTCOME_CERTIFICATE_2026-07-15.md`). Includes a single 16.8 MB `watcher_state.jsonl`.
4. **`results/paper2_bound_harness/` — 294 MB**, of which ~241 MB is five near-identical `source.bundle` git bundles (48.2 MB each, only two distinct commit hashes: `179f7c2`, `0a1e3be`) — plus a 25 MB `switch_complexity_screen/9672e21_local_20260713_sieve1/result.json`. `.gitignore` already carves out `switch_complexity_screen_v2/` as "force-add only after audit"; the v1 dir was not.
5. **`results/step3_s{1..4}_*/` — 418 MB of raw tapes, all tracked**, four ~51 MB `traces.json` plus ~8.7 MB per-seed shards. Only `results/step3_pooled/result.json` (12 K) is the analysable artifact — and `docs/REGISTRO_DE_HUECOS_2026-08-07.md` A1 says the step-3 run must be **re-run anyway** because `worst_product_fill` was never persisted.
6. **`.git` — 696 MB (587 MB packed).** Driven by items 2–5; 100 MB of loose objects (1686) suggests `git gc` has not run recently.
7. **`results/headroom/l_program_gate{1,2}_crn_v{2,3}_2026-07-10/manifest.json` — 11.9 MB each**, and `_crn_v2` is superseded by `_crn_v3`.
8. **`docs/artifacts/` — 20 MB**, the largest doc subdir, and `.gitignore` already excludes only `docs/artifacts/control_reward/*_smoke/`.
9. **`docs/track_b_q1_stats_2026-07-02_final/` — 4.6 MB**, sitting beside three sibling versions totalling another 772 KB.
10. **Root clutter**: `scres_ia_manuscript_overleaf.zip` (1.5 MB), `notebookffc7b2c5ff.ipynb` (89 KB), `Untitled90.ipynb` (72 KB), `thesis.txt` (436 KB), `.DS_Store` (12 KB, tracked-adjacent), plus `tmp/` (154 MB) and `archive_untracked/` (2.9 MB).

---

## One thing worth flagging before any reorganisation

Two orthogonal custody rules are recorded in the corpus and would be violated by a naive cleanup:

- **`docs/DONDE_PODEMOS_SER_LAXOS_2026-08-02.md:56`** — *"No editar artefactos fechados en sitio. Se supersede con banner."* And `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` §8: a correction is issued as a **successor file**, never an in-place edit.
- **`docs/GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md:30`** — *"Un resultado retirado se conserva y se etiqueta; no se borra."* Confirmed in practice by `results/headroom/estar_capacity_sweep_INFLATED_H_REGIME/`, kept deliberately *so the correction stays auditable*, and by `SUPERSEDED_ORDERING` on `results/grid_transfer_v2/` — *"no se borra y no se cita como confirmación."*

So the safe reorganisation is **label and relocate, never delete**: move superseded families under a `superseded/` or `historical/` prefix with the pointer to the authoritative member, rather than pruning them. The only genuinely deletable candidates I found are the three `results/expanded_contract_comparators_v2_preflight_*` dirs (no `result.json`, zero references anywhere in `docs/`), and `outputs/`'s nested repo copies — and even those I'd archive rather than remove, since the July triage's lesson was precisely that provenance breaks silently.
