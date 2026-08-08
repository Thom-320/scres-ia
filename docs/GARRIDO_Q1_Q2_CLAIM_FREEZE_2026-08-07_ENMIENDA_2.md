# Enmienda 2 a la congelación de claims — los artefactos que el Paper 1 necesita citar

Generada `2026-08-08T01:08:14+00:00` por `scripts/build_claim_freeze_amendment_v1.py`. **No editar la tabla a mano.**

La regla del congelamiento dice que una cifra sin fila no circula. Estos 16 artefactos sostienen el Paper 1 y no estaban listados.

## Los dos hashes, y por qué van los dos

`self_sha256` es el digest del *payload sellado*, calculado **antes** de insertar el propio sello; `file_sha256` es el digest de los **bytes en disco**. Los dos son correctos y contestan preguntas distintas —¿cambió el payload? ¿es éste el fichero que me dieron?— y **no coinciden nunca**. Una revisión de diseño de este repositorio citó los digests de fichero bajo el nombre `self_sha256` para dieciséis artefactos: todos los valores correctos, todas las etiquetas equivocadas. Por eso van los dos, con su nombre.

| § | artefacto | `claim_status` | grado | falsadores | `self_sha256` | `file_sha256` | bloque |
|---|---|---|---|---|---|---|---|
| §3 el abandono | `results/sensitivity/contention_headroom_v1_2/result.json` | `CONTENTION_DOES_NOT_OPEN_THE_DOOR` | DEVELOPMENT | ✅ | `291e8e07233fc0a0…` | `a47e89831fe6f130…` | — |
| §3 replicación | `results/metric_audit/abandonment_v1/result.json` | `COBB_DOUGLAS_SURVIVES_THE_ABANDONMENT_TEST` | DEVELOPMENT | ✅ | `1d37752cbca3649b…` | `48746e7c386782e5…` | — |
| §4 el mecanismo refutado | `results/sensitivity/contention_headroom_v1_3/result.json` | `CONTENTION_DOES_NOT_OPEN_THE_DOOR` | DEVELOPMENT | ✅ | `6730b4d1e65f68d2…` | `60537d88bbc2b1a9…` | — |
| §5 TTR sin estimando | `results/manuscript/h1_h3_v1/result.json` | `HALTED_FALSIFIER_FAILED` | NEGATIVE_OR_HALTED | ⚠️ no todos | `5a01aa2e47aff29c…` | `12abb332afe73c85…` | — |
| §6 la reparación | `results/manuscript/h1_h3_originales_v3/result.json` | `H1_SUPPORTED__H3_NOT_SUPPORTED` | DEVELOPMENT | ✅ | `dc46ce6069755a28…` | `39061791dd37eef4…` | — |
| §7 la fuga | `results/twin_surface_v2/result.json` | `PREFIX_NORMALISER_IS_BLIND_TO_THE_UNRUN_SURFACE` | REPLAY | ✅ | `04b8137157e83a61…` | `cbefeb716d1eda3f…` | — |
| §8 no invariante | `results/monotone_transform_family_v4/result.json` | `A_MONOTONE_RESCALING_SURVIVES_ALL_THREE` | REPLAY | ✅ | `ff3803946fea76fc…` | `e7514c85e7be3141…` | — |
| §9 la cadencia | `results/metric_audit/ret_cadence_corrective_v2/result.json` | `DEVELOPMENT_CORRECTIVE_AUDIT` | UNCONTRACTED | — | `dfe0dc80356a3548…` | `03025ec734721ee8…` | — |
| §9 el defecto | `results/metric_audit/ret_defects_v1/result.json` | `DEVELOPMENT_FINDING_METRIC_DEFECT` | UNCONTRACTED | — | `60065512b08b432c…` | `37208e3c141270a3…` | — |
| §9 las reparaciones | `results/metric_audit/ret_repair_variants_v1/result.json` | `DEVELOPMENT_PREREGISTRATION_INPUT_NO_METRIC_CHANGED` | UNCONTRACTED | — | `aca840e863a0500f…` | `b9ac35565f47cb9a…` | — |
| §10 diagnóstico | `results/determinism_diagnostic/result.json` | `DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT` | DIAGNOSTIC | ✅ | `b8f23013df815add…` | `2126f260a20a9953…` | — |
| §10 reparación | `results/determinism_repair_control/result.json` | `DETERMINISM_REPAIRED_SEED_IS_A_REPLICATION_UNIT_AGAIN` | DIAGNOSTIC | ✅ | `6aea0c525e2f2449…` | `9e988da25ee8c9f5…` | — |
| §12 lo que sobrevive | `results/endpoint_headroom_atlas/result.json` | `NO_ENDPOINT_CARRIES_REGIME_HEADROOM` | REPLAY | ✅ | `86cf97da8a96c9df…` | `14317d68c9bebf49…` | — |
| §2 la demanda | `results/demand_process/result.json` | `DEMAND_PROCESS_CHARACTERISED` | — | ⚠️ no todos | `9711cb366b74f4d8…` | `cb4f88398c4f93a4…` | 8600001-8600012 |
| §5 contexto | `results/surface_gates_v2/result.json` | `NON_SEPARABLE_BUT_CONTEXT_INVARIANT` | REPLAY | ⚠️ no todos | `27244b4ee4505bbc…` | `5abd006f27be0d55…` | — |
| §12 curva de aprendizaje | `results/manuscript/h2_learning_curve/result.json` | `H2_SUPPORTED_LEARNING_CURVE` | DEVELOPMENT | ✅ | `74b75141241ba763…` | `2894e525dc360c8f…` | — |

## Artefactos con falsadores en rojo — se citan CON su fallo, o no se citan

* `results/manuscript/h1_h3_v1/result.json` — `HALTED_FALSIFIER_FAILED`. El fallo es parte del resultado y va en la misma frase que el número.
* `results/demand_process/result.json` — `DEMAND_PROCESS_CHARACTERISED`. El fallo es parte del resultado y va en la misma frase que el número.
* `results/surface_gates_v2/result.json` — `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. El fallo es parte del resultado y va en la misma frase que el número.

## Lo que esta enmienda NO hace

No adjudica nada, no cambia ningún número y no autoriza ninguna afirmación nueva. Sólo admite estos artefactos a la tabla de citables. Las prohibiciones del §7 del congelamiento y de la Enmienda 1 siguen vigentes íntegras.
