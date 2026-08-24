# Nota de diseño — ladder clásico Vía B (exploratorio)

**Fecha:** 2026-08-07
**Estado:** análisis exploratorio; no es un prerregistro confirmatorio.

## Propósito

El contrato B existente compara learners con un incumbente estático congelado y con
el máximo estático in-sample. Para evitar que un learner sea comparado sólo contra
un baseline débil, esta nota añade un screen de políticas clásicas observables en
las mismas 24 cintas de validation ya quemadas.

## Diseño congelado para este screen

- No se abren semillas DES.
- Unidad emparejada: tape de validation `7400097–7400120`.
- Misma física: `fixed_clock_physical_v1`, `rho=0,75`, `share=0,90`, scheduler primario.
- Endpoint principal del screen: `ret_service_full_clipped_v1`.
- Secundarios: `ret_excel_clipped_0_1`, `ret_excel_full_ledger`, `ret_thesis`,
  `flow_fill_rate`, `delivered_rations`, `lost_orders`, `unresolved_orders`,
  `terminal_stock`, `worst_product_fill`, `actual_payload`.
- Comparadores: incumbente estático congelado, incumbente estático in-sample y la
  familia finita de `finite_state_rich_configurations()`.
- La familia incluye `base_stock`, `max_pressure`, `min_cost_flow`, `belief_mpc`
  y `belief_dp` con los parámetros enumerados por el módulo; no hay tuning sobre
  validation.
- Intervalos: bootstrap de 20.000 remuestras, LCB unilateral al percentil 5,
  idéntico al evaluador de learners.

## Límite

El screen no cambia el gate B ni autoriza PPO/KAN. Sólo responde si el learner,
cuando termine la evaluación, está siendo contrastado con una referencia clásica
observable suficientemente fuerte. El resultado se guarda en
`results/program_b_classical_baselines_exploratory_v1/result.json` y lo ejecuta
`scripts/evaluate_program_b_classical_baselines.py`.
