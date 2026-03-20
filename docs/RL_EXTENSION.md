# Reinforcement Learning Extension (SCRES+IA)

This section documents the Gymnasium-compatible RL wrapper that extends Garrido's static DES model.

## Action Space (Strategy Mapping)
The thesis explores resilience via two static buffering strategies (Strategy I & II). The RL agent extends this into continuous, dynamic control.

- **`assembly_shifts` (S):** Maps to Thesis Strategy II. Agent controls active shifts (1, 2, or 3).
- **`op3_q`, `op9_q_max`, `op9_q_min`:** Maps to Thesis Strategy I (Inventory buffering). Agent dynamically adjusts dispatch quantities.
- **`op3_rop`, `op9_rop`:** Extension beyond the thesis, allowing dynamic reorder points to counter emerging risks.

## Observation Space (State Vector)
The baseline 15-dimensional state vector tracks:
1. `raw_material_wdc` (Op3 upstream)
2. `raw_material_al` (Op5 upstream)
3. `rations_al` (Op7 buffer)
4. `rations_sb` (Op9 buffer)
5. `rations_cssu` (Op11 buffer)
6. `rations_theatre` (Op13)
7. `fill_rate`
8. `backorder_rate`
9. `assembly_line_down` (Binary: Op5/6/7)
10. `any_loc_down` (Binary: Op4/8/10/12)
11. `op9_down` (Binary)
12. `op11_down` (Binary)
13. `time_fraction` (Progress through simulation)
14. `pending_batch / batch_size`
15. `contingent_demand_pending / 2600`

*Note: Future v4 observation spaces will add `rations_sb_dispatch` and `assembly_shifts_active` to fully represent the system state.*

## Reward Function
The reward aligns the agent's objective with the thesis ReT metric, heavily penalizing backorders and throughput failures while rewarding sustained fill rates.
