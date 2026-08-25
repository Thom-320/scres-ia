# Sensibilidad de endpoint para cierre de Paper 2

Estado: análisis descriptivo de preparación; no re-adjudica O, O-R ni Q y no modifica contratos congelados.

## Recomendación

El endpoint primario recomendado para el manuscrito es `ret_excel_full_ledger`. Es el ReT alineado con el ledger completo y conserva la misma dirección negativa en los cuatro bloques mostrados. El primario histórico de la confirmación correctiva sigue siendo `ret_excel_clipped_0_1`, tal como fija `contracts/ret_metric_repair_confirmation_v1.json::primary_endpoint`; la recomendación de este informe es prospectiva para el manuscrito y no cambia esa adjudicación.

La convención en todas las celdas es `brazo − incumbente` sobre la misma tape. En la confirmación se usan los IC pareados ya publicados en `results/metric_audit/ret_metric_repair_confirmation_v1/result.json::families.{R1r,R2r}.comparisons.<endpoint>.ci95`. En Step 3 se reconstruye el bloque desde las cuatro fuentes `rows.json` declaradas en `results/step3_pooled/result.json::shards`; el IC es el bootstrap percentil pareado de 5000 remuestras con semilla 20260806, exactamente el procedimiento de `scripts/merge_step3_shards.py::N_BOOT` y `main.rng`.

## Tabla endpoint × bloque

Los números de la tabla son vistas redondeadas de los campos `estimate`/`ci95` del JSON que acompaña este informe. Cada celda conserva allí su ruta y campo fuente. `C` es la confirmación correctiva; `S3` es `replay_mpc_v2` frente a la mejor postura estática pooled de su familia. Los diagnósticos oracle/greedy y DDMRP no se presentan como controladores convertidores.

| Endpoint | C-R1r (16 tapes) | C-R2r (16 tapes) | S3-R1r (12 tapes) | S3-R2r (12 tapes) |
|---|---:|---:|---:|---:|
| `delivered_rations` | +3756.3125 [634.1875, 7964.8125] | −25399.0625 [−29344.4594, −21173.5484] | +3947.9167 [826.6417, 7639.5750] | −3149.8333 [−8207.4167, 216.9167] |
| `flow_fill_rate` | +0.00002016 [0.00000345, 0.00004371] | +0.00233963 [−0.00144914, 0.00625994] | +0.00002076 [0.00000373, 0.00003999] | −0.00147207 [−0.00464265, 0.00052841] |
| `lost_orders` | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| `ret_excel` | −0.00001954 [−0.00004956, −0.00000017] | +0.01251565 [0.00900389, 0.01595500] | −0.00002165 [−0.00004689, 0.00000424] | −0.00016751 [−0.00308789, 0.00214861] |
| `ret_excel_clipped_0_1` | −0.00001954 [−0.00004940, −0.00000021] | +0.01247474 [0.00910860, 0.01590910] | −0.00002165 [−0.00004689, 0.00000424] | −0.00016751 [−0.00308789, 0.00214861] |
| `ret_excel_full_ledger` **← recomendado** | −0.00001938 [−0.00004863, −0.00000007] | −0.00448348 [−0.00660046, −0.00238802] | −0.00002138 [−0.00004636, 0.00000431] | −0.00099085 [−0.00370060, 0.00068258] |
| `ret_excel_quantity_time_clipped_0_1` | −0.00001965 [−0.00004863, −0.00000029] | +0.01238668 [0.00924427, 0.01525866] | −0.00002165 [−0.00004689, 0.00000424] | −0.00016504 [−0.00308629, 0.00215375] |
| `ret_thesis` | −0.00001938 [−0.00004833, −0.00000027] | +0.00037000 [−0.00108445, 0.00173061] | −0.00002138 [−0.00004636, 0.00000431] | −0.00000571 [−0.00003294, 0.00002001] |
| `strategic_injected` | −88133.25 [−107522.49, −64976.60] | −99072.25 [−112528.84, −87077.25] | −89327.50 [−103840.05, −67992.86] | −111784.75 [−133362.53, −87610.59] |
| `terminal_stock` | −112037.1875 [−130459.7297, −97796.3281] | −13446.0 [−29312.2938, 13532.8937] | −105072.4167 [−144736.7229, −64326.4292] | −17427.0 [−23915.4583, −12165.5271] |
| `unresolved` | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |

## Lectura y límites

- `ret_excel`, `ret_excel_clipped_0_1`, `ret_excel_quantity_time_clipped_0_1` y `ret_thesis` cambian de signo entre bloques/fuentes; no son una base estable para escoger un primario único. El campo `sign_inversion_across_blocks` de cada endpoint en `results/paper_prep/endpoint_sensitivity.json::table` deja esa decisión explícita.
- `ret_excel_full_ledger` no cambia de signo en esta matriz y es negativo en ambos bloques de Step 3; por eso se recomienda como endpoint principal, sin convertir esa recomendación en una nueva adjudicación.
- Step 3 aplicó `flow_fill_rate`, no el guardrail preregistrado `worst_product_fill`: `results/step3_pooled/result.json::service_metric_applied` y `::service_guardrail_deviation`. Por tanto, esta tabla no declara seguridad por producto.
- No existe literalmente `results/step3_pooled/rows.json`. El JSON lo registra como ausente en `results/paper_prep/endpoint_sensitivity.json::pooled_rows_resolution`; no se creó un archivo fuente ficticio. El pooled se recompone sólo con `results/step3_s1_r1r_a/full/rows.json`, `results/step3_s2_r1r_b/full/rows.json`, `results/step3_s3_r2r_a/full/rows.json` y `results/step3_s4_r2r_b/full/rows.json`, que son los cuatro shards consignados en `results/step3_pooled/result.json::shards`.
- Se mantienen los estados ya publicados: `results/metric_audit/ret_metric_repair_confirmation_v1/result.json::families.R1r.verdict`, `::families.R2r.verdict` y `results/step3_pooled/result.json::claim_status`.

El detalle machine-readable, incluidos los `source` path+campo por celda, está en [`endpoint_sensitivity.json`](endpoint_sensitivity.json).
