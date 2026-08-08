# Paper 2 seasonal sensitivity runbook

This is the only new scientific execution authorised by the consolidated plan. It is bounded
development sensitivity and cannot alter the RQ2 confirmation grade.

## Order

1. Finish and seal `results/frozen_path_equivalence_v2/result.json`.
2. Run the amended characterisation with the explicit, already-used development seed list:

```sh
.venv/bin/python scripts/characterise_seasonal_demand_engine_v1.py \
  --seed0 8600001 --seeds 12 \
  --contract docs/ENMIENDA_DEMANDA_ESTACIONAL_P2_2026-08-08.md
```

3. If the characterisation remains partial, report the literal generator numbers and stop the
   robustness interpretation; do not retune.
4. If a sensitivity surface is needed, build only from explicitly supplied replay seeds:

```sh
.venv/bin/python scripts/build_seasonal_sensitivity_cache_v1.py \
  --seed 8600001 --seed 8600002 --replay-of EXISTING_REGISTRY_BLOCK_ID \
  --shard 0 --of 1
```

The command intentionally has no default seed block. Before any large surface is built, the PI must
bind the supplied seeds to an existing registry entry. Cache slices must then be sealed with
`scripts/seal_garrido_surface_cache_v1.py` using the amendment and the seasonal result as reference.

5. Run the outer-loop comparator with `--development-sensitivity`; never pass `--confirmation`:

```sh
.venv/bin/python scripts/run_grid_transfer_v1.py \
  --base-cache results/surface_cache/seasonal_sensitivity_v1/base \
  --ext-cache results/surface_cache/seasonal_sensitivity_v1/ext \
  --development-sensitivity \
  --contract docs/ENMIENDA_DEMANDA_ESTACIONAL_P2_2026-08-08.md \
  --reference results/demand_seasonal_engine/result.json \
  --output results/seasonal_sensitivity_transfer/result.json
```

The hard wall is 72 hours. A timeout or incomplete cache is a limitation, not a reason to change
the estimand. The main manuscript remains valid with its scope limited to the inherited demand
process.
