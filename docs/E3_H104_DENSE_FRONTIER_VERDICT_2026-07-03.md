# E3 h104 Per-Cell Dense Frontier Verdict (2026-07-03)

## Verdict

The per-cell dense frontier follow-up strengthens the Paper 1 E3 claim for the
two canonical-horizon transfer cells. Frozen Track B PPO remains positive
against the best in-cell 147-policy dense downstream-dispatch static frontier in
both `current/h104` and `increased/h104`.

Artifact:
`outputs/experiments/track_b_e3_h104_per_cell_dense_frontier_2026-07-02/`

## Results

| Cell | Best dense static | PPO ReT | Static ReT | Delta | Seed-clustered CI95 | Seeds positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current/h104 | `S3_op10_2.00_op12_1.50` | 0.005648 | 0.005405 | +0.000244 | [+0.000209, +0.000278] | 5/5 |
| increased/h104 | `S2_op10_2.00_op12_1.25` | 0.003660 | 0.003037 | +0.000623 | [+0.000584, +0.000663] | 5/5 |

Service interpretation is preserved: PPO improves flow fill in both dense
frontier cells (`0.984` vs `0.840` for current/h104; `0.941` vs `0.613` for
increased/h104).

## Manuscript Implication

Allowed claim: the Track B policy transfers positively to current and increased
Garrido-native risk levels at the canonical horizon even after replacing the
stress-screen comparator with the full in-cell dense downstream-dispatch static
frontier.

Boundary claim retained: severe risk remains a real failure/boundary regime at
both horizons. The dense-frontier caveat now applies only to h52 cells and the
severe boundary cells.
