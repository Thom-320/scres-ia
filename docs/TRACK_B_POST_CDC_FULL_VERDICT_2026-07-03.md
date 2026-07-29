# Track B Post-CDC Full Verdict (2026-07-03)

## Verdict

The full post-CDC Track B ablation is positive. PPO still beats the evaluated
static/heuristic comparator set when authority over the CDC itself (Op3 quantity
and reorder-point controls) is frozen at Garrido baseline.

This supports the claim that the Track B gain is not driven by controlling the
CDC. The result remains a Track B mechanism/robustness check, not a new paper
spine.

## Artifact

`outputs/experiments/track_b_ablation_8d_final_2026-07-01/post_cdc_only/`

Log:

`outputs/experiments/track_b_ablation_8d_final_2026-07-01/post_cdc_only_run.log`

## Protocol Check

- Seeds: `1..5`
- Eval episodes: `12` per seed
- CRN keys: `60`
- Policies per CRN key: `16`
- Comparators include: PPO, dense static family, and five heuristics
- Primary reported metric in the runner decision: `order_level_ret_mean`
- Best static comparator in the decision object: `s2_d1.50`

The episode ledger contains `960` rows: `16` policies x `5` seeds x `12`
episodes. Every `(seed, episode, eval_seed)` key has all `16` policies.

## Result

Against the runner-selected best static (`s2_d1.50`):

- PPO order-level ReT: `0.005611`
- best static order-level ReT: `0.005214`
- mean paired delta: `+0.0003967`
- paired episode signs: `60/60` positive
- seed signs: `5/5` positive
- minimum paired episode delta: `+0.0001714`

Seed-level deltas:

| Seed | PPO - static |
|---:|---:|
| 1 | `+0.000368` |
| 2 | `+0.000358` |
| 3 | `+0.000420` |
| 4 | `+0.000417` |
| 5 | `+0.000421` |

PPO ranks first by mean `order_level_ret_mean`; the next policies are
`heur_forecast_threshold`, `heur_tuned`, and `s2_d1.50`.

## Manuscript-Safe Wording

Use:

> A full post-CDC ablation, freezing Op3/CDC quantity and reorder-point controls
> at Garrido baseline while retaining downstream dispatch and shift authority,
> preserved the Track B advantage over the evaluated static and heuristic
> comparators.

Avoid:

> Track B never depends on upstream or distribution-centre decisions.

The ablation freezes CDC controls in the current Track B contract. It does not
prove a universal non-role for distribution-centre decisions in all supply-chain
settings.
