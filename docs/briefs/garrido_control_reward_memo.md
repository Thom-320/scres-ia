**MFSC Hybrid Benchmark Memo**

This memo summarizes the current DES+RL lane for the military food supply chain project. It is intended as a short briefing for project status, not as a claim of final PPO superiority.

**Current Environment Interface**

The simulation lane uses the validated 13-operation DES as the primary system model and exposes a compact Gymnasium-compatible sequential decision interface for weekly control. The observation is a 15-dimensional operational snapshot combining normalized inventories, service indicators, disruption flags, and time-progress features. The action space is five-dimensional: four continuous signals modulate inventory-policy parameters at Operations 3 and 9 through multiplicative adjustments to order quantities and reorder points, while the fifth signal selects assembly capacity over three shift regimes (`S1`, `S2`, `S3`). This interface is best described as a practical control approximation rather than a formal proof of full Markov sufficiency; the DES evolves with a richer internal state than the agent observes, so partial observability remains a valid caveat.

**What Is Already Validated**

The DES itself is validated and operational. Static shift-control baselines are clearly differentiated under `risk_level="increased"`, which confirms that shift allocation is a meaningful control lever before any learning claim is made. In the current control-reward benchmark, `ReT_thesis` is retained as a reporting metric, while PPO is trained on a separate operational reward (`control_v1`) because direct control learning needs an explicit service/cost objective rather than a resilience score designed for evaluation.

| Policy | Control reward mean | Fill rate | Backorder rate | Shift mix |
| --- | ---: | ---: | ---: | --- |
| `static_s1` | -359.41 | 0.650 | 0.350 | 100% `S1` |
| `static_s2` | -179.58 | 0.827 | 0.173 | 100% `S2` |
| `static_s3` | -181.92 | 0.830 | 0.170 | 100% `S3` |

Static source: `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/benchmarks/control_reward/policy_summary.csv` for `w_bo=4.0`, `w_cost=0.02`, `w_disr=0.0`.

**What Is Still Being Tested**

Current PPO results under `control_v1` are preliminary but promising in a narrow weight regime. The most favorable existing run (`w_bo=4.0`, `w_cost=0.02`, `w_disr=0.0`) suggests adaptive switching can match the service level of the best static baseline without collapsing to a single shift mode, but those gains are not yet globally validated. The next robustness lane is therefore scoped to longer runs, more seeds, and stochastic processing times, first under `risk_level="increased"` and then under `risk_level="severe"`. The goal of that lane is to test whether the adaptive signal remains stable under stronger training and harsher operational stress, not to pre-announce a definitive PPO win.
