# S3 Non-Monotonicity Audit (2026-06-28)

Question: why does a simple `f0.10_S1` static policy beat more aggressive S2/S3 policies under the
war-stress continuous lane (`phi=4`, `psi=1.5`, h104)?

## Result

The effect is real in the current DES, not a reporting artifact. Under the dense-CRN evaluation seeds
(`9000..9007`), `f0.10_S3` is worse than `f0.10_S1` on Excel ReT, flow, lost orders, service-loss AUC,
and TTR.

| policy | Excel ReT | flow_fill | lost_rate | service AUC | TTR mean | resource |
|---|---:|---:|---:|---:|---:|---:|
| `f0.10_S1` | 0.002279 | 0.7876 | 0.1637 | 4.04e9 | 873.6h | 0.05 |
| `f0.10_S2` | 0.002255 | 0.7882 | 0.1661 | 4.21e9 | 892.3h | 0.30 |
| `f0.10_S3` | 0.002091 | 0.7706 | 0.1847 | 4.60e9 | 989.6h | 0.55 |

## Isolation Tests

Temporary in-process capacity-table patches show that the degradation is not primarily caused by
S3's larger batch size (`7000`). It is driven mainly by the S3 upstream raw-material dispatch
quantity (`op3_q=47000`).

| variant | `f0.10_S3` Excel | flow_fill | lost_rate | service AUC | TTR mean |
|---|---:|---:|---:|---:|---:|
| original S3 | 0.002091 | 0.7706 | 0.1847 | 4.60e9 | 989.6h |
| S3 with batch fixed to 5000 | 0.002089 | 0.7706 | 0.1847 | 4.60e9 | 989.7h |
| S3 with `op3_q` fixed to S1 value | 0.002347 | 0.7987 | 0.1534 | 4.02e9 | 786.0h |
| S3 with batch 5000 and `op3_q` fixed to S1 | 0.002240 | 0.7852 | 0.1667 | 4.32e9 | 975.4h |

## Mechanism

S3 creates much more upstream production/WIP, but the downstream stages do not convert it into theatre
service under war-risk disruptions. The extra upstream flow accumulates as inventory and rework rather
than delivered resilience.

| policy | total produced | total delivered | rework mean | rations_sb mean | rations_cssu mean |
|---|---:|---:|---:|---:|---:|
| `f0.10_S1` | 0.597e6 | 0.828e6 | 62 | 0.019e6 | 0.038e6 |
| `f0.10_S2` | 1.233e6 | 0.815e6 | 120 | 0.467e6 | 0.054e6 |
| `f0.10_S3` | 1.798e6 | 0.792e6 | 238 | 1.036e6 | 0.067e6 |

Interpretation: in this DES, "more aggressive capacity" is not a monotone improvement. S3 increases
upstream throughput and WIP exposure under risks, while Op9/Op10/Op12/theatre delivery remain the real
service bottleneck. The result is more inventory and rework, not better Excel ReT.

## Implication

This is a strong reason to try a more expressive, still thesis-adjacent continuous action contract:
per-node buffer fractions instead of one common fraction:

`Box([op3_frac, op5_frac, op9_frac, shift_signal])`

That would keep Garrido's Track-A decision family (inventory buffers + shifts) but let the agent avoid
over-buffering/upstream overfeeding when the bottleneck is downstream.

Audit artifact: `outputs/audits/s3_nonmonotonicity_2026-06-28/s3_nonmonotonicity_audit.json`.
