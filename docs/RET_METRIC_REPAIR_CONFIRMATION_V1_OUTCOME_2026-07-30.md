# ReT metric repair confirmation v1 — terminal outcome

**Status:** `COMPLETE_PROSPECTIVE_CORRECTIVE_CONFIRMATION`

**Frozen contract:** `contracts/ret_metric_repair_confirmation_v1.json`

**Contract SHA-256:** `c1efdc20fc9d75743a5789ab3ddf90108d7f18cf1230dcd62fb5a85815dc441e`

**Adjudication self hash:** `bde02309f72c9ee11222704a63a4c38dc27ca74cb9c15743dc42bd8b104c0ff4`

The confirmation used sixteen previously unopened tapes per family, five future
scenarios per candidate, the complete 216-posture static domain, paired tape-level
inference, and replay-prefix state-hash checks. Both eligible runs completed and all
prefix hashes matched. A duplicate local R2r attempt was stopped before its dynamic
phase, preserved under quarantine, and excluded from adjudication.

## Contract adjudication

| family | primary delta: MPC − frozen static | paired 95% CI | positive tapes | contract verdict |
|---|---:|---:|---:|---|
| R1r | −0.00001954 | [−0.00004940, −0.00000021] | 5/16 | `NOT_CONFIRMED` |
| R2r | **+0.01247474** | **[+0.00910860, +0.01590910]** | **15/16** | `PASS_MATERIAL_REPAIRED_MPC` |

The mandatory quantity-to-time sensitivity, explicitly treated as
`DISCLOSED_PROXY_NOT_EXACT_ATTRIBUTION`, agrees directionally:

| family | proxy delta | paired 95% CI | positive tapes |
|---|---:|---:|---:|
| R1r | −0.00001965 | [−0.00004863, −0.00000029] | 5/16 |
| R2r | **+0.01238668** | **[+0.00924427, +0.01525866]** | **15/16** |

R2r clears the preregistered materiality threshold of 0.005 and the frozen
flow-fill guardrail. R1r does not confirm a positive MPC effect.

## The stronger scientific reading is endpoint-bounded

The R2r pass is real under the frozen bounded endpoint, but it is not universal
physical superiority:

| R2r endpoint, MPC − static | mean delta | paired 95% CI |
|---|---:|---:|
| bounded `ret_excel` | **+0.012475** | **[+0.009109, +0.015909]** |
| canonical `ret_excel` | **+0.012516** | **[+0.009004, +0.015955]** |
| `ret_excel_full_ledger` | **−0.004483** | **[−0.006600, −0.002388]** |
| `ret_thesis` | +0.000370 | [−0.001084, +0.001731] |
| flow fill | +0.002340 | [−0.001449, +0.006260] |
| delivered rations | **−25,399** | **[−29,344, −21,174]** |
| strategic material injected | **−99,072** | **[−112,529, −87,077]** |

The canonical and clipped endpoints agree on the new R2r tapes. Therefore this
prospective result does **not** reproduce the development claim that clipping itself
reverses the verdict. Instead, it confirms an MPC advantage under both the canonical
visible-order score and its bounded version. The full-ledger score points in the
opposite direction, while MPC uses substantially less strategic material and delivers
fewer total rations. The defensible interpretation is an endpoint- and
resource-dependent trade-off, not dominance.

R1r exposes the same boundary from the other side. Its primary delta is slightly
negative and far below the 0.005 SESOI, although MPC delivers about 3,756 more rations,
raises flow fill slightly, and injects about 88,133 fewer strategic units. Thus scalar
ReT ordering and the physical/resource panel can disagree even after bounding the
per-order score.

## Claim boundary

Allowed:

> In a prospective 16-tape confirmation over the complete 216-posture buffer domain,
> replay MPC achieved a material positive delta over the frozen static incumbent in
> R2r under the preregistered bounded ReT endpoint; the effect did not generalize to
> R1r and did not constitute physical or resource-adjusted dominance.

Not allowed:

- replacing or relabelling historical canonical results;
- claiming that clipping caused the prospective R2r result;
- claiming MPC dominates the static incumbent across metrics or service endpoints;
- treating the quantity-to-time proxy as identified causal attribution;
- using this result to authorize a neural or KAN bakeoff;
- extending the result beyond the shifts=1 buffer subcontract.

## Terminal decision

```text
PASS_MATERIAL_REPAIRED_MPC_R2R
NOT_CONFIRMED_R1R
HOLD_RESOURCE_ADJUSTED_OR_FULL_LEDGER_SUPERIORITY
NO_GO_NEURAL_OR_KAN_AUTHORIZATION
```
