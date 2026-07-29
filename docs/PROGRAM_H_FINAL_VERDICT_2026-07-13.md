# Program H final verdict — 2026-07-13

## Verdict

**`STOP_PROGRAM_H_NO_BELIEF_POLICY_PASS_INFORMATION_BOUND_REMAINS_LOOSE`.**

Program H executed its frozen O0 audit on 200 development tapes (`1060001+`) and 400 locked
belief-policy tapes (`1070001+`). The result was reproduced bit-for-bit. No tape from
`1080001+` was opened and no PPO or RecurrentPPO was trained.

The Bayes filter was informative: log loss was 0.8568 versus 1.0016 for the frozen prior.
Information existed, but no belief-aware policy converted it into a qualifying advantage.

| Policy vs ABAB | Delta ReT order (CI95) | Favorable | PI conversion | Verdict |
|---|---:|---:|---:|---|
| Regret fitted-Q | +0.00225 [-0.00021, +0.00460] | 19.0% | 13.7% | FAIL |
| Belief-MPC 2w | -0.01752 [-0.02154, -0.01363] | 9.0% | -106.8% | FAIL |
| Belief point rollout | -0.02298 [-0.02777, -0.01839] | 8.25% | -140.0% | FAIL |

The regret policy improved quantity-weighted ReT (+0.00625), but failed the primary MCID,
positive LCB, attended orders (-0.535), worst-CSSU fill (-0.0385), unfulfilled-ration
non-inferiority (+226), favorable-tape threshold, and 30% conversion gate.

The exact full-tape 81-sequence information-relaxation ceiling remained material: +0.01641,
CI95 [+0.01397,+0.01886]. It is rigorous but loose. Program H therefore does not prove that
the mathematically optimal O0 policy cannot beat ABAB. It proves that an informative Bayes
filter, regret fitted-Q, belief-MPC, and point rollout did not produce a qualifying policy.
Formal information sufficiency remains unresolved, while the program terminates under its
frozen last-program rule.

Binding consequences: Program G remains terminal; Program H is terminal; no Programs I--K;
1080001+ stays closed; PPO, RecurrentPPO, and persistent/reset/frozen are not authorized; and
no reward, risk, signal, horizon, metric, architecture, or physics rescue is allowed.

Machine artifact: `results/program_h/locked_belief_policy_gate/verdict.json`.
