# Program K3 Final Corrective Verdict

## Verdict

`RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND`

The strong-MPC development and confirmation gates appeared to pass against a
static frontier restricted to periods 1-4. The authorized PPO run then beat
both `(s,S)` and MPC in five of six seeds. A mandatory policy-trajectory audit
found that PPO seed 0 emitted exactly one sequence across all 120 learner-test
tapes:

```text
(1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.0) · D0
```

Replacing PPO with this fixed eight-week schedule reproduces its ReT exactly.
The schedule was absent from the comparator because it has period eight. It
also beats the purportedly adaptive MPC under identical total and weekly
resources. The apparent neural/adaptive victory is therefore an open-loop
calendar discovery, not state-feedback value.

The raw confirmation and PPO artifacts remain versioned as an audit trail, but
their promotion fields are superseded by
`results/k3/open_loop_confound_audit.json`. Paper 2 adaptive control and Paper
3 neural retention are not authorized. Seeds `6800001+` and `6900001+` are now
opened and cannot be reused as virgin confirmation.

No further architecture, horizon, signal or reward is permitted within K3.
