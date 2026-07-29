# Program I Op7 release-period instrument notice

Date: 2026-07-28

Status: `INVALIDATED_BY_INERT_ACTION_MAPPING`

## Scope

This notice applies only to the Program I Morris sensitivity estimate for
`op7_release_period`.

It does not rewrite the historical Program I verdict, declare the factor live,
or reopen any scientific data.

## Defect

`scripts/run_program_i_sensitivity.py` maps the factor to:

```python
"op8_rop": params["op7_release_period"]
```

`op8_rop` is not a mutable key in `MFSCSimulation.params`. At the time of the
screen, `MFSCSimulation.step()` silently ignored unknown action keys.

The resulting Morris values

```text
mu = 0
mu_star = 0
sigma = 0
sign_stability = 1
```

therefore cannot be interpreted as evidence that changing the Op7 release
period has no physical effect.

## Evidence separation

Two evidence streams must remain distinct:

1. **Morris screen:** invalid for this factor because the action mapping was
   inert.
2. **Static code trace:** the current catalog classifies
   `op7_release_period` as `transition_dead_configuration_field` using
   `supply_chain/config.py` and `supply_chain/supply_chain.py`.

Invalidating the first does not establish that the second is false. The
catalog disposition remains unchanged until an explicit Op7 actuator is
implemented and passes liveness.

## Required successor

Before any focal rerun:

1. define the operational meaning of an Op7 release-period decision;
2. define its native review epoch and carry-forward behavior;
3. add an explicit mutable key or adapter;
4. make unknown keys fail closed;
5. prove that two valid values change a relevant transition while holding
   tapes fixed;
6. run only on development or burned tapes;
7. assess whether the corrected factor was gate-critical before considering
   any broader reopening.

No rerun is authorized by this notice.
