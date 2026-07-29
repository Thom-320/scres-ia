# Program D D1-v1 audit — historical authority result

**Audited status: `NOMINAL_AUTHORITY_ONLY`.** The original artifact and verdict
remain unchanged for provenance. They must not be cited as evidence that Op9
rationing improves resilience under R1/R2 disruptions.

## Verified facts

- `scripts/run_program_d_authority_screen.py` never set `risks_enabled=True`.
- `FAMILIES=(R1,R2,mixed)` affected index arithmetic only; it did not select an
  enabled-risk set.
- Promotion was `ret_lo > 0`; the declared authority threshold and service/lost
  guardrails did not enter the decision.
- `age_threshold` was best on all 30 nominal tapes. The tape oracle over that
  constant therefore had zero headroom.
- The run did establish physical authority under nominal congestion: queue order
  changed service, loss, and Excel ReT.

D1-v2 independently activates R1/R2/mixed tapes, separates selection and
validation, binds service/loss guardrails, and freezes the comparator before
counterfactual branching.

