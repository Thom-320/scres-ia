# Invalidated Program L Gate 1 CRN-v2 artifact

This version correctly materialized risk tapes but still initialized each
static policy with its own shift during warm-up. The historical wrapper also
continued beyond physical warm-up while demand was already active, consuming
different positions of the demand stream. It is superseded by
`../l_program_gate1_crn_v3_2026-07-10/`, which uses endogenous physical warm-up,
a common S1 initial state, and applies every static policy through the delayed
weekly action contract.
