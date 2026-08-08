# The sealed artifact was stale relative to its own producer

`result_2026-08-08_pre_q1_decision.json` is the version that sat in this directory until
2026-08-08. It carries `q1_decision: null`, because it was produced before the script learned to
compute that block. Re-running the producer to add `run_role` and `scope` recomputed it.

Nothing scientific moved. `claim_status` is identical, all five falsifiers still pass, and the
numbers the manuscript cites -- the Fig. 5 identity at R^2 = 1.0 with maximum error 3.22e-15 -- are
byte-identical between the two. The difference is one previously-null field now filled in.

It is kept because a sealed artifact that is replaced does not vanish, and because "the artifact on
disk predated its producer" is the kind of drift worth being able to point at later.
