# Preventive Reserve v2 — Gate 1 verdict

**Verdict:** `STOP_NO_PREVENTIVE_HEADROOM`

**Artifact:** `outputs/preventive_reserve_v2/gate1_30x52/verdict.json`

**PPO trained:** no

Across 30 paired 52-week tapes, the physical reserve was live on 25/30 perfect
warning tapes, but the information policy did not create deployable value:

- perfect warning vs static zero service gain: +11.36%, CI95
  [+5.62%, +17.42%];
- imperfect warning vs shuffled placebo: -2.27%, CI95
  [-5.22%, -0.04%];
- the imperfect warning was dominated by static 15k;
- perfect warning was also static-dominated and used more inventory-time than
  static 15k.

This stop is valid for the executed contract: its fixed replenishment lead was
336 h, exactly equal to the warning lead. Route-aware delivery therefore met
the disruption onset and could be held until recovery. It is not the final test
of the proposed *real-lead* mechanism. Garrido's downstream path specifies two
24 h legs (Op10 and Op12), with Op11 availability between them. Changing from
an administrative 336 h lead to that physical path is a change of contract,
not a reinterpretation of v2. It is preregistered separately as v3.
