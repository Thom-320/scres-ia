# K3 Confirmation Freeze

The terminal pre-learner screen passed on development seeds `6720001-6720300`.
Before opening confirmation, freeze:

- classical comparator: budgeted `(s,S)=(2.0,3.0)`;
- observable candidate: paced MPC `(alpha,beta,gamma)=(1.25,0.0,1.5)`;
- exact total order commitment `10·D0`, weekly cap `1.5·D0`, lead one week;
- primary metric canonical full-ledger order ReT;
- guardrails quantity-ReT, lost orders, remaining quantity and exact resources;
- confirmation seeds `6800001-6800120`, opened once;
- no reselection, coefficient change, metric change or additional confirmation block.

Confirmation passes only if mean ReT delta is at least 0.01 with positive
paired CI95 lower bound, all guardrails pass and at least 70% of tapes are
nonnegative. A pass authorizes a learner study on still-virgin `6900001+`; it
does not by itself establish neural incremental value.
