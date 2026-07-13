# K3 Strong-MPC Amendment — frozen before dev2 seeds

The first K3 development screen found a mean `+0.0103` ReT signal, but failed
CI, lost-order and actual-resource gates. This amendment does not alter demand,
lead time, ReT, budget, capacity or confirmation seeds.

Before opening any new data, it freezes:

- exact commitment of `10·D0` by every policy, enforced through a feasibility
  floor that prevents an infeasible unspent budget near episode end;
- static frontier: every period-1..4 calendar on the 0.25 grid;
- classical adaptive frontier: budgeted `(s,S)` and inventory-only pacing;
- candidate: receding forecast/inventory/backlog controller with coefficients
  selected only on `6710001-6710120` from
  `alpha,beta ∈ {0,.25,...,1.5}`, `gamma ∈ {0,.5,1,1.5,2}`;
- evaluation: `6720001-6720300`, used once for this terminal pre-learner gate;
- confirmation `6800001+` and learner `6900001+` remain sealed.

Promotion requires ReT delta at least 0.01 with positive paired CI95 lower
bound, quantity-ReT and lost-order non-inferiority, exact resource equality and
at least 70% nonnegative tapes. Failure closes K3 without PPO.
