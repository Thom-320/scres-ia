# Program U — Garrido policy-search and risk gate

## Binding interpretation

Garrido (2017) did **not** enumerate or optimize sequential decision policies.
The thesis used a design of simulation experiments over 90 supply-chain
configurations and measured ReT over long runs.  Its decision variables were
configuration factors: risk levels, on-hand inventory and short-term assembly
capacity.  Therefore the thesis supports efficient experimental design and the
physical ranges below; it does not establish that every DES variable, every
risk interaction, or adaptive headroom was optimized.

The new Program U separates three claims:

1. **Static-policy discovery:** a learner must find a single open-loop calendar
   using a limited simulator budget.  The exhaustive 65,536 matrix is hidden
   from training and used only as a small-case answer key.
2. **Endogenous review:** a learner observes deployable state and allocates a
   finite budget of review rights.  It selects both product mix and when to
   review again.
3. **Risk interaction:** a cascaded, fail-closed screen locates thesis-native
   regions before any neural learner is trained.

## Thesis evidence recovered

- The baseline assembly line used one shift, 320.5 rations/hour and 15,384
  effective rations/week.  Garrido evaluated one, two and three shifts.
- The nine modeled risks were R11–R14, R21–R24 and R3.  Tables 6.11–6.12 give
  current and increased frequency/probability levels.  Program U uses these
  two levels as its primary native envelope rather than tuning damage until a
  learner wins.
- Scenario I used efficient design matrices for increased risk frequency.
  Scenario II evaluated buffers equivalent to 168, 336, 504, 672 and 1,344
  hours at Op3, Op5 and Op9.  Scenario III evaluated 1–3 shifts.
- Chapter 8 explicitly leaves two gaps: the interaction among inventory,
  capacity and lead time, and the search for an optimum level of SCRes.
- Chapter 7 reports that increased frequency was associated with lower ReT for
  most risks; R12 and R21 were pruned from the association-rule result.  The
  prose says seven of nine risks but the parenthetical list is internally
  inconsistent, so Program U will preserve the raw risk-specific outcomes and
  will not silently repair that count.

## What has and has not been optimized

Already evaluated substantially:

- Program O product-mix persistence/share region;
- the full 65,536 open-loop frontier in the Program O/Q contract;
- a bounded relevant-risk screen;
- Program Q learned feedback versus the complete static frontier;
- thesis-like buffer and shift configurations in earlier repository lanes.

Not completed:

- frequency × impact × product × timing sensitivity for adaptive value;
- all-DES-variable sensitivity;
- a certified risk-aware transducer for every Program S mask;
- simulator-budgeted recovery of the static optimum by a learner;
- endogenous review timing for product mix;
- residual value over reinforced event-triggered MPC.

Program S S1 cannot answer those questions: its risk-aware transducer failed a
direct replay tolerance before completing the screen.  Its partial shards are
custody evidence, not scientific outcome evidence.

## Researcher-designed adverse environments

The thesis is a base, not a prohibition on prospective extensions.  Program U
therefore permits two additional, explicitly labeled strata:

- **Decision-relevant stress:** risk-specific frequency, recovery/impact,
  concurrency and timing may vary inside a frozen plausible envelope.  The
  environment designer optimizes safe headroom and action-ranking diversity,
  never the performance of PPO or another learner.
- **Dynamic regime stress:** normal, surge and recovery regimes change during
  an episode.  The regime is latent; the controller observes only realized
  outages, recovery, demand and physical state.  This is the cleanest way to
  create a legitimate advantage for feedback over a robust constant policy.

This is not “increasing damage until RL wins.”  Before seeing learner results,
the least-severe connected region must show safe perfect-information headroom,
observable conversion by a strong classical controller, multiple optimal
actions and favorable guardrails.  A null with stationary regimes must remove
the incremental timing mechanism.

Stochastic processing time is also allowed, but the existing repository option
uses `Tri(0.75*PT, PT, 1.5*PT)`, whose mean is `1.0833*PT`.  It changes average
capacity as well as variability.  The primary Program U extension instead
pre-generates mean-one lognormal potential processing multipliers under CRN.
The deterministic spread-zero cell must reproduce the base DES exactly.

## Computational ladder

The successor must run in this order:

1. Force every mask/extreme on burned diagnostic tapes and prove direct-DES
   exactness before generating a score matrix.
2. Run one tape per point.  Stop an inexact mask without blocking exact masks.
3. Expand only exact points with preliminary safe headroom.
4. Batch several points per worker and cache common demand/risk tapes and
   action-independent prefixes.
5. Use approximate policy search to localize connected regions.
6. Use branch-and-bound, DP or certified optimization where available.
7. Enumerate all 65,536 calendars only for at most 6–12 finalists and for the
   small U0 answer-key experiment.

This changes exhaustive enumeration from the search algorithm into a bounded
validation instrument.

## Burned-seed static-discovery smoke

The first implementation check used three non-scientific smoke tapes.  PPO was
allowed to query one calendar/tape pair per episode and never read the complete
score matrix during training.  The exact matrix was generated only after the
candidate was frozen.

| Budget | Calendar/tape evaluations | Rank among 65,536 | Simple ReT regret |
|---:|---:|---:|---:|
| 30,000 steps | 3,776 | 2,720 | 0.02201 |
| 100,000 steps | 12,512 | 233 | 0.00559 |

This supports the feasibility of simulator-budgeted static-policy discovery,
but does not show exact recovery and is not scientific evidence.  It also shows
why “RL will converge to the optimum” cannot be assumed: even in a small space,
vanilla PPO approached but did not recover the exact optimum.  The next frozen
bakeoff must compare PPO with random search, cross-entropy search, policy
gradient, Bayesian optimization and parameterized simulation optimization on
new burned development tapes, followed by a separate held-out qualification.
