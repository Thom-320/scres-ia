# Program I — review of the 2-lane full-DES GSA dictamen (2026-07-12)

Verdict: **methodologically sound and well-cited, with 3–4 genuinely additive pieces (adopted below).
But its core proposal — a full-DES 36-factor two-lane GSA campaign — contradicts the locked scope and
the frozen "finish the manuscript first" discipline, and largely re-does what Programs D–I already
executed. Recommendation: adopt the cheap additive pieces into the existing stylized Program I; DEFER
the full-DES 36-factor campaign to the post-submission paper. The scope decision is the PI's.**

## What is correct and ADDITIVE (adopted / noted)
- **Synthetic acceptance tests — ADOPTED.** `tests/test_program_i_gsa.py` (4/4 green): Sobol recovers
  Ishigami; Morris ranks active≫inert; a *sensitive-but-constant-optimal* headroom STOPS (the Program
  G/H anchor case); a *ranking-reversal* headroom promotes only if it does NOT starve a node (the exact
  behaviour of the GP-located region). This directly validates that the estimator distinguishes
  sensitivity from headroom — the project's whole thesis — and strengthens the manuscript's rigor.
- **Dependent-input GSA caveat — CORRECT, but only bites the full-DES lane.** Standard Sobol assumes
  independent factors; Q/ROP/capacity coupling makes full-DES inputs dependent → Shapley-effect /
  Kucherenko-style indices needed there. Program I's stylized factors (signal_q, lead, surge, dwell,
  commonality, r22) are INDEPENDENT by construction, so the standard Sobol we ran is valid; noted for
  the deferred full-DES lane.
- **Typed decision-right catalog (Op3–Op13) — worth building as a paper artifact** (systematizes the
  decision surface the exploration found scattered across config.py + contracts). Cheap doc/JSON; can
  be added to the manuscript's methods without any new compute.
- **D-optimal design for unordered categoricals** (priority rules, routing, maintenance doctrine,
  dispatch mode) and **grouped Morris** for high-dim screening — correct; relevant only if the full-DES
  lane is opened. **Stochastic kriging + ranking-and-selection with CRN** — the right tools for the
  expensive full-DES GSA, correctly cited. **SALib pin** — reasonable IF we go to full 36-factor
  grouped Morris/dependent Sobol; our hand-rolled estimators are validated for the stylized lane.

## Where it overreaches (the critical flags)
1. **Contradicts the locked scope.** The PI already chose (this session) STYLIZED fast lanes over the
   full DES, explicitly because the full-DES spatial port is deferred. Carril A here IS the full-DES
   36-factor GSA, which requires that deferred port.
2. **Contradicts the frozen finish-the-manuscript discipline.** Multiple prior dictámenes + the PI set:
   Program H/I are the LAST computational extensions; the full-DES port is a SEPARATE post-submission
   paper; "not giving up = finishing the paper." A full-DES 36-factor two-lane Morris+Sobol+kriging
   campaign (~740 configs × 12 tapes × full-DES horizon, plus surrogates) is months of VPS compute — the
   "sophisticated, more-JSON way of not finishing anything" the 5th dictamen explicitly warned against.
3. **Largely re-does D–I.** The two-lane structure (realistic screen → engineered phase diagram),
   headroom-as-output, regret-weighted tree, branching gates, convertibility gates, placebos, and the
   anti-p-hacking discipline are exactly Programs D/F/G/H/I. The stylized Program I this turn already
   returned the answer: information/risk-magnitude INERT (Morris μ*=0), only structural
   scarcity×concurrency moves headroom, and the sole H_obs>0 region is a spatial-fairness violation. A
   full-DES 36-factor GSA would very likely reproduce this at ~100× the cost — low marginal value now.

## Recommendation
- **Now (cheap, adopted):** the acceptance tests (done); optionally the `decision_right_catalog_v1` doc
  as a manuscript methods artifact (no compute).
- **Defer to the post-submission full-DES paper:** the 36-factor grouped-Morris / dependent-Sobol /
  stochastic-kriging Carril A, SALib pin, D-optimal categoricals. Keep the design (it is good) as the
  preregistered plan for that separate paper.
- **Do NOT reopen the full-DES port before submitting the current manuscript.** The stylized Program I
  answered the PI's question with a principled GSA; the manuscript is the remaining artifact.
