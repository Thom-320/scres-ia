# ChatGPT Pro Review Package (2026-07-01)

Use this package to ask ChatGPT Pro for a Q1-level critique of the current
SCRES-IA paper evidence.

Repository:

- https://github.com/Thom-320/scres-ia
- Branch: `codex/garrido-replication-experiments`

Primary files for ChatGPT Pro:

1. Prompt:
   [`docs/chatgpt_pro_review_package_2026-07-01/CHATGPT_PRO_Q1_REVIEW_PROMPT.md`](chatgpt_pro_review_package_2026-07-01/CHATGPT_PRO_Q1_REVIEW_PROMPT.md)

2. Literature scouting brief:
   [`docs/chatgpt_pro_review_package_2026-07-01/LITERATURE_SCOUTING_BRIEF.md`](chatgpt_pro_review_package_2026-07-01/LITERATURE_SCOUTING_BRIEF.md)

3. Q1 evidence bundle:
   [`docs/track_b_q1_stats_2026-07-01/README.md`](track_b_q1_stats_2026-07-01/README.md)

4. Claim registry:
   [`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`](CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md)

Current headline evidence:

- Primary metric: Garrido/Excel ReT.
- Track B PPO vs best dense static by Excel ReT:
  - PPO: `0.005893`
  - Best dense static: `0.005466`
  - Delta: `+0.000426`
  - CI95: `[+0.000389, +0.000463]`
  - Paired Cohen's d: `2.87`
- PPO is non-dominated versus the dense static frontier on Excel ReT, cost,
  CTj p99 tail, and flow fill.

Current framing boundary:

- Track B is an operational extension that controls the downstream dispatch
  bottleneck. It is not Garrido's original decision-variable set.
- Track A is boundary evidence: Garrido's original buffer/shift-style controls
  do not expose a publishable dynamic frontier after dense frontier, CRN, and
  rich metrics.
- Mechanism language should be "adaptive recovery / backlog control" unless
  lead-lag analysis later proves anticipation.
- H4 retained-vs-reset is not yet a claim.
