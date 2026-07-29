# Literature Scouting Brief for ChatGPT Pro

Purpose: identify the citation set and reviewer-defense literature for a Q1 paper on reinforcement learning for supply-chain resilience using a Garrido-grounded DES.

## Core Paper Claim To Support

The paper is not simply "RL beats static policies." The sharper claim is:

> The value of neural learning in supply-chain resilience is frontier-dependent. In a validated Garrido-style DES, Garrido's original buffer/shift controls expose little dynamic headroom: a dense static frontier captures the publishable optimum. When the action space is extended to the downstream dispatch bottleneck, PPO learns adaptive recovery/backlog control that improves Garrido/Excel ReT, lower-tail resilience, service continuity, and balanced Cobb-Douglas resilience against dense static policies.

## Citation Roles Needed

Please find and rank papers for these roles:

1. Garrido / SCRES foundation
   - Garrido-Rios thesis and Garrido et al. SCRES / Alzheimer-effect / Cobb-Douglas papers.
   - Papers on resilience metrics APj/RPj/DPj/CTj/ReT or similar recovery/absorption metrics.

2. Supply-chain resilience measurement
   - Composite resilience indices.
   - Cobb-Douglas / weighted multiplicative resilience indices.
   - Service-loss / recovery-time / time-to-recover metrics.

3. DRL in supply chains
   - DRL for inventory management.
   - DRL for disruption response.
   - DRL for supply-chain resilience specifically, if any.
   - PPO/SAC/TD3 comparisons in continuous supply-chain actions.

4. DES + RL / simulation optimization
   - Papers combining discrete-event simulation with reinforcement learning.
   - Papers using simulation environments as testbeds for operational resilience.

5. Bottleneck/action-space theory
   - Theory of Constraints and bottleneck control.
   - Papers showing action-space alignment / controllability matters for RL.
   - Dispatch or last-mile bottleneck control in logistics/supply chains.

6. Tail risk / CVaR / multi-objective RL
   - CVaR-aware RL.
   - Multi-objective RL in operations or supply chains.
   - Pareto frontier reporting and risk-sensitive policy comparison.

7. Non-stationarity / retained learning / continual RL
   - Only needed if H4 retained-vs-reset becomes a claim.
   - Papers on catastrophic forgetting, backward/forward transfer, continual RL metrics.

8. Reviewer safety citations
   - Surveys that reviewers expect: RL in SCM, DES+ML in supply chains, SCRES reviews.
   - Papers that could claim similar novelty; identify and differentiate them.

## Known Candidate Papers / Areas To Verify

Please verify bibliographic details and whether these are real/current/relevant:

- Garrido-Rios (2017), PhD thesis on military food supply chain resilience.
- Garrido, Ponguta, Adarme / Garcia-Reyes (2024), SCRES / Cobb-Douglas / AI-SCRES related papers.
- Gijsbrechts et al. (2022), Deep reinforcement learning for inventory control / M&SOM.
- Boute et al. (2022), DRL inventory roadmap / EJOR.
- Rolf et al. (2023 or 2025), RL in supply-chain management review.
- Ivanov, digital supply chain twins / ripple effect / resilience via simulation.
- Haleh et al. (2018), Cobb-Douglas resilience measurement foundation if applicable.
- Goldratt & Cox / Theory of Constraints for bottleneck logic.
- Recent papers on DES+DRL in manufacturing/supply chains.
- Recent papers on CVaR/multi-objective RL in operations.

## Output Format Requested

Return a table with columns:

- Priority: must-cite / useful / optional / avoid
- Full citation
- DOI/arXiv/URL
- Venue and year
- Claim role
- What it supports in our paper
- How we differ
- Risk if omitted

Then provide:

1. Top 10 must-cite papers.
2. Top 10 closest competitors and our differentiator.
3. Suggested related-work outline.
4. Missing-citation risks a Q1 reviewer would flag.
5. A short paragraph we can use in the paper to position our contribution.

Search broadly and verify details. Do not hallucinate citations. If a paper is uncertain, label it as unverified.
