# Literature Research Request for ChatGPT Pro

## Supporting the Q1 Paper: "Neural Reinforcement Learning for Supply Chain Resilience — Action Space Alignment as the Mechanism"

### Repository: `github.com/Thom-320/scres-ia` (branch: `codex/garrido-replication-experiments`)

---

## Objective

Find and evaluate **30-50 papers** to build a comprehensive Q1 submission bibliography. We already have ~34 references from existing bibliography. We need you to **find, assess, and rank** papers we should add.

For each paper, return:
1. Full citation (authors, year, title, venue, volume/pages)
2. **Q1 grade**: Q1-top / Q1 / Q2 / non-Q1
3. **Citation count** (Google Scholar estimate)
4. **Relevance score** 1-10 (10 = directly on point)
5. **One-line relevance** to our specific claim
6. **Risk to our paper** (competes directly, supports, or irrelevant)

---

## Search Areas (ordered by priority)

### A. Direct Competitors (HIGHEST PRIORITY — we MUST differentiate)

**Search terms:**
- "reinforcement learning" + "supply chain resilience" + "disruption" + "dispatch" or "recovery"
- "PPO" + "inventory" + "supply chain" + "disruption"
- "deep RL" + "multi-echelon" + "inventory" + "disruption recovery"
- "action space" + "supply chain" + "reinforcement learning" + "bottleneck"

**What we need:** Papers that directly address RL-for-SCRES with operational control (dispatch, capacity, recovery). NOT forecasting papers, NOT network design papers.

**Known competitors (must-grade):**
| Paper | What They Do | How We Differentiate |
|---|---|---|
| Ding et al. (2026, IJPE) | MARL for SC resilience reconfiguration | They do network topology; we do operational dispatch |
| Lu et al. (2025, Symmetry) | PPO for multi-echelon inventory | They study replenishment; we study 13-op DES |
| Kim (2024, IISE Trans.) | MARL for inventory transshipment | Different mechanism (transshipment vs dispatch) |
| Rolf et al. (2024, IJPR) | SD+RL for recovery policies | SD vs DES; different fidelity |
| Bussieweke, Mula et al. (2025, IJPR) | SD+RL for disruption recovery | SD simulation; we use DES |

**Grade each on:** relevance, Q1-ness, how urgently we must cite them, what we must say to differentiate.

---

### B. DES + RL Integration (VALIDATE METHODOLOGY)

**Search terms:**
- "discrete event simulation" + "reinforcement learning" + "production" or "inventory"
- "DES" + "DRL" + "scheduling" or "dispatching"
- "simulation optimization" + "deep reinforcement learning" + "supply chain"

**What we need:** Papers that use DES as the environment for RL training, in a production/inventory context. These validate our methodological choice.

**Examples we know:**
- Wang & Liao (2024) — J. Intelligent Manufacturing, DES+DRL for job shop (82 cites)
- Sakr et al. (2023) — J. Intelligent Manufacturing, DES+RL for semiconductor dispatching (46 cites)
- Rolf et al. (2025) / Kogler & Maxera (2026) — J. Simulation, DES+ML review (72 papers surveyed)

**Search for more:** We need papers that specifically link DES state to RL observation (not just using DES as a black box reward generator).

---

### C. Cobb-Douglas / Composite Resilience Metrics (VALIDATE METRIC)

**Search terms:**
- "Cobb-Douglas" + "resilience" + "supply chain"
- "composite resilience index" + "supply chain"
- "resilience measurement" + "order-level metrics"
- "recovery time" + "resilience metric" + "disruption"

**What we need:** Papers that use composite or order-level metrics for supply chain resilience. These validate our use of Garrido's Excel ReT formula and the Cobb-Douglas resilience index.

**Examples:**
- Garrido, Ponguta & Garcia-Reyes (2024, IJPR) — C-D factory resilience index (our source)
- Haleh et al. (2018) — C-D for resilience assessment
- Ma, Li & Pan (2023, Systems) — C-D for industrial chain resilience
- Sun (2026, Discover AI) — SC resilience index with trained models

**Search for:** Other authors using C-D or composite resilience indexes. Any debate in the literature about composite vs single-dimension metrics.

---

### D. Retained Learning / H4 Scaffolding

**Search terms:**
- "continual reinforcement learning" + "nonstationary"
- "catastrophic forgetting" + "reinforcement learning"
- "policy retention" + "transfer" + "reinforcement learning"
- "recurrent PPO" + "supply chain" or "inventory"

**What we need:** Theoretical and methodological scaffolding for our H4 experiment (not yet run). If H4 is deferred, we still need to cite these as "future work" or "open hypothesis."

**Examples:**
- Pan et al. (2025) — Continual RL survey (23 cites)
- Abbas et al. (2023) — Loss of plasticity in continual deep RL
- Kumar et al. (2025) — Continual learning as computationally constrained RL

**Search for:** New 2024-2026 papers on continued learning in operational/industrial contexts, not just games.

---

### E. Reward Design for SCRES (SUPPORTING)

**Search terms:**
- "reward shaping" + "inventory" + "reinforcement learning"
- "Cobb-Douglas" + "reward function"
- "multi-objective RL" + "supply chain" + "resilience"
- "CVaR" + "reinforcement learning" + "inventory" or "operations"

**What we need:** Papers that design custom reward functions for SC/inventory RL, especially if they use C-D or multi-objective framing.

**Examples:**
- De Moor et al. (2022, EJOR) — Reward shaping for perishable inventory
- Kotecha & del Rio Chanona (2025) — MORSE: Multi-objective RL + CVaR
- Sabal Bermudez et al. (2023) — Distributional constrained RL for SC

---

## Literature Differentiation Matrix

For each paper found, tell us:
1. **Does it compete directly with our Track B claim?** (RL-for-SCRES win over static)
2. **Can we cite it as supporting evidence?** (similar methodology, different domain)
3. **Is it a "must-differentiate"?** (reviewer will ask "how is this different from X?")
4. **What specific sentence should we write to differentiate?**

---

## Deliverable Format

Return a structured bibliography with:

```
### Area A: Direct Competitors

**Paper 1** — Author et al. (Year) — Venue. Citations: ~XX
  Relevance: 9/10 | Q1 grade: Q1-top | Risk: COMPETES
  One-line: "They show RL beats heuristic in inventory; we show PPO beats frontier at bottleneck."
  Differentiation: "Unlike Author et al. who study retail replenishment, we validate on a 13-operation thesis-grounded DES with order-level Excel metrics."
  Must-cite urgency: HIGH

**Paper 2** — ...
```

For each of the 5 areas, provide at least 5-10 papers. Total target: 30-50 papers.
