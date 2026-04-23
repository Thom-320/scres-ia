# Minimal Additions to v.0 — Thom's DES Contributions Only

Instructions: paste each block into the corresponding section of the v.0 Word document.
Do NOT change David's sections (3.3, 4.2) or Garrido's framing (Intro, Hypotheses).

---

## 1. Addition to Section 3.2 (after the last paragraph about warm-up)

> INSERT AFTER: "...only after that point are steady-state observations interpreted."
> INSERT BEFORE: "In this way, the DES provides a standalone resilience model..."

**New paragraph to add:**

To enable integration with learning algorithms, the DES was wrapped in a Gymnasium-compatible interface (Towers et al. 2024). At each decision epoch—set at 168 simulated hours (one operational week)—the interface exposes a state observation vector derived from the simulation's internal variables and accepts a continuous action vector that modifies operational parameters before advancing the simulation by one step. The interface supports multiple observation versions (ranging from 15 to 46 dimensions) and action contracts (from 5 to 7 continuous dimensions), allowing systematic comparison of different control scopes over the same DES backbone. This modular design means that the DES validation described in this section remains unchanged regardless of which learning algorithm is coupled to it.

---

## 2. Expansion of Section 4.1 (after Table 4 and the deterministic results paragraph)

> The v.0 already has the deterministic validation and Table 4.
> ADD the following paragraphs and table AFTER the existing 4.1 content.

**New content to add:**

To characterize the DES behavior under disruption, two stochastic risk configurations were evaluated over the 20-year simulation horizon. Table 5 summarizes the results.

**Table 5.** DES performance under stochastic risk configurations

| Configuration | Avg. annual delivery | Fill rate | Total backorders | Disruptive events |
|---|---|---|---|---|
| Deterministic (Cf0, S=1) | 733,621 | 99.3% | 41 | 0 |
| Current risk | 677,750 | 68.3% | 1,825 | 8,247 |
| Increased risk | 549,250 | 45.6% | 3,132 | 13,705 |

Under deterministic conditions with risks disabled, the simulation produces 734,458 rations per year and delivers 733,621, with a fill rate of 99.3% and only 41 backorders accumulated over the 20-year horizon. These results indicate that the baseline network is well configured for nominal demand. At the same time, the weekly demand of roughly 17,500 rations against a single-shift production capacity of approximately 17,948 rations shows that the assembly line already operates near saturation.

When stochastic risks are enabled at the current level, average annual delivery falls to 677,750 rations (−7.6% relative to the deterministic baseline), the fill rate drops from 99.3% to 68.3%, and total backorders rise from 41 to 1,825. Across the 20-year horizon, the model registers 8,247 disruptive events, mainly driven by internal workstation breakdowns (R11, 1,868 events), defective products (R14, 5,591 events), and recurrent demand surges (R24, 478 events), together with one black-swan interruption lasting 672 consecutive hours.

A stronger stress configuration with increased risks confirms the same structural tendency. Under that setting, average annual delivery decreases to 549,250 rations, the fill rate falls to 45.6%, total backorders increase to 3,132, and the number of disruptive events rises to 13,705. The DES therefore shows a clear transition from a well-calibrated nominal system to a highly fragile network once disruption intensity escalates.

Taken together, these findings establish the DES as both empirically credible and analytically informative: it reproduces the thesis baseline with acceptable error, while making visible how disruption propagation erodes service performance under stochastic risk.

---

## 3. Section 2 — Background: minimal addition

> REPLACE the placeholder "XXX et al. (2025) developed a theoretical analysis..."
> WITH the following paragraph. This is just enough to hold the section together
> until we write the full related work.

**Replacement paragraph:**

The literature on RL applied to supply chain management has grown substantially in recent years. Comprehensive reviews by Yan et al. (2022) and Rolf et al. (2023) classify a wide range of RL algorithms and applications in logistics and SCM, noting that inventory control is the dominant application domain. Rolf et al. (2025) review 72 papers combining DES with ML, confirming that DES+RL is the most frequent hybrid paradigm. In the resilience domain, Ding et al. (2026) use multi-agent RL (MAPPO) for supply chain resilience reconfiguration under disruption, though their focus is network topology redesign rather than operational control on a detailed DES. Garrido et al. (2024) proposed integrating AI algorithms into SCRES simulation models, identifying neural networks and reinforcement learning as the most suitable candidates for bridging the gap between static simulation and adaptive learning. The present work operationalizes this proposal.

---

## Notes for Thom

- These additions are ONLY from your DES work. Nothing about Track A/B results yet.
- The literature paragraph gives Garrido and David enough context to understand we are NOT first movers, and positions against Ding.
- When you're ready to share the full results, use the manuscript in docs/manuscript_current/.
- The literature_links.md file (separate) gives them the references they were missing in the WhatsApp chat.
