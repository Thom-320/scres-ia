# SCRES-IA — objective and decision hierarchy

## The objective, and it does not change

**Answer the two questions Garrido, Pongutá & Adarme pose in *Enhancing the Operationalization of
SCRES-Based Simulation Models with AI Algorithms* (ICCL 2024, LNCS 15168, pp. 80–94):**

1. **What category of AI algorithms best mimics the supply-chain-learning (SCL) attribute?**
2. **How can that family be integrated into the internal structure of a DES model for SCRES
   assessment?**

Their Fig. 2 marks the gap precisely: nodes ③ (data gathering) and ⑧ (verification & validation)
are **the two ends of an open-loop supply chain**. An AI algorithm placed between them is what
converts it into a **closed loop**. They call the absence "the **Alzheimer's effect**": the
modelled network cannot retain what it learned from earlier runs. Their Fig. 5 is the bridge —
a neuron whose dendrites are the four SCRES drivers `d_i`, weighted by `ρ`, with an activation
function such as *"is the SCRES measure at configuration x higher than at configuration x−1?"*

**The clock is real.** The paper is from 2024, it is nearly 2027, and the gap is published and
open to anyone.

## The decision hierarchy — read top to bottom

1. **Answer Garrido's questions.** Everything else is instrumental. If a piece of work does not
   move one of the two questions, it is not the priority.
2. **Maximise resilience.** That is the object being operationalised.
3. **Create headroom first.** There is no point training on a problem with nothing to capture.
   Headroom means: an environment where the optimal action *varies with state*, the surface is
   *non-linear*, and a static policy or a well-trained MLP *loses*. Measure `H_regime` **before**
   spending on a learner.
4. **Only then train.** Ladder: constant → threshold → MLP → PPO → recurrent PPO. The MLP is
   trained properly; a straw-man comparator proves nothing.

## Standing constraints from the PI

- **We do not depend on Garrido.** Where the thesis does not settle a fact, **we decide it**,
  declare it as our assumption, and price it. `blocked_domain_fact` is not a terminal state.
- **The code is not written in stone.** This codebase was built by older models. Where something
  is wrong, change it.
- **No artificial or useless blockers.** Guardrails must earn their place; several of our own
  past constraints — not Garrido's — are what killed the only headroom we ever found.
- **Never lose a lane.** `docs/PROMISING_LANES_REGISTRY.md` is living. When the model or the base
  changes, **re-test hypotheses that failed before** — a negative under old physics is not a
  negative under new physics.
- **Be critical.** Report what the measurement says, including when it refutes something I wrote
  an hour earlier.
- **Save findings** so a context compaction does not lose the thread.

## Two models, on purpose

| layer | what it is | what it is for |
|---|---|---|
| **thesis-faithful DES** | frozen; reproduces Garrido-Ríos (2017) Ch. 6 to −4,43 % on ECS | validation (§4.1 of the manuscript) |
| **extended DES** | our decision environment; new physics allowed | headroom and the learner |

Every departure from the thesis is declared as **our** assumption, with its fidelity price
**measured** — never presented as if it came from the thesis.

## Where headroom lives (measured, not assumed)

**Contention over a scarce, non-fungible shared resource.** Program O measured `H_PI = 0,1515`
(LCB95 0,1156) under a non-fungible two-product share of Op5–Op7, and — the decisive control —
**with the resource made fully fungible the headroom is exactly 0**. That is a causal mechanism,
about 1.000× the `1e-4` scale of the thesis-native envelope.

Three measured reasons the thesis-native envelope has none: the assembly line runs at a **2,6 %
capacity margin** (17.948 vs ~17.500 rations/week at S=1), so there is no allocation decision;
the highest-weight ReT branch (**autotomy**, weight 1,0) is **structurally unreachable** because
`GARRIDO_FULFILLMENT_DELAY_HOURS = 54 > LT = 48`; and at `op12` the **uninformed placebo beats
the state-conditioned rule**, so the value is in the period varying, not in what varies it.

## Scientific discipline (non-negotiable)

- **Preregister before running.** Falsifiers must state *why they can fail*.
- **Uninformed placebo** in every headroom measurement.
- **Virgin, disjoint seeds** for every confirmation; the fit never reads the test block.
- **Never edit a frozen contract or a dated artifact in place.**
- Measure through the pipeline (`arm_runner.py`), never with an ad-hoc script.
