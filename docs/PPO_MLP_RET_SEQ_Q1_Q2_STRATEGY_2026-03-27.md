# PPO + MLP + ReT_seq_v1: Q1/Q2 strategy and environment audit

Date: 2026-03-27

## Working decision

The repo should concentrate on a single paper backbone first:

- Algorithm: PPO with MLP policy
- Environment family: shift-control DES wrapper
- Reward: ReT_seq_v1
- Main reward parameter: kappa = 0.20

Reason: the highest-ROI publication path is not architectural novelty. It is a
defensible environment, a resilience-grounded reward, and clear evidence that
the agent learns adaptive behavior under disruption.

## Journal strategy

### Best overall target

1. Computers & Industrial Engineering (CAIE)

Why:
- Strong fit for industrial engineering + computerized decision methods.
- Good home for DES + RL + operational policy adaptation.
- Better fit than a pure AI journal if the contribution is "validated
  environment + adaptive control policy + resilience metrics".

Submit here if the final manuscript centers on:
- the DES environment as a serious operational model,
- the reward contract as the main methodological contribution,
- adaptive shift/inventory control under disruption,
- cross-scenario benchmarking against static baselines.

### Best AI-facing target

2. Engineering Applications of Artificial Intelligence (EAAI)

Why:
- Strong fit if the paper foregrounds the RL method as an engineering AI
  application.
- Good outlet if the story is "resilience-aware reward enables learning in a
  mission-critical supply chain simulator".

Submit here if the final manuscript centers on:
- PPO + ReT_seq_v1 as the core scientific contribution,
- proof-of-learning artifacts,
- replicable engineering benchmark design.

### Best simulation-facing target

3. Simulation Modelling Practice and Theory (SMPAT)

Why:
- Strong fit if the paper emphasizes the DES model, validation, verification,
  and the coupling of simulation with RL.
- This is the cleanest home if the environment audit and simulation argument
  become the main strength.

Submit here if the final manuscript centers on:
- model construction and validation,
- why the DES is credible,
- how RL uses the simulator to learn adaptive control.

### High-prestige but harder fits

4. International Journal of Production Economics (IJPE)
5. Transportation Research Part E (TRE-E)

Why harder:
- These journals want stronger managerial/economic generalization than "an RL
  agent learned in one military-inspired simulator".
- To be competitive there, the paper needs a broader supply chain insight:
  regime-dependent value of adaptive capacity, cost-service-resilience tradeoff,
  and stronger external validity.

### Not recommended as first target right now

6. Reliability Engineering & System Safety (RESS)

Reason:
- Interesting stretch target, but only if the paper is reframed around
  reliability/resilience methodology and the environment validation is made much
  tighter than it is now.

## Local environment audit

### What looks structurally correct

1. The DES backbone still preserves hourly assembly-line granularity.
   - `supply_chain/supply_chain.py` launches `_assembly_hourly()`.
   - `_assembly_hourly()` advances with `yield self.env.timeout(1)`.
   - The RL wrapper applies weekly or custom decision intervals on top of an
     hourly internal simulator, which is methodologically acceptable.

2. Macro deterministic throughput is plausible.
   - A deterministic two-year post-warmup run currently gives about:
     - `warmup_time = 919`
     - `avg_annual_delivery = 732500`
     - `avg_annual_production = 738432`
   - This remains broadly aligned with the repo's deterministic validation
     tolerance.

3. Observation-space structure is internally consistent.
   - `v1 = 15 dims`
   - `v2 = 18 dims`
   - `v3 = 20 dims`
   - `v4 = 24 dims`
   - The declared Gym box is nonnegative with upper bound `20.0`, and short
     rollout checks did not exceed it.

4. Action-space structure is internally consistent.
   - Action shape is `(5,)`.
   - Dims `[0:4]` map to inventory multipliers in `[0.5, 2.0]`.
   - Dim `[4]` maps to shifts `{1, 2, 3}`.
   - This is a coherent RL extension of Garrido's static scenario controls.

### The main scientific risk

The current RL reset regime is the biggest publication vulnerability.

At `env.reset()`, the environment skips directly to:
- `warmup_hours = 838.8`

But the actual simulator state at that time is not a healthy operating regime.
Repeated checks on `current`, `increased`, and `severe`, across multiple seeds,
all start from the same degraded state:

- `fill_rate ~= 0.0345`
- `backorder_rate ~= 0.9655`
- `pending_backorders_count = 29`
- `rations_theatre = 0`

Implication:
- the agent is not starting from "post-warmup stable operations",
- it is starting from a pre-existing backlog crisis,
- so PPO is currently learning crisis recovery plus ongoing adaptation, not
  clean adaptive resilience from a nominal operating state.

This does not make the environment unusable, but it must be fixed or made
explicit in the paper. As it stands, this is the strongest reviewer attack
surface.

### Warmup inconsistency that drives the issue

- `config.py` states the warmup estimate is `838.8h` until the first batch
  reaches Op9.
- `supply_chain.py` marks `warmup_complete` when total production reaches one
  batch in assembly, which currently lands around `919h` in deterministic mode.
- The RL wrapper does not use the simulator's warmup flag. It always skips to
  the fixed config estimate.

So the paper-facing environment contract and the live DES state are not aligned.

### Observation-space assessment

Current assessment:
- The observation space is syntactically correct for Gymnasium.
- `v4` is the best current representation for learning because it adds recent
  dynamics, cumulative diagnostics, current shifts, and upstream disruptions.
- `v1` is simpler and cleaner for a benchmark contract, but it hides dynamics
  that matter for adaptation.

Publication recommendation:
- For the first paper, freeze exactly one observation contract.
- If you want the simplest story, use `v1`.
- If you want the strongest learning signal, use `v4`.
- Do not mix multiple observation contracts in the main result section.

### Action-space assessment

Current assessment:
- The shift action is strong, interpretable, and scientifically meaningful.
- Inventory-control dimensions are valid, but their effect is slower and less
  directly interpretable than the shift action.

Quick behavior checks show:
- after one step, shift and inventory changes barely change fill/backorder
  because the system is still dominated by inherited backlog;
- after several steps, shift choice clearly changes outcomes;
- inventory-control dimensions do affect downstream inventory and service, but
  with delayed and noisier effects.

Publication recommendation:
- Keep the 5D action space available in the repo.
- But for the first paper, be ready to run a shift-only ablation if learning
  with all five controls looks unstable or weakly interpretable.

## What to do next before submission

1. Fix or explicitly redefine reset semantics.
   - Best option: reset into a true post-warmup operating state, not a backlog
     collapse state.
   - If not fixed, the paper must clearly say it studies adaptive recovery from
     an already degraded system state.

2. Freeze the paper contract.
   - PPO + MLP
   - ReT_seq_v1
   - kappa = 0.20
   - one observation version
   - one primary training regime

3. Keep the contribution narrow and strong.
   - reward contract,
   - validated DES-backed environment,
   - proof that RL adapts under disruption,
   - cross-scenario evaluation against static baselines.

4. Treat architecture novelty as phase 2.
   - RecurrentPPO is the next rational algorithmic extension.
   - KAN / GNN / DKANA should not distract the first paper unless PPO + MLP is
     already solid and reproducible.

## Recommendation in one sentence

Yes: concentrate on PPO + MLP + ReT_seq_v1. But do not submit until the reset
state / warmup semantics are cleaned up or explicitly justified, because that is
currently the main methodological weakness of the environment.
