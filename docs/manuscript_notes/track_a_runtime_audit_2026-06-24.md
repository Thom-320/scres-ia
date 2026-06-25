# Track A Runtime Audit - ReT_cd rho Sweep

## What Ran

Command audited:

```text
scripts/pilot_learning_regime.py
--label sweep_retcd_v1
--reward-mode ReT_cd
--rhos 0.3334,0.6,0.9
--cycles 8
--online-timesteps-per-cycle 1200
--pretrain-timesteps 1500
--learning-starts 100
--buffer-size 5000
```

Observed runtime was about 33 minutes. The process used one CPU core heavily
throughout and memory remained stable, so it was computing rather than stalled.

## Why It Took So Long

The expensive part is not a single SimPy block. A static `run_episode` block with
`max_steps=8` takes roughly `0.06s` on this machine.

The expensive part is the interaction between DQN timesteps and
`decision_cadence=block`:

```text
1 DQN timestep = 1 BlockDecisionEnv step
1 BlockDecisionEnv step = a full disruption block
1 disruption block = up to max_steps weekly DES steps
max_steps = 8
```

So `online_timesteps_per_cycle=1200` means about 1200 full 8-week simulated
blocks, not 1200 lightweight weekly decisions.

Per rho, the command executed approximately:

```text
pretrain                       = 1,500 DQN block timesteps
retained updates: 8 * 1,200     = 9,600 DQN block timesteps
reset updates:    8 * 1,200     = 9,600 DQN block timesteps
total per rho                  = 20,700 DQN block timesteps
```

Across three rho values:

```text
3 * 20,700 = 62,100 DQN block timesteps
62,100 * 8 = 496,800 base weekly DES steps
```

This excludes the smaller evaluation overhead:

```text
3 rho * 8 cycles * 3 policy arms * 8 weekly steps = 576 base weekly DES steps
```

The runtime is therefore dominated by DQN learning, especially the retained and
reset online updates. The reset arm is intentionally expensive because it reloads
the initial checkpoint and adapts from scratch for every cycle.

## Methodological Implication

The current setting is more than a smoke. It gives each arm a large synthetic
within-cycle adaptation budget:

```text
1200 full disruption blocks per cycle per adaptive arm
```

That may be defensible as an algorithmic pilot, but it is much larger than the
literal experience available from one observed disruption block. For the paper,
the data budget must be described as synthetic online training budget, not as
one real-world disruption observation.

## Output Limitation Found

The pilot wrote only final output. During the run, `/tmp/sweep_retcd_v1.log`
remained empty because the script had no progress prints before final summary.
This made the run look opaque despite active CPU use.

Patch applied after the run:

- progress print at rho start;
- progress print after initial model creation;
- progress print after each cycle;
- elapsed seconds per rho in `pilot.json`;
- run configuration and estimated timestep counts in `pilot.json`.

## Recommendation

For future go/no-go pilots:

1. Keep `ReT_cd` fixed unless the contract is explicitly amended.
2. Use a smaller first-pass budget before full sweeps, e.g.:

```text
--cycles 4
--pretrain-timesteps 300
--online-timesteps-per-cycle 100
```

3. If the goal is a realistic retained-learning contrast, pre-register the
online update budget as a substantive design choice. In block cadence,
`online_timesteps_per_cycle` should be interpreted as a number of synthetic
block-level learning episodes.
4. Consider adding a cached/static robust baseline when the pilot does not need
to re-select static policies.
5. Keep the final confirmatory run larger, but only after the rho pilot shows
the desired dose-response direction.

