# Q-R1 factorial v3 external preflight audit

## Verdict

`STOP_INSTRUMENT_CONTRACT_IMPLEMENTATION_MISMATCH`

The factorial design is scientifically useful.

The VPS execution launched on 2026-07-26 is not an admissible execution of the
frozen contract.

The five full workers and a later diagnostic worker were terminated before a
full result was produced.

No confirmation root was opened.

## What passed

The contract was frozen before the first VPS launch.

The contract hash in the freeze receipt matches the contract.

The within-campaign persistence is explicitly frozen at rho 0.90.

The runtime identity probe reconstructed the expected rho-0.90 skeleton.

The identical-calendar objective matched the exact frontier lookup with error
zero.

The four declared prior/hidden-state factor combinations exist.

The predecessor run using the per-batch action mapping was correctly classified
VOID.

## Blocking discrepancies

### 1. The static-bar split was not executed as frozen

The contract assigns selection roots 7662101--7662116.

The runner defaults to only eight roots.

The full VPS commands overrode that default with `--bar-roots 4`.

The smoke used only root 7662101.

Therefore the deployed static bar was not selected on the contracted selection
split.

### 2. Runtime choices were not frozen

The contract does not freeze:

- total training timesteps;
- RecurrentPPO hyperparameters;
- the number of roots used for the static bar;
- checkpoint cadence or checkpoint-selection rule.

All remain mutable CLI or library defaults.

The `--seed-indices` implementation was added after the contract freeze.

The five workers started at 20:10:17 UTC.

The commit recording that implementation was created at approximately 20:12
UTC.

The VPS directory has no `.git`, so the execution cannot attest an immutable
source commit or clean worktree.

### 3. The contracted gates cannot be adjudicated

The contract freezes kappa cells 0.50, 0.75 and 0.90.

The runner evaluates only 0.75 and 0.90.

It therefore cannot test the iid-null gate or the frozen dose response.

### 4. Neural premium is declared but not implemented

The runner evaluates the four neural factorial arms and a static bar.

It does not evaluate or join the retained structured MPC on the same histories.

It therefore cannot compute:

`P1_H1 - strongest_tested_retained_structured_controller`.

### 5. Checkpoint selection is not implemented

The contract says checkpoint selection uses only the selection split.

The runner trains one final 96,000-step checkpoint.

It computes `selection_mean_ret` after training but does not select among
checkpoints.

### 6. Mandatory secondary service disclosure is absent

The runner does not report worst-product fill, unresolved orders, lost orders
or the complete service ledger.

### 7. Custody was not opened explicitly

The freeze receipt still reports training and selection roots unopened.

No development-opening receipt records the actual command, runtime, source
hashes, dependency environment or opened roots.

The smoke nevertheless opened development data and optimizer seed 7663001.

The full commands started all five optimizer seeds and began enumerating
selection roots 7662101--7662104.

## Custody consequence

Treat the following as burned development:

- training roots 7662001--7662040;
- selection roots 7662101--7662116;
- optimizer seeds 7663001--7663005.

This conservative block-level burn prevents later selection based on which
individual histories happened to be touched before termination.

Confirmation roots 7662201--7662264 remain unopened.

The smoke result and partial logs are instrumental evidence only.

## Required successor

Preserve the four-arm factorial estimands.

Freeze the complete executable protocol before opening new roots:

- rho and kappa separately;
- all three kappa cells, including iid;
- complete training and selection root ranges;
- static-bar root count equal to the full selection split;
- optimizer and model hyperparameters;
- training budget and checkpoint cadence;
- checkpoint-selection rule;
- retained MPC comparator and join keys;
- service disclosures;
- execution commit, environment and artifact hashes;
- one development-opening receipt.

Run the static-bar calculation once and share the immutable artifact across
optimizer seeds.

Do not open a new development block until implementation tests prove that every
contracted estimand and gate can be emitted.
