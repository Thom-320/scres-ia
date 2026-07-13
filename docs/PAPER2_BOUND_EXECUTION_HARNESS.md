# Paper 2 bound execution harness

`scripts/paper2_bound_execution_harness.py` is the single local/VPS execution
entrypoint for the reduced W12/W16 transducer certificates, the W24 profile/state
audit, and the later authorized full frontier. A submission or completed process
is never evidence by itself; retrieval, checksum verification, and independent
audit remain required.

## Immutable prerequisites

Run from a clean, tracked `HEAD` with the intended Python environment active.
The harness records and rechecks the commit, primary contract, result contract,
runner, harness, Python ABI/package/requirements environment digest, seed
manifest, and command manifest. The frozen modes reject seed, context, split,
horizon, runner, or output-path overrides that differ from
`contracts/paper2_bottleneck_primary_bound_v2.json`.

For VPS execution, first run `remote-preflight`. It writes
`scientific_environment.json`, containing the exact remote Python ABI, package
versions and tracked-requirement hashes. Supply that file to `prepare` with
`--scientific-environment-json`; the preparing machine must match every field
except the platform-specific SOABI, and the executing VPS must reproduce the
complete digest exactly.

The contractual outputs inside each run directory are:

- `results/paper2_bottleneck/exact_transducer_certification_w12.json`
- `results/paper2_bottleneck/exact_transducer_certification_w16_hard.json`
- `results/paper2_bottleneck/w24_profile_state_audit.json`

## Prepare and execute locally

Prepare one mode without opening a tape:

```bash
./.venv/bin/python scripts/paper2_bound_execution_harness.py prepare \
  --mode reduced_w12 \
  --run-id paper2-reduced-w12 \
  --run-dir outputs/paper2-reduced-w12 \
  --scientific-environment-json outputs/paper2-remote-preflight/scientific_environment.json \
  --runner-workers 4
```

Use `--mode reduced_w16` or `--mode w24_audit` with a distinct run id and
directory for the other gates. Execute a prepared envelope explicitly:

```bash
./.venv/bin/python scripts/paper2_bound_execution_harness.py execute \
  --run-dir outputs/paper2-reduced-w12 \
  --location local

./.venv/bin/python scripts/paper2_bound_execution_harness.py verify \
  --run-dir outputs/paper2-reduced-w12
```

Each execution writes stdout/stderr, heartbeat, per-tape and job status,
machine-readable progress, an execution receipt, runtime environment snapshots,
and an artifact checksum manifest.

## VPS flow

Preparation remains local. Then stage the exact Git commit and control envelope,
launch it, poll it, and retrieve it:

```bash
./.venv/bin/python scripts/paper2_bound_execution_harness.py stage-vps \
  --run-dir outputs/paper2-reduced-w12

./.venv/bin/python scripts/paper2_bound_execution_harness.py launch-vps \
  --run-dir outputs/paper2-reduced-w12 \
  --remote-python '~/scres-ia/.venv/bin/python'

./.venv/bin/python scripts/paper2_bound_execution_harness.py remote-status \
  --run-dir outputs/paper2-reduced-w12

./.venv/bin/python scripts/paper2_bound_execution_harness.py retrieve-vps \
  --run-dir outputs/paper2-reduced-w12
```

Retrieval creates `retrieved/` and immediately verifies all recorded hashes.
The result remains `NOT_EVIDENCE` until independent scientific audit.

## Full-frontier authorization without commit circularity

For a future full `scientific` frontier, first prepare without authorization.
This freezes the seed and command hashes and writes
`authorization_required_template.json`. Fill a separate control JSON that pins
the prepared Git `HEAD` and the tracked W12/W16/W24 artifacts, then seal the
existing run directory without regenerating either manifest:

```bash
./.venv/bin/python scripts/paper2_bound_execution_harness.py seal \
  --run-dir outputs/paper2-frontier-calibration \
  --authorization-json /secure/control/authorization.json
```

The authorization JSON itself is control-plane input and need not be committed;
the certificates and W24 audit that it references must be tracked and identical
to the pinned `HEAD`.
