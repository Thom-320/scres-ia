# Repository Source of Truth

This note freezes the current paper-facing repository story. Treat it as the
primary reference for README examples, manuscript edits, benchmark commands,
and reviewer-facing explanations unless a later note explicitly supersedes it.

> **Superseded lane notice (2026-07-10).** Everything below replaces the
> pre-Track-B version of this document, which described the
> `shift_control`/`ReT_seq_v1`/`v1` lane as the frozen paper backbone. That
> lane and its 500k bundles under `outputs/paper_benchmarks/` are HISTORICAL
> context only. The claim-by-claim authority is
> `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`.

## Canonical scientific story (Paper 1)

The manuscript in `docs/manuscript_current/submission/elsevier/` makes one
central claim:

> In a thesis-grounded military food supply-chain DES, a learned policy (PPO)
> improves Garrido/Excel resilience over dense static frontiers when — and,
> in the tested contracts, only when — the action contract exposes the
> downstream dispatch bottleneck (Op10/Op12). The measured gain is adaptive
> recovery, not anticipatory prevention.

The repository roles that must not be conflated:

- `thesis_faithful` lane (`docs/thesis_faithful/CONTRACT.md`,
  `scripts/run_thesis_faithful.py`): strict Garrido-Rios reproduction, a
  validation gate for the DES, not a training benchmark.
- **Track A** (`track_a_*` contracts): the thesis-grounded buffer/shift
  decision family. Boundary result: no tested learner converts the measured
  oracle headroom (claims registry C8).
- **Track B** (`track_b_v1`, 8D): the canonical positive lane — upstream
  qty/ROP + Op5 + shift + Op10/Op12 dispatch. Primary result C1/C21.
- **Track B-P** (`track_bp_v1`, 11D; `supply_chain/track_bp_env.py`): the
  Paper-2 extension lane (strategic reserve postures under lead-time
  commitment). Outside Paper 1. See C28/C29.

## Frozen benchmark backbone (Track B canonical)

- Environment factory: `external_env_interface.make_track_b_env()`
- Action contract: `track_b_v1` (8D)
- Training reward: `control_v1`
- Observation: `v7` (note: 48-dim at the time seeds 1-5 were trained; 52-dim
  for seeds 6-10 — four tail fields appended between runs; disclosed and
  handled by exact slicing in held-out evaluation)
- Risk level: `adaptive_benchmark_v2`; horizon h104 (weekly steps, 168 h)
- Year basis: `thesis`; stochastic PT: on; learning rate 3e-4; 60k timesteps
- Primary metric: `ret_excel` (Garrido/Excel ReT). Never `ret_thesis`.

## Primary artifact bundles (current)

- Headline 10-seed paired dense-CRN stats:
  `docs/track_b_q1_stats_2026-07-02_final_10seed/`
- **Crossed held-out evaluation (Blocker 1, 2026-07-09):**
  `outputs/experiments/track_b_crossed_eval_2026-07-09/` — 10 checkpoints x
  60 fresh tapes (eval seeds 200001+), Excel ReT delta `+0.000486`, two-way
  CI95 `[+0.000456, +0.000517]`, 10/10 checkpoints and 60/60 tapes positive.
- Corrected decision-contract factorial (Blocker 2, mechanism gate):
  `outputs/experiments/track_b_factorial_{joint,upstream_shift,dispatch_only}_2026-07-09/`
- Frozen checkpoints: see `docs/REPRODUCIBILITY.md`.
- E3 cross-regime + dense-frontier: `docs/track_b_q1_stats_2026-07-02_final/`
  and `outputs/experiments/track_b_e3_dense_frontier_2026-07-02/` (use the
  conservative dense-best values; see C11 provenance note).

## What is not the main paper lane

Valuable but secondary or retired:

- `shift_control`/`ReT_seq_v1` 500k lane (historical; pre-Track-B)
- `ReT_thesis` as a training reward or reported metric
- KAN / DKANA / GNN as a contribution (sidecars only)
- SAC/TD3 beyond the screen-scale scope check
- Prevention/anticipation claims (retracted; boundary result only — C25/C26)
- H4 retained/reset as a central theory (small effect; future work)
- Track B-P reserve postures (Paper 2, gated; C28/C29)

## Document hierarchy

When documents disagree:

1. `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` (claim-by-claim authority)
2. This file
3. `docs/REPRODUCIBILITY.md`
4. Dated verdict documents (`docs/*_VERDICT_*.md`, autopsies, audits)
5. Historical reports, manuscript notes, and meeting notes

## Required language discipline

Preferred: "thesis-grounded reconstruction with forensic workbook replay and
throughput checks"; "adaptive recovery"; "decision-contract/action-space
alignment (pattern evidence pending the corrected factorial)"; "boundary
result"; "no detected difference at current precision".

Avoid: "validated digital twin"; "empirically validated"; the invented
"±15% validation threshold"; "prevention"/"anticipation"; "organizational
learning"/"path dependency"; "worst-case" for p99 statistics; "equivalent"
for a CI that spans zero; "regardless of algorithm choice"; "full 8D static
frontier" for the downstream 147-cell enumeration; "first DES–RL for SCRES".
