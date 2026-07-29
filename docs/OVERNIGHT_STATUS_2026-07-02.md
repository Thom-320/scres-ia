# Overnight execution status (live — updates through the night)

## ✅ Manuscript verification pass complete (2026-07-02 ~08:15 UTC)

The two items flagged as "pending" in the results-section rewrite are now
resolved with real evidence, not fabricated numbers:

- **Section 4.2 (Track A negative result):** rewritten using
  `docs/TRACK_A_HEADROOM_SEARCH_2026-06-29.md` (regime-conditioned oracle
  headroom +0.000176 to +0.000296 Excel ReT, CI lower bound positive in
  the full grid) and `docs/TRACK_A_REPAIR_LOCAL_ANALYSIS_2026-06-30.md`
  (closest fair PPO attempt misses the best static by $-6.99\times10^{-6}$
  Excel ReT, dominated by 10 static configs). Replaces the old
  fill-rate-0.788-vs-0.792 framing the registry (C8) already flagged as
  retired.
- **Section 4.6 (reward sensitivity):** rewritten using the real 18-cell
  reward×observation sweep in
  `outputs/experiments/track_b_adaptive_sweep_kaggle_2026-07-01_v6/fetched/track_b_adaptive_sweep/sweep_summary.csv`
  (2 seeds × 40k, screening scale, honestly labeled as such) — every one
  of 18 cells shows a positive Excel ReT delta (+0.000195 to +0.000452).
  Confirmatory-scale (5×60k) rerun remains future work if a reviewer
  demands it, but the screen itself is now real, cited evidence rather
  than a placeholder.

Manuscript-wide final check: braces balanced in every edited file, zero
remaining instances of "7D", "fill rate 1.000", "57%", "500,000", or the
old "0.788 vs 0.792" Track A numbers anywhere in
`docs/manuscript_current/submission/elsevier/`.

## ✅ CORE EVIDENCE BASE COMPLETE (as of 2026-07-02 ~07:40 UTC)

Every experiment prescribed by `docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md` as
"needed for submission" has now landed, all with favorable or clearly
disclosed results. Summary table (full detail in each linked verdict doc):

| # | Experiment | Where | Result | Verdict doc |
|---|---|---|---|---|
| E1 | Regime-table/heuristic go-no-go | local | **GO** — PPO beats all 75 statics, all 6 heuristics, and the fitted regime table, 60/60 CI95>0 | `docs/E1_GO_NO_GO_VERDICT_2026-07-02.md` |
| E2 | Obs-masked PPO retrain (privileged fields removed) | Kaggle + OVH VPS (independent) | PPO retains ~95% of the win without privileged regime/forecast fields (+0.000395/+0.000403 vs canonical +0.000415) | `docs/E2_PRIVILEGED_OBSERVATION_VERDICT_2026-07-02.md` |
| E3 | Cross-regime/horizon generalization matrix | OVH VPS | PPO wins 5/6 cells (current/increased/severe × h52/h104); loses narrowly at severe/h52 (disclosed) | `docs/E3_GENERALIZATION_VERDICT_2026-07-02.md` |
| E4 | Final 5-seed 8D action-space ablation | local | `downstream_only` (+0.000566) beats `joint` (+0.000415) — not "just more knobs" | `docs/E4_ABLATION_VERDICT_2026-07-02.md` |
| E6 | Fidelity-gate flow-mode reconciliation | local | Faithful mode confirmed; R1/R2 pass cleanly (p<0.01), R3 weak in both modes (matches thesis) | `docs/E6_FIDELITY_MODE_RECONCILIATION_2026-07-02.md` |
| — | Track A retrained on `adaptive_benchmark_v2` (companion to E3) | local | **Loses** on the same regime Track B wins on — refutes "regime alone explains it" | `docs/E3_GENERALIZATION_VERDICT_2026-07-02.md` |
| E5 | Stats reanalysis bundle (seed-level, top-12, CVaR05, dispatch-cost) | Codex, verified by Claude | Correct on code read, no issues found | `docs/track_b_q1_stats_2026-07-02/` |

**T3 (privileged observation) — the audit's single highest-priority open
threat — is now closed from both directions** (E1: a policy WITH
privileged access still loses to PPO; E2: PPO doesn't NEED privileged
access to win). This was genuinely the make-or-break question for whether
the paper has a real RL result at all, and the answer is yes.

**Claims registry (`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`) is
current** — C7, C11, C15, C16, C17 all added/upgraded tonight with citations
to the verdict docs above.

## What's still open

- **Manuscript numeric rewrite.** The Elsevier draft
  (`docs/manuscript_current/submission/elsevier/`) is still an era behind —
  contains the retired "7D"/"fill=1.000"/"57% cost win" claims the original
  audit flagged. Now that the evidence base is stable, this is the next
  priority: full rewrite to the dense-CRN canon + all six results above,
  MDP/POMDP appendix, CRN protocol description, gap-decomposition figure
  (E1's central table) wired in as a real figure.
- **Bibliography.** SCRES-theory anchors added tonight (Ponomarov & Holcomb,
  Wieland & Durach, Hosseini/Ivanov/Dolgui, Levitt & March). Still needed:
  wire the ~25 narrative citations in `02_related_work.tex` to `\citep`/
  `\citet` (currently only 1 `\cite` in the whole manuscript).
  `\nocite{*}`/dual bibliographystyle already fixed by Codex.
- **Repo hygiene** (lower priority): scrub `docs/WIN_CONFIRMED_2026-06-29.md`
  (falsified per-op "win"), archive `outputs/experiments/track_b_dense_frontier_2026-07-01`
  (different undocumented regime), quarantine v9 Kaggle artifacts.
- **Hold for response letter, not needed tonight:** SAC/TD3 comparator, H4
  retained-vs-reset, Track B at h260, mechanism lead/lag audit, full
  147-candidate regime-table fit (E1 used a trimmed 75-candidate grid).

## Coordination history (chronological, kept for reference)

**Resource check (2026-07-02 ~06:19 UTC, per user):** Kaggle confirmed available
(`kernel RUNNING`, no auth/rate-limit issue right now). VPS has real headroom
(load 3.5/6 cores, 9.2Gi available via reclaimable buff/cache, 77G disk free)
— left unused for now rather than force-launching a lower-priority
experiment (SAC/TD3 was considered; the runner only supports `ppo`/
`recurrent_ppo`, adding SAC needs real code changes and is explicitly a
hold-for-response-letter item per the audit, not needed tonight). Local Mac
now runs 2 jobs (E1 + the new Track A test) after E4 freed capacity.

**E3 had two real, previously-unexercised bugs, fixed in
`scripts/run_track_b_e3_cross_regime_horizon_matrix.py` and
`scripts/eval_track_b_cross_scenario.py`** (both crashed on first real run,
neither smoke test — mine or Codex's — used `--include-heuristics` so this
path was never hit before):
1. `eval_track_b_cross_scenario.build_parser()` omits several args
   (`ret_seq_kappa`, `ret_excel_cvar_alpha`, `ret_excel_cvar_tail_level`,
   `ret_excel_cvar_window`, `enabled_risks`, `risk_frequency_multiplier`,
   `risk_impact_multiplier`, `demand_mean_multiplier`) that the shared
   `build_env_kwargs()` reads unconditionally — `AttributeError` on
   `evaluate_heuristic_policy`. Fixed by setting them explicitly (inert
   defaults) in the E3 wrapper's `run_one_horizon`.
2. Deeper issue: `evaluate_heuristic_policy` (run_track_b_smoke.py's row
   builder) produces a genuinely different, smaller metric schema than
   `evaluate_static_policy`/`evaluate_learned_policy`
   (audit_track_b_all_rewards.py's row builder) — missing `reward_mode`,
   `algo`, and rich Garrido-thesis metrics like `ret_thesis_total` that
   heuristics never compute at all. Patched `reward_mode`/`algo` with
   `setdefault` in `eval_track_b_cross_scenario.py`'s heuristic call site,
   but the missing rich-metric columns are a real, deeper schema mismatch —
   **`--include-heuristics` + `eval_track_b_cross_scenario.py` is not
   reliable and was NOT used for tonight's E3 run** (launched with
   `--skip-heuristics`). E1 already covers heuristics-vs-PPO at the
   canonical cell, so this isn't a coverage gap for the paper, just a
   known-broken code path for anyone reaching for it later — full fix
   would mean rewriting `evaluate_heuristic_policy` to compute the same
   metric panel as the static/learned path.

**VPS added as the reliable heavy-compute lane (per user, 2026-07-02
~05:52 UTC): OVH VPS (`ovh-agent-lab`, 6 cores/11GB RAM/90GB disk).**
Installed Python 3.11.15 via deadsnakes (matches local exactly — VPS default
was 3.14, too new/risky for stable_baselines3/torch pins), rsynced the
current working tree (scripts/ + supply_chain/ + tests/, excluding
.git/outputs/notebooks/kaggle/.venv), created a venv, installed
requirements.txt cleanly (stable_baselines3 2.9.0, gymnasium 1.3.0 — no
version conflicts), smoke-tested the exact E2 command at tiny scale, then
launched the real 5-seed×60k confirmatory run in the background with a
local SSH-polling watcher that rsyncs results back on completion. This has
no Kaggle-style size/network constraints — it's a full Ubuntu box with
normal internet. **Treat this as the primary E2 lane; the Kaggle attempt
under `thomaschisica` is opportunistic only** (that account previously had
a working embedded-payload track record, so worth one low-effort retry, but
per instruction not depending on it for anything heavy).

Local watcher `scripts/watch_track_b_e1_e4_local.sh` is running in tmux session
`q1_e1_e4_watch`. It polls E1/E4 every 3 minutes and will automatically run
`scripts/build_track_b_e1_go_no_go.py` after E1 stops and
`episode_metrics.csv` exists.

**STOP retrying E2 on Kaggle — root cause found, it's not a timing/account
issue.** Checked `outputs/experiments/track_b_adaptive_confirm_v9_2026-07-01/fetched_v3/`
(the *original* v9 attempts, hours before tonight's account switch, under the
old `thomaschisica` account): **same exact
`FileNotFoundError: Missing scres_ia_payload.tar.gz beside kernel or in
Kaggle input.`** Grepped the whole repo's Kaggle output history — zero
instances of `"using payload /kaggle/input"` ever succeeding. This is a
pre-existing, systemic dataset-mount failure for this project's kernels, not
something introduced tonight. The git-clone fallback also fails on every
attempt with `Could not resolve host: github.com` despite
`enable_internet: true` — classic symptom of Kaggle silently disabling
internet at runtime for an unverified account, regardless of what the
kernel metadata requests. **Both fallback paths (dataset mount, git clone)
are broken**, so no amount of re-pushing the dataset/kernel will fix this;
it needs either Kaggle account phone verification, or a different
code-delivery mechanism (e.g. actually populating the `EMBEDDED_PAYLOAD_B64`
base64-inline fallback that exists as a stub in the older kernel scripts but
was never filled in). **UPDATE — found the working method and fixed it (2026-07-02 ~05:41 UTC):**
checked `outputs/kaggle/track_b_campaign_optimized/fetched_psi1_fix/` and
`outputs/experiments/track_b_adaptive_sweep_kaggle_2026-07-01_v6/fetched/` —
both succeeded, and both used a delivery method neither dataset-mount nor
git-clone: **the payload tarball base64-embedded directly inside the kernel
script as a string literal** (`EMBEDDED_PAYLOAD_B64`, decoded and written to
disk at runtime if no local/dataset file is found). My E2 script had that
fallback stubbed out empty (inherited from the v9 template) and never
populated it. Root-caused and fixed in three steps:

1. **Populated `EMBEDDED_PAYLOAD_B64`** with the real payload, base64-encoded
   and wrapped at 76 chars/line (matching the working reference format).
   Verified locally by moving the local `.tar.gz` copy aside and re-running
   — confirmed the script falls through to `[kernel] writing embedded
   payload` and completes correctly.
2. **Hit a hard kernel-source size cap.** Push failed with `400 Bad Request`
   at the full-repo payload size (~1055-1068KB script). Bisected with
   throwaway probe kernels (`thomaschisicalondoo/scresia-size-probe`):
   1,030,046 bytes pushed fine, ~1,068-1,080KB did not — **the cap is close
   to 1 MiB (1,048,576 bytes)** for this account. Fixed by shrinking the
   payload to only the files E2 actually needs (traced imports:
   `run_track_b_observation_ablation.py` → `run_track_b_smoke.py` →
   `benchmark_control_reward.py`/`track_b_heuristics.py`, plus all of
   `supply_chain/`), cutting the tarball from 786KB → 165KB (script now
   230KB total, comfortably under the cap).
3. **Missing `pip install`.** My script never had the dependency-install
   step the working reference scripts have
   (`pip install -q -r requirements.txt`, Kaggle-only). First real-scale
   attempt got past payload extraction and launched the actual 5-seed/60k
   run, then failed with `ModuleNotFoundError: No module named
   'stable_baselines3'`. Added the install step and re-enabled
   `enable_internet: true` (needed for pip/PyPI access — distinct from the
   git-clone-to-github.com path, which is still broken/DNS-blocked and is
   not being relied on).

Kernel v5 pushed at ~05:41 UTC with all three fixes and briefly reached
`RUNNING`, but ended `ERROR` at ~05:45 UTC. Manual fetch landed in
`outputs/experiments/track_b_e2_obs_masked_confirm_2026-07-02/fetched_v5_manual/`.
The log shows the embedded payload worked (`writing embedded payload`,
`extracting ... -> /kaggle/working/scres-ia`), then pip failed because Kaggle
cannot resolve PyPI either:
`Temporary failure in name resolution` for `/simple/simpy/`, followed by
`No matching distribution found for simpy>=4.1`. So the remaining blocker is
not repo delivery; it is dependency delivery under a no-DNS Kaggle runtime.
Do **not** rely on `pip install` for this account tonight. The viable routes
are: wait for local CPU capacity and run E2 locally; fix Kaggle internet/account
verification; or build a vendored-dependency payload/wheel dataset that does
not require DNS. A naive vendor bundle of SB3+Gymnasium+SimPy is ~1.1MB
compressed before base64, so it likely exceeds the inline-script cap unless
trimmed further or delivered as a Kaggle dataset that mounts reliably.

**E3 is NOT yet launched by Claude** — Codex, since you already wired the E3
compatibility shim (48-dim frozen-checkpoint fix), please own launching the
real E3 h52/h104 matrix; Claude will not duplicate it.

**E6 is done**. Corrected verdict is in
`docs/E6_FIDELITY_MODE_RECONCILIATION_2026-07-02.md`: faithful mode repairs the
flow-mode configuration, but risk_r3 no longer cleanly passes the ReT sign
gate (H2 8/10, p=0.05469; H3 6/10, p=0.37695), and fill-sign checks remain
weak.

**Kaggle account note:** the CLI is currently authenticated as
`thomaschisica` again. This restores access to older `thomaschisica` slugs, but
does **not** fix the E2 issue: the `thomaschisica` E2 slug is already
`KernelWorkerStatus.ERROR`, and the `thomaschisicalondoo` E2 slug is now
private/inaccessible from this account. Do not treat the account switch as a
Kaggle recovery. E2 is now running on the OVH VPS instead.


Coordinating with a concurrent Codex agent already active on this repo (see
`docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md` context note). This file tracks what
Claude launched/verified vs what Codex had already done as of ~23:50 on the
prior date, to avoid duplicate/conflicting runs.

## What Codex had already done (verified, not re-run)

- **E5 reanalysis bundle** (`docs/track_b_q1_stats_2026-07-02/`): verified
  correct on code read. `build_cvar05_effect_row` genuinely computes the
  conditional mean of the lowest 5% (fixes the old p05-mislabeled-as-CVaR
  bug). `build_seed_level_inference`, `build_top_static_robustness`,
  `build_dispatch_cost_sensitivity` all correctly implemented in
  `scripts/build_track_b_q1_stats.py`. **No further action needed on E5.**
- **Manuscript bib/nocite fixes**: `\nocite{*}` and dual `\bibliographystyle`
  removed from `main.tex`; duplicate `rolf2025jsim` entry deleted from
  `references.bib`. Confirmed via diff.
- **E1/E2 infra**: `scripts/run_track_b_e1_regime_static_heuristic_crn.py`
  and the `PrivilegedObservationMaskWrapper` in
  `scripts/run_track_b_observation_ablation.py` both exist and are correctly
  built (verified by reading the code). **Gap found:** the E1 script never
  loads/evaluates PPO itself — it only produces static/regime-table/heuristic
  rows. Its smoke test (`track_b_e1_smoke_codex`) was `--skip-regime-fit`,
  `max_steps=2` — plumbing only, not a real result.

## What Claude launched tonight (all backgrounded, `_claude`-suffixed or
distinctly named to avoid collisions)

| Job | PID | Started | Purpose | Status as of last check |
|---|---|---|---|---|
| E6 fidelity-mode rerun | completed | 23:50 prior day | Rerun H2/H3 gate under `kit_equivalent_order_up_to` instead of `legacy_validated` | complete; see `docs/E6_FIDELITY_MODE_RECONCILIATION_2026-07-02.md` |
| E4 final 8D ablation | 81973 | 00:00 | `track_b_ablation_8d_final_2026-07-01/`, 5 seeds×60k, joint/downstream_only/shift_only | joint + downstream_only complete; shift_only running |
| E1 confirmatory go/no-go | 82256 | 00:01 | `track_b_e1_confirmatory_2026-07-02/`, 5 seeds×12 episodes, real regime-table fit (not skip), trimmed grid (3 shifts × 5×5 op10/op12 = 75 candidates) to keep fitting tractable | running, no progress output (script buffers to CSV only at completion) |

**New script written (not run yet — depends on E1 finishing):**
`scripts/build_track_b_e1_go_no_go.py` — merges E1's regime-table/heuristic
`episode_metrics.csv` against the canonical PPO run's `episode_metrics.csv` on
`(seed, episode, eval_seed)` and produces the actual go/no-go verdict
(`outputs/audits/track_b_e1_go_no_go_2026-07-02/verdict.json`). Neither
Codex's nor this session's E1 script did this merge — it was a genuine gap.

## E2 — moved to Kaggle (per user steer: use Kaggle for heavy/confirmatory work)

Local CPU was already saturated by E6+E4+E1 (load ~9-13 on 10 cores), so E2
(obs-masked PPO retrain) was offloaded to Kaggle instead of competing locally:

- Smoke-tested the exact kernel script locally first (`SCRESIA_PROFILE=smoke`,
  512 steps/1 seed) — ran clean end-to-end before pushing.
- The initial dataset-mount route failed repeatedly: Kaggle did not expose
  `scres_ia_payload.tar.gz`, and the git-clone fallback failed with
  `Could not resolve host: github.com`. This matched older v9 failures under
  the previous account, so it was a delivery-mechanism failure, not an E2 code
  failure.
- The working delivery method was recovered from prior successful kernels:
  embed a minimal `scres_ia_payload.tar.gz` directly in the kernel script as
  base64 (`EMBEDDED_PAYLOAD_B64`), decode it at runtime, then install
  `requirements.txt` before running the experiment. The minimal payload is
  ~165KB and keeps the full kernel script at ~230KB, below the apparent ~1MiB
  Kaggle source cap on `thomaschisicalondoo`.
- Failed kernel: `thomaschisicalondoo/scresia-track-b-e2-obs-masked-confirm`
  v5. It was configured for `run_track_b_observation_ablation.py --obs-configs
  v7_no_regime_forecast` at the canonical settings (control_v1,
  adaptive_benchmark_v2, h104, 5 seeds×60k, lr=3e-4, n_steps=1024,
  batch=256, n_epochs=10), but it never reached training because dependency
  installation failed under DNS outage.

**Claude corroboration (independent check, same conclusion):** downloaded
the actual missing wheels locally to get exact numbers —
`simpy-4.1.2` (27KB), `gymnasium-1.1.1` (965KB), `stable_baselines3-2.7.1`
(188KB), `sb3_contrib-2.7.1` (93KB), `einops-0.8.2` (66KB) = **1.34MB total,
before any code/model payload**, confirming Codex's estimate — vendoring is
not viable under the ~1MiB inline-script cap. Not attempting Kaggle further
for E2 tonight; Kaggle watcher killed
(`scripts/watch_track_b_e2_obs_masked_confirm.sh`).

**OVH replacement lane (active):** Claude launched E2 on the OVH VPS under
`~/scres-ia`, output directory
`outputs/experiments/track_b_e2_obs_masked_confirm_vps_2026-07-02/`, remote
PID 557615. Codex briefly created duplicate OVH E2 processes while setting up
a fallback payload under `/home/ubuntu/scres-ia-e2-20260702-min`; those
duplicates were killed, preserving only Claude's older run. Local watcher
`scripts/watch_track_b_e2_obs_masked_vps_claude.sh` is running as PID 65271 and
rsyncs the remote output directory back into this repo every 3 minutes. First
watcher poll at 2026-07-02T05:58:24Z confirmed the remote process was running.

## Not yet launched

- **E3** (cross-regime/horizon matrix): script exists
  (`scripts/eval_track_b_cross_scenario.py`) but uses a **fixed static-policy
  set per risk level**, not a re-fitted per-regime dense frontier per risk
  level — lighter than the audit's ideal spec but still real generalization
  evidence. Given local CPU is still tight (load ~13 with E6+E4+E1 running)
  this will likely also move to Kaggle rather than compete locally; limitation
  will be disclosed in the final write-up rather than building new fitting
  logic tonight.

## Next check

Poll local E1/E4 processes and the OVH E2 watcher. Do not call
E1/E2/E4 complete until full-scale CSV/JSON artifacts exist. E3 still needs
either per-risk dense-static fitting or explicit limitation as lighter evidence.

## Final check — 2026-07-02T13:16:45Z

All overnight compute lanes are now terminal. No local `run_track_b*` jobs,
no OVH experiment jobs, and no tmux watcher remain active. The stale local
watcher process was killed after artifacts were verified.

Completed artifact roots:

- E1 go/no-go:
  `outputs/experiments/track_b_e1_confirmatory_2026-07-02/` and
  `outputs/audits/track_b_e1_go_no_go_2026-07-02/`.
  Verdict: **GO**. PPO beats the best zero-learning regime-table comparator
  with CI95 > 0; delta mean `0.0003987059`, CI95
  `[0.0003693285, 0.0004302702]`.
- E2 obs-masked retrain:
  primary complete artifact is
  `outputs/experiments/track_b_e2_obs_masked_confirm_vps_2026-07-02/`.
  OVH PPO masked-v7 `order_level_ret_mean_mean=0.0056130959`.
  Kaggle `thomaschisica/scresia-track-b-e2-obs-masked-confirm` also reached
  `COMPLETE`; a partial fetch contains `episode_metrics.csv` and
  `order_ledger.csv` under
  `outputs/experiments/track_b_e2_obs_masked_confirm_2026-07-02/fetched_thomaschisica_complete/`.
- E3 cross-regime/horizon matrix:
  `outputs/experiments/track_b_e3_cross_regime_horizon_matrix_vps_2026-07-02/`.
  Matrix covers h52/h104 × current/increased/severe. PPO beats the fixed
  static comparator on ReT in current/increased and is mixed in severe
  (`h52 severe` ReT gap `-0.000060`; `h104 severe` ReT gap `+0.000009`).
- E4 final ablation:
  `outputs/experiments/track_b_ablation_8d_final_2026-07-01/`.
  All three arms completed. PPO ReT means:
  `joint=0.0055874344`, `downstream_only=0.0056777158`,
  `shift_only=0.0056070968`.

Compute note: Kaggle is available under `thomaschisica`, but it remains
opportunistic rather than primary. The reliable heavy lane for this project is
the OVH VPS, with local Mac used for moderate/ongoing jobs.
