# Track C — From-zero contract/env/learner redesign (2026-07-10)

**Status: DESIGN + pre-registration skeleton. New research program authorized by
the PI after the same-contract gate reversed the Track B adaptive claim and the
clean replication failed. This is a NEW question with NEW tape ranges (600001+),
not a reopening of the closed Paper-1 gates. The C&IE pivot paper is unaffected.**

Inputs: full-thesis deep read (Ch5 metric mechanics, Ch6 risks/design, Ch7
moderation results, Ch8 limitations/future work) + full repo capability/failure
audit (8-agent workflow wf_a52b22b3-bd1, 2026-07-10; results archived in the
session scratchpad and summarized here).

---

## 1. Why everything lost — the three root-cause classes (verified)

**ENVIRONMENTAL — the tested worlds make constants optimal.** Same-contract
constant matches/beats PPO in the native regime (−1.8e-5 CI<0); prevention
channels are exact-zero at native intensities (bit-identical arms, 10/16 R21
grid cells); where headroom exists (engineered R21 starvation, +0.0285) it is
captured by a FIXED posture because the regime is stationary within episodes
and holding was never priced during training — "the env never gave the learner
any reason to time its buffers" (lead-lag traces flat).

**PROTOCOL — comparator/tape artifacts manufactured every prior 'win'.**
Restricted static families (147-cell grid holding upstream fixed) produced the
retired +0.000438/+0.000486 headline; ≥6 Track A lanes died under dense CRN;
non-nested ablations inverted the mechanism story; neutral anchors inflated the
dispatch increment 4.4×; post-hoc tape reads shrank on virgin tapes.

**LEARNER — PPO fails to convert sharp headroom.** Track A 5D had real oracle
headroom (+0.0042..+0.0097) and three pre-registered conversion attempts failed
0/5 seeds each; PPO erodes converged BC inits; sharp discrete optima get
smoothed away. Warm-start maintenance works (CF20); delayed-credit (43.9% of
step reward is delayed) is the one untested learner-side fix.

Design consequence: **the environment must punish constants; the comparator
must share the full contract from day 0; oracle gates must precede any
training; and the learner gets a warm-start arm.**

## 2. What we never did (the gap list)

Environment/physics:
1. Intra-episode non-stationarity with material amplitude and controllable
   dwell. The adaptive benchmark's 5-regime process swings intensity only
   0.90→1.85 on downstream risks with ~4–6 DAY dwells against a 168h decision
   epoch — sub-Nyquist for a weekly controller — and E1 measured its
   regime-conditioned optima ≈ one constant (+2.77e-5).
2. Costs priced in training. All cost pressure was reward-side and mostly
   post-hoc; holding cost on ACTUAL time-weighted stock never existed
   (target-fraction proxy only); surge_inertia exists and was exercised in
   audits but never in a training benchmark.
3. Route-aware replenishment (top-ups currently inject through downed routes —
   the physics hole that made "prevention" indistinguishable from reaction).
4. A regime built on R22/R23/R24 — the risks Garrido's OWN Ch7 moderation
   evidence says buffers/capacity have authority over (H2b: buffers inhibit
   R22 92%, R23 67%; H3b: capacity inhibits R24 67%; association rules driving
   ReT Low: R22+R24, R22+R23, R23+R24). We engineered R21 instead — the risk
   his Table 7.1/7.3 found NOT associated with ReT drops at native rates.
5. 24h decision epoch (thesis order cycle is daily; we always ran 168h).

Contract/learner:
6. op5_q is a DEAD DIMENSION in every env built with initial_buffers={} —
   most published Track A/B runs trained a no-op dim (supply_chain.py:1093-97).
7. Budgeted/Lagrangian constrained RL (advisor-preferred P1) — never run.
8. Delayed-credit hyperparameters (gamma 0.995+/gae 0.98) — never tested.
9. Stage-2 PPO hparam grid — never run (task #13).
10. BC-warm-start from a known-good threshold policy on a lane with verified
    headroom — never combined with the above.

Protocol (now fixed, keep):
11. Same-contract static co-optimization as a standard harness element (added
    2026-07-10 only); switching-oracle entry gate BEFORE training (E1 was
    post-hoc); virgin-tape discipline per gate.

Paper-2 leftovers folded in here: classical reserve frontier (Gate C),
route-aware sensitivity (Gate D), actual-inventory economics (Gate E).

## 3. The from-zero design (thesis-grounded)

**Research question (in Garrido's own terms).** His §8.5.1 names stationarity
as THE decisive limitation ("the effectiveness of the buffering strategy is
guaranteed as long as the demand for items is steady over time"); §8.5.2/8.6.2
ask for the cost factor; §8.6.1 asks buffers-vs-capacity synergy + lead-time;
§8.4 literally describes dynamic buffering ("any change detected in the
long-term pattern ... must be quickly incorporated into the buffering
strategy"). So:

> Under non-stationary campaign-style risk (relaxing §6.5.8) and with the cost
> factor included (§8.5.2), is the optimal buffering/capacity/dispatch strategy
> still an open-loop constant — or does state-contingent control create
> measurable resilience value, and can model-free RL learn it from
> non-privileged observations?

Either answer is publishable: a positive result is the adaptive/preventive win;
a null extends the C&IE comparator-design paper with "even under
non-stationarity + costs, constants suffice" — a strong boundary.

### 3.1 Environment `campaign_v1` (new risk_level preset)

- Base: `current` risk tables, all 9 risks enabled, thesis year basis,
  stochastic_pt on, h104 × 168h (24h epoch reserved as sensitivity).
- **Campaign regime process** (extends the existing adaptive-benchmark regime
  machinery): 2 states, weekly review (168h, matching the epoch).
  - `calm`: native intensities (multipliers 1.0).
  - `campaign`: R22 freq ×4, R23 freq ×4, R24 freq ×3 (surge ×1.5), R21
    freq ×4 / impact (recovery) ×3 — inside the measured headroom-on region
    (R21 grid turns on at freq ≥×4 AND impact ≥×3; Case B/C showed R22/23/24
    stress creates the largest adaptive margins ever measured, +0.014..+0.046
    vs restricted statics).
  - Dwell: geometric, mean 8 weeks calm / 5 weeks campaign (≫ epoch and
    ≫ 168h replenishment lead; several transitions per h104 episode).
  - NO injected precursor signal in v1: detectability comes from realized
    minor-event rates (campaign state ≈4× event frequency → hazard EWMA rises
    within 1–3 weeks). A `pre_campaign` intel-ramp state is a v2 option.
- **Costs (priced in training AND in the endpoint):**
  - λ_h = 0.05 per unit-week on ACTUAL time-weighted buffer stock (new
    ledger; the measured preventive increment survives to λ≈0.05 — this is the
    calibrated "real but not lethal" level; sensitivity 0.02/0.10).
  - Dispatch expediting λ_d = 0.025 on (m10−1)+ + (m12−1)+ (C18's threshold
    where cost separates policies).
  - surge_inertia ON (ramp 1 level/step, finite budget) — S3 is no longer free
    or instant.
- **Route-aware replenishment ON** (finish the pending edit): buffer top-ups
  cannot arrive while the supplying LOC is down; queued until route recovery.
- Primary endpoint `J_v3 = mean ret_excel − λ_h·holding_actual − λ_d·dispatch
  − λ_s·(S−1)·step` — ONE quantity for training reward, static optimization,
  and evaluation (the alignment principle; ret_excel alone reported alongside).

### 3.2 Decision contract `track_c_v1` (12D = track_bp_v1 + live op5)

Dims 0–7: track_b_v1 (op3_q, op9_q, op3_rop, op9_rop, op5_q, shift, op10_q,
op12_q). Dims 8–10: buffer-target fractions op3_rm/op5_rm/op9_rations under
the 168h commitment lag (track_bp machinery). Fix: construct the env WITH
initial op5_rm buffer so dim 4 is live. The static comparator family gets THE
SAME 12 dims (Sobol full-contract search) plus Garrido's own clock family
(I168..I1344 × S1–3, his Scenario II/III grid) as named baselines.

### 3.3 Observation

`v10_no_regime_forecast` (realized-history only: per-risk active/recent flags,
weeks-since clocks, 8w/26w counts, hazard EWMAs, calendar phases, queue/backlog
state; NO regime one-hot, NO transition-matrix forecasts). Reviewer-safe by
construction. Optional supervised probe (offline AUC of campaign-state
detection from obs) as a diagnostic, not a claim.

### 3.4 Pre-registered gates (cheapest first; STOP rules before results)

Tape discipline: calibration 600001–600060; gate evals 610001+; each
confirmatory battery a fresh never-opened range. All contrasts CRN-paired,
two-way (seed × tape) bootstrap + seed-clustered t-CI.

- **Gate C0 — env sanity (eval-only, hours).** Regime process visibly switches
  (logged); never/always posture spread > 0 in campaign weeks; always-max pays
  measurable cost under J_v3; route-aware blocks mid-outage top-ups (assert).
- **Gate C1 — switching-oracle headroom (eval-only, THE decisive gate).**
  Policies: (a) best single constant (Sobol 128+refine over 12D on J_v3,
  calibration tapes); (b) privileged phase-switching constant pair
  (calm-constant, campaign-constant, each Sobol-fit on disjoint calibration
  tapes, switched by TRUE regime — an upper bound no learner can beat);
  (c) never/always anchors; (d) Garrido clock family.
  **PROMOTE iff (b) − (a) ≥ +0.002 J_v3 with CI95 > 0.** Below threshold:
  iterate env design (amplitude/dwell/λ) — allowed BEFORE any training and
  BEFORE any test tape; each iteration logged. If no reasonable calibration
  reaches threshold → the null paper (boundary extension) and STOP.
- **Gate C2 — non-privileged detectability.** Same switching pair triggered by
  a hazard-EWMA threshold on v10 features (grid over threshold/half-life on
  calibration tapes). PASS iff ≥50% of the C1 gap is captured. FAIL → regime
  is undetectable → add pre_campaign ramp state or stop.
- **Gate C3 — learner conversion.** PPO-MLP [256,256], 5 seeds × 200k,
  lr 3e-4 (1e-4 sensitivity), ent_coef 0.005, gamma 0.995, gae_lambda 0.98,
  n_steps 1024, batch 256, norm obs+reward. TWO arms: (A) from scratch;
  (B) BC-warm-started from the Gate-C2 threshold policy (the CF20 lesson:
  PPO maintains what it cannot discover). Eval on a virgin battery vs the
  same-contract Sobol constant re-optimized on calibration tapes under J_v3.
  **Adaptive WIN iff PPO − best-constant CI95 > 0, ≥4/5 seeds, ≥50/60 tapes.**
  KAN sidecar only after a PPO pass (KAN ties MLP everywhere; architecture is
  not the bottleneck).
- **Gate C4 — preventive decomposition (only after C3 passes).** Existing
  cross-fitted within-checkpoint machinery (clamp-to-own-mean with disjoint
  episodes, exogenous replay, pre/post-onset blocks) + lead-lag vs campaign
  onsets. **PREVENTIVE claim iff the timed pre-onset buildup channel survives
  controls with CI95 > 0** — physically possible now (route-aware: stock must
  be placed BEFORE the route drops) and economically forced (λ_h punishes
  always-high). Otherwise the result is "adaptive, posture-switching" only.

### 3.5 Build list (code)

1. `campaign_v1` risk-level preset: per-regime multiplier dicts applied to
   {R21,R22,R23,R24} via the existing regime-intensity branch; weekly review;
   dwell params in config. (config.py + supply_chain.py, small.)
2. Actual-inventory holding ledger (time-weighted stock integral per op,
   exposed in info + episode metrics).
3. Route-aware replenishment flag (`replenishment_route_aware=True`): top-up
   arrival requires the op's supplying LOC up at arrival time; else queued.
4. `track_c_v1` factory = make_track_bp_env + live op5_rm + campaign_v1 +
   costs; `J_v3` reward mode + episode metric.
5. Gate runners C0–C3 (adapt run_track_b_static_contract_search /
   run_track_bp_gate1_oracle / evaluate harnesses; two-arm PPO trainer with
   BC warm-start reusing the existing BC machinery).

### 3.6 Honesty guardrails (answering the p-hacking concern in
STRATEGIC_DECISION_CLOSE_CIE_2026-07-10.md)

The contamination critique is mitigated, not waved away: (a) new question in
Garrido's own future-work terms, new env physics, new tape universe; (b) env
design iteration happens ONLY at the eval-only oracle stage, pre-training,
logged; (c) the same-contract static is co-optimized under every condition
change — the asymmetry that manufactured old wins cannot recur; (d) both
outcomes are pre-committed papers (win → adaptive/preventive under
non-stationarity; null → boundary extension of the comparator-design paper);
(e) the C&IE pivot manuscript proceeds independently and is not hostage to
Track C.
