# CD-lane audit note (2026-06-27) — for coordination (Codex + Claude)

Three findings from the Discrete(18) / CD audit. **The committed `WIN=YES` (4-case) result is a
measurement artifact and must not be reused as-is.**

## 1. `war_cd_train.py` eval is buggy — the war-CD "win" is false
`scripts/war_cd_train.py` (used for cases 3-4) measures the policy with **non-existent info keys**
that silently default to 0, and a CD aggregation that mismatches the static frontier:
- `info.get("fill_rate_state_terminal", 0.0)`, `info.get("n_lost", 0)`,
  `info.get("service_loss_auc_per_order", 0.0)` are **not** in step `info` → flow_fill / lost /
  service all reported as 0 (a measurement artifact, NOT real zeros).
- PPO CD = the **terminal-step** sigmoid; the static frontier CD = **episode-mean** sigmoid →
  apples-to-oranges; the "win" partly rides on this.
- The aggregated table compared PPO against a phantom-weak `static_robust=0.55 / oracle=0.57`,
  but the true best static (measured consistently) is **S1_I168 cd≈0.63**.

Corrected (consistent episode-mean + `compute_episode_metrics` real service, 3 seeds × 15k):
`scripts/cd_component_decomposition.py` → **PPO LOSES** (cd 0.588 vs static 0.628) and delivers far
worse service (flow_fill 0.47 vs 0.80; lost 0.48 vs 0.08). Decomposition: PPO's CD deficit is
dominated by **κ̇/cost** and it is worse on ε/backorders too.

→ **Action:** fix `war_cd_train.py` to use `compute_episode_metrics` + episode-mean CD, or stop
using its summary. Do not publish the 4-case `WIN=YES`.

## 2. Why no win: full-cost CD rewards "no buffer", so the optimum is a degenerate constant
`scripts/audit_pepe_discrete18.py` trains RL on **Discrete(18)** (which *does* expose buffer×shift,
unlike `track_a_v1`) with the CD reward in the war cell, and compares to every constant action:
- With **`kappa_train_frac=1.0` (full cost)** the CD index is dominated by κ̇ (n_kappa≈0.31, the
  largest exponent) → the **best constant action is `S1_I0` (no buffer)** and PPO correctly
  **converges to `S1_I0`**. So PPO *is* learning and *does* match the best static — but the optimum
  is the cheap no-buffer policy, which has terrible tail service (high CVaR). No dynamic headroom,
  no resilience win, by construction of the reward.
- This is the "full cost → S1 collapse" the older audit notes warned about.

→ The full-cost CD bar is **cost-dominated**: maximizing it means cutting cost (no buffer), not
delivering. That is why CVaR loses and why nothing beats the static — the reward doesn't value the
buffer's delayed service benefit (anticipation is not credited).

## 3. Implication for the win definition + reward
- **Evaluate resilience on the Excel bar** (service/recovery, no cost) OR on a **reduced-cost CD**
  (`kappa_train_frac≈0.2`). The full-cost CD same-bar is degenerate.
- When CD is the *same-bar* reward, reduce the cost weight so the buffer/service terms (ζ, ε) drive
  it; then anticipatory buffering has value and a dynamic policy can plausibly beat a fixed static.
- Test underway: `audit_pepe_discrete18` at κ=1.0 vs κ=0.2 to confirm a buffer policy emerges and
  improves Excel ReT + CVaR under the reduced-cost reward.

## Artifacts
`outputs/experiments/cd_component_decomposition_2026-06-27/decomposition.json`,
`outputs/experiments/audit_pepe_{fullcost,lowcost}_2026-06-27/audit.json`,
`outputs/audits/des_garrido_format_2026-06-27.xlsx` (per-order ledger in Garrido Raw_data format).

## A↔B contrast (Pepe Discrete(18), 2 seeds × 40k, war φ4/ψ1.5) — confirms the fix direction

| metric | A (full-cost κ=1.0) | B (low-cost κ=0.2) | best_const S1_I0 |
|---|---|---|---|
| PPO cd_mean | 0.5796 | 0.5778 | — |
| best_const_cd | 0.6133 (S1_I0) | 0.6133 (S1_I0) | — |
| PPO − best_cd | **−0.0336** | **−0.0355** | — |
| PPO flow_fill | 0.714 | 0.707 | 0.530 |
| PPO lost_rate | 0.170 | 0.179 | 0.403 |
| PPO ret_excel | 0.00182 | 0.00177 | 0.00112 |
| PPO action_hist (seed 1) | S2_I168×160, S1_I504×76, S1_I0×24 | **S1_I336×136, S3_I672×107**, S1_I168×11 | — |

**Key finding — even reduced-cost CD (κ=0.2) collapses to S1_I0** as the best constant (cd=0.6133). Why: the CD cost term (`n_kappa=0.31`) is the **largest exponent** in the 5-var Cobb-Douglas product, so the CD bar *always* rewards the cheap no-buffer constant regardless of κ. B's PPO learned a **buffer-heavy policy** (S1_I336 + S3_I672 — meaningful strategic buffers), which is the *right* RL behavior for real service, but the CD bar still picks S1_I0 as best (cd 0.6133 > 0.5778). Real service is good in both (PPO lost_rate 0.17–0.18 vs S1_I0 0.40 — ~2.3× better).

**This confirms the user's resilience insight:** the CD bar (even reduced-cost) is the wrong eval because the cost term dominates. The correct fix is **eval = Excel ReT (the real Garrido metric) with a service-aligned reward** (reward ≠ eval), so the reward genuinely rewards anticipation/buffering without collapsing to the cheap constant. The A/B show that tuning the CD κ alone doesn't unlock a win; we need to change the bar.
