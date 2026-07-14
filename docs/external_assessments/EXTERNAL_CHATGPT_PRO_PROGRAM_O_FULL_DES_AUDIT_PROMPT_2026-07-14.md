# External adversarial audit — Program O full-DES promotion gate

Repository: <https://github.com/Thom-320/scres-ia>

Draft PR: <https://github.com/Thom-320/scres-ia/pull/3>

Pinned commit: `07572663ca5a7e0bf42ce568b5d3eaa1c766c68e`

Act as an independent senior simulation, stochastic-control and supply-chain
reviewer. Do not propose PPO, accept a proxy endpoint or infer full-DES value
from a reduced model. Audit whether Program O has legitimately earned the right
to implement a full-DES H_PI screen, and specify the smallest full-DES contract
that can falsify the translation.

Read at least:

- `contracts/program_o_multi_ration_product_mix_v1.json`
- `contracts/program_o_exact_transducer_v1.json`
- `research/paper2_exhaustive_search/PROGRAM_O_MULTI_RATION_PREREGISTRATION_2026-07-14.md`
- `research/paper2_exhaustive_search/PROGRAM_O_EXACT_TRANSDUCER_SCREEN_FREEZE_2026-07-14.md`
- `research/paper2_exhaustive_search/program_o_exact_transducer_validation_freeze_20260714.json`
- `results/program_o/affected_order_bound_v1/result.json`
- `results/program_o/exact_transducer_screen_v1/result.json`
- `results/program_o/exact_transducer_validation_v1/result.json`
- `scripts/screen_program_o_exact_transducer.py`
- `scripts/validate_program_o_exact_transducer.py`
- `research/paper2_exhaustive_search/toy_screen_adversarial_audit.md`
- `supply_chain/ret_thesis.py`
- `supply_chain/episode_metrics.py`
- `supply_chain/supply_chain.py`

Verified current facts:

1. Program M is stopped: two cells passed individually, but its frozen connected
   H_PI region failed. It must not be rescued.
2. Program O is a disclosed two-product researcher extension anchored only by
   the thesis fact that the real system has 21 ration types while the DES uses
   one. Its numerical mix, substitution and setup assumptions are not MFSC
   estimates.
3. The corrected binary transducer enumerates all `2^8=256` calendars. On fresh
   validation tapes, all four cells pass simultaneous H_PI LCB95
   (`0.0781–0.1141`), 12–19 distinct oracle calendars occur, backlog and
   worst-product fill improve, and the fully fungible null is exact.
4. This is not a full-DES result. No H_obs policy or learner is authorized.
5. The governing endpoint is `ret_excel_request_snapshot_v2`; full-order ReT,
   remaining backlog, worst-product service and lost orders are mandatory
   guardrails because the visible ledger omits unfinished rows.

Attack the following possible failure modes with concrete code references,
counterexamples and tests:

- whether the transducer's request-snapshot capture or order-completion timing
  inflates ReT;
- whether selecting the per-tape oracle on visible ReT can exploit omitted
  unfinished orders despite the reported guardrails;
- whether fixed 2:1/1:2 production rights create headroom by construction that
  disappears when the full `k in {0,1,2,3}` action and all `4^8=65,536`
  open-loop calendars are admitted;
- whether an initial-inventory, end-horizon, within-week batch-order or
  same-time-event convention drives the result;
- whether product tagging can be added to Op5–Op13 without duplicating mass,
  bypassing the real Op10–Op12 delivery path or hand-rolling ReT;
- whether setup loss, BOM components or partial substitution are necessary, or
  whether simple nonfungibility alone is the cleanest claim-limited mechanism;
- exact product/component resource and conservation ledgers required;
- strongest product base-stock, hysteresis, min-cost-flow, rolling MILP/MPC and
  belief-DP comparators that would later be mandatory;
- null and ablation cells needed to distinguish product feedback from a fixed
  seasonal calendar;
- seed, watcher, custody and raw-matrix requirements for the VPS run.

Return one of:

1. `GO_FULL_DES_HPI_ONLY`, with an exact minimal implementation contract and
   fail-closed tests; or
2. `STOP_BEFORE_FULL_DES`, with a concrete defect or dominance argument that
   invalidates the translation.

Do not call this Paper 2, H_obs, learned value or retained learning. Do not ask
Garrido to find the mechanism. Later expert review may bound domain claims, but
the development search is the research team's responsibility.
