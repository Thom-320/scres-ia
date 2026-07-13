# Adversarial audit of concurrent F8/F10 toy screens

Date: 2026-07-13
Scientific use: exploratory boundary evidence only. The concurrent toy scripts/results are preserved but excluded from the authoritative artifact allowlist.

## F8 two-product shared-line toy

The result labels itself `TOY_F8_HEADROOM_BELOW_GATE_OR_NULL_NOT_CLEAN`: no candidate `H_obs` LCB reaches 0.01 and the declared null cells do not all collapse. It cannot be promoted for additional reasons:

- `Y` is unit fill, not canonical order-level `ret_excel_visible_v1`.
- The “open-loop” set contains ten short repeated cycles, not all full-horizon binary calendars.
- `best_non_signal_comparator` selects the best comparator separately on each tape. That is an ex-post oracle over comparator identities, not one deployable comparator, so the stored `H_obs` is not the defined estimand.
- The setup implementation does not freeze a pending target: during `setup_rem > 0`, completion assigns `last = a` from the current action. A policy can redirect a changeover while it is under way, contradicting the documented commitment mechanism.
- Backlog is not served from later inventory, and the base-stock policy builds its sequence under different aggregate backlog dynamics before evaluation.
- Product BOMs, demand shares, substitution rules, setup/minimum-run physics and signal timing remain absent from Garrido sources.

The toy therefore demonstrates neither product-mix headroom nor a quantitative ceiling on a Garrido-defensible contract.

## F10 finite-fleet toy

The result labels itself `TOY_F10_HEADROOM_BELOW_GATE_OR_NULL_NOT_CLEAN`; its largest stored `H_obs` LCB remains negative and its chosen null does not collapse. Further defects prevent scientific use:

- `Y` is batch-clear unit fill, not canonical ReT.
- The stored `H_PI` is a per-tape maximum over a small tested policy list, hence a feasible-subclass lower bound on the real oracle, not a PI upper bound.
- The stored `H_obs` takes a per-tape maximum over three belief weights before subtracting the baseline. This opens each evaluation tape to choose the policy parameter and is not one frozen non-anticipative policy.
- Only eight periodic calendars are tested; the full 50-period open-loop family is not bounded.
- Vehicle state is a return time only. There is no conserved vehicle location, origin-dependent travel, payload or mode-specific arc, so the code does not establish the requested location/commitment mechanism.
- The null verdict inspects only the first zero-correlation cell rather than all matched physics cells.

## Disposition

Both toys are negative exploratory diagnostics. They do not change `blocked_domain_fact` for product mix or finite-fleet reservation, do not consume virgin tapes, and do not authorize a learner. Reopening either family still requires Garrido's operational facts followed by a frozen canonical-DES contract and full comparator gate.
