# Program M: finite shared-lift advance reservation

Status: **frozen after adversarial correction; no tapes have been opened, and no
learner or virgin tapes are authorized.**

## Why this family leads the next search

The pasted R21/R3 finite-crew result is not an executable repository result. The
governing record contains no producer, raw rows, tests or verdict for the quoted
`0.0034`, `0.0042` or `0.00004` values. In contrast, the historical Program G
spatial adapter exhibited order-level perfect-information headroom near `0.018`.
That evidence is non-governing because it predates the request-snapshot-v2 metric,
uses a stylized adapter, fails observable conversion and sacrifices worst-CSSU
service. It is nevertheless the strongest quantitative pointer to a mechanism,
not a claim.

Program M changes the mechanism, not merely its architecture or risk magnitude.
The full terrestrial service remains balanced. The new decision is an irrevocable
weekly assignment of one conserved protected-lift slot to A or B before a
destination-local access disruption. This targets the two Program G failures:

1. the pre-booking warning supplies information before commitment; and
2. every policy is charged for the same eight protected round trips, including
   empty movements, so serving the threatened destination does not take ordinary
   land capacity from the other CSSU or buy more protected departures.

The exact machine contract is
[`contracts/program_m_shared_lift_reservation_v1.json`](../../contracts/program_m_shared_lift_reservation_v1.json).

## Theory-first necessary condition

Let `d` be the destination, `Z_t` the localized access state, `S_t` the warning,
and `a_t` the destination receiving the single protected slot. For the slot to
have adaptive value, the paired marginal canonical outcome must reverse:

`Delta_t(X_t,Z_t) = Y(a_t=A)-Y(a_t=B)`

must be positive on a material set of A-threatened states and negative on a
material set of B-threatened states. The slot is reserved before the affected
departure. Decisions occur every 168 hours, activate 24 hours later and expire at
the next decision epoch. All policies are charged exactly eight movements,
20,800 units of payload capacity and 384 vehicle-hours, loaded or empty. The
action cannot change demand, production, stock, risk events or baseline transport.

If the warning carries no location information, arrives after booking, or the
protected mode obeys the same local outage, `H_obs` should collapse to zero. Those
are mandatory null/placebo cells rather than optional sensitivity checks.

## Frozen search and selection rule

The positive-hazard development grid is the Cartesian product of:

- independent weekly event hazard: `0.25, 0.50, 0.75`;
- disruption duration: `24, 72, 120` hours; and
- warning sensitivity/specificity: `0.70/0.80`, `0.85/0.90`.

Event occurrence, location and start are independent exogenous draws; locations
are never balanced within a tape, realized episode counts are hidden, and starts
are uniform within the predeclared weekly window rather than synchronized to a
departure. The 18 positive cells, one unique zero-hazard cell, mode-equivalence
ablation and all signal placebos are reported. The first gate enumerates all
`2^8 = 256` reservation calendars per cell.

A candidate connected region must contain at least three adjacent cells and span
at least two hazard and two duration levels. It is selected only on 24 screening
tapes, then every member is recomputed on a fresh 24-tape H_PI block with
simultaneous one-sided bounds. The cell with the smallest passing validation
lower bound is frozen for H_obs. If no region has resource-restricted `H_PI`
LCB95 at least `0.01`, this contract stops before policy fitting.

The predeclared burned roles are disjoint: `7300001-7300024` for screening,
`7300025-7300048` for H_PI validation, `7300049-7300096` for H_obs fitting and
`7300097-7300144` for one-time H_obs validation. A seed is burned only when its
named role is opened and can never be called virgin. Confirmation seeds are
deliberately not assigned.

## Comparator and audit obligations

The adaptive policy must beat the complete 256-calendar open-loop frontier,
strong backlog/age/fill/max-pressure rules, the signal rule, robust reservation,
belief-state stochastic lookahead, two-week exact-action MPC and resource-equal
mixtures. `H_obs` is adaptive-versus-open-loop; `H_learned` is separately defined
as learner-versus-the maximum of open-loop and every classical comparator. Every
action sequence is retained and replaced by its modal calendar,
calibration-only fixed calendar and phase-only controller.

The primary endpoint is `ret_excel_request_snapshot_v2`. Quantity ReT, service
loss, lost orders, backlog age, worst-CSSU fill, tail risk, mass, baseline
transport and protected-slot ledgers are simultaneous guardrails. A spatial mean
gain with worse worst-CSSU fill is a failure, not headroom.

## Claim boundary

This is a disclosed synthetic researcher extension anchored to the thesis's two CSSUs,
two 24-hour downstream legs, daily order scale and stated use of land, water and
air modes. The thesis does not provide fleet counts or mode parameters. A
positive result would therefore support adaptive value **conditional on the
frozen multimodal contested-access contract**, not claim numerical reproduction
of the original fleet. The numerical hazard and warning intensities are scenario
assumptions, not thesis estimates. Garrido or another domain expert can review
face validity later; they are not being asked to find the headroom or choose the
winning cell.
