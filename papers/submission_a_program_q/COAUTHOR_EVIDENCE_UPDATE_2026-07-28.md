# Coauthor evidence update — Program Q and Track B

Date: 2026-07-28

Status: READY FOR COAUTHOR REVIEW — no scientific result is changed by this note.

Canonical paper branch: `codex/submission-a-program-q`

Canonical reviewed commit before the editorial updates in this branch:
`031d0af9479fcf73e95f34cece9a0ea76a218c97`

## Why this note exists

Two results have sometimes been compressed into the sentence “RecurrentPPO
beats all static policies.” They answer different questions and must remain
separate.

### Track B 8D

The same-contract challenge retired the general Track B claim that PPO beats a
strong policy fixed over the complete 8D contract.

- Canonical PPO minus the calibration-selected full-contract constant:
  `-0.000018049` Excel ReT.
- Two-way 95% interval:
  `[-0.000028615, -0.000008087]`.
- Binding source:
  `docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md`.

Allowed wording:

> The historical Track B advantage over a restricted frontier did not survive
> a same-contract challenge with a stronger constant comparator.

This does not alter Program Q.

### Program Q

Program Q prospectively evaluated ten frozen RecurrentPPO checkpoints on 256
new common-random-number tapes in each of three demand-regime cells.

- RecurrentPPO minus the complete 65,536-calendar open-loop frontier:
  `+0.07952`, `+0.07255`, and `+0.11724`.
- Simultaneous lower bounds:
  `+0.06608`, `+0.06233`, and `+0.10614`.
- Favorable tapes:
  `84.8%` to `95.7%`.
- Positive optimizer seeds:
  `10/10` in every cell.
- RecurrentPPO minus the best reselected structured controller:
  `-0.00159`, `-0.00072`, and `-0.00041`.
- Every simultaneous interval for the neural contrast lies inside the frozen
  practical-equivalence region `[-0.01, +0.01]`.
- Worst-product-fill simultaneous lower bounds:
  `-0.02266`, `-0.02566`, and `-0.02632`, which do not establish the frozen
  `-0.02` non-inferiority margin.

Binding sources:

- `papers/submission_a_program_q/source_of_truth.json`;
- `docs/PROGRAM_Q_TERMINAL_VERDICT_2026-07-18.md`;
- `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json`.

Allowed wording:

> In the evaluated Program Q contract, RecurrentPPO outperformed every one of
> the 65,536 deterministic open-loop calendars, but provided no material
> premium over the strongest tested structured controller and did not
> establish worst-product service non-inferiority.

Program Q does not establish superiority over every possible static or dynamic
policy, an optimal POMDP controller, the original monoproduct physics, or
Garrido-native active risks.

## Structured comparator identity

The structured comparator is a frozen ten-member family, not a single
belief-MPC and not a proof of optimal dynamic control.

- Base-stock: one- and two-week targets.
- Max-pressure: switching tolerance 0 or 5,000 units.
- Min-cost-flow: one- and two-week belief-projection horizons.
- Deterministic belief-MPC: three- and four-week horizons.
- Approximate belief-DP: three- and four-week horizons.

The selected members are:

- `rho75_share90`: `min_cost_flow__2`;
- `rho90_share75`: `min_cost_flow__2`;
- `rho90_share90`: `max_pressure__0`.

All members receive causal operational history and the same four weekly
product-mix actions as RecurrentPPO. None receives future orders, the latent
regime, tape identity, or score-time outcomes.

## Risk, product, and learning boundary

- The Program Q product labels and demand regimes are researcher-defined.
- The confirmatory study is risk-off and does not test Garrido-native
  disruption adaptation.
- Learning is within campaign. Physical and recurrent state reset between
  episodes.
- The paper does not test accumulated learning or retained knowledge across
  campaigns.
- The binding compound label
  `STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION` is preserved. Its decomposition,
  rather than its name alone, determines the paper claim.

## Clean-room boundary

The existing Kaggle receipt is a passing independent-environment regeneration
of the frozen evidence package:

- source-of-truth and eight evidence tables were byte-identical;
- four figure pairs were regenerated and checked for exact expected outputs;
- five tests passed;
- the tracked PDF was present and hash-checked.

It did not retrain RecurrentPPO, rerun the full prospective confirmation, or
recompile TeX on Kaggle. Required wording:

> Passing clean-room regeneration of the frozen manuscript evidence package.

The direct scientific evidence remains the prospective confirmation and its
21,696 full-DES replay audits.

## Computational result

The existing benchmark is descriptive and hardware-specific:

- RecurrentPPO median batch-one action-selection latency: `0.573479 ms`;
- reselected structured family median: `0.081834 ms`.

DES construction and observation replay were excluded. No universal compute
claim follows. Under practically equivalent ReT in this contract, the
structured controller is the preferred engineering choice because it was
faster on the measured hardware, is more transparent, and represents
feasibility through explicit rules.

## Ready-to-send note for Prof. Garrido and David

> Profesor, quisiera dejar una precisión metodológica antes de iniciar la
> expansión de variables. Tenemos dos resultados distintos. En Track B 8D, la
> ventaja frente a políticas constantes no sobrevivió cuando la política
> constante recibió el mismo contrato completo. En Program Q, en cambio,
> RecurrentPPO sí superó prospectivamente los 65.536 calendarios open-loop de
> su contrato en las tres celdas evaluadas. Sin embargo, fue prácticamente
> equivalente al mejor controlador estructurado probado y no pasó la
> no-inferioridad del servicio del peor producto.
>
> Por tanto, la escalera propuesta sigue siendo la pregunta correcta, pero debe
> medir por separado cuándo la autoridad adicional crea valor contingente,
> cuánto convierte el mejor controlador estructurado y si queda un residual
> desplegable para MLP o KAN. Un resultado en el que DDMRP o MPC siga siendo
> suficiente será igualmente válido.
>
> La sumisión de Program Q avanzará como resultado cerrado. La nueva escalera
> será un estudio prospectivo independiente, sin modificar resultados
> históricos y sin depender de corridas de colaboradores para selección.

## Written confirmations requested

The coauthors are asked to confirm:

1. author order, affiliations, corresponding author, and CRediT roles;
2. the interpretation of ReT and its workbook-visible population;
3. the wording for worst-product fill and unresolved orders;
4. use of “closed-loop” for within-campaign feedback;
5. the distinction between the thesis-grounded base and researcher-defined
   multiproduct extension;
6. risk-off and within-campaign boundaries;
7. funding, conflicts, security, and AI-assistance disclosures.
