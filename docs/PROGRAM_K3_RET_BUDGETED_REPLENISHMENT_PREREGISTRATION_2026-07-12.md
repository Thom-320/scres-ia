# Program K3 — ReT under budgeted replenishment

## Binding question

Can a policy using observable inventory, backlog, pipeline and a noisy demand
forecast improve canonical order-level Excel ReT by reallocating the same finite
replenishment budget over time?

K2 remains a correct STOP for the holding-cost objective. K3 is a separate ReT
estimand. Holding and waste are diagnostics only and cannot determine promotion.

## Physical contract

The eight-week episode starts with one `D0` of stock. Every policy may order at
most `10·D0` in total and `1.5·D0` per week, on a `0.25·D0` grid. Orders arrive
after one week. The shelf-life is 156 weeks and therefore does not bind. Weekly
customer demand creates auditable `OrderRecord` objects; inventory is allocated
FIFO and unfulfilled orders remain in backlog. ReT is computed by the canonical
repository aggregator.

The finite procurement/transport budget is a researcher-imposed extension and
requires Garrido validation before a managerial claim. It is equal for all
policies by construction, preventing a ReT win through greater expenditure.

## Comparator and execution boundary

The primary static bar enumerates every periodic calendar of period 1-4 on the
same order grid and budget. Adaptive comparators include budgeted `(s,S)`, an
inventory-only paced controller, and signal/inventory MPC. Development seeds
`6700001+` were used to establish feasibility and are permanently non-virgin.
Confirmation seeds `6800001+` remain sealed until this contract and analysis
code are frozen. PPO remains blocked until an interpretable policy passes ReT,
quantity-ReT, lost-order and resource gates on confirmation.
