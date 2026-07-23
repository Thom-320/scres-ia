"""Product-coupled R24 carrier for the Q-R1 risk-door screen.

Contract: contracts/q_r1_product_coupled_execution_amendment_v1.json (frozen
2026-07-23) under parent contracts/q_r1_product_coupled_risk_door_v1.json.

Mechanism (amendment ``mechanism_frozen``): a latent per-history composition
state sigma in {"C", "H"} persists across campaigns with kappa_r (flip prob
1 - kappa_r).  When a Garrido-native R24 contingent-demand surge fires, its
quantity is assigned to the sigma-favored product with probability s_r (else
the other product).  R24 runs at the thesis's own "increased" frequency level
(uniform(1, 336) h) with unscaled thesis surge quantities (2400..2600).

The product coupling is a RESEARCHER-DEFINED extension (PI full-autonomy
grant); it is doctrine-grounded and never attributed to Garrido.  The R24
family, frequency level, and surge quantities are Garrido-native (Table 6.7).

Implementation notes
--------------------
* Evaluation is DIRECT SimPy (``ProgramOFullDESSimulation`` physics with
  ``risks_enabled=True``); risk masks break transducer exactness, so the fast
  transducer is not used anywhere here.
* CRN: the R24 event timeline and surge quantities come from the parent's
  ``per_risk`` RNG stream (action-independent).  The product assignment uses a
  DEDICATED stream seeded from (coupling salt, campaign seed, event index), so
  the assignment sequence is also identical across arms of the same
  (cell, root, campaign).
* The surge merge point mirrors the parent scalar mechanism
  (supply_chain.py::_sample_calendar_demand_quantity L4765-4784) but is
  product-typed: two pending buckets, and a surge quantity only attaches to
  orders whose tape product matches the surge's assigned product.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from supply_chain.program_o_full_des import (
    PRODUCTS,
    ProgramOFullDESSimulation,
    ProgramOOrderRecord,
    product_demand_tape,
)
from supply_chain.q_r1_retained_learning import early_cohort_metrics_from_orders
from supply_chain.supply_chain import HOURS_PER_WEEK, RiskEvent

SIGMA_VALUES = ("C", "H")
SIGMA_TO_PRODUCT = {"C": "P_C", "H": "P_H"}
# Researcher-defined stream salts, frozen here (amendment: parameters are
# researcher-defined).  SIGMA_PATH_SALT seeds the latent sigma chain;
# COUPLING_STREAM_SALT seeds the per-event product-assignment stream.
SIGMA_PATH_SALT = 0x516A0C
COUPLING_STREAM_SALT = 0xC0AB7E
# Frozen demand-tape cell: the Q-R1 primary cell rho90_share90 with the
# binary_0.9 cross-campaign demand chain (scripts/c6_perbatch_ceiling.py
# REGIME_PERSISTENCE/DOMINANT_SHARE and retained_context_discovery salt).
DEMAND_REGIME_PERSISTENCE = 0.90
DEMAND_DOMINANT_SHARE = 0.90
DEMAND_KAPPA = 0.90
DEMAND_CHAIN_SALT = 0x5A17C0DE  # identical to retained_context_discovery L99
R24_PENDING_CAP = 5 * 2600  # parent cap, supply_chain.py L5435


def sigma_path(
    history_root: int,
    campaigns: int,
    kappa_r: float,
    seed_salt: int = SIGMA_PATH_SALT,
) -> tuple[str, ...]:
    """Persistent latent sigma chain in {"C","H"} across campaigns.

    Flip probability between consecutive campaigns is ``1 - kappa_r``.  The
    chain is deterministic in (history_root, seed_salt) and independent of
    every simulation RNG stream.
    """
    if not 0.5 <= float(kappa_r) <= 1.0:
        raise ValueError("kappa_r must be in [0.5, 1]")
    if int(campaigns) < 1:
        raise ValueError("at least one campaign is required")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(history_root), int(seed_salt)])
    )
    sigma = SIGMA_VALUES[int(rng.integers(0, 2))]
    output: list[str] = []
    for _campaign in range(int(campaigns)):
        output.append(sigma)
        if float(rng.random()) >= float(kappa_r):
            sigma = SIGMA_VALUES[1 - SIGMA_VALUES.index(sigma)]
    return tuple(output)


def draw_surge_product(
    *, campaign_seed: int, event_index: int, sigma: str, s_r: float,
    coupling_salt: int = COUPLING_STREAM_SALT,
) -> str:
    """Assign one R24 surge to a product: favored with probability s_r.

    Dedicated stream seeded from (salt, campaign seed, event index) so the
    assignment sequence is action-independent (CRN across arms).
    """
    if sigma not in SIGMA_VALUES:
        raise ValueError(f"unknown sigma: {sigma}")
    if not 0.0 <= float(s_r) <= 1.0:
        raise ValueError("s_r must be in [0, 1]")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(coupling_salt), int(campaign_seed), int(event_index)]
        )
    )
    favored = SIGMA_TO_PRODUCT[sigma]
    other = PRODUCTS[1 - PRODUCTS.index(favored)]
    return favored if float(rng.random()) < float(s_r) else other


def demand_initial_regime_chain(
    history_root: int,
    campaigns: int,
    *,
    kappa_demand: float = DEMAND_KAPPA,
    regime_persistence: float = DEMAND_REGIME_PERSISTENCE,
    dominant_share: float = DEMAND_DOMINANT_SHARE,
    weeks: int = 8,
) -> tuple[str, ...]:
    """Per-campaign demand-tape initial regimes, mirroring the burned Q-R1
    history convention (retained_context_discovery.build_campaign_history
    L99-125: same salt, same draw order) without running any DES."""
    rng = np.random.default_rng(int(history_root) ^ DEMAND_CHAIN_SALT)
    initial = PRODUCTS[int(rng.integers(0, 2))]
    output: list[str] = []
    for campaign_index in range(int(campaigns)):
        tape = product_demand_tape(
            int(history_root) * 100 + campaign_index,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
            weeks=int(weeks),
            initial_regime=initial,
        )
        regimes = tuple(map(str, tape["regimes"]))
        output.append(initial)
        stays = bool(rng.random() < float(kappa_demand))
        initial = regimes[-1] if stays else PRODUCTS[1 - PRODUCTS.index(regimes[-1])]
    return tuple(output)


class ProductCoupledProgramODES(ProgramOFullDESSimulation):
    """Program O full DES with the product-coupled R24 surge carrier.

    Overrides (both are no-ops when ``product_coupling_enabled=False`` so the
    class reproduces the parent bit-exactly for the regression self-check):

    * ``_apply_risk_R24_event`` (parent: supply_chain.py L5418-5483): the
      surge quantity draw and event bookkeeping are replicated verbatim, but
      the quantity goes into a per-product pending bucket chosen by
      ``draw_surge_product`` instead of the scalar
      ``_contingent_demand_pending``.
    * ``_op13_demand`` (parent: program_o_full_des.py L570-595): replicated
      verbatim, plus a product-typed merge -- an order absorbs only the
      pending surge bucket of its own tape product.
    """

    def __init__(
        self,
        *,
        product_coupling_enabled: bool = False,
        sigma: str | None = None,
        s_r: float = 1.0,
        coupling_salt: int = COUPLING_STREAM_SALT,
        risk_overrides: Mapping[str, str] | None = None,
        **program_o_kwargs: Any,
    ) -> None:
        super().__init__(**program_o_kwargs)
        # The Program O constructor does not expose risk_overrides; the parent
        # MFSCSimulation reads self.risk_overrides lazily inside the risk
        # loops (supply_chain.py L4886), so setting it before run_contract()
        # is exact.
        self.risk_overrides = dict(risk_overrides or {})
        self.pc_coupling_enabled = bool(product_coupling_enabled)
        self.pc_sigma = str(sigma) if sigma is not None else None
        self.pc_s_r = float(s_r)
        self.pc_coupling_salt = int(coupling_salt)
        self.pc_pending: dict[str, float] = {product: 0.0 for product in PRODUCTS}
        self.pc_surge_log: list[dict[str, Any]] = []
        self.pc_merge_log: list[dict[str, Any]] = []
        self.pc_event_index = 0
        if self.pc_coupling_enabled:
            if self.pc_sigma not in SIGMA_VALUES:
                raise ValueError("coupling requires sigma in {'C','H'}")
            if not 0.0 <= self.pc_s_r <= 1.0:
                raise ValueError("s_r must be in [0, 1]")
            if self.risk_attribution_source == "causal_exposure":
                raise ValueError(
                    "product coupling does not support causal_exposure "
                    "attribution (episodes are product-blind)"
                )

    # ------------------------------------------------------------------
    # R24 surge: product-typed pending buckets
    # ------------------------------------------------------------------
    def _apply_risk_R24_event(self) -> None:
        if not self.pc_coupling_enabled:
            super()._apply_risk_R24_event()
            return
        # --- verbatim quantity draw (supply_chain.py L5419-5426): identical
        # r24 stream consumption keeps the event timeline and quantities CRN
        # with the product-blind parent.
        surge_lo, surge_hi = self._get_risk_surge()
        r24_rng = self._risk_rng_for("R24")
        surge = r24_rng.integers(surge_lo, surge_hi + 1)
        target_cssu = (
            str(r24_rng.choice(("A", "B")))
            if self.cssu_topology_mode == "split_v1"
            else None
        )
        if target_cssu is not None:
            self._contingent_cssu_destination_pending = target_cssu
        # --- product assignment from the dedicated CRN stream.
        event_index = int(self.pc_event_index)
        self.pc_event_index += 1
        assigned = draw_surge_product(
            campaign_seed=int(self.seed or 0),
            event_index=event_index,
            sigma=str(self.pc_sigma),
            s_r=self.pc_s_r,
            coupling_salt=self.pc_coupling_salt,
        )
        favored = SIGMA_TO_PRODUCT[str(self.pc_sigma)]
        # --- bucket admission with the parent's total-pending cap
        # (supply_chain.py L5429-5447 semantics on the bucket total).
        pending_before = float(sum(self.pc_pending.values()))
        accepted_surge = max(
            0.0, min(float(surge), float(R24_PENDING_CAP) - pending_before)
        )
        clipped_surge = max(0.0, float(surge) - accepted_surge)
        self.pc_pending[assigned] += accepted_surge
        self.r24_generated_surge_quantity += float(surge)
        self.r24_admitted_surge_quantity += accepted_surge
        self.r24_clipped_surge_quantity += clipped_surge
        if clipped_surge > 1e-9:
            self.r24_cap_hit_count += 1
        # --- event record identical to the parent (supply_chain.py
        # L5457-5470); description format preserved for comparability.
        window = self.r24_attribution_window_hours
        event = RiskEvent(
            "R24",
            self.env.now,
            self.env.now + window,
            window,
            [13],
            f"+{surge}",
            magnitude=float(surge),
            unit="rations",
            affected_cssu=target_cssu,
        )
        self.risk_events.append(event)
        self._add_ret_quantity_risk(event)
        self.pc_surge_log.append(
            {
                "event_index": event_index,
                "time": float(self.env.now),
                "surge": float(surge),
                "admitted": float(accepted_surge),
                "clipped": float(clipped_surge),
                "sigma": str(self.pc_sigma),
                "favored_product": favored,
                "assigned_product": assigned,
                "assigned_is_favored": bool(assigned == favored),
            }
        )

    # ------------------------------------------------------------------
    # Demand: product-typed contingent merge
    # ------------------------------------------------------------------
    def _op13_demand(self):
        # Verbatim copy of ProgramOFullDESSimulation._op13_demand
        # (program_o_full_des.py L570-595) plus the product-typed merge.
        yield self.program_o_warmup_event
        start = float(self.program_o_decision_start or self.env.now)
        labels = tuple(self.program_o_tape["order_products"])
        order_num = 0
        for week in range(self.program_o_decision_weeks):
            for day, offset in enumerate(self.program_o_demand_offsets_hours):
                target_time = start + week * float(HOURS_PER_WEEK) + float(offset)
                if target_time > self.env.now:
                    yield self.env.timeout(target_time - float(self.env.now))
                demand_qty, is_contingent, causal_ids = (
                    self._sample_calendar_demand_quantity()
                )
                product_id = labels[week * 6 + day]
                if self.pc_coupling_enabled:
                    extra = float(self.pc_pending.get(product_id, 0.0))
                    if extra > 1e-12:
                        demand_qty += extra
                        self.pc_pending[product_id] = 0.0
                        is_contingent = True
                        self.pc_merge_log.append(
                            {
                                "time": float(self.env.now),
                                "order_num": int(order_num + 1),
                                "product_id": product_id,
                                "merged_quantity": extra,
                            }
                        )
                self.total_demanded += demand_qty
                order_num += 1
                order = ProgramOOrderRecord(
                    j=order_num,
                    OPTj=float(self.env.now),
                    quantity=float(demand_qty),
                    remaining_qty=float(demand_qty),
                    contingent=bool(is_contingent),
                    causal_r24_event_ids=causal_ids,
                    requested_product_id=product_id,
                )
                yield from self._place_demand_order(order)


def direct_campaign_metrics(sim: ProgramOFullDESSimulation) -> dict[str, Any]:
    """Metrics for a completed DIRECT run, mirroring the transducer path.

    Early-cohort metrics reuse the shared canonical helper
    (q_r1_retained_learning.early_cohort_metrics_from_orders, the same
    definition as program_o_full_des_transducer.py L827-885: cohort =
    orders with OPTj < decision_start + 336 h).  Full-campaign product
    metrics come from the live sim's product_outcome_panel.
    """
    decision_start = float(sim.program_o_decision_start or 0.0)
    score_time = (
        decision_start
        + float(sim.program_o_decision_weeks) * float(HOURS_PER_WEEK)
        + float(sim.program_o_clearance_hours)
    )
    orders = [
        order
        for order in sim.orders
        if not order.metrics_excluded and float(order.OPTj) >= decision_start
    ]
    early = early_cohort_metrics_from_orders(
        orders=orders, decision_start=decision_start, score_time=score_time
    )
    panel = sim.product_outcome_panel()
    products = panel["products"]
    conservation = panel["conservation"]
    unresolved_orders = sum(
        1 for order in orders if order.OATj is None and not order.lost
    )
    lost_orders = sum(1 for order in orders if order.lost)
    return {
        **early,
        "ret_excel": float(panel["metrics"]["ret_excel"]),
        "worst_product_fill": float(panel["worst_product_fill"]),
        "fill_P_C": float(products["P_C"]["fill"]),
        "fill_P_H": float(products["P_H"]["fill"]),
        "unresolved_orders": float(unresolved_orders),
        "unresolved_quantity": float(
            sum(row["unresolved_quantity"] for row in products.values())
        ),
        "lost_orders": float(lost_orders),
        "lost_quantity": float(
            sum(
                float(order.quantity) for order in orders if order.lost
            )
        ),
        "generated_orders": float(len(orders)),
        "total_demanded_quantity": float(
            sum(float(order.quantity) for order in orders)
        ),
        "r24_events": float(len([e for e in sim.risk_events if e.risk_id == "R24"])),
        "r24_generated_surge_quantity": float(sim.r24_generated_surge_quantity),
        "r24_admitted_surge_quantity": float(sim.r24_admitted_surge_quantity),
        "r24_clipped_surge_quantity": float(sim.r24_clipped_surge_quantity),
        "max_abs_product_residual": float(
            conservation["max_abs_product_residual"]
        ),
        "max_abs_partition_residual": float(
            conservation["max_abs_partition_residual"]
        ),
        "aggregate_state_hash": str(panel["aggregate_state_hash"]),
        "tape_sha256": str(panel["tape_sha256"]),
        "decision_start": decision_start,
        "score_time": score_time,
    }


def run_campaign(
    *,
    root: int,
    campaign_index: int,
    sigma: str,
    s_r: float,
    kappa: float,
    calendar: Sequence[int],
    scheduler: Mapping[str, Sequence[str]],
    regime_persistence: float = DEMAND_REGIME_PERSISTENCE,
    dominant_share: float = DEMAND_DOMINANT_SHARE,
    initial_regime: str | None = None,
    coupling_enabled: bool = True,
    risks_enabled: bool = True,
    risk_overrides: Mapping[str, str] | None = None,
    downstream_freight_physics_mode: str = "fixed_clock_physical_v1",
) -> dict[str, Any]:
    """One direct-DES campaign with the product-coupled R24 carrier.

    ``kappa`` is the cell's kappa_r; it is recorded on the row (the sigma
    chain itself is built by the caller via ``sigma_path``).  The campaign
    seed follows the burned Q-R1 convention: root * 100 + campaign_index.
    """
    seed = int(root) * 100 + int(campaign_index)
    overrides = dict(risk_overrides) if risk_overrides is not None else (
        {"R24": "increased"} if risks_enabled else {}
    )
    sim = ProductCoupledProgramODES(
        product_coupling_enabled=bool(coupling_enabled),
        sigma=str(sigma) if coupling_enabled else None,
        s_r=float(s_r),
        risk_overrides=overrides,
        seed=seed,
        calendar=tuple(int(action) for action in calendar),
        scheduler=scheduler,
        regime_persistence=float(regime_persistence),
        dominant_share=float(dominant_share),
        downstream_freight_physics_mode=str(downstream_freight_physics_mode),
        risks_enabled=bool(risks_enabled),
        enabled_risks={"R24"} if risks_enabled else None,
        risk_rng_mode="per_risk",
        initial_regime=initial_regime,
    ).run_contract()
    metrics = direct_campaign_metrics(sim)
    surge_log = list(sim.pc_surge_log)
    favored_count = sum(1 for row in surge_log if row["assigned_is_favored"])
    return {
        "root": int(root),
        "campaign_index": int(campaign_index),
        "seed": seed,
        "sigma": str(sigma),
        "s_r": float(s_r),
        "kappa_r": float(kappa),
        "calendar": [int(action) for action in calendar],
        "posture": (
            int(calendar[0])
            if len(set(int(action) for action in calendar)) == 1
            else None
        ),
        "coupling_enabled": bool(coupling_enabled),
        "risks_enabled": bool(risks_enabled),
        "risk_overrides": overrides,
        "initial_regime": initial_regime,
        "n_surges": len(surge_log),
        "n_surges_favored": int(favored_count),
        "surge_log": surge_log,
        "merge_log_count": len(sim.pc_merge_log),
        **metrics,
    }


def beta_bernoulli_sigma_posterior(
    history_surge_logs: Iterable[Iterable[Mapping[str, Any]]],
) -> dict[str, float]:
    """P(sigma = C) from past campaigns' surge product assignments.

    Uniform Beta(1, 1) prior; each surge assigned to P_C counts toward C and
    each surge assigned to P_H counts toward H, exactly as frozen in the
    amendment (``observable`` arm).  Only PREVIOUS campaigns' logs may be
    passed in; the caller enforces that boundary.
    """
    n_c = 0
    n_h = 0
    for campaign_log in history_surge_logs:
        for row in campaign_log:
            product = str(row["assigned_product"])
            if product == "P_C":
                n_c += 1
            elif product == "P_H":
                n_h += 1
            else:
                raise ValueError(f"unknown product in surge log: {product}")
    alpha = 1.0 + float(n_c)
    beta = 1.0 + float(n_h)
    return {
        "p_sigma_c": alpha / (alpha + beta),
        "alpha": alpha,
        "beta": beta,
        "n_c": float(n_c),
        "n_h": float(n_h),
    }
