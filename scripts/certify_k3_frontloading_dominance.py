#!/usr/bin/env python3
"""Certify the frozen K3 exact-budget front-loading dominance theorem.

This checker consumes no stochastic tape.  It enumerates the *effective*
action graph induced by ``_budget_feasible_quantity(..., exact_budget=True)``
and binds the proof assumptions to the current Python sources with AST checks
and source hashes.

The theorem is deliberately contract-local.  It certifies zero perfect-
information headroom for K3's frozen ``ret_excel_full_ledger_order`` endpoint.
It does not promote the workbook-visible sparse-ledger endpoint: that endpoint
changes its scored population when an order is lost, so it remains subject to
the mandatory lost-order guardrail and to a separate metric-specific audit.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import inspect
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRACT_PATH = ROOT / "contracts" / "program_k3_ret_budgeted_replenishment_v1.json"
KERNEL_PATH = ROOT / "supply_chain" / "replenish_ret.py"
METRIC_ADAPTER_PATH = ROOT / "supply_chain" / "program_g.py"
RET_PATH = ROOT / "supply_chain" / "ret_thesis.py"
ORDER_RECORD_PATH = ROOT / "supply_chain" / "supply_chain.py"
TAPE_PATH = ROOT / "supply_chain" / "replenish.py"
OUTPUT_PATH = (
    ROOT
    / "research"
    / "paper2_exhaustive_search"
    / "k3_frontloading_dominance_certificate.json"
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one top-level function {name!r}, got {len(matches)}")
    return matches[0]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one top-level class {name!r}, got {len(matches)}")
    return matches[0]


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError("AST node has no source segment")
    return segment


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        parts = [target.attr]
        value = target.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate) == name
    ]


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


@dataclass(frozen=True)
class ResourceGraph:
    scale_D0: float
    weeks: int
    budget_units: int
    weekly_cap_units: int
    level_units: tuple[int, ...]
    states: tuple[tuple[int, int], ...]
    edges: tuple[tuple[int, int, int, int], ...]
    schedules: tuple[tuple[int, ...], ...]


def enumerate_effective_resource_graph() -> ResourceGraph:
    """Enumerate reachable actions using the actual K3 exact-budget coercion."""
    from supply_chain import replenish_ret as kernel

    contract = _load_json(CONTRACT_PATH)
    physics = contract["physics"]
    scale = 0.25
    weeks = int(physics["episode_weeks"])
    budget_units = int(round(float(physics["total_replenishment_budget_D0"]) / scale))
    cap_units = int(round(float(physics["weekly_order_cap_D0"]) / scale))
    level_units = tuple(
        int(round(float(level) / scale)) for level in physics["order_grid_D0"]
    )
    assert tuple(float(level) for level in kernel.LEVELS) == tuple(
        float(level) for level in physics["order_grid_D0"]
    )
    assert kernel.WEEKS == weeks
    assert int(round(kernel.BUDGET_D0 / scale)) == budget_units
    assert int(round(kernel.WEEKLY_CAP_D0 / scale)) == cap_units

    prefixes: set[tuple[int, ...]] = {()}
    states: set[tuple[int, int]] = {(0, 0)}
    edges: set[tuple[int, int, int, int]] = set()
    for week in range(weeks):
        next_prefixes: set[tuple[int, ...]] = set()
        for prefix in sorted(prefixes):
            spent_units = sum(prefix)
            actual_next_units: set[int] = set()
            for requested_units in level_units:
                quantity = kernel._budget_feasible_quantity(
                    requested_units * scale,
                    spent_units * scale,
                    week,
                    exact_budget=True,
                )
                quantity_units = int(round(quantity / scale))
                actual_next_units.add(quantity_units)

            # Independently characterize every feasible exact-budget edge.
            weeks_after = weeks - week - 1
            mathematical_next_units = {
                quantity_units
                for quantity_units in level_units
                if spent_units + quantity_units <= budget_units
                and spent_units + quantity_units + cap_units * weeks_after
                >= budget_units
            }
            assert actual_next_units == mathematical_next_units

            for quantity_units in sorted(actual_next_units):
                next_spent = spent_units + quantity_units
                edges.add((week, spent_units, quantity_units, next_spent))
                states.add((week + 1, next_spent))
                next_prefixes.add(prefix + (quantity_units,))
        prefixes = next_prefixes

    assert {sum(schedule) for schedule in prefixes} == {budget_units}
    return ResourceGraph(
        scale_D0=scale,
        weeks=weeks,
        budget_units=budget_units,
        weekly_cap_units=cap_units,
        level_units=level_units,
        states=tuple(sorted(states)),
        edges=tuple(sorted(edges)),
        schedules=tuple(sorted(prefixes)),
    )


def _front_loaded_schedule(graph: ResourceGraph) -> tuple[int, ...]:
    remaining = graph.budget_units
    result: list[int] = []
    for _ in range(graph.weeks):
        quantity = min(graph.weekly_cap_units, remaining)
        result.append(quantity)
        remaining -= quantity
    assert remaining == 0
    return tuple(result)


def _prefix_sums(row: Iterable[int]) -> tuple[int, ...]:
    total = 0
    result = []
    for value in row:
        total += value
        result.append(total)
    return tuple(result)


def verify_under_budget_envelope_dominance(
    graph: ResourceGraph, front: tuple[int, ...]
) -> dict[str, Any]:
    """Exhaust the broader ``total spend <= B`` calendar envelope by DP.

    The frozen K3 action projector forces exact terminal spend.  Paper 2's
    comparator rule is broader, however: an open-loop policy may consume
    non-superior resources.  This finite-state count therefore includes every
    eight-week calendar on the declared quarter-D0 action grid whose total is
    at most the budget.  It is a resource-envelope dominance result, not a
    claim that an under-spending calendar uses resources equal to ``front``.
    """
    front_prefix = _prefix_sums(front)
    path_counts: dict[int, int] = {0: 1}
    states: set[tuple[int, int]] = {(0, 0)}
    edges: set[tuple[int, int, int, int]] = set()
    prefix_violations = 0
    maximum_spend_by_layer: dict[int, int] = {0: 0}
    for week in range(graph.weeks):
        next_counts: dict[int, int] = {}
        for spent, count in sorted(path_counts.items()):
            for quantity in graph.level_units:
                next_spent = spent + quantity
                if next_spent > graph.budget_units:
                    continue
                edges.add((week, spent, quantity, next_spent))
                states.add((week + 1, next_spent))
                next_counts[next_spent] = next_counts.get(next_spent, 0) + count
                if next_spent > front_prefix[week]:
                    prefix_violations += count
        path_counts = next_counts
        maximum_spend_by_layer[week + 1] = max(path_counts)

    schedule_count = sum(path_counts.values())
    exact_budget_count = path_counts.get(graph.budget_units, 0)
    assert schedule_count == 5_758_374
    assert exact_budget_count == len(graph.schedules)
    assert prefix_violations == 0
    assert all(
        maximum_spend_by_layer[week] <= front_prefix[week - 1]
        for week in range(1, graph.weeks + 1)
    )
    return {
        "method": "exact finite-state dynamic-programming enumeration",
        "schedule_count_total_spend_le_budget": schedule_count,
        "exact_budget_schedule_count": exact_budget_count,
        "strictly_under_budget_schedule_count": schedule_count - exact_budget_count,
        "reachable_state_count": len(states),
        "reachable_edge_count": len(edges),
        "maximum_spend_quarter_units_by_layer": {
            str(week): maximum_spend_by_layer[week]
            for week in range(graph.weeks + 1)
        },
        "prefix_dominance_violation_count": prefix_violations,
        "resource_scope": "Calendars may use less than 10*D0; they are within the same upper-bound resource envelope but are not described as pairwise equal-resource to F.",
    }


def visible_ledger_nonmonotonicity_counterexample() -> dict[str, Any]:
    """Give an exact-budget K3 counterexample to visible-v1 monotonicity.

    Both action schedules are feasible on the frozen grid, spend exactly B,
    and see the same deterministic demand.  F prefix-dominates q and completes
    every order weakly earlier, with the same visible/lost population.  The
    sparse workbook ledger nevertheless rewards q's later batching.
    """
    from supply_chain import replenish_ret as kernel
    from supply_chain.replenish import D0
    from supply_chain.ret_thesis import compute_order_level_ret_excel_visible_ledger
    from supply_chain.supply_chain import OrderRecord

    front_actions = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.0)
    batched_actions = (1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5)
    demand_D0 = (3.2, 1.0, 0.3, 1.1, 0.3, 1.0, 1.0, 3.2)

    def rollout(actions: tuple[float, ...]) -> tuple[list[OrderRecord], float]:
        inventory = D0
        pipeline = [0.0]
        pending: list[OrderRecord] = []
        orders: list[OrderRecord] = []
        spent = 0.0
        for week in range(kernel.WEEKS):
            hour = float(week * 168)
            inventory += pipeline.pop(0)
            quantity = kernel._budget_feasible_quantity(
                actions[week], spent, week, exact_budget=True
            )
            spent += quantity
            pipeline.append(quantity * D0)
            demand = demand_D0[week] * D0
            order = OrderRecord(
                j=week + 1,
                OPTj=hour,
                quantity=demand,
                remaining_qty=demand,
                LTj=168.0,
            )
            pending.append(order)
            orders.append(order)
            pending, inventory = kernel._complete_pending(
                pending, inventory, hour
            )
        horizon = float(kernel.WEEKS * 168)
        for order in pending:
            order.lost = True
            order.lost_time = horizon
        return orders, spent

    front_orders, front_spend = rollout(front_actions)
    batched_orders, batched_spend = rollout(batched_actions)
    front_result = compute_order_level_ret_excel_visible_ledger(
        front_orders, current_time=kernel.WEEKS * 168.0
    )
    batched_result = compute_order_level_ret_excel_visible_ledger(
        batched_orders, current_time=kernel.WEEKS * 168.0
    )
    front_oat = [
        None if order.OATj is None else float(order.OATj / 168.0)
        for order in front_orders
    ]
    batched_oat = [
        None if order.OATj is None else float(order.OATj / 168.0)
        for order in batched_orders
    ]
    assert front_spend == batched_spend == kernel.BUDGET_D0
    assert all(
        front_value >= batched_value
        for front_value, batched_value in zip(
            _prefix_sums(int(4 * value) for value in front_actions),
            _prefix_sums(int(4 * value) for value in batched_actions),
            strict=True,
        )
    )
    assert sum(order.lost for order in front_orders) == 1
    assert sum(order.lost for order in batched_orders) == 1
    assert all(
        f is None or (q is not None and f <= q)
        for f, q in zip(front_oat, batched_oat, strict=True)
    )
    assert abs(front_result["mean_ret_excel"] - float(Fraction(6, 7))) < 1e-15
    assert batched_result["mean_ret_excel"] == 1.0
    return {
        "status": "PASS_LIVE_AGGREGATOR_COUNTEREXAMPLE",
        "purpose": "Proves inside the frozen K3 action/resource kernel that pathwise earlier completion and equal lost counts do not order ret_excel_visible_v1.",
        "demand_D0": list(demand_D0),
        "front_loaded_path": {
            "actions_D0": list(front_actions),
            "completion_weeks": front_oat,
            "lost_order_ids": [order.j for order in front_orders if order.lost],
            "ordered_D0": front_spend,
            "ret_values": front_result["ret_values"],
            "mean_ret_excel": front_result["mean_ret_excel"],
        },
        "later_batched_path": {
            "actions_D0": list(batched_actions),
            "completion_weeks": batched_oat,
            "lost_order_ids": [order.j for order in batched_orders if order.lost],
            "ordered_D0": batched_spend,
            "ret_values": batched_result["ret_values"],
            "mean_ret_excel": batched_result["mean_ret_excel"],
        },
        "mechanism": "F completes order 1 at week 2 when order 2 has activated but is still open, so its j=1 row sees Bt=1 and scores zero. q completes orders 1-3 together at week 3; completion-removal events are applied before all same-time rows, so its j=1 row sees Bt=0.",
        "demand_support_scope": "Every demand value is in the lower-clipped Gaussian generator's mathematical support, but the extreme 3.2*D0 draws are rare; this is a universal-theorem counterexample, not a practical-frequency estimate.",
        "claim_limit": "The feasible open-loop example disproves H_PI=0 when F is assumed to be a universal visible-v1 upper comparator. It does not estimate K3 H_PI, establish H_obs>0, or reopen the blocked K3 lane without a new mechanism.",
    }


def _abstract_full_ledger_score(statuses: tuple[int, ...]) -> Fraction:
    """Exact risk-free K3 full-ledger score.

    Status order is ``0=on_time < 1=late < 2=lost``.  Late and lost rows both
    increment the cumulative bad-order ledger before scoring; a lost row itself
    scores zero.  This is the K3 specialization of the bound source functions.
    """
    cumulative_bad = 0
    total = Fraction(0, 1)
    for j, status in enumerate(statuses, start=1):
        if status > 0:
            cumulative_bad += 1
        if status != 2:
            total += Fraction(j - cumulative_bad, j)
    return total / len(statuses)


def verify_full_ledger_metric_monotonicity(weeks: int) -> dict[str, Any]:
    """Exhaust the full status lattice and bind it to the live aggregator."""
    from supply_chain.ret_thesis import compute_order_level_ret_excel_formula
    from supply_chain.supply_chain import OrderRecord

    vectors = tuple(product(range(3), repeat=weeks))
    exact_scores = {vector: _abstract_full_ledger_score(vector) for vector in vectors}

    # First prove the finite abstraction is exactly what the current aggregator
    # computes for every on-time/late/lost status vector.
    abstraction_mismatches = 0
    maximum_abs_difference = 0.0
    for statuses in vectors:
        orders = []
        for j, status in enumerate(statuses, start=1):
            opt = float((j - 1) * 168)
            order = OrderRecord(
                j=j,
                OPTj=opt,
                quantity=1.0,
                remaining_qty=1.0 if status == 2 else 0.0,
                LTj=168.0,
            )
            if status == 0:
                order.OATj = opt
                order.CTj = 0.0
            elif status == 1:
                order.OATj = opt + 336.0
                order.CTj = 336.0
                order.backorder = True
            else:
                order.lost = True
                order.lost_time = float(weeks * 168)
            orders.append(order)
        observed = float(
            compute_order_level_ret_excel_formula(
                orders, j_source="row_index"
            )["mean_ret_excel"]
        )
        expected = float(exact_scores[statuses])
        difference = abs(observed - expected)
        maximum_abs_difference = max(maximum_abs_difference, difference)
        if difference > 1e-12:
            abstraction_mismatches += 1
    assert abstraction_mismatches == 0

    # Earlier completion maps each order weakly downward in this status order.
    # Exhaust every coordinatewise pair F<=q, a superset of physically reachable
    # FIFO pairs, and require score(F)>=score(q).
    pair_count = 0
    violation_count = 0
    strict_pair_count = 0
    for candidate in vectors:
        candidate_score = exact_scores[candidate]
        for front in product(*(range(status + 1) for status in candidate)):
            pair_count += 1
            front_score = exact_scores[front]
            if front_score < candidate_score:
                violation_count += 1
            elif front_score > candidate_score:
                strict_pair_count += 1
    assert violation_count == 0
    return {
        "status_vectors_checked_against_live_aggregator": len(vectors),
        "abstraction_mismatch_count": abstraction_mismatches,
        "maximum_abs_difference": maximum_abs_difference,
        "coordinatewise_status_pairs_checked": pair_count,
        "strict_metric_improvement_pair_count": strict_pair_count,
        "metric_dominance_violation_count": violation_count,
        "status_order": ["on_time", "late", "lost"],
    }


def audit_source_semantics() -> dict[str, Any]:
    """Fail closed unless current source implements every theorem premise."""
    contract = _load_json(CONTRACT_PATH)
    kernel_source = KERNEL_PATH.read_text()
    metric_source = METRIC_ADAPTER_PATH.read_text()
    ret_source = RET_PATH.read_text()
    order_record_source = ORDER_RECORD_PATH.read_text()
    tape_source = TAPE_PATH.read_text()
    kernel_tree = ast.parse(kernel_source)
    metric_tree = ast.parse(metric_source)
    ret_tree = ast.parse(ret_source)
    order_record_tree = ast.parse(order_record_source)
    tape_tree = ast.parse(tape_source)
    rollout = _function(kernel_tree, "rollout_actions")
    complete = _function(kernel_tree, "_complete_pending")
    feasible = _function(kernel_tree, "_budget_feasible_quantity")
    metric = _function(metric_tree, "ret_order_metrics")
    full_ledger = _function(ret_tree, "compute_order_level_ret_excel_formula")
    order_record = _class(order_record_tree, "OrderRecord")
    materialize_tape = _function(tape_tree, "materialize_tape")

    checks: dict[str, bool] = {}

    # One-week lead: receive the pipeline head at the start of a week, and only
    # then append this week's complete order to the tail.
    checks["contract_one_week_lead"] = contract["physics"]["lead_time_weeks"] == 1
    checks["kernel_one_week_lead_constant"] = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "LEAD_WEEKS" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 1
        for node in kernel_tree.body
    )
    rollout_text = _source_segment(kernel_source, rollout)
    receive_index = rollout_text.index("inventory += pipeline.pop(0)")
    append_index = rollout_text.index("pipeline.append(quantity * D0)")
    checks["receive_before_order_append"] = receive_index < append_index
    checks["pipeline_receives_head_exactly"] = (
        "inventory += pipeline.pop(0)" in _source_segment(kernel_source, rollout)
    )
    checks["new_order_enters_tail_unclipped"] = (
        "pipeline.append(quantity * D0)" in _source_segment(kernel_source, rollout)
    )
    checks["identical_initial_inventory_one_D0"] = (
        contract["physics"]["initial_inventory_D0"] == 1.0
        and "inventory = D0" in rollout_text
    )

    # Demand is a frozen exogenous tape load, and no code in rollout stores to tape.
    demand_assignments = [
        node
        for node in ast.walk(rollout)
        if isinstance(node, ast.Assign)
        and "demand = float(tape.demand[week])" in _source_segment(kernel_source, node)
    ]
    tape_stores = [
        node
        for node in ast.walk(rollout)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if _root_name(target) == "tape"
    ]
    checks["demand_loaded_from_frozen_tape"] = len(demand_assignments) == 1
    checks["rollout_does_not_mutate_tape"] = not tape_stores
    tape_text = _source_segment(tape_source, materialize_tape)
    checks["rtape_generator_demand_is_nonnegative"] = (
        "demand = np.clip" in tape_text
        and "0.3 * D0" in tape_text
        and "RTape(" in tape_text
        and "demand=demand" in tape_text
    )
    order_constructor_calls = _calls(rollout, "OrderRecord")
    checks["k3_orders_do_not_inject_risk_fields"] = (
        len(order_constructor_calls) == 1
        and not {
            keyword.arg
            for keyword in order_constructor_calls[0].keywords
            if keyword.arg is not None
        }
        & {"APj", "RPj", "DPj", "ret_risk_indicators"}
    )
    record_defaults = {
        statement.target.id: statement.value
        for statement in order_record.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is not None
    }
    checks["order_record_risk_periods_default_zero"] = all(
        isinstance(record_defaults.get(name), ast.Constant)
        and record_defaults[name].value == 0.0
        for name in ("APj", "RPj", "DPj")
    )
    risk_default = record_defaults.get("ret_risk_indicators")
    checks["order_record_risk_indicators_default_empty"] = (
        isinstance(risk_default, ast.Call)
        and _call_name(risk_default) == "field"
        and any(
            keyword.arg == "default_factory"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "dict"
            for keyword in risk_default.keywords
        )
    )

    # FIFO and work conservation: iterate the pending list in-place, use all
    # possible inventory for each oldest order, and preserve list order.
    fifo_loops = [
        node
        for node in complete.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "order"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "pending"
    ]
    complete_text = _source_segment(kernel_source, complete)
    checks["fifo_iterates_pending_in_original_order"] = len(fifo_loops) == 1
    checks["fifo_work_conserving_take"] = (
        "take = min(inventory, float(order.remaining_qty))" in complete_text
        and "inventory -= take" in complete_text
        and "order.remaining_qty -= take" in complete_text
    )
    checks["fifo_filter_preserves_survivor_order"] = (
        "[order for order in pending if order.remaining_qty > 1e-9]" in complete_text
    )
    checks["completion_time_is_current_review_hour"] = (
        "order.OATj = float(hour)" in complete_text
    )

    # The frozen inventory kernel has no storage capacity, spoilage, expiry, or
    # holding-cost branch.  The only cap is the declared order resource cap in
    # the exact-budget action projector.
    rollout_and_fifo = (complete_text + "\n" + _source_segment(kernel_source, rollout)).lower()
    forbidden_inventory_physics = ("spoil", "expir", "shelf", "storage_cap", "inventory_cap")
    checks["no_spoilage_or_inventory_capacity_branch"] = not any(
        token in rollout_and_fifo for token in forbidden_inventory_physics
    )
    feasible_text = _source_segment(kernel_source, feasible)
    checks["only_declared_action_caps_in_projector"] = (
        "WEEKLY_CAP_D0" in feasible_text
        and "BUDGET_D0" in feasible_text
        and "exact_budget" in feasible_text
    )
    checks["holding_not_in_primary_contract"] = contract["physics"]["holding_cost_in_primary"] is False
    checks["holding_excluded_from_metric_call"] = (
        "metrics = ret_order_metrics(orders)" in _source_segment(kernel_source, rollout)
        and "holding" not in _source_segment(metric_source, metric).lower()
    )

    # Episode/horizon and loss convention.
    checks["contract_and_kernel_horizon_eight_weeks"] = (
        contract["physics"]["episode_weeks"] == 8
        and any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "WEEKS" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value == 8
            for node in kernel_tree.body
        )
        and "for week in range(WEEKS)" in rollout_text
        and "horizon = float(WEEKS * 168)" in rollout_text
    )
    checks["pending_orders_lost_only_at_horizon"] = (
        "for order in pending:" in rollout_text
        and "order.lost = True" in rollout_text
        and "order.lost_time = horizon" in rollout_text
    )

    # Metric binding: K3 calls the complete-order ledger, not the sparse visible
    # ledger, with the canonical increment-before-score logic.
    metric_text = _source_segment(metric_source, metric)
    full_ledger_text = _source_segment(ret_source, full_ledger)
    checks["frozen_contract_names_full_ledger_metric"] = (
        contract["primary_metric"] == "ret_excel_full_ledger_order"
    )
    checks["k3_adapter_calls_full_ledger_aggregator"] = (
        len(_calls(metric, "compute_order_level_ret_excel_formula")) == 1
        and "compute_order_level_ret_excel_visible_ledger" not in metric_text
        and 'j_source="row_index"' in metric_text
    )
    checks["full_ledger_increments_lost_and_late_before_score"] = (
        "cumulative_unattended += 1" in full_ledger_text
        and "cumulative_backorders += 1" in full_ledger_text
        and "compute_ret_per_order_excel_formula" in full_ledger_text
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError(f"K3 source-semantic bindings failed: {failed}")
    return {
        "status": "PASS",
        "checks": checks,
        "source_sha256": {
            str(CONTRACT_PATH.relative_to(ROOT)): _sha256(CONTRACT_PATH),
            str(KERNEL_PATH.relative_to(ROOT)): _sha256(KERNEL_PATH),
            str(METRIC_ADAPTER_PATH.relative_to(ROOT)): _sha256(METRIC_ADAPTER_PATH),
            str(RET_PATH.relative_to(ROOT)): _sha256(RET_PATH),
            str(ORDER_RECORD_PATH.relative_to(ROOT)): _sha256(ORDER_RECORD_PATH),
            str(TAPE_PATH.relative_to(ROOT)): _sha256(TAPE_PATH),
        },
        "loaded_function_source_sha256": {
            "rollout_actions": sha256(inspect.getsource(__import__(
                "supply_chain.replenish_ret", fromlist=["rollout_actions"]
            ).rollout_actions).encode()).hexdigest(),
            "_complete_pending": sha256(inspect.getsource(__import__(
                "supply_chain.replenish_ret", fromlist=["_complete_pending"]
            )._complete_pending).encode()).hexdigest(),
        },
    }


def build_certificate() -> dict[str, Any]:
    graph = enumerate_effective_resource_graph()
    front = _front_loaded_schedule(graph)
    assert front in graph.schedules
    front_prefix = _prefix_sums(front)

    prefix_violations: list[dict[str, Any]] = []
    minimum_margin = None
    strict_schedule_count = 0
    for schedule in graph.schedules:
        schedule_prefix = _prefix_sums(schedule)
        margins = tuple(
            front_value - candidate_value
            for front_value, candidate_value in zip(front_prefix, schedule_prefix, strict=True)
        )
        if any(margin < 0 for margin in margins):
            prefix_violations.append(
                {"schedule": schedule, "margins": margins}
            )
        if any(margin > 0 for margin in margins):
            strict_schedule_count += 1
        row_minimum = min(margins)
        minimum_margin = row_minimum if minimum_margin is None else min(minimum_margin, row_minimum)
    assert not prefix_violations
    assert minimum_margin == 0  # all exact-budget schedules tie at terminal spend
    assert strict_schedule_count == len(graph.schedules) - 1

    state_counts = Counter(week for week, _ in graph.states)
    source_audit = audit_source_semantics()
    metric_monotonicity = verify_full_ledger_metric_monotonicity(graph.weeks)
    visible_counterexample = visible_ledger_nonmonotonicity_counterexample()
    scale = graph.scale_D0
    front_D0 = [value * scale for value in front]
    under_budget = verify_under_budget_envelope_dominance(graph, front)

    return {
        "schema_version": "k3_frontloading_dominance_certificate_v1",
        "status": "PASS_EXACT_PATHWISE_FRONTLOADING_DOMINANCE__FROZEN_K3_FULL_LEDGER_H_PI_ZERO",
        "generated_without_stochastic_tapes": True,
        "contract_id": _load_json(CONTRACT_PATH)["contract_id"],
        "scope": {
            "included": "Frozen eight-week K3 exact-budget replenishment kernel and ret_excel_full_ledger_order metric.",
            "excluded": [
                "ret_excel_visible_v1 as an unconditional monotone objective",
                "any replenishment contract with spoilage, storage clipping, receipt clipping, action-dependent holding cost, endogenous demand, non-FIFO allocation, or a different lead/horizon",
                "all new or unburned tapes",
            ],
        },
        "resource_graph": {
            "quarter_D0_scale": scale,
            "weeks": graph.weeks,
            "budget_quarter_units": graph.budget_units,
            "weekly_cap_quarter_units": graph.weekly_cap_units,
            "action_quarter_units": list(graph.level_units),
            "reachable_state_count": len(graph.states),
            "reachable_state_counts_by_layer": {
                str(week): state_counts[week] for week in range(graph.weeks + 1)
            },
            "reachable_edge_count": len(graph.edges),
            "effective_exact_budget_schedule_count": len(graph.schedules),
            "terminal_states": [list(state) for state in graph.states if state[0] == graph.weeks],
            "all_terminal_spends_equal_budget": all(
                sum(schedule) == graph.budget_units for schedule in graph.schedules
            ),
            "enumeration_sha256": sha256(
                json.dumps(
                    {
                        "states": graph.states,
                        "edges": graph.edges,
                        "schedules": graph.schedules,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "front_loaded_schedule": {
            "actions_D0": front_D0,
            "actions_quarter_units": list(front),
            "prefix_spend_D0": [value * scale for value in front_prefix],
            "total_D0": sum(front_D0),
            "is_reachable": front in graph.schedules,
            "identity": "F=(1.5,1.5,1.5,1.5,1.5,1.5,1.0,0.0)*D0",
        },
        "exhaustive_prefix_certificate": {
            "schedules_checked": len(graph.schedules),
            "prefix_comparisons_checked": len(graph.schedules) * graph.weeks,
            "violation_count": len(prefix_violations),
            "strictly_prefix_dominated_schedule_count": strict_schedule_count,
            "unique_nondominated_schedule_count": len(graph.schedules) - strict_schedule_count,
            "algebraic_identity": "sum_{t<k} q_t <= min(1.5*k,10.0) = sum_{t<k} F_t for every k=1,...,8",
        },
        "non_superior_resource_envelope_certificate": under_budget,
        "source_semantics": source_audit,
        "full_ledger_metric_monotonicity": metric_monotonicity,
        "visible_ledger_nonmonotonicity_counterexample": visible_counterexample,
        "pathwise_coupling_theorem": {
            "premises": [
                "Demand D_j is the same nonnegative exogenous tape under every policy.",
                "A week-t order is received in full at the start of week t+1.",
                "Initial inventory is identical and service is work-conserving FIFO.",
                "There is no spoilage, receipt/storage capacity clipping, or action-dependent service penalty.",
            ],
            "supply_order": "With zero-based review epoch w in {0,...,7}, S_F(w)=D0*(1+sum_{t<w}F_t) >= D0*(1+sum_{t<w}q_t)=S_q(w).",
            "completion_implication": "Order j is generated at zero-based epoch j-1. For any w>=j-1, OAT_q(j)<=w implies cumulative demand through j <= S_q(w) <= S_F(w), hence OAT_F(j)<=w. Therefore OAT_F(j)<=OAT_q(j), with infinity for horizon-unfulfilled orders.",
            "backlog_implication": "B_F(w)=max(0,cumulative demand through w-S_F(w)) <= B_q(w) at every review epoch.",
            "loss_implication": "Every order attended by q by the horizon is attended by F; lost_F<=lost_q and attended_F>=attended_q pathwise.",
            "resource_implication": "Every effective exact-budget K3 policy trajectory and F spend exactly 10*D0. Separately, all 5,758,374 grid calendars spending <=10*D0 obey the same weekly cap and are prefix-dominated by F; those under-spending comparisons are within the resource envelope, not pairwise equal-resource.",
        },
        "metric_conclusions": {
            "frozen_k3_full_ledger": {
                "metric": "ret_excel_full_ledger_order",
                "result": "H_PI=0 exactly",
                "reason": "Earlier completion can only turn a full-ledger row from lost/late to late/on-time. F weakly dominates every exact-budget perfect-information or non-anticipative trajectory pathwise and every grid calendar using non-superior total resources. Because F itself is admissible open loop inside B, oracle minus strongest open loop is exactly zero.",
                "h_pi": 0.0,
                "h_obs": 0.0,
                "learned_incremental_value_upper_bound": 0.0,
            },
            "ret_excel_visible_v1": {
                "result": "NOT_UNCONDITIONALLY_CERTIFIED_BY_THIS_THEOREM",
                "mandatory_guardrail": "Any candidate compared with F must have lost_candidate<=lost_F. The coupling already gives lost_F<=lost_candidate, so eligibility requires exact equality of lost counts.",
                "reason_for_scope_limit": "The sparse visible ledger omits unfulfilled rows and scores rows at their completion times. A policy that completes fewer or later rows can change the scored population or ledger snapshot; OAT dominance alone is not asserted here to imply mean visible-ReT dominance.",
                "paper2_rule": "No visible-v1 advantage may reopen K3 unless lost counts are equal, the visible-ledger aggregator is recomputed on the same path, and the ranking survives a separate metric-specific proof/audit. This certificate burns no tapes and supplies no such rescue.",
            },
        },
        "scientific_verdict": {
            "k3_full_ledger_family_state": "FORMALLY_DOMINATED_BY_FULL_HORIZON_OPEN_LOOP_F",
            "paper2_adaptive_value": False,
            "paper3_authorized": False,
            "no_new_tapes_consumed": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the checked-in JSON byte-for-byte")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    certificate = build_certificate()
    rendered = json.dumps(certificate, indent=2, sort_keys=True) + "\n"
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.check:
        if not output.exists() or output.read_text() != rendered:
            raise SystemExit(f"stale or missing K3 certificate: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    try:
        output_label = str(output.relative_to(ROOT))
    except ValueError:
        output_label = str(output)
    print(
        json.dumps(
            {
                "status": certificate["status"],
                "states": certificate["resource_graph"]["reachable_state_count"],
                "schedules": certificate["resource_graph"]["effective_exact_budget_schedule_count"],
                "prefix_violations": certificate["exhaustive_prefix_certificate"]["violation_count"],
                "output": output_label,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
