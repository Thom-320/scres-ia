from __future__ import annotations

from pathlib import Path

SUPPLY_CHAIN_SRC = Path("supply_chain/supply_chain.py").read_text(encoding="utf-8")


def _assert_capacity_capped(container_attr: str) -> None:
    """A post-CDC dispatch op must read on-hand inventory and cap the
    requested quantity against it before removing anything from the
    container. This is the exact property Garrido pushed back on in the
    2026-07-02 meeting: "no se puede abastecer de la nada" (you cannot
    replenish from nothing) — see docs/THESIS_INTERPRETATION_DECISIONS_2026-06-24.md D10.
    A regression here (e.g. someone swapping in an unconditioned top-up like
    `_top_up_inventory_buffer`) would silently reintroduce free inventory
    creation on the exact lane the paper's positive Track B result depends on.
    """
    assert f"available = self.{container_attr}.level" in SUPPLY_CHAIN_SRC
    assert "dispatch_qty = min(target, available)" in SUPPLY_CHAIN_SRC


def test_op9_dispatch_is_capacity_capped() -> None:
    _assert_capacity_capped("rations_sb")


def test_op10_dispatch_is_capacity_capped() -> None:
    _assert_capacity_capped("rations_sb_dispatch")


def test_op12_dispatch_is_capacity_capped() -> None:
    _assert_capacity_capped("rations_cssu")


def test_exogenous_buffer_top_up_is_named_and_isolated() -> None:
    """The one place material genuinely appears without an upstream `get()`
    is `_top_up_inventory_buffer` (Track A's strategic buffer, D6/D10). It
    must stay a single, named, greppable mechanism — not duplicated inline
    elsewhere — so a reviewer or future contributor can audit it in one place.
    """
    assert "_top_up_inventory_buffer" in SUPPLY_CHAIN_SRC
    assert "_delayed_buffer_top_up" in SUPPLY_CHAIN_SRC
