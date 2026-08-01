from supply_chain.service_first_metric import (
    service_first_better,
    service_first_components,
    service_first_key,
)


def test_abandonment_loses_even_when_visible_ret_is_higher():
    abandoned = {
        "lost_orders": 76,
        "flow_fill_rate": 0.51,
        "backorder_qty_final": 0,
        "ret_excel_visible_clipped_0_1": 0.36,
    }
    served = {
        "lost_orders": 0,
        "flow_fill_rate": 0.79,
        "backorder_qty_final": 100,
        "ret_excel_visible_clipped_0_1": 0.001,
    }

    assert service_first_better(served, abandoned)
    assert service_first_key(served)[0] == 1.0
    assert service_first_key(abandoned)[0] == 0.0


def test_service_first_uses_fill_and_queue_before_ret_as_tiebreakers():
    lower_fill = {
        "lost_orders": 0,
        "flow_fill_rate": 0.70,
        "backorder_qty_final": 0,
        "ret_excel_visible_clipped_0_1": 1.0,
    }
    higher_fill = {
        "lost_orders": 0,
        "flow_fill_rate": 0.80,
        "backorder_qty_final": 10_000,
        "ret_excel_visible_clipped_0_1": 0.0,
    }
    assert service_first_better(higher_fill, lower_fill)

    same_fill_lower_queue = dict(higher_fill, backorder_qty_final=0)
    assert service_first_better(same_fill_lower_queue, higher_fill)


def test_components_are_named_and_json_safe():
    components = service_first_components(
        {
            "lost_orders": 0,
            "flow_fill_rate": 0.8,
            "backorder_qty_final": 2.0,
            "ret_excel_visible_clipped_0_1": 0.2,
        }
    )
    assert components == {
        "no_lost_orders": 1.0,
        "flow_fill_rate": 0.8,
        "negative_backorder_qty_final": -2.0,
        "ret_excel_visible_clipped_0_1": 0.2,
    }
