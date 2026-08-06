from scripts.run_search_comparator_ladder_v1 import CONTEXT_ORDER as V1_CONTEXT_ORDER
from scripts.run_search_comparator_ladder_v2 import CONTEXT_ORDER as V2_CONTEXT_ORDER


EXPECTED = ("R1r", "R2r", "R1r+R2r", "R1r|esc", "R2r|esc", "R1r+R2r|esc")


def test_both_search_ladders_use_the_contractual_career_order():
    assert V1_CONTEXT_ORDER == EXPECTED
    assert V2_CONTEXT_ORDER == EXPECTED

