from scripts.evaluate_program_q_replication import scheduler
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar
from supply_chain.program_t_full_des_mpc import FullDEST0Config, choose_ret_transducer_action


def test_ret_transducer_provider_is_deterministic():
    sched = scheduler()
    skeleton, _ = extract_full_des_skeleton(seed=949200003, scheduler=sched, regime_persistence=.75, dominant_share=.9, downstream_freight_physics_mode="fixed_clock_physical_v1")
    _, rows = state_rich_calendar(skeleton=skeleton.as_dict(), scheduler=sched, config=StateRichConfiguration("belief_mpc", 1), regime_persistence=.75, dominant_share=.9, action_overrides=(0,) * 8)
    config = FullDEST0Config(1, "scenario", particles=3)
    left = choose_ret_transducer_action(rows[0].observation, base_skeleton=skeleton, prefix=(), scheduler=sched, config=config)
    right = choose_ret_transducer_action(rows[0].observation, base_skeleton=skeleton, prefix=(), scheduler=sched, config=config)
    assert left == right
