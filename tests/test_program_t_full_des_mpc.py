from supply_chain.program_o_state_rich import StateRichObservation
from supply_chain.program_t_full_des_mpc import FullDEST0Config, choose_t0_action


SCHEDULER = {"0": ["P_C"] * 3, "1": ["P_C", "P_C", "P_H"], "2": ["P_C", "P_H", "P_H"], "3": ["P_H"] * 3}


def observation(backlog=(0.0, 0.0)):
    return StateRichObservation(0, 0.0, (5000.0, 5000.0), (0.0, 0.0), backlog, (0, 0), (0.0, 0.0), (0.0, 0.0), 0.9, 0.82, None, 8, "a" * 64)


def test_t0_action_is_deterministic_and_observable_only():
    config = FullDEST0Config(3, "scenario", particles=8)
    assert choose_t0_action(observation(), scheduler=SCHEDULER, config=config) == choose_t0_action(observation(), scheduler=SCHEDULER, config=config)


def test_t0_action_responds_to_product_backlog():
    config = FullDEST0Config(1, "nominal")
    c, _ = choose_t0_action(observation((20000.0, 0.0)), scheduler=SCHEDULER, config=config)
    h, _ = choose_t0_action(observation((0.0, 20000.0)), scheduler=SCHEDULER, config=config)
    assert c < h
