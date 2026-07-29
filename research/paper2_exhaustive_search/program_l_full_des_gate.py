"""Full-DES pre-learner gate for Program L route recourse.

Runs the FAITHFUL ProgramLRouteRecourseEnv (flags-off identity PROVEN against ProgramEConvoyEnv)
under a controlled Op8 R22 schedule (frequency x duration), from thesis-native (rare, ~24h) to
the disclosed multi-day relaxation. Measures canonical ret_excel headroom of a deployable
signal-routing policy over the best full-contract static route policy. NO learner is trained.

Shared dispatch TRIGGER across all policies (staging-threshold, = DRA-2 closed question) so the
screen isolates the ROUTE-CHOICE value, not dispatch timing. Policies differ only in which route.
"""

from __future__ import annotations
import copy
import json
import numpy as np
from supply_chain.dra2_experiment import digest, materialize_tape
from supply_chain.program_l_route_recourse_env import (
    ProgramLRouteRecourseEnv,
    RouteContract,
    make_identity_normalizers,
)

EP_DAYS = 56
EP_HOURS = EP_DAYS * 24


def build_tape(seed: int, n_r22: int, dur: float, base_family: str = "op8_interruption"):
    """Canonical warmup tape with a CONTROLLED Op8 R22 schedule (disclosed relaxation knob)."""
    t = copy.deepcopy(
        materialize_tape(seed=seed, family=base_family, horizon_weeks=EP_DAYS // 7 + 1, split="dev")
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x2202]))
    events = []
    for _ in range(n_r22):
        st = float(rng.integers(0, max(1, EP_HOURS - int(dur))))
        events.append(
            {
                "risk_id": "R22",
                "start_time": st,
                "end_time": st + dur,
                "duration": dur,
                "affected_ops": [8],
                "description": "programL_controlled",
                "magnitude": 1.0,
                "unit": "incidents",
            }
        )
    t["risk_events"] = sorted(events, key=lambda e: e["start_time"])
    # The controlled intervention changes the tape body; retain no stale identity.
    t.pop("sha256", None)
    t["sha256"] = digest(t)
    return t


DISPATCH_STAGING = 2500.0
DISPATCH_WAIT = 48.0
DISPATCH_URGENT = 96.0


def _want_dispatch(raw) -> bool:
    return (
        raw["op7_staged_inventory"] >= DISPATCH_STAGING
        or raw["staging_age"] >= DISPATCH_WAIT
        or raw["oldest_backlog_age"] >= DISPATCH_URGENT
    )


def choose_const(route):
    def f(raw, mask, env):
        if not _want_dispatch(raw):
            return 0
        return route if (route < len(mask) and mask[route]) else (1 if mask[1] else 0)

    return f


def choose_alternate():
    st = {"n": 0}

    def f(raw, mask, env):
        if not _want_dispatch(raw):
            return 0
        st["n"] += 1
        r = 1 + (st["n"] % 2)
        return r if mask[r] else (1 if mask[1] else (2 if mask[2] else 0))

    return f


def choose_signal():
    """Deployable: if route-1 (Op8) is signalled down and the alternate is feasible + signalled
    healthy, take ROUTE_2; else ROUTE_1 (fast primary)."""

    def f(raw, mask, env):
        if not _want_dispatch(raw):
            return 0
        r1_down = raw["sig_route1_down"] > 0.5
        r2_bad = raw["sig_route2_degraded"] > 0.5
        if r1_down and mask[2] and not r2_bad:
            return 2
        if r1_down and not mask[1] and mask[2]:
            return 2  # primary infeasible -> alternate
        return 1 if mask[1] else (2 if mask[2] else 0)

    return f


def choose_clairvoyant():
    """Diagnostic true-state myopic rule; this is not a full-horizon PI oracle."""

    def f(raw, mask, env):
        if not _want_dispatch(raw):
            return 0
        now_rel = int(float(env.sim.env.now) - env.start)
        lead = int(env.contract.signal_lead_hours)
        lo = max(0, now_rel)
        hi = min(len(env._op8_down), lo + max(1, lead))
        op8_down = hi > lo and env._op8_down[lo:hi].any()
        d = min(env._cur_day(), len(env._Z2) - 1)
        z2_deg = bool(env._Z2[d])
        # route-1 effective transit ~24h (or stalled while down); route-2 ~36h(+24 if deg)
        r1_cost = 24.0 + (env._op8_down[lo:hi].sum() if op8_down else 0.0)
        r2_cost = env.contract.route2_base_outbound_h + (
            env.contract.route2_degraded_penalty_h if z2_deg else 0.0
        )
        pref = 2 if (r2_cost < r1_cost) else 1
        if pref == 1 and mask[1]:
            return 1
        if pref == 2 and mask[2]:
            return 2
        return 1 if mask[1] else (2 if mask[2] else 0)

    return f


def choose_placebo(seed):
    rng = np.random.default_rng(seed)

    def f(raw, mask, env):
        if not _want_dispatch(raw):
            return 0
        r1_down = rng.random() < 0.5
        if r1_down and mask[2]:
            return 2
        return 1 if mask[1] else (2 if mask[2] else 0)

    return f


def run_episode(tape, chooser, contract):
    env = ProgramLRouteRecourseEnv(
        [tape],
        make_identity_normalizers(),
        contract=contract,
        episode_days=EP_DAYS,
        random_tapes=False,
    )
    env.reset(options={"tape": tape})
    done = False
    last = {}
    while not done:
        raw = env.raw_observation()
        mask = env.action_masks()
        a = chooser(raw, mask, env)
        _, _, done, _, info = env.step(a)
        last = info
    return float(last["ret_excel"]), float(last.get("episode_departures", 0.0))


def boot_ci(delta, nb=3000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(nb, len(delta)))
    bm = delta[idx].mean(axis=1)
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def screen(n_r22, dur, contract, n_tapes=40, seed0=8_500_001):
    tapes = [build_tape(seed0 + i, n_r22, dur) for i in range(n_tapes)]

    def ev(chooser_factory, tag_seed=None):
        ch = chooser_factory() if tag_seed is None else chooser_factory(tag_seed)
        r = np.array([run_episode(t, ch, contract) for t in tapes])
        return r[:, 0], r[:, 1]

    s1, d1 = ev(lambda: choose_const(1))
    s2, d2 = ev(lambda: choose_const(2))
    sa, da = ev(choose_alternate)
    statics = {"const_R1": (s1, d1), "const_R2": (s2, d2), "alternate": (sa, da)}
    bkey = max(statics, key=lambda k: statics[k][0].mean())
    static, static_dep = statics[bkey]
    obs, obs_dep = ev(choose_signal)
    clv, _ = ev(choose_clairvoyant)
    plac, _ = ev(lambda: choose_placebo(seed0))
    heuristic_true_state_delta = float((clv - static).mean())
    Hobs = float((obs - static).mean())
    return {
        "n_r22": n_r22,
        "dur_h": dur,
        "n": n_tapes,
        "best_static": bkey,
        "heuristic_true_state_delta": heuristic_true_state_delta,
        "H_PI_certified": False,
        "H_obs": Hobs,
        "H_obs_ci95": boot_ci(obs - static),
        "H_obs_placebo": float((plac - static).mean()),
        "real_beats_placebo": bool(Hobs > (plac - static).mean()),
        "eta_diagnostic": (
            float(Hobs / heuristic_true_state_delta)
            if abs(heuristic_true_state_delta) > 1e-9
            else 0.0
        ),
        "dep_obs_minus_static": float(obs_dep.mean() - static_dep.mean()),
    }


if __name__ == "__main__":
    C = RouteContract()
    grid = []
    # thesis-native-ish -> disclosed multi-day/frequent
    for n_r22, dur in [(1, 24), (2, 24), (4, 24), (2, 72), (4, 72), (4, 120), (6, 120), (8, 72)]:
        grid.append(screen(n_r22, dur, C, n_tapes=40))
        r = grid[-1]
        print(
            f"n_R22={r['n_r22']} dur={r['dur_h']:.0f}h  heuristic_true_state_delta={r['heuristic_true_state_delta']:+.4f}  H_obs={r['H_obs']:+.4f} "
            f"LCB95={r['H_obs_ci95'][0]:+.4f}  eta_diagnostic={r['eta_diagnostic']:.2f}  plac={r['H_obs_placebo']:+.4f} "
            f"real>plac={r['real_beats_placebo']}  ddep={r['dep_obs_minus_static']:+.1f}  static={r['best_static']}"
        )
    json.dump(
        {
            "schema": "program_l_full_des_development_screen_v2",
            "generated": "2026-07-13",
            "contract": C.__dict__,
            "episode_days": EP_DAYS,
            "note": "FULL-DES development screen on canonical ret_excel; diagnostic true-state rule is NOT H_PI; comparator frontier incomplete; NO learner trained",
            "grid": grid,
        },
        open("results/paper2_search/program_l_full_des_gate.json", "w"),
        indent=2,
    )
    print("saved results/paper2_search/program_l_full_des_gate.json")
