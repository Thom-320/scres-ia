"""G3a reconstruction: latent regime, warning, endpoint and controller library.

Contract: `docs/PREREGISTRO_G3A_V2_RECONSTRUCCION_2026-08-08.md`, frozen before this file.

WHY IT IS REBUILT RATHER THAN CITED. The external package describes this campaign and its own
manifest records `g3a_code_and_raw_results_in_remote_head: false` -- runner, contract and raw
results were deleted before they were pushed. Prose cannot be replayed, audited or superseded, so
its numbers are TARGETS TO REPRODUCE here, never evidence.

THE ENDPOINT EXISTS BECAUSE THE OBVIOUS ONE IS DEGENERATE. On-time fill is identically zero under
every policy: the recorded fulfilment delay is 54 h against a 48 h promise, so no policy can ever
deliver on time and the metric cannot discriminate control. Measured, not assumed --
`claimant_fills` returns 0.0 for every allocation share. Late EXPOSURE keeps the information that a
binary deadline throws away: how much is late, and for how long.

It is POST-HOC and says so. It was defined after discovering the timing conflict, it is not a
validated SCRES measure, and it cannot carry a confirmatory claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

REGIMES = ("A_PRESSURE", "NEUTRAL", "B_PRESSURE")
SELF_TRANSITION = 0.78
PRESSURED_SHARE = 0.75
WARNING_ACCURACY = 0.72
WEEKS = 16
HOURS_PER_WEEK = 168.0


@dataclass(frozen=True)
class Cell:
    """One factorial cell. Frozen so an arm cannot edit the world it is scored in."""

    process: str          # "iid" | "persistent"
    demand: str           # "uniform" | "seasonal"
    contract: str         # "hard_quota" | "spare_reallocation" | "global_pool"

    @property
    def label(self) -> str:
        return f"{self.process}_{self.demand}_{self.contract}"

    def sim_kwargs(self) -> dict:
        return {
            "cssu_topology_mode": "split_v1",
            "cssu_reallocate_unused": self.contract == "spare_reallocation",
            "cssu_global_pool": self.contract == "global_pool",
        }


def regime_tape(seed: int, process: str) -> list[str]:
    """Weekly latent regime. Drawn from a hash of the seed, NOT from a simulator RNG stream.

    Keeping it out of the simulator's generators is what makes the nine cells comparable: the same
    tape keeps the same risks, demand magnitudes and draw order whichever regime process is in
    force, so a difference between cells cannot be a different world.
    """
    out, current = [], "NEUTRAL"
    for week in range(WEEKS):
        u = _u(seed, f"regime:{process}:{week}")
        if process == "iid" or not out:
            current = REGIMES[min(int(u * 3), 2)]
        elif u >= SELF_TRANSITION:
            others = [r for r in REGIMES if r != current]
            current = others[0] if _u(seed, f"switch:{week}") < 0.5 else others[1]
        out.append(current)
    return out


def warning_tape(seed: int, regimes: list[str]) -> list[str]:
    """Observable signal, right with probability `WARNING_ACCURACY`, available BEFORE the action."""
    out = []
    for week, regime in enumerate(regimes):
        if _u(seed, f"warn:{week}") < WARNING_ACCURACY:
            out.append(regime)
        else:
            others = [r for r in REGIMES if r != regime]
            out.append(others[0] if _u(seed, f"warnpick:{week}") < 0.5 else others[1])
    return out


def share_schedule(regimes: list[str]) -> list[float]:
    return [PRESSURED_SHARE if r == "A_PRESSURE"
            else (1.0 - PRESSURED_SHARE) if r == "B_PRESSURE" else 0.5 for r in regimes]


def _u(seed: int, key: str) -> float:
    digest = sha256(f"g3a-v2:{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def worst_claimant_late_exposure_service(sim) -> dict[str, float]:
    """1 - normalised late exposure, per claimant; the endpoint is the WORSE of the two.

    Exposure integrates outstanding quantity from an order's deadline to its delivery or to the
    measurement close, normalised by the maximum exposure that order could accrue. Orders whose
    deadline falls after the close are excluded from the denominator rather than counted as
    perfect, because an order that could not yet be late says nothing about control.
    """
    horizon, start = float(sim.env.now), float(sim.warmup_time)
    num = {"A": 0.0, "B": 0.0}
    den = {"A": 0.0, "B": 0.0}
    for order in sim.orders:
        if bool(getattr(order, "metrics_excluded", False)):
            continue
        cssu = getattr(order, "cssu_destination", None)
        if cssu not in ("A", "B"):
            continue
        opt = float(getattr(order, "OPTj", 0.0) or 0.0)
        if opt < start:
            continue
        due = opt + float(order.LTj or 0.0)
        if due >= horizon:            # could not have been late before the close
            continue
        qty = float(order.quantity or 0.0)
        end = float(order.OATj) if getattr(order, "OATj", None) is not None else horizon
        num[cssu] += qty * max(0.0, end - due)
        den[cssu] += qty * (horizon - due)
    service = {c: (1.0 - num[c] / den[c]) if den[c] > 1e-9 else 1.0 for c in ("A", "B")}
    service["worst"] = min(service["A"], service["B"])
    return service


# ---------------------------------------------------------------------------------------------
# Controller library. Enumerated in the contract and closed. Each returns the allocation share for
# a week given ONLY what its interface is allowed to see.
# ---------------------------------------------------------------------------------------------

def _from_signal(signal: str, *, inverted: bool = False) -> float:
    share = {"A_PRESSURE": 0.7, "NEUTRAL": 0.5, "B_PRESSURE": 0.3}[signal]
    return 1.0 - share if inverted else share


class Controller:
    """`privileged` marks an arm that reads the true regime: diagnostic, never deployable."""

    def __init__(self, name: str, kind: str, *, privileged: bool = False, **cfg):
        self.name, self.kind, self.privileged, self.cfg = name, kind, privileged, cfg

    def shares(self, tape: dict) -> list[float]:
        k, cfg = self.kind, self.cfg
        warn, regimes = tape["warnings"], tape["regimes"]
        if k == "constant":
            return [cfg["level"]] * WEEKS
        if k == "warning":
            return [_from_signal(w, inverted=cfg.get("inverted", False)) for w in warn]
        if k == "warning_shuffled":          # placebo: mechanism kept, information destroyed
            order = np.random.default_rng(cfg["salt"]).permutation(WEEKS)
            return [_from_signal(warn[int(i)]) for i in order]
        if k == "warning_delayed":           # placebo: the signal arrives one week too late
            return [0.5] + [_from_signal(w) for w in warn[:-1]]
        if k == "belief":
            # ONE common transition model in every cell -- never the cell's generating matrix,
            # which is the leak the package's internal audit found in its own first analysis.
            belief = np.array([1 / 3, 1 / 3, 1 / 3])
            common = np.full((3, 3), (1 - 0.7) / 2)
            np.fill_diagonal(common, 0.7)
            out = []
            for week in range(WEEKS):
                like = np.array([WARNING_ACCURACY if r == warn[week]
                                 else (1 - WARNING_ACCURACY) / 2 for r in REGIMES])
                post = belief * like
                post = post / max(post.sum(), 1e-12)
                out.append(float(0.7 * post[0] + 0.5 * post[1] + 0.3 * post[2]))
                belief = post @ common if cfg.get("stateful", True) else np.array([1/3, 1/3, 1/3])
            return out
        if k == "lagged_demand":
            return [0.5] + [0.7 if d > 0 else 0.3 for d in tape["lagged_a_minus_b"][:-1]]
        if k == "backlog_threshold":
            return [0.7 if b > cfg["theta"] else 0.5 for b in tape["backlog_a_share"]]
        if k == "true_state":                # PRIVILEGED diagnostic: not deployable, not a bound
            return [_from_signal(r) for r in regimes]
        raise ValueError(k)


LIBRARY: list[Controller] = (
    [Controller(f"const_{lvl:.1f}", "constant", level=lvl) for lvl in (0.3, 0.4, 0.5, 0.6, 0.7)]
    + [Controller("warning_lookup", "warning"),
       Controller("warning_inverted", "warning", inverted=True),
       Controller("placebo_shuffled", "warning_shuffled", salt=20260808),
       Controller("placebo_delayed", "warning_delayed"),
       Controller("belief_stateful", "belief", stateful=True),
       Controller("belief_reset", "belief", stateful=False),
       Controller("lagged_demand", "lagged_demand"),
       Controller("backlog_threshold", "backlog_threshold", theta=0.5),
       Controller("privileged_true_state", "true_state", privileged=True)]
)
