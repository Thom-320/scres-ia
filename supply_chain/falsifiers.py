"""Shared falsifiers and a mandatory pre-flight, so a check learned once is never dropped again.

WHY THIS EXISTS. On 2026-08-08 nineteen defects shipped across six runners, and they had four
causes rather than nineteen. This module removes two of them structurally:

  1. **A falsifier written for one runner did not reach the next.** The decision-space degeneracy
     check was written twice and omitted the third time -- in the benchmark, where it was needed
     most. Checks live here now and are imported, not retyped.
  2. **`passed` was hardcoded to `True` and the literal was counted in "N falsifiers pass".** A
     project memory named `falsifier-must-be-seen-to-fail` exists because that once let a real data
     leak ship, and it happened again in every runner that day. `check()` refuses a literal and
     `summarise()` counts only computed checks, with disclosures kept in their own field.

USE:
    from supply_chain.falsifiers import check, disclosure, summarise, preflight

    f = {}
    f["f1_endpoint_responds"] = check(spread > 2 * se, "if the endpoint does not move when the "
                                      "action moves, a zero is about the instrument",
                                      spread=spread, two_se=2 * se)
    f["d1_scope"] = disclosure("x16 is a stress probe, not part of the claim", cells=[...])
    summary = summarise(f)      # -> passed / n_computed / n_disclosures / n_not_applicable
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


class FalsifierConstructionError(AssertionError):
    """Raised when a falsifier cannot fail. That is a build error, not a passing test."""


def check(passed: Any, why_it_can_fail: str, *, computed_from: Mapping[str, Any],
          **evidence: Any) -> dict:
    """A falsifier whose verdict came from DATA. `computed_from` must carry the operands.

    Identity cannot separate a computed bool from a literal one -- Python interns `True` -- which
    is why the first version of this module was itself broken. The guard is therefore structural:
    you must hand over the quantities the verdict was derived from, they are recorded as evidence,
    and at least one must be numeric. `check(True, ...)` cannot satisfy that without fabricating
    operands, which is a visible lie rather than an invisible default.
    """
    if not why_it_can_fail.strip():
        raise FalsifierConstructionError("a falsifier must state why it can fail")
    if not computed_from:
        raise FalsifierConstructionError(
            "`computed_from` must carry the operands the verdict came from. If the statement "
            "cannot fail, it is a disclosure: use `disclosure()` so it is reported and NOT counted.")
    if not any(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
               for v in computed_from.values()):
        raise FalsifierConstructionError(
            "`computed_from` needs at least one numeric operand; a verdict with no measured "
            "quantity behind it is a claim, not a test")
    if isinstance(passed, np.bool_):
        passed = bool(passed)
    if not isinstance(passed, bool):
        raise FalsifierConstructionError(f"`passed` must be a bool, got {type(passed).__name__}")
    return {"passed": passed, "computed": True,
            "evidence": {"why_it_can_fail": why_it_can_fail,
                         "computed_from": dict(computed_from), **evidence}}


def gt(observed, threshold, why_it_can_fail: str, **evidence: Any) -> dict:
    """`observed > threshold`, with both operands recorded. The commonest shape, made unfakeable."""
    return check(float(observed) > float(threshold), why_it_can_fail,
                 computed_from={"observed": float(observed), "threshold": float(threshold)},
                 **evidence)


def lt(observed, threshold, why_it_can_fail: str, **evidence: Any) -> dict:
    return check(float(observed) < float(threshold), why_it_can_fail,
                 computed_from={"observed": float(observed), "threshold": float(threshold)},
                 **evidence)


def ge(observed, threshold, why_it_can_fail: str, **evidence: Any) -> dict:
    return check(float(observed) >= float(threshold), why_it_can_fail,
                 computed_from={"observed": float(observed), "threshold": float(threshold)},
                 **evidence)


def disclosure(statement: str, **evidence: Any) -> dict:
    """Something that must travel with the artifact but cannot fail. Never counted as a falsifier."""
    return {"passed": None, "computed": False, "disclosure": True,
            "evidence": {"statement": statement, **evidence}}


def not_applicable(reason: str, **evidence: Any) -> dict:
    return {"passed": None, "computed": False, "not_applicable": True,
            "evidence": {"reason": reason, **evidence}}


def summarise(falsifiers: Mapping[str, Any]) -> dict:
    """`all_passed` is over COMPUTED checks only; disclosures and N/A are reported separately.

    Counting a disclosure or a not-applicable inside "N falsifiers pass" overstates validation, and
    that overstatement reached three sealed artifacts in one day.
    """
    computed = {k: v for k, v in falsifiers.items()
                if isinstance(v, dict) and v.get("computed") is True}
    disclosures = [k for k, v in falsifiers.items()
                   if isinstance(v, dict) and v.get("disclosure")]
    na = [k for k, v in falsifiers.items()
          if isinstance(v, dict) and v.get("not_applicable")]
    failed = [k for k, v in computed.items() if not v["passed"]]
    return {"all_passed": not failed and bool(computed),
            "n_computed": len(computed), "n_failed": len(failed), "failed": failed,
            "n_disclosures": len(disclosures), "disclosures": disclosures,
            "n_not_applicable": len(na), "not_applicable": na}


# --------------------------------------------------------------------------------------------
# The permutation null. Added 2026-08-08 after it retracted a positive claim of our own.
# --------------------------------------------------------------------------------------------

def selection_gap(matrix: np.ndarray, train_idx: Sequence[int],
                  test_idx: Sequence[int]) -> float:
    """`E_test[min over options]` minus the train-selected fixed column, evaluated on test.

    This is the shape of every "clairvoyant ceiling" this project computes, and it is BIASED
    UPWARD: a minimum over K noisy draws sits below the minimum of their means by Jensen, so the
    gap is positive even when every option has the same true value.
    """
    fixed = int(np.argmin(np.asarray(matrix)[list(train_idx)].mean(axis=0)))
    block = np.asarray(matrix)[list(test_idx)]
    return float((block[:, fixed] - block.min(axis=1)).mean())


def permutation_null(matrix: np.ndarray, train_idx: Sequence[int], test_idx: Sequence[int],
                     *, n_draws: int = 20_000, rng=None,
                     statistic: Callable[..., float] = selection_gap) -> dict:
    """Null distribution of `statistic` when the TAPE x OPTION INTERACTION carries no information.

    THE FIRST VERSION OF THIS FUNCTION WAS WRONG, and the mutation tests below caught it. It
    permuted labels within each row, which leaves `row.min()` numerically UNCHANGED -- a
    permutation of a row cannot move its minimum -- so it never touched the term it claimed to
    test. It only randomised the fixed column, which means a gap SMALLER than that null was
    evidence the train-selected schedule beats a random one, the opposite of how it was read.

    Per-tape headroom IS the interaction: some schedules suit some tapes. So the null fits the
    additive model `J[i,j] ~ mu + a_i + b_j`, keeps it, and permutes only the RESIDUALS. Tape
    difficulty and schedule quality both survive; only "this schedule suits this tape" is
    destroyed. Under that null a positive gap is exactly the Jensen bias of a minimum over K
    options, which is what an observed gap has to beat.
    """
    rng = np.random.default_rng() if rng is None else rng
    m = np.asarray(matrix, dtype=float)
    observed = float(statistic(m, train_idx, test_idx))
    grand = m.mean()
    fitted = m.mean(axis=1, keepdims=True) + m.mean(axis=0, keepdims=True) - grand
    resid = m - fitted
    flat = resid.ravel()
    draws = np.empty(int(n_draws))
    for b in range(int(n_draws)):
        draws[b] = statistic(fitted + rng.permutation(flat).reshape(m.shape),
                             train_idx, test_idx)
    return {"gap_observed": observed, "n_draws": int(n_draws),
            "null_mean": float(draws.mean()),
            "null_p50": float(np.percentile(draws, 50)),
            "null_p95": float(np.percentile(draws, 95)),
            "p_value": float((draws >= observed).mean()),
            "null_model": "additive mu + a_i + b_j retained; residuals permuted"}


def survives_permutation_null(matrix: np.ndarray, train_idx: Sequence[int],
                              test_idx: Sequence[int], *, alpha: float = 0.05,
                              n_draws: int = 20_000, rng=None,
                              statistic: Callable[..., float] = selection_gap) -> dict:
    """A falsifier: does the measured gap exceed what a no-interaction world produces?

    Without it, the winner's curse gets reported as headroom: a minimum over K options is biased
    upward whatever the truth. With the WRONG null it is worse than useless, which is why the
    mutation tests below assert that it both rejects pure noise AND accepts a genuine per-tape
    advantage. A check that can only fail is a rejection rule, not a test.
    """
    out = permutation_null(matrix, train_idx, test_idx, n_draws=n_draws, rng=rng,
                           statistic=statistic)
    return lt(out["p_value"], alpha,
              "a minimum over K options is biased upward, so a positive gap is what a world with "
              "no tape-option interaction already produces; without this the winner's curse is "
              "reported as headroom",
              **out)


# --------------------------------------------------------------------------------------------
# The pre-flight. Every item below corresponds to a defect that reached a sealed artifact.
# --------------------------------------------------------------------------------------------

def preflight(*, probe: Callable[[Any], float], options: Sequence[Any],
              reset_now: float, horizon: float, scenario: Mapping[str, Any],
              expected_scenario: Mapping[str, Any], min_distinct: int = 3) -> dict:
    """Run before any expensive campaign. Every item below is a defect that reached a sealed
    artifact on 2026-08-08; a failure here means do not run.

    `probe(option) -> endpoint value` is evaluated on every option in `options`.
    """
    arr = np.asarray([float(probe(o)) for o in options], dtype=float)
    distinct = int(len(np.unique(np.round(arr, 9))))
    n_matching = sum(1 for k, v in expected_scenario.items() if scenario.get(k) == v)
    return {
        "p1_endpoint_responds_to_the_action": gt(
            float(arr.max() - arr.min()), 0.0,
            "a dead action reads as a measured null: a posture grid that never reached S1 returned "
            "byte-identical episodes and shipped H = 0 with every other falsifier green",
            values=[float(x) for x in arr]),
        "p2_decision_space_has_more_than_one_effective_level": ge(
            distinct, min_distinct,
            "an effective dimension of one means a single bit predicts the optimum, so any "
            "comparison between things that choose is unidentifiable -- the benchmark's decision "
            "space collapsed to 'activate in the first eligible week'",
            values=[float(x) for x in arr]),
        "p3_reset_leaves_time_inside_the_horizon": lt(
            reset_now, horizon,
            "a warm-up that consumes the horizon makes the cell ineligible rather than hard: "
            "R13 at x16 left env.now at 161,280 h against a 4,368 h horizon and was reported as a "
            "one-step episode"),
        "p4_scenario_is_the_declared_one": ge(
            n_matching, len(expected_scenario),
            "a runner that silently falls back to a default scenario is not measuring the "
            "experiment it names: the per-risk sensitivity reverted to thesis_uniform while the "
            "benchmark it was meant to explain used garrido_seasonal_v1",
            declared=dict(expected_scenario), actual=dict(scenario)),
    }
