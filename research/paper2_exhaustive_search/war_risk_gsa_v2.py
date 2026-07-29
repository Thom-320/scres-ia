"""Validated GSA primitives for the preregistered war-stress timing atlas.

This module does not execute the DES and never selects a scientific scenario by
itself.  It provides four fail-closed pieces for a future custodied runner:

* SALib's standard Morris design and analysis inside one fixed mask/coupling
  stratum;
* a paired-CRN target for the *safe within-cell timing increment*;
* an explicit decomposition of Monte-Carlo noise in that target;
* a cross-fitted stochastic surrogate, with Sobol analysis allowed only after
  the surrogate passes frozen out-of-sample gates and only for independent
  continuous factors.

The target is not regime-tailoring among constants.  It is the restricted
event-timed policy minus the strongest comparator in the same risk cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


MASK_FACTORS: dict[str, tuple[str, ...]] = {
    "LOC_SURGE": ("phi_R22", "phi_R24", "psi_R22", "psi_R24"),
    "THEATER_CAPACITY_SURGE": (
        "phi_R21",
        "phi_R23",
        "phi_R24",
        "psi_R21",
        "psi_R23",
        "psi_R24",
    ),
    "PRODUCTION_QUALITY_SURGE": (
        "phi_R11",
        "phi_R14",
        "phi_R24",
        "psi_R11",
        "psi_R24",
    ),
}

COUPLED_MODES = frozenset(
    {
        "disruption_leads_surge_72h",
        "coincident",
        "surge_leads_disruption_72h",
    }
)

DEFAULT_HIGHER_GUARDRAILS = (
    "ret_excel_full_ledger",
    "ration_ret_excel",
    "ret_excel_cvar10",
    "worst_node_or_product_fill",
)
DEFAULT_LOWER_GUARDRAILS = (
    "lost_orders",
    "ret_excel_omitted_n",
    "backorder_qty_final",
    "backlog_age_max",
    "shift_hours",
    "surge_hours",
    "buffer_target_unit_hours",
    "op8_convoy_vehicle_hours",
    "clipped_surge_quantity",
    "risk_cap_hits",
)


@dataclass(frozen=True)
class FactorSpace:
    """Independent log2 factor space for one fixed mask/coupling stratum."""

    mask: str
    coupling: str
    names: tuple[str, ...]
    log2_bounds: tuple[tuple[float, float], ...]

    @classmethod
    def for_mask(cls, mask: str, coupling: str) -> "FactorSpace":
        if mask not in MASK_FACTORS:
            raise ValueError(f"unknown war-risk mask {mask!r}")
        names = MASK_FACTORS[mask]
        if coupling in COUPLED_MODES:
            # The parent contract replaces every native non-R24 occurrence
            # process with the R24-driven cluster schedule in coupled modes.
            # Consequently only phi_R24 controls cluster frequency; retaining
            # any other phi would create an inert GSA dimension by definition.
            names = tuple(
                name
                for name in names
                if not name.startswith("phi_") or name == "phi_R24"
            )
        bounds = tuple((0.0, 3.0) if name.startswith("phi_") else (0.0, 2.0) for name in names)
        return cls(mask=mask, coupling=coupling, names=names, log2_bounds=bounds)

    @property
    def salib_problem(self) -> dict[str, Any]:
        return {
            "num_vars": len(self.names),
            "names": list(self.names),
            "bounds": [list(bounds) for bounds in self.log2_bounds],
        }

    def physical(self, log2_points: np.ndarray) -> np.ndarray:
        points = np.asarray(log2_points, dtype=float)
        if points.shape[-1] != len(self.names):
            raise ValueError("point dimension does not match factor space")
        return np.power(2.0, points)


@dataclass(frozen=True)
class MorrisDesign:
    space: FactorSpace
    log2_points: np.ndarray
    physical_points: np.ndarray
    trajectories: int
    levels: int


def salib_morris_design(
    space: FactorSpace,
    *,
    candidate_trajectories: int = 20,
    selected_trajectories: int = 10,
    levels: int = 8,
    seed: int = 7470901,
) -> MorrisDesign:
    """Generate a standard optimized Morris design with no clipping shortcuts."""
    if selected_trajectories > candidate_trajectories:
        raise ValueError("selected trajectories cannot exceed candidate trajectories")
    points = morris_sample.sample(
        space.salib_problem,
        N=int(candidate_trajectories),
        num_levels=int(levels),
        optimal_trajectories=int(selected_trajectories),
        local_optimization=True,
        seed=int(seed),
    )
    expected = selected_trajectories * (len(space.names) + 1)
    if points.shape != (expected, len(space.names)):
        raise RuntimeError(f"unexpected Morris shape {points.shape}; expected {(expected, len(space.names))}")
    delta = np.diff(points.reshape(selected_trajectories, len(space.names) + 1, -1), axis=1)
    changed = np.sum(np.abs(delta) > 1e-12, axis=2)
    if not np.all(changed == 1):
        raise RuntimeError("SALib Morris trajectory does not change exactly one factor per step")
    return MorrisDesign(
        space=space,
        log2_points=points,
        physical_points=space.physical(points),
        trajectories=int(selected_trajectories),
        levels=int(levels),
    )


def analyze_salib_morris(
    design: MorrisDesign,
    responses: Sequence[float],
    *,
    bootstrap_resamples: int = 1_000,
    seed: int = 7470902,
) -> dict[str, Any]:
    """Analyze Morris effects; sigma is labelled nonlinearity *or* interaction."""
    values = np.asarray(responses, dtype=float)
    if values.shape != (design.log2_points.shape[0],):
        raise ValueError("one response is required for every Morris point")
    result = morris_analyze.analyze(
        design.space.salib_problem,
        design.log2_points,
        values,
        num_resamples=int(bootstrap_resamples),
        conf_level=0.95,
        scaled=False,
        num_levels=design.levels,
        seed=int(seed),
    )
    return {
        "names": list(result["names"]),
        "mu": np.asarray(result["mu"], dtype=float),
        "mu_star": np.asarray(result["mu_star"], dtype=float),
        "sigma_nonlinearity_or_interaction": np.asarray(result["sigma"], dtype=float),
        "mu_star_conf95": np.asarray(result["mu_star_conf"], dtype=float),
        "n_evaluations": int(values.size),
    }


@dataclass(frozen=True)
class PolicyMetrics:
    policy_id: str
    metrics: Mapping[str, float]
    event_tape_sha256: str
    exogenous_base_stream_sha256: str


@dataclass(frozen=True)
class TimingTapeEvaluation:
    tape_id: int
    comparator: PolicyMetrics
    restricted_candidates: tuple[PolicyMetrics, ...]


@dataclass(frozen=True)
class StochasticTimingResponse:
    config_id: str
    tape_ids: tuple[int, ...]
    event_tape_sha256s: tuple[str, ...]
    exogenous_base_stream_sha256s: tuple[str, ...]
    deltas: np.ndarray
    selected_policy_ids: tuple[str, ...]
    mean: float
    standard_error: float
    favorable_tapes: int


def safe_against(
    candidate: Mapping[str, float],
    comparator: Mapping[str, float],
    *,
    higher_guardrails: Sequence[str] = DEFAULT_HIGHER_GUARDRAILS,
    lower_guardrails: Sequence[str] = DEFAULT_LOWER_GUARDRAILS,
    tolerance: float = 1e-12,
) -> bool:
    """Resource/service safety gate used before any oracle maximization."""
    missing = [
        key
        for key in (*higher_guardrails, *lower_guardrails)
        if key not in candidate or key not in comparator
    ]
    if missing:
        raise KeyError(f"missing safe-oracle metrics: {sorted(set(missing))}")
    return all(
        float(candidate[key]) >= float(comparator[key]) - tolerance
        for key in higher_guardrails
    ) and all(
        float(candidate[key]) <= float(comparator[key]) + tolerance
        for key in lower_guardrails
    )


def compute_h_timing_safe(
    config_id: str,
    tapes: Sequence[TimingTapeEvaluation],
    *,
    primary: str = "ret_excel",
    higher_guardrails: Sequence[str] = DEFAULT_HIGHER_GUARDRAILS,
    lower_guardrails: Sequence[str] = DEFAULT_LOWER_GUARDRAILS,
) -> StochasticTimingResponse:
    """Compute the paired safe within-cell timing increment over a frozen comparator.

    The comparator must already have been selected without tape-level future
    information.  On each tape, the restricted oracle may choose only among
    candidates that are safe relative to that comparator.  If none is safe, the
    comparator itself is retained and the tape delta is exactly zero.
    """
    if not tapes:
        raise ValueError("at least one tape is required")
    ordered = sorted(tapes, key=lambda row: int(row.tape_id))
    tape_ids = tuple(int(row.tape_id) for row in ordered)
    if len(set(tape_ids)) != len(tape_ids):
        raise ValueError("duplicate tape IDs")

    deltas: list[float] = []
    selected: list[str] = []
    event_tape_sha256s: list[str] = []
    base_stream_sha256s: list[str] = []
    for row in ordered:
        comparator = row.comparator
        event_hashes = {comparator.event_tape_sha256}
        event_hashes.update(candidate.event_tape_sha256 for candidate in row.restricted_candidates)
        if len(event_hashes) != 1:
            raise ValueError(f"policy-dependent risk tape on tape {row.tape_id}")
        event_tape_sha256s.append(next(iter(event_hashes)))
        base_hashes = {comparator.exogenous_base_stream_sha256}
        base_hashes.update(
            candidate.exogenous_base_stream_sha256
            for candidate in row.restricted_candidates
        )
        if len(base_hashes) != 1 or not next(iter(base_hashes)):
            raise ValueError(f"policy-dependent or missing base CRN stream on tape {row.tape_id}")
        base_stream_sha256s.append(next(iter(base_hashes)))
        if primary not in comparator.metrics:
            raise KeyError(f"comparator missing primary metric {primary!r}")
        eligible = [
            candidate
            for candidate in row.restricted_candidates
            if safe_against(
                candidate.metrics,
                comparator.metrics,
                higher_guardrails=higher_guardrails,
                lower_guardrails=lower_guardrails,
            )
        ]
        winner = max(
            [comparator, *eligible],
            key=lambda candidate: (float(candidate.metrics[primary]), candidate.policy_id),
        )
        delta = float(winner.metrics[primary]) - float(comparator.metrics[primary])
        deltas.append(delta)
        selected.append(winner.policy_id)

    vector = np.asarray(deltas, dtype=float)
    se = float(np.std(vector, ddof=1) / math.sqrt(vector.size)) if vector.size > 1 else 0.0
    return StochasticTimingResponse(
        config_id=str(config_id),
        tape_ids=tape_ids,
        event_tape_sha256s=tuple(event_tape_sha256s),
        exogenous_base_stream_sha256s=tuple(base_stream_sha256s),
        deltas=vector,
        selected_policy_ids=tuple(selected),
        mean=float(np.mean(vector)),
        standard_error=se,
        favorable_tapes=int(np.sum(vector > 0.0)),
    )


@dataclass(frozen=True)
class NoiseAudit:
    configuration_count: int
    tapes_per_configuration: int
    observed_variance_of_means: float
    configuration_signal_variance: float
    tape_main_effect_variance: float
    configuration_by_tape_variance: float
    monte_carlo_variance_of_paired_configuration_means: float
    monte_carlo_fraction: float
    residuals: np.ndarray


def audit_crn_noise(responses: Sequence[StochasticTimingResponse]) -> NoiseAudit:
    """Two-way CRN decomposition of configuration, tape, and residual variation.

    With common random numbers, configuration-mean errors are correlated through
    the shared tape main effect.  Treating each configuration's within-tape
    variance as independent therefore misstates the noise in comparisons across
    configurations.  This balanced two-way decomposition removes the shared tape
    effect and treats the configuration-by-tape residual as the Monte-Carlo
    uncertainty relevant to paired configuration contrasts.
    """
    if len(responses) < 2:
        raise ValueError("at least two configurations are required")
    reference = responses[0].tape_ids
    if any(response.tape_ids != reference for response in responses[1:]):
        raise ValueError("all configurations must use the same ordered CRN tape IDs")
    reference_hashes = responses[0].exogenous_base_stream_sha256s
    if any(
        response.exogenous_base_stream_sha256s != reference_hashes
        for response in responses[1:]
    ):
        raise ValueError("exogenous base CRN stream hashes differ across GSA configurations")
    n_tapes = len(reference)
    if n_tapes < 2:
        raise ValueError("at least two tapes per configuration are required")
    matrix = np.asarray([response.deltas for response in responses], dtype=float)
    if matrix.shape != (len(responses), n_tapes):
        raise ValueError("response vectors do not form a balanced configuration-by-tape panel")
    n_configurations = matrix.shape[0]
    grand = float(np.mean(matrix))
    configuration_means = np.mean(matrix, axis=1)
    tape_means = np.mean(matrix, axis=0)
    residuals = (
        matrix
        - configuration_means[:, None]
        - tape_means[None, :]
        + grand
    )
    ss_configuration = n_tapes * float(np.sum((configuration_means - grand) ** 2))
    ss_tape = n_configurations * float(np.sum((tape_means - grand) ** 2))
    ss_residual = float(np.sum(residuals**2))
    ms_configuration = ss_configuration / (n_configurations - 1)
    ms_tape = ss_tape / (n_tapes - 1)
    ms_residual = ss_residual / ((n_configurations - 1) * (n_tapes - 1))
    signal = max(0.0, (ms_configuration - ms_residual) / n_tapes)
    tape_variance = max(0.0, (ms_tape - ms_residual) / n_configurations)
    interaction_variance = max(0.0, ms_residual)
    mc_for_paired_means = interaction_variance / n_tapes
    denominator = signal + mc_for_paired_means
    return NoiseAudit(
        configuration_count=n_configurations,
        tapes_per_configuration=n_tapes,
        observed_variance_of_means=float(np.var(configuration_means, ddof=1)),
        configuration_signal_variance=signal,
        tape_main_effect_variance=tape_variance,
        configuration_by_tape_variance=interaction_variance,
        monte_carlo_variance_of_paired_configuration_means=mc_for_paired_means,
        monte_carlo_fraction=(
            mc_for_paired_means / denominator if denominator > 0.0 else 0.0
        ),
        residuals=residuals,
    )


@dataclass
class SurrogateFit:
    model: ExtraTreesRegressor
    variance_model: ExtraTreesRegressor
    factor_space: FactorSpace
    cv_r2: float
    cv_nrmse: float
    cv_spearman: float
    variance_cv_log_rmse: float
    variance_cv_spearman: float
    gate_pass: bool
    minimum_training_points: int

    def predict(self, log2_points: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(log2_points, dtype=float)), dtype=float)

    def predict_variance(self, log2_points: np.ndarray) -> np.ndarray:
        log_variance = self.variance_model.predict(np.asarray(log2_points, dtype=float))
        return np.exp(log_variance)


def fit_cross_validated_surrogate(
    factor_space: FactorSpace,
    log2_points: np.ndarray,
    responses: Sequence[StochasticTimingResponse],
    *,
    seed: int = 7470910,
    folds: int = 5,
    minimum_r2: float = 0.80,
    maximum_nrmse: float = 0.15,
    minimum_spearman: float = 0.90,
) -> SurrogateFit:
    """Fit mean/variance emulators and adjudicate a configuration-level OOS gate."""
    X = np.asarray(log2_points, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(factor_space.names):
        raise ValueError("surrogate design does not match factor space")
    if len(responses) != X.shape[0]:
        raise ValueError("one stochastic response is required per design point")
    minimum_points = max(40, 10 * len(factor_space.names))
    if X.shape[0] < minimum_points:
        raise ValueError(f"surrogate requires at least {minimum_points} configurations")
    noise_audit = audit_crn_noise(responses)
    y = np.asarray([response.mean for response in responses], dtype=float)
    internal_variance = np.maximum(
        np.sum(noise_audit.residuals**2, axis=1) / max(1, noise_audit.tapes_per_configuration - 1),
        1e-12,
    )
    mean_sampling_variance = internal_variance / noise_audit.tapes_per_configuration
    weights = 1.0 / mean_sampling_variance
    weights = np.clip(weights / np.median(weights), 0.1, 10.0)
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=int(seed),
        n_jobs=1,
    )
    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    predictions = cross_val_predict(model, X, y, cv=splitter, params={"sample_weight": weights})
    scale = max(float(np.ptp(y)), 1e-12)
    cv_r2 = float(r2_score(y, predictions))
    cv_nrmse = float(math.sqrt(mean_squared_error(y, predictions)) / scale)
    correlation = spearmanr(y, predictions).statistic
    cv_spearman = float(correlation) if np.isfinite(correlation) else 0.0
    model.fit(X, y, sample_weight=weights)

    variance_model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=int(seed) + 1,
        n_jobs=1,
    )
    log_internal_variance = np.log(internal_variance)
    variance_predictions = cross_val_predict(
        variance_model,
        X,
        log_internal_variance,
        cv=splitter,
        params={"sample_weight": weights},
    )
    variance_cv_log_rmse = float(
        math.sqrt(mean_squared_error(log_internal_variance, variance_predictions))
    )
    if float(np.ptp(log_internal_variance)) <= 1e-12:
        variance_cv_spearman = 1.0
    else:
        variance_correlation = spearmanr(
            log_internal_variance, variance_predictions
        ).statistic
        variance_cv_spearman = (
            float(variance_correlation) if np.isfinite(variance_correlation) else 0.0
        )
    variance_model.fit(X, log_internal_variance, sample_weight=weights)
    passed = bool(
        cv_r2 >= minimum_r2
        and cv_nrmse <= maximum_nrmse
        and cv_spearman >= minimum_spearman
    )
    return SurrogateFit(
        model=model,
        variance_model=variance_model,
        factor_space=factor_space,
        cv_r2=cv_r2,
        cv_nrmse=cv_nrmse,
        cv_spearman=cv_spearman,
        variance_cv_log_rmse=variance_cv_log_rmse,
        variance_cv_spearman=variance_cv_spearman,
        gate_pass=passed,
        minimum_training_points=minimum_points,
    )


def surrogate_sobol_indices(
    fit: SurrogateFit,
    *,
    base_n: int = 4_096,
    bootstrap_resamples: int = 1_000,
    seed: int = 7470920,
    dependent_inputs: bool = False,
) -> dict[str, Any]:
    """Estimate raw Sobol indices only after the frozen surrogate/independence gates."""
    if dependent_inputs:
        raise ValueError("classical Sobol is forbidden for dependent/coupled factor inputs")
    if not fit.gate_pass:
        raise ValueError("surrogate failed the frozen OOS gate; Sobol is not authorized")
    points = sobol_sample.sample(
        fit.factor_space.salib_problem,
        int(base_n),
        calc_second_order=False,
        scramble=True,
        seed=int(seed),
    )
    predictions = fit.predict(points)
    result = sobol_analyze.analyze(
        fit.factor_space.salib_problem,
        predictions,
        calc_second_order=False,
        num_resamples=int(bootstrap_resamples),
        conf_level=0.95,
        print_to_console=False,
        seed=int(seed) + 1,
    )
    s1 = np.asarray(result["S1"], dtype=float)
    st = np.asarray(result["ST"], dtype=float)
    return {
        "names": list(fit.factor_space.names),
        "S1_raw": s1,
        "S1_conf95": np.asarray(result["S1_conf"], dtype=float),
        "ST_raw": st,
        "ST_conf95": np.asarray(result["ST_conf"], dtype=float),
        "overlapping_higher_order_gap_ST_minus_S1": st - s1,
        "base_n": int(base_n),
        "surrogate_cv": {
            "r2": fit.cv_r2,
            "nrmse": fit.cv_nrmse,
            "spearman": fit.cv_spearman,
            "internal_variance_log_rmse": fit.variance_cv_log_rmse,
            "internal_variance_spearman": fit.variance_cv_spearman,
        },
        "claim_limit": "ST-S1 overlaps across factors and is not a unique global interaction mass",
    }
