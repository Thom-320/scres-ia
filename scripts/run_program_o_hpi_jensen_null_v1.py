#!/usr/bin/env python3
"""Is Program O's H_PI = 0.1515 a prize, or the Jensen bias of a max over 65,536 calendars?

Preregistration: `docs/PREREGISTRO_NULO_JENSEN_SOBRE_H_PI_DE_PROGRAM_O_2026-08-12.md`

H_PI is the only material headroom figure this project has left, and its form is the one we now
know noise inflates: mean-of-per-tape-maxima minus the best single calendar, with the max taken
over 65,536 options. Yesterday the same shape gave +0.000065 observed against a null whose mean was
+0.003978, and in August a clairvoyant ceiling that had passed its interaction null died to exactly
this test on fresh seeds.

O'S FUNGIBLE NULL DOES NOT CONTROL THIS. `exact_fungible_null_h_pi = 0.0` exactly can only arise if
all 65,536 calendars are identical under fungibility -- zero variance across actions, hence no
selection bias to accumulate. It is a null of PHYSICS, not of ESTIMATOR, and the estimator has never
been tested.

THE NULL permutes the calendar axis WITHIN each tape, applying the same permutation to every metric
so each calendar keeps its whole metric vector and the safety mask stays coherent. Only calendar
identity ACROSS tapes is destroyed. The per-tape max is therefore unchanged; what moves is the best
single calendar, which stops being informative. What remains is exactly the selection bias.

This run cannot promote Program O -- it is closed and immutable. It can only withdraw it.
No seed is opened: the calendar matrices are already on disk.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from screen_program_o_full_des_hpi import profile_summary                        # noqa: E402
from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402

CONTRACT = Path("docs/PREREGISTRO_NULO_JENSEN_SOBRE_H_PI_DE_PROGRAM_O_2026-08-12.md")
OUT = Path("results/program_o/hpi_jensen_null_v1/result.json")
MATRIX_ROOT = Path("outputs/program_o_runs/program-o-full-des-validation-v2-20260715/"
                   "artifacts/validation/raw_calendar_matrix")
SEALED = Path("results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json")
O_CONTRACT = Path("contracts/program_o_full_des_hpi_translation_v1.json")
PRIMARY_PROFILE = "rho75_share90__centered_minority_v1"
FUNGIBLE_PROFILE = "fungible_null__centered_minority_v1"
N_DRAWS = 1_000
RNG_SEED = 20260812


def load_panel(profile: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(MATRIX_ROOT / profile / "tape_*.npz")))
    if not files:
        raise FileNotFoundError(f"no calendar matrices for {profile!r}")
    keys = list(np.load(files[0]).keys())
    return {k: np.stack([np.load(f)[k] for f in files]) for k in keys}


def permute_panel(panel: dict[str, np.ndarray], rng) -> dict[str, np.ndarray]:
    """One permutation of the calendar axis per tape, applied to EVERY metric identically."""
    n_tapes, n_cal = panel["ret_visible"].shape
    perms = np.stack([rng.permutation(n_cal) for _ in range(n_tapes)])
    rows = np.arange(n_tapes)[:, None]
    return {k: v[rows, perms] for k, v in panel.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()
    started = time.perf_counter()

    contract = json.loads(O_CONTRACT.read_text())
    sealed = json.loads(SEALED.read_text())["primary"]

    panel = load_panel(PRIMARY_PROFILE)
    observed = profile_summary(panel, contract)
    reproduction = {
        "raw_h_pi": {"recomputed": observed["raw_h_pi"], "sealed": sealed["raw_h_pi"]},
        "safe_h_pi": {"recomputed": observed["safe_h_pi"], "sealed": sealed["safe_h_pi"]},
    }
    exact = (abs(observed["raw_h_pi"] - sealed["raw_h_pi"]) < 1e-12
             and abs(observed["safe_h_pi"] - sealed["safe_h_pi"]) < 1e-12)
    print(f"  reproduccion exacta: {exact}  safe={observed['safe_h_pi']:.17g}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    null_raw, null_safe = np.empty(args.draws), np.empty(args.draws)
    for i in range(args.draws):
        s = profile_summary(permute_panel(panel, rng), contract)
        null_raw[i], null_safe[i] = s["raw_h_pi"], s["safe_h_pi"]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.draws} ({time.perf_counter() - started:.0f}s) "
                  f"media safe {null_safe[:i + 1].mean():.6f}", flush=True)

    def summarise_null(draws, obs):
        return {"mean": float(draws.mean()), "sd": float(draws.std(ddof=1)),
                "p95": float(np.quantile(draws, 0.95)),
                "p_value": float((draws >= obs).mean()), "n_draws": int(draws.size),
                "observed": float(obs)}

    safe_null = summarise_null(null_safe, observed["safe_h_pi"])
    raw_null = summarise_null(null_raw, observed["raw_h_pi"])

    # n4: is the fungible null degenerate in the way the diagnosis claims?
    fung = load_panel(FUNGIBLE_PROFILE)
    fung_var = float(np.mean(fung["ret_visible"].var(axis=1)))
    fung_summary = profile_summary(fung, contract)

    checks = {
        "n2_the_reproduction_is_exact": F.check(
            exact,
            "if the sealed figure cannot be recomputed from the raw matrices to 1e-12 then I am "
            "reading the data wrong and nothing else in this artifact means anything",
            computed_from={"raw_gap": abs(observed["raw_h_pi"] - sealed["raw_h_pi"]),
                           "safe_gap": abs(observed["safe_h_pi"] - sealed["safe_h_pi"])},
            reproduction=reproduction),
        "n3_the_null_is_not_degenerate": F.check(
            safe_null["sd"] > 0.0,
            "a permutation that moves nothing tests nothing; the null must have spread",
            computed_from={"sd": safe_null["sd"], "n_draws": safe_null["n_draws"]}),
        "n1_the_observed_beats_its_null": F.check(
            observed["safe_h_pi"] > safe_null["p95"],
            "this can fail, and if it fails it withdraws the only material headroom figure the "
            "project has. That is the point of running it",
            computed_from={"observed": observed["safe_h_pi"], "null_p95": safe_null["p95"],
                           "null_mean": safe_null["mean"], "p_value": safe_null["p_value"]}),
        "n4_the_fungible_null_is_degenerate_as_claimed": F.check(
            fung_var < 1e-12,
            "the diagnosis says the fungible null returns exactly 0 because calendar variance "
            "vanishes, so it cannot price selection bias. If there IS variance and H_PI is still "
            "exactly 0, that diagnosis is wrong and the fungible null did control something",
            computed_from={"mean_within_tape_variance": fung_var,
                           "fungible_safe_h_pi": fung_summary["safe_h_pi"]}),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: the calendar matrices were written by a sealed "
                                        "run and are read here. No seed is opened, no episode runs.",
                     "seeds_opened": 0, "episodes_run": 0, "matrix_root": str(MATRIX_ROOT)}}
    summary = F.summarise(checks)

    if not checks["n2_the_reproduction_is_exact"]["passed"]:
        status = "BLOCKED_CANNOT_REPRODUCE_THE_SEALED_FIGURE"
    elif not checks["n3_the_null_is_not_degenerate"]["passed"]:
        status = "BLOCKED_NULL_IS_DEGENERATE"
    elif checks["n1_the_observed_beats_its_null"]["passed"]:
        status = "H_PI_SURVIVES_ITS_JENSEN_NULL"
    else:
        status = "H_PI_IS_SELECTION_BIAS"

    payload = {
        "schema_version": "program_o_hpi_jensen_null_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "REANALYSIS_OF_SEALED_CALENDAR_MATRICES",
        "scope": "CAN_ONLY_WITHDRAW_PROGRAM_O_NEVER_PROMOTE_IT_NO_SEEDS_NO_EPISODES",
        "endpoint": "safe_h_pi__mean_of_per_tape_maxima_minus_best_single_calendar",
        "profile": PRIMARY_PROFILE, "n_tapes": int(panel["ret_visible"].shape[0]),
        "n_calendars": int(panel["ret_visible"].shape[1]),
        "observed": {"raw_h_pi": observed["raw_h_pi"], "safe_h_pi": observed["safe_h_pi"]},
        "reproduction": reproduction,
        "jensen_null_safe": safe_null, "jensen_null_raw": raw_null,
        "fungible_profile_check": {"mean_within_tape_variance": fung_var,
                                   "safe_h_pi": fung_summary["safe_h_pi"]},
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started, "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT, reference=SEALED)

    print(f"\nveredicto: {status}\n")
    print(f"  safe_h_pi observado   {observed['safe_h_pi']:+.6f}")
    print(f"  nulo de Jensen        media {safe_null['mean']:+.6f}  sd {safe_null['sd']:.6f}  "
          f"p95 {safe_null['p95']:+.6f}  p={safe_null['p_value']:.4f}")
    print(f"  raw_h_pi observado    {observed['raw_h_pi']:+.6f}   nulo p95 {raw_null['p95']:+.6f}")
    print(f"  perfil fungible       varianza intra-tapa {fung_var:.3e}  "
          f"safe_h_pi {fung_summary['safe_h_pi']:+.6f}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:48} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
