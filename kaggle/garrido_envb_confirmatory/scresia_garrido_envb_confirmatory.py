"""Kaggle kernel: Garrido Env-B dynamic-vs-static confirmatory run.

Runs only the three frozen candidates selected by the local Env-B screens:

* aggr_g24_raw_ppo: Excel-ReT route.
* aggr_g24_raw_recurrent: history/memory ablation on the same route.
* cons_control_v2_ppo: Pareto/resource-efficiency route.

Outputs one machine-readable decision file plus CSV/Markdown summaries.  The
kernel intentionally does not tune rewards or environment settings.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
from typing import Any


PAYLOAD = Path("/kaggle/input/scres-ia-payload/scres_ia_payload.tar.gz")
LOCAL_PAYLOAD = Path("scres_ia_payload.tar.gz")
KAGGLE_REPO_DIR = Path("/kaggle/working/scres-ia")
KAGGLE_OUTPUT_ROOT = Path("/kaggle/working/scresia_garrido_envb_confirmatory_outputs")
LOCAL_OUTPUT_ROOT = Path("outputs/kaggle/garrido_envb_confirmatory")

CONTROL_V2_SERVICE_HEAVY = {
    "--control-v2-w-fill": "1.2",
    "--control-v2-w-service": "6.0",
    "--control-v2-w-lost": "4.0",
    "--control-v2-w-inventory": "0.04",
    "--control-v2-w-shift": "0.06",
    "--control-v2-w-switch": "0.01",
}

FULL_PROFILE = {
    "seeds": "8501,8502,8503,8504,8505,8506,8507,8508,8509,8510",
    "eval_episodes": "5",
    "max_steps": "52",
    "train_timesteps": "65536",
}

DEBUG_PROFILE = {
    "seeds": "8501",
    "eval_episodes": "1",
    "max_steps": "4",
    "train_timesteps": "128",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def repo_root_from_context() -> Path:
    cwd = Path.cwd()
    if (cwd / "scripts" / "compare_garrido_dynamic_vs_static.py").exists():
        return cwd
    parent = Path(__file__).resolve().parents[2]
    if (parent / "scripts" / "compare_garrido_dynamic_vs_static.py").exists():
        return parent
    raise FileNotFoundError("Could not locate scres-ia repo root.")


def find_payload() -> Path | None:
    candidates = [PAYLOAD, LOCAL_PAYLOAD]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob("**/scres_ia_payload.tar.gz")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def describe_kaggle_input() -> str:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return "/kaggle/input does not exist"
    lines = []
    for path in sorted(input_root.rglob("*"))[:200]:
        kind = "dir" if path.is_dir() else "file"
        try:
            rel = path.relative_to(input_root)
        except ValueError:
            rel = path
        lines.append(f"{kind}: {rel}")
    if not lines:
        return "/kaggle/input is empty"
    return "\n".join(lines)


def extract_payload_or_use_local_repo() -> Path:
    payload = find_payload()
    if payload is not None:
        print(f"[kernel] Extracting payload from {payload}", flush=True)
        if KAGGLE_REPO_DIR.exists():
            shutil.rmtree(KAGGLE_REPO_DIR)
        KAGGLE_REPO_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(payload, "r:gz") as archive:
            archive.extractall(KAGGLE_REPO_DIR)
        return KAGGLE_REPO_DIR
    if is_kaggle():
        raise FileNotFoundError(
            "Missing scres_ia_payload.tar.gz in Kaggle input.\n"
            + describe_kaggle_input()
        )
    return repo_root_from_context()


def profile() -> dict[str, str]:
    value = (subprocess.os.environ.get("SCRESIA_PROFILE") or "confirmatory").strip()
    if value == "debug":
        return DEBUG_PROFILE
    if value != "confirmatory":
        raise ValueError("SCRESIA_PROFILE must be 'debug' or 'confirmatory'.")
    return FULL_PROFILE


def output_root() -> Path:
    return KAGGLE_OUTPUT_ROOT if is_kaggle() else LOCAL_OUTPUT_ROOT


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "label": "envb_aggr_g24_raw_ppo",
            "reward_mode": "ReT_garrido2024_raw",
            "algo": "ppo",
            "phi": "2.0",
            "psi": "1.5",
            "extra": {},
            "claim_path": "Partial Win A: Excel ReT",
        },
        {
            "label": "envb_aggr_g24_raw_recurrent",
            "reward_mode": "ReT_garrido2024_raw",
            "algo": "recurrent_ppo",
            "phi": "2.0",
            "psi": "1.5",
            "extra": {},
            "claim_path": "History ablation for Excel ReT",
        },
        {
            "label": "envb_cons_control_v2_ppo",
            "reward_mode": "control_v2",
            "algo": "ppo",
            "phi": "1.0",
            "psi": "1.25",
            "extra": CONTROL_V2_SERVICE_HEAVY,
            "claim_path": "Partial Win B: Pareto/resources",
        },
    ]


def compare_command(repo_dir: Path, out_root: Path, spec: dict[str, Any], prof: dict[str, str]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/compare_garrido_dynamic_vs_static.py",
        "--output-dir",
        str(out_root / "runs"),
        "--label",
        spec["label"],
        "--regimes",
        "severe",
        "--seeds",
        prof["seeds"],
        "--eval-episodes",
        prof["eval_episodes"],
        "--max-steps",
        prof["max_steps"],
        "--train-timesteps",
        prof["train_timesteps"],
        "--reward-mode",
        spec["reward_mode"],
        "--algo",
        spec["algo"],
        "--risk-frequency-multiplier",
        spec["phi"],
        "--risk-impact-multiplier",
        spec["psi"],
    ]
    for key, value in spec["extra"].items():
        cmd.extend([key, value])
    return cmd


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def ci95(values: list[float]) -> dict[str, float | int]:
    clean = [v for v in values if math.isfinite(v)]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sem": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    mean = statistics.mean(clean)
    sem = statistics.stdev(clean) / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "ci95_lo": mean - 1.96 * sem if n > 1 else float("nan"),
        "ci95_hi": mean + 1.96 * sem if n > 1 else float("nan"),
    }


def paired_deltas(episode_rows: list[dict[str, str]], static_policy: str, metric: str) -> list[float]:
    dynamic: dict[tuple[str, str], float] = {}
    static: dict[tuple[str, str], float] = {}
    for row in episode_rows:
        key = (row.get("regime", ""), row.get("eval_seed", ""))
        if row.get("policy") == "ppo_dynamic":
            dynamic[key] = f(row, metric)
        elif row.get("policy") == static_policy:
            static[key] = f(row, metric)
    out: list[float] = []
    for key, value in dynamic.items():
        if key in static:
            out.append(value - static[key])
    return out


def summarize_cell(out_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = out_root / "runs" / spec["label"]
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparisons = [
        row
        for row in summary.get("comparison_table", [])
        if row.get("dynamic_policy") == "ppo_dynamic"
    ]
    episode_rows = read_csv(run_dir / "episode_metrics.csv")

    target_rows: list[dict[str, Any]] = []
    for target in ("frozen_efficient", "static_S3_I1344", "original_S1_I0"):
        if target == "frozen_efficient":
            rows = [row for row in comparisons if bool(row.get("is_frozen_efficient_static"))]
            static_policy = rows[0]["static_policy"] if rows else ""
        else:
            rows = [row for row in comparisons if row.get("static_policy") == target]
            static_policy = target
        if not rows:
            continue
        excel_delta = ci95(paired_deltas(episode_rows, static_policy, "mean_ret_excel_formula"))
        cd_delta = ci95(paired_deltas(episode_rows, static_policy, "cd_sigmoid_mean"))
        resource_delta = ci95(paired_deltas(episode_rows, static_policy, "resource_composite_total"))
        comparison = rows[0]
        target_rows.append(
            {
                "label": spec["label"],
                "claim_path": spec["claim_path"],
                "algo": spec["algo"],
                "reward_mode": spec["reward_mode"],
                "phi": spec["phi"],
                "psi": spec["psi"],
                "target": target,
                "static_policy": static_policy,
                "strict_win": bool(comparison.get("strict_service_resource_dominates")),
                "pareto_win": bool(comparison.get("resource_pareto_dominates")),
                "excel_noninferior": bool(comparison.get("excel_noninferior")),
                "excel_delta_mean": excel_delta["mean"],
                "excel_delta_ci95_lo": excel_delta["ci95_lo"],
                "excel_delta_ci95_hi": excel_delta["ci95_hi"],
                "cd_delta_mean": cd_delta["mean"],
                "cd_delta_ci95_lo": cd_delta["ci95_lo"],
                "cd_delta_ci95_hi": cd_delta["ci95_hi"],
                "resource_delta_mean": resource_delta["mean"],
                "resource_delta_ci95_lo": resource_delta["ci95_lo"],
                "resource_delta_ci95_hi": resource_delta["ci95_hi"],
                "paired_n": excel_delta["n"],
                "summary_json": str(summary_path),
            }
        )

    static_rows = [row for row in comparisons if row.get("static_policy", "").startswith("static_") or row.get("static_policy") == "original_S1_I0"]
    excel_deltas_all = [float(row.get("delta_excel_ret", 0.0)) for row in static_rows]
    cd_deltas_all = [float(row.get("delta_cd_sigmoid_mean", 0.0)) for row in static_rows]
    resource_deltas_all = [float(row.get("delta_resource_composite_total", 0.0)) for row in static_rows]
    all_statics = {
        "label": spec["label"],
        "beats_all_statics_excel": bool(excel_deltas_all and min(excel_deltas_all) > 0.0),
        "beats_all_statics_cd": bool(cd_deltas_all and min(cd_deltas_all) > 0.0),
        "resource_lower_than_all_statics": bool(resource_deltas_all and max(resource_deltas_all) < 0.0),
        "min_delta_excel_all_statics": min(excel_deltas_all) if excel_deltas_all else float("nan"),
        "min_delta_cd_all_statics": min(cd_deltas_all) if cd_deltas_all else float("nan"),
        "max_delta_resource_all_statics": max(resource_deltas_all) if resource_deltas_all else float("nan"),
    }
    return {"targets": target_rows, "all_statics": all_statics}


def make_decision(rows: list[dict[str, Any]], all_statics: list[dict[str, Any]]) -> dict[str, Any]:
    by_label_target = {(row["label"], row["target"]): row for row in rows}
    labels = sorted({row["label"] for row in rows})
    decisions: dict[str, Any] = {"by_label": {}, "overall": {}}
    for label in labels:
        frozen = by_label_target.get((label, "frozen_efficient"), {})
        s3 = by_label_target.get((label, "static_S3_I1344"), {})
        all_row = next((row for row in all_statics if row["label"] == label), {})
        partial_a = bool(all_row.get("beats_all_statics_excel", False))
        partial_b = bool(
            frozen.get("excel_delta_mean", float("-inf")) >= -0.005
            and frozen.get("resource_delta_mean", float("inf")) < 0.0
            and frozen.get("cd_delta_mean", float("-inf")) >= 0.0
        )
        complete = bool(partial_a and partial_b and all_row.get("beats_all_statics_cd", False))
        decisions["by_label"][label] = {
            "complete_win": complete,
            "partial_win_a_excel_all_statics": partial_a,
            "partial_win_b_pareto_vs_frozen_efficient": partial_b,
            "s3_pareto_win": bool(s3.get("pareto_win", False)),
            "frozen_efficient_excel_delta_mean": frozen.get("excel_delta_mean"),
            "frozen_efficient_cd_delta_mean": frozen.get("cd_delta_mean"),
            "frozen_efficient_resource_delta_mean": frozen.get("resource_delta_mean"),
            "min_delta_excel_all_statics": all_row.get("min_delta_excel_all_statics"),
        }
    decisions["overall"] = {
        "complete_win_labels": [k for k, v in decisions["by_label"].items() if v["complete_win"]],
        "partial_win_a_labels": [
            k for k, v in decisions["by_label"].items() if v["partial_win_a_excel_all_statics"]
        ],
        "partial_win_b_labels": [
            k for k, v in decisions["by_label"].items() if v["partial_win_b_pareto_vs_frozen_efficient"]
        ],
        "s3_pareto_labels": [k for k, v in decisions["by_label"].items() if v["s3_pareto_win"]],
    }
    return decisions


def write_report(out_root: Path, rows: list[dict[str, Any]], all_statics: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    lines = [
        "# Garrido Env-B Confirmatory Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(decision["overall"], indent=2),
        "```",
        "",
        "## Target Summary",
        "",
        "| label | target | Excel Δ | Excel CI95 | CD Δ | Resource Δ | strict | pareto |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {target} | {excel_delta_mean:+.6f} | [{excel_delta_ci95_lo:+.6f}, {excel_delta_ci95_hi:+.6f}] | {cd_delta_mean:+.6f} | {resource_delta_mean:+.0f} | {strict_win} | {pareto_win} |".format(
                **row
            )
        )
    lines.extend(["", "## All-Static Checks", ""])
    lines.extend([
        "| label | beats all Excel | beats all CD | resource lower than all | min Excel Δ | min CD Δ | max resource Δ |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    for row in all_statics:
        lines.append(
            "| {label} | {beats_all_statics_excel} | {beats_all_statics_cd} | {resource_lower_than_all_statics} | {min_delta_excel_all_statics:+.6f} | {min_delta_cd_all_statics:+.6f} | {max_delta_resource_all_statics:+.0f} |".format(
                **row
            )
        )
    (out_root / "confirmatory_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = datetime.now(timezone.utc)
    prof = profile()
    out_root = output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    repo_dir = extract_payload_or_use_local_repo()

    print("SCRESIA Garrido Env-B confirmatory", started.isoformat(), flush=True)
    print("profile", prof, flush=True)
    print("repo_dir", repo_dir, flush=True)
    print("output_root", out_root, flush=True)
    try:
        import torch

        print(
            "torch_cuda",
            {
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            },
            flush=True,
        )
    except Exception as exc:
        print("torch_cuda_probe_failed", repr(exc), flush=True)

    if is_kaggle():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo_dir)

    run_summaries: list[dict[str, Any]] = []
    for spec in candidate_specs():
        print(f"\n=== {spec['label']} ===", flush=True)
        run(compare_command(repo_dir, out_root, spec, prof), cwd=repo_dir)
        run_summaries.append(spec)

    target_rows: list[dict[str, Any]] = []
    all_static_rows: list[dict[str, Any]] = []
    for spec in candidate_specs():
        summary = summarize_cell(out_root, spec)
        target_rows.extend(summary["targets"])
        all_static_rows.append(summary["all_statics"])

    write_csv(out_root / "confirmatory_summary.csv", target_rows)
    write_csv(out_root / "confirmatory_all_statics.csv", all_static_rows)
    decision = make_decision(target_rows, all_static_rows)
    payload = {
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": prof,
        "candidates": run_summaries,
        "decision": decision,
        "artifacts": {
            "confirmatory_summary_csv": str(out_root / "confirmatory_summary.csv"),
            "confirmatory_all_statics_csv": str(out_root / "confirmatory_all_statics.csv"),
            "confirmatory_report_md": str(out_root / "confirmatory_report.md"),
        },
    }
    (out_root / "confirmatory_decision.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_report(out_root, target_rows, all_static_rows, decision)

    print("\n=== CONFIRMATORY DECISION ===", flush=True)
    print(json.dumps(decision["overall"], indent=2), flush=True)
    print("outputs:", out_root, flush=True)


if __name__ == "__main__":
    main()
