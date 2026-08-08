#!/usr/bin/env python3
"""Generate the citable-artifact table for a claim-freeze amendment. Two hashes, both labelled.

WHY TWO. `seal_and_write` stores `self_sha256`, the digest of the payload serialised under the
sealing convention. That is NOT the digest of the file on disk: the seal is computed before the
seal itself is inserted, so the bytes you would download hash to something else. Both are correct
and they answer different questions --

    self_sha256   : did the sealed payload change?
    file_sha256   : is this the file I was given?

-- and a citable table that prints one under the other's name is a table a reader will conclude is
wrong. A design pass on this repository quoted file digests as `self_sha256` for sixteen artifacts;
every value was right and every label was wrong. Hence both columns, named.

WHY GENERATED. The claim freeze's own rule is that a number without a row does not circulate. A
hand-written row is how the drift count in the document written to stop drift ended up 37 commits
stale. Paths and prose are the analyst's; hashes, grades and statuses are read from disk.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only over artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
REGISTRY = Path("research/evidence_registry.jsonl")

#: Paper 1 depends on these and the freeze does not list them. Order is the paper's section order.
PAPER1 = [
    ("§3 el abandono", "results/sensitivity/contention_headroom_v1_2/result.json"),
    ("§3 replicación", "results/metric_audit/abandonment_v1/result.json"),
    ("§4 el mecanismo refutado", "results/sensitivity/contention_headroom_v1_3/result.json"),
    ("§5 TTR sin estimando", "results/manuscript/h1_h3_v1/result.json"),
    ("§6 la reparación", "results/manuscript/h1_h3_originales_v3/result.json"),
    ("§7 la fuga", "results/twin_surface_v2/result.json"),
    ("§8 no invariante", "results/monotone_transform_family_v4/result.json"),
    ("§9 la cadencia", "results/metric_audit/ret_cadence_corrective_v2/result.json"),
    ("§9 el defecto", "results/metric_audit/ret_defects_v1/result.json"),
    ("§9 las reparaciones", "results/metric_audit/ret_repair_variants_v1/result.json"),
    ("§10 diagnóstico", "results/determinism_diagnostic/result.json"),
    ("§10 reparación", "results/determinism_repair_control/result.json"),
    ("§12 lo que sobrevive", "results/endpoint_headroom_atlas/result.json"),
    ("§2 la demanda", "results/demand_process/result.json"),
    ("§5 contexto", "results/surface_gates_v2/result.json"),
    ("§12 curva de aprendizaje", "results/manuscript/h2_learning_curve/result.json"),
]


def grades() -> dict[str, dict]:
    out = {}
    if REGISTRY.exists():
        for line in REGISTRY.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["artifact_path"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_2.md"))
    ap.add_argument("--receipt", type=Path,
                    default=Path("results/claim_freeze_amendment_2/result.json"))
    args = ap.parse_args()
    reg = grades()

    rows, missing = [], []
    for section, path in PAPER1:
        f = Path(path)
        if not f.exists():
            missing.append(path)
            continue
        raw = f.read_bytes()
        d = json.loads(raw)
        rows.append({
            "section": section, "path": path,
            "claim_status": d.get("claim_status"),
            "self_sha256": d.get("self_sha256"),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "contract_path": d.get("contract_path") or d.get("preregistration"),
            "seed_block": (f"{d['seed_block']['start']}-{d['seed_block']['end']}"
                           if isinstance(d.get("seed_block"), dict)
                           and "start" in d["seed_block"] else None),
            "evidence_grade": reg.get(path, {}).get("evidence_grade"),
            "falsifiers_all_passed": (d.get("falsifiers") or {}).get("all_passed"),
        })

    L = ["# Enmienda 2 a la congelación de claims — los artefactos que el Paper 1 necesita citar\n"]
    L.append(f"Generada `{datetime.now(timezone.utc).isoformat(timespec='seconds')}` por "
             "`scripts/build_claim_freeze_amendment_v1.py`. **No editar la tabla a mano.**\n")
    L.append("La regla del congelamiento dice que una cifra sin fila no circula. Estos "
             f"{len(rows)} artefactos sostienen el Paper 1 y no estaban listados.\n")
    L.append("## Los dos hashes, y por qué van los dos\n")
    L.append("`self_sha256` es el digest del *payload sellado*, calculado **antes** de insertar el "
             "propio sello; `file_sha256` es el digest de los **bytes en disco**. Los dos son "
             "correctos y contestan preguntas distintas —¿cambió el payload? ¿es éste el fichero "
             "que me dieron?— y **no coinciden nunca**. Una revisión de diseño de este repositorio "
             "citó los digests de fichero bajo el nombre `self_sha256` para dieciséis artefactos: "
             "todos los valores correctos, todas las etiquetas equivocadas. Por eso van los dos, "
             "con su nombre.\n")
    L.append("| § | artefacto | `claim_status` | grado | falsadores | `self_sha256` | "
             "`file_sha256` | bloque |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        fp = {True: "✅", False: "⚠️ no todos", None: "—"}[r["falsifiers_all_passed"]]
        L.append(f"| {r['section']} | `{r['path']}` | `{r['claim_status']}` | "
                 f"{r['evidence_grade'] or '—'} | {fp} | `{str(r['self_sha256'])[:16]}…` | "
                 f"`{r['file_sha256'][:16]}…` | {r['seed_block'] or '—'} |")
    L.append("")
    warn = [r for r in rows if r["falsifiers_all_passed"] is False]
    if warn:
        L.append("## Artefactos con falsadores en rojo — se citan CON su fallo, o no se citan\n")
        for r in warn:
            L.append(f"* `{r['path']}` — `{r['claim_status']}`. El fallo es parte del resultado "
                     "y va en la misma frase que el número.")
        L.append("")
    if missing:
        L.append("## Ausentes\n")
        for m in missing:
            L.append(f"* `{m}` — no está en el árbol; el Paper 1 no puede citarlo.")
        L.append("")
    L.append("## Lo que esta enmienda NO hace\n")
    L.append("No adjudica nada, no cambia ningún número y no autoriza ninguna afirmación nueva. "
             "Sólo admite estos artefactos a la tabla de citables. Las prohibiciones del §7 del "
             "congelamiento y de la Enmienda 1 siguen vigentes íntegras.\n")
    args.out.write_text("\n".join(L))

    payload = {
        "schema_version": "claim_freeze_amendment_v1",
        "claim_status": "ARTIFACTS_ADMITTED_TO_THE_CITABLE_TABLE",
        "scope": "INDEX_ONLY_NO_SCIENTIFIC_CLAIM_NO_SEEDS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": "none_index_only",
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "generated_document": str(args.out),
        "n_admitted": len(rows), "n_missing": len(missing), "missing": missing,
        "rows": rows,
        "two_hashes_note": ("self_sha256 is the sealed payload digest computed before the seal is "
                            "inserted; file_sha256 is the digest of the bytes on disk. They never "
                            "coincide, and printing one under the other's name makes a citable "
                            "table look wrong."),
    }
    digest = seal_and_write(payload, args.receipt, contract=args.contract,
                            reference=Path("results/evidence_registry/result.json"))
    print(f"  {len(rows)} admitidos · {len(missing)} ausentes · "
          f"{len(warn)} con falsadores en rojo")
    for r in warn:
        print(f"    ⚠️  {r['path']}")
    print(f"  -> {args.out}  ·  recibo {args.receipt} (sello {digest[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
