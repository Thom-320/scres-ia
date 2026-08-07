#!/usr/bin/env python3
"""Generate the canonical claims table FROM the evidence registry, instead of writing it by hand.

WHY. Six external audits in a row found stale figures in hand-maintained tables: a drift count that
was 786 in the document written to stop drift and 820 by the time it was read, a lower bound quoted
from a superseded ladder, a retired normaliser figure still sitting in a hypothesis row. None of
those are carelessness in the ordinary sense -- they are what happens when a derived fact is stored
by hand.

WHAT IS GENERATED AND WHAT IS NOT. Everything countable comes from the registry and the custody
file: how many experiments exist, how they grade, which are confirmatory, how much custody is left,
how far the branch has drifted. Interpretation is NOT generated -- the prohibited-claims list and
the reading of each result stay in the hand-written amendments, and this file links to them rather
than paraphrasing them. A generator that also wrote the interpretation would just be a slower way
of going stale.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only. No seeds, no simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
REGISTRY = Path("research/evidence_registry.jsonl")
CUSTODY = Path("research/seed_custody_registry.json")
# Interpretation lives here and is linked, never paraphrased.
INTERPRETATION = (
    ("qué se puede afirmar y qué está prohibido",
     "docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md"),
    ("enmienda 1: H2, la cota de ofat, la cifra prohibida de H4",
     "docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md"),
    ("qué fichero manda por familia", "docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md"),
    ("huecos abiertos", "docs/REGISTRO_DE_HUECOS_2026-08-07.md"),
)


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/TABLA_CANONICA_GENERADA.md"))
    ap.add_argument("--receipt", type=Path,
                    default=Path("results/canonical_table_generated/result.json"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]
    survivors = [r for r in rows if r["duplicate_of"] is None]
    grades = Counter(r["evidence_grade"] for r in survivors)
    confirmatory = sorted((r for r in survivors if r["evidence_grade"] == "CONFIRMATORY"),
                          key=lambda r: r["artifact_path"])
    incomplete = sum(1 for r in rows if not r["dedup_key_complete"])
    missing = Counter()
    for r in rows:
        for k, v in r["dedup_key"].items():
            if v is None:
                missing[k] += 1

    custody = json.loads(CUSTODY.read_text())
    blocks = custody["blocks"]
    virgin = [b for b in blocks if b["status"] == "RESERVED_NOT_OPENED"]
    burned_conf = [b for b in blocks if "CONFIRMATION_COMPLETE" in b.get("status", "")]

    head = git("rev-parse", "--short", "HEAD")
    ahead = git("rev-list", "--count", "main..HEAD") or "?"
    behind = git("rev-list", "--count", "HEAD..main") or "?"
    main_sha = git("rev-parse", "--short", "main")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    L: list[str] = []
    L.append("# Tabla canónica — **generada**, no escrita a mano\n")
    L.append(f"Generada `{now}` desde `{REGISTRY}` y `{CUSTODY}` por "
             f"`scripts/generate_canonical_table_v1.py`.\n")
    L.append("> **No editar a mano.** Seis auditorías seguidas encontraron cifras rancias en "
             "tablas mantenidas a mano. Lo contable se genera; **la interpretación no**, y vive en "
             "los documentos enlazados abajo.\n")

    L.append("## Procedencia\n")
    L.append(f"| rama científica | `{head}` |\n|---|---|")
    L.append(f"| `main` | `{main_sha}` |")
    L.append(f"| adelante / detrás | **{ahead}** / {behind} |\n")

    L.append("## Evidencia, por grado derivado\n")
    L.append(f"{len(rows)} artefactos leídos · {len(rows) - len(survivors)} colapsados como "
             f"re-reporte · **{len(survivors)} experimentos distintos**.\n")
    L.append("| grado | n |\n|---|---:|")
    for g, n in grades.most_common():
        L.append(f"| `{g}` | {n} |")
    L.append("")
    L.append("El grado **se deriva** de hechos comprobables (rol de confirmación, bloque de "
             "custodia, presencia de contrato), nunca del `claim_status` que escribió un autor.\n")

    L.append("## Confirmatorios\n")
    if confirmatory:
        L.append("| artefacto | autorado como | bloque |\n|---|---|---|")
        for r in confirmatory:
            L.append(f"| `{r['artifact_path']}` | `{r['claim_status_as_authored']}` | "
                     f"`{r['dedup_key']['seed_block']}` |")
    else:
        L.append("_ninguno_")
    L.append("")

    L.append("## Custodia\n")
    L.append(f"Estado del registro: `{custody['status']}` · "
             f"`new_seed_opening: {custody['new_seed_opening']}`\n")
    L.append(f"**Bloques vírgenes restantes: {len(virgin)}**"
             + (f" — {', '.join(b['id'] for b in virgin)}" if virgin else
                " — no queda ninguno; ninguna confirmación más es posible sin autorizar semillas "
                "nuevas.") + "\n")
    L.append(f"Quemados por confirmación completa: {len(burned_conf)} "
             f"({', '.join(b['id'] for b in burned_conf)}).\n")

    L.append("## Lo que el corpus todavía no permite\n")
    L.append(f"**{incomplete} de {len(rows)} claves de experimento están incompletas "
             f"({100*incomplete/max(1,len(rows)):.0f} %)**, así que el corpus no se puede "
             f"deduplicar del todo. Campo ausente:\n")
    L.append("| campo | artefactos sin él |\n|---|---:|")
    for k, n in missing.most_common():
        L.append(f"| `{k}` | {n} |")
    L.append("\n`seal_and_write` ya deriva `seed_block` y `endpoint` cuando el payload los "
             "contiene, así que esta cifra debe **bajar** con cada corrida nueva. Si sube, algo "
             "dejó de sellar.\n")

    L.append("## La interpretación, que no se genera\n")
    for label, path in INTERPRETATION:
        L.append(f"* [{label}]({Path(path).name}) — `{path}`")
    L.append("")

    args.out.write_text("\n".join(L))

    payload = {
        "schema_version": "canonical_table_generated_v1",
        "claim_status": "CANONICAL_TABLE_GENERATED",
        "scope": "INDEX_ONLY_NO_SCIENTIFIC_CLAIM_NO_SEEDS",
        "created_at": now, "endpoint": "none_index_only",
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "generated_document": str(args.out),
        "sources": {"registry": str(REGISTRY), "custody": str(CUSTODY)},
        "counts": {"artifacts": len(rows), "distinct_experiments": len(survivors),
                   "by_grade": dict(grades), "incomplete_keys": incomplete,
                   "missing_fields": dict(missing),
                   "virgin_blocks": len(virgin), "confirmatory": len(confirmatory)},
        "what_is_not_generated": ("interpretation: the prohibited-claims list and the reading of "
                                  "each result stay hand-written and are linked, not paraphrased"),
    }
    # The registry itself is JSONL; seal_and_write reads its reference as a single JSON object,
    # so the reference is the registry's own sealed receipt rather than the line-delimited file.
    digest = seal_and_write(payload, args.receipt, contract=args.contract,
                            reference=Path("results/evidence_registry/result.json"))
    print(f"  {len(rows)} artefactos · {len(survivors)} experimentos · "
          f"{len(confirmatory)} confirmatorios · {len(virgin)} bloques vírgenes")
    print(f"  claves incompletas {incomplete}/{len(rows)} ({100*incomplete/max(1,len(rows)):.0f}%)")
    print(f"  -> {args.out}  ·  recibo {args.receipt} (sello {digest[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
