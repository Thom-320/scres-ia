#!/usr/bin/env python3
"""Give an unsealed artifact a payload digest without editing it.

WHY A SIDECAR AND NOT A FIELD. `results/garrido_risk_headroom_sensitivity_v1/result.json` carries no
`self_sha256`: its bytes are pinned by a file digest and nothing detects an edit that preserved the
file size. Writing the digest INTO the file would fix the gap by doing the thing the gap makes
dangerous -- modifying a dated artifact -- and would also change the bytes every other record of it
refers to. So the seal goes beside it, and the artifact is not touched.

WHAT THE TWO DIGESTS ARE, BECAUSE THIS PROJECT HAS CONFUSED THEM BEFORE.

`file_sha256` is the SHA-256 of the bytes on disk. It answers "is this the file I was handed?".

`content_sha256` is the SHA-256 of the payload serialised canonically -- sorted keys, no indent,
UTF-8, integrity fields excluded -- so it answers "did the payload change?" and survives a
reformat that changes nothing. A design review of this repository once cited file digests under the
name `self_sha256` for sixteen artifacts: every value right, every label wrong. Hence both, each
under its own name, and neither called `self_sha256`, which in this codebase means specifically the
digest a sealing runner computed BEFORE inserting its own seal. This artifact never had one and a
sidecar cannot invent it retroactively.

WHAT THIS DOES NOT DO. It does not make the artifact confirmatory, does not re-run anything, and
does not claim the payload was correct when written -- only that from this moment it is pinned. A
seal applied after the fact certifies the future, not the past, and the sidecar says so.

Development tooling. Reads one JSON, writes one JSON beside it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

#: Fields a sealing runner would have excluded from its own payload digest.
INTEGRITY_FIELDS = ("self_sha256", "content_sha256", "file_sha256", "sealed_payload_sha256")


def canonical(payload: dict) -> bytes:
    body = {k: v for k, v in payload.items() if k not in INTEGRITY_FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <artifact>.seal.json beside it")
    args = ap.parse_args()

    raw = args.artifact.read_bytes()
    payload = json.loads(raw)
    already = [f for f in INTEGRITY_FIELDS if payload.get(f)]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    seal = {
        "schema_version": "retrospective_seal_v1",
        "artifact": str(args.artifact),
        "file_sha256": sha256(raw).hexdigest(),
        "content_sha256": sha256(canonical(payload)).hexdigest(),
        "canonicalisation": ("json.dumps(payload minus integrity fields, sort_keys=True, "
                             "separators=(',',':'), ensure_ascii=False), UTF-8"),
        "excluded_fields": list(INTEGRITY_FIELDS),
        "artifact_already_carried": already,
        "artifact_top_level_keys": sorted(payload),
        "artifact_verdict_field": ("status" if "status" in payload else
                                   "claim_status" if "claim_status" in payload else None),
        "artifact_verdict": payload.get("status") or payload.get("claim_status"),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "sealed_at_commit": commit,
        "what_this_certifies": ("that these bytes and this payload are what existed at the commit "
                                "named above. From this point an edit is detectable"),
        "what_this_does_not_certify": [
            "that the payload was correct when it was written -- a seal applied after the fact "
            "certifies the future, not the past",
            "that the artifact is confirmatory or has any grade it did not already declare",
            "the value a sealing runner would have written as self_sha256, which is computed "
            "before its own seal is inserted and cannot be reconstructed from outside",
        ],
        "verify_with": ("python3 -c \"import json,hashlib;p=json.load(open(ARTIFACT));"
                        "b={k:v for k,v in p.items() if k not in " + str(list(INTEGRITY_FIELDS)) +
                        "};print(hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':'),"
                        "ensure_ascii=False).encode()).hexdigest())\""),
    }
    out = args.out or args.artifact.with_suffix(".seal.json")
    out.write_text(json.dumps(seal, indent=1, sort_keys=True) + "\n")

    print(f"artefacto  : {args.artifact}")
    print(f"  veredicto: {seal['artifact_verdict']}  (campo `{seal['artifact_verdict_field']}`)")
    print(f"  ya traía : {already or 'ningún digest'}")
    print(f"  file     : {seal['file_sha256']}")
    print(f"  content  : {seal['content_sha256']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
