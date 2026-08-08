# Kaggle slices, quarantined: the environment was never pinned

These fifteen extended-surface slices were produced on a third architecture — Linux 6.12.90 x86_64,
glibc 2.35, Python 3.12.13, recorded in `host_env_scresia-ext-surface-a.json` — by a kernel that ran
`pip install -q simpy numpy pandas scipy scikit-learn` **without version pins**. They are held here
rather than in `shards/` for that reason alone: a verification whose environment is unspecified
cannot be the authoritative one, whatever it reports.

**What they report is nonetheless worth keeping.** Fourteen of the fifteen show a maximum absolute
difference between 7e-18 and 6e-17 across roughly 4,500 of 4,608 cells each. That is last-bit
floating-point divergence, three to four orders of magnitude inside the `atol = rtol = 1e-12` band
frozen in `docs/ENMIENDA_TOLERANCIA_EQUIVALENCIA_CROSS_PLATFORM_2026-08-08.md` before any of these
numbers existed. Under that rule they are `CURRENT_HEAD_NUMERICALLY_EQUIVALENT_NOT_BIT_EXACT`, and
the rule earned its keep: it was written for exactly this and it fired on the first case it met.

That is a result about the reproducibility claim, and it narrows it honestly. The base surface
reproduced **bit-exactly** from macOS arm64 to Linux x86_64 glibc 2.43 / Python 3.14.4 over 103,680
cells. A third environment, differing in glibc and Python minor version and running unpinned
packages, reproduces the science and not the bits. So bit-exactness is a property of a *pinned*
environment pair, not of the code, and the manuscript should say so rather than generalise from the
pair that happened to agree.

**The fifteenth slice is different and is not covered by the above.** `ext__R1r_esc__8200005.json`
reports `max_abs_delta = 7662.0`, six orders of magnitude outside the tolerance band. It is not
adjudicated here. It shares its context with the one slice our own machines produced with a gross
difference (`ext__R1r_esc__8200011.json`, 39 cells, delta 10006.0), which is under separate
investigation, and until that returns neither is attributed to Kaggle or to us.

**Operational note.** These files were briefly copied into `shards/` before their mismatch counts
were read. That was the wrong order — merge after inspection, never before — and they were moved
here within minutes. Partitions 24-29 are being recomputed locally so the authoritative surface
carries no unpinned slice.
