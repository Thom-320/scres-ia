# Program F implementation freeze

Status: **FROZEN BEFORE ANY PROGRAM F TAPE**.

This document closes the event-generation details intentionally left abstract
in the Program F charter. Dominant contexts use the thesis Table 6.12 increased
occurrence windows; background contexts use current windows. R11 equipment
pressure uses the already disclosed severe-extended five-hour recovery anchor;
other R11 events retain the current two-hour mean. R22, R23 and R24 magnitudes
use thesis current-level anchors.

Counts are generated independently by risk and week from Poisson rates equal to
`168 / b`, where `b` is the frozen uniform occurrence window. Within-week onset,
target operation and base damage are event keyed. This is a computational
context extension rather than a claim that the thesis used a Poisson process.

The threat hash contains latent contexts, signals and unmitigated events. It
never contains realized duration, reserve issue or policy state. A CRN audit
must show the threat hash is identical across all six actions.

The initial active posture is `(1,1,0)` for the unavoidable first activation
week. Every requested action takes effect exactly 168 hours later.
