# Track B Q1 stats bundle - 10-seed extension (2026-07-02)

This bundle combines canonical seeds 1-5 with the VPS seed expansion 6-10.
It recomputes the primary paired comparison against the canonical best dense
static comparator `S2_op10_2.00_op12_1.50` under matching CRN keys.

- Pairs: 120 (10 seeds x 12 episodes)
- Seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- Order-level ReT delta: +0.000426168, CI95 [+0.000398484, +0.000454626]
- Excel ReT delta: +0.000438164, CI95 [+0.000409361, +0.000467629]
- Seed-clustered mean delta: +0.000438164, CI95 [+0.000420699, +0.000458220]
- All seed mean deltas positive: True

Top-12 static robustness is not recomputed here; the paper appendix remains the
5-seed top-12 check unless top-12 statics are evaluated on seeds 6-10.
