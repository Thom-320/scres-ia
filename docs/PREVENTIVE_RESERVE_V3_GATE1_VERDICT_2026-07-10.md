# Preventive Reserve v3 — Gate 1 verdict

**Verdict:** `STOP_NO_PREVENTIVE_HEADROOM`

**Artifact:** `outputs/preventive_reserve_v3/gate1_30x52/verdict.json`

**PPO trained:** no

v3 replaced the administrative 336 h replenishment lead with the physical
downstream path (Op10 24 h, Op11 availability, Op12 24 h). Thirty new paired
52-week tapes were evaluated.

## Results

- Reserve liveness: 30/30 perfect-warning tapes.
- Perfect warning vs static zero service-loss improvement: **+17.86%**, CI95
  **[+10.72%, +25.51%]**.
- Imperfect warning vs shuffled placebo: **+1.18%**, CI95
  **[−0.09%, +2.59%]**; gate threshold +5%.
- Perfect warning inventory-time: 166.77m ration-hours.
- Static 15k inventory-time: 125.51m ration-hours.
- Imperfect warning was Pareto non-dominated, but its placebo contrast failed.
- Perfect warning was Pareto non-dominated, but it exceeded the resource-match
  cap relative to static 15k.

## Interpretation

The physical intervention works: stock positioned behind the threatened arc is
issued and materially improves service relative to no reserve. The adaptive or
preventive value does not meet the pre-registered bar. A small permanent reserve
captures most of the deployable benefit, and an 80/80 imperfect warning does
not outperform an equal-count shuffled warning by a practically or
statistically sufficient margin.

This is the terminal result for the downstream-reserve + 14-day-warning class.
No capacity, target, warning-error, risk-frequency, lead, reward, or PPO tuning
is authorized. The result belongs in the consolidated boundary paper as a
strong distinction between **value of the physical reserve** and **value of
adaptive timing/information**.
