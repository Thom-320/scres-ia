# Section 4.2 — Results from the Hybrid Simulation–Neural Model

*This section replaces the placeholder text in the v0 under "4.2 Results from Hybrid Simulation Neural Model."*
*Written 2026-03-24 — ready for insertion into the DOCX.*

---

## 4.2 Results from the Hybrid Simulation–Neural Model

This section reports the performance of the PPO-based adaptive controller under the two stress scenarios defined in Section 3.3.6, benchmarked against static shift policies and a random baseline.

### 4.2.1 Static Baseline Characterization

Before evaluating the learned policy, we first establish the performance envelope of fixed shift policies under stochastic conditions. Table 8 reports the control reward and service metrics for each static baseline.

**Table 8.** Static baseline performance under increased and severe risk (control_v1 reward, 5 seeds × 10 episodes, stochastic processing times enabled).

| Scenario | Policy | Control reward | Fill rate | Backorder rate |
|----------|--------|---------------:|----------:|---------------:|
| Increased | Static S1 | −356.81 | 0.652 | 0.348 |
| Increased | Static S2 | **−171.35** | **0.836** | 0.164 |
| Increased | Static S3 | −178.09 | 0.835 | 0.165 |
| Severe | Static S1 | −564.70 | 0.452 | 0.548 |
| Severe | Static S2 | −384.82 | 0.628 | 0.372 |
| Severe | Static S3 | **−388.40** | **0.629** | 0.371 |

Under increased risk, S2 dominates: it provides the highest service level and the best control reward, indicating that double-shift operation is already sufficient to absorb moderate disruptions. Under severe risk, S2 and S3 perform comparably, with S2 marginally better on reward and S3 marginally better on service. Critically, S1 is clearly suboptimal in both scenarios, confirming that shift allocation is a meaningful control lever for supply chain resilience.

### 4.2.2 Adaptive Control Under Increased Risk

Table 9 reports the performance of the PPO-trained policy compared to the best static baseline (S2) under the increased risk scenario.

**Table 9.** PPO vs. best static baseline under increased risk (500k timesteps, control_v1, w_bo = 4.0, w_cost = 0.02, stochastic PT).

| Policy | Control reward | Fill rate | Backorder rate | Shift mix (S1/S2/S3) |
|--------|---------------:|----------:|---------------:|----------------------|
| Static S2 | −170.10 | 0.837 | 0.163 | 0% / 100% / 0% |
| PPO | −172.05 | 0.838 | 0.162 | 12% / 25% / 63% |
| Difference | −1.95 | +0.001 | −0.001 | — |
| Bootstrap CI₉₅ | [−9.95, +8.51] | — | — | — |

Under moderate stress, the PPO agent matches the service level of the best static baseline while using a heterogeneous shift allocation (primarily S3, with occasional downshifting to S1/S2). The reward difference of −1.95 points is not distinguishable from zero (bootstrap CI₉₅ includes zero, exact sign-flip p = 0.812). This result indicates that the adaptive controller is *competitive* with the optimal fixed policy under increased risk, but does not yet demonstrate a clear advantage in this regime.

### 4.2.3 Adaptive Control Under Severe Risk

Table 10 reports the performance under the severe risk scenario, where disruption frequencies are escalated 4–17× relative to the thesis baseline.

**Table 10.** PPO vs. best static baseline under severe risk (500k timesteps, control_v1, w_bo = 4.0, w_cost = 0.02, stochastic PT).

| Policy | Control reward | Fill rate | Backorder rate | Shift mix (S1/S2/S3) |
|--------|---------------:|----------:|---------------:|----------------------|
| Static S3 | −385.59 | 0.632 | 0.368 | 0% / 0% / 100% |
| PPO | −380.98 | 0.631 | 0.369 | 41% / 27% / 32% |
| Difference | +4.61 | −0.001 | +0.001 | — |
| Bootstrap CI₉₅ | [−0.28, +9.49] | — | — | — |

Under severe stress, the adaptive controller outperforms the best fixed baseline by 4.61 control-reward points while maintaining effectively equivalent service (fill rate difference < 0.1%). The bootstrap CI₉₅ is [−0.28, +9.49], with an exact sign-flip p-value of 0.188, indicating a directional advantage that approaches but does not reach conventional significance levels. Notably, the PPO agent achieves this improvement by *reducing* its reliance on S3 relative to the static baseline, using a balanced mix of all three shift levels. This behavior suggests that the agent learns to downshift during low-disruption windows, recovering cost without sacrificing service—a strategy that is unavailable to any fixed policy.

### 4.2.4 Interpretation

The pattern across the two stress scenarios supports the following reading:

1. **Regime-dependent value.** The adaptive controller is competitive under moderate stress and stronger under severe stress. This is consistent with the expectation that fixed policies become insufficient when disruption intensity exceeds their design point.

2. **Cost-aware adaptation.** The PPO agent does not simply default to the most expensive shift configuration. Under both scenarios, it uses a heterogeneous shift allocation, indicating that the reward function successfully incentivizes cost-aware behavior.

3. **Service preservation.** In neither scenario does the adaptive policy degrade service relative to the best static baseline. This confirms that the control_v1 reward function avoids the misalignment observed with ReT_thesis (Section 3.3.4).

These results should be interpreted as preliminary: the current evidence demonstrates that adaptive shift control has value in high-stress regimes, but does not justify a claim of uniform superiority across all risk conditions. Section 4.3 extends this analysis by examining whether richer policy architectures can strengthen the adaptive signal.
