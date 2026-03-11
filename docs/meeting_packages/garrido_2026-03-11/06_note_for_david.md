**Note for David: why there is no cross-validation loss plot here**

The current method is PPO over a Gymnasium environment, not a supervised learner trained on fixed folds. Because of that, there is no standard cross-validation loss curve analogous to ANN/GNN validation loss. The evaluation equivalents we can show right now are:

- PPO training reward vs timesteps
- held-out evaluation against fixed baselines on the same seeds
- fill rate, backorder rate, and shift-mix diagnostics

So the correct sentence in the meeting is:

> For PPO, we do not evaluate with cross-validation loss; we evaluate with training reward diagnostics plus out-of-sample policy comparison on fixed seeds and operational metrics.
