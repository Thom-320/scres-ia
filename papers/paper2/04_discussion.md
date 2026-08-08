# 4. Discussion

## 4.1 What transfers

The evidence supports a narrow conclusion: development point estimates favour retaining search
state as a way to avoid rediscovery, but transfer is not an intrinsic property of “learning” or of
neural architecture. The prespecified factorized-UCB arm meets the prospective transfer criterion
against cold start and state-blind marginal replay. Secondary carrier contrasts show no corresponding
advantage for the evaluated neural arm.

This makes search strategy the correct level of explanation. The result does not show that a real
supply chain learns, that a neural controller adapts within an episode, or that UCB is universally
superior. It shows what crossed the declared design-space boundary under this contract.

## 4.2 Architecture is secondary to the estimand

The Figure 5 result is an identity check, not evidence of learning. The matched KAN/MLP result is a
search comparison, not a universal statement about KANs. The latent PPO result belongs to a different
contract and is therefore supplementary. Treating these as one architecture story would conflate
fit, sequential search efficiency and within-episode outcome.

## 4.3 Implications for SCRES modelling

The formal state variable (L_{k-1}) is useful for modelling path dependence in the optimization
procedure. It does not by itself establish the v0 claim that resilience performance is a
learning-dependent property of the physical supply chain. That stronger interpretation requires a
different contract with repeated operational decisions, an explicit recovery estimand, and a
prospectively adjudicated learner.

## 4.4 Limitations

The principal results inherit the demand and risk process of the confirmation contract. The seasonal
extension is development evidence only. Context-level retention effects cannot be recovered from the
stored averaged AUC arrays. The KAN fit and search contracts differ. The physical panels establish
targeted correspondence, not complete source-model reproduction. These limitations define the
generalization boundary rather than disappear through additional wording.
