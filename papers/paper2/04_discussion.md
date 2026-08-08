# 4. Discussion

## 4.1 What transfers

The evidence supports a narrow conclusion: development point estimates favour retaining search
state as a way to avoid rediscovery, but transfer is not an intrinsic property of “learning” or of
neural architecture. The prespecified factorized-UCB arm meets the prospective transfer criterion
against cold start and against an online replay of its own visit marginals. Secondary carrier contrasts show no corresponding
advantage for the evaluated neural arm.

This makes search strategy the correct level of explanation. The result does not show that a real
supply chain learns, that a neural controller adapts within an episode, or that UCB is universally
superior. It shows what crossed the declared design-space boundary under this contract.

## 4.2 Two loops, one question, one answer

The question this literature asks — which family of algorithms best reproduces supply-chain learning
— can be put to either of two loops, and the answer differs by loop. In the **outer** loop, state is
carried across runs of the search over configurations; that is the closure between data gathering
and validation in the source framework, and it is where every positive result in this paper lives.
In the **inner** loop, a decision within an episode is conditioned on observed state against a fixed
policy; that is where a controller would live, and it is empty.

We separate them because the report is asymmetric and the asymmetry is easy to lose. A preregistered
replication of the within-episode arm on 48 previously unopened seeds does not reproduce its own
ceiling: the clairvoyant per-tape gap is +0.024054 against an interaction null whose mean is
+0.026641 (p = 0.7482), where twelve reused tapes had given +0.045103 at p = 0.0132. The observed
gap is smaller than the average of its own null — the downward bias of a minimum taken over 27 noisy
options. By the frozen reading rule of that authorisation, the feature-level tests beneath the
ceiling are not read; they are retained and declared unread.

The consequence for the framing is direct. A result about carrying search state across runs is not
evidence that a controller adapts within an episode, and the surviving half must not be allowed to
stand in for the half that did not survive. Both are reported here under the same endpoint, and the
claim register records which loop each claim answers.

## 4.3 Architecture is secondary to the estimand

The Figure 5 result is an identity check, not evidence of learning. The matched KAN/MLP result is a
search comparison, not a universal statement about KANs. The latent PPO result belongs to a different
contract and is therefore supplementary. Treating these as one architecture story would conflate
fit, sequential search efficiency and within-episode outcome.

## 4.4 Implications for SCRES modelling

The formal state variable (L_{k-1}) is useful for modelling path dependence in the optimization
procedure. It does not by itself establish the v0 claim that resilience performance is a
learning-dependent property of the physical supply chain. That stronger interpretation requires a
different contract with repeated operational decisions, an explicit recovery estimand, and a
prospectively adjudicated learner.

## 4.5 Limitations

The principal results inherit the demand and risk process of the confirmation contract. The seasonal
extension is development evidence only. Context-level retention effects cannot be recovered from the
stored averaged AUC arrays. The KAN fit and search contracts differ. The physical panels establish
targeted correspondence, not complete source-model reproduction. These limitations define the
generalization boundary rather than disappear through additional wording.
