# Retained Search State Before Neural Architecture

## Prospective Transfer in Supply-Chain Resilience Simulation Optimization

### Abstract framing

Simulation-based supply-chain resilience studies often evaluate a configuration one disruption
scenario at a time. That design resets the physical system correctly, but it can also reset the
search procedure: information acquired in one discrete-event simulation run is discarded before the
next configuration is selected. We study the outer-loop alternative in which only search state is
retained across runs. The physical DES, random realization and within-run state remain reset.

Within the evaluated contract, development point estimates favour state retention in all six matched
families relative to their memoryless twins. In a prospective expansion from 288 to 4,608
configurations, the prespecified factorized UCB arm meets the confirmatory transfer criterion
against both cold start and a state-blind replay of its own visit marginals. Secondary carrier
contrasts show no corresponding advantage for the evaluated neural, GP-EI or OFAT arms.
The result separates retained search state from neural architecture and from learning inside the
physical chain.

### Research questions

1. Does retaining search state across DES runs reduce development regret relative to a matched
   memoryless search procedure?
2. Does retained state transfer when the design space expands from 288 to 4,608 configurations,
   beyond cold start and state-blind marginal replay?
3. Is any transfer advantage specific to the neural carrier, or does it arise from a simpler search
   statistic?

### Contribution and boundary

The contribution is a reproducible outer-loop protocol, not a claim of complete source-model
validity or that the physical supply chain learns. We distinguish trajectory-dependent search from within-
episode control, and prospective transfer from development ranking. Architecture comparisons are
reported only under their own contracts. The demand process is inherited unless the bounded seasonal
sensitivity is explicitly identified as development evidence.

The central claim is:

> Within the evaluated outer-loop contract and the inherited demand process, development point
> estimates favoured state retention in all six matched families. In a prospective expansion from
> 288 to 4,608 configurations, the prespecified factorized UCB search strategy met the
> confirmatory transfer criterion against both cold start and a state-blind replay of its own
> search marginals; secondary carrier contrasts showed no corresponding neural-specific advantage.
