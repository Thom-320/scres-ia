# Function Reference: `MFSCSimulation`

Core methods of the SimPy simulation mapping to the thesis.

## `__init__(self, env, params)`
- **Purpose:** Initializes buffers, processes, and RNG.
- **Thesis Ref:** Section 6.3 Architecture.
- **Config:** Uses `SIM_PARAMS` from `config.py`.

## `_op1_contracts(self)`
- **Purpose:** Generates raw material contracts every 672h.
- **Thesis Ref:** Section 6.3.1, Op1.

## `_op5_to_op7_assembly(self)`
- **Purpose:** Continuous assembly line processing at rate λ (320.5 rations/h).
- **Thesis Ref:** Section 6.3.3, Op5-7.

## `_op13_demand(self)`
- **Purpose:** Generates daily demand U(2400,2600) and triggers system pull.
- **Thesis Ref:** Section 6.4, Table 6.4.

## Risk Generators (`_risk_R11` through `_risk_R3`)
- **Purpose:** Inject disruptions based on exponential/uniform/binomial distributions.
- **Thesis Ref:** Table 6.12.
- **Usage:** Yields timeouts in the SimPy environment to trigger and resolve outages.

## `_backorder_rate(self)`
- **Purpose:** Calculates the fraction of delayed orders.
- **Thesis Ref:** Equation 5.4.
- **Note:** Uses order counts, directly supporting Re(FRt) calculation.
