import numpy as np
from supply_chain.supply_chain import MFSCSimulation
from supply_chain.config import SIMULATION_HORIZON, HOURS_PER_YEAR_GREGORIAN

seeds = [42, 123, 456, 789, 1000]
results = []
for seed in seeds:
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=True,
        risk_level="current",
        seed=seed,
        horizon=SIMULATION_HORIZON,
        year_basis="gregorian",
    ).run()
    throughput = sim.get_annual_throughput()
    results.append(throughput["avg_annual_delivery"])
    print(f"Seed {seed}: {throughput['avg_annual_delivery']:,.0f}")

print(f"Mean: {np.mean(results):,.0f} | Std: {np.std(results):,.0f}")
