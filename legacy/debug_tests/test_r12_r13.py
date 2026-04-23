from supply_chain.supply_chain import MFSCSimulation

sim = MFSCSimulation(shifts=1, risks_enabled=True, seed=42, horizon=80640)
sim.run()
print(f"R12 events: {[e.duration for e in sim.risk_events if e.risk_id == 'R12']}")
print(f"R13 events: {[e.duration for e in sim.risk_events if e.risk_id == 'R13']}")
