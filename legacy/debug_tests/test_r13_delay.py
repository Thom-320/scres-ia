from supply_chain.supply_chain import MFSCSimulation


class TestSim(MFSCSimulation):
    def _op2_supplier_delivery(self):
        while True:
            yield self.env.timeout(self.params["op2_rop"])
            print(f"[{self.env.now}] Op2 waking up, is_down: {self._is_down(2)}")
            while self._is_down(2):
                yield self.env.timeout(1)
            print(f"[{self.env.now}] Op2 starting processing")
            yield self.env.timeout(self._pt("op2_pt"))
            print(f"[{self.env.now}] Op2 finished processing, delivering")
            total_delivery = self.params["op2_q"] * 12
            yield self.raw_material_wdc.put(total_delivery)

    def _risk_R13(self):
        n = 12
        p = 1.0  # Force delay
        while True:
            yield self.env.timeout(self.params["op2_rop"])
            delayed = self.rng.binomial(n, p)
            if delayed > 0:
                delay = delayed * 24
                print(f"[{self.env.now}] R13 taking down Op2 for {delay}")
                self._take_down(2)
                yield self.env.timeout(delay)
                self._bring_up(2)
                print(f"[{self.env.now}] R13 bringing up Op2")


sim = TestSim(shifts=1, risks_enabled=True, seed=42, horizon=2000)
sim.run()
