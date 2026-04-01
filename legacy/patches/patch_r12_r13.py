import re

with open("supply_chain/supply_chain.py", "r") as f:
    content = f.read()

# Fix _op1_contracting
content = re.sub(
    r'yield self\.env\.timeout\(self\._pt\("op1_pt"\)\)',
    r'''pt_remaining = self._pt("op1_pt")
            while pt_remaining > 0:
                while self._is_down(1):
                    yield self.env.timeout(1)
                yield self.env.timeout(1)
                pt_remaining -= 1''',
    content
)

# Fix _op2_supplier_delivery
content = re.sub(
    r'yield self\.env\.timeout\(self\._pt\("op2_pt"\)\)',
    r'''pt_remaining = self._pt("op2_pt")
            while pt_remaining > 0:
                while self._is_down(2):
                    yield self.env.timeout(1)
                yield self.env.timeout(1)
                pt_remaining -= 1''',
    content
)

with open("supply_chain/supply_chain.py", "w") as f:
    f.write(content)

