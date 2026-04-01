with open('supply_chain/supply_chain.py', 'r') as f:
    content = f.read()

content = content.replace(
    "RISKS_SEVERE_EXTENDED,",
    "RISKS_SEVERE_EXTENDED,\n    RISKS_SEVERE_TRAINING,"
)

content = content.replace(
    '"severe_extended": RISKS_SEVERE_EXTENDED,',
    '"severe_extended": RISKS_SEVERE_EXTENDED,\n        "severe_training": RISKS_SEVERE_TRAINING,'
)

with open('supply_chain/supply_chain.py', 'w') as f:
    f.write(content)
