import sys
with open('supply_chain/config.py', 'r') as f:
    content = f.read()

new_profile = """
# Severe Training Profile - Curriculum Learning
# Extrema disrupción para forzar al RL a aprender robustness.
RISKS_SEVERE_TRAINING = {
    "R11": {"dist": "uniform", "a": 1, "b": 10, "recovery_mean": 6},  # 2x freq of severe (21->10), 3x recovery (2->6)
    "R12": {"dist": "binomial", "n": 12, "p": 8 / 11},
    "R13": {"dist": "binomial", "n": 12, "p": 8 / 10},
    "R14": {"dist": "binomial", "n": 2564, "p": 12 / 100},
    "R21": {"dist": "uniform", "a": 1, "b": 1008, "recovery_mean": 240}, # 2x freq of severe, 2x recovery
    "R22": {"dist": "uniform", "a": 1, "b": 336, "recovery_mean": 48},   # 2x freq of severe, 2x recovery
    "R23": {"dist": "uniform", "a": 1, "b": 672},
    "R24": {"dist": "uniform", "a": 1, "b": 168, "surge_lo": 7200, "surge_hi": 7800}, # 3x surge
    "R3": {"dist": "uniform", "a": 1, "b": 40_320}, # 4x more freq than current
}
"""

if 'RISKS_SEVERE_TRAINING' not in content:
    content = content.replace("RISKS_SEVERE_EXTENDED = {", new_profile + "\nRISKS_SEVERE_EXTENDED = {")
    with open('supply_chain/config.py', 'w') as f:
        f.write(content)
