
with open("supply_chain/env.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "self.action_space = spaces.Box(" in line:
        new_lines.append(line)
    elif "shape=(4,)" in line and "spaces.Box" not in line: # Fix action space shape
        new_lines.append(line.replace("shape=(4,)", "shape=(5,)"))
    elif "multipliers = 1.25 + 0.75 * action" in line:
        # We need to split action array because action[4] is for shifts
        new_lines.append("        multipliers = 1.25 + 0.75 * action[:4]\n")
        new_lines.append("        \n")
        new_lines.append("        # Shift control: Continuous to Discrete (S=1 or S=2)\n")
        new_lines.append("        # If action[4] < 0 -> S=1, else S=2\n")
        new_lines.append("        chosen_shifts = 1 if action[4] < 0 else 2\n")
    elif "'op9_rop': OPERATIONS[9][\"rop\"] * multipliers[3]," in line:
        new_lines.append(line)
        new_lines.append("            'assembly_shifts': chosen_shifts,\n")
    else:
        new_lines.append(line)

with open("supply_chain/env.py", "w") as f:
    f.writelines(new_lines)
