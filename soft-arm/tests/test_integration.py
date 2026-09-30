import sys
import numpy as np

sys.path.insert(0, "soft-arm")
from soft_arm_sim import SoftArmSim

sim = SoftArmSim()
sim.set_pre_inflation(0.0)
sim.reset()

pressure = np.zeros((4, 5))

# Supply 3 psi to only the East column's level-2 pouch.
pressure[0, 2] = 3.0

for _ in range(300):
    observation = sim.step(pressure)

print("Tip position:", observation["tip_pos"])
print("Actual pressures:\n", observation["p_actual"])