# Soft robotic arm coursework

The installable package now lives at the repository's `mujoco` root so the
coursework and calibrated research simulator use the same arm mechanics.

## Install

For instructor development from the repository's `mujoco` directory:

```bash
python -m pip install -e ".[coursework]"
```

For students, first create and push an immutable release tag. After a tag such
as `v0.4.0` exists, they can install the same model directly from GitHub:

```bash
python -m pip install "soft-robotic-arm[coursework] @ git+https://github.com/Jeevan-HM/Soft-Robotic-Arm.git@v0.4.0#subdirectory=mujoco"
```

That tag is a release example, not an assertion that it already exists. The
package is not currently documented as published on PyPI.

## Usage

```python
import numpy as np

from soft_robotic_arm import make_sim

sim = make_sim(control_hz=100)
obs = sim.reset()
pouch_command = np.full((4, 5), 2.0)   # rows S1--S4, columns P1--P5
pouch_command[0, 2] = 4.0              # command only S1, P3
obs = sim.step(pouch_command)
print(obs["pouch_pressures"])          # shape (4, 5)
```

The simulator accepts either four segment setpoints or a `(4, 5)` matrix that
commands all twenty pouches independently, clipped to 0--9 psi.

## What's in each section of the notebook

| Public library symbol | Notebook use |
|-----------------------|--------------|
| `make_sim`, `SoftArmSim` | Twenty-pouch teaching plant |
| `obs["pouch_pressures"]` | Four-by-five pouch pressure readings |
| `obs["segment_pressures"]` | Mean pressure for S1--S4 |

The notebook includes an individual-pouch example and a PD-controller example
that tracks a circular tip trajectory, followed by prompts for students to tune
or replace the controller.
