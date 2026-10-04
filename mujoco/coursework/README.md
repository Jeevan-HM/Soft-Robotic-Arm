# Soft robotic arm coursework

The installable package now lives at the repository's `mujoco` root so the
coursework and calibrated research simulator use the same arm mechanics.

## Install

For instructor development from the repository's `mujoco` directory:

```bash
python -m pip install -e ".[coursework]"
```

For students, first create and push an immutable release tag. After a tag such
as `v0.3.0` exists, they can install the same model directly from GitHub:

```bash
python -m pip install "soft-robotic-arm[coursework] @ git+https://github.com/Jeevan-HM/Soft-Robotic-Arm.git@v0.3.0#subdirectory=mujoco"
```

That tag is a release example, not an assertion that it already exists. The
package is not currently documented as published on PyPI.

## Usage

```python
from soft_robotic_arm import make_sim

sim = make_sim(control_hz=100)
obs = sim.reset()
obs = sim.step([2.0, 2.0, 2.0, 2.0])  # S1, S2, S3, S4
print(obs["pouch_pressures"])          # shape (4, 5)
```

Each controller output must be exactly four absolute pressure setpoints in
`[S1, S2, S3, S4]` order, clipped to the hardware-safe 0--9 psi range.

## What's in each section of the notebook

| Public library symbol | Notebook use |
|-----------------------|--------------|
| `make_sim`, `SoftArmSim` | Four-segment teaching plant |
| `obs["pouch_pressures"]` | Four-by-five pouch pressure readings |
| `obs["segment_pressures"]` | Mean pressure for S1--S4 |

The notebook intentionally provides no reference controller or fixed tracking
challenge. Students receive the interface, a direction illustration, and a
controller template, then build the controller themselves.
