# Soft robotic arm coursework

The installable package now lives at the repository's `mujoco` root so the
coursework and calibrated research simulator use exactly the same model.

## Install

For instructor development from the repository's `mujoco` directory:

```bash
python -m pip install -e ".[coursework]"
```

For students, first create and push an immutable release tag. After a tag such
as `v0.2.0` exists, they can install the same model directly from GitHub:

```bash
python -m pip install "soft-robotic-arm[coursework] @ git+https://github.com/Jeevan-HM/Soft-Robotic-Arm.git@v0.2.0#subdirectory=mujoco"
```

That tag is a release example, not an assertion that it already exists. The
package is not currently documented as published on PyPI.

## Usage

```python
from soft_robotic_arm import make_sim

sim = make_sim(reservoir_pressure_psi=2.0, control_hz=100)
obs = sim.reset()
obs = sim.step([2.0, 2.0, 2.0])  # S2, S3, and S4; S1 stays sealed
```

Each controller output must be exactly three absolute pressure setpoints in
`[S2, S3, S4]` order. The shared coursework evaluator clips them to the
hardware-safe 0--9 psi range. The calibrated internal model uses an 11 psi
state bound to accommodate regulator gain and bias; that is not permission to
send 11 psi to the physical arm.

## What's in each section of the notebook

| Public library symbol | Notebook use |
|-----------------------|--------------|
| `make_sim`, `SoftArmSim` | Calibrated three-pressure plant |
| `ArmConfig`, `build_arm_xml` | Model exploration |
| `RobotCalibration` | Inspect or construct from the packaged calibration |
| `CircularTrackingTask`, `make_reference` | Common circular reference |
| `evaluate_controller`, `TrackingResult` | Common scoring, signals, and plots |
