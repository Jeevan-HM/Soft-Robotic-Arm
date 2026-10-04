# MuJoCo Digital Twin — Soft Robotic Arm

MuJoCo teaching model and calibrated digital twin of the fabric pneumatic
soft arm. The coursework interface keeps controller work simple: students
command all four segments and read all twenty pouch pressures.

## Quick start

```bash
uv sync
```

```python
from soft_robotic_arm import make_sim

sim = make_sim(control_hz=100.0)

obs = sim.step([2.0, 5.0, 2.0, 2.0])  # S1, S2, S3, S4 [psi]
print(obs["tip_pos"])             # [x, y, z] in metres
print(obs["pouch_pressures"])     # shape (4 segments, 5 pouches)
```

The command order is `[S1 East, S2 North, S3 West, S4 South]`. The coursework
notebook contains only usage instructions, a manual experiment, and an empty
controller template; it does not provide a controller or fixed tracking task.

Research replay of the measured configuration remains available through
`make_calibrated_sim()`. That interface charges and seals S1 and commands only
`[S2, S3, S4]`.

## Files

| File | Purpose |
|------|---------|
| `soft_robotic_arm/model.py` | Generates the MuJoCo MJCF XML for the arm geometry and physics |
| `soft_robotic_arm/simulator.py` | `SoftArmSim` — the digital twin: `step()`, `observe()`, `render()` |
| `soft_robotic_arm/calibration.py` | Loads the packaged calibration and constructs the simulator |
| `soft_robotic_arm/evaluation.py` | Shared circular trajectory and controller metrics |
| `soft_robotic_arm/data/calibration.json` | Fitted pressure, reservoir, and mechanics parameters |

## Install for coursework

From a checkout of this repository, install the package and notebook extras with:

```bash
python -m pip install -e ".[coursework]"
```

Before distributing the assignment, push these package changes and create an
immutable Git tag (for example, `v0.3.0`). Students can then install that exact
version without cloning the repository:

```bash
python -m pip install "soft-robotic-arm[coursework] @ git+https://github.com/Jeevan-HM/Soft-Robotic-Arm.git@v0.3.0#subdirectory=mujoco"
```

The tag in this example still has to be created and pushed. The package has not
been published to PyPI, so do not use a `soft-robotic-arm==...` command unless
you publish it there separately.

Coursework controller code returns four absolute pressure commands in
`[S1, S2, S3, S4]` order, limited to the hardware-safe 0--9 psi range.

The supported public imports are `ArmConfig`, `CircularTrackingTask`,
`RobotCalibration`, `SoftArmSim`, `TrackingResult`, `build_arm_xml`,
`evaluate_controller`, `make_calibrated_sim`, `make_reference`, and `make_sim`.

## Docs

| File | Contents |
|------|---------|
| `docs/hardware_integration.md` | Network IPs, ZMQ protocol, OptiTrack data format, CSV format, coordinate frames, safety limits — everything needed to reconnect the twin to real hardware |
| `docs/arm_parameters.md` | Arm geometry and calibrated parameter reference |
| `docs/arm_model_guide.md` | Guide to the MJCF model structure |

## Calibration accuracy

Across 36 experiments, the calibrated mechanics replay achieves:
- **3.03 mm** mean lateral error
- **1.07 mm** mean axial error

Full desired-command replay (including actuator and reservoir dynamics):
- **3.34 mm** lateral, **1.17 mm** axial
- **0.44 psi** active-pressure, **0.09 psi** reservoir-pressure error
