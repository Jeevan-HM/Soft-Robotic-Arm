# MuJoCo Digital Twin — Soft Robotic Arm

Calibrated MuJoCo digital twin of the fabric pneumatic soft arm.
The model is fitted to 36 physical-arm recordings and replicates
measured actuator delay, pressure mapping, pneumatic lag, and
sealed-reservoir response.

## Quick start

```bash
uv sync
```

```python
from soft_robotic_arm import make_sim

sim = make_sim(
    topology="parallel",          # or "coupled"
    reservoir_pressure_psi=2.0,
    control_hz=100.0,
)

obs = sim.step([2.0, 5.0, 2.0])  # S2, S3, S4 pressures [psi]
print(obs["tip_pos"])             # [x, y, z] in metres
print(obs["time"])                # simulation time [s]
```

Segment 1 is a charged, sealed five-pouch reservoir — it is set once at
construction time and not commanded at runtime. Segments 2–4 are the three
active actuators; commands are always three absolute pressure setpoints in psi.

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
immutable Git tag (for example, `v0.2.0`). Students can then install that exact
version without cloning the repository:

```bash
python -m pip install "soft-robotic-arm[coursework] @ git+https://github.com/Jeevan-HM/Soft-Robotic-Arm.git@v0.2.0#subdirectory=mujoco"
```

The tag in this example still has to be created and pushed. The package has not
been published to PyPI, so do not use a `soft-robotic-arm==...` command unless
you publish it there separately.

Controller code returns three absolute pressure commands in `[S2, S3, S4]`
order. Segment 1 is a sealed reservoir and is never a controller output.

The shared circular evaluation defaults to a 7 mm radius around the settled
`[4.5, 4.5, 4.5]` psi operating point. That radius is chosen from the calibrated
workspace: a 30 mm circle is not reachable with S1 sealed and S2--S4 limited
to the hardware-safe 0--9 psi command range. The model's 11 psi bound leaves
headroom for calibrated regulator gain and bias; it is not the coursework
command limit.

The supported public imports are `ArmConfig`, `CircularTrackingTask`,
`RobotCalibration`, `SoftArmSim`, `TrackingResult`, `build_arm_xml`,
`evaluate_controller`, `make_reference`, and `make_sim`.

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
