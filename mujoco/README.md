# MuJoCo Digital Twin — Soft Robotic Arm

MuJoCo teaching model and calibrated digital twin of the fabric pneumatic
soft arm. The coursework interface lets students command all twenty pouches
independently and read all twenty pouch pressures.

## Quick start

```bash
pip install soft-robotic-arm
```

This installs MuJoCo, NumPy, Matplotlib, and the ImageIO video encoder used by
the coursework notebook to render controller runs as MP4 files.

```python
import numpy as np

from soft_robotic_arm import make_sim

sim = make_sim(control_hz=100.0)
obs = sim.reset()

pouch_command = np.full((4, 5), 2.0)   # rows S1--S4, columns P1--P5
pouch_command[1, 0] = 5.0              # command only S2, P1
obs = sim.step(pouch_command)
print(obs["tip_pos"])             # [x, y, z] in metres
print(obs["pouch_pressures"])     # shape (4 segments, 5 pouches)
```

Rows are `[S1 East, S2 North, S3 West, S4 South]`; columns are pouches P1--P5.
The coursework notebook explains both segment and pouch commands and includes
a PD-control example for circular tip tracking.

Research replay of the measured configuration remains available through
`make_calibrated_sim()`. That interface charges and seals S1 and commands only
`[S2, S3, S4]`.

## Coursework interface

### Commands

`make_sim()` returns the symmetric teaching plant used by the Colab notebook.
Pass either command form to `sim.step()`:

| Command shape | Meaning |
|---|---|
| `(4,)` | One pressure for each segment S1--S4; broadcast to its five pouches |
| `(4, 5)` | One pressure for every `(segment, pouch)` pair |

Coursework commands are absolute pressures in psi and should remain in the
0--9 psi range. One call to `step()` advances one control interval.

### Observations

`reset()`, `step()`, and `observe()` return an observation dictionary:

| Key | Shape | Meaning |
|---|---:|---|
| `time` | scalar | simulation time in seconds |
| `tip_pos` | `(3,)` | Cartesian tip position `[x, y, z]` in metres |
| `tip_vel` | `(3,)` | Cartesian tip velocity in m/s |
| `tip_quat` | `(4,)` | tip orientation quaternion |
| `pouch_pressures` | `(4, 5)` | measured pressure for every pouch in psi |
| `segment_pressures` | `(4,)` | mean measured pressure for S1--S4 |
| `p_actual` | `(4, 5)` | internal pressure after pneumatic lag |
| `q` | model-dependent | MuJoCo generalized position vector |

The calibrated simulator additionally returns `reservoir_pressures` with shape
`(5,)`, `actuator_pressures` with shape `(3,)`, and `actuator_columns`, the
three zero-based segment indices that remain actively commanded.

### `SoftArmSim` methods

Construct this class through `make_sim()` for coursework or
`make_calibrated_sim()` for measured-plant replay.

| Member | Purpose |
|---|---|
| `reset(clear_log=True)` | reset physics, pneumatics, sensors, and optionally the pressure log |
| `step(command)` | apply a pressure command, advance one control tick, and return an observation |
| `observe()` | read the current observation without advancing time |
| `set_pre_inflation(psi)` | change the symmetric pre-inflation used for stiffness scaling |
| `actuator_columns` | indices of segments that accept runtime pressure commands |
| `set_reservoir_pressure(pressure, column=0)` | charge and seal a reservoir segment; advanced calibrated use |
| `clear_pressure_log()` | discard recorded pressure samples without resetting the plant |
| `get_pressure_log()` | return `time`, `p_cmd`, `actuator_cmd`, and `p_actual` NumPy arrays |
| `save_pressure_log(path, which="p_cmd", relative_time=False)` | export a selected pressure log to CSV |
| `render_frame(cam_azimuth=135, cam_elevation=-20, cam_distance=0.8, width=640, height=480)` | return an RGB image from an offscreen MuJoCo camera |
| `close()` | release the optional renderer |

For `save_pressure_log`, `which` may be `"p_cmd"`, `"actuator_cmd"`, or
`"p_actual"`. The method returns the written `Path`. Call `close()` when a
notebook is finished if rendering was used.

## Public package API

### Main construction functions

| Import | Purpose |
|---|---|
| `make_sim(control_hz=100, seed=0, **overrides)` | create the four-segment, twenty-pouch coursework simulator |
| `make_calibrated_sim(reservoir_pressure_psi=2, control_hz=100, topology="parallel", seed=0, **overrides)` | create the measured plant with sealed S1 and commands for S2--S4 |
| `SoftArmSim` | simulator class returned by both construction functions |

### Shared trajectory evaluation

These helpers reproduce the measured three-actuator evaluation task. They are
separate from the notebook's twenty-pouch controller. An evaluation controller
implements `compute(t, obs, ref)` and returns three pressures for S2--S4.

| Import | Purpose |
|---|---|
| `CircularTrackingTask` | immutable settings for control rate, settling, radius, frequency, baseline, and pressure limit |
| `CircularTrackingTask.baseline_command(max_pressure)` | validate and return the three-pressure baseline |
| `CircularTrackingTask.settle_steps` / `track_steps` | convert task durations into sample counts |
| `make_reference(home, t, task=None)` | return the task's three-dimensional circular position reference |
| `evaluate_controller(controller, task=None, sim=None)` | run settling and tracking, then return metrics and sampled signals |
| `TrackingResult` | result object containing RMSE, maximum error, phase lag, effort, trajectories, and commands |
| `TrackingResult.as_dict()` | return metrics and copies of the sampled arrays in a dictionary |
| `TrackingResult.plot(label="Controller")` | plot path, tracking error, and commanded pressures |

### Model and calibration tools

Most coursework users do not need these lower-level interfaces.

| Import | Purpose |
|---|---|
| `ArmConfig` | dataclass containing geometry, mechanics, pneumatics, and simulation parameters |
| `ArmConfig.n_channels` | total number of pressure channels |
| `ArmConfig.level_height()` | height of one modeled pouch level |
| `ArmConfig.col_azimuths()` / `azimuths()` | segment azimuths in radians |
| `ArmConfig.to_json(path)` / `from_json(path)` | save or load a validated configuration |
| `build_arm_xml(config)` | generate the MuJoCo MJCF XML string for an `ArmConfig` |
| `RobotCalibration.load(path=None)` | load and validate packaged or external calibration JSON |
| `RobotCalibration.make_arm_config()` | construct an `ArmConfig` from measured parameters |
| `RobotCalibration.simulator_kwargs(...)` | obtain calibrated simulator keyword arguments |
| `RobotCalibration.make_sim(...)` | construct a simulator directly from the calibration object |

## Repository files

| File | Purpose |
|------|---------|
| `soft_robotic_arm/model.py` | Generates the MuJoCo MJCF XML for the arm geometry and physics |
| `soft_robotic_arm/simulator.py` | `SoftArmSim` — the digital twin: `step()`, `observe()`, `render()` |
| `soft_robotic_arm/calibration.py` | Loads the packaged calibration and constructs the simulator |
| `soft_robotic_arm/evaluation.py` | Shared circular trajectory and controller metrics |
| `soft_robotic_arm/data/calibration.json` | Fitted pressure, reservoir, and mechanics parameters |

## Development installation

From a checkout of this repository, install the package in editable mode with:

```bash
python -m pip install -e .
```

Coursework controller code may return four segment pressures or a `(4, 5)`
matrix of individual pouch pressures, limited to the safe 0--9 psi range.

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
