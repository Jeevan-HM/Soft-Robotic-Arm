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
from calibration import RobotCalibration

sim = RobotCalibration.load().make_sim(
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
| `mjcf_model.py` | Generates the MuJoCo MJCF XML for the arm geometry and physics |
| `simulator.py` | `SoftArmSim` — the digital twin: `step()`, `observe()`, `render()` |
| `calibration.py` | Loads `calibration.json` and constructs the calibrated simulator |
| `calibration.json` | Fitted physical parameters (pressure gains, delays, mechanics) |

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
