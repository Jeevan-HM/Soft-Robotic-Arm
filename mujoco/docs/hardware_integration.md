# Hardware Integration Reference

> Extracted from `hardware.py`, `hardware_test.py`, `visualize.py`, and `experiment_data.py`
> before those files were removed from the digital-twin core.
> These notes are needed if you reconnect the twin to the physical testbed.

---

## Network Addresses (Tailscale)

| Machine | Role | Tailscale IP | Notes |
|---------|------|-------------|-------|
| `raspberrypi-testbed` | Regulator server (Pi) | `100.82.152.108` | Runs ZMQ REQ/REP + PUB servers |
| `rise-testbed-laptop` | OptiTrack publisher | `100.124.65.8` | Runs ZMQ PUB for mocap rigid bodies |

---

## ZMQ Protocol

### Regulator server (Pi) — REQ/REP

| Port | Socket type | Direction |
|------|-------------|-----------|
| `5555` | REQ/REP | Client → Pi (command) |
| `5556` | PUB | Pi → clients (pressure monitor feed) |

**Commands sent to `tcp://100.82.152.108:5555`:**

```json
{ "cmd": "zero_all" }
{ "cmd": "set_all",  "psi": 3.0 }
{ "cmd": "set_many", "values": { "0": 2.5, "1": 3.0, "2": 2.5, "3": 3.0 } }
```

These are low-level regulator-server messages, not the three-value controller
API. Channel 0 is used only while charging Segment 1 during setup; isolate it
before tracking. Runtime controller commands address channels 1--3 (S2--S4).

**Response:**
```json
{ "status": "ok" }
{ "status": "partial" }
{ "status": "error", "message": "..." }
```

> ⚠️ Always send `zero_all` on connect and on close. There is no watchdog on the Pi — if the client crashes, the arm stays pressurised.

---

### OptiTrack publisher (testbed laptop) — ZMQ PUB

| Port | Socket type |
|------|-------------|
| `5556` | SUB (subscribe to all) |

**Message format** (JSON string, one per rigid body per frame, ~50 Hz):

```json
{
  "id": 1,
  "name": "RB 1",
  "position":   [x, y, z],
  "quaternion": [qx, qy, qz, qw]
}
```

- Units: metres, normalised quaternion
- OptiTrack is **Y-up**; MuJoCo is **Z-up** — conversion needed (see below)

---

## Coordinate Frame Conversion (OptiTrack → MuJoCo)

```python
# Position
mj_pos = [opti[0], -opti[2], opti[1]]   # X_mj=X_o, Y_mj=-Z_o, Z_mj=Y_o

# Rotation matrix from quaternion [qx, qy, qz, qw]
T = [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]]
R_mj = T @ R_opti @ T.T
```

---

## OptiTrack Rigid Body Roles

Three rigid bodies, told apart by **height** (Z in MuJoCo frame):

| Height | Role | ID |
|--------|------|----|
| Highest | Mount — fixed, defines the base frame | RB 1 |
| Middle  | Base — top of soft arm | RB 2 |
| Lowest  | Tip  — tip of soft arm | RB 3 |

**Tip position in the digital twin frame:**
```
tip_pos = [af_tip_x, af_tip_y, MOUNT_Z + af_tip_z]
```
where `af_tip` is the tip in the **mount's own frame** and `MOUNT_Z = 0.5 m`
(the fixed mount height in the MuJoCo model).

> Assumption: arm hangs down, tip never rises above the base RB. Valid for active pressures ≤ 9 psi.

---

## Regulator → Column Wiring

Simulator column convention:

| Column | Segment | Direction | Runtime role |
|-------:|--------:|-----------|--------------|
| 0 | S1 | East  | Charged once, then sealed |
| 1 | S2 | North | Active command 1 (`+Y`) |
| 2 | S3 | West  | Active command 2 (`-X`) |
| 3 | S4 | South | Active command 3 (`-Y`) |

Default mapping (verify against actual wiring before first real run):
```python
REG_FOR_COLUMN = {0: 0, 1: 1, 2: 2, 3: 3}
```

Do not use column 0 for a runtime direction test: it is Segment 1, which must
be pre-inflated and isolated before control begins. Verify active wiring with
one low-pressure command at a time: S2/column 1 should bend toward North
(`+Y`), S3/column 2 toward West (`-X`), and S4/column 3 toward South (`-Y`).
Confirm the regulator indices on the physical testbed before closing the loop.

---

## Safety Limits

| Parameter | Value |
|-----------|-------|
| Arm pressure limit (documented) | 10 psi |
| Hardware/coursework command limit | **9 psi** |
| Internal calibrated model bound (`ArmConfig.p_max`) | 11 psi |
| Control rate | **100 Hz** |
| Reservoir charge (Segment 1) | 1–3 psi, pre-inflated then sealed |

The 11 psi model bound provides headroom for calibrated regulator gain and
bias. It is not a physical-arm command limit; clamp real-arm and coursework
controller setpoints to 0--9 psi.

---

## Experiment CSV Format

Files in `data/_data_extract/`. Naming convention:
```
<waveform>_<charge>-<max>_<topology>.csv
```
- `waveform`: `axial`, `circular`, or `triangular`
- `charge`: Segment-1 reservoir pressure in psi
- `max`: peak active command pressure in psi
- `topology`: `coupled` or `parallel`

**Example:** `axial_2-10_parallel.csv`

### 36 CSV columns

```
step_id, time,
Desired_pressure_segment_1,
Desired_pressure_segment_2,
Desired_pressure_segment_3,
Desired_pressure_segment_4,
Measured_pressure_Segment_1_pouch_1 … pouch_5,
Measured_pressure_Segment_2,
Measured_pressure_Segment_3,
Measured_pressure_Segment_4,
Rigid_body_1_x/y/z, Rigid_body_1_qx/qy/qz/qw,
Rigid_body_2_x/y/z, Rigid_body_2_qx/qy/qz/qw,
Rigid_body_3_x/y/z, Rigid_body_3_qx/qy/qz/qw,
mocap_time_rel_s
```

- Control log: ~100 Hz; OptiTrack: ~50 Hz (deduplicated before interpolation)
- Active window: 180 s, starts when Segment-1 desired pressure drops to ~0
- Active segment order: **S2, S3, S4** (Segment 1 is sealed reservoir)

---

## Digital Twin Interface

```python
from soft_robotic_arm import make_calibrated_sim

sim = make_calibrated_sim(
    topology="parallel",          # or "coupled"
    reservoir_pressure_psi=2.0,
    control_hz=100.0,
)

obs = sim.step([2.0, 5.0, 2.0])  # S2, S3, S4 pressures in psi
print(obs["tip_pos"])             # [x, y, z] in metres
print(obs["time"])                # simulation time in seconds
```

The former `HardwareArm` class had the identical `step()` / `reset()` / `observe()`
interface — swapping sim ↔ real arm was a single constructor change.
