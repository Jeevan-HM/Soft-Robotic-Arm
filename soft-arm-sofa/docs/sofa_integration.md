# SOFA Integration Reference

> Extracted from `pressure_control.py`, `pressure_panel_app.py`, and `demo.py`
> before those files were removed from the digital-twin core.
> These notes are needed if you want to reconnect the pressure control UI
> or add motion demos.

---

## How to Launch

```bash
./run.sh                          # open scene in SOFA GUI with pressure panel
./run.sh mesh/generate_mesh.py   # run any other script in the SOFA environment
```

`run.sh` sets three required environment variables on macOS:

| Variable | Value |
|----------|-------|
| `SOFA_ROOT` | `<project>/SOFA` |
| `PYTHONPATH` | `$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages` |
| `DYLD_FRAMEWORK_PATH` | `/Library/Frameworks` |

---

## Pressure Panel ↔ SOFA Protocol (UDP, localhost)

The pressure panel and SOFA communicate over **UDP on localhost**.
Port is randomised per run to allow multiple simulations to coexist:

```bash
export SOFT_ARM_PRESSURE_PORT=$((47631 + RANDOM % 1000))
```

**Default port:** `47631` — override via `SOFT_ARM_PRESSURE_PORT` env var.

### Panel → SOFA (command packet, JSON)

```json
{
  "pressures": [[col0_lvl0, col0_lvl1, col0_lvl2, col0_lvl3, col0_lvl4],
                [col1_lvl0, ...],
                [col2_lvl0, ...],
                [col3_lvl0, col3_lvl1, col3_lvl2, col3_lvl3, col3_lvl4]],
  "pre_inflation_psi": 0.0
}
```

- Shape: **4 columns × 5 levels** (East, North, West, South)
- Units: **psi** (converted to Pa inside SOFA: `1 psi = 6894.76 Pa`)
- Safety: if no command received for **1.0 s**, pressures reset to zero

### SOFA → Panel (feedback, ~20 Hz)

```json
{
  "actual": [[col0_lvl0, ..., col0_lvl4], ..., [col3_lvl0, ..., col3_lvl4]],
  "pre_inflation_psi": 0.0,
  "sim_time": 1.23
}
```

Feedback interval: **0.05 s** (20 Hz)

---

## Unit Conversions

| From | To | Factor |
|------|----|--------|
| psi | Pa (SOFA SI) | × **6894.76** |
| Pa | psi | ÷ **6894.76** |

SOFA uses **SI units (Pa)** internally. All Python-facing APIs use **psi**.

---

## Digital Twin Interface

```python
from soft_arm_sim import SoftArmSim
import numpy as np

sim = SoftArmSim()
sim.set_pre_inflation(0.0)   # baseline pressure on all 20 pouches
sim.reset()

pressure = np.zeros((4, 5))  # 4 columns × 5 levels [psi]
pressure[0, 2] = 3.0         # 3 psi on East column, level 2

obs = sim.step(pressure)
print(obs["tip_pos"])        # [x, y, z] in metres
print(obs["p_actual"])       # 4×5 actual pressures [psi]
```

---

## Arm Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Arm length (rest) | 0.220 | m |
| Arm outer radius | 0.046 | m |
| Column axis offset | 0.028 | m |
| Column cavity radius | 0.015 | m |
| Columns | 4 | — |
| Levels (pouches per column) | 5 | — |
| Pouch gap (wall thickness) | 0.004 | m |
| Young's modulus E | 3 000 000 | Pa |
| Poisson ratio ν | 0.45 | — |
| Arm mass (fabric + fittings) | 0.35 | kg |
| Marker frame mass | 0.08 | kg |
| Pneumatic time constant τ | 0.12 | s |

**Column convention:**

| Index | Direction |
|-------|-----------|
| 0 | East (azimuth 0°) |
| 1 | North |
| 2 | West |
| 3 | South |

Local +Z runs from mount toward the tip (arm hangs down).

---

## Motion Primitives Reference

| Motion | Pressure pattern |
|--------|-----------------|
| Single column bend | `pressure[col, :] = p` — one column fully pressurised |
| Axial extension | `pressure[:, :] = p` — all 4 columns equal |
| Circular tip path | 90° phase-shifted sinusoids across the 4 columns |
| Triangular tip path | Cycle through 3 column-dominant pressure vertices |

---

## Mesh

The FEM mesh is pre-generated and committed:

```
mesh/soft_arm.msh    ← GMSH tetrahedral mesh, loaded directly by SOFA
```

To regenerate (only needed if arm geometry changes):
```bash
./run.sh mesh/generate_mesh.py
```
