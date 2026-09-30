# SOFA FEM Simulation — Soft Robotic Arm

This directory contains the **SOFA + SoftRobots** simulation of the fabric
pneumatic soft arm, replacing the rigid-body MuJoCo approximation with a
Finite Element Method (FEM) continuum mechanics model.

---

## Why SOFA instead of MuJoCo?

| | MuJoCo (`mujoco/`) | SOFA (`soft-arm/`) |
|---|---|---|
| Physics | Rigid links + hinge joints | FEM continuum solid mechanics |
| Deformation | Approximate (piecewise rigid) | Realistic (arbitrary bending shapes) |
| Material law | Spring-damper constants `k`, `d` | Young's modulus `E`, Poisson ratio `ν` |
| Actuation | Generalised forces on DOFs | `SurfacePressureConstraint` on cavity surfaces |
| Pre-inflation | Stiffness scaling hack | Natural cavity pressure |
| Coupling effects | Not modelled | Pressure ↔ deformation naturally coupled |
| Speed | Very fast (< 1 ms/step) | Slower (~10–100 ms/step depending on mesh) |
| Colab support | ✅ | ❌ (native binary required) |

**Use SOFA when you need:**
- Realistic deformation shapes (not just tip position)
- Accurate stress/strain fields
- Physically grounded material parameters (measured from tensile tests)
- Coupling between pressurisation, geometry change, and stiffness

---

## Directory structure

```
soft-arm/
├── mesh/
│   ├── generate_mesh.py    ← Gmsh script: hollow cylinder + 20 pouches
│   └── soft_arm.msh        ← Generated tetrahedral mesh (run generate_mesh.py)
├── arm_config.py           ← ArmConfig dataclass (FEM parameters)
├── pressure_control.py      ← bridge from the pressure window to SOFA
├── pressure_panel_app.py    ← native editable 20-pouch pressure window
├── robot_visual.py         ← Smooth, photo-matched 20-pouch visual skin
├── soft_arm_scene.py       ← SOFA scene definition (SofaPython3)
├── soft_arm_sim.py         ← Python wrapper (same API as MuJoCo SoftArmSim)
├── demo.py                 ← Four motion demos (bend, axial, circle, triangle)
└── README.md               ← This file
```

---

## Installation

### 1. Install SOFA with SoftRobots plugin

SOFA is not pip-installable.  The easiest way is to download a pre-built binary:

1. Go to **https://www.sofa-framework.org/download/**
2. Download the latest stable release for your OS (v23.12 or newer recommended)
3. Extract to a directory, e.g. `/opt/sofa`

The binary includes `SoftRobots`, `SofaPython3`, and `STLIB` plugins.

### 2. Set PYTHONPATH

```bash
# Add to ~/.zshrc or ~/.bashrc
export SOFA_ROOT=/opt/sofa
export PYTHONPATH=$SOFA_ROOT/lib/python3/site-packages:$PYTHONPATH
export PATH=$SOFA_ROOT/bin:$PATH
```

Reload your shell, then verify:

```bash
python3 -c "import Sofa; print(Sofa.__version__)"
```

### 3. Install Gmsh (for mesh generation)

```bash
pip install gmsh
```

### 4. Generate the mesh

```bash
# From the repository root
python soft-arm/mesh/generate_mesh.py

# Optional: view in Gmsh GUI
python soft-arm/mesh/generate_mesh.py --view
```

This writes `soft-arm/mesh/soft_arm.msh` with 20 tagged, closed pouch surfaces.

---

## Running the simulation

### Headless (Python API)

```python
from soft_arm_sim import SoftArmSim
import numpy as np

sim = SoftArmSim()
sim.set_pre_inflation(1.5)   # psi
sim.reset()

P = np.zeros((4, 5))         # (n_cols, n_pouches)
P[0, :] = 3.0                # East column @ 3 psi
for _ in range(300):         # 3 s at 100 Hz
    obs = sim.step(P)

print("Tip position [cm]:", obs["tip_pos"] * 100)
```

### With SOFA GUI

```bash
cd soft-arm
./run.sh
```

The GUI hides the coarse tetrahedral computation mesh and displays a separate
deforming visual skin based on the hardware photos in `mujoco/docs`: 20 black
heat-sealed fabric pouches, blue carrier panels, individual airline ports,
connector plates, the plywood/standoff mount, and the external marker cross.

`run.sh` opens a separate **Soft Arm Pressure Control** window with ordinary
editable PSI spin boxes.  Each direction has five independent commands:
level 0 is next to the plywood mount and level 4 is next to the end effector.
The window also provides column/all-pouch fills, global pre-inflation, and
**VENT ALL**.  Pressure entries accept any non-negative finite PSI value; there
is no 10 PSI software ceiling.  Four live plots show the filtered pressure reported by SOFA for
levels 0–4 in the East, North, West, and South columns over the last 20 seconds.
Changes are sent to SOFA immediately over localhost.  `run.sh` starts the SOFA
animation automatically, and the pressure plots redraw continuously from the
live SOFA readback stream while the arm moves.  The time axis expands from a
one-second startup view into a rolling 20-second history so early samples stay
visible.
The values shown inside SofaImGui's component inspector remain read-only because
that GUI cannot edit numeric Data created by Python.

### Run motion demos

```bash
# Single demo
python soft-arm/demo.py --demo circle --plot

# All demos
python soft-arm/demo.py --demo all --plot
```

---

## Physical parameters

### Material (FEM)

| Parameter | Value | Unit | Basis |
|---|---|---|---|
| Effective Young's modulus `E` | **3 MPa** | Pa | Composite calibration for TPU-coated heat-sealable pouches, blue carriers, seams, fittings, and separator plates. |
| Poisson ratio `ν` | **0.45** | — | Near-incompressible effective continuum; avoids FEM locking. |
| Moving mass | **0.35 kg arm + 0.08 kg tip** | kg | Matches the MuJoCo construction model. |
| FEM method | `large` (co-rotational) | — | Handles large bending displacements (>15°). Required for soft arms. |

### Rayleigh damping

SOFA uses Rayleigh damping to model viscous dissipation:

```
C = α·M + β·K
```

| Coefficient | Value | Meaning |
|---|---|---|
| α (mass) | 0.1 | Low-frequency (rigid-body) damping |
| β (stiffness) | 0.1 | High-frequency (stiffness-proportional) damping |

These values are starting points.  Tune α/β to match the step-response
settling time observed on the physical arm (~0.8–1.2 s from cold press to
steady state at 4 psi).

### Pneumatics

| Parameter | Value | Basis |
|---|---|---|
| Software pressure ceiling | None | Accepts any non-negative finite PSI command |
| Pre-inflation | 1.5 psi (10.3 kPa) | Operational default (from MuJoCo experiments) |
| Time constant τ | 0.12 s | Identified from hardware step response (sysid) |
| Physics timestep | 1 ms | Matches MuJoCo and keeps pressure actuation stable |
| SOFA pressure scale | 0.08 | Maps physical psi to the effective fabric continuum |
| Max cavity growth | 1 initial pouch volume | Strain limit for heat-sealed fabric |

Pressure is applied as a first-order filtered input:

```
p_actual += (dt / τ) × (p_cmd − p_actual)   [per physics substep]
```

### Geometry

| Parameter | Value |
|---|---|
| Arm length | 220 mm |
| Outer body radius | 46 mm (col_offset 28 mm + col_radius 18 mm) |
| Cavity radius | 15 mm (slightly smaller for wall thickness ≥ 3 mm) |
| Number of levels | 5 × 44 mm |
| Pouch height | 40 mm |
| Inter-pouch wall | 4 mm |
| Outer radial wall | ~3 mm minimum |
| Number of columns | 4 (East / North / West / South) |
| Plywood mount | 150 × 150 × 9 mm with four ~100 mm standoffs |
| Marker-frame offset | 12 mm beyond the free face |
| Marker cross | 140 mm span (70 mm half-arm) |

---

## Parameter comparison: SOFA vs MuJoCo

The MuJoCo spring-damper parameters and SOFA FEM parameters describe the same
physical system at different levels of abstraction.

| Physical property | MuJoCo parameter | MuJoCo value (sysid) | SOFA parameter | SOFA value |
|---|---|---|---|---|
| Bending stiffness | `base_stiffness` | 0.459 N·m/rad | effective `young_modulus` | 3,000,000 Pa |
| Bending damping | `base_damping` | 0.053 N·m·s/rad | Rayleigh β | 0.1 |
| Axial stiffness | `axial_stiffness` | 555 N/m | (FEM-intrinsic) | — |
| Mass | `mass` + `tip_mass` | 0.35 + 0.08 kg | distributed arm + mapped tip mass | 0.35 + 0.08 kg |
| Bending actuator | `pressure_gain` | 0.679 N/psi | `SurfacePressureConstraint` | FEM-computed |
| Extension actuator | `extension_gain` | 0.0348 N/psi | (FEM-intrinsic) | — |

The SOFA FEM model does not require explicit `pressure_gain` or `extension_gain`
parameters: the bending and extension forces emerge naturally from the cavity
geometry, internal pressure, and material stiffness.

---

## Mesh details

The mesh is a tetrahedral volume mesh of the arm body:

- **Outer shell**: cylinder, radius 46 mm, length 220 mm
- **20 independent pouches**: 4 columns × 5 levels, radius 15 mm, offset 28 mm from centre
- **Physical groups**:
  - `ArmBody` (tag 1): tetrahedral volume
  - `Base` (tag 2): top face — fixed (FixedConstraint)
  - `Tip` (tag 3): bottom face — free
  - `OuterSurface` (tag 4): lateral outer surface
  - `CavityCol0_Lvl0` … `CavityCol0_Lvl4`: tags 10–14
  - `CavityCol1_Lvl0` … `CavityCol1_Lvl4`: tags 15–19
  - `CavityCol2_Lvl0` … `CavityCol2_Lvl4`: tags 20–24
  - `CavityCol3_Lvl0` … `CavityCol3_Lvl4`: tags 25–29

Mesh element sizes: 8 mm on the outer surface, 5 mm on cavity walls.
Total: ~10,000 nodes, ~40,000 tetrahedra (tunable in `generate_mesh.py`).

---

## Troubleshooting

**`ImportError: No module named 'Sofa'`**
→ SOFA is not on `PYTHONPATH`.  See Installation step 2.

**`MeshGmshLoader` fails to load `soft_arm.msh`**
→ Run `python soft-arm/mesh/generate_mesh.py` first.

**`"Object must have a tetrahedric topology"`**
→ The mesh has only surface triangles.  Re-generate with `generate_mesh.py`
  (it produces a proper volumetric tetrahedral mesh).

**Arm immediately collapses / explodes**
→ Check that `youngModulus` is in Pa (not kPa).  300 kPa = 300,000 Pa.

**Simulation is very slow**
→ Reduce mesh density in `generate_mesh.py` (increase `MESH_SIZE_OUTER`).

---

## References

- Polygerinos et al. (2015). "Modeling of soft fiber-reinforced bending actuators."
  *IEEE Trans. Robotics*, 31(3), 778–789.
- Mosadegh et al. (2014). "Pneumatic networks for soft robotics that actuate
  rapidly." *Adv. Functional Materials*, 24(15).
- SOFA documentation: https://www.sofa-framework.org/
- SoftRobots plugin: https://softrobots.readthedocs.io/
- Inria DEFROST team tutorials: https://project.inria.fr/softrobot/
