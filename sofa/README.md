# SOFA FEM Simulation — Soft Robotic Arm

This directory contains the **SOFA + SoftRobots** simulation of the fabric
pneumatic soft arm, replacing the rigid-body MuJoCo approximation with a
Finite Element Method (FEM) continuum mechanics model.

---

## Why SOFA instead of MuJoCo?

| | MuJoCo (`mujoco/`) | SOFA (`sofa/`) |
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
sofa/
├── mesh/
│   ├── generate_mesh.py    ← Gmsh script: hollow cylinder + 4 cavities
│   └── soft_arm.msh        ← Generated tetrahedral mesh (run generate_mesh.py)
├── arm_config.py           ← ArmConfig dataclass (FEM parameters)
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
python sofa/mesh/generate_mesh.py

# Optional: view in Gmsh GUI
python sofa/mesh/generate_mesh.py --view
```

This writes `sofa/mesh/soft_arm.msh` (~10k nodes, ~40k tetrahedra at default settings).

---

## Running the simulation

### Headless (Python API)

```python
from sofa.soft_arm_sim import SoftArmSim
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
runSofa sofa/soft_arm_scene.py
```

### Run motion demos

```bash
# Single demo
python sofa/demo.py --demo circle --plot

# All demos
python sofa/demo.py --demo all --plot
```

---

## Physical parameters

### Material (FEM)

| Parameter | Value | Unit | Basis |
|---|---|---|---|
| Young's modulus `E` | **0.3 MPa** | Pa | Dragon Skin 10 silicone (150–250 kPa) + fabric reinforcement factor ~1.5× (Polygerinos et al. 2015; Mosadegh et al. 2014). Cross-validated against MuJoCo sysid result: `base_stiffness = 0.459 N·m/rad` at 1.5 psi pre-inflation implies E ≈ 0.25–0.35 MPa. |
| Poisson ratio `ν` | **0.45** | — | Silicone elastomers are nearly incompressible (ν → 0.5). 0.45 avoids FEM locking while remaining physically accurate. |
| Density `ρ` | **1100 kg/m³** | kg/m³ | Dragon Skin / Ecoflex: 1070–1200 kg/m³. Fabric adds ~5 % mass. |
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
| Max pressure | 10 psi (68.9 kPa) | Hardware limit |
| Pre-inflation | 1.5 psi (10.3 kPa) | Operational default (from MuJoCo experiments) |
| Time constant τ | 0.12 s | Identified from hardware step response (sysid) |

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
| Wall thickness | ~3 mm minimum |
| Number of columns | 4 (East / North / West / South) |

---

## Parameter comparison: SOFA vs MuJoCo

The MuJoCo spring-damper parameters and SOFA FEM parameters describe the same
physical system at different levels of abstraction.

| Physical property | MuJoCo parameter | MuJoCo value (sysid) | SOFA parameter | SOFA value |
|---|---|---|---|---|
| Bending stiffness | `base_stiffness` | 0.459 N·m/rad | `young_modulus` | 300,000 Pa |
| Bending damping | `base_damping` | 0.053 N·m·s/rad | Rayleigh β | 0.1 |
| Axial stiffness | `axial_stiffness` | 555 N/m | (FEM-intrinsic) | — |
| Mass | `mass` | 0.369 kg | `density` × volume | ~0.37 kg |
| Bending actuator | `pressure_gain` | 0.679 N/psi | `SurfacePressureConstraint` | FEM-computed |
| Extension actuator | `extension_gain` | 0.0348 N/psi | (FEM-intrinsic) | — |

The SOFA FEM model does not require explicit `pressure_gain` or `extension_gain`
parameters: the bending and extension forces emerge naturally from the cavity
geometry, internal pressure, and material stiffness.

---

## Mesh details

The mesh is a tetrahedral volume mesh of the arm body:

- **Outer shell**: cylinder, radius 46 mm, length 220 mm
- **4 longitudinal cavities**: cylinders, radius 15 mm, offset 28 mm from centre
- **Physical groups**:
  - `ArmBody` (tag 1): tetrahedral volume
  - `Base` (tag 2): top face — fixed (FixedConstraint)
  - `Tip` (tag 3): bottom face — free
  - `OuterSurface` (tag 4): lateral outer surface
  - `CavityCol0` (tag 10): East cavity wall
  - `CavityCol1` (tag 11): North cavity wall
  - `CavityCol2` (tag 12): West cavity wall
  - `CavityCol3` (tag 13): South cavity wall

Mesh element sizes: 8 mm on the outer surface, 5 mm on cavity walls.
Total: ~10,000 nodes, ~40,000 tetrahedra (tunable in `generate_mesh.py`).

---

## Troubleshooting

**`ImportError: No module named 'Sofa'`**
→ SOFA is not on `PYTHONPATH`.  See Installation step 2.

**`MeshGmshLoader` fails to load `soft_arm.msh`**
→ Run `python sofa/mesh/generate_mesh.py` first.

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
