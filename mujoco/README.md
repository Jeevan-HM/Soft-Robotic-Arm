# MuJoCo Simulation (Legacy / Reference)

This folder contains the original **rigid-body approximation** of the soft robotic arm
built with [MuJoCo](https://mujoco.org/).

## Why "legacy"?

The MuJoCo model treats the arm as 5 rigid links connected by hinge and slide joints
with explicit spring-damper terms. This is fast and easy to work with, but it
fundamentally differs from how a continuum elastic body actually deforms:

| Property | This model (MuJoCo) | SOFA FEM model |
|---|---|---|
| Deformation | Rigid links + joints | Continuum FEM (tetrahedral mesh) |
| Material law | Spring-damper constants | Young's modulus + Poisson ratio |
| Actuation | Generalised forces on DOFs | `SurfacePressureConstraint` |
| Pre-inflation | Stiffness scaling factor | Natural cavity pressure |
| Accuracy | Good for fast control loops | Closer to real physics |

## Status

All files are preserved exactly as they were. The system-identified parameters
(`identified_params.json`) from the sysid run on `sysid_data.csv` are baked into
`soft_robotic_arm-2.ipynb`.

The notebook (`soft_robotic_arm-2.ipynb`) runs in **Google Colab** — no local
install needed.

## Files

| File | Purpose |
|------|---------|
| `arm_model.py` | `ArmConfig` dataclass + MJCF XML generator |
| `soft_arm_sim.py` | `SoftArmSim` wrapper (step/observe/render API) |
| `sysid.py` | System identification (Nelder-Mead trajectory fitting) |
| `collect_sysid_data.py` | Hardware data collection for sysid |
| `demo.py` | Four motion demos (bend, axial, circle, triangle) |
| `digital_twin.py` | Real-time digital twin (hardware + sim side-by-side) |
| `hardware_arm.py` | Physical arm interface |
| `controller_example.py` | Example closed-loop controller |
| `pressure_test.py` | Hardware pressure tests |
| `identified_params.json` | Sysid-identified physical parameters |
| `sysid_data.csv` | 8 s of OptiTrack + pressure data used for sysid |
| `soft_robotic_arm-2.ipynb` | Colab notebook (uses identified params) |
| `arm_parameters.md` | Detailed parameter reference |

## For the new SOFA-based simulation

See [`../sofa/README.md`](../sofa/README.md).
