# SOFA Digital Twin — Soft Robotic Arm

FEM-based SOFA digital twin of the fabric pneumatic soft arm using
SoftRobots + SofaPython3. The physics model uses a tetrahedral FEM
body with continuum-mechanics parameters matched to the physical arm.

## Quick start

```bash
./run.sh      # launches SOFA GUI with the arm scene
```

Or use programmatically:

```python
from soft_arm_sim import SoftArmSim
import numpy as np

sim = SoftArmSim()
sim.set_pre_inflation(0.0)
sim.reset()

pressure = np.zeros((4, 5))   # 4 columns × 5 levels [psi]
pressure[0, 2] = 3.0           # 3 psi on East column, level 2

obs = sim.step(pressure)
print(obs["tip_pos"])          # [x, y, z] in metres
print(obs["p_actual"])         # 4×5 actual pressures [psi]
```

## Files

| File | Purpose |
|------|---------|
| `soft_arm_sim.py` | `SoftArmSim` — the digital twin: `step()`, `observe()`, `reset()` |
| `soft_arm_scene.py` | SOFA scene builder — called by SOFA on startup via `createScene()` |
| `arm_config.py` | FEM physical parameters (geometry, E, ν, mass, pneumatic τ) |
| `robot_visual.py` | Procedural visual skin mapped onto the FEM degrees of freedom |
| `mesh/soft_arm.msh` | Pre-generated GMSH tetrahedral FEM mesh |
| `run.sh` | Sets SOFA env vars and launches `runSofa` |

## Docs

| File | Contents |
|------|---------|
| `docs/sofa_integration.md` | UDP pressure protocol, unit conversions, launch env vars, motion primitive reference, mesh regeneration — extracted from removed files |

## Dependencies

SOFA framework is installed locally in `SOFA/` (not tracked in git — clone
from the SOFA GitHub releases). `run.sh` points `SOFA_ROOT` and `PYTHONPATH`
there automatically.

```bash
uv sync   # install Python dependencies (gmsh, matplotlib, numpy)
```
