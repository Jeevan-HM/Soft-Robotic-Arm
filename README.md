# Soft Robotic Arm — Simulation

This repository contains two simulation backends for a 4-column, 5-level fabric
pneumatic soft robotic arm.

| Folder | Backend | Physics | Status |
|--------|---------|---------|--------|
| [`sofa/`](sofa/) | **SOFA + SoftRobots** | FEM continuum mechanics | ✅ Active |
| [`mujoco/`](mujoco/) | MuJoCo rigid-body | Spring-damper joints | 📦 Reference |

---

## Quick start

### SOFA (active development)

See [`sofa/README.md`](sofa/README.md) for full installation and usage.

```bash
# 1. Install SOFA binaries from https://www.sofa-framework.org/download/
# 2. Set PYTHONPATH (see sofa/README.md)
# 3. Generate the mesh
pip install gmsh
python sofa/mesh/generate_mesh.py

# 4. Run a demo
python sofa/demo.py --demo circle --plot
```

### MuJoCo (Colab-compatible reference)

```bash
cd mujoco
uv sync
uv run demo.py
```

Or open [`mujoco/soft_robotic_arm-2.ipynb`](mujoco/soft_robotic_arm-2.ipynb)
in Google Colab.

---

## Physical architecture

The arm has **4 pneumatic columns** (East / North / West / South) arranged at
the corners of a 28 mm-radius square, each with **5 inflatable levels**
stacked vertically along a 220 mm arm.

```
      N (col 1)
      |  r = 18 mm
W ----+---- E (col 0)    centre-to-column = 28 mm
      |
      S (col 3)
```

Pressurising a column elongates one side → bends the tip toward that column.

---

## Key parameters

### SOFA FEM (`sofa/arm_config.py`)

| Parameter | Value | Basis |
|---|---|---|
| Young's modulus | 0.3 MPa | Dragon Skin 10 silicone + fabric reinforcement |
| Poisson ratio | 0.45 | Near-incompressible elastomer |
| Density | 1100 kg/m³ | Silicone (1070–1200 kg/m³) |
| Max pressure | 10 psi (68.9 kPa) | Hardware limit |
| Pneumatic τ | 0.12 s | Measured from hardware step response |

### MuJoCo sysid (`mujoco/identified_params.json`)

| Parameter | Identified value |
|---|---|
| Pressure gain | 0.679 N/psi |
| Base stiffness | 0.459 N·m/rad |
| Base damping | 0.053 N·m·s/rad |
| Axial stiffness | 555 N/m |
| Mass | 0.369 kg |

---

## Repository structure

```
simulation/
├── sofa/                    ← Active: SOFA FEM simulation
│   ├── mesh/
│   │   ├── generate_mesh.py
│   │   └── soft_arm.msh     (generated — not committed)
│   ├── arm_config.py
│   ├── soft_arm_scene.py
│   ├── soft_arm_sim.py
│   ├── demo.py
│   └── README.md
│
├── mujoco/                  ← Reference: MuJoCo rigid-body simulation
│   ├── arm_model.py
│   ├── soft_arm_sim.py
│   ├── sysid.py
│   ├── identified_params.json
│   ├── soft_robotic_arm-2.ipynb
│   └── README.md
│
└── README.md                ← This file
```
