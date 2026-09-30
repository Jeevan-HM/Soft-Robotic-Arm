# Soft Robotic Arm — Calibrated MuJoCo Simulation

This repository uses one simulation model: the robot-data-calibrated MuJoCo
digital twin in [`mujoco/`](mujoco/). The model represents the hanging fabric
pneumatic arm as five axial/bending levels and includes the measured actuator
delay, pressure mapping, pneumatic lag, and sealed-reservoir response.

The calibration is fitted to the 36 public robot experiments in
[`data/_data_extract`](https://github.com/Jeevan-HM/Soft-Robotic-Arm/tree/main/data/_data_extract).

## Quick start

```bash
cd mujoco
uv sync
```

The default constructor is already the calibrated physical robot:

```python
from simulator import SoftArmSim

sim = SoftArmSim()                       # parallel plumbing, 2 psi charge
observation = sim.step([2.0, 5.0, 2.0]) # Segment 2, 3, and 4 pressures

# Reproduce the coupled-plumbing trials when needed.
coupled = SoftArmSim(topology="coupled", reservoir_pressure_psi=3.0)
```

Segment 1 is a charged, sealed five-pouch reservoir. Segments 2–4 are the only
runtime actuators, so commands always contain three absolute pressure
setpoints. The calibrated rest length is 0.296 m and the model pressure limit
is 11 psi; the recorded validation range uses 1–3 psi reservoir charges,
active-command peaks of 5 or 10 psi, and 0.1 Hz excitation.

Across all 36 experiments, the calibrated mechanics replay has 3.03 mm mean
lateral and 1.07 mm mean axial error. Full desired-command replay, including
actuator and reservoir dynamics, has 3.34 mm lateral, 1.17 mm axial,
0.44 psi active-pressure, and 0.09 psi reservoir-pressure error. These are
centered held-cycle metrics for motion pattern and amplitude.

See [`mujoco/README.md`](mujoco/README.md) for calibration details, validation,
controller prototypes, demos, and file-level documentation.

## Main files

```text
simulation/
├── mujoco/                         MuJoCo digital twin (main project)
│   ├── mjcf_model.py                MJCF model + calibrated mechanics
│   ├── simulator.py             canonical step/observe/render interface
│   ├── calibration.json      fitted model parameters
│   ├── calibration.py        strict loader and simulator factory
│   ├── experiment_data.py    synchronized robot-data loader
│   ├── hardware.py             physical arm interface
│   │
│   ├── controllers/                controller implementations
│   │   ├── pid.py
│   │   ├── prc.py
│   │   ├── rl_teacher.py
│   │   └── rl_student.py
│   │
│   ├── simulations/                training pipelines + evaluation
│   │   ├── run_prc.py
│   │   ├── run_rl_prc.py
│   │   ├── calibrate.py
│   │   ├── compare.py
│   │   ├── visualize.py
│   │   └── hardware_test.py
│   │
│   ├── tests/                      automated unit tests
│   ├── notebooks/                  Jupyter notebooks
│   ├── docs/                       images, diagrams, parameter docs
│   ├── output/                     generated artefacts (PNGs, CSVs, NPZs)
│   └── README.md                   complete usage guide
│
├── soft-arm/                       SOFA-based simulation (separate project)
│   ├── soft_arm_scene.py
│   ├── simulator.py
│   ├── arm_config.py
│   ├── pressure_control.py
│   ├── pressure_panel_app.py
│   ├── robot_visual.py
│   ├── mesh/                       geometry files
│   ├── SOFA/                       SOFA framework install
│   ├── tests/
│   └── README.md
│
└── README.md
```
