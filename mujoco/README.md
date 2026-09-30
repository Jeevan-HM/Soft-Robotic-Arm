# MuJoCo Soft Robotic Arm

This project provides a MuJoCo model of the fabric pneumatic soft arm. The
model represents four radial segments with five pouch levels per segment, for
a total of 20 pressure commands. Its geometry, pressure-to-force gains,
passive mechanics, and pneumatic response use the identified arm parameters.

Two entry points are provided:

- `soft_robotic_arm.ipynb` builds and demonstrates the plant model in Google
  Colab.
- `rl_run_prc.py` trains an RL teacher and distils its behaviour into a
  physical-reservoir-computing (PRC) student.

## Colab plant model

Open `soft_robotic_arm.ipynb` in Google Colab and select **Runtime > Run all**.
The notebook installs its dependencies, builds the arm, runs axial, circular,
and triangular pressure examples, and produces a simulation video.

The notebook contains the plant and motion examples only. It does not contain
a controller, controller training, or the RL-to-PRC pipeline. It is intended
as the minimum model needed for students to build and test their own control
methods.

## RL teacher and PRC student

The local training pipeline has two stages:

1. A nonlinear policy is trained with the Cross-Entropy Method (CEM). CEM is
   model-free episodic reinforcement learning here: candidate policies are
   ranked only by scalar rewards from MuJoCo rollouts. Training does not use
   PID demonstrations or inverse-model labels.
2. A 20-output PRC readout is fitted to the safe pressure commands actually
   applied by the trained RL policy. The saved PRC controller then runs on its
   own; the RL teacher is not loaded or called during PRC deployment.

The teacher and student each produce all 20 pressure commands. Pressure
measurements from all 20 pouch states support runtime safety, and the PRC
student also uses their causal history. Both controllers enforce pressure and
per-tick slew limits. The training and held-out rollouts include axial,
circular, and triangular trajectories.

### Run the complete pipeline

Install the project environment once:

```bash
uv sync
```

Then train the RL teacher, fit the PRC student, and evaluate both:

```bash
uv run python rl_run_prc.py
```

Training is seeded for reproducibility. CEM evaluates many MuJoCo episodes, so
the complete default run can take several minutes depending on the computer.

### Generated files

The pipeline writes these files under `output/`:

| File | Contents |
|------|----------|
| `rl_teacher_cem.npz` | Trained nonlinear RL teacher and its policy settings |
| `rl_distilled_prc.npz` | Reloadable 20-output PRC student |
| `rl_teacher_demonstrations.npz` | PRC features and applied RL commands used for distillation |
| `rl_prc_metrics.json` | Training and held-out evaluation settings and metrics |
| `rl_prc_comparison.png` | Held-out target, teacher, and PRC tracking comparison |

The metrics report compares the untrained policy, RL teacher, and PRC student,
including tracking error, reward, command imitation error, pressure range,
slew rate, projections, and safety fallbacks.

### Load the PRC without the RL teacher

```python
import numpy as np

from rl_student import ImitationPRCController

controller = ImitationPRCController.load("output/rl_distilled_prc.npz")

initial_pressures = np.full(20, 3.5)
controller.reset(
    initial_command_psi=initial_pressures,
    measured_pressures_psi=initial_pressures,
)

step = controller.compute(
    target_xyz_m=np.array([0.004, 0.000, 0.001]),
    preview_target_xyz_m=np.array([0.004, 0.001, 0.001]),
    measured_xyz_m=np.zeros(3),
    measured_pressures_psi=initial_pressures,
)

# Segment-by-pouch pressure command expected by the arm model.
pressure_command = step.command_psi.reshape(4, 5)
```

The caller supplies the measured tip position and the 20 measured pressures at
each control tick, then applies `pressure_command` to the plant or hardware
pressure interface.

## Tests

Run the automated checks with:

```bash
uv run python -m unittest discover -s tests
```

## Directory layout

```
mujoco/
├── mjcf_model.py               Arm configuration and MuJoCo XML generation
├── simulator.py            MuJoCo stepping, observation, and rendering interface
├── calibration.py       Calibration loader and simulator factory
├── calibration.json     Fitted model parameters
├── experiment_data.py   Synchronized robot-data loader
├── hardware.py            Physical arm interface
│
├── controllers/               Controller implementations
│   ├── pid.py      Safety-constrained PID bend controller
│   ├── prc.py      Physical Reservoir Computing controller
│   ├── rl_teacher.py       20-output RL teacher (CEM optimiser)
│   └── rl_student.py              PRC tracking features, RL imitation fit, save/load
│
├── simulations/               Training pipelines and evaluation scripts
│   ├── run_prc.py      PRC training and closed-loop evaluation
│   ├── rl_run_prc.py   End-to-end RL training + PRC distillation
│   ├── calibrate.py Validation pipeline
│   ├── compare.py PRC vs PID comparison
│   ├── visualize.py        Hardware + simulation visualisation
│   └── hardware_test.py       Pressure hardware test
│
├── tests/                     Automated unit tests
├── notebooks/                 Jupyter notebooks (Colab plant model)
├── docs/                      Reference images, diagrams, and parameter docs
└── output/                    Generated artefacts (PNGs, CSVs, NPZs, JSONs)
```
