# soft-robotic-arm

`ArmConfig` and `build_arm_xml` for the fabric pneumatic soft arm coursework.

## Install

```bash
pip install soft-robotic-arm
```

Or directly from GitHub (before PyPI publication):

```bash
pip install "soft-robotic-arm @ git+https://github.com/Jeevan-HM/Soft-Robotic-Arm.git@simulation#subdirectory=mujoco/coursework"
```

## Usage

```python
import mujoco
from soft_robotic_arm import ArmConfig, build_arm_xml

cfg = ArmConfig()                                        # default calibrated parameters
model = mujoco.MjModel.from_xml_string(build_arm_xml(cfg))
print(f"{model.nq} DOF, {model.nbody} bodies")

# Modify a parameter
cfg.length = 0.25
model2 = mujoco.MjModel.from_xml_string(build_arm_xml(cfg))
```

## What's in each section of the notebook

| Library symbol | Notebook section |
|----------------|-----------------|
| `ArmConfig` | § 2 Embedded parameters |
| `build_arm_xml` | § 3 MJCF model |
| *(install cell)* | § 1 → replaced by `pip install soft-robotic-arm` |
