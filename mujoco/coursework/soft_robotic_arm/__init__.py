"""
soft_robotic_arm
================
Physical parameters and MuJoCo MJCF builder for the fabric pneumatic soft arm.

Quick start
-----------
>>> from soft_robotic_arm import ArmConfig, build_arm_xml
>>> import mujoco
>>> cfg = ArmConfig()
>>> model = mujoco.MjModel.from_xml_string(build_arm_xml(cfg))
"""

from .arm_config import ArmConfig
from .mjcf_builder import build_arm_xml

__all__ = ["ArmConfig", "build_arm_xml"]
__version__ = "0.1.0"
