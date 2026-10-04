"""Backward-compatible imports for the packaged MJCF model.

New code should import from :mod:`soft_robotic_arm`.
"""

from soft_robotic_arm.model import ArmConfig, build_arm_xml

__all__ = ["ArmConfig", "build_arm_xml"]
