"""Backward-compatible imports for the packaged calibration API.

New code should import from :mod:`soft_robotic_arm`.
"""

from soft_robotic_arm.calibration import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CALIBRATION_RESOURCE,
    RobotCalibration,
    SUPPORTED_TOPOLOGIES,
)

__all__ = [
    "DEFAULT_CALIBRATION_PATH",
    "DEFAULT_CALIBRATION_RESOURCE",
    "RobotCalibration",
    "SUPPORTED_TOPOLOGIES",
]
