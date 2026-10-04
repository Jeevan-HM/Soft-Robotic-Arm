"""Calibrated MuJoCo digital twin of the pneumatic soft robotic arm.

The public plant has three controller inputs: absolute pressure setpoints for
Segments 2, 3, and 4.  Segment 1 is charged once at construction and remains a
sealed five-pouch reservoir throughout an experiment.
"""

from .calibration import RobotCalibration
from .evaluation import (
    CircularTrackingTask,
    TrackingResult,
    evaluate_controller,
    make_reference,
)
from .model import ArmConfig, build_arm_xml
from .simulator import SoftArmSim

__version__ = "0.2.0"


def make_sim(
    *,
    reservoir_pressure_psi=2.0,
    control_hz: float = 100.0,
    topology: str = "parallel",
    seed: int | None = 0,
    **overrides,
) -> SoftArmSim:
    """Construct the calibrated three-input teaching simulator.

    Parameters are keyword-only to keep classroom experiments reproducible.
    ``overrides`` is intended for documented calibration/sensor parameters;
    normal controller code only needs the first four arguments.
    """
    return RobotCalibration.load().make_sim(
        topology=topology,
        reservoir_pressure_psi=reservoir_pressure_psi,
        control_hz=control_hz,
        seed=seed,
        **overrides,
    )


__all__ = [
    "ArmConfig",
    "CircularTrackingTask",
    "RobotCalibration",
    "SoftArmSim",
    "TrackingResult",
    "build_arm_xml",
    "evaluate_controller",
    "make_reference",
    "make_sim",
]
