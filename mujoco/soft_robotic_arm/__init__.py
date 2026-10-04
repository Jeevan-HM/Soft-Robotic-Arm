"""MuJoCo teaching model and calibrated digital twin of the soft arm."""

from .calibration import RobotCalibration
from .evaluation import (
    CircularTrackingTask,
    TrackingResult,
    evaluate_controller,
    make_reference,
)
from .model import ArmConfig, build_arm_xml
from .simulator import SoftArmSim

__version__ = "0.3.0"


def make_sim(
    *,
    control_hz: float = 100.0,
    seed: int | None = 0,
    **overrides,
) -> SoftArmSim:
    """Construct the barebones four-segment coursework simulator.

    ``step([p1, p2, p3, p4])`` commands Segments 1--4 directly. Each scalar
    setpoint is broadcast to that segment's five pouches. The arm mechanics
    and pouch-sensor response come from the measured calibration, while the
    teaching command interface is deliberately symmetric and leaves controller
    design to the student.
    """
    calibration = RobotCalibration.load()
    measured = calibration.simulator_kwargs("parallel", 0.0)
    teaching_defaults = {
        "sensor_noise_psi": measured["sensor_noise_psi"],
        "curvature_coupling": measured["curvature_coupling"],
        "extension_coupling": measured["extension_coupling"],
    }
    teaching_defaults.update(overrides)
    return SoftArmSim(
        cfg=calibration.make_arm_config(),
        control_hz=control_hz,
        seed=seed,
        reservoir_column=None,
        strict_segment_commands=True,
        **teaching_defaults,
    )


def make_calibrated_sim(
    *,
    reservoir_pressure_psi=2.0,
    control_hz: float = 100.0,
    topology: str = "parallel",
    seed: int | None = 0,
    **overrides,
) -> SoftArmSim:
    """Construct the measured three-actuator plant with sealed Segment 1."""
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
    "make_calibrated_sim",
    "make_reference",
    "make_sim",
]
