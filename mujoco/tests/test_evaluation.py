import numpy as np
import pytest

from soft_robotic_arm import (
    CircularTrackingTask,
    evaluate_controller,
    make_calibrated_sim,
    make_reference,
)
from soft_robotic_arm.evaluation import _phase_lag_seconds


class RecordingController:
    def __init__(self, command=(5.0, 5.0, 5.0)):
        self.command = np.asarray(command, dtype=float)
        self.calls = []

    def compute(self, t, obs, ref):
        self.calls.append((t, float(obs["time"]), np.asarray(ref).copy()))
        return self.command


def short_task(**overrides):
    values = {
        "control_hz": 100.0,
        "settle_s": 0.02,
        "track_s": 0.03,
        "radius_m": 0.001,
        "frequency_hz": 1.0,
        "baseline_psi": (5.0, 5.0, 5.0),
    }
    values.update(overrides)
    return CircularTrackingTask(**values)


def test_evaluator_uses_monotonic_plant_time_and_tracking_relative_arrays():
    controller = RecordingController()
    task = short_task()
    result = evaluate_controller(controller, task=task)

    np.testing.assert_allclose(result.times, [0.0, 0.01, 0.02])
    assert result.tips.shape == (3, 3)
    assert result.references.shape == (3, 3)
    assert result.commands.shape == (3, 3)
    for controller_time, observation_time, _ in controller.calls:
        assert controller_time == pytest.approx(observation_time)
    assert controller.calls[0][0] == pytest.approx(task.settle_s)
    assert set(result.as_dict()) >= {
        "rmse_mm",
        "max_error_mm",
        "phase_lag_s",
        "effort_psi",
    }


def test_evaluator_validates_shape_and_clips_to_coursework_limit():
    sim = make_calibrated_sim()
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        evaluate_controller(
            RecordingController(command=(1.0, 2.0)),
            task=short_task(),
            sim=sim,
        )

    result = evaluate_controller(
        RecordingController(command=(-1.0, 20.0, 4.0)),
        task=short_task(command_limit_psi=9.0),
        sim=make_calibrated_sim(),
    )
    np.testing.assert_allclose(result.commands, [[0.0, 9.0, 4.0]] * 3)


def test_positive_phase_lag_for_delayed_signal():
    sample_hz = 100.0
    delay_samples = 7
    t = np.arange(1000) / sample_hz
    reference = np.sin(2.0 * np.pi * 1.0 * t)
    measured = np.concatenate(
        [np.zeros(delay_samples), reference[:-delay_samples]]
    )

    assert _phase_lag_seconds(reference, measured, sample_hz) == pytest.approx(
        delay_samples / sample_hz
    )

    bounded = _phase_lag_seconds(
        reference, measured, sample_hz, max_lag_s=0.5
    )
    assert bounded == pytest.approx(delay_samples / sample_hz)
    assert abs(bounded) <= 0.5


def test_phase_lag_is_zero_when_measured_motion_is_negligible():
    t = np.arange(1000) / 100.0
    reference = np.sin(2.0 * np.pi * t)
    sensor_drift = 1e-6 * t

    assert _phase_lag_seconds(reference, sensor_drift, 100.0) == 0.0


def test_reference_starts_on_negative_x_side_of_circle():
    home = np.array([0.1, -0.2, 0.3])
    task = short_task(radius_m=0.008)

    np.testing.assert_allclose(
        make_reference(home, 0.0, task),
        home + np.array([-task.radius_m, 0.0, 0.0]),
        atol=1e-15,
    )
