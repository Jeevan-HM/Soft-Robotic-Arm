from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from pid import PIDController, PIDGains
from prc import PRCConfig


class PIDGainValidationTests(unittest.TestCase):
    def test_defaults_are_the_provisional_calibrated_plant_baseline(self):
        gains = PIDGains()

        self.assertEqual(gains.kp_psi_per_deg, 1.0)
        self.assertEqual(gains.ki_psi_per_deg_s, 0.12)
        self.assertEqual(gains.kd_psi_s_per_deg, 0.03)

    def test_gains_must_be_finite_nonnegative_and_have_unit_direction(self):
        invalid = (
            {"kp_psi_per_deg": -0.1},
            {"ki_psi_per_deg_s": -0.1},
            {"kd_psi_s_per_deg": -0.1},
            {"kp_psi_per_deg": float("nan")},
            {"control_direction": 0.0},
            {"control_direction": 2.0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PIDGains(**values)

        self.assertEqual(PIDGains(control_direction=-1.0).control_direction, -1.0)


class PIDControllerTests(unittest.TestCase):
    @staticmethod
    def config(**overrides) -> PRCConfig:
        values = {
            "control_hz": 10.0,
            "history_length": 1,
            "pressure_min_psi": 0.0,
            "pressure_max_psi": 6.0,
            "slew_rate_psi_s": 1000.0,
            "initial_command_psi": (3.0, 3.0, 3.0),
            "max_tick_s": None,
        }
        values.update(overrides)
        return PRCConfig(**values)

    @staticmethod
    def inputs(**overrides):
        values = {
            "reference_deg": 0.0,
            "measured_deg": 0.0,
            "reservoir_pressures": np.zeros(5),
            "actuator_pressures_psi": np.full(3, 3.0),
            "dt": 0.1,
        }
        values.update(overrides)
        return values

    def test_error_maps_to_segment_3_with_configured_sign(self):
        positive = PIDController(
            PIDGains(0.5, 0.0, 0.0, control_direction=1.0),
            self.config(),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )
        negative = PIDController(
            PIDGains(0.5, 0.0, 0.0, control_direction=-1.0),
            self.config(),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )

        positive_step = positive.compute(**self.inputs(reference_deg=2.0))
        negative_step = negative.compute(**self.inputs(reference_deg=2.0))

        np.testing.assert_allclose(positive_step.raw_command_psi, [3.0, 4.0, 3.0])
        np.testing.assert_allclose(negative_step.raw_command_psi, [3.0, 2.0, 3.0])
        np.testing.assert_allclose(positive_step.command_psi, [3.0, 4.0, 3.0])
        np.testing.assert_allclose(negative_step.command_psi, [3.0, 2.0, 3.0])

    def test_segments_2_and_4_remain_at_bias_during_pid_motion(self):
        controller = PIDController(
            PIDGains(0.4, 0.2, 0.05),
            self.config(),
            bias_pressure_psi=[2.5, 3.0, 3.5],
        )
        controller.reset([2.5, 3.0, 3.5])

        for reference, preview, measured in (
            (1.0, 1.5, 0.0),
            (-1.0, -1.5, 0.2),
            (0.5, 0.8, -0.1),
        ):
            step = controller.compute(**self.inputs(
                reference_deg=reference,
                preview_reference_deg=preview,
                measured_deg=measured,
            ))
            np.testing.assert_allclose(step.raw_command_psi[[0, 2]], [2.5, 3.5])
            np.testing.assert_allclose(step.command_psi[[0, 2]], [2.5, 3.5])

    def test_preview_affects_only_the_proportional_setpoint(self):
        preview_pid = PIDController(
            PIDGains(0.5, 0.0, 0.0),
            self.config(),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=True,
        )
        causal_pid = PIDController(
            PIDGains(0.5, 0.0, 0.0),
            self.config(),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )

        inputs = self.inputs(
            reference_deg=0.0,
            preview_reference_deg=2.0,
            measured_deg=0.0,
        )
        preview_step = preview_pid.compute(**inputs)
        causal_step = causal_pid.compute(**inputs)

        self.assertEqual(preview_step.error_deg, 0.0)
        self.assertEqual(causal_step.error_deg, 0.0)
        self.assertEqual(preview_step.integral_deg_s, 0.0)
        self.assertEqual(causal_step.integral_deg_s, 0.0)
        self.assertEqual(preview_step.raw_command_psi[1], 4.0)
        self.assertEqual(causal_step.raw_command_psi[1], 3.0)

    def test_projection_enforces_slew_and_hard_pressure_bounds(self):
        slew_limited = PIDController(
            PIDGains(100.0, 0.0, 0.0),
            self.config(
                pressure_min_psi=1.0,
                pressure_max_psi=4.0,
                slew_rate_psi_s=2.0,
            ),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )
        rising = slew_limited.compute(**self.inputs(reference_deg=10.0))
        self.assertTrue(rising.projected)
        np.testing.assert_allclose(rising.command_psi, [3.0, 3.2, 3.0])

        bounds_limited = PIDController(
            PIDGains(100.0, 0.0, 0.0),
            self.config(
                pressure_min_psi=1.0,
                pressure_max_psi=4.0,
                slew_rate_psi_s=1000.0,
            ),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )
        upper = bounds_limited.compute(**self.inputs(reference_deg=10.0))
        bounds_limited.reset([3.0, 3.0, 3.0])
        lower = bounds_limited.compute(**self.inputs(reference_deg=-10.0))
        np.testing.assert_allclose(upper.command_psi, [3.0, 4.0, 3.0])
        np.testing.assert_allclose(lower.command_psi, [3.0, 1.0, 3.0])

    def test_derivative_uses_measurement_and_has_no_setpoint_kick(self):
        controller = PIDController(
            PIDGains(0.0, 0.0, 1.0),
            self.config(),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )
        initial = controller.compute(**self.inputs(reference_deg=0.0, measured_deg=1.0))
        reference_jump = controller.compute(**self.inputs(
            reference_deg=10.0,
            measured_deg=1.0,
        ))
        measurement_change = controller.compute(**self.inputs(
            reference_deg=10.0,
            measured_deg=2.0,
        ))

        self.assertEqual(initial.filtered_measurement_rate_deg_s, 0.0)
        self.assertEqual(reference_jump.filtered_measurement_rate_deg_s, 0.0)
        self.assertEqual(reference_jump.raw_command_psi[1], 3.0)
        self.assertGreater(measurement_change.filtered_measurement_rate_deg_s, 0.0)
        self.assertLess(measurement_change.raw_command_psi[1], 3.0)

    def test_integral_clamps_at_configured_limit(self):
        controller = PIDController(
            PIDGains(0.0, 1.0, 0.0),
            self.config(integral_limit_deg_s=0.2),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )
        step = None
        for _ in range(10):
            step = controller.compute(**self.inputs(reference_deg=1.0))

        self.assertIsNotNone(step)
        self.assertAlmostEqual(step.integral_deg_s, 0.2)
        self.assertAlmostEqual(step.raw_command_psi[1], 3.2)

    def test_projection_backcalculates_integral_to_prevent_windup(self):
        controller = PIDController(
            PIDGains(0.0, 1.0, 0.0),
            self.config(
                pressure_max_psi=4.0,
                integral_limit_deg_s=100.0,
                antiwindup_gain=1.0,
            ),
            bias_pressure_psi=3.0,
            use_preview_for_proportional=False,
        )
        step = controller.compute(**self.inputs(reference_deg=20.0))

        self.assertTrue(step.projected)
        self.assertAlmostEqual(step.raw_command_psi[1], 5.0)
        self.assertAlmostEqual(step.command_psi[1], 4.0)
        self.assertAlmostEqual(step.integral_deg_s, 1.9)

    def test_invalid_input_uses_safe_fallback(self):
        controller = PIDController(
            config=self.config(fallback_mode="hold"),
            bias_pressure_psi=3.0,
        )
        result = controller.compute(**self.inputs(dt="invalid"))

        self.assertTrue(result.fallback)
        self.assertTrue(result.projected)
        self.assertIn("invalid_input", result.gate_reason)
        np.testing.assert_allclose(result.command_psi, 0.0)

    def test_save_load_round_trip_preserves_configuration_and_behavior(self):
        controller = PIDController(
            PIDGains(0.7, 0.3, 0.08, control_direction=-1.0),
            self.config(
                derivative_cutoff_hz=7.0,
                integral_limit_deg_s=4.0,
                antiwindup_gain=0.4,
            ),
            bias_pressure_psi=[2.5, 3.0, 3.5],
            controlled_actuator=1,
            use_preview_for_proportional=False,
            metadata={"controller": "pid", "fixed_segments": [2, 4]},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pid.json"
            returned = controller.save(path)
            restored = PIDController.load(path)

            self.assertEqual(returned, path)
            self.assertTrue(path.is_file())

        self.assertEqual(restored.gains, controller.gains)
        self.assertEqual(restored.config.control_hz, controller.config.control_hz)
        self.assertEqual(
            restored.config.derivative_cutoff_hz,
            controller.config.derivative_cutoff_hz,
        )
        self.assertEqual(
            restored.config.integral_limit_deg_s,
            controller.config.integral_limit_deg_s,
        )
        self.assertEqual(
            restored.config.antiwindup_gain,
            controller.config.antiwindup_gain,
        )
        np.testing.assert_allclose(
            restored.config.initial_command_psi,
            controller.config.initial_command_psi,
        )
        np.testing.assert_array_equal(
            restored.bias_pressure_psi,
            controller.bias_pressure_psi,
        )
        self.assertEqual(restored.controlled_actuator, 1)
        self.assertFalse(restored.use_preview_for_proportional)
        self.assertEqual(restored.metadata, controller.metadata)

        expected = controller.compute(**self.inputs(reference_deg=1.0))
        actual = restored.compute(**self.inputs(reference_deg=1.0))
        np.testing.assert_allclose(actual.command_psi, expected.command_psi)
        np.testing.assert_allclose(actual.raw_command_psi, expected.raw_command_psi)

    def test_pre_calibration_v1_artifact_is_rejected(self):
        controller = PIDController(config=self.config(), bias_pressure_psi=3.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-pid.json"
            controller.save(path)
            payload = json.loads(path.read_text())
            payload["format_version"] = 1
            path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "version 1"):
                PIDController.load(path)


if __name__ == "__main__":
    unittest.main()
