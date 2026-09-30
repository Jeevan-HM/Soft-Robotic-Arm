import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prc import (
    FeatureNormalizer,
    PRCConfig,
    PRCController,
    PRCFeatureBuilder,
    PressureProjector,
    fit_ridge_readout,
    quaternion_relative_rotvec,
    signed_bend_angle_deg,
)


class RidgeReadoutTests(unittest.TestCase):
    def test_twenty_output_fit_returns_expected_shapes(self):
        rng = np.random.default_rng(42)
        features = np.column_stack((np.ones(32), rng.normal(size=(32, 7))))
        targets = rng.normal(size=(32, 20))

        weights, normalizer = fit_ridge_readout(features, targets)

        self.assertEqual(weights.shape, (20, features.shape[1]))
        self.assertEqual(normalizer.mean.shape, (features.shape[1] - 1,))
        self.assertEqual(normalizer.scale.shape, (features.shape[1] - 1,))

    def test_readout_rejects_invalid_target_shapes(self):
        features = np.ones((4, 3))
        invalid_targets = (
            np.ones(4),
            np.ones((3, 20)),
            np.empty((4, 0)),
        )

        for targets in invalid_targets:
            with self.subTest(shape=targets.shape):
                with self.assertRaises(ValueError):
                    fit_ridge_readout(features, targets)

    def test_twenty_output_fit_reconstructs_linear_targets(self):
        rng = np.random.default_rng(7)
        features = np.column_stack((np.ones(128), rng.normal(size=(128, 6))))
        expected_weights = rng.normal(size=(20, features.shape[1]))
        targets = features @ expected_weights.T

        weights, normalizer = fit_ridge_readout(features, targets, ridge=0.0)
        reconstructed = normalizer.transform(features) @ weights.T

        np.testing.assert_allclose(reconstructed, targets, rtol=1e-11, atol=1e-11)


class PRCFeatureBuilderTests(unittest.TestCase):
    def test_feature_dimension_order_and_causal_history(self):
        config = PRCConfig(
            control_hz=10.0,
            history_length=2,
            n_reservoir=2,
            derivative_cutoff_hz=1.0,
            integral_limit_deg_s=100.0,
        )
        builder = PRCFeatureBuilder(config)

        first = builder.update(
            reservoir_pressures=[1.0, 2.0],
            reference_deg=10.0,
            preview_reference_deg=11.0,
            measured_deg=7.0,
            actuator_pressures_psi=[4.0, 5.0, 6.0],
            dt=0.1,
        )
        first_snapshot = first.copy()

        self.assertEqual(config.feature_size, 12)
        self.assertEqual(first.shape, (config.feature_size,))
        np.testing.assert_allclose(
            first,
            [
                1.0,
                1.0, 2.0,
                1.0, 2.0,
                11.0, 3.0, 0.0, 0.3,
                4.0, 5.0, 6.0,
            ],
        )

        second = builder.update(
            reservoir_pressures=[7.0, 8.0],
            reference_deg=12.0,
            preview_reference_deg=13.0,
            measured_deg=8.0,
            actuator_pressures_psi=[9.0, 10.0, 11.0],
            dt=0.1,
        )
        alpha = 0.1 / (1.0 / (2.0 * math.pi) + 0.1)
        expected_derivative = alpha * 10.0
        np.testing.assert_allclose(
            second,
            [
                1.0,
                7.0, 8.0,
                1.0, 2.0,
                13.0, 4.0, expected_derivative, 0.7,
                9.0, 10.0, 11.0,
            ],
        )
        # Later samples must not mutate a feature vector already handed to the
        # readout at an earlier control tick.
        np.testing.assert_array_equal(first, first_snapshot)

    def test_previous_command_affects_only_projection_not_readout_features(self):
        config = PRCConfig(
            control_hz=10.0,
            history_length=1,
            slew_rate_psi_s=1.0,
            max_tick_s=None,
        )
        weights = np.zeros((config.n_actuators, config.feature_size))
        weights[:, 0] = 9.0
        from_zero = PRCController(weights, config)
        from_five = PRCController(weights, config)
        from_zero.reset(initial_command_psi=0.0)
        from_five.reset(initial_command_psi=5.0)

        inputs = dict(
            reference_deg=5.0,
            measured_deg=1.0,
            reservoir_pressures=[1.0, 2.0, 3.0, 4.0, 5.0],
            actuator_pressures_psi=[2.0, 3.0, 4.0],
            preview_reference_deg=6.0,
        )
        low = from_zero.compute(**inputs)
        high = from_five.compute(**inputs)

        np.testing.assert_array_equal(low.feature, high.feature)
        np.testing.assert_array_equal(low.raw_command_psi, high.raw_command_psi)
        np.testing.assert_allclose(low.command_psi, 0.1)
        np.testing.assert_allclose(high.command_psi, 5.1)


class PressureProjectorTests(unittest.TestCase):
    def test_box_bounds_and_slew_limits(self):
        config = PRCConfig(
            pressure_min_psi=(1.0, 2.0, 0.0),
            pressure_max_psi=(5.0, 6.0, 7.0),
            slew_rate_psi_s=(10.0, 20.0, 30.0),
        )
        result = PressureProjector(config).project(
            raw_command_psi=[-5.0, 10.0, 3.5],
            previous_command_psi=[3.0, 3.0, 3.0],
            dt=0.1,
        )

        np.testing.assert_allclose(result.command_psi, [2.0, 5.0, 3.5])
        np.testing.assert_allclose(result.pressure_ceiling_psi, [5.0, 6.0, 7.0])
        self.assertTrue(result.projected)
        self.assertFalse(result.slew_relaxed)

    def test_sudden_supply_ceiling_wins_over_slew(self):
        config = PRCConfig(
            pressure_max_psi=9.0,
            slew_rate_psi_s=1.0,
        )
        result = PressureProjector(config).project(
            raw_command_psi=[9.0, 9.0, 9.0],
            previous_command_psi=[8.0, 8.0, 8.0],
            dt=0.1,
            supply_ceiling_psi=[4.0, 5.0, 6.0],
        )

        np.testing.assert_allclose(result.command_psi, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(result.pressure_ceiling_psi, [4.0, 5.0, 6.0])
        self.assertTrue(result.projected)
        self.assertTrue(result.slew_relaxed)

    def test_ceiling_drop_preserves_unaffected_channel_slew(self):
        config = PRCConfig(pressure_max_psi=9.0, slew_rate_psi_s=1.0)
        result = PressureProjector(config).project(
            raw_command_psi=[9.0, 9.0, 9.0],
            previous_command_psi=[8.0, 0.0, 0.0],
            dt=0.1,
            supply_ceiling_psi=[4.0, 9.0, 9.0],
        )

        np.testing.assert_allclose(result.command_psi, [4.0, 0.1, 0.1])
        self.assertTrue(result.slew_relaxed)

    def test_coupled_total_pressure_projection(self):
        config = PRCConfig(
            pressure_min_psi=1.0,
            pressure_max_psi=10.0,
            slew_rate_psi_s=100.0,
            max_total_pressure_psi=6.0,
        )
        result = PressureProjector(config).project(
            raw_command_psi=[5.0, 4.0, 3.0],
            previous_command_psi=[2.0, 2.0, 2.0],
            dt=1.0,
        )

        np.testing.assert_allclose(result.command_psi, [3.0, 2.0, 1.0], atol=1e-10)
        self.assertAlmostEqual(float(result.command_psi.sum()), 6.0, places=10)
        self.assertTrue(result.projected)
        self.assertFalse(result.slew_relaxed)


class PRCControllerStateTests(unittest.TestCase):
    def test_projection_backcalculates_integrator_update(self):
        config = PRCConfig(
            control_hz=10.0,
            history_length=1,
            integral_limit_deg_s=100.0,
            pressure_max_psi=9.0,
            slew_rate_psi_s=1000.0,
            initial_command_psi=1.0,
            max_tick_s=None,
        )
        weights = np.zeros((config.n_actuators, config.feature_size))
        weights[:, config.integral_feature_index] = 2.0
        controller = PRCController(weights, config)
        inputs = dict(
            reference_deg=100.0,
            measured_deg=0.0,
            reservoir_pressures=np.zeros(config.n_reservoir),
            actuator_pressures_psi=np.zeros(config.n_actuators),
        )

        limited = controller.compute(**inputs)
        self.assertTrue(limited.projected)
        np.testing.assert_allclose(limited.command_psi, 9.0)
        # eta first reaches 10 deg*s. d(p_raw)/d(eta)=2 psi/(deg*s),
        # so the projected 20->9 psi discrepancy back-calculates -5.5 deg*s.
        self.assertAlmostEqual(controller.features.integral, 4.5)

    def test_save_load_round_trip(self):
        config = PRCConfig(
            control_hz=50.0,
            history_length=2,
            pressure_min_psi=(0.1, 0.2, 0.3),
            pressure_max_psi=(7.0, 8.0, 9.0),
            slew_rate_psi_s=(4.0, 5.0, 6.0),
            max_total_pressure_psi=18.0,
            initial_command_psi=(1.0, 2.0, 3.0),
            max_tick_s=None,
        )
        weights = np.arange(
            config.n_actuators * config.feature_size, dtype=float
        ).reshape(config.n_actuators, config.feature_size) / 100.0
        normalizer = FeatureNormalizer(
            mean=np.linspace(-1.0, 1.0, config.feature_size - 1),
            scale=np.linspace(0.5, 2.0, config.feature_size - 1),
        )
        controller = PRCController(
            weights,
            config,
            normalizer,
            metadata={
                "reservoir_column": 0,
                "actuator_columns": [1, 2, 3],
                "reference_preview_steps": 12,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.npz"
            returned = controller.save(path)
            restored = PRCController.load(path)

            self.assertEqual(returned, path)
            self.assertTrue(path.is_file())

        np.testing.assert_array_equal(restored.weights, controller.weights)
        np.testing.assert_array_equal(
            restored.normalizer.mean, controller.normalizer.mean
        )
        np.testing.assert_array_equal(
            restored.normalizer.scale, controller.normalizer.scale
        )
        self.assertEqual(restored.config.control_hz, config.control_hz)
        self.assertEqual(restored.config.history_length, config.history_length)
        self.assertEqual(
            list(restored.config.pressure_min_psi),
            list(config.pressure_min_psi),
        )
        self.assertEqual(restored.config.max_total_pressure_psi, 18.0)
        np.testing.assert_allclose(restored.previous_command, [1.0, 2.0, 3.0])
        self.assertEqual(restored.metadata, controller.metadata)

    def test_fallback_obeys_reduced_supply_and_keeps_fresh_measurement(self):
        config = PRCConfig(
            history_length=1,
            pressure_min_psi=1.0,
            pressure_max_psi=9.0,
            initial_command_psi=8.0,
            fallback_mode="hold",
            max_tick_s=None,
        )
        weights = np.zeros((3, config.feature_size))
        controller = PRCController(weights, config)

        # This reaches projection after updating the feature state, then finds
        # the hard set empty because the supplied ceiling is below p_min.
        result = controller.compute(
            reference_deg=10.0,
            measured_deg=0.0,
            reservoir_pressures=np.zeros(5),
            actuator_pressures_psi=np.ones(3),
            supply_ceiling_psi=0.5,
        )

        self.assertTrue(result.fallback)
        np.testing.assert_allclose(result.command_psi, 0.0)
        self.assertLessEqual(float(result.command_psi.max()), 0.5)
        self.assertAlmostEqual(controller.features.integral, 0.0)
        self.assertAlmostEqual(controller.features.previous_error, 10.0)
        self.assertEqual(len(controller.features._history), 1)

    def test_save_without_npz_suffix_uses_exact_path(self):
        config = PRCConfig(history_length=1, max_tick_s=None)
        controller = PRCController(np.zeros((3, config.feature_size)), config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.weights"
            returned = controller.save(path)
            self.assertEqual(returned, path)
            self.assertTrue(path.is_file())
            PRCController.load(path)

    def test_invalid_supply_ceiling_forces_vent_even_in_hold_mode(self):
        config = PRCConfig(
            history_length=1,
            initial_command_psi=4.0,
            fallback_mode="hold",
            max_tick_s=None,
        )
        controller = PRCController(np.zeros((3, config.feature_size)), config)
        result = controller.compute(
            reference_deg=0.0,
            measured_deg=0.0,
            reservoir_pressures=np.zeros(5),
            actuator_pressures_psi=np.zeros(3),
            sample_age_s=1.0,
            supply_ceiling_psi=float("nan"),
        )
        self.assertTrue(result.fallback)
        np.testing.assert_allclose(result.command_psi, 0.0)

    def test_invalid_dt_returns_fallback_instead_of_raising(self):
        config = PRCConfig(history_length=1, max_tick_s=None)
        controller = PRCController(np.zeros((3, config.feature_size)), config)
        result = controller.compute(
            reference_deg=0.0,
            measured_deg=0.0,
            reservoir_pressures=np.zeros(5),
            actuator_pressures_psi=np.zeros(3),
            dt="not-a-number",
        )
        self.assertTrue(result.fallback)
        self.assertIn("invalid_input", result.gate_reason)

    def test_vent_fallback_means_zero_even_with_nonzero_operating_minimum(self):
        config = PRCConfig(
            history_length=1,
            pressure_min_psi=1.5,
            initial_command_psi=4.0,
            fallback_mode="vent",
            max_tick_s=None,
        )
        controller = PRCController(np.zeros((3, config.feature_size)), config)
        result = controller.compute(
            reference_deg=0.0,
            measured_deg=0.0,
            reservoir_pressures=np.zeros(5),
            actuator_pressures_psi=np.full(3, 2.0),
            sample_age_s=1.0,
        )

        self.assertTrue(result.fallback)
        np.testing.assert_allclose(result.command_psi, 0.0)

    def test_pressure_trip_forces_vent_even_in_hold_mode(self):
        config = PRCConfig(
            history_length=1,
            initial_command_psi=4.0,
            pressure_trip_psi=8.0,
            fallback_mode="hold",
            max_tick_s=None,
        )
        controller = PRCController(np.zeros((3, config.feature_size)), config)
        result = controller.compute(
            reference_deg=0.0,
            measured_deg=0.0,
            reservoir_pressures=[9.0, 0.0, 0.0, 0.0, 0.0],
            actuator_pressures_psi=np.zeros(3),
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.gate_reason, "pressure_trip")
        np.testing.assert_allclose(result.command_psi, 0.0)

    def test_stale_overpressure_still_forces_vent(self):
        config = PRCConfig(
            history_length=1,
            initial_command_psi=4.0,
            pressure_trip_psi=8.0,
            stale_after_s=0.03,
            fallback_mode="hold",
            max_tick_s=None,
        )
        controller = PRCController(np.zeros((3, config.feature_size)), config)
        result = controller.compute(
            reference_deg=0.0,
            measured_deg=0.0,
            reservoir_pressures=np.zeros(5),
            actuator_pressures_psi=[0.0, 9.0, 0.0],
            sample_age_s=0.10,
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.gate_reason, "pressure_trip")
        np.testing.assert_allclose(result.command_psi, 0.0)


class QuaternionTests(unittest.TestCase):
    def test_quaternion_sign_does_not_change_relative_rotation(self):
        angle = math.radians(70.0)
        current = np.array([math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0])
        home = np.array([1.0, 0.0, 0.0, 0.0])

        expected = np.array([angle, 0.0, 0.0])
        np.testing.assert_allclose(
            quaternion_relative_rotvec(current, home), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            quaternion_relative_rotvec(-current, home), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            quaternion_relative_rotvec(current, -home), expected, atol=1e-12
        )
        self.assertAlmostEqual(
            signed_bend_angle_deg(-current, home, bend_axis_xy=(1.0, 0.0)),
            70.0,
            places=10,
        )
        np.testing.assert_array_equal(
            current,
            [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0],
        )
        np.testing.assert_array_equal(home, [1.0, 0.0, 0.0, 0.0])

    def test_nonfinite_safety_configuration_is_rejected(self):
        for keyword in (
            "control_hz",
            "derivative_cutoff_hz",
            "integral_limit_deg_s",
            "antiwindup_gain",
            "stale_after_s",
            "max_tick_s",
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    PRCConfig(**{keyword: float("nan")})


if __name__ == "__main__":
    unittest.main()
