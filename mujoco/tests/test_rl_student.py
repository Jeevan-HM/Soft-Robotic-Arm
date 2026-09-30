import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from prc import (
    FeatureNormalizer,
    InfeasiblePressureConstraints,
    PRCConfig,
)
from rl_student import (
    ImitationPRCController,
    N_PRESSURES,
    TrackingFeatureBuilder,
    TrackingFeatureConfig,
    fit_rl_imitation_readout,
)


class TrackingFeatureBuilderTests(unittest.TestCase):
    def test_feature_shape_and_newest_pressure_history(self):
        config = TrackingFeatureConfig(control_hz=10.0, history_length=2)
        builder = TrackingFeatureBuilder(config)
        builder.reset(np.zeros(N_PRESSURES))
        pressure = np.arange(N_PRESSURES, dtype=float)

        feature = builder.update(
            target_xyz_m=[0.01, 0.0, 0.0],
            preview_target_xyz_m=[0.02, 0.0, 0.0],
            measured_xyz_m=[0.0, 0.0, 0.0],
            measured_pressures_psi=pressure,
        )

        self.assertEqual(feature.shape, (config.feature_size,))
        np.testing.assert_array_equal(feature[1:1 + N_PRESSURES], pressure)
        np.testing.assert_array_equal(
            feature[1 + N_PRESSURES:1 + 2 * N_PRESSURES],
            np.zeros(N_PRESSURES),
        )


class ImitationReadoutTests(unittest.TestCase):
    def test_fits_twenty_output_linear_teacher(self):
        rng = np.random.default_rng(4)
        feature_count = 17
        features = rng.normal(size=(300, feature_count))
        features[:, 0] = 1.0
        expected = rng.normal(scale=0.1, size=(N_PRESSURES, feature_count))
        commands = features @ expected.T

        weights, normalizer = fit_rl_imitation_readout(
            features,
            commands,
            ridge=0.0,
        )
        reconstructed = normalizer.transform(features) @ weights.T

        self.assertEqual(weights.shape, (N_PRESSURES, feature_count))
        np.testing.assert_allclose(reconstructed, commands, atol=1e-9)

    def test_rejects_non_twenty_channel_labels(self):
        features = np.ones((10, 3))
        with self.assertRaisesRegex(ValueError, "samples, 20"):
            fit_rl_imitation_readout(features, np.ones((10, 19)))


class ImitationControllerTests(unittest.TestCase):
    @staticmethod
    def make_controller(
        max_tick_s: float | None = None,
        array_pressure_config: bool = False,
    ) -> ImitationPRCController:
        feature_config = TrackingFeatureConfig(
            control_hz=100.0,
            history_length=1,
        )
        pressure_min = np.zeros(N_PRESSURES) if array_pressure_config else 0.0
        pressure_max = (
            np.full(N_PRESSURES, 10.0) if array_pressure_config else 10.0
        )
        slew = np.full(N_PRESSURES, 6.0) if array_pressure_config else 6.0
        initial = np.full(N_PRESSURES, 3.5)
        pressure_config = PRCConfig(
            control_hz=100.0,
            history_length=1,
            n_actuators=N_PRESSURES,
            pressure_min_psi=pressure_min,
            pressure_max_psi=pressure_max,
            slew_rate_psi_s=slew,
            initial_command_psi=(initial if array_pressure_config else tuple(initial)),
            pressure_trip_psi=10.5,
            max_tick_s=max_tick_s,
            fallback_mode="hold",
        )
        weights = np.zeros((N_PRESSURES, feature_config.feature_size))
        weights[:, 0] = np.linspace(0.0, 12.0, N_PRESSURES)
        return ImitationPRCController(
            weights,
            pressure_config,
            feature_config,
            FeatureNormalizer.identity(feature_config.feature_size),
            metadata={"teacher": "rl"},
        )

    def test_projection_enforces_per_channel_slew_and_bounds(self):
        controller = self.make_controller()
        step = controller.compute(
            target_xyz_m=[0.0, 0.0, 0.0],
            preview_target_xyz_m=[0.0, 0.0, 0.0],
            measured_xyz_m=[0.0, 0.0, 0.0],
            measured_pressures_psi=np.full(N_PRESSURES, 3.5),
        )

        self.assertFalse(step.fallback)
        self.assertEqual(step.command_psi.shape, (N_PRESSURES,))
        self.assertTrue(np.all(step.command_psi >= 0.0))
        self.assertTrue(np.all(step.command_psi <= 10.0))
        self.assertLessEqual(
            float(np.max(np.abs(step.command_psi - 3.5))),
            0.06 + 1e-12,
        )
        self.assertGreater(np.ptp(step.raw_command_psi), 1.0)

    def test_pressure_trip_and_invalid_input_force_vent_but_stale_holds(self):
        controller = self.make_controller()
        common = dict(
            target_xyz_m=[0.0, 0.0, 0.0],
            preview_target_xyz_m=[0.0, 0.0, 0.0],
            measured_xyz_m=[0.0, 0.0, 0.0],
            measured_pressures_psi=np.full(N_PRESSURES, 3.5),
        )

        stale = controller.compute(**common, sample_age_s=1.0)
        self.assertEqual(stale.gate_reason, "stale_sensor")
        np.testing.assert_array_equal(stale.command_psi, np.full(N_PRESSURES, 3.5))

        tripped = controller.compute(
            **{**common, "measured_pressures_psi": np.full(N_PRESSURES, 10.6)}
        )
        self.assertEqual(tripped.gate_reason, "pressure_trip")
        np.testing.assert_array_equal(tripped.command_psi, np.zeros(N_PRESSURES))

        controller.reset()
        invalid = controller.compute(
            **{**common, "target_xyz_m": [np.nan, 0.0, 0.0]}
        )
        self.assertTrue(invalid.gate_reason.startswith("invalid_input:"))
        np.testing.assert_array_equal(invalid.command_psi, np.zeros(N_PRESSURES))

    @staticmethod
    def assert_feature_state_equal(test, actual, expected):
        actual_history, actual_error, actual_rate, actual_integral = actual
        expected_history, expected_error, expected_rate, expected_integral = expected
        for actual_item, expected_item in zip(actual_history, expected_history):
            np.testing.assert_array_equal(actual_item, expected_item)
        if expected_error is None:
            test.assertIsNone(actual_error)
        else:
            np.testing.assert_array_equal(actual_error, expected_error)
        np.testing.assert_array_equal(actual_rate, expected_rate)
        np.testing.assert_array_equal(actual_integral, expected_integral)

    def test_projection_failure_restores_feature_state(self):
        controller = self.make_controller()
        before = controller.features._snapshot_state()
        with patch.object(
            controller.projector,
            "project",
            side_effect=InfeasiblePressureConstraints("test"),
        ):
            step = controller.compute(
                target_xyz_m=[0.001, -0.002, 0.003],
                preview_target_xyz_m=[0.002, -0.001, 0.004],
                measured_xyz_m=[0.0, 0.0, 0.0],
                measured_pressures_psi=np.full(N_PRESSURES, 3.5),
            )

        self.assertTrue(step.fallback)
        self.assertTrue(step.gate_reason.startswith("infeasible_projection:"))
        self.assert_feature_state_equal(
            self, controller.features._snapshot_state(), before
        )

    def test_missed_deadline_restores_feature_state(self):
        controller = self.make_controller(max_tick_s=0.5)
        before = controller.features._snapshot_state()
        with patch("rl_prc.perf_counter", side_effect=[0.0, 1.0, 1.0]):
            step = controller.compute(
                target_xyz_m=[0.001, -0.002, 0.003],
                preview_target_xyz_m=[0.002, -0.001, 0.004],
                measured_xyz_m=[0.0, 0.0, 0.0],
                measured_pressures_psi=np.full(N_PRESSURES, 3.5),
            )

        self.assertTrue(step.fallback)
        self.assertEqual(step.gate_reason, "missed_deadline")
        self.assert_feature_state_equal(
            self, controller.features._snapshot_state(), before
        )

    def test_save_load_preserves_deterministic_action(self):
        controller = self.make_controller()
        inputs = dict(
            target_xyz_m=[0.001, -0.002, 0.0],
            preview_target_xyz_m=[0.002, -0.001, 0.0],
            measured_xyz_m=[0.0, 0.0, 0.0],
            measured_pressures_psi=np.full(N_PRESSURES, 3.5),
        )
        expected = controller.compute(**inputs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl_prc.npz"
            controller = self.make_controller()
            controller.save(path)
            restored = ImitationPRCController.load(path)
            actual = restored.compute(**inputs)

        np.testing.assert_array_equal(actual.command_psi, expected.command_psi)
        np.testing.assert_array_equal(
            actual.raw_command_psi,
            expected.raw_command_psi,
        )
        self.assertEqual(restored.metadata, {"teacher": "rl"})

    def test_save_load_accepts_numpy_array_pressure_config(self):
        controller = self.make_controller(array_pressure_config=True)

        with tempfile.TemporaryDirectory() as directory:
            path = controller.save(Path(directory) / "rl_prc_arrays.npz")
            restored = ImitationPRCController.load(path)

        np.testing.assert_array_equal(
            restored.projector.p_min, controller.projector.p_min
        )
        np.testing.assert_array_equal(
            restored.projector.p_max, controller.projector.p_max
        )
        np.testing.assert_array_equal(
            restored.projector.slew, controller.projector.slew
        )


if __name__ == "__main__":
    unittest.main()
