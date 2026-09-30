import tempfile
import unittest
from pathlib import Path

import numpy as np

from prc import PRCConfig
from rl_teacher import (
    CEMConfig,
    PARAMETER_SIZE,
    RLPolicyConfig,
    RLTeacherController,
    default_pressure_config,
    optimize_cem,
)


class CEMTests(unittest.TestCase):
    def test_optimizer_is_seeded_and_improves_best_so_far(self):
        target = np.array([0.18, -0.12, 0.07, 0.03])

        def evaluator(parameters):
            return -float(np.sum((parameters - target) ** 2))

        config = CEMConfig(
            population_size=24,
            elite_count=5,
            generations=8,
            initial_std=0.2,
            minimum_std=0.002,
            seed=41,
        )
        first = optimize_cem(evaluator, 4, config)
        second = optimize_cem(evaluator, 4, config)

        np.testing.assert_array_equal(first.best_parameters, second.best_parameters)
        np.testing.assert_array_equal(
            first.generation_best_scores, second.generation_best_scores
        )
        np.testing.assert_array_equal(
            first.best_so_far_scores, second.best_so_far_scores
        )
        self.assertTrue(np.all(np.diff(first.best_so_far_scores) >= 0.0))
        self.assertGreater(first.best_score, evaluator(np.zeros(4)))

    def test_optimizer_rejects_nonfinite_return(self):
        config = CEMConfig(population_size=2, elite_count=1, generations=1)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            optimize_cem(lambda _: np.nan, 2, config)

    def test_optimizer_retains_initial_policy_when_samples_are_worse(self):
        initial = np.array([0.25, -0.5, 0.75])

        def evaluator(parameters):
            return -float(np.sum((parameters - initial) ** 2))

        result = optimize_cem(
            evaluator,
            parameter_size=3,
            config=CEMConfig(
                population_size=6,
                elite_count=2,
                generations=3,
                initial_std=0.5,
                minimum_std=0.01,
                seed=8,
            ),
            initial_mean=initial,
        )

        np.testing.assert_array_equal(result.best_parameters, initial)
        self.assertEqual(result.best_score, 0.0)
        np.testing.assert_array_equal(result.best_so_far_scores, np.zeros(3))


class RLTeacherTests(unittest.TestCase):
    @staticmethod
    def observation():
        return dict(
            target_xyz=np.array([0.004, -0.002, 0.001]),
            current_xyz=np.zeros(3),
            preview_target_xyz=np.array([0.005, -0.001, 0.0015]),
            measured_pressures_psi=np.full((4, 5), 3.5),
        )

    def test_fixed_encoder_and_twenty_independent_outputs(self):
        pressure = PRCConfig(
            control_hz=100.0,
            history_length=1,
            n_reservoir=20,
            n_actuators=20,
            pressure_min_psi=0.0,
            pressure_max_psi=10.0,
            slew_rate_psi_s=1e6,
            initial_command_psi=tuple([3.5] * 20),
            pressure_trip_psi=10.5,
            max_tick_s=None,
            fallback_mode="hold",
        )
        controller = RLTeacherController(pressure_config=pressure)
        parameters = np.zeros(PARAMETER_SIZE)
        parameters[-20:] = np.linspace(-0.8, 0.8, 20)
        controller.set_parameter_vector(parameters)
        step = controller.compute(**self.observation())

        self.assertEqual(step.feature.shape, (13,))
        self.assertEqual(step.encoded_feature.shape, (8,))
        self.assertEqual(step.command_psi.shape, (20,))
        self.assertEqual(step.command_matrix_psi.shape, (4, 5))
        self.assertEqual(len(np.unique(np.round(step.command_psi, 8))), 20)
        self.assertFalse(step.fallback)

    def test_box_and_slew_limits_are_per_channel(self):
        controller = RLTeacherController()
        parameters = np.zeros(PARAMETER_SIZE)
        parameters[-20:] = np.linspace(-20.0, 20.0, 20)
        controller.set_parameter_vector(parameters)
        step = controller.compute(**self.observation())

        allowed = 6.0 / 100.0
        self.assertTrue(np.all(step.command_psi >= 0.0))
        self.assertTrue(np.all(step.command_psi <= 10.0))
        self.assertTrue(np.all(np.abs(step.command_psi - 3.5) <= allowed + 1e-12))
        self.assertGreater(np.ptp(step.command_psi), 0.0)

    def test_flat_and_grid_pressure_observations_are_equivalent(self):
        first = RLTeacherController()
        second = RLTeacherController()
        grid = self.observation()
        flat = {**grid, "measured_pressures_psi": np.full(20, 3.5)}

        a = first.compute(**grid)
        b = second.compute(**flat)
        np.testing.assert_allclose(a.command_psi, b.command_psi)
        np.testing.assert_allclose(a.feature, b.feature)

    def test_pressure_trip_vents_all_channels(self):
        controller = RLTeacherController()
        observation = self.observation()
        observation["measured_pressures_psi"] = np.full(20, 10.6)
        step = controller.compute(**observation)

        self.assertTrue(step.fallback)
        self.assertEqual(step.gate_reason, "pressure_trip")
        np.testing.assert_array_equal(step.command_psi, np.zeros(20))

    def test_pickle_free_save_load_preserves_policy(self):
        controller = RLTeacherController(
            policy_config=RLPolicyConfig(encoder_seed=19),
            metadata={"algorithm": "cem", "seed": 5},
        )
        rng = np.random.default_rng(8)
        controller.set_parameter_vector(rng.normal(0.0, 0.1, PARAMETER_SIZE))

        with tempfile.TemporaryDirectory() as directory:
            path = controller.save(Path(directory) / "teacher.npz")
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(int(saved["format_version"]), 1)
                self.assertNotEqual(saved["output_weights"].dtype, object)
                self.assertNotEqual(saved["metadata_json"].dtype, object)
            loaded = RLTeacherController.load(path)

        controller.reset()
        loaded.reset()
        expected = controller.compute(**self.observation())
        actual = loaded.compute(**self.observation())
        np.testing.assert_array_equal(
            controller.parameter_vector(), loaded.parameter_vector()
        )
        np.testing.assert_array_equal(
            controller.encoder_weights, loaded.encoder_weights
        )
        np.testing.assert_allclose(expected.command_psi, actual.command_psi)
        np.testing.assert_allclose(expected.feature, actual.feature)
        self.assertEqual(loaded.metadata, controller.metadata)

    def test_save_load_accepts_numpy_array_pressure_config(self):
        values = default_pressure_config().__dict__.copy()
        values.update(
            pressure_min_psi=np.zeros(20),
            pressure_max_psi=np.full(20, 10.0),
            slew_rate_psi_s=np.linspace(5.0, 7.0, 20),
            initial_command_psi=np.full(20, 3.5),
        )
        controller = RLTeacherController(
            pressure_config=PRCConfig(**values),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = controller.save(Path(directory) / "teacher_arrays.npz")
            loaded = RLTeacherController.load(path)

        np.testing.assert_array_equal(
            loaded.projector.p_min, controller.projector.p_min
        )
        np.testing.assert_array_equal(
            loaded.projector.p_max, controller.projector.p_max
        )
        np.testing.assert_array_equal(
            loaded.projector.slew, controller.projector.slew
        )

    def test_zero_width_pressure_range_is_rejected(self):
        values = default_pressure_config().__dict__.copy()
        lower = np.zeros(20)
        upper = np.full(20, 10.0)
        upper[7] = lower[7]
        values["pressure_min_psi"] = lower
        values["pressure_max_psi"] = upper

        with self.assertRaisesRegex(ValueError, "strictly greater"):
            RLTeacherController(pressure_config=PRCConfig(**values))

    def test_default_configuration_has_no_cross_channel_pressure_cap(self):
        config = default_pressure_config()
        self.assertEqual(config.n_actuators, 20)
        self.assertIsNone(config.max_total_pressure_psi)

    def test_coupled_total_pressure_limit_is_rejected(self):
        config = default_pressure_config()
        values = config.__dict__.copy()
        values["max_total_pressure_psi"] = 100.0
        with self.assertRaisesRegex(ValueError, "independent per-channel"):
            RLTeacherController(pressure_config=PRCConfig(**values))


if __name__ == "__main__":
    unittest.main()
