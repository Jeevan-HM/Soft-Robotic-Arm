import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rl_teacher import CEMConfig, PARAMETER_SIZE, RLTeacherController, default_pressure_config
from rl_student import TrackingFeatureConfig
from run_rl_prc import (
    PipelineConfig,
    collect_teacher_demonstrations,
    make_arm_simulator,
    make_tracking_scenario,
    rollout_controller,
    run_pipeline,
    tracking_reward,
)


class TwentyChannelPlantTests(unittest.TestCase):
    def test_default_plant_matches_notebook_pneumatics(self):
        plant = make_arm_simulator(control_hz=20.0)

        self.assertEqual(plant.cfg.p_max, 10.0)
        self.assertEqual(plant.cfg.tau_pneumatic, 0.6)
        self.assertEqual(plant.simulation.actuator_delay_s, 0.5)
        self.assertIsNone(plant.simulation.reservoir_column)
        np.testing.assert_array_equal(
            plant.simulation.actuator_pressure_gain,
            np.ones(4),
        )

    def test_public_step_requires_exact_four_by_five_matrix(self):
        plant = make_arm_simulator(control_hz=20.0, actuator_delay_s=0.0)
        plant.reset()
        with self.assertRaisesRegex(ValueError, r"shape \(4, 5\)"):
            plant.step(np.zeros(20))
        with self.assertRaisesRegex(ValueError, r"shape \(4, 5\)"):
            plant.step(np.zeros(4))

    def test_one_commanded_pouch_changes_only_its_pressure_state(self):
        plant = make_arm_simulator(control_hz=20.0, actuator_delay_s=0.0)
        plant.reset()
        command = np.zeros((4, 5))
        command[2, 3] = 7.0

        observation = plant.step(command)
        changed = np.argwhere(observation["p_actual"] > 1e-12)

        np.testing.assert_array_equal(changed, np.array([[2, 3]]))
        self.assertGreater(observation["p_actual"][2, 3], 0.0)
        log = plant.get_pressure_log()
        np.testing.assert_array_equal(log["p_cmd"][0], command)


class TrackingScenarioTests(unittest.TestCase):
    def test_scenario_contains_all_three_motion_examples_and_preview(self):
        scenario = make_tracking_scenario(
            seconds_per_motion=0.2,
            control_hz=20.0,
            preview_s=0.1,
            seed=14,
        )

        self.assertEqual(scenario.target_xyz_m.shape, (12, 3))
        self.assertEqual(scenario.preview_xyz_m.shape, (12, 3))
        self.assertEqual(set(scenario.motion), {"axial", "circular", "triangular"})
        np.testing.assert_array_equal(
            scenario.preview_xyz_m[:-2],
            scenario.target_xyz_m[2:],
        )
        self.assertLess(float(np.min(scenario.target_xyz_m[:4, 2])), 0.0)
        self.assertGreater(float(np.ptp(scenario.target_xyz_m[4:8, 0])), 0.0)
        self.assertGreater(float(np.ptp(scenario.target_xyz_m[8:, 1])), 0.0)

    def test_reward_prefers_matching_tip_position(self):
        command = np.full(20, 3.5)
        matched = tracking_reward(
            np.zeros(3), np.zeros(3), command, command, command, dt=0.01
        )
        missed = tracking_reward(
            np.zeros(3), np.array([0.01, 0.0, 0.0]), command, command, command, dt=0.01
        )
        self.assertTrue(np.isfinite(matched))
        self.assertGreater(matched, missed)


class DistillationIntegrationTests(unittest.TestCase):
    def test_demonstration_labels_are_applied_teacher_commands(self):
        control_hz = 20.0
        teacher = RLTeacherController(
            pressure_config=replace(default_pressure_config(), control_hz=control_hz)
        )
        parameters = np.zeros(PARAMETER_SIZE)
        parameters[-20:] = np.linspace(-0.4, 0.4, 20)
        teacher.set_parameter_vector(parameters)
        scenario = make_tracking_scenario(
            seconds_per_motion=0.05,
            control_hz=control_hz,
            preview_s=0.5,
            seed=3,
        )
        feature_config = TrackingFeatureConfig(
            control_hz=control_hz,
            history_length=2,
        )

        demonstrations = collect_teacher_demonstrations(
            teacher,
            [scenario],
            feature_config,
            control_hz=control_hz,
            actuator_delay_s=0.5,
            settle_s=0.01,
            seed=8,
        )
        rollout = rollout_controller(
            teacher,
            scenario,
            plant_seed=8,
            control_hz=control_hz,
            actuator_delay_s=0.5,
            settle_s=0.01,
        )

        self.assertEqual(
            demonstrations.features.shape,
            (scenario.sample_count, feature_config.feature_size),
        )
        self.assertEqual(demonstrations.teacher_commands_psi.shape, (12, 20))
        np.testing.assert_allclose(
            demonstrations.teacher_commands_psi,
            rollout.command_psi,
            atol=0.0,
            rtol=0.0,
        )

    def test_tiny_pipeline_saves_loadable_artifacts(self):
        config = PipelineConfig(
            seed=9,
            control_hz=20.0,
            actuator_delay_s=0.5,
            preview_s=0.5,
            seconds_per_motion=0.05,
            settle_s=0.01,
            training_scenarios=1,
            history_length=2,
            ridge=1e-2,
            cem=CEMConfig(
                population_size=2,
                elite_count=1,
                generations=1,
                initial_std=0.02,
                minimum_std=0.01,
                seed=9,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_pipeline(config, directory)
            paths = (
                result.artifacts.teacher_path,
                result.artifacts.student_path,
                result.artifacts.demonstrations_path,
                result.artifacts.metrics_path,
                result.artifacts.figure_path,
            )
            for path in paths:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0)

            restored_teacher = RLTeacherController.load(result.artifacts.teacher_path)
            restored_student = result.student.load(result.artifacts.student_path)
            payload = json.loads(Path(result.artifacts.metrics_path).read_text())
            with np.load(result.artifacts.demonstrations_path, allow_pickle=False) as saved:
                self.assertEqual(saved["teacher_commands_psi"].shape[1], 20)

        self.assertEqual(restored_teacher.parameter_vector().shape, (PARAMETER_SIZE,))
        self.assertEqual(restored_student.weights.shape[0], 20)
        self.assertIn("untrained_baseline", payload["metrics"])
        self.assertIn("teacher", payload["metrics"])
        self.assertIn("prc_student", payload["metrics"])
        self.assertEqual(result.teacher_rollout.command_psi.shape[1], 20)
        self.assertEqual(result.student_rollout.command_psi.shape[1], 20)


if __name__ == "__main__":
    unittest.main()
