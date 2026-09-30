import unittest
from pathlib import Path

import numpy as np

from prc import PRCConfig
from run_prc import (
    ClosedLoopData,
    DelayAffinePredictor,
    PredictorReport,
    build_argument_parser,
    calibrated_plant_metadata,
    evaluate_student_acceptance,
    generate_inverse_control_labels,
    make_demo_reference,
    reachable_reference_limit,
    tracking_metrics,
    run_pipeline,
)


class ReferenceEnvelopeTests(unittest.TestCase):
    def test_symmetric_limit_never_expands_observed_envelope(self):
        bend = np.concatenate((np.full(50, -0.2), np.full(50, 3.0)))
        limit = reachable_reference_limit(bend)

        self.assertGreater(limit, 0.0)
        self.assertLessEqual(limit, 0.2)

    def test_one_sided_excitation_has_no_symmetric_safe_envelope(self):
        self.assertEqual(
            reachable_reference_limit(np.linspace(0.1, 3.0, 200)),
            0.0,
        )

    def test_video_arguments_are_optional_and_configurable(self):
        parser = build_argument_parser()
        default = parser.parse_args([])
        self.assertIsNone(default.video_out)
        self.assertEqual(default.video_fps, 30.0)
        self.assertTrue(default.live)
        self.assertTrue(default.live_hold)
        self.assertEqual(default.live_fps, 15.0)
        self.assertEqual(default.frequency_scale, 1.0)
        self.assertEqual(default.delay_steps, 50)
        self.assertEqual(default.settle_seconds, 4.0)
        self.assertEqual(default.train_seconds, 60.0)
        self.assertEqual(default.profile_seconds, 20.0)
        self.assertEqual(default.pressure_max, 10.0)
        self.assertEqual(default.history_length, 50)
        self.assertEqual(default.predictor_history, 50)
        self.assertEqual(default.amplitude, 2.8)
        self.assertEqual(default.slew_rate, 6.0)

        configured = parser.parse_args([
            "--video-out", "output/controller.mp4",
            "--video-fps", "24",
            "--no-live",
            "--no-live-hold",
            "--live-fps", "12",
        ])
        self.assertEqual(configured.video_out, Path("output/controller.mp4"))
        self.assertEqual(configured.video_fps, 24.0)
        self.assertFalse(configured.live)
        self.assertFalse(configured.live_hold)
        self.assertEqual(configured.live_fps, 12.0)

    def test_frequency_scale_changes_demo_without_changing_amplitude(self):
        slow, _, _ = make_demo_reference(20.0, 50, 2.0, frequency_scale=0.75)
        fast, _, _ = make_demo_reference(20.0, 50, 2.0, frequency_scale=1.0)

        np.testing.assert_array_equal(slow[:2000], fast[:2000])
        slow_sine = slow[2000:4000]
        fast_sine = fast[2000:4000]
        self.assertAlmostEqual(float(np.max(np.abs(slow_sine))), 1.44, places=3)
        self.assertAlmostEqual(float(np.max(np.abs(fast_sine))), 1.44, places=3)
        self.assertFalse(np.allclose(slow[2000:], fast[2000:]))

    def test_defaults_embed_measured_robot_provenance(self):
        metadata = calibrated_plant_metadata()

        self.assertEqual(metadata["reservoir_topology"], "parallel")
        self.assertEqual(metadata["actuator_delay_s"], 0.5)
        self.assertEqual(metadata["pressure_time_constant_s"], 0.6)
        self.assertEqual(metadata["simulator_pressure_ceiling_psi"], 11.0)
        self.assertEqual(metadata["recorded_command_ceiling_psi"], 10.0)
        self.assertEqual(metadata["recorded_excitation_frequency_hz"], 0.1)
        self.assertEqual(len(metadata["source_commit"]), 40)

    def test_command_limit_uses_calibrated_simulator_ceiling(self):
        args = build_argument_parser(live_default=False).parse_args([
            "--pressure-max", "11.01",
        ])

        with self.assertRaisesRegex(ValueError, "11 psi"):
            run_pipeline(args, persist_artifacts=False)


class InverseLabelTests(unittest.TestCase):
    @staticmethod
    def predictor() -> DelayAffinePredictor:
        state_dimension = 10
        basis_dimension = 1 + 3 * state_dimension
        coefficients = np.zeros(basis_dimension + 3)
        coefficients[basis_dimension:] = [0.2, 1.0, -0.2]
        return DelayAffinePredictor(
            delay_steps=1,
            history_length=1,
            state_mean=np.zeros(state_dimension),
            state_scale=np.ones(state_dimension),
            command_mean=np.zeros(3),
            command_scale=np.ones(3),
            coefficients=coefficients,
        )

    def test_recursive_labels_obey_bounds_slew_and_fixed_plane_bias(self):
        config = PRCConfig(
            control_hz=10.0,
            history_length=1,
            pressure_min_psi=0.0,
            pressure_max_psi=6.0,
            slew_rate_psi_s=1.0,
            initial_command_psi=(2.0, 2.0, 2.0),
            max_tick_s=None,
        )
        n = 12
        states = np.zeros((n, 10))
        references = np.linspace(-2.0, 2.0, n)
        previous = np.full((n, 3), 2.0)
        labels = generate_inverse_control_labels(
            self.predictor(),
            states,
            references,
            previous,
            config,
            fixed_plane_bias_psi=2.0,
            recursive_previous=True,
        )

        np.testing.assert_allclose(labels[:, [0, 2]], 2.0)
        self.assertTrue(np.all(labels >= 0.0))
        self.assertTrue(np.all(labels <= 6.0))
        trajectory = np.vstack((np.full(3, 2.0), labels))
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(trajectory, axis=0)))),
            0.1 + 1e-12,
        )


class StudentAcceptanceTests(unittest.TestCase):
    @staticmethod
    def demo(fallback: bool = False) -> ClosedLoopData:
        reference = np.tile([0.0, 1.0], 3)
        bend = 0.8 * reference
        command = np.array([
            [2.0, 2.0, 2.0],
            [2.0, 2.1, 2.0],
            [2.0, 2.2, 2.0],
            [2.0, 2.1, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.1, 2.0],
        ])
        return ClosedLoopData(
            time_s=np.arange(6) * 0.1,
            profile=np.repeat(["step", "sine", "multisine"], 2),
            reference_deg=reference,
            preview_reference_deg=reference.copy(),
            reservoir_psi=np.full((6, 5), 2.0),
            bend_deg=bend,
            measured_pressure_psi=command.copy(),
            command_psi=command,
            raw_command_psi=command.copy(),
            projected=np.zeros(6, dtype=bool),
            fallback=np.array([False, False, False, False, False, fallback]),
        )

    def test_acceptance_passes_valid_student_and_rejects_fallback(self):
        config = PRCConfig(
            control_hz=10.0,
            history_length=1,
            pressure_max_psi=6.0,
            slew_rate_psi_s=1.0,
            initial_command_psi=(2.0, 2.0, 2.0),
            max_tick_s=None,
        )
        report = PredictorReport(0.1, 0.2, 100, 20)
        good = self.demo()
        accepted, failures = evaluate_student_acceptance(
            good,
            tracking_metrics(good),
            report,
            config,
            np.full(3, 2.0),
        )
        self.assertTrue(accepted)
        self.assertEqual(failures, ())

        bad = self.demo(fallback=True)
        accepted, failures = evaluate_student_acceptance(
            bad,
            tracking_metrics(bad),
            report,
            config,
            np.full(3, 2.0),
        )
        self.assertFalse(accepted)
        self.assertIn("student rollout invoked a controller fallback", failures)


if __name__ == "__main__":
    unittest.main()
