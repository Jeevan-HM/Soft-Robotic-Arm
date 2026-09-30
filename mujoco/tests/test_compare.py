from dataclasses import replace
from pathlib import Path
import csv
import tempfile
import unittest

import numpy as np

from compare import (
    build_argument_parser,
    common_safety_failures,
    comparison_metrics,
    make_comparison_scenario,
    pid_calibration_metadata,
    save_comparison_log,
)
from run_prc import ClosedLoopData, EvaluationScenario


def make_demo() -> ClosedLoopData:
    reference = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    bend = reference - 1.0
    command = np.array([
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
        [2.0, 4.0, 2.0],
        [2.0, 2.0, 2.0],
        [2.0, 3.0, 2.0],
        [2.0, 1.0, 2.0],
    ])
    return ClosedLoopData(
        time_s=np.arange(6, dtype=float) * 0.1,
        profile=np.repeat(["step", "sine", "multisine"], 2),
        reference_deg=reference,
        preview_reference_deg=np.roll(reference, -1),
        reservoir_psi=np.full((6, 5), 2.0),
        bend_deg=bend,
        measured_pressure_psi=command.copy(),
        command_psi=command,
        raw_command_psi=command.copy(),
        projected=np.zeros(6, dtype=bool),
        fallback=np.zeros(6, dtype=bool),
    )


class ComparisonParserTests(unittest.TestCase):
    def test_defaults_use_matched_preview_and_do_not_open_prc_dashboard(self):
        parser = build_argument_parser()
        defaults = parser.parse_args([])

        self.assertTrue(defaults.pid_preview)
        self.assertFalse(defaults.live)
        self.assertIsNone(defaults.video_out)
        self.assertEqual(defaults.pid_kp, 1.0)
        self.assertEqual(defaults.pid_ki, 0.12)
        self.assertEqual(defaults.pid_kd, 0.03)
        self.assertEqual(defaults.pid_derivative_cutoff, 2.0)
        self.assertEqual(defaults.pid_integral_limit, 8.0)
        self.assertEqual(defaults.pid_antiwindup_gain, 0.5)

        causal = parser.parse_args(["--no-pid-preview"])
        self.assertFalse(causal.pid_preview)

    def test_help_reports_live_display_default(self):
        help_text = build_argument_parser().format_help()
        self.assertIn("show the model and tracking dashboard live", help_text)
        self.assertIn("disabled)", help_text)

    def test_custom_gain_calibration_provenance_is_unknown(self):
        parser = build_argument_parser()
        calibrated = pid_calibration_metadata(parser.parse_args([]))
        custom = pid_calibration_metadata(
            parser.parse_args(["--pid-kp", "0.56"])
        )

        self.assertFalse(calibrated["evaluation_scenario_used_for_selection"])
        self.assertEqual(calibrated["provenance"], "pid_calibration_record.json")
        self.assertEqual(calibrated["status"], "provisional_migration_baseline")
        self.assertEqual(
            calibrated["plant_calibration"]["reservoir_topology"],
            "parallel",
        )
        self.assertIsNone(custom["evaluation_scenario_used_for_selection"])
        self.assertEqual(custom["provenance"], "unknown_for_custom_settings")


class EvaluationScenarioTests(unittest.TestCase):
    def test_scenario_copies_and_freezes_shared_reference_arrays(self):
        reference = np.array([0.0, 1.0, -1.0])
        preview = np.array([1.0, -1.0, -1.0])
        profile = np.array(["step", "sine", "multisine"])
        scenario = EvaluationScenario(10.0, reference, preview, profile)

        reference[0] = 99.0
        preview[0] = 99.0
        profile[0] = "changed"
        np.testing.assert_array_equal(scenario.reference_deg, [0.0, 1.0, -1.0])
        np.testing.assert_array_equal(scenario.preview_reference_deg, [1.0, -1.0, -1.0])
        np.testing.assert_array_equal(
            scenario.profile,
            ["step", "sine", "multisine"],
        )
        self.assertFalse(scenario.reference_deg.flags.writeable)
        self.assertFalse(scenario.preview_reference_deg.flags.writeable)
        self.assertFalse(scenario.profile.flags.writeable)
        with self.assertRaises(ValueError):
            scenario.reference_deg[0] = 3.0
        np.testing.assert_allclose(scenario.time_s, [0.0, 0.1, 0.2])

    def test_comparison_scenario_has_aligned_bounded_profiles_and_preview(self):
        scenario = make_comparison_scenario(
            seconds_per_profile=1.0,
            delay_steps=3,
            limit_deg=2.0,
            control_hz=100.0,
            frequency_scale=1.0,
        )

        self.assertEqual(len(scenario.reference_deg), 300)
        self.assertEqual(np.count_nonzero(scenario.profile == "step"), 100)
        self.assertEqual(np.count_nonzero(scenario.profile == "sine"), 100)
        self.assertEqual(np.count_nonzero(scenario.profile == "multisine"), 100)
        self.assertLessEqual(float(np.max(np.abs(scenario.reference_deg))), 2.0)
        np.testing.assert_array_equal(
            scenario.preview_reference_deg[:-3],
            scenario.reference_deg[3:],
        )


class ComparisonLogTests(unittest.TestCase):
    def test_paired_csv_contains_both_controller_records(self):
        prc = make_demo()
        pid = replace(prc, bend_deg=prc.bend_deg + 0.25)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            returned = save_comparison_log(path, prc, pid)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(returned, path)
        self.assertEqual(len(rows), len(prc.time_s))
        self.assertIn("prc_bend_deg", rows[0])
        self.assertIn("pid_bend_deg", rows[0])
        self.assertIn("prc_command_s3_psi", rows[0])
        self.assertIn("pid_command_s3_psi", rows[0])
        self.assertEqual(float(rows[0]["prc_bend_deg"]), 0.0)
        self.assertEqual(float(rows[0]["pid_bend_deg"]), 0.25)

    def test_csv_rejects_rollouts_that_do_not_share_the_scenario(self):
        prc = make_demo()
        unpaired = replace(
            prc,
            preview_reference_deg=prc.preview_reference_deg + 0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            with self.assertRaisesRegex(ValueError, "not paired"):
                save_comparison_log(path, prc, unpaired)


class ComparisonMetricTests(unittest.TestCase):
    def test_metrics_report_tracking_effort_range_and_saturation(self):
        metrics = comparison_metrics(
            make_demo(),
            bias_psi=2.0,
            pressure_min_psi=0.0,
            pressure_max_psi=4.0,
        )

        for name in ("step", "sine", "multisine", "all"):
            self.assertAlmostEqual(metrics[name]["rmse_deg"], 1.0)
            self.assertAlmostEqual(metrics[name]["mae_deg"], 1.0)
        self.assertAlmostEqual(
            metrics["step"]["normalized_rmse"],
            1.0 / np.sqrt(5.0),
        )
        self.assertAlmostEqual(metrics["all"]["bend_peak_to_peak_deg"], 4.0)
        self.assertAlmostEqual(
            metrics["all"]["segment3_rms_excursion_psi"],
            np.sqrt(10.0 / 6.0),
        )
        self.assertAlmostEqual(metrics["step"]["saturation_fraction"], 0.5)
        self.assertAlmostEqual(metrics["sine"]["saturation_fraction"], 0.5)
        self.assertAlmostEqual(metrics["multisine"]["saturation_fraction"], 0.0)
        self.assertAlmostEqual(metrics["all"]["saturation_fraction"], 2.0 / 6.0)

    def test_common_safety_checks_detect_fallback_bounds_and_slew(self):
        demo = make_demo()
        failures = common_safety_failures(
            demo,
            pressure_min_psi=0.0,
            pressure_max_psi=4.0,
            slew_rate_psi_s=100.0,
            initial_pressure_psi=2.0,
            control_hz=10.0,
        )
        self.assertEqual(failures, ())

        unsafe_command = demo.command_psi.copy()
        unsafe_command[0, 1] = 5.0
        unsafe = replace(
            demo,
            command_psi=unsafe_command,
            fallback=np.array([True, False, False, False, False, False]),
        )
        failures = common_safety_failures(
            unsafe,
            pressure_min_psi=0.0,
            pressure_max_psi=4.0,
            slew_rate_psi_s=1.0,
            initial_pressure_psi=2.0,
            control_hz=10.0,
        )
        self.assertIn("controller fallback occurred", failures)
        self.assertIn("pressure bound violation", failures)
        self.assertIn("pressure slew violation", failures)


if __name__ == "__main__":
    unittest.main()
