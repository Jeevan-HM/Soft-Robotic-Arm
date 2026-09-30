import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiment_data import (
    CSV_COLUMNS,
    RobotExperimentDataError,
    load_robot_experiment_csv,
    parse_experiment_metadata,
)


class FilenameMetadataTests(unittest.TestCase):
    def test_parses_case_and_triangle_alias(self):
        metadata = parse_experiment_metadata("TriAnGlE_3-10_Coupled.csv")

        self.assertEqual(metadata.waveform, "triangular")
        self.assertEqual(metadata.topology, "coupled")
        self.assertEqual(metadata.charge_psi, 3.0)
        self.assertEqual(metadata.pre_inflation_psi, 3.0)
        self.assertEqual(metadata.max_psi, 10.0)

    def test_rejects_bad_filename_and_parent_topology_mismatch(self):
        with self.assertRaisesRegex(RobotExperimentDataError, "filename must match"):
            parse_experiment_metadata("experiment.csv")

        path = Path("coupled") / "Axial_1-5_parallel.csv"
        with self.assertRaisesRegex(RobotExperimentDataError, "disagrees"):
            parse_experiment_metadata(path)


class RobotExperimentLoaderTests(unittest.TestCase):
    def test_control_signals_keep_logger_timing_across_held_mocap_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Axial_1-5_parallel.csv"
            rows = [
                self.row(0.00, 1.0, logger_time=0.00, command=(0, 0, 0)),
                self.row(0.08, 0.0, logger_time=0.10, command=(0, 0, 0)),
                # This control update shares the prior held mocap frame. It
                # must remain at logger t=.15 rather than replacing t=.08.
                self.row(0.08, 0.0, logger_time=0.15, command=(10, 10, 10)),
                self.row(0.18, 0.0, logger_time=0.20, command=(20, 20, 20)),
                self.row(0.28, 0.0, logger_time=0.25, command=(30, 30, 30)),
            ]
            self.write_csv(path, rows)

            data = load_robot_experiment_csv(
                path,
                sample_rate_hz=20.0,
                active_duration_s=0.1,
                max_interpolation_gap_s=0.2,
            )

        np.testing.assert_allclose(data.time_s, [0.0, 0.05, 0.1])
        np.testing.assert_allclose(
            data.commands_psi,
            [[0, 0, 0], [10, 10, 10], [20, 20, 20]],
        )
        self.assertEqual(data.metadata.active_start_s, 0.1)

    def test_deduplicates_resamples_and_uses_rb2_to_rb3_in_rb1_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Triangular_3-10_coupled.csv"
            rows = [
                self.row(0.0, 3.0, base_vector=(1.0, 0.0, 0.0), command=(1, 1, 1)),
                self.row(0.1, 3.0, base_vector=(1.0, 0.0, 0.0), command=(1, 1, 1)),
                # Same mocap frame as the last prefill row. Last values must win.
                self.row(0.1, 0.0, base_vector=(1.0, 0.0, 0.0), command=(2, 3, 4)),
                self.row(0.2, 0.0, base_vector=(1.1, 0.1, 0.0), command=(3, 4, 5)),
                self.row(
                    0.2,
                    0.0,
                    base_vector=(1.1, 0.1, 0.0),
                    command=(4, 5, 6),
                    measured_s2="",
                ),
                self.row(0.3, 0.0, base_vector=(1.2, 0.2, 0.1), command=(6, 7, 8)),
                self.row(0.4, 0.0, base_vector=(1.3, 0.3, 0.2), command=(8, 9, 10)),
            ]
            self.write_csv(path, rows)

            data = load_robot_experiment_csv(
                path,
                sample_rate_hz=10.0,
                active_duration_s=0.3,
                max_interpolation_gap_s=0.21,
            )

        np.testing.assert_allclose(data.time_s, [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            data.commands_psi,
            [[2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10]],
        )
        # The missing S2 measurement at t=.2 is safely interpolated.
        np.testing.assert_allclose(data.measured_pressures_psi[:, 0], [20, 30, 40, 50])
        self.assertEqual(data.reservoir_pressures_psi.shape, (4, 5))

        # The synthetic RB1 orientation is +90 degrees about Z. The loader
        # rotates the world vector back to the requested base-frame vectors.
        np.testing.assert_allclose(
            data.tip_vector_m,
            [[1.0, 0.0, 0.0], [1.1, 0.1, 0.0], [1.2, 0.2, 0.1], [1.3, 0.3, 0.2]],
            atol=1e-12,
        )
        np.testing.assert_allclose(data.initial_axis, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(data.axial_displacement_m, [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            data.transverse_displacement_m,
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1], [0.3, 0.2]],
            atol=1e-12,
        )
        # Fixture-to-tip is deliberately different and cannot contaminate the
        # calibrated arm length/displacement calculation.
        self.assertFalse(np.allclose(data.fixture_to_tip_vector_m, data.tip_vector_m))
        self.assertEqual(data.metadata.active_start_s, 0.1)
        self.assertEqual(data.metadata.raw_row_count, 7)
        self.assertEqual(data.metadata.unique_mocap_samples, 5)

    def test_rejects_missing_column_absent_transition_and_long_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.row(0.0, 1.0),
                self.row(0.1, 0.0),
                self.row(0.4, 0.0),
            ]

            missing_column = root / "Axial_1-5_parallel.csv"
            self.write_csv(
                missing_column,
                rows,
                columns=CSV_COLUMNS[:-1],
            )
            with self.assertRaisesRegex(RobotExperimentDataError, "expected 36 columns"):
                load_robot_experiment_csv(
                    missing_column,
                    sample_rate_hz=10,
                    active_duration_s=0.2,
                )

            no_transition = root / "Circular_1-5_parallel.csv"
            self.write_csv(no_transition, [self.row(0.0, 1.0), self.row(0.2, 1.0)])
            with self.assertRaisesRegex(RobotExperimentDataError, "active-window start"):
                load_robot_experiment_csv(
                    no_transition,
                    sample_rate_hz=10,
                    active_duration_s=0.1,
                )

            long_gap = root / "Triangular_1-5_parallel.csv"
            self.write_csv(long_gap, rows)
            with self.assertRaisesRegex(RobotExperimentDataError, "missing-data gap"):
                load_robot_experiment_csv(
                    long_gap,
                    sample_rate_hz=10,
                    active_duration_s=0.3,
                    max_interpolation_gap_s=0.2,
                )

    @staticmethod
    def row(
        mocap_time,
        desired_s1,
        *,
        base_vector=(1.0, 0.0, 0.0),
        command=(2.0, 3.0, 4.0),
        measured_s2=None,
        logger_time=None,
    ):
        row = {name: 0.0 for name in CSV_COLUMNS}
        row["step_id"] = mocap_time * 100
        row["time"] = mocap_time if logger_time is None else logger_time
        row["mocap_time_rel_s"] = mocap_time
        row["Desired_pressure_segment_1"] = desired_s1
        for segment, value in zip((2, 3, 4), command):
            row[f"Desired_pressure_segment_{segment}"] = value

        row["Measured_pressure_Segment_2"] = (
            10.0 + 100.0 * mocap_time if measured_s2 is None else measured_s2
        )
        row["Measured_pressure_Segment_3"] = 20.0 + 100.0 * mocap_time
        row["Measured_pressure_Segment_4"] = 30.0 + 100.0 * mocap_time
        for pouch in range(1, 6):
            row[f"Measured_pressure_Segment_1_pouch_{pouch}"] = pouch + mocap_time

        # RB1 has a +90 degree Z rotation; world = R * base.
        half_sqrt = math.sqrt(0.5)
        row.update(
            {
                "Rigid_body_1_x": 10.0,
                "Rigid_body_1_y": 20.0,
                "Rigid_body_1_z": 30.0,
                "Rigid_body_1_qx": 0.0,
                "Rigid_body_1_qy": 0.0,
                "Rigid_body_1_qz": half_sqrt,
                "Rigid_body_1_qw": half_sqrt,
                "Rigid_body_2_x": 1.0,
                "Rigid_body_2_y": 2.0,
                "Rigid_body_2_z": 3.0,
            }
        )
        bx, by, bz = base_vector
        world_vector = (-by, bx, bz)
        row["Rigid_body_3_x"] = row["Rigid_body_2_x"] + world_vector[0]
        row["Rigid_body_3_y"] = row["Rigid_body_2_y"] + world_vector[1]
        row["Rigid_body_3_z"] = row["Rigid_body_2_z"] + world_vector[2]
        # RB2/RB3 quaternions exist in the schema but are not needed here.
        row["Rigid_body_2_qw"] = 1.0
        row["Rigid_body_3_qw"] = 1.0
        return row

    @staticmethod
    def write_csv(path, rows, columns=CSV_COLUMNS):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
