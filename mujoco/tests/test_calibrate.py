import unittest

import numpy as np

from calibrate import fit_global_rotation, motion_comparison


class MotionMetricTests(unittest.TestCase):
    @staticmethod
    def trajectory(samples=201):
        phase = np.linspace(0.0, 2.0 * np.pi, samples)
        return np.column_stack(
            (
                0.012 * np.cos(phase),
                0.008 * np.sin(phase),
                -0.30 + 0.002 * np.sin(2.0 * phase),
            )
        )

    def test_identity_motion_has_unit_amplitude_and_zero_error(self):
        points = self.trajectory()
        metrics, real_xy, simulated_xy = motion_comparison(
            points,
            points.copy(),
            sample_rate_hz=20.0,
            warmup_s=5.0,
            rotation=np.eye(3),
        )

        self.assertLess(metrics.lateral_rmse_mm, 1e-10)
        self.assertLess(metrics.axial_rmse_mm, 1e-10)
        self.assertAlmostEqual(metrics.lateral_amplitude_ratio, 1.0)
        np.testing.assert_allclose(simulated_xy, real_xy)

    def test_global_alignment_is_proper_and_cannot_accept_axis_preserving_mirror(self):
        real = self.trajectory()
        mirrored = real @ np.diag([1.0, -1.0, 1.0])
        rotation = fit_global_rotation(
            [(real, mirrored)],
            sample_rate_hz=20.0,
            warmup_s=5.0,
        )
        metrics, _, _ = motion_comparison(
            real,
            mirrored,
            sample_rate_hz=20.0,
            warmup_s=5.0,
            rotation=rotation,
        )

        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=10)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertGreater(metrics.lateral_rmse_mm, 5.0)


if __name__ == "__main__":
    unittest.main()
