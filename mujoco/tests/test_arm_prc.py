import unittest

import numpy as np

from mjcf_model import ArmConfig
from simulator import SoftArmSim


class SoftArmReservoirTests(unittest.TestCase):
    @staticmethod
    def make_sim(reservoir_pressure=1.5, **kwargs):
        config = ArmConfig(tau_pneumatic=0.001)
        sensor_noise = kwargs.pop("sensor_noise_psi", 0.0)
        return SoftArmSim(
            cfg=config,
            control_hz=100.0,
            sensor_noise_psi=sensor_noise,
            seed=123,
            reservoir_column=0,
            reservoir_pressure_psi=reservoir_pressure,
            **kwargs,
        )

    def test_three_commands_map_to_columns_two_through_four(self):
        charge = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        sim = self.make_sim(reservoir_pressure=charge)

        obs = sim.step([2.0, 3.0, 4.0])
        logged = sim.get_pressure_log()["p_cmd"][-1]

        np.testing.assert_array_equal(sim.actuator_columns, [1, 2, 3])
        self.assertTrue(np.isnan(logged[0]).all())
        np.testing.assert_allclose(logged[1], 2.0)
        np.testing.assert_allclose(logged[2], 3.0)
        np.testing.assert_allclose(logged[3], 4.0)
        np.testing.assert_allclose(
            sim.get_pressure_log()["actuator_cmd"][-1], [2.0, 3.0, 4.0]
        )
        np.testing.assert_allclose(obs["p_actual"][1], 2.0)
        np.testing.assert_allclose(obs["p_actual"][2], 3.0)
        np.testing.assert_allclose(obs["p_actual"][3], 4.0)

    def test_sealed_segment_ignores_commands_and_changes_with_deformation(self):
        charge = np.full(5, 2.0)
        sim = self.make_sim(
            reservoir_pressure=charge,
            curvature_coupling=5.0,
            extension_coupling=20.0,
        )

        full_command = np.zeros((4, 5))
        full_command[0] = 9.0
        sim.step(full_command)
        logged = sim.get_pressure_log()["p_cmd"][-1]
        self.assertTrue(np.isnan(logged[0]).all())

        sim.reset()
        sim.data.qpos[sim.ext_dofs] = np.linspace(0.001, 0.005, 5)
        sim.data.qpos[sim.bend_dofs[:, 1]] = np.linspace(0.005, 0.025, 5)
        expected = np.clip(
            charge + sim._deformation_pressure_offset()[0],
            0.0,
            sim.cfg.p_max,
        )
        sim._update_sealed_reservoir()

        np.testing.assert_allclose(sim.reservoir_pressure, expected)
        np.testing.assert_allclose(sim.p_actual[0], charge)
        self.assertFalse(np.allclose(sim.reservoir_pressure, charge))

    def test_charging_at_a_deformed_pose_uses_that_pose_as_pressure_reference(self):
        sim = self.make_sim(
            reservoir_pressure=1.0,
            curvature_coupling=5.0,
            extension_coupling=20.0,
        )
        sim.data.qpos[sim.ext_dofs] = np.linspace(0.001, 0.005, 5)
        sim.data.qpos[sim.bend_dofs[:, 1]] = np.linspace(0.005, 0.025, 5)
        sim.set_reservoir_pressure(2.0, column=0)
        sim._update_sealed_reservoir()

        # The pressure is x at the instant the valves close, even when the arm
        # is not at its zero-coordinate pose.
        np.testing.assert_allclose(sim.reservoir_pressure, 2.0)

        sim.data.qpos[sim.ext_dofs] += 0.001
        sim._update_sealed_reservoir()
        self.assertFalse(np.allclose(sim.reservoir_pressure, 2.0))

    def test_returned_reservoir_sample_matches_returned_deformation(self):
        sim = self.make_sim(
            reservoir_pressure=2.0,
            curvature_coupling=50.0,
            extension_coupling=200.0,
        )
        obs = sim.step([2.0, 5.0, 2.0])
        expected = np.clip(
            sim.reservoir_charge
            + sim._deformation_pressure_offset()[0]
            - sim._reservoir_reference_offset,
            0.0,
            sim.cfg.p_max,
        )

        np.testing.assert_allclose(obs["reservoir_pressures"], expected)

    def test_observation_shapes_and_reset_restore_charged_reservoir(self):
        charge = np.array([1.0, 1.2, 1.4, 1.6, 1.8])
        sim = self.make_sim(reservoir_pressure=charge)

        initial = sim.reset()
        self.assertEqual(initial["tip_pos"].shape, (3,))
        self.assertEqual(initial["tip_quat"].shape, (4,))
        self.assertEqual(initial["tip_vel"].shape, (3,))
        self.assertEqual(initial["pouch_pressures"].shape, (4, 5))
        self.assertEqual(initial["p_actual"].shape, (4, 5))
        self.assertEqual(initial["q"].shape, (sim.model.nq,))
        self.assertEqual(initial["reservoir_pressures"].shape, (5,))
        self.assertEqual(initial["actuator_pressures"].shape, (3,))
        self.assertEqual(initial["actuator_columns"].shape, (3,))
        np.testing.assert_allclose(initial["reservoir_pressures"], charge)
        np.testing.assert_allclose(initial["actuator_pressures"], 0.0)

        sim.step([2.0, 3.0, 4.0])
        self.assertGreater(sim.data.time, 0.0)
        self.assertEqual(sim.get_pressure_log()["time"].shape, (1,))

        reset = sim.reset()
        self.assertEqual(reset["time"], 0.0)
        np.testing.assert_allclose(reset["p_actual"][0], charge)
        np.testing.assert_allclose(reset["p_actual"][1:], 0.0)
        np.testing.assert_allclose(reset["reservoir_pressures"], charge)
        np.testing.assert_allclose(reset["actuator_pressures"], 0.0)
        self.assertEqual(sim.get_pressure_log()["time"].shape, (0,))

    def test_prc_setpoints_are_absolute_even_with_global_preinflation(self):
        sim = self.make_sim(reservoir_pressure=2.0)
        sim.set_pre_inflation(3.0)
        sim.step([7.0, 8.0, 9.0])

        np.testing.assert_allclose(
            sim.get_pressure_log()["actuator_cmd"][-1], [7.0, 8.0, 9.0]
        )
        np.testing.assert_allclose(sim.p_actual[0], 2.0)

    def test_actuator_observations_are_measured_not_exact_state(self):
        sim = self.make_sim(reservoir_pressure=2.0, sensor_noise_psi=0.5)
        obs = sim.step([2.0, 3.0, 4.0])
        exact = obs["p_actual"][sim.actuator_columns].mean(axis=1)

        self.assertFalse(np.allclose(obs["actuator_pressures"], exact))

    def test_nonfinite_command_is_rejected_before_state_changes(self):
        sim = self.make_sim(reservoir_pressure=2.0)
        before = sim.p_actual.copy()
        with self.assertRaises(ValueError):
            sim.step([np.nan, 1.0, 1.0])
        np.testing.assert_array_equal(sim.p_actual, before)
        self.assertEqual(sim.data.time, 0.0)


if __name__ == "__main__":
    unittest.main()
