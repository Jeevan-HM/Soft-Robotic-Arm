import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mjcf_model import ArmConfig
from calibration import DEFAULT_CALIBRATION_PATH, RobotCalibration
from simulator import SoftArmSim


class RobotCalibrationTests(unittest.TestCase):
    @staticmethod
    def load_payload():
        with Path(DEFAULT_CALIBRATION_PATH).open(encoding="utf-8") as stream:
            return json.load(stream)

    def assert_payload_rejected(self, payload, pattern):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            with path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            with self.assertRaisesRegex(ValueError, pattern):
                RobotCalibration.load(path)

    def test_plain_constructor_is_the_calibrated_parallel_robot(self):
        calibration = RobotCalibration.load()
        sim = SoftArmSim(seed=7)
        self.addCleanup(sim.close)

        self.assertTrue(sim.uses_robot_calibration)
        self.assertEqual(sim.topology, "parallel")
        self.assertEqual(sim.reservoir_column, 0)
        np.testing.assert_array_equal(sim.actuator_columns, [1, 2, 3])
        np.testing.assert_allclose(sim.reservoir_nominal_charge, 2.0)
        self.assertEqual(sim.p_pre, 2.0)
        self.assertEqual(sim.actuator_delay_steps, 50)
        self.assertEqual(asdict(sim.cfg), dict(calibration.arm_config))
        np.testing.assert_allclose(
            sim.actuator_pressure_gain[sim.actuator_columns],
            calibration.actuator["pressure_gain"],
        )
        np.testing.assert_allclose(
            sim.actuator_pressure_bias[sim.actuator_columns],
            calibration.actuator["pressure_bias_psi"],
        )

    def test_arm_config_defaults_match_the_canonical_calibration(self):
        calibration = RobotCalibration.load()
        self.assertEqual(asdict(ArmConfig()), dict(calibration.arm_config))

    def test_calibration_contains_no_historical_default_baseline(self):
        payload = self.load_payload()
        keys = {
            key
            for section in payload.get("validation", {}).values()
            if isinstance(section, dict)
            for key in section
        }
        self.assertFalse(any(key.startswith("default_") for key in keys))

    def test_plain_coupled_constructor_uses_coupled_reservoir_fit(self):
        calibration = RobotCalibration.load()
        sim = SoftArmSim(
            topology="coupled",
            reservoir_pressure_psi=3.0,
            seed=8,
        )
        self.addCleanup(sim.close)

        coupled = calibration.reservoirs["coupled"]
        self.assertEqual(sim.topology, "coupled")
        self.assertEqual(sim.reservoir_equalization, coupled["equalization"])
        np.testing.assert_allclose(
            sim.reservoir_fast_relaxation_fraction,
            coupled["fast_fraction_by_charge"][-1],
        )
        np.testing.assert_allclose(
            sim.reservoir_fast_relaxation_tau,
            coupled["fast_tau_s_by_charge"][-1],
        )

    def test_default_json_loads_and_maps_parallel_charge(self):
        calibration = RobotCalibration.load()

        self.assertEqual(calibration.source["conditions"], 36)
        self.assertEqual(calibration.make_arm_config().p_max, 11.0)

        charge = 2.0
        sim = calibration.make_sim(
            topology="parallel",
            reservoir_pressure_psi=charge,
            seed=7,
        )
        self.addCleanup(sim.close)
        reservoir = calibration.reservoirs["parallel"]
        expected_charge = np.clip(
            charge * np.asarray(reservoir["charge_gain"])
            + np.asarray(reservoir["charge_bias_psi"]),
            0.0,
            sim.cfg.p_max,
        )

        self.assertTrue(sim.uses_robot_calibration)
        self.assertEqual(sim.reservoir_column, 0)
        np.testing.assert_array_equal(sim.actuator_columns, [1, 2, 3])
        np.testing.assert_allclose(sim.reservoir_nominal_charge, charge)
        np.testing.assert_allclose(sim.reservoir_charge, expected_charge)
        np.testing.assert_allclose(sim.p_actual[0], expected_charge)
        np.testing.assert_allclose(
            sim.curvature_coupling,
            reservoir["curvature_coupling_psi_per_rad"],
        )
        np.testing.assert_allclose(sim.reservoir_leak_tau, reservoir["leak_tau_s"])
        self.assertEqual(sim.reservoir_equalization, 0.0)
        self.assertEqual(sim.p_pre, charge)
        with self.assertRaisesRegex(ValueError, "requires sealed Segment 1"):
            calibration.make_sim(reservoir_column=None)

    def test_unknown_calibration_keys_fail_loudly(self):
        mutations = (
            (
                "root",
                lambda payload: payload.__setitem__("actutor", {}),
                "unknown calibration root keys",
            ),
            (
                "arm_config",
                lambda payload: payload["arm_config"].__setitem__(
                    "pressure_gian", 0.2
                ),
                "unknown ArmConfig calibration keys",
            ),
            (
                "actuator",
                lambda payload: payload["actuator"].__setitem__(
                    "pressure_gian", [1.0, 1.0, 1.0]
                ),
                "unknown actuator calibration keys",
            ),
            (
                "reservoir",
                lambda payload: payload["reservoirs"]["parallel"].__setitem__(
                    "equalizaton", 0.5
                ),
                "unknown parallel reservoir calibration keys",
            ),
            (
                "topology",
                lambda payload: payload["reservoirs"].__setitem__("serial", {}),
                "unknown reservoir topologies",
            ),
        )
        original = self.load_payload()
        for name, mutate, pattern in mutations:
            with self.subTest(section=name):
                payload = copy.deepcopy(original)
                mutate(payload)
                self.assert_payload_rejected(payload, pattern)

    def test_actuator_delay_gain_and_above_zero_bias(self):
        calibration = RobotCalibration.load()
        sim = calibration.make_sim(
            topology="parallel",
            reservoir_pressure_psi=2.0,
            control_hz=100.0,
            seed=11,
        )
        self.addCleanup(sim.close)

        # Make chamber tracking exact so this test isolates the calibrated
        # transport delay and static pressure map.
        sim.cfg.tau_pneumatic = sim.cfg.timestep
        command = np.array([1.0, 2.0, 0.0])
        expected = np.clip(
            np.asarray(calibration.actuator["pressure_gain"]) * command
            + np.asarray(calibration.actuator["pressure_bias_psi"])
            * (command > 0.0),
            0.0,
            sim.cfg.p_max,
        )

        self.assertEqual(sim.actuator_delay_steps, 50)
        for _ in range(sim.actuator_delay_steps):
            observation = sim.step(command)
            np.testing.assert_allclose(
                observation["p_actual"][sim.actuator_columns], 0.0
            )

        observation = sim.step(command)
        np.testing.assert_allclose(
            observation["p_actual"][sim.actuator_columns],
            np.repeat(expected[:, None], sim.cfg.n_pouches, axis=1),
        )
        # Zero commands do not receive the fitted positive-pressure offset.
        np.testing.assert_allclose(observation["p_actual"][3], 0.0)

    def test_coupled_charge_selects_fast_relaxation_and_decays(self):
        calibration = RobotCalibration.load()
        charge = 2.5
        coupled = calibration.reservoirs["coupled"]
        sim = calibration.make_sim(
            topology="coupled",
            reservoir_pressure_psi=charge,
            seed=13,
        )
        self.addCleanup(sim.close)

        expected_fraction = np.interp(
            charge,
            (1.0, 2.0, 3.0),
            coupled["fast_fraction_by_charge"],
        )
        expected_fast_tau = np.interp(
            charge,
            (1.0, 2.0, 3.0),
            coupled["fast_tau_s_by_charge"],
        )
        np.testing.assert_allclose(
            sim.reservoir_fast_relaxation_fraction, expected_fraction
        )
        np.testing.assert_allclose(
            sim.reservoir_fast_relaxation_tau, expected_fast_tau
        )
        self.assertEqual(sim.reservoir_equalization, coupled["equalization"])

        initial_charge = sim.reservoir_charge.copy()
        clock_time = 10.0
        sim.data.time = clock_time
        sim._update_sealed_reservoir()

        elapsed = clock_time - coupled["relaxation_delay_s"]
        slow_fraction = np.exp(-elapsed / sim.reservoir_leak_tau)
        fast_fraction = np.exp(-elapsed / sim.reservoir_fast_relaxation_tau)
        relaxation = (
            (1.0 - sim.reservoir_fast_relaxation_fraction) * slow_fraction
            + sim.reservoir_fast_relaxation_fraction * fast_fraction
        )
        independent_pressure = initial_charge * relaxation
        expected_pressure = (
            (1.0 - sim.reservoir_equalization) * independent_pressure
            + sim.reservoir_equalization * np.mean(independent_pressure)
        )

        np.testing.assert_allclose(sim.reservoir_pressure, expected_pressure)
        np.testing.assert_allclose(sim.p_actual[0], expected_pressure)
        self.assertLess(
            float(np.mean(sim.reservoir_pressure)),
            float(np.mean(initial_charge)),
        )


if __name__ == "__main__":
    unittest.main()
