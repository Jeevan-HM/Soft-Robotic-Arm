import numpy as np
import pytest

from soft_robotic_arm import make_calibrated_sim, make_sim


def test_calibrated_sim_has_sealed_s1_and_three_pressure_api():
    sim = make_calibrated_sim(control_hz=100, seed=7)
    obs = sim.reset()

    assert sim.uses_robot_calibration
    assert sim.reservoir_column == 0
    np.testing.assert_array_equal(sim.actuator_columns, [1, 2, 3])
    assert obs["tip_pos"].shape == (3,)
    assert obs["tip_quat"].shape == (4,)
    assert obs["tip_vel"].shape == (3,)
    assert obs["pouch_pressures"].shape == (4, 5)
    assert obs["reservoir_pressures"].shape == (5,)
    assert obs["actuator_pressures"].shape == (3,)
    np.testing.assert_array_equal(obs["actuator_columns"], [1, 2, 3])

    obs = sim.step([2.0, 3.0, 4.0])
    assert obs["time"] == pytest.approx(0.01)
    log = sim.get_pressure_log()
    np.testing.assert_allclose(log["actuator_cmd"][0], [2.0, 3.0, 4.0])
    assert np.isnan(log["p_cmd"][0, 0]).all()


@pytest.mark.parametrize(
    "bad_command",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        np.zeros((4, 5)),
        [1.0, np.nan, 3.0],
    ],
)
def test_calibrated_step_rejects_non_three_pressure_commands(bad_command):
    sim = make_calibrated_sim()
    with pytest.raises(ValueError):
        sim.step(bad_command)


def test_calibrated_step_clips_to_model_bounds_without_commanding_s1():
    sim = make_calibrated_sim()
    sim.reset()
    sim.step([-2.0, 5.0, 20.0])
    log = sim.get_pressure_log()

    np.testing.assert_allclose(log["actuator_cmd"][0], [0.0, 5.0, 11.0])
    assert np.isnan(log["p_cmd"][0, 0]).all()


def test_actuator_transport_delay_does_not_turn_s1_into_an_actuator():
    sim = make_calibrated_sim(control_hz=100, seed=0)
    obs = sim.reset()
    assert sim.actuator_delay_steps == 50
    assert np.all(obs["p_actual"][1:] == 0.0)
    assert np.all(obs["p_actual"][0] > 0.0)

    for _ in range(sim.actuator_delay_steps):
        obs = sim.step([4.0, 4.0, 4.0])
        assert np.all(obs["p_actual"][1:] == 0.0)
    obs = sim.step([4.0, 4.0, 4.0])

    assert np.all(obs["p_actual"][1:] > 0.0)
    assert sim.reservoir_column == 0
    assert np.isnan(sim.get_pressure_log()["p_cmd"][:, 0]).all()


def test_segment_commands_have_the_documented_tip_directions():
    def settled_tip(command):
        sim = make_calibrated_sim(control_hz=100, seed=0)
        obs = sim.reset()
        for _ in range(600):
            obs = sim.step(command)
        return obs["tip_pos"]

    zero = settled_tip([0.0, 0.0, 0.0])
    s2 = settled_tip([8.0, 0.0, 0.0])
    s3 = settled_tip([0.0, 8.0, 0.0])
    s4 = settled_tip([0.0, 0.0, 8.0])

    assert s2[1] - zero[1] > 0.005  # Segment 2 moves toward +y.
    assert s3[0] - zero[0] < -0.005  # Segment 3 moves toward -x.
    assert s4[1] - zero[1] < -0.005  # Segment 4 moves toward -y.


def test_coursework_sim_commands_all_four_segments_and_reports_pouches():
    sim = make_sim(control_hz=100, seed=0)
    obs = sim.reset()

    assert sim.reservoir_column is None
    np.testing.assert_array_equal(sim.actuator_columns, [0, 1, 2, 3])
    assert obs["pouch_pressures"].shape == (4, 5)
    assert obs["segment_pressures"].shape == (4,)

    obs = sim.step([1.0, 2.0, 3.0, 4.0])
    assert obs["time"] == pytest.approx(0.01)
    np.testing.assert_allclose(
        sim.get_pressure_log()["actuator_cmd"][0], [1.0, 2.0, 3.0, 4.0]
    )


@pytest.mark.parametrize(
    "bad_command",
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], np.zeros((4, 5))),
)
def test_coursework_sim_requires_exactly_four_segment_commands(bad_command):
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        make_sim().step(bad_command)


def test_coursework_segment_commands_match_cardinal_directions():
    def settled_tip(command):
        sim = make_sim(control_hz=100, seed=0)
        obs = sim.reset()
        for _ in range(600):
            obs = sim.step(command)
        return obs["tip_pos"]

    zero = settled_tip([0.0, 0.0, 0.0, 0.0])
    s1 = settled_tip([8.0, 0.0, 0.0, 0.0])
    s2 = settled_tip([0.0, 8.0, 0.0, 0.0])
    s3 = settled_tip([0.0, 0.0, 8.0, 0.0])
    s4 = settled_tip([0.0, 0.0, 0.0, 8.0])

    assert s1[0] - zero[0] > 0.005  # East, +x.
    assert s2[1] - zero[1] > 0.005  # North, +y.
    assert s3[0] - zero[0] < -0.005  # West, -x.
    assert s4[1] - zero[1] < -0.005  # South, -y.
