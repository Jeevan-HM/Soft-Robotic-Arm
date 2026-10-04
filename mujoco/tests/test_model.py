import json

import mujoco
import numpy as np

from soft_robotic_arm import ArmConfig, RobotCalibration, build_arm_xml


def test_packaged_calibration_builds_canonical_model():
    calibration = RobotCalibration.load()
    cfg = calibration.make_arm_config()

    assert isinstance(cfg, ArmConfig)
    assert cfg.n_segments == 4
    assert cfg.n_pouches == 5
    assert cfg.p_max == 11.0
    np.testing.assert_allclose(
        np.degrees(cfg.col_azimuths()), [0.0, 90.0, 180.0, 270.0]
    )

    model = mujoco.MjModel.from_xml_string(build_arm_xml(cfg))
    assert model.nq == 15
    assert model.nv == 15
    assert mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "tip"
    ) >= 0


def test_arm_config_json_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"not_a_parameter": 1}), encoding="utf-8")

    try:
        ArmConfig.from_json(path)
    except ValueError as exc:
        assert "unknown ArmConfig keys" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("unknown fields must not be silently ignored")
