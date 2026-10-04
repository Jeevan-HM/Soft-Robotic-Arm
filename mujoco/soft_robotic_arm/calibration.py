"""Load the physical-robot calibration used by :mod:`simulator`.

``calibration.json`` is the authoritative description of the deployed
robot.  ``SoftArmSim()`` loads it automatically; this class also exposes the
validated values to calibration tools and callers that need to choose the
parallel or coupled reservoir topology explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .model import ArmConfig


DEFAULT_CALIBRATION_RESOURCE = files("soft_robotic_arm").joinpath(
    "data", "calibration.json"
)
# Compatibility alias: this may be an importlib ``Traversable`` rather than a
# filesystem Path when the package is imported directly from a wheel.
DEFAULT_CALIBRATION_PATH = DEFAULT_CALIBRATION_RESOURCE
SUPPORTED_TOPOLOGIES = ("parallel", "coupled")
ACTUATOR_KEYS = {
    "delay_s",
    "pressure_gain",
    "pressure_bias_psi",
    "bias_applies_above_zero_only",
}
RESERVOIR_KEYS = {
    "charge_gain",
    "charge_bias_psi",
    "leak_tau_s",
    "fast_relaxation_fraction",
    "fast_relaxation_tau_s",
    "relaxation_delay_s",
    "fast_fraction_by_charge",
    "fast_tau_s_by_charge",
    "curvature_coupling_psi_per_rad",
    "extension_coupling_psi_per_m",
    "equalization",
    "response_tau_s",
    "sensor_noise_psi",
    "force_feedback",
}


def _finite_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array))
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or shape ({size},)")
    return tuple(float(item) for item in array)


def _positive_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    result = _finite_vector(value, size, name)
    if any(item <= 0.0 for item in result):
        raise ValueError(f"{name} must contain only positive values")
    return result


@dataclass(frozen=True)
class RobotCalibration:
    """Validated measured-robot calibration payload."""

    source: Mapping[str, Any]
    arm_config: Mapping[str, Any]
    actuator: Mapping[str, Any]
    reservoirs: Mapping[str, Mapping[str, Any]]
    validation: Mapping[str, Any]

    @classmethod
    def load(
        cls, path: str | Path | None = None
    ) -> "RobotCalibration":
        """Load and validate the packaged calibration or an explicit JSON file.

        The default uses :mod:`importlib.resources`, so calibration data works
        from an installed wheel as well as a source checkout.
        """
        resource = DEFAULT_CALIBRATION_RESOURCE if path is None else Path(path)
        with resource.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        required = {"source", "arm_config", "actuator", "reservoirs"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                f"calibration {resource} is missing keys: {sorted(missing)}"
            )
        unknown_root = set(payload).difference(required | {"validation"})
        if unknown_root:
            raise ValueError(
                f"unknown calibration root keys: {sorted(unknown_root)}"
            )
        calibration = cls(
            source=dict(payload["source"]),
            arm_config=dict(payload["arm_config"]),
            actuator=dict(payload["actuator"]),
            reservoirs={
                str(name): dict(values)
                for name, values in payload["reservoirs"].items()
            },
            validation=dict(payload.get("validation", {})),
        )
        calibration._validate()
        return calibration

    def _validate(self) -> None:
        # ArmConfig performs field/type validation through construction and
        # ignores no keys here: a typo in a calibration must fail loudly.
        valid_arm_fields = set(ArmConfig.__dataclass_fields__)
        unknown = set(self.arm_config).difference(valid_arm_fields)
        if unknown:
            raise ValueError(f"unknown ArmConfig calibration keys: {sorted(unknown)}")
        arm = self.make_arm_config()
        if arm.n_segments != 4 or arm.n_pouches != 5:
            raise ValueError(
                "measured-robot calibration requires four segments and five pouches"
            )
        positive_arm_fields = (
            "length", "col_offset", "col_radius", "mass", "tip_mass",
            "moment_arm", "pressure_gain", "extension_gain",
            "base_stiffness", "base_damping", "axial_stiffness",
            "axial_damping", "p_max", "tau_pneumatic", "timestep",
        )
        for name in positive_arm_fields:
            value = float(getattr(arm, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"arm_config.{name} must be positive and finite")
        if (not np.isfinite(float(arm.stiffness_per_psi))
                or arm.stiffness_per_psi < 0.0):
            raise ValueError(
                "arm_config.stiffness_per_psi must be nonnegative and finite"
            )

        unknown_actuator = set(self.actuator).difference(ACTUATOR_KEYS)
        if unknown_actuator:
            raise ValueError(
                f"unknown actuator calibration keys: {sorted(unknown_actuator)}"
            )

        _positive_vector(
            self.actuator.get("pressure_gain", 1.0), 3,
            "actuator.pressure_gain",
        )
        _finite_vector(
            self.actuator.get("pressure_bias_psi", 0.0), 3,
            "actuator.pressure_bias_psi",
        )
        delay = float(self.actuator.get("delay_s", 0.0))
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError("actuator.delay_s must be finite and nonnegative")
        if self.actuator.get("bias_applies_above_zero_only", True) is not True:
            raise ValueError(
                "only above-zero actuator pressure bias is supported"
            )

        for topology in SUPPORTED_TOPOLOGIES:
            if topology not in self.reservoirs:
                raise ValueError(f"calibration has no {topology!r} reservoir")
            values = self.reservoirs[topology]
            unknown_reservoir = set(values).difference(RESERVOIR_KEYS)
            if unknown_reservoir:
                raise ValueError(
                    f"unknown {topology} reservoir calibration keys: "
                    f"{sorted(unknown_reservoir)}"
                )
            _positive_vector(values.get("charge_gain", 1.0), 5,
                             f"reservoirs.{topology}.charge_gain")
            _finite_vector(values.get("charge_bias_psi", 0.0), 5,
                           f"reservoirs.{topology}.charge_bias_psi")
            _positive_vector(values.get("leak_tau_s", 1e12), 5,
                             f"reservoirs.{topology}.leak_tau_s")
            has_fast_fraction = "fast_fraction_by_charge" in values
            has_fast_tau = "fast_tau_s_by_charge" in values
            if has_fast_fraction != has_fast_tau:
                raise ValueError(
                    f"reservoirs.{topology} must define both charge-dependent "
                    "fast-relaxation arrays"
                )
            if has_fast_fraction:
                fractions = _finite_vector(
                    values["fast_fraction_by_charge"], 3,
                    f"reservoirs.{topology}.fast_fraction_by_charge",
                )
                if any(not 0.0 <= value <= 1.0 for value in fractions):
                    raise ValueError(
                        f"reservoirs.{topology}.fast_fraction_by_charge "
                        "must lie in [0, 1]"
                    )
            if "fast_tau_s_by_charge" in values:
                _positive_vector(
                    values["fast_tau_s_by_charge"], 3,
                    f"reservoirs.{topology}.fast_tau_s_by_charge",
                )
            static_fraction = float(
                values.get("fast_relaxation_fraction", 0.0)
            )
            if (not np.isfinite(static_fraction)
                    or not 0.0 <= static_fraction <= 1.0):
                raise ValueError(
                    f"reservoirs.{topology}.fast_relaxation_fraction "
                    "must lie in [0, 1]"
                )
            static_tau = float(values.get("fast_relaxation_tau_s", 1.0))
            if not np.isfinite(static_tau) or static_tau <= 0.0:
                raise ValueError(
                    f"reservoirs.{topology}.fast_relaxation_tau_s "
                    "must be positive and finite"
                )
            relaxation_delay = float(values.get("relaxation_delay_s", 0.0))
            if not np.isfinite(relaxation_delay) or relaxation_delay < 0.0:
                raise ValueError(
                    f"reservoirs.{topology}.relaxation_delay_s must be "
                    "nonnegative and finite"
                )
            _finite_vector(values.get("curvature_coupling_psi_per_rad", 0.0), 5,
                           f"reservoirs.{topology}.curvature_coupling_psi_per_rad")
            _finite_vector(values.get("extension_coupling_psi_per_m", 0.0), 5,
                           f"reservoirs.{topology}.extension_coupling_psi_per_m")
            equalization = float(values.get("equalization", 0.0))
            if not np.isfinite(equalization) or not 0.0 <= equalization <= 1.0:
                raise ValueError(
                    f"reservoirs.{topology}.equalization must lie in [0, 1]"
                )
            response_tau = float(values.get("response_tau_s", 0.0))
            if not np.isfinite(response_tau) or response_tau < 0.0:
                raise ValueError(
                    f"reservoirs.{topology}.response_tau_s must be "
                    "nonnegative and finite"
                )
            sensor_noise = float(values.get("sensor_noise_psi", 0.0))
            if not np.isfinite(sensor_noise) or sensor_noise < 0.0:
                raise ValueError(
                    f"reservoirs.{topology}.sensor_noise_psi must be "
                    "nonnegative and finite"
                )
            if not isinstance(values.get("force_feedback", True), bool):
                raise ValueError(
                    f"reservoirs.{topology}.force_feedback must be boolean"
                )
        unknown_topologies = set(self.reservoirs).difference(SUPPORTED_TOPOLOGIES)
        if unknown_topologies:
            raise ValueError(
                f"unknown reservoir topologies: {sorted(unknown_topologies)}"
            )

    def make_arm_config(self) -> ArmConfig:
        return ArmConfig(**dict(self.arm_config))

    def simulator_kwargs(
        self,
        topology: str = "parallel",
        reservoir_pressure_psi: float | np.ndarray = 0.0,
    ) -> dict[str, Any]:
        topology = str(topology).lower()
        if topology not in SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {SUPPORTED_TOPOLOGIES}, got {topology!r}"
            )
        reservoir = self.reservoirs[topology]
        kwargs: dict[str, Any] = {
            "sensor_noise_psi": float(reservoir.get("sensor_noise_psi", 0.0)),
            "curvature_coupling": np.asarray(
                reservoir.get("curvature_coupling_psi_per_rad", 0.0),
                dtype=float,
            ),
            "extension_coupling": np.asarray(
                reservoir.get("extension_coupling_psi_per_m", 0.0),
                dtype=float,
            ),
            "actuator_delay_s": float(self.actuator.get("delay_s", 0.0)),
            "actuator_pressure_gain": np.asarray(
                self.actuator.get("pressure_gain", 1.0), dtype=float
            ),
            "actuator_pressure_bias_psi": np.asarray(
                self.actuator.get("pressure_bias_psi", 0.0), dtype=float
            ),
            "reservoir_charge_gain": np.asarray(
                reservoir.get("charge_gain", 1.0), dtype=float
            ),
            "reservoir_charge_bias_psi": np.asarray(
                reservoir.get("charge_bias_psi", 0.0), dtype=float
            ),
            "reservoir_leak_tau_s": np.asarray(
                reservoir.get("leak_tau_s", 1e12), dtype=float
            ),
            "reservoir_fast_relaxation_fraction": float(
                reservoir.get("fast_relaxation_fraction", 0.0)
            ),
            "reservoir_fast_relaxation_tau_s": float(
                reservoir.get("fast_relaxation_tau_s", 1.0)
            ),
            "reservoir_relaxation_delay_s": float(
                reservoir.get("relaxation_delay_s", 0.0)
            ),
            "reservoir_equalization": float(
                reservoir.get("equalization", 0.0)
            ),
            "reservoir_response_tau_s": float(
                reservoir.get("response_tau_s", 0.0)
            ),
            "reservoir_force_feedback": bool(
                reservoir.get("force_feedback", True)
            ),
        }
        nominal_charge = float(np.mean(np.asarray(reservoir_pressure_psi)))
        if "fast_fraction_by_charge" in reservoir:
            kwargs["reservoir_fast_relaxation_fraction"] = float(np.interp(
                nominal_charge,
                (1.0, 2.0, 3.0),
                reservoir["fast_fraction_by_charge"],
            ))
        if "fast_tau_s_by_charge" in reservoir:
            kwargs["reservoir_fast_relaxation_tau_s"] = float(np.interp(
                nominal_charge,
                (1.0, 2.0, 3.0),
                reservoir["fast_tau_s_by_charge"],
            ))
        return kwargs

    def make_sim(
        self,
        *,
        topology: str = "parallel",
        reservoir_pressure_psi: float | np.ndarray = 2.0,
        reservoir_column: int | None = 0,
        control_hz: float = 100.0,
        seed: int | None = 0,
        **overrides: Any,
    ):
        """Construct a ``SoftArmSim`` with the calibrated mechanics/pneumatics."""
        from .simulator import SoftArmSim

        if reservoir_column != 0:
            raise ValueError(
                "the measured calibration requires sealed Segment 1 "
                "(reservoir_column=0)"
            )
        kwargs = self.simulator_kwargs(topology, reservoir_pressure_psi)
        nominal_charge = float(np.mean(np.asarray(reservoir_pressure_psi)))
        kwargs.update(overrides)
        sim = SoftArmSim(
            cfg=self.make_arm_config(),
            control_hz=control_hz,
            seed=seed,
            topology=topology,
            reservoir_column=reservoir_column,
            reservoir_pressure_psi=reservoir_pressure_psi,
            **kwargs,
        )
        sim.uses_robot_calibration = True
        # Segment-1 charge also changes the arm's effective stiffness. In
        # reservoir mode set_pre_inflation does not add pressure to the
        # absolute commands for Segments 2--4.
        sim.set_pre_inflation(nominal_charge)
        return sim
