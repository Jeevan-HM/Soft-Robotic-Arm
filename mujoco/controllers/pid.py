"""Safety-constrained PID bend controller for the MuJoCo soft arm.

The scalar commissioning task bends about the same +Y coordinate used by the
PRC controller.  Segments 2 and 4 stay at the balanced ``x``-psi bias while
the PID law changes Segment 3.  Its optional two-degree-of-freedom mode uses
reference preview only for the proportional term; integral action remains on
current error and derivative action remains on measured bend. Both controllers
share ``PRCConfig`` and ``PressureProjector`` so pressure, slew, deadline, and
fallback rules are identical in comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from prc import (
    InfeasiblePressureConstraints,
    PRCConfig,
    PressureProjector,
)


ArrayLike = Iterable[float] | np.ndarray


def _vector(value: float | ArrayLike, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).copy()
    if out.ndim == 0:
        out = np.full(size, float(out))
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite and scalar or shape ({size},)")
    return out


@dataclass(frozen=True)
class PIDGains:
    """Conservative calibrated-plant baseline, pending a full robustness tune."""

    kp_psi_per_deg: float = 1.0
    ki_psi_per_deg_s: float = 0.12
    kd_psi_s_per_deg: float = 0.03
    control_direction: float = 1.0

    def __post_init__(self) -> None:
        values = np.array([
            self.kp_psi_per_deg,
            self.ki_psi_per_deg_s,
            self.kd_psi_s_per_deg,
            self.control_direction,
        ])
        if not np.all(np.isfinite(values)):
            raise ValueError("PID gains and control direction must be finite")
        if np.any(values[:3] < 0.0):
            raise ValueError("PID gains must be nonnegative")
        if self.control_direction not in {-1.0, 1.0}:
            raise ValueError("control_direction must be either -1 or +1")


@dataclass(frozen=True)
class PIDStep:
    command_psi: np.ndarray
    raw_command_psi: np.ndarray
    error_deg: float
    filtered_measurement_rate_deg_s: float
    integral_deg_s: float
    projected: bool
    slew_relaxed: bool
    fallback: bool
    gate_reason: str
    elapsed_s: float


class PIDController:
    """Filtered PID feedback with projection and integral anti-windup."""

    # Version 2 deliberately makes pre-calibration controller files fail
    # closed instead of looking like results from the measured-robot plant.
    FORMAT_VERSION = 2

    def __init__(
        self,
        gains: PIDGains | None = None,
        config: PRCConfig | None = None,
        bias_pressure_psi: float | ArrayLike | None = None,
        controlled_actuator: int = 1,
        use_preview_for_proportional: bool = True,
        metadata: dict | None = None,
    ):
        self.gains = gains or PIDGains()
        self.config = config or PRCConfig()
        if self.config.n_actuators != 3:
            raise ValueError("the scalar bend PID requires exactly three actuators")
        if not 0 <= controlled_actuator < self.config.n_actuators:
            raise ValueError("controlled_actuator is out of range")
        self.controlled_actuator = int(controlled_actuator)
        self.use_preview_for_proportional = bool(use_preview_for_proportional)
        default_bias = self.config.initial_command_psi
        self.bias_pressure_psi = _vector(
            default_bias if bias_pressure_psi is None else bias_pressure_psi,
            self.config.n_actuators,
            "bias pressure",
        )
        self.projector = PressureProjector(self.config)
        if (np.any(self.bias_pressure_psi < self.projector.p_min)
                or np.any(self.bias_pressure_psi > self.projector.p_max)):
            raise ValueError("bias pressure must lie inside the pressure bounds")
        try:
            self.metadata = json.loads(json.dumps(metadata or {}))
        except (TypeError, ValueError) as exc:
            raise ValueError("controller metadata must be JSON serializable") from exc
        self.reset()

    def reset(
        self,
        initial_command_psi: ArrayLike | None = None,
        reservoir_pressures: ArrayLike | None = None,
    ) -> None:
        """Reset dynamic PID state; the reservoir argument keeps the PRC API."""
        if reservoir_pressures is not None:
            _vector(
                reservoir_pressures,
                self.config.n_reservoir,
                "reservoir pressures",
            )
        initial = (
            self.bias_pressure_psi
            if initial_command_psi is None
            else _vector(
                initial_command_psi,
                self.config.n_actuators,
                "initial command",
            )
        )
        self.previous_command = np.clip(
            initial,
            self.projector.p_min,
            self.projector.p_max,
        )
        self.previous_measurement: float | None = None
        self.filtered_measurement_rate = 0.0
        self.integral = 0.0

    def _fallback(
        self,
        reason: str,
        started: float,
        raw: np.ndarray | None = None,
        supply_ceiling_psi: float | ArrayLike | None = None,
    ) -> PIDStep:
        ceiling = self.projector.p_max.copy()
        force_vent = reason.startswith((
            "invalid_input",
            "invalid_projection",
            "infeasible_projection",
            "nonfinite_pid",
            "pressure_trip",
        ))
        try:
            if supply_ceiling_psi is not None:
                ceiling = np.minimum(
                    ceiling,
                    _vector(
                        supply_ceiling_psi,
                        self.config.n_actuators,
                        "supply ceiling",
                    ),
                )
        except (TypeError, ValueError):
            force_vent = True
            reason = f"{reason}; invalid_supply_ceiling"

        emergency_vent = self.config.fallback_mode == "vent" or force_vent
        if emergency_vent:
            command = np.zeros(self.config.n_actuators)
        elif np.any(ceiling < self.projector.p_min):
            command = np.maximum(
                np.minimum(np.zeros(self.config.n_actuators), ceiling), 0.0
            )
            reason = f"{reason}; hard_set_empty"
        else:
            try:
                command = self.projector._project_capped_box(
                    self.previous_command,
                    self.projector.p_min,
                    ceiling,
                    self.config.max_total_pressure_psi,
                )
            except InfeasiblePressureConstraints:
                command = self.projector.p_min.copy()
                reason = f"{reason}; hard_set_empty"
        self.previous_command = command.copy()
        raw_command = command.copy() if raw is None else np.asarray(raw).copy()
        return PIDStep(
            command_psi=command,
            raw_command_psi=raw_command,
            error_deg=np.nan,
            filtered_measurement_rate_deg_s=self.filtered_measurement_rate,
            integral_deg_s=self.integral,
            projected=True,
            slew_relaxed=True,
            fallback=True,
            gate_reason=reason,
            elapsed_s=perf_counter() - started,
        )

    def compute(
        self,
        reference_deg: float,
        measured_deg: float,
        reservoir_pressures: ArrayLike,
        actuator_pressures_psi: ArrayLike,
        preview_reference_deg: float | None = None,
        dt: float | None = None,
        sample_age_s: float = 0.0,
        supply_ceiling_psi: float | ArrayLike | None = None,
    ) -> PIDStep:
        """Compute one safe absolute pressure command for Segments 2--4."""
        started = perf_counter()
        try:
            dt = self.config.dt if dt is None else float(dt)
            preview = reference_deg if preview_reference_deg is None else float(
                preview_reference_deg
            )
            reservoir = _vector(
                reservoir_pressures,
                self.config.n_reservoir,
                "reservoir pressures",
            )
            measured_pressure = _vector(
                actuator_pressures_psi,
                self.config.n_actuators,
                "actuator pressures",
            )
            scalars = np.array([
                reference_deg,
                measured_deg,
                preview,
                dt,
                sample_age_s,
            ])
            if not np.all(np.isfinite(scalars)) or dt <= 0.0:
                raise ValueError("references, measurement, age, and dt must be valid")
        except (TypeError, ValueError) as exc:
            return self._fallback(
                f"invalid_input: {exc}",
                started,
                supply_ceiling_psi=supply_ceiling_psi,
            )

        if (np.any(measured_pressure > self.config.pressure_trip_psi)
                or np.any(reservoir > self.config.pressure_trip_psi)):
            return self._fallback(
                "pressure_trip",
                started,
                supply_ceiling_psi=supply_ceiling_psi,
            )
        if sample_age_s < 0.0:
            return self._fallback(
                "invalid_sample_age",
                started,
                supply_ceiling_psi=supply_ceiling_psi,
            )
        if sample_age_s > self.config.stale_after_s:
            return self._fallback(
                "stale_sensor",
                started,
                supply_ceiling_psi=supply_ceiling_psi,
            )

        previous_state = (
            self.previous_measurement,
            self.filtered_measurement_rate,
            self.integral,
        )
        error = float(reference_deg - measured_deg)
        proportional_error = float(
            preview - measured_deg
            if self.use_preview_for_proportional
            else error
        )
        raw_measurement_rate = 0.0
        if self.previous_measurement is not None:
            raw_measurement_rate = (
                float(measured_deg) - self.previous_measurement
            ) / dt
        tau = 1.0 / (2.0 * np.pi * self.config.derivative_cutoff_hz)
        alpha = dt / (tau + dt)
        filtered_rate = self.filtered_measurement_rate + alpha * (
            raw_measurement_rate - self.filtered_measurement_rate
        )
        candidate_integral = float(np.clip(
            self.integral + error * dt,
            -self.config.integral_limit_deg_s,
            self.config.integral_limit_deg_s,
        ))
        effort = (
            self.gains.kp_psi_per_deg * proportional_error
            + self.gains.ki_psi_per_deg_s * candidate_integral
            - self.gains.kd_psi_s_per_deg * filtered_rate
        )
        raw = self.bias_pressure_psi.copy()
        raw[self.controlled_actuator] += (
            self.gains.control_direction * effort
        )
        if not np.all(np.isfinite(raw)):
            return self._fallback(
                "nonfinite_pid",
                started,
                raw=raw,
                supply_ceiling_psi=supply_ceiling_psi,
            )

        try:
            result = self.projector.project(
                raw,
                self.previous_command,
                dt,
                supply_ceiling_psi=supply_ceiling_psi,
            )
        except InfeasiblePressureConstraints as exc:
            return self._fallback(
                f"infeasible_projection: {exc}",
                started,
                raw=raw,
                supply_ceiling_psi=supply_ceiling_psi,
            )
        except (TypeError, ValueError) as exc:
            return self._fallback(
                f"invalid_projection: {exc}",
                started,
                raw=raw,
                supply_ceiling_psi=supply_ceiling_psi,
            )

        elapsed = perf_counter() - started
        if self.config.max_tick_s is not None and elapsed > self.config.max_tick_s:
            (
                self.previous_measurement,
                self.filtered_measurement_rate,
                self.integral,
            ) = previous_state
            return self._fallback(
                "missed_deadline",
                started,
                raw=raw,
                supply_ceiling_psi=supply_ceiling_psi,
            )

        # Back-calculation prevents both hard-bound and slew-limit windup.
        if result.projected and self.gains.ki_psi_per_deg_s > 1e-12:
            output_error = (
                result.command_psi[self.controlled_actuator]
                - raw[self.controlled_actuator]
            )
            candidate_integral += (
                self.config.antiwindup_gain
                * self.gains.control_direction
                * output_error
                / self.gains.ki_psi_per_deg_s
                * dt
            )
            candidate_integral = float(np.clip(
                candidate_integral,
                -self.config.integral_limit_deg_s,
                self.config.integral_limit_deg_s,
            ))

        self.previous_measurement = float(measured_deg)
        self.filtered_measurement_rate = float(filtered_rate)
        self.integral = candidate_integral
        self.previous_command = result.command_psi.copy()
        return PIDStep(
            command_psi=result.command_psi,
            raw_command_psi=raw,
            error_deg=error,
            filtered_measurement_rate_deg_s=self.filtered_measurement_rate,
            integral_deg_s=self.integral,
            projected=result.projected,
            slew_relaxed=result.slew_relaxed,
            fallback=False,
            gate_reason="ok",
            elapsed_s=elapsed,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "gains": asdict(self.gains),
            "config": asdict(self.config),
            "bias_pressure_psi": self.bias_pressure_psi.tolist(),
            "controlled_actuator": self.controlled_actuator,
            "use_preview_for_proportional": self.use_preview_for_proportional,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PIDController":
        payload = json.loads(Path(path).read_text())
        version = int(payload.get("format_version", -1))
        if version != cls.FORMAT_VERSION:
            raise ValueError(f"unsupported PID file version {version}")
        return cls(
            gains=PIDGains(**payload["gains"]),
            config=PRCConfig(**payload["config"]),
            bias_pressure_psi=payload["bias_pressure_psi"],
            controlled_actuator=int(payload["controlled_actuator"]),
            use_preview_for_proportional=bool(
                payload.get("use_preview_for_proportional", True)
            ),
            metadata=payload.get("metadata", {}),
        )
