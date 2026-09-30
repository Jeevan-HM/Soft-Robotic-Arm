"""Twenty-channel PRC student distilled from an RL teacher.

The student is a deterministic linear readout over causal pressure history and
Cartesian tracking features.  It never calls the teacher at runtime.  Every
output corresponds to one independently supplied pouch in segment-major order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from prc import (
    FeatureNormalizer,
    InfeasiblePressureConstraints,
    PRCConfig,
    PressureProjector,
    fit_ridge_readout,
)


ArrayLike = Iterable[float] | np.ndarray
N_SEGMENTS = 4
N_POUCHES = 5
N_PRESSURES = N_SEGMENTS * N_POUCHES


def _json_numpy(value):
    """Convert NumPy config values while keeping pickle-free JSON storage."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"object of type {type(value).__name__} is not JSON serializable"
    )


def _vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).copy()
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return out


@dataclass(frozen=True)
class TrackingFeatureConfig:
    """Causal feature settings shared by imitation training and deployment."""

    control_hz: float = 100.0
    history_length: int = 8
    derivative_cutoff_hz: float = 5.0
    integral_limit_m_s: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive")
        if self.history_length < 1:
            raise ValueError("history_length must be at least one")
        if (
            not np.isfinite(self.derivative_cutoff_hz)
            or self.derivative_cutoff_hz <= 0.0
        ):
            raise ValueError("derivative_cutoff_hz must be finite and positive")
        if (
            not np.isfinite(self.integral_limit_m_s)
            or self.integral_limit_m_s < 0.0
        ):
            raise ValueError("integral_limit_m_s must be finite and nonnegative")

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def feature_size(self) -> int:
        # bias + pressure history + error/rate/integral/preview delta + pressure
        return 1 + self.history_length * N_PRESSURES + 12 + N_PRESSURES


class TrackingFeatureBuilder:
    """Build causal features from tip tracking and all pouch pressures."""

    def __init__(self, config: TrackingFeatureConfig | None = None):
        self.config = config or TrackingFeatureConfig()
        self._history: deque[np.ndarray] = deque(
            maxlen=self.config.history_length
        )
        self.reset()

    def reset(self, measured_pressures_psi: ArrayLike | None = None) -> None:
        initial = (
            np.zeros(N_PRESSURES)
            if measured_pressures_psi is None
            else _vector(
                measured_pressures_psi,
                N_PRESSURES,
                "measured_pressures_psi",
            )
        )
        self._history.clear()
        for _ in range(self.config.history_length):
            self._history.append(initial.copy())
        self.previous_error: np.ndarray | None = None
        self.filtered_error_rate = np.zeros(3)
        self.integral = np.zeros(3)

    def _snapshot_state(self) -> tuple:
        """Capture mutable causal state for a transactional controller tick."""
        return (
            tuple(item.copy() for item in self._history),
            None if self.previous_error is None else self.previous_error.copy(),
            self.filtered_error_rate.copy(),
            self.integral.copy(),
        )

    def _restore_state(self, snapshot: tuple) -> None:
        """Restore a state returned by :meth:`_snapshot_state`."""
        history, previous_error, filtered_error_rate, integral = snapshot
        self._history.clear()
        self._history.extend(item.copy() for item in history)
        self.previous_error = (
            None if previous_error is None else previous_error.copy()
        )
        self.filtered_error_rate = filtered_error_rate.copy()
        self.integral = integral.copy()

    def update(
        self,
        target_xyz_m: ArrayLike,
        preview_target_xyz_m: ArrayLike,
        measured_xyz_m: ArrayLike,
        measured_pressures_psi: ArrayLike,
        dt: float | None = None,
    ) -> np.ndarray:
        dt = self.config.dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        target = _vector(target_xyz_m, 3, "target_xyz_m")
        preview = _vector(preview_target_xyz_m, 3, "preview_target_xyz_m")
        measured = _vector(measured_xyz_m, 3, "measured_xyz_m")
        pressure = _vector(
            measured_pressures_psi,
            N_PRESSURES,
            "measured_pressures_psi",
        )

        error = target - measured
        raw_rate = (
            np.zeros(3)
            if self.previous_error is None
            else (error - self.previous_error) / dt
        )
        tau = 1.0 / (2.0 * np.pi * self.config.derivative_cutoff_hz)
        alpha = dt / (tau + dt)
        self.filtered_error_rate += alpha * (
            raw_rate - self.filtered_error_rate
        )
        limit = self.config.integral_limit_m_s
        self.integral = np.clip(
            self.integral + error * dt,
            -limit,
            limit,
        )
        self.previous_error = error.copy()
        self._history.appendleft(pressure.copy())

        history = np.concatenate(tuple(self._history))
        feature = np.concatenate(
            (
                np.ones(1),
                history,
                error,
                self.filtered_error_rate,
                self.integral,
                preview - target,
                pressure,
            )
        )
        if feature.shape != (self.config.feature_size,):
            raise AssertionError("internal tracking feature dimension error")
        return feature


@dataclass(frozen=True)
class ImitationStep:
    command_psi: np.ndarray
    raw_command_psi: np.ndarray
    feature: np.ndarray
    projected: bool
    slew_relaxed: bool
    fallback: bool
    gate_reason: str
    elapsed_s: float


class ImitationPRCController:
    """Safe 20-output PRC readout trained from RL actions."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        weights: np.ndarray,
        pressure_config: PRCConfig,
        feature_config: TrackingFeatureConfig,
        normalizer: FeatureNormalizer,
        metadata: dict | None = None,
    ):
        if pressure_config.n_actuators != N_PRESSURES:
            raise ValueError("pressure_config must define 20 actuator channels")
        self.pressure_config = pressure_config
        self.feature_config = feature_config
        self.weights = np.asarray(weights, dtype=float).copy()
        expected = (N_PRESSURES, feature_config.feature_size)
        if self.weights.shape != expected or not np.all(np.isfinite(self.weights)):
            raise ValueError(f"weights must be finite with shape {expected}")
        if (
            normalizer.mean.shape != (feature_config.feature_size - 1,)
            or normalizer.scale.shape != (feature_config.feature_size - 1,)
            or not np.all(np.isfinite(normalizer.mean))
            or not np.all(np.isfinite(normalizer.scale))
            or np.any(normalizer.scale <= 0.0)
        ):
            raise ValueError("normalizer does not match the tracking features")
        try:
            self.metadata = json.loads(json.dumps(metadata or {}))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        self.normalizer = normalizer
        self.features = TrackingFeatureBuilder(feature_config)
        self.projector = PressureProjector(pressure_config)
        self.reset()

    @property
    def config(self) -> PRCConfig:
        """Compatibility alias for shared rollout utilities."""
        return self.pressure_config

    def reset(
        self,
        initial_command_psi: ArrayLike | None = None,
        measured_pressures_psi: ArrayLike | None = None,
    ) -> None:
        initial = (
            self.pressure_config.initial_command_psi
            if initial_command_psi is None
            else initial_command_psi
        )
        self.previous_command = np.clip(
            _vector(initial, N_PRESSURES, "initial_command_psi"),
            self.projector.p_min,
            self.projector.p_max,
        )
        self.features.reset(measured_pressures_psi)

    def _fallback(
        self,
        reason: str,
        started: float,
        raw: np.ndarray | None = None,
        force_vent: bool = False,
    ) -> ImitationStep:
        if force_vent or self.pressure_config.fallback_mode == "vent":
            command = np.zeros(N_PRESSURES)
        else:
            command = self.previous_command.copy()
        self.previous_command = command.copy()
        return ImitationStep(
            command_psi=command,
            raw_command_psi=command.copy() if raw is None else raw.copy(),
            feature=np.empty(0),
            projected=True,
            slew_relaxed=True,
            fallback=True,
            gate_reason=reason,
            elapsed_s=perf_counter() - started,
        )

    def compute(
        self,
        target_xyz_m: ArrayLike,
        measured_xyz_m: ArrayLike,
        measured_pressures_psi: ArrayLike,
        preview_target_xyz_m: ArrayLike | None = None,
        dt: float | None = None,
        sample_age_s: float = 0.0,
    ) -> ImitationStep:
        started = perf_counter()
        try:
            dt = self.feature_config.dt if dt is None else float(dt)
            target = _vector(target_xyz_m, 3, "target_xyz_m")
            preview = (
                target
                if preview_target_xyz_m is None
                else _vector(
                    preview_target_xyz_m,
                    3,
                    "preview_target_xyz_m",
                )
            )
            measured = _vector(measured_xyz_m, 3, "measured_xyz_m")
            pressure = _vector(
                measured_pressures_psi,
                N_PRESSURES,
                "measured_pressures_psi",
            )
            if (
                not np.isfinite(dt)
                or dt <= 0.0
                or not np.isfinite(sample_age_s)
                or sample_age_s < 0.0
            ):
                raise ValueError("dt and sample age must be valid")
        except (TypeError, ValueError) as exc:
            return self._fallback(
                f"invalid_input: {exc}", started, force_vent=True
            )

        if np.any(pressure > self.pressure_config.pressure_trip_psi):
            return self._fallback("pressure_trip", started, force_vent=True)
        if sample_age_s > self.pressure_config.stale_after_s:
            return self._fallback("stale_sensor", started)

        feature_state = self.features._snapshot_state()
        try:
            raw_feature = self.features.update(
                target,
                preview,
                measured,
                pressure,
                dt,
            )
            feature = self.normalizer.transform(raw_feature)
            raw = self.weights @ feature
            if not np.all(np.isfinite(raw)):
                self.features._restore_state(feature_state)
                return self._fallback("nonfinite_readout", started, raw)
            result = self.projector.project(
                raw,
                self.previous_command,
                dt,
            )
        except InfeasiblePressureConstraints as exc:
            self.features._restore_state(feature_state)
            return self._fallback(f"infeasible_projection: {exc}", started)
        except (TypeError, ValueError) as exc:
            self.features._restore_state(feature_state)
            return self._fallback(f"invalid_projection: {exc}", started)

        elapsed = perf_counter() - started
        if (
            self.pressure_config.max_tick_s is not None
            and elapsed > self.pressure_config.max_tick_s
        ):
            self.features._restore_state(feature_state)
            return self._fallback("missed_deadline", started, raw)

        self.previous_command = result.command_psi.copy()
        return ImitationStep(
            command_psi=result.command_psi,
            raw_command_psi=raw,
            feature=feature,
            projected=result.projected,
            slew_relaxed=result.slew_relaxed,
            fallback=False,
            gate_reason="ok",
            elapsed_s=elapsed,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                format_version=np.array(self.FORMAT_VERSION),
                pressure_config_json=np.array(
                    json.dumps(
                        asdict(self.pressure_config), default=_json_numpy
                    )
                ),
                feature_config_json=np.array(
                    json.dumps(asdict(self.feature_config), default=_json_numpy)
                ),
                weights=self.weights,
                feature_mean=self.normalizer.mean,
                feature_scale=self.normalizer.scale,
                metadata_json=np.array(json.dumps(self.metadata)),
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ImitationPRCController":
        with np.load(Path(path), allow_pickle=False) as saved:
            version = int(saved["format_version"])
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"unsupported PRC file version {version}")
            return cls(
                weights=saved["weights"].copy(),
                pressure_config=PRCConfig(
                    **json.loads(str(saved["pressure_config_json"]))
                ),
                feature_config=TrackingFeatureConfig(
                    **json.loads(str(saved["feature_config_json"]))
                ),
                normalizer=FeatureNormalizer(
                    saved["feature_mean"].copy(),
                    saved["feature_scale"].copy(),
                ),
                metadata=json.loads(str(saved["metadata_json"])),
            )


def fit_rl_imitation_readout(
    features: np.ndarray,
    teacher_commands_psi: np.ndarray,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, FeatureNormalizer]:
    """Fit a 20-output PRC readout to applied RL teacher commands."""
    commands = np.asarray(teacher_commands_psi, dtype=float)
    if commands.ndim != 2 or commands.shape[1] != N_PRESSURES:
        raise ValueError("teacher_commands_psi must have shape (samples, 20)")
    return fit_ridge_readout(features, commands, ridge)
