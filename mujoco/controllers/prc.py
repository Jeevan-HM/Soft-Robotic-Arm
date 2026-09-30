"""Physical-reservoir-computing controller for the soft-arm model.

The real-time law follows the project proposal without its optional LLM
residual::

    phi_k = col(1, s_k ... s_{k-L+1}, r_preview, e, de, eta, p_meas)
    p_raw = Wc @ phi_k
    p_cmd = project(p_raw)

``s`` contains the five sealed-pouch pressures from Segment 1 and ``p_cmd``
contains the three absolute pressure setpoints for Segments 2--4.  The
previous command is intentionally used only by the safety projector, never as
a learned feature.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


class PRCError(RuntimeError):
    """Base class for controller/configuration errors."""


class InfeasiblePressureConstraints(PRCError):
    """Raised when even the hard pressure limits have an empty feasible set."""


def _vector(value: float | ArrayLike, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).copy()
    if out.ndim == 0:
        out = np.full(size, float(out))
    if out.shape != (size,):
        raise ValueError(f"{name} must be scalar or shape ({size},), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True)
class PRCConfig:
    """Configuration for the 100 Hz PRC outer loop."""

    control_hz: float = 100.0
    history_length: int = 8
    n_reservoir: int = 5
    n_actuators: int = 3
    derivative_cutoff_hz: float = 5.0
    integral_limit_deg_s: float = 30.0
    antiwindup_gain: float = 1.0
    pressure_min_psi: float | tuple[float, ...] = 0.0
    pressure_max_psi: float | tuple[float, ...] = 9.0
    slew_rate_psi_s: float | tuple[float, ...] = 20.0
    max_total_pressure_psi: float | None = None
    initial_command_psi: float | tuple[float, ...] = 0.0
    pressure_trip_psi: float = 9.5
    stale_after_s: float = 0.03
    max_tick_s: float | None = 0.008
    fallback_mode: str = "vent"

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if self.history_length < 1:
            raise ValueError("history_length must be at least one")
        if self.n_reservoir < 1 or self.n_actuators < 1:
            raise ValueError("reservoir and actuator dimensions must be positive")
        if (not np.isfinite(self.derivative_cutoff_hz)
                or self.derivative_cutoff_hz <= 0):
            raise ValueError("derivative_cutoff_hz must be finite and positive")
        if (not np.isfinite(self.integral_limit_deg_s)
                or self.integral_limit_deg_s < 0):
            raise ValueError("integral_limit_deg_s must be finite and nonnegative")
        if not np.isfinite(self.antiwindup_gain) or self.antiwindup_gain < 0:
            raise ValueError("antiwindup_gain must be finite and nonnegative")
        if not np.isfinite(self.stale_after_s) or self.stale_after_s <= 0:
            raise ValueError("stale_after_s must be finite and positive")
        if not np.isfinite(self.pressure_trip_psi) or self.pressure_trip_psi <= 0:
            raise ValueError("pressure_trip_psi must be finite and positive")
        if (self.max_tick_s is not None
                and (not np.isfinite(self.max_tick_s) or self.max_tick_s <= 0)):
            raise ValueError("max_tick_s must be finite and positive or None")
        if self.fallback_mode not in {"vent", "hold"}:
            raise ValueError("fallback_mode must be 'vent' or 'hold'")

        p_min = _vector(self.pressure_min_psi, self.n_actuators,
                        "pressure_min_psi")
        p_max = _vector(self.pressure_max_psi, self.n_actuators,
                        "pressure_max_psi")
        slew = _vector(self.slew_rate_psi_s, self.n_actuators,
                       "slew_rate_psi_s")
        if np.any(p_min < 0) or np.any(p_max < p_min):
            raise ValueError("pressure limits must satisfy 0 <= min <= max")
        if np.any(slew <= 0):
            raise ValueError("slew_rate_psi_s must be positive")
        if self.max_total_pressure_psi is not None:
            if (not np.isfinite(self.max_total_pressure_psi)
                    or self.max_total_pressure_psi < float(p_min.sum())):
                raise ValueError("invalid max total pressure")
        _vector(self.initial_command_psi, self.n_actuators,
                "initial_command_psi")

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def feature_size(self) -> int:
        # bias + L*5 pouch history + preview/error/derivative/integral + 3 pmeas
        return (1 + self.history_length * self.n_reservoir
                + 4 + self.n_actuators)

    @property
    def integral_feature_index(self) -> int:
        return self.history_length * self.n_reservoir + 4


@dataclass(frozen=True)
class ProjectionResult:
    command_psi: np.ndarray
    projected: bool
    slew_relaxed: bool
    pressure_ceiling_psi: np.ndarray


class PressureProjector:
    """Exact Euclidean projection for box, slew, and optional sum limits.

    No QP package is needed for the current three-variable feasible set.  If a
    coupled total-pressure constraint is configured, its capped-box projection
    is solved by a monotone scalar bisection.  A newly reduced hard pressure
    ceiling is allowed to override slew, as required by the proposal.
    """

    def __init__(self, config: PRCConfig):
        self.config = config
        n = config.n_actuators
        self.p_min = _vector(config.pressure_min_psi, n, "pressure_min_psi")
        self.p_max = _vector(config.pressure_max_psi, n, "pressure_max_psi")
        self.slew = _vector(config.slew_rate_psi_s, n, "slew_rate_psi_s")

    @staticmethod
    def _project_capped_box(raw: np.ndarray, lower: np.ndarray,
                            upper: np.ndarray, total_max: float | None) -> np.ndarray:
        command = np.clip(raw, lower, upper)
        if total_max is None or command.sum() <= total_max + 1e-12:
            return command
        if lower.sum() > total_max + 1e-12:
            raise InfeasiblePressureConstraints(
                "coupled pressure cap is below the active lower bounds"
            )

        # KKT solution: x_i = clip(raw_i - lambda, lower_i, upper_i).
        lo = 0.0
        hi = max(1.0, float(np.max(raw - lower)))
        while np.clip(raw - hi, lower, upper).sum() > total_max:
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.clip(raw - mid, lower, upper).sum() > total_max:
                lo = mid
            else:
                hi = mid
        return np.clip(raw - hi, lower, upper)

    def project(self, raw_command_psi: ArrayLike, previous_command_psi: ArrayLike,
                dt: float, supply_ceiling_psi: float | ArrayLike | None = None
                ) -> ProjectionResult:
        n = self.config.n_actuators
        raw = _vector(raw_command_psi, n, "raw command")
        previous = _vector(previous_command_psi, n, "previous command")
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")

        ceiling = self.p_max.copy()
        if supply_ceiling_psi is not None:
            ceiling = np.minimum(
                ceiling,
                _vector(supply_ceiling_psi, n, "supply ceiling"),
            )
        if np.any(ceiling < self.p_min):
            raise InfeasiblePressureConstraints(
                "supply ceiling is below the hard minimum pressure"
            )

        delta = self.slew * dt
        lower = np.maximum(self.p_min, previous - delta)
        upper = np.minimum(ceiling, previous + delta)
        conflict = lower > upper
        slew_relaxed = bool(np.any(conflict))
        total_max = self.config.max_total_pressure_psi

        # A ceiling drop or sum-limit change can make the slew-constrained set
        # empty. Relax slew first; never relax a pressure ceiling.
        if np.any(conflict):
            # Relax only channels whose downward slew bound is now above the
            # hard ceiling; unaffected channels retain both slew bounds.
            lower[conflict] = self.p_min[conflict]
            upper[conflict] = ceiling[conflict]

        if total_max is not None and lower.sum() > total_max + 1e-12:
            if self.p_min.sum() > total_max + 1e-12:
                raise InfeasiblePressureConstraints(
                    "coupled pressure cap is below the hard minima"
                )
            # Minimum-L2 relaxation of the downward slew bounds: find lambda
            # such that sum(clip(lower-lambda, p_min, lower)) == total_max.
            original_lower = lower.copy()
            lo = 0.0
            hi = max(1.0, float(np.max(original_lower - self.p_min)))
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                candidate = np.maximum(self.p_min, original_lower - mid)
                if candidate.sum() > total_max:
                    lo = mid
                else:
                    hi = mid
            lower = np.maximum(self.p_min, original_lower - hi)
            slew_relaxed = True

        command = self._project_capped_box(raw, lower, upper, total_max)
        return ProjectionResult(
            command_psi=command,
            projected=not np.allclose(command, raw, rtol=0.0, atol=1e-10),
            slew_relaxed=slew_relaxed,
            pressure_ceiling_psi=ceiling,
        )


@dataclass(frozen=True)
class FeatureNormalizer:
    """Training-only standardization for all features except the bias."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def identity(cls, feature_size: int) -> "FeatureNormalizer":
        return cls(np.zeros(feature_size - 1), np.ones(feature_size - 1))

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureNormalizer":
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError("features must have shape (samples, feature_size)")
        if not np.all(np.isfinite(x)):
            raise ValueError("training features contain nonfinite values")
        mean = x[:, 1:].mean(axis=0)
        scale = x[:, 1:].std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return cls(mean, scale)

    def transform(self, feature: np.ndarray) -> np.ndarray:
        x = np.asarray(feature, dtype=float)
        expected = self.mean.size + 1
        if x.shape[-1] != expected:
            raise ValueError(
                f"feature dimension {x.shape[-1]} does not match {expected}"
            )
        out = x.copy()
        out[..., 0] = 1.0
        out[..., 1:] = (out[..., 1:] - self.mean) / self.scale
        return out


class PRCFeatureBuilder:
    """Stateful construction of the causal PRC feature vector."""

    def __init__(self, config: PRCConfig):
        self.config = config
        self._history: deque[np.ndarray] = deque(maxlen=config.history_length)
        self.reset()

    def reset(self, reservoir_pressures: ArrayLike | None = None) -> None:
        self._history.clear()
        if reservoir_pressures is not None:
            s = _vector(reservoir_pressures, self.config.n_reservoir,
                        "reservoir pressures")
            for _ in range(self.config.history_length):
                self._history.append(s.copy())
        self.previous_error: float | None = None
        self.filtered_derivative = 0.0
        self.integral = 0.0
        self._integral_before_update = 0.0
        self._snapshot = None
        self._update_pending = False

    def update(self, reservoir_pressures: ArrayLike, reference_deg: float,
               preview_reference_deg: float, measured_deg: float,
               actuator_pressures_psi: ArrayLike, dt: float) -> np.ndarray:
        s = _vector(reservoir_pressures, self.config.n_reservoir,
                    "reservoir pressures")
        p_meas = _vector(actuator_pressures_psi, self.config.n_actuators,
                         "actuator pressures")
        scalars = np.array([reference_deg, preview_reference_deg, measured_deg, dt],
                           dtype=float)
        if not np.all(np.isfinite(scalars)) or dt <= 0:
            raise ValueError("references, measurement, and dt must be finite")

        # Direct users of the feature builder need not call commit explicitly;
        # beginning another update means the prior sample was accepted.
        if self._update_pending:
            self._snapshot = None
            self._update_pending = False
        self._snapshot = (
            [item.copy() for item in self._history],
            self.previous_error,
            self.filtered_derivative,
            self.integral,
        )
        self._update_pending = True

        if not self._history:
            for _ in range(self.config.history_length):
                self._history.append(s.copy())
        else:
            self._history.appendleft(s.copy())

        error = float(reference_deg - measured_deg)
        raw_derivative = 0.0
        if self.previous_error is not None:
            raw_derivative = (error - self.previous_error) / dt
        tau = 1.0 / (2.0 * np.pi * self.config.derivative_cutoff_hz)
        alpha = dt / (tau + dt)
        self.filtered_derivative += alpha * (
            raw_derivative - self.filtered_derivative
        )
        self.previous_error = error

        self._integral_before_update = self.integral
        limit = self.config.integral_limit_deg_s
        self.integral = float(np.clip(self.integral + error * dt, -limit, limit))

        history = np.concatenate(list(self._history), axis=0)
        feature = np.concatenate((
            np.array([1.0]),
            history,
            np.array([
                preview_reference_deg,
                error,
                self.filtered_derivative,
                self.integral,
            ]),
            p_meas,
        ))
        if feature.shape != (self.config.feature_size,):
            raise AssertionError("internal PRC feature dimension error")
        return feature

    def commit_projection(self, was_projected: bool,
                          integral_correction: float | None = None) -> None:
        """Apply back-calculation (or conditional) integral anti-windup."""
        if (was_projected and integral_correction is not None
                and np.isfinite(integral_correction)):
            limit = self.config.integral_limit_deg_s
            self.integral = float(np.clip(
                self.integral
                + self.config.antiwindup_gain * integral_correction,
                -limit,
                limit,
            ))
        self._snapshot = None
        self._update_pending = False

    def rollback_update(self) -> None:
        """Undo a feature-state update when a safety gate rejects the tick."""
        if not self._update_pending or self._snapshot is None:
            return
        history, previous_error, derivative, integral = self._snapshot
        self._history = deque(
            (item.copy() for item in history),
            maxlen=self.config.history_length,
        )
        self.previous_error = previous_error
        self.filtered_derivative = derivative
        self.integral = integral
        self._integral_before_update = integral
        self._snapshot = None
        self._update_pending = False

    def reject_command_keep_measurement(self) -> None:
        """Keep fresh sensor history but freeze eta for a rejected command."""
        if not self._update_pending:
            return
        self.integral = self._integral_before_update
        self._snapshot = None
        self._update_pending = False


@dataclass(frozen=True)
class PRCStep:
    command_psi: np.ndarray
    raw_command_psi: np.ndarray
    feature: np.ndarray
    projected: bool
    slew_relaxed: bool
    fallback: bool
    gate_reason: str
    elapsed_s: float


class PRCController:
    """Reference-conditioned PRC readout plus deterministic safety gates."""

    FORMAT_VERSION = 1

    def __init__(self, weights: np.ndarray, config: PRCConfig | None = None,
                 normalizer: FeatureNormalizer | None = None,
                 metadata: dict | None = None):
        self.config = config or PRCConfig()
        self.weights = np.asarray(weights, dtype=float)
        expected = (self.config.n_actuators, self.config.feature_size)
        if self.weights.shape != expected:
            raise ValueError(f"Wc must have shape {expected}, got {self.weights.shape}")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Wc contains nonfinite values")
        self.normalizer = normalizer or FeatureNormalizer.identity(
            self.config.feature_size
        )
        if (self.normalizer.mean.shape != (self.config.feature_size - 1,)
                or self.normalizer.scale.shape != (self.config.feature_size - 1,)):
            raise ValueError("normalizer dimension does not match PRC features")
        if (not np.all(np.isfinite(self.normalizer.mean))
                or not np.all(np.isfinite(self.normalizer.scale))
                or np.any(self.normalizer.scale <= 0)):
            raise ValueError("invalid feature normalizer")
        try:
            # JSON round-tripping both validates and detaches nested values.
            self.metadata = json.loads(json.dumps(metadata or {}))
        except (TypeError, ValueError) as exc:
            raise ValueError("controller metadata must be JSON serializable") from exc

        self.features = PRCFeatureBuilder(self.config)
        self.projector = PressureProjector(self.config)
        self.reset()

    def reset(self, initial_command_psi: ArrayLike | None = None,
              reservoir_pressures: ArrayLike | None = None) -> None:
        self.features.reset(reservoir_pressures)
        initial = (self.config.initial_command_psi if initial_command_psi is None
                   else initial_command_psi)
        self.previous_command = _vector(
            initial, self.config.n_actuators, "initial command"
        )
        self.previous_command = np.clip(
            self.previous_command,
            self.projector.p_min,
            self.projector.p_max,
        )

    def _fallback(self, reason: str, started: float,
                  raw: np.ndarray | None = None,
                  feature: np.ndarray | None = None,
                  supply_ceiling_psi: float | ArrayLike | None = None) -> PRCStep:
        # If feature construction completed, the measurement itself is still
        # useful even though the command is rejected. Keep causal history and
        # derivative state, but do not integrate error during fallback.
        self.features.reject_command_keep_measurement()
        ceiling = self.projector.p_max.copy()
        hard_faults = (
            "invalid_input",
            "invalid_projection",
            "infeasible_projection",
            "nonfinite_readout",
            "pressure_trip",
        )
        force_vent = reason.startswith(hard_faults)
        try:
            if supply_ceiling_psi is not None:
                ceiling = np.minimum(
                    ceiling,
                    _vector(supply_ceiling_psi, self.config.n_actuators,
                            "supply ceiling"),
                )
        except (TypeError, ValueError):
            # The ceiling itself is invalid, so the fail-safe choice is vent.
            ceiling = self.projector.p_max.copy()
            force_vent = True
            reason = f"{reason}; invalid_supply_ceiling"

        emergency_vent = self.config.fallback_mode == "vent" or force_vent
        if emergency_vent:
            # Vent is an emergency state outside the nominal operating box:
            # zero gauge pressure intentionally overrides p_min and slew.
            command = np.zeros(self.config.n_actuators)
        elif np.any(ceiling < self.projector.p_min):
            # No command can meet both bounds. The pressure ceiling wins and
            # an emergency zero/ceiling command is returned.
            command = np.minimum(np.zeros(self.config.n_actuators), ceiling)
            command = np.maximum(command, 0.0)
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
        return PRCStep(
            command_psi=command,
            raw_command_psi=(command.copy() if raw is None else raw),
            feature=(np.empty(0) if feature is None else feature),
            projected=True,
            slew_relaxed=True,
            fallback=True,
            gate_reason=reason,
            elapsed_s=perf_counter() - started,
        )

    def compute(self, reference_deg: float, measured_deg: float,
                reservoir_pressures: ArrayLike,
                actuator_pressures_psi: ArrayLike,
                preview_reference_deg: float | None = None,
                dt: float | None = None, sample_age_s: float = 0.0,
                supply_ceiling_psi: float | ArrayLike | None = None) -> PRCStep:
        started = perf_counter()

        try:
            dt = self.config.dt if dt is None else float(dt)
            preview = (reference_deg if preview_reference_deg is None
                       else preview_reference_deg)
            s = _vector(reservoir_pressures, self.config.n_reservoir,
                        "reservoir pressures")
            p_meas = _vector(actuator_pressures_psi, self.config.n_actuators,
                             "actuator pressures")
            if not np.all(np.isfinite([reference_deg, measured_deg, preview, dt,
                                       sample_age_s])):
                raise ValueError("nonfinite controller input")
        except (TypeError, ValueError) as exc:
            return self._fallback(f"invalid_input: {exc}", started,
                                  supply_ceiling_psi=supply_ceiling_psi)

        # A recorded overpressure remains a hard fault even if the sample is
        # also late; do not let the ordinary stale-data hold policy mask it.
        if (np.any(p_meas > self.config.pressure_trip_psi)
                or np.any(s > self.config.pressure_trip_psi)):
            return self._fallback("pressure_trip", started,
                                  supply_ceiling_psi=supply_ceiling_psi)
        if sample_age_s < 0:
            return self._fallback("invalid_sample_age", started,
                                  supply_ceiling_psi=supply_ceiling_psi)
        if sample_age_s > self.config.stale_after_s:
            return self._fallback("stale_sensor", started,
                                  supply_ceiling_psi=supply_ceiling_psi)

        try:
            raw_feature = self.features.update(
                s, float(reference_deg), float(preview), float(measured_deg),
                p_meas, dt,
            )
            feature = self.normalizer.transform(raw_feature)
            raw = self.weights @ feature
            if not np.all(np.isfinite(raw)):
                return self._fallback("nonfinite_readout", started,
                                      raw=raw, feature=feature,
                                      supply_ceiling_psi=supply_ceiling_psi)
            result = self.projector.project(
                raw, self.previous_command, dt,
                supply_ceiling_psi=supply_ceiling_psi,
            )
        except InfeasiblePressureConstraints as exc:
            return self._fallback(
                f"infeasible_projection: {exc}", started,
                supply_ceiling_psi=supply_ceiling_psi,
            )
        except (TypeError, ValueError) as exc:
            return self._fallback(
                f"invalid_projection: {exc}", started,
                supply_ceiling_psi=supply_ceiling_psi,
            )

        elapsed = perf_counter() - started
        if self.config.max_tick_s is not None and elapsed > self.config.max_tick_s:
            return self._fallback("missed_deadline", started,
                                  raw=raw, feature=feature,
                                  supply_ceiling_psi=supply_ceiling_psi)

        integral_correction = None
        if result.projected:
            index = self.config.integral_feature_index
            sensitivity = (
                self.weights[:, index] / self.normalizer.scale[index - 1]
            )
            sensitivity_norm_sq = float(sensitivity @ sensitivity)
            if sensitivity_norm_sq > 1e-12:
                integral_correction = float(
                    sensitivity @ (result.command_psi - raw)
                    / sensitivity_norm_sq
                )
        self.features.commit_projection(
            result.projected,
            integral_correction=integral_correction,
        )

        self.previous_command = result.command_psi.copy()
        return PRCStep(
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
        """Save weights, training normalization, and controller settings."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                format_version=np.array(self.FORMAT_VERSION),
                config_json=np.array(json.dumps(asdict(self.config))),
                weights=self.weights,
                feature_mean=self.normalizer.mean,
                feature_scale=self.normalizer.scale,
                metadata_json=np.array(json.dumps(self.metadata)),
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PRCController":
        with np.load(Path(path), allow_pickle=False) as saved:
            version = int(saved["format_version"])
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"unsupported PRC file version {version}")
            config = PRCConfig(**json.loads(str(saved["config_json"])))
            normalizer = FeatureNormalizer(
                saved["feature_mean"].copy(), saved["feature_scale"].copy()
            )
            metadata = (
                json.loads(str(saved["metadata_json"]))
                if "metadata_json" in saved.files
                else {}
            )
            return cls(saved["weights"].copy(), config, normalizer, metadata)


def fit_ridge_readout(features: np.ndarray, target_pressures_psi: np.ndarray,
                      ridge: float = 1e-3
                      ) -> tuple[np.ndarray, FeatureNormalizer]:
    """Fit ``Wc`` for one or more pressure channels with ridge regression."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(target_pressures_psi, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features/targets must be 2-D with matching samples")
    if x.shape[0] < 2 or y.shape[1] < 1:
        raise ValueError("need at least two samples and at least one target")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("training data contain nonfinite values")

    normalizer = FeatureNormalizer.fit(x)
    x_scaled = normalizer.transform(x)
    penalty = np.eye(x_scaled.shape[1]) * ridge
    penalty[0, 0] = 0.0  # do not shrink the pressure bias
    gram = x_scaled.T @ x_scaled + penalty
    rhs = x_scaled.T @ y
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs
    return coefficients.T, normalizer


def quaternion_relative_rotvec(current_wxyz: ArrayLike,
                               home_wxyz: ArrayLike) -> np.ndarray:
    """Shortest home-relative rotation vector for MuJoCo ``wxyz`` quaternions."""
    current = _vector(current_wxyz, 4, "current quaternion")
    home = _vector(home_wxyz, 4, "home quaternion")
    current_norm = np.linalg.norm(current)
    home_norm = np.linalg.norm(home)
    if current_norm < 1e-12 or home_norm < 1e-12:
        raise ValueError("quaternions must have nonzero norm")
    current /= current_norm
    home /= home_norm
    if np.dot(current, home) < 0:
        current = -current

    # conjugate(home) * current
    w1, x1, y1, z1 = home[0], -home[1], -home[2], -home[3]
    w2, x2, y2, z2 = current
    relative = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    relative /= np.linalg.norm(relative)
    if relative[0] < 0:
        relative = -relative
    sin_half = np.linalg.norm(relative[1:])
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, relative[0])
    return relative[1:] * (angle / sin_half)


def signed_bend_angle_deg(current_wxyz: ArrayLike, home_wxyz: ArrayLike,
                          bend_axis_xy: ArrayLike = (1.0, 0.0)) -> float:
    """Project base-to-distal rotation onto a configured horizontal bend axis."""
    axis_xy = _vector(bend_axis_xy, 2, "bend axis")
    norm = np.linalg.norm(axis_xy)
    if norm < 1e-12:
        raise ValueError("bend axis cannot be zero")
    axis = np.array([axis_xy[0] / norm, axis_xy[1] / norm, 0.0])
    return float(np.rad2deg(quaternion_relative_rotvec(
        current_wxyz, home_wxyz
    ) @ axis))
