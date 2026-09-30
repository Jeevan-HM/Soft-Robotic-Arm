"""Dependency-free reinforcement-learning teacher for the 20-channel arm.

The policy is trained with episodic Cross-Entropy Method (CEM) search.  CEM
uses only scalar rollout returns, so it does not require PID demonstrations,
inverse-model labels, automatic differentiation, or an RL framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from prc import (
    InfeasiblePressureConstraints,
    PRCConfig,
    PressureProjector,
)


ArrayLike = Iterable[float] | np.ndarray
Evaluator = Callable[[np.ndarray], float]

N_CHANNELS = 20
FEATURE_SIZE = 13
ENCODED_SIZE = 8
PARAMETER_SIZE = N_CHANNELS * (ENCODED_SIZE + 1)


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
    """Return a finite one-dimensional vector, accepting a 4-by-5 pressure grid."""
    out = np.asarray(value, dtype=float)
    if size == N_CHANNELS and out.shape == (4, 5):
        out = out.reshape(-1)
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return out.copy()


def default_pressure_config() -> PRCConfig:
    """Return independent per-pouch pressure and slew limits for the arm."""
    return PRCConfig(
        control_hz=100.0,
        history_length=1,
        n_reservoir=N_CHANNELS,
        n_actuators=N_CHANNELS,
        pressure_min_psi=0.0,
        pressure_max_psi=10.0,
        slew_rate_psi_s=6.0,
        max_total_pressure_psi=None,
        initial_command_psi=tuple([3.5] * N_CHANNELS),
        pressure_trip_psi=10.5,
        stale_after_s=0.03,
        max_tick_s=None,
        fallback_mode="hold",
    )


@dataclass(frozen=True)
class RLPolicyConfig:
    """Scaling and state settings for the fixed 13-to-8 policy encoder."""

    encoder_seed: int = 2026
    position_scale_m: tuple[float, float, float] = (0.008, 0.008, 0.004)
    error_rate_scale_m_s: tuple[float, float, float] = (0.02, 0.02, 0.01)
    integral_scale_m_s: tuple[float, float, float] = (0.08, 0.08, 0.04)
    integral_limit_m_s: tuple[float, float, float] = (0.08, 0.08, 0.04)
    derivative_cutoff_hz: float = 3.0
    baseline_pressure_psi: float = 3.5

    def __post_init__(self) -> None:
        for name in (
            "position_scale_m",
            "error_rate_scale_m_s",
            "integral_scale_m_s",
            "integral_limit_m_s",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values")
            if name == "integral_limit_m_s":
                if np.any(value < 0.0):
                    raise ValueError("integral limits must be nonnegative")
            elif np.any(value <= 0.0):
                raise ValueError(f"{name} must be positive")
        if (
            not np.isfinite(self.derivative_cutoff_hz)
            or self.derivative_cutoff_hz <= 0.0
        ):
            raise ValueError("derivative_cutoff_hz must be finite and positive")
        if not np.isfinite(self.baseline_pressure_psi):
            raise ValueError("baseline_pressure_psi must be finite")


@dataclass(frozen=True)
class RLStep:
    """One policy result with a safe command for all twenty pouches."""

    command_psi: np.ndarray
    raw_command_psi: np.ndarray
    feature: np.ndarray
    encoded_feature: np.ndarray
    projected: bool
    slew_relaxed: bool
    fallback: bool
    gate_reason: str
    elapsed_s: float

    @property
    def command_matrix_psi(self) -> np.ndarray:
        """Return the command in segment-by-pouch order."""
        return self.command_psi.reshape(4, 5)


class RLTeacherController:
    """CEM-trainable nonlinear policy with twenty independent pressure outputs."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        output_weights: np.ndarray | None = None,
        output_bias: np.ndarray | None = None,
        pressure_config: PRCConfig | None = None,
        policy_config: RLPolicyConfig | None = None,
        encoder_weights: np.ndarray | None = None,
        encoder_bias: np.ndarray | None = None,
        metadata: dict | None = None,
    ):
        self.config = pressure_config or default_pressure_config()
        self.policy_config = policy_config or RLPolicyConfig()
        if self.config.n_actuators != N_CHANNELS:
            raise ValueError("RL teacher requires exactly 20 pressure outputs")
        if self.config.max_total_pressure_psi is not None:
            raise ValueError(
                "RL teacher uses independent per-channel limits; "
                "max_total_pressure_psi must be None"
            )

        if encoder_weights is None or encoder_bias is None:
            if encoder_weights is not None or encoder_bias is not None:
                raise ValueError("provide both encoder arrays or neither")
            rng = np.random.default_rng(self.policy_config.encoder_seed)
            encoder_weights = rng.normal(
                0.0,
                1.0 / np.sqrt(FEATURE_SIZE),
                size=(ENCODED_SIZE, FEATURE_SIZE),
            )
            encoder_bias = rng.uniform(-0.25, 0.25, size=ENCODED_SIZE)
        self.encoder_weights = np.asarray(encoder_weights, dtype=float).copy()
        self.encoder_bias = np.asarray(encoder_bias, dtype=float).copy()
        if self.encoder_weights.shape != (ENCODED_SIZE, FEATURE_SIZE):
            raise ValueError(
                f"encoder_weights must have shape ({ENCODED_SIZE}, {FEATURE_SIZE})"
            )
        if self.encoder_bias.shape != (ENCODED_SIZE,):
            raise ValueError(f"encoder_bias must have shape ({ENCODED_SIZE},)")

        if output_weights is None:
            output_weights = np.zeros((N_CHANNELS, ENCODED_SIZE))
        if output_bias is None:
            output_bias = np.zeros(N_CHANNELS)
        self.output_weights = np.asarray(output_weights, dtype=float).copy()
        self.output_bias = np.asarray(output_bias, dtype=float).copy()
        if self.output_weights.shape != (N_CHANNELS, ENCODED_SIZE):
            raise ValueError(
                f"output_weights must have shape ({N_CHANNELS}, {ENCODED_SIZE})"
            )
        if self.output_bias.shape != (N_CHANNELS,):
            raise ValueError(f"output_bias must have shape ({N_CHANNELS},)")
        for name, value in (
            ("encoder_weights", self.encoder_weights),
            ("encoder_bias", self.encoder_bias),
            ("output_weights", self.output_weights),
            ("output_bias", self.output_bias),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains a nonfinite value")

        try:
            self.metadata = json.loads(json.dumps(metadata or {}))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc

        self.projector = PressureProjector(self.config)
        if np.any(self.projector.p_max <= self.projector.p_min):
            raise ValueError(
                "each pressure maximum must be strictly greater than its minimum"
            )
        baseline = self.policy_config.baseline_pressure_psi
        if np.any(baseline < self.projector.p_min) or np.any(
            baseline > self.projector.p_max
        ):
            raise ValueError("baseline pressure must lie inside every channel limit")
        self._position_scale = np.asarray(
            self.policy_config.position_scale_m, dtype=float
        )
        self._rate_scale = np.asarray(
            self.policy_config.error_rate_scale_m_s, dtype=float
        )
        self._integral_scale = np.asarray(
            self.policy_config.integral_scale_m_s, dtype=float
        )
        self._integral_limit = np.asarray(
            self.policy_config.integral_limit_m_s, dtype=float
        )
        self.reset()

    @property
    def parameter_size(self) -> int:
        return PARAMETER_SIZE

    def parameter_vector(self) -> np.ndarray:
        """Return the CEM-searchable output layer as one vector."""
        return np.concatenate((self.output_weights.reshape(-1), self.output_bias))

    def set_parameter_vector(self, parameters: ArrayLike) -> None:
        """Replace the CEM-searchable output layer without changing the encoder."""
        value = _vector(parameters, PARAMETER_SIZE, "policy parameters")
        split = N_CHANNELS * ENCODED_SIZE
        self.output_weights[:] = value[:split].reshape(
            N_CHANNELS, ENCODED_SIZE
        )
        self.output_bias[:] = value[split:]

    def reset(self, initial_command_psi: ArrayLike | None = None) -> None:
        """Reset tracking memory and the per-channel slew reference."""
        initial = (
            self.config.initial_command_psi
            if initial_command_psi is None
            else initial_command_psi
        )
        initial_vector = _vector(initial, N_CHANNELS, "initial command")
        self.previous_command = np.clip(
            initial_vector, self.projector.p_min, self.projector.p_max
        )
        self.previous_error: np.ndarray | None = None
        self.filtered_error_rate = np.zeros(3)
        self.integral_error = np.zeros(3)

    def _tracking_feature(
        self,
        target_xyz: np.ndarray,
        current_xyz: np.ndarray,
        preview_xyz: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        error = target_xyz - current_xyz
        raw_rate = np.zeros(3)
        if self.previous_error is not None:
            raw_rate = (error - self.previous_error) / dt
        tau = 1.0 / (2.0 * np.pi * self.policy_config.derivative_cutoff_hz)
        alpha = dt / (tau + dt)
        self.filtered_error_rate += alpha * (
            raw_rate - self.filtered_error_rate
        )
        self.integral_error = np.clip(
            self.integral_error + error * dt,
            -self._integral_limit,
            self._integral_limit,
        )
        self.previous_error = error.copy()
        feature = np.concatenate(
            (
                np.ones(1),
                error / self._position_scale,
                self.filtered_error_rate / self._rate_scale,
                self.integral_error / self._integral_scale,
                (preview_xyz - target_xyz) / self._position_scale,
            )
        )
        if feature.shape != (FEATURE_SIZE,):
            raise AssertionError("internal RL feature dimension error")
        return feature

    def _raw_command(self, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        encoded = np.tanh(self.encoder_weights @ feature + self.encoder_bias)
        latent = self.output_weights @ encoded + self.output_bias

        # Offset the bounded output so a zero output layer produces the
        # configured baseline rather than the midpoint of the pressure box.
        fraction = (
            self.policy_config.baseline_pressure_psi - self.projector.p_min
        ) / (self.projector.p_max - self.projector.p_min)
        if np.any((fraction <= 0.0) | (fraction >= 1.0)):
            baseline_latent = np.where(
                fraction <= 0.0, -20.0, np.where(fraction >= 1.0, 20.0, 0.0)
            )
            interior = (fraction > 0.0) & (fraction < 1.0)
            baseline_latent[interior] = np.arctanh(
                2.0 * fraction[interior] - 1.0
            )
        else:
            baseline_latent = np.arctanh(2.0 * fraction - 1.0)
        bounded = 0.5 * (np.tanh(latent + baseline_latent) + 1.0)
        raw = self.projector.p_min + bounded * (
            self.projector.p_max - self.projector.p_min
        )
        return raw, encoded

    def _fallback(
        self,
        reason: str,
        started: float,
        supply_ceiling_psi: ArrayLike | float | None = None,
        force_vent: bool = False,
    ) -> RLStep:
        command = self.previous_command.copy()
        if force_vent or self.config.fallback_mode == "vent":
            command = np.zeros(N_CHANNELS)
        else:
            try:
                result = self.projector.project(
                    command,
                    command,
                    self.config.dt,
                    supply_ceiling_psi=supply_ceiling_psi,
                )
                command = result.command_psi
            except (InfeasiblePressureConstraints, TypeError, ValueError):
                command = np.zeros(N_CHANNELS)
                reason = f"{reason}; invalid_supply_ceiling"
        self.previous_command = command.copy()
        return RLStep(
            command_psi=command,
            raw_command_psi=command.copy(),
            feature=np.empty(0),
            encoded_feature=np.empty(0),
            projected=True,
            slew_relaxed=True,
            fallback=True,
            gate_reason=reason,
            elapsed_s=perf_counter() - started,
        )

    def compute(
        self,
        target_xyz: ArrayLike,
        current_xyz: ArrayLike,
        preview_target_xyz: ArrayLike | None,
        measured_pressures_psi: ArrayLike,
        *,
        dt: float | None = None,
        sample_age_s: float = 0.0,
        supply_ceiling_psi: ArrayLike | float | None = None,
    ) -> RLStep:
        """Return one safe flat 20-pressure command from Cartesian observations."""
        started = perf_counter()
        try:
            target = _vector(target_xyz, 3, "target_xyz")
            current = _vector(current_xyz, 3, "current_xyz")
            preview = target if preview_target_xyz is None else _vector(
                preview_target_xyz, 3, "preview_target_xyz"
            )
            measured = _vector(
                measured_pressures_psi, N_CHANNELS, "measured_pressures_psi"
            )
            dt_value = self.config.dt if dt is None else float(dt)
            age = float(sample_age_s)
            if (
                not np.isfinite(dt_value)
                or dt_value <= 0.0
                or not np.isfinite(age)
                or age < 0.0
            ):
                raise ValueError("dt must be positive and sample age nonnegative")
        except (TypeError, ValueError) as exc:
            return self._fallback(
                f"invalid_input: {exc}",
                started,
                supply_ceiling_psi,
                force_vent=True,
            )

        if np.any(measured > self.config.pressure_trip_psi):
            return self._fallback(
                "pressure_trip", started, supply_ceiling_psi, force_vent=True
            )
        if age > self.config.stale_after_s:
            return self._fallback("stale_sensor", started, supply_ceiling_psi)

        state_snapshot = (
            None if self.previous_error is None else self.previous_error.copy(),
            self.filtered_error_rate.copy(),
            self.integral_error.copy(),
        )
        try:
            feature = self._tracking_feature(target, current, preview, dt_value)
            raw, encoded = self._raw_command(feature)
            result = self.projector.project(
                raw,
                self.previous_command,
                dt_value,
                supply_ceiling_psi=supply_ceiling_psi,
            )
        except (InfeasiblePressureConstraints, TypeError, ValueError) as exc:
            (
                self.previous_error,
                self.filtered_error_rate,
                self.integral_error,
            ) = state_snapshot
            return self._fallback(
                f"invalid_projection: {exc}",
                started,
                supply_ceiling_psi,
                force_vent=True,
            )

        elapsed = perf_counter() - started
        if self.config.max_tick_s is not None and elapsed > self.config.max_tick_s:
            (
                self.previous_error,
                self.filtered_error_rate,
                self.integral_error,
            ) = state_snapshot
            return self._fallback("missed_deadline", started, supply_ceiling_psi)

        self.previous_command = result.command_psi.copy()
        return RLStep(
            command_psi=result.command_psi,
            raw_command_psi=raw,
            feature=feature,
            encoded_feature=encoded,
            projected=result.projected,
            slew_relaxed=result.slew_relaxed,
            fallback=False,
            gate_reason="ok",
            elapsed_s=elapsed,
        )

    def save(self, path: str | Path) -> Path:
        """Save the complete deterministic policy without pickle objects."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(
                stream,
                format_version=np.array(self.FORMAT_VERSION, dtype=np.int64),
                pressure_config_json=np.array(
                    json.dumps(asdict(self.config), default=_json_numpy)
                ),
                policy_config_json=np.array(
                    json.dumps(asdict(self.policy_config), default=_json_numpy)
                ),
                encoder_weights=self.encoder_weights,
                encoder_bias=self.encoder_bias,
                output_weights=self.output_weights,
                output_bias=self.output_bias,
                metadata_json=np.array(json.dumps(self.metadata)),
            )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RLTeacherController":
        """Load a teacher saved by :meth:`save` with pickle disabled."""
        with np.load(Path(path), allow_pickle=False) as saved:
            version = int(saved["format_version"])
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"unsupported RL teacher file version {version}")
            pressure_config = PRCConfig(
                **json.loads(str(saved["pressure_config_json"]))
            )
            policy_config = RLPolicyConfig(
                **json.loads(str(saved["policy_config_json"]))
            )
            metadata = json.loads(str(saved["metadata_json"]))
            return cls(
                output_weights=saved["output_weights"].copy(),
                output_bias=saved["output_bias"].copy(),
                pressure_config=pressure_config,
                policy_config=policy_config,
                encoder_weights=saved["encoder_weights"].copy(),
                encoder_bias=saved["encoder_bias"].copy(),
                metadata=metadata,
            )


@dataclass(frozen=True)
class CEMConfig:
    """Deterministic diagonal-CEM settings for episodic policy search."""

    population_size: int = 64
    elite_count: int = 8
    generations: int = 25
    initial_std: float = 0.12
    minimum_std: float = 0.015
    old_distribution_weight: float = 0.2
    seed: int = 7

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")
        if not 1 <= self.elite_count <= self.population_size:
            raise ValueError("elite_count must lie within the population")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if (
            not np.isfinite(self.initial_std)
            or self.initial_std <= 0.0
            or not np.isfinite(self.minimum_std)
            or self.minimum_std <= 0.0
            or self.minimum_std > self.initial_std
        ):
            raise ValueError("CEM standard deviations are invalid")
        if not 0.0 <= self.old_distribution_weight < 1.0:
            raise ValueError("old_distribution_weight must lie in [0, 1)")


@dataclass(frozen=True)
class CEMResult:
    """Best policy and deterministic learning history from CEM search."""

    best_parameters: np.ndarray
    best_score: float
    final_mean: np.ndarray
    final_std: np.ndarray
    generation_best_scores: np.ndarray
    best_so_far_scores: np.ndarray
    generation_mean_scores: np.ndarray


def optimize_cem(
    evaluator: Evaluator,
    parameter_size: int = PARAMETER_SIZE,
    config: CEMConfig | None = None,
    initial_mean: ArrayLike | None = None,
) -> CEMResult:
    """Maximize an episodic evaluator with deterministic diagonal CEM."""
    settings = config or CEMConfig()
    if parameter_size < 1:
        raise ValueError("parameter_size must be positive")
    mean = (
        np.zeros(parameter_size)
        if initial_mean is None
        else _vector(initial_mean, parameter_size, "initial_mean")
    )
    std = np.full(parameter_size, settings.initial_std)
    rng = np.random.default_rng(settings.seed)
    generation_best = np.empty(settings.generations)
    best_so_far = np.empty(settings.generations)
    generation_mean = np.empty(settings.generations)
    # The untrained policy is a valid candidate.  Evaluating it once ensures
    # exploration can never replace a safe baseline with a worse sampled policy.
    best_score = float(evaluator(mean.copy()))
    if not np.isfinite(best_score):
        raise ValueError("CEM evaluator returned a nonfinite score")
    best_parameters = mean.copy()

    for generation in range(settings.generations):
        population = mean + std * rng.standard_normal(
            (settings.population_size, parameter_size)
        )
        scores = np.empty(settings.population_size)
        for index, parameters in enumerate(population):
            score = float(evaluator(parameters.copy()))
            if not np.isfinite(score):
                raise ValueError("CEM evaluator returned a nonfinite score")
            scores[index] = score

        # Stable ordering makes ties deterministic as well as seeded samples.
        order = np.argsort(scores, kind="mergesort")
        elite = population[order[-settings.elite_count :]]
        elite_mean = elite.mean(axis=0)
        elite_std = elite.std(axis=0)
        old = settings.old_distribution_weight
        mean = old * mean + (1.0 - old) * elite_mean
        std = np.maximum(
            settings.minimum_std,
            old * std + (1.0 - old) * elite_std,
        )

        winner = int(order[-1])
        winner_score = float(scores[winner])
        if winner_score > best_score:
            best_score = winner_score
            best_parameters = population[winner].copy()
        generation_best[generation] = winner_score
        best_so_far[generation] = best_score
        generation_mean[generation] = float(scores.mean())

    return CEMResult(
        best_parameters=best_parameters,
        best_score=best_score,
        final_mean=mean,
        final_std=std,
        generation_best_scores=generation_best,
        best_so_far_scores=best_so_far,
        generation_mean_scores=generation_mean,
    )
