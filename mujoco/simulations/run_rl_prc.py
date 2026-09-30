"""Train a model-free RL teacher and distil it into a 20-output PRC.

The RL teacher is optimized with the Cross-Entropy Method (CEM).  Candidate
policies receive only scalar rewards from complete MuJoCo rollouts.  The PRC
student is then fitted to the teacher's safe, applied pressure commands and is
evaluated without calling the teacher.

Run a short example with::

    python run_rl_prc.py --generations 2 --population-size 8

The default run writes the teacher, student, metrics, demonstrations, and a
comparison figure to ``output``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import platform
from typing import Iterable

import numpy as np


def _configure_mujoco_backend() -> None:
    """Select a renderer backend that is valid on the current platform.

    MuJoCo validates ``MUJOCO_GL`` while it is imported, even for numeric
    rollouts that never create a renderer.  Correcting a backend left over
    from another platform keeps the command-line pipeline portable.
    """
    system = platform.system()
    configured = os.environ.get("MUJOCO_GL", "").lower()
    if system == "Darwin" and configured not in {"cgl", "glfw"}:
        os.environ["MUJOCO_GL"] = "cgl"
        os.environ.pop("PYOPENGL_PLATFORM", None)
    elif system == "Linux" and configured not in {"egl", "glfw", "osmesa"}:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif system == "Windows" and configured not in {"glfw", "wgl"}:
        os.environ["MUJOCO_GL"] = "glfw"
        os.environ.pop("PYOPENGL_PLATFORM", None)


_configure_mujoco_backend()

from mjcf_model import ArmConfig
from rl_teacher import (
    CEMConfig,
    CEMResult,
    RLTeacherController,
    default_pressure_config,
    optimize_cem,
)
from rl_student import (
    ImitationPRCController,
    N_POUCHES,
    N_PRESSURES,
    N_SEGMENTS,
    TrackingFeatureBuilder,
    TrackingFeatureConfig,
    fit_rl_imitation_readout,
)
from simulator import SoftArmSim


BASELINE_PRESSURE_PSI = 3.5


def _finite_vector(value: Iterable[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return result.copy()


def _pressure_matrix(value: np.ndarray | Iterable[float]) -> np.ndarray:
    """Validate the public plant command contract without broadcasting."""
    matrix = np.asarray(value, dtype=float)
    expected = (N_SEGMENTS, N_POUCHES)
    if matrix.shape != expected:
        raise ValueError(f"pressure command must have shape {expected}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pressure command must contain only finite values")
    return matrix.copy()


class TwentyChannelArm:
    """Strict 4-by-5 command interface around the MuJoCo arm simulation."""

    def __init__(self, simulation: SoftArmSim):
        if (
            simulation.cfg.n_segments != N_SEGMENTS
            or simulation.cfg.n_pouches != N_POUCHES
        ):
            raise ValueError("the RL/PRC pipeline requires a 4-segment, 5-pouch arm")
        if simulation.reservoir_column is not None:
            raise ValueError("all 20 pressure channels must be active")
        self.simulation = simulation

    @property
    def cfg(self) -> ArmConfig:
        return self.simulation.cfg

    @property
    def control_dt(self) -> float:
        return self.simulation.control_dt

    def reset(self, clear_log: bool = True) -> dict:
        return self.simulation.reset(clear_log=clear_log)

    def step(self, pressure_command_psi: np.ndarray) -> dict:
        return self.simulation.step(_pressure_matrix(pressure_command_psi))

    def get_pressure_log(self) -> dict:
        return self.simulation.get_pressure_log()


def make_arm_simulator(
    *,
    control_hz: float = 100.0,
    seed: int = 0,
    actuator_delay_s: float = 0.5,
    arm_config: ArmConfig | None = None,
) -> TwentyChannelArm:
    """Build the strict 20-input plant used for training and evaluation."""
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be finite and positive")
    if not np.isfinite(actuator_delay_s) or actuator_delay_s < 0.0:
        raise ValueError("actuator_delay_s must be finite and nonnegative")
    # These values are the same plant parameters used by the standalone
    # Colab notebook.  Supplying a config remains useful for focused tests,
    # but the public default never loads the older runtime calibration.
    cfg = (
        replace(ArmConfig(), p_max=10.0, tau_pneumatic=0.6)
        if arm_config is None
        else arm_config
    )
    simulation = SoftArmSim(
        cfg=cfg,
        control_hz=control_hz,
        sensor_noise_psi=0.0,
        curvature_coupling=0.0,
        extension_coupling=0.0,
        seed=seed,
        reservoir_column=None,
        actuator_delay_s=actuator_delay_s,
        actuator_pressure_gain=np.ones(N_SEGMENTS),
        actuator_pressure_bias_psi=np.zeros(N_SEGMENTS),
    )
    return TwentyChannelArm(simulation)


@dataclass(frozen=True)
class TrackingScenario:
    """Home-relative Cartesian target and its time-aligned preview."""

    time_s: np.ndarray
    target_xyz_m: np.ndarray
    preview_xyz_m: np.ndarray
    motion: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        target = np.asarray(self.target_xyz_m, dtype=float)
        preview = np.asarray(self.preview_xyz_m, dtype=float)
        motion = np.asarray(self.motion)
        sample_count = time.size
        if time.shape != (sample_count,) or sample_count < 1:
            raise ValueError("scenario time must contain at least one sample")
        if target.shape != (sample_count, 3) or preview.shape != (sample_count, 3):
            raise ValueError("scenario targets must have shape (samples, 3)")
        if motion.shape != (sample_count,):
            raise ValueError("scenario motion labels must have shape (samples,)")
        if not (
            np.all(np.isfinite(time))
            and np.all(np.isfinite(target))
            and np.all(np.isfinite(preview))
        ):
            raise ValueError("scenario arrays must be finite")
        if sample_count > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("scenario time must be strictly increasing")

        # Frozen dataclasses do not make mutable arrays immutable, so keep
        # private copies and mark them read-only.
        for name, value in (
            ("time_s", time),
            ("target_xyz_m", target),
            ("preview_xyz_m", preview),
            ("motion", motion.astype("U12")),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @property
    def sample_count(self) -> int:
        return int(self.time_s.size)


def _triangle_path(phase: np.ndarray, radius_m: float) -> np.ndarray:
    """Return one closed triangular path that starts at the home position."""
    vertices = radius_m * np.array(
        [
            [0.0, 0.0],
            [-1.5, np.sqrt(3.0) / 2.0],
            [-1.5, -np.sqrt(3.0) / 2.0],
            [0.0, 0.0],
        ]
    )
    edge_position = np.clip(phase, 0.0, 1.0) * 3.0
    edge = np.minimum(edge_position.astype(int), 2)
    fraction = edge_position - edge
    return vertices[edge] * (1.0 - fraction[:, None]) + vertices[edge + 1] * fraction[:, None]


def make_tracking_scenario(
    *,
    seconds_per_motion: float = 2.0,
    control_hz: float = 100.0,
    preview_s: float = 0.5,
    seed: int = 0,
) -> TrackingScenario:
    """Create axial, circular, and triangular targets for one episode."""
    if not np.isfinite(seconds_per_motion) or seconds_per_motion <= 0.0:
        raise ValueError("seconds_per_motion must be finite and positive")
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be finite and positive")
    if not np.isfinite(preview_s) or preview_s < 0.0:
        raise ValueError("preview_s must be finite and nonnegative")

    rng = np.random.default_rng(seed)
    count = max(4, int(round(seconds_per_motion * control_hz)))
    phase = np.linspace(0.0, 1.0, count, endpoint=True)
    angle = float(rng.uniform(-np.pi, np.pi))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    xy_radius = 0.004 * float(rng.uniform(0.85, 1.15))
    axial_amplitude = 0.0015 * float(rng.uniform(0.85, 1.15))

    axial = np.zeros((count, 3))
    axial[:, 2] = -0.5 * axial_amplitude * (1.0 - np.cos(2.0 * np.pi * phase))

    theta = 2.0 * np.pi * phase
    circle_xy = xy_radius * np.column_stack((np.cos(theta) - 1.0, np.sin(theta)))
    circle = np.zeros((count, 3))
    circle[:, :2] = circle_xy @ rotation.T

    triangle = np.zeros((count, 3))
    triangle[:, :2] = _triangle_path(phase, xy_radius) @ rotation.T

    target = np.concatenate((axial, circle, triangle), axis=0)
    labels = np.repeat(np.array(["axial", "circular", "triangular"]), count)
    preview_steps = int(round(preview_s * control_hz))
    preview_indices = np.minimum(
        np.arange(target.shape[0]) + preview_steps,
        target.shape[0] - 1,
    )
    preview = target[preview_indices]
    time = np.arange(target.shape[0], dtype=float) / control_hz
    return TrackingScenario(time, target, preview, labels, seed)


@dataclass(frozen=True)
class RewardConfig:
    """Dimensionless rollout reward scales and regularization weights."""

    position_scale_m: tuple[float, float, float] = (0.006, 0.006, 0.003)
    effort_weight: float = 0.002
    command_rate_weight: float = 0.003
    pouch_profile_weight: float = 0.001
    projection_weight: float = 0.01
    fallback_penalty: float = 1.0

    def __post_init__(self) -> None:
        scale = np.asarray(self.position_scale_m, dtype=float)
        if scale.shape != (3,) or np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
            raise ValueError("position_scale_m must contain three positive values")
        for name in (
            "effort_weight",
            "command_rate_weight",
            "pouch_profile_weight",
            "projection_weight",
            "fallback_penalty",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


def tracking_reward(
    target_xyz_m: np.ndarray,
    measured_xyz_m: np.ndarray,
    command_psi: np.ndarray,
    previous_command_psi: np.ndarray,
    raw_command_psi: np.ndarray,
    *,
    dt: float,
    fallback: bool = False,
    config: RewardConfig | None = None,
) -> float:
    """Score one MuJoCo transition using tracking and command regularity."""
    settings = config or RewardConfig()
    target = _finite_vector(target_xyz_m, 3, "target_xyz_m")
    measured = _finite_vector(measured_xyz_m, 3, "measured_xyz_m")
    command = _finite_vector(command_psi, N_PRESSURES, "command_psi")
    previous = _finite_vector(previous_command_psi, N_PRESSURES, "previous_command_psi")
    raw = _finite_vector(raw_command_psi, N_PRESSURES, "raw_command_psi")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    scale = np.asarray(settings.position_scale_m)
    tracking = np.exp(-0.5 * float(np.sum(((target - measured) / scale) ** 2)))
    effort = float(np.mean(((command - BASELINE_PRESSURE_PSI) / 6.5) ** 2))
    command_rate = float(np.mean(((command - previous) / (6.0 * dt)) ** 2))
    matrix = command.reshape(N_SEGMENTS, N_POUCHES)
    pouch_profile = float(np.mean((np.diff(matrix, axis=1) / 10.0) ** 2))
    projection = float(np.mean(((raw - command) / 10.0) ** 2))
    reward = (
        tracking
        - settings.effort_weight * effort
        - settings.command_rate_weight * command_rate
        - settings.pouch_profile_weight * pouch_profile
        - settings.projection_weight * projection
    )
    if fallback:
        reward -= settings.fallback_penalty
    return float(reward)


@dataclass(frozen=True)
class Rollout:
    """Complete controller trajectory on one target scenario."""

    time_s: np.ndarray
    target_xyz_m: np.ndarray
    measured_xyz_m: np.ndarray
    command_psi: np.ndarray
    raw_command_psi: np.ndarray
    measured_pressure_psi: np.ndarray
    reward: np.ndarray
    projected: np.ndarray
    fallback: np.ndarray
    motion: np.ndarray
    home_xyz_m: np.ndarray

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.reward))

    @property
    def tracking_rmse_m(self) -> float:
        error = self.target_xyz_m - self.measured_xyz_m
        return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def _reset_and_settle(
    plant: TwentyChannelArm,
    settle_s: float,
    baseline_pressure_psi: float = BASELINE_PRESSURE_PSI,
) -> tuple[np.ndarray, dict]:
    if not np.isfinite(settle_s) or settle_s < 0.0:
        raise ValueError("settle_s must be finite and nonnegative")
    observation = plant.reset(clear_log=True)
    command = np.full((N_SEGMENTS, N_POUCHES), baseline_pressure_psi)
    steps = int(round(settle_s / plant.control_dt))
    for _ in range(steps):
        observation = plant.step(command)
    home = np.asarray(observation["tip_pos"], dtype=float).copy()
    # Do not mix the settling interval with the controlled episode log.
    plant.simulation.clear_pressure_log()
    return home, observation


def _controller_reset(controller, pressure: np.ndarray) -> None:
    initial = np.full(N_PRESSURES, BASELINE_PRESSURE_PSI)
    if isinstance(controller, ImitationPRCController):
        controller.reset(initial, pressure)
    elif isinstance(controller, RLTeacherController):
        controller.reset(initial)
    else:
        try:
            controller.reset(initial, pressure)
        except TypeError:
            controller.reset(initial)


def _controller_compute(
    controller,
    target: np.ndarray,
    measured: np.ndarray,
    preview: np.ndarray,
    pressure: np.ndarray,
    dt: float,
):
    if isinstance(controller, RLTeacherController):
        return controller.compute(
            target_xyz=target,
            current_xyz=measured,
            preview_target_xyz=preview,
            measured_pressures_psi=pressure,
            dt=dt,
        )
    return controller.compute(
        target_xyz_m=target,
        measured_xyz_m=measured,
        measured_pressures_psi=pressure,
        preview_target_xyz_m=preview,
        dt=dt,
    )


def rollout_controller(
    controller,
    scenario: TrackingScenario,
    *,
    plant: TwentyChannelArm | None = None,
    plant_seed: int = 0,
    control_hz: float = 100.0,
    actuator_delay_s: float = 0.5,
    settle_s: float = 4.0,
    reward_config: RewardConfig | None = None,
) -> Rollout:
    """Run a teacher or a teacher-free PRC student on the MuJoCo plant."""
    arm = plant or make_arm_simulator(
        control_hz=control_hz,
        seed=plant_seed,
        actuator_delay_s=actuator_delay_s,
    )
    if not np.isclose(arm.control_dt, 1.0 / control_hz):
        raise ValueError("plant and rollout control rates do not match")
    home, observation = _reset_and_settle(arm, settle_s)
    pressure = np.asarray(observation["pouch_pressures"], dtype=float).reshape(-1)
    _controller_reset(controller, pressure)

    count = scenario.sample_count
    measured_xyz = np.empty((count, 3))
    command = np.empty((count, N_PRESSURES))
    raw_command = np.empty_like(command)
    measured_pressure = np.empty_like(command)
    rewards = np.empty(count)
    projected = np.empty(count, dtype=bool)
    fallback = np.empty(count, dtype=bool)
    previous_command = np.full(N_PRESSURES, BASELINE_PRESSURE_PSI)

    for index in range(count):
        current = np.asarray(observation["tip_pos"], dtype=float) - home
        pressure = np.asarray(observation["pouch_pressures"], dtype=float).reshape(-1)
        step = _controller_compute(
            controller,
            scenario.target_xyz_m[index],
            current,
            scenario.preview_xyz_m[index],
            pressure,
            arm.control_dt,
        )
        applied = _finite_vector(step.command_psi, N_PRESSURES, "controller command")
        raw = _finite_vector(step.raw_command_psi, N_PRESSURES, "raw controller command")
        observation = arm.step(applied.reshape(N_SEGMENTS, N_POUCHES))
        next_xyz = np.asarray(observation["tip_pos"], dtype=float) - home

        measured_xyz[index] = next_xyz
        command[index] = applied
        raw_command[index] = raw
        measured_pressure[index] = np.asarray(
            observation["pouch_pressures"], dtype=float
        ).reshape(-1)
        projected[index] = bool(step.projected)
        fallback[index] = bool(step.fallback)
        rewards[index] = tracking_reward(
            scenario.target_xyz_m[index],
            next_xyz,
            applied,
            previous_command,
            raw,
            dt=arm.control_dt,
            fallback=bool(step.fallback),
            config=reward_config,
        )
        previous_command = applied

    return Rollout(
        time_s=scenario.time_s.copy(),
        target_xyz_m=scenario.target_xyz_m.copy(),
        measured_xyz_m=measured_xyz,
        command_psi=command,
        raw_command_psi=raw_command,
        measured_pressure_psi=measured_pressure,
        reward=rewards,
        projected=projected,
        fallback=fallback,
        motion=scenario.motion.copy(),
        home_xyz_m=home,
    )


@dataclass(frozen=True)
class DemonstrationDataset:
    """Causal PRC features paired with the teacher's applied actions."""

    features: np.ndarray
    teacher_commands_psi: np.ndarray
    scenario_index: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=float)
        commands = np.asarray(self.teacher_commands_psi, dtype=float)
        scenarios = np.asarray(self.scenario_index, dtype=int)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("features must be a finite two-dimensional array")
        if commands.shape != (features.shape[0], N_PRESSURES):
            raise ValueError("teacher commands must have shape (samples, 20)")
        if scenarios.shape != (features.shape[0],):
            raise ValueError("scenario_index must have shape (samples,)")
        if not np.all(np.isfinite(commands)):
            raise ValueError("teacher commands must be finite")


def collect_teacher_demonstrations(
    teacher: RLTeacherController,
    scenarios: list[TrackingScenario],
    feature_config: TrackingFeatureConfig,
    *,
    control_hz: float = 100.0,
    actuator_delay_s: float = 0.5,
    settle_s: float = 4.0,
    seed: int = 0,
) -> DemonstrationDataset:
    """Collect post-projection RL actions on the teacher's own trajectories."""
    if not scenarios:
        raise ValueError("at least one demonstration scenario is required")
    feature_rows: list[np.ndarray] = []
    command_rows: list[np.ndarray] = []
    scenario_rows: list[int] = []

    for scenario_number, scenario in enumerate(scenarios):
        plant = make_arm_simulator(
            control_hz=control_hz,
            seed=seed + scenario_number,
            actuator_delay_s=actuator_delay_s,
        )
        home, observation = _reset_and_settle(plant, settle_s)
        pressure = np.asarray(observation["pouch_pressures"], dtype=float).reshape(-1)
        teacher.reset(np.full(N_PRESSURES, BASELINE_PRESSURE_PSI))
        builder = TrackingFeatureBuilder(feature_config)
        builder.reset(pressure)

        for index in range(scenario.sample_count):
            current = np.asarray(observation["tip_pos"], dtype=float) - home
            pressure = np.asarray(observation["pouch_pressures"], dtype=float).reshape(-1)
            feature = builder.update(
                scenario.target_xyz_m[index],
                scenario.preview_xyz_m[index],
                current,
                pressure,
                dt=plant.control_dt,
            )
            step = teacher.compute(
                target_xyz=scenario.target_xyz_m[index],
                current_xyz=current,
                preview_target_xyz=scenario.preview_xyz_m[index],
                measured_pressures_psi=pressure,
                dt=plant.control_dt,
            )
            applied = _finite_vector(
                step.command_psi, N_PRESSURES, "teacher applied command"
            )
            feature_rows.append(feature)
            command_rows.append(applied)
            scenario_rows.append(scenario_number)
            observation = plant.step(applied.reshape(N_SEGMENTS, N_POUCHES))

    return DemonstrationDataset(
        features=np.asarray(feature_rows),
        teacher_commands_psi=np.asarray(command_rows),
        scenario_index=np.asarray(scenario_rows),
    )


def fit_prc_student(
    demonstrations: DemonstrationDataset,
    feature_config: TrackingFeatureConfig,
    *,
    control_hz: float = 100.0,
    ridge: float = 1e-2,
    metadata: dict | None = None,
) -> ImitationPRCController:
    """Fit a safe 20-output PRC to the RL teacher demonstrations."""
    pressure_config = replace(default_pressure_config(), control_hz=control_hz)
    weights, normalizer = fit_rl_imitation_readout(
        demonstrations.features,
        demonstrations.teacher_commands_psi,
        ridge=ridge,
    )
    details = {
        "teacher": "model-free CEM policy",
        "samples": int(demonstrations.features.shape[0]),
        "ridge": float(ridge),
        **(metadata or {}),
    }
    return ImitationPRCController(
        weights=weights,
        pressure_config=pressure_config,
        feature_config=feature_config,
        normalizer=normalizer,
        metadata=details,
    )


def train_rl_teacher(
    scenarios: list[TrackingScenario],
    *,
    cem_config: CEMConfig | None = None,
    control_hz: float = 100.0,
    actuator_delay_s: float = 0.5,
    settle_s: float = 4.0,
    reward_config: RewardConfig | None = None,
    seed: int = 0,
) -> tuple[RLTeacherController, CEMResult]:
    """Optimize an RL teacher using only MuJoCo rollout returns."""
    if not scenarios:
        raise ValueError("at least one training scenario is required")
    pressure_config = replace(default_pressure_config(), control_hz=control_hz)
    teacher = RLTeacherController(pressure_config=pressure_config)
    plants = [
        make_arm_simulator(
            control_hz=control_hz,
            seed=seed + index,
            actuator_delay_s=actuator_delay_s,
        )
        for index in range(len(scenarios))
    ]

    def evaluate(parameters: np.ndarray) -> float:
        teacher.set_parameter_vector(parameters)
        returns = [
            rollout_controller(
                teacher,
                scenario,
                plant=plant,
                control_hz=control_hz,
                actuator_delay_s=actuator_delay_s,
                settle_s=settle_s,
                reward_config=reward_config,
            ).mean_reward
            for scenario, plant in zip(scenarios, plants)
        ]
        return float(np.mean(returns))

    result = optimize_cem(
        evaluate,
        parameter_size=teacher.parameter_size,
        config=cem_config,
        initial_mean=teacher.parameter_vector(),
    )
    teacher.set_parameter_vector(result.best_parameters)
    teacher.metadata = {
        "algorithm": "episodic CEM policy search",
        "training_scenarios": len(scenarios),
        "best_training_reward": float(result.best_score),
        "seed": int((cem_config or CEMConfig()).seed),
    }
    return teacher, result


def rollout_metrics(rollout: Rollout) -> dict:
    """Return JSON-ready tracking and safety metrics for one rollout."""
    error = rollout.target_xyz_m - rollout.measured_xyz_m
    if rollout.command_psi.shape[0] > 1:
        dt = float(np.median(np.diff(rollout.time_s)))
        maximum_slew = float(np.max(np.abs(np.diff(rollout.command_psi, axis=0))) / dt)
    else:
        maximum_slew = 0.0
    return {
        "cartesian_rmse_mm": float(np.sqrt(np.mean(np.sum(error * error, axis=1))) * 1000.0),
        "axis_rmse_mm": (np.sqrt(np.mean(error * error, axis=0)) * 1000.0).tolist(),
        "mean_reward": rollout.mean_reward,
        "pressure_min_psi": float(np.min(rollout.command_psi)),
        "pressure_max_psi": float(np.max(rollout.command_psi)),
        "maximum_slew_psi_s": maximum_slew,
        "projection_fraction": float(np.mean(rollout.projected)),
        "fallback_count": int(np.count_nonzero(rollout.fallback)),
    }


def evaluate_teacher_and_student(
    teacher: RLTeacherController,
    student: ImitationPRCController,
    scenario: TrackingScenario,
    *,
    control_hz: float = 100.0,
    actuator_delay_s: float = 0.5,
    settle_s: float = 4.0,
    reward_config: RewardConfig | None = None,
    seed: int = 0,
) -> tuple[dict, Rollout, Rollout]:
    """Evaluate teacher and standalone student on identical held-out targets."""
    baseline = RLTeacherController(
        pressure_config=teacher.config,
        policy_config=teacher.policy_config,
        encoder_weights=teacher.encoder_weights,
        encoder_bias=teacher.encoder_bias,
        metadata={"role": "untrained_baseline"},
    )
    baseline_rollout = rollout_controller(
        baseline,
        scenario,
        plant_seed=seed,
        control_hz=control_hz,
        actuator_delay_s=actuator_delay_s,
        settle_s=settle_s,
        reward_config=reward_config,
    )
    teacher_rollout = rollout_controller(
        teacher,
        scenario,
        plant_seed=seed,
        control_hz=control_hz,
        actuator_delay_s=actuator_delay_s,
        settle_s=settle_s,
        reward_config=reward_config,
    )
    student_rollout = rollout_controller(
        student,
        scenario,
        plant_seed=seed,
        control_hz=control_hz,
        actuator_delay_s=actuator_delay_s,
        settle_s=settle_s,
        reward_config=reward_config,
    )
    baseline_metrics = rollout_metrics(baseline_rollout)
    teacher_metrics = rollout_metrics(teacher_rollout)
    student_metrics = rollout_metrics(student_rollout)
    baseline_rmse = baseline_metrics["cartesian_rmse_mm"]
    teacher_rmse = teacher_metrics["cartesian_rmse_mm"]
    metrics = {
        "untrained_baseline": baseline_metrics,
        "teacher": teacher_metrics,
        "prc_student": student_metrics,
        "teacher_rmse_improvement_percent": float(
            100.0 * (baseline_rmse - teacher_rmse) / baseline_rmse
            if baseline_rmse > 1e-12
            else 0.0
        ),
        "teacher_reward_improvement": float(
            teacher_metrics["mean_reward"] - baseline_metrics["mean_reward"]
        ),
        "time_aligned_command_rmse_psi": float(
            np.sqrt(np.mean((teacher_rollout.command_psi - student_rollout.command_psi) ** 2))
        ),
        "heldout_scenario_seed": int(scenario.seed),
    }
    return metrics, teacher_rollout, student_rollout


@dataclass(frozen=True)
class PipelineConfig:
    """Reproducible training, distillation, and evaluation settings."""

    seed: int = 7
    control_hz: float = 100.0
    actuator_delay_s: float = 0.5
    preview_s: float = 0.5
    seconds_per_motion: float = 2.0
    settle_s: float = 4.0
    training_scenarios: int = 2
    history_length: int = 8
    ridge: float = 1.0
    cem: CEMConfig = field(
        default_factory=lambda: CEMConfig(
            population_size=32,
            elite_count=6,
            generations=12,
        )
    )
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.training_scenarios < 1:
            raise ValueError("training_scenarios must be positive")
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        for name in (
            "control_hz",
            "seconds_per_motion",
            "ridge",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("actuator_delay_s", "preview_s", "settle_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class PipelineArtifacts:
    teacher_path: Path
    student_path: Path
    demonstrations_path: Path
    metrics_path: Path
    figure_path: Path


@dataclass(frozen=True)
class PipelineResult:
    teacher: RLTeacherController
    student: ImitationPRCController
    cem_result: CEMResult
    demonstrations: DemonstrationDataset
    metrics: dict
    teacher_rollout: Rollout
    student_rollout: Rollout
    artifacts: PipelineArtifacts


def _save_comparison_figure(
    path: Path,
    cem_result: CEMResult,
    teacher: Rollout,
    student: Rollout,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    labels = ("x", "y", "z")
    for axis_index, label in enumerate(labels):
        axes[0].plot(
            teacher.time_s,
            1000.0 * teacher.target_xyz_m[:, axis_index],
            linestyle="--",
            linewidth=1.5,
            label=f"target {label}",
        )
        axes[0].plot(
            teacher.time_s,
            1000.0 * teacher.measured_xyz_m[:, axis_index],
            linewidth=1.0,
            label=f"RL {label}",
        )
        axes[0].plot(
            student.time_s,
            1000.0 * student.measured_xyz_m[:, axis_index],
            linewidth=1.0,
            alpha=0.8,
            label=f"PRC {label}",
        )
    axes[0].set_ylabel("home-relative tip [mm]")
    axes[0].set_title("Held-out Cartesian tracking")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(teacher.time_s, teacher.command_psi.mean(axis=1), label="RL mean")
    axes[1].fill_between(
        teacher.time_s,
        teacher.command_psi.min(axis=1),
        teacher.command_psi.max(axis=1),
        alpha=0.18,
        label="RL range",
    )
    axes[1].plot(student.time_s, student.command_psi.mean(axis=1), label="PRC mean")
    axes[1].fill_between(
        student.time_s,
        student.command_psi.min(axis=1),
        student.command_psi.max(axis=1),
        alpha=0.18,
        label="PRC range",
    )
    axes[1].set_ylabel("pressure [psi]")
    axes[1].set_title("Twenty-channel command envelope")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.25)

    generations = np.arange(1, cem_result.best_so_far_scores.size + 1)
    axes[2].plot(generations, cem_result.generation_mean_scores, label="generation mean")
    axes[2].plot(generations, cem_result.best_so_far_scores, label="best so far")
    axes[2].set_xlabel("CEM generation")
    axes[2].set_ylabel("mean rollout reward")
    axes[2].set_title("RL teacher learning curve")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_pipeline_artifacts(
    output_dir: str | Path,
    config: PipelineConfig,
    teacher: RLTeacherController,
    student: ImitationPRCController,
    cem_result: CEMResult,
    demonstrations: DemonstrationDataset,
    metrics: dict,
    teacher_rollout: Rollout,
    student_rollout: Rollout,
) -> PipelineArtifacts:
    """Write pickle-free models, data, metrics, and comparison figure."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = PipelineArtifacts(
        teacher_path=directory / "rl_teacher_cem.npz",
        student_path=directory / "rl_distilled_prc.npz",
        demonstrations_path=directory / "rl_teacher_demonstrations.npz",
        metrics_path=directory / "rl_prc_metrics.json",
        figure_path=directory / "rl_prc_comparison.png",
    )
    teacher.save(artifacts.teacher_path)
    student.save(artifacts.student_path)
    with artifacts.demonstrations_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            features=demonstrations.features,
            teacher_commands_psi=demonstrations.teacher_commands_psi,
            scenario_index=demonstrations.scenario_index,
        )
    payload = {
        "pipeline": asdict(config),
        "metrics": metrics,
        "training": {
            "best_reward": float(cem_result.best_score),
            "generation_best_reward": cem_result.generation_best_scores.tolist(),
            "best_so_far_reward": cem_result.best_so_far_scores.tolist(),
            "generation_mean_reward": cem_result.generation_mean_scores.tolist(),
            "demonstration_samples": int(demonstrations.features.shape[0]),
        },
    }
    artifacts.metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    _save_comparison_figure(
        artifacts.figure_path,
        cem_result,
        teacher_rollout,
        student_rollout,
    )
    return artifacts


def run_pipeline(
    config: PipelineConfig | None = None,
    output_dir: str | Path = "output",
) -> PipelineResult:
    """Train the RL teacher, distil the PRC, evaluate, and save artifacts."""
    settings = config or PipelineConfig()
    training_scenarios = [
        make_tracking_scenario(
            seconds_per_motion=settings.seconds_per_motion,
            control_hz=settings.control_hz,
            preview_s=settings.preview_s,
            seed=settings.seed + 100 + index,
        )
        for index in range(settings.training_scenarios)
    ]
    teacher, cem_result = train_rl_teacher(
        training_scenarios,
        cem_config=settings.cem,
        control_hz=settings.control_hz,
        actuator_delay_s=settings.actuator_delay_s,
        settle_s=settings.settle_s,
        reward_config=settings.reward,
        seed=settings.seed + 1000,
    )
    feature_config = TrackingFeatureConfig(
        control_hz=settings.control_hz,
        history_length=settings.history_length,
        derivative_cutoff_hz=teacher.policy_config.derivative_cutoff_hz,
        integral_limit_m_s=max(teacher.policy_config.integral_limit_m_s),
    )
    demonstrations = collect_teacher_demonstrations(
        teacher,
        training_scenarios,
        feature_config,
        control_hz=settings.control_hz,
        actuator_delay_s=settings.actuator_delay_s,
        settle_s=settings.settle_s,
        seed=settings.seed + 2000,
    )
    student = fit_prc_student(
        demonstrations,
        feature_config,
        control_hz=settings.control_hz,
        ridge=settings.ridge,
        metadata={"training_seed": settings.seed},
    )
    heldout = make_tracking_scenario(
        seconds_per_motion=settings.seconds_per_motion,
        control_hz=settings.control_hz,
        preview_s=settings.preview_s,
        seed=settings.seed + 10_000,
    )
    metrics, teacher_rollout, student_rollout = evaluate_teacher_and_student(
        teacher,
        student,
        heldout,
        control_hz=settings.control_hz,
        actuator_delay_s=settings.actuator_delay_s,
        settle_s=settings.settle_s,
        reward_config=settings.reward,
        seed=settings.seed + 3000,
    )
    artifacts = save_pipeline_artifacts(
        output_dir,
        settings,
        teacher,
        student,
        cem_result,
        demonstrations,
        metrics,
        teacher_rollout,
        student_rollout,
    )
    return PipelineResult(
        teacher=teacher,
        student=student,
        cem_result=cem_result,
        demonstrations=demonstrations,
        metrics=metrics,
        teacher_rollout=teacher_rollout,
        student_rollout=student_rollout,
        artifacts=artifacts,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a model-free CEM teacher and distil a standalone 20-output PRC."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--seconds-per-motion", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=4.0)
    parser.add_argument("--actuator-delay-seconds", type=float, default=0.5)
    parser.add_argument("--preview-seconds", type=float, default=0.5)
    parser.add_argument("--training-scenarios", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--initial-std", type=float, default=0.12)
    parser.add_argument("--minimum-std", type=float, default=0.015)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    cem = CEMConfig(
        population_size=args.population_size,
        elite_count=args.elite_count,
        generations=args.generations,
        initial_std=args.initial_std,
        minimum_std=args.minimum_std,
        seed=args.seed,
    )
    config = PipelineConfig(
        seed=args.seed,
        control_hz=args.control_hz,
        actuator_delay_s=args.actuator_delay_seconds,
        preview_s=args.preview_seconds,
        seconds_per_motion=args.seconds_per_motion,
        settle_s=args.settle_seconds,
        training_scenarios=args.training_scenarios,
        history_length=args.history_length,
        ridge=args.ridge,
        cem=cem,
    )
    result = run_pipeline(config, args.output_dir)
    print(f"RL teacher best training reward: {result.cem_result.best_score:.6f}")
    print(
        "Held-out Cartesian RMSE [mm]: "
        f"baseline={result.metrics['untrained_baseline']['cartesian_rmse_mm']:.3f}, "
        f"RL={result.metrics['teacher']['cartesian_rmse_mm']:.3f}, "
        f"PRC={result.metrics['prc_student']['cartesian_rmse_mm']:.3f}"
    )
    print(
        "Held-out command imitation RMSE [psi]: "
        f"{result.metrics['time_aligned_command_rmse_psi']:.3f}"
    )
    print(f"Saved teacher: {result.artifacts.teacher_path}")
    print(f"Saved standalone PRC: {result.artifacts.student_path}")
    print(f"Saved metrics: {result.artifacts.metrics_path}")
    print(f"Saved comparison: {result.artifacts.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
