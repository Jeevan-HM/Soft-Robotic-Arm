"""Fit and validate MuJoCo behaviour against the physical-arm recordings.

The public experiment matrix is slow (0.1 Hz), so the calibration is staged:

1. measured pouch pressures are replayed directly to identify mechanics;
2. desired-to-measured actuator dynamics are represented by a delay, gain,
   bias, and first-order time constant; and
3. the stored calibration is replayed end-to-end for validation.

By default this command evaluates ``robot_calibration.json``.  Pass
``--optimize`` to rerun the bounded differential-evolution mechanics fit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from mjcf_model import ArmConfig
from calibration import DEFAULT_CALIBRATION_PATH, RobotCalibration
from experiment_data import (
    RobotExperimentData,
    load_robot_experiment_csv,
    parse_experiment_metadata,
)
from simulator import SoftArmSim


PARAMETER_NAMES = (
    "base_stiffness",
    "base_damping",
    "axial_stiffness",
    "axial_damping",
    "pressure_gain",
    "extension_gain",
    "stiffness_per_psi",
)

PARAMETER_BOUNDS = (
    (0.30, 3.00),
    (0.01, 0.40),
    (150.0, 1200.0),
    (2.0, 100.0),
    (0.05, 0.80),
    (0.01, 0.15),
    (0.00, 0.80),
)

# Balanced across topology, waveform, charge, and maximum pressure.  The
# remaining 30 conditions act as out-of-selection validation data.
FIT_CONDITIONS = (
    ("coupled", "axial", 1.0, 10.0),
    ("coupled", "circular", 2.0, 5.0),
    ("coupled", "triangular", 3.0, 10.0),
    ("parallel", "axial", 2.0, 10.0),
    ("parallel", "circular", 1.0, 5.0),
    ("parallel", "triangular", 3.0, 5.0),
)


@dataclass(frozen=True)
class CalibrationClip:
    path: Path
    topology: str
    waveform: str
    charge_psi: float
    max_psi: float
    sample_rate_hz: float
    time_s: np.ndarray
    commands_psi: np.ndarray
    measured_pressures_psi: np.ndarray
    reservoir_pressures_psi: np.ndarray
    tip_vector_m: np.ndarray
    prehistory_commands_psi: np.ndarray

    @property
    def full_measured_pressures_psi(self) -> np.ndarray:
        return np.column_stack(
            (self.reservoir_pressures_psi, self.measured_pressures_psi)
        )

    @property
    def condition(self) -> tuple[str, str, float, float]:
        return self.topology, self.waveform, self.charge_psi, self.max_psi


@dataclass(frozen=True)
class MotionMetrics:
    score: float
    lateral_rmse_mm: float
    axial_rmse_mm: float
    lateral_amplitude_ratio: float


@dataclass(frozen=True)
class CommandReplay:
    tip_vector_m: np.ndarray
    actuator_pressures_psi: np.ndarray
    reservoir_pressures_psi: np.ndarray


def discover_experiments(data_dir: str | Path) -> list[Path]:
    paths = sorted(Path(data_dir).glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no experiment CSVs found below {Path(data_dir)!s}"
        )
    seen: set[tuple[str, str, float, float]] = set()
    for path in paths:
        meta = parse_experiment_metadata(path)
        key = (meta.topology, meta.waveform, meta.charge_psi, meta.max_psi)
        if key in seen:
            raise ValueError(f"duplicate experiment condition: {key}")
        seen.add(key)
    return paths


def load_clip(
    path: str | Path,
    *,
    sample_rate_hz: float,
    clip_start_s: float,
    clip_duration_s: float,
    max_interpolation_gap_s: float,
) -> CalibrationClip:
    end_s = clip_start_s + clip_duration_s
    data = load_robot_experiment_csv(
        path,
        sample_rate_hz=sample_rate_hz,
        active_duration_s=end_s,
        max_interpolation_gap_s=max_interpolation_gap_s,
    )
    start = int(round(clip_start_s * sample_rate_hz))
    stop = int(round(end_s * sample_rate_hz)) + 1
    selection = slice(start, stop)
    return CalibrationClip(
        path=Path(path),
        topology=data.metadata.topology,
        waveform=data.metadata.waveform,
        charge_psi=data.metadata.charge_psi,
        max_psi=data.metadata.max_psi,
        sample_rate_hz=sample_rate_hz,
        time_s=data.time_s[selection] - data.time_s[start],
        commands_psi=data.commands_psi[selection],
        measured_pressures_psi=data.measured_pressures_psi[selection],
        reservoir_pressures_psi=data.reservoir_pressures_psi[selection],
        tip_vector_m=data.tip_vector_m[selection],
        prehistory_commands_psi=data.commands_psi[:start],
    )


def parameter_vector(config: ArmConfig) -> np.ndarray:
    return np.asarray([getattr(config, name) for name in PARAMETER_NAMES])


def config_from_vector(vector: Iterable[float], base: ArmConfig) -> ArmConfig:
    values = asdict(base)
    for name, value in zip(PARAMETER_NAMES, vector):
        values[name] = float(value)
    # A 2 ms optimization step preserves the 0.1 Hz behaviour while making
    # the global search practical. Stored deployment calibration remains 1 ms.
    values["timestep"] = 0.002
    values["tau_pneumatic"] = values.get("tau_pneumatic", 0.6)
    return ArmConfig(**values)


def replay_measured_pressures(
    clip: CalibrationClip,
    config: ArmConfig,
) -> np.ndarray:
    """Replay measured pressure directly, bypassing regulator dynamics."""
    sim = SoftArmSim(
        cfg=config,
        control_hz=clip.sample_rate_hz,
        sensor_noise_psi=0.0,
    )
    sim.set_pre_inflation(clip.charge_psi)
    points = np.empty((len(clip.time_s), 3), dtype=float)
    try:
        mount = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, "mount"
        )
        for row_index, row in enumerate(clip.full_measured_pressures_psi):
            # Samples at t_i describe the state before pressure p_i acts over
            # [t_i, t_i + dt). Returning the pose first avoids a one-sample
            # pressure lead in the comparison.
            points[row_index] = (
                sim.data.sensordata[:3] - sim.data.xpos[mount]
            )
            pressure = np.empty((config.n_segments, config.n_pouches))
            pressure[0] = row[: config.n_pouches]
            pressure[1:] = row[config.n_pouches :, None]
            for _ in range(sim.n_sub_steps):
                sim.p_actual[:] = pressure
                sim._apply_pressure_wrench()
                mujoco.mj_step(sim.model, sim.data)
    finally:
        sim.close()
    return points


def replay_commands(
    clip: CalibrationClip,
    calibration: RobotCalibration,
) -> CommandReplay:
    """Replay desired S2--S4 commands through the complete calibrated model."""
    sim = calibration.make_sim(
        topology=clip.topology,
        reservoir_pressure_psi=clip.charge_psi,
        control_hz=clip.sample_rate_hz,
        seed=0,
        sensor_noise_psi=0.0,
    )
    mount = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, "mount"
    )
    tip = np.empty((len(clip.time_s), 3), dtype=float)
    actuator = np.empty((len(clip.time_s), 3), dtype=float)
    reservoir = np.empty((len(clip.time_s), 5), dtype=float)
    try:
        # Preserve the 50 s of command history preceding the scored clip so
        # pneumatic and mechanical states are not reinitialized mid-wave.
        for command in clip.prehistory_commands_psi:
            sim.step(command)
        for index, command in enumerate(clip.commands_psi):
            observation = sim.observe()
            tip[index] = (
                observation["tip_pos"] - sim.data.xpos[mount]
            )
            actuator[index] = observation["actuator_pressures"]
            reservoir[index] = observation["reservoir_pressures"]
            sim.step(command)
    finally:
        sim.close()
    return CommandReplay(tip, actuator, reservoir)


def _transverse_basis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    reference = next(
        candidate for candidate in np.eye(3)
        if abs(float(candidate @ axis)) < 0.9
    )
    first = reference - axis * float(reference @ axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return np.vstack((first, second))


def fit_global_rotation(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    sample_rate_hz: float,
    warmup_s: float,
) -> np.ndarray:
    """Fit one proper rotation from simulation coordinates to robot RB1.

    Offsets are removed per trial, but the same orientation is frozen for all
    fit and validation conditions. Reflections are explicitly forbidden.
    """
    simulated_rows = []
    real_rows = []
    simulated_axis_rows = []
    real_axis_rows = []
    warmup = int(round(warmup_s * sample_rate_hz))
    for real_points, simulated_points in pairs:
        real = real_points[warmup:]
        simulated = simulated_points[warmup:]
        real_rows.append(real - np.mean(real, axis=0))
        simulated_rows.append(simulated - np.mean(simulated, axis=0))
        real_axis = np.mean(real_points[: max(2, warmup)], axis=0)
        simulated_axis = np.mean(
            simulated_points[: max(2, warmup)], axis=0
        )
        real_axis_rows.append(0.1 * real_axis / np.linalg.norm(real_axis))
        simulated_axis_rows.append(
            0.1 * simulated_axis / np.linalg.norm(simulated_axis)
        )
    real = np.vstack(real_rows)
    simulated = np.vstack(simulated_rows)
    # Anchor the hanging-arm direction as well as centered motion. Without
    # this constraint a proper 3D rotation could imitate a 2D reflection by
    # also flipping the arm's axial direction.
    real = np.vstack((real, real_axis_rows))
    simulated = np.vstack((simulated, simulated_axis_rows))
    u, _, vt = np.linalg.svd(simulated.T @ real)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    if np.linalg.det(rotation) < 1.0 - 1e-8:
        raise RuntimeError("motion-frame fit did not produce a proper rotation")
    return rotation


def motion_comparison(
    real_points: np.ndarray,
    simulated_points: np.ndarray,
    *,
    sample_rate_hz: float,
    warmup_s: float,
    rotation: np.ndarray,
) -> tuple[MotionMetrics, np.ndarray, np.ndarray]:
    warmup = int(round(warmup_s * sample_rate_hz))
    if warmup < 1 or warmup >= len(real_points) - 2:
        raise ValueError("warmup must leave at least three scored samples")
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or np.linalg.det(rotation) < 1.0 - 1e-7):
        raise ValueError("rotation must be orthonormal and proper")
    real_scored = real_points[warmup:]
    sim_scored = simulated_points[warmup:] @ rotation
    real_centered = real_scored - np.mean(real_scored, axis=0)
    sim_centered = sim_scored - np.mean(sim_scored, axis=0)
    axis = np.mean(real_points[: max(2, warmup)], axis=0)
    axis /= np.linalg.norm(axis)
    basis = _transverse_basis(axis)
    real_lateral = real_centered @ basis.T
    sim_lateral = sim_centered @ basis.T
    lateral_error = sim_lateral - real_lateral
    real_axial = real_centered @ axis
    sim_axial = sim_centered @ axis
    lateral_rmse = float(np.sqrt(np.mean(np.sum(lateral_error ** 2, axis=1))))
    axial_rmse = float(np.sqrt(np.mean((sim_axial - real_axial) ** 2)))
    real_amplitude = float(np.sqrt(np.mean(np.sum(real_lateral ** 2, axis=1))))
    sim_amplitude = float(np.sqrt(np.mean(np.sum(sim_lateral ** 2, axis=1))))
    amplitude_ratio = (
        sim_amplitude / real_amplitude
        if real_amplitude > 1e-9 else float("nan")
    )
    score = float(np.sqrt(
        (lateral_rmse / 0.010) ** 2
        + 0.35 * (axial_rmse / 0.005) ** 2
    ))
    return (
        MotionMetrics(
            score=score,
            lateral_rmse_mm=1000.0 * lateral_rmse,
            axial_rmse_mm=1000.0 * axial_rmse,
            lateral_amplitude_ratio=amplitude_ratio,
        ),
        real_lateral,
        sim_lateral,
    )


def fit_mechanics(
    clips: list[CalibrationClip],
    base_config: ArmConfig,
    *,
    seed: int,
    maxiter: int,
    popsize: int,
) -> tuple[ArmConfig, object]:
    selected = [clip for clip in clips if clip.condition in FIT_CONDITIONS]
    missing = set(FIT_CONDITIONS).difference(clip.condition for clip in selected)
    if missing:
        raise ValueError(f"fit set is missing conditions: {sorted(missing)}")

    evaluation = [0]

    def objective(vector: np.ndarray) -> float:
        config = config_from_vector(vector, base_config)
        simulations = []
        for clip in selected:
            simulations.append(replay_measured_pressures(clip, config))
        warmup_s = 0.5 * selected[0].time_s[-1]
        rotation = fit_global_rotation(
            [
                (clip.tip_vector_m, simulated)
                for clip, simulated in zip(selected, simulations)
            ],
            sample_rate_hz=selected[0].sample_rate_hz,
            warmup_s=warmup_s,
        )
        scores = []
        for clip, simulated in zip(selected, simulations):
            metrics, _, _ = motion_comparison(
                clip.tip_vector_m,
                simulated,
                sample_rate_hz=clip.sample_rate_hz,
                warmup_s=0.5 * clip.time_s[-1],
                rotation=rotation,
            )
            scores.append(metrics.score)
        evaluation[0] += 1
        value = float(np.mean(scores))
        if evaluation[0] == 1 or evaluation[0] % 10 == 0:
            print(f"[calibration] evaluation {evaluation[0]:4d}: score={value:.5f}")
        return value

    result = differential_evolution(
        objective,
        PARAMETER_BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        workers=1,
        x0=parameter_vector(base_config),
    )
    return config_from_vector(result.x, base_config), result


def evaluate_mechanics(
    clips: list[CalibrationClip],
    calibrated_config: ArmConfig,
) -> tuple[
    list[dict],
    dict[Path, tuple[np.ndarray, np.ndarray]],
    np.ndarray,
]:
    rows: list[dict] = []
    traces: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    simulations: dict[Path, np.ndarray] = {}
    for clip in clips:
        simulations[clip.path] = replay_measured_pressures(
            clip, calibrated_config
        )
    fit_pairs = [
        (clip.tip_vector_m, simulations[clip.path])
        for clip in clips
        if clip.condition in FIT_CONDITIONS
    ]
    rotation = fit_global_rotation(
        fit_pairs,
        sample_rate_hz=clips[0].sample_rate_hz,
        warmup_s=0.5 * clips[0].time_s[-1],
    )
    for index, clip in enumerate(clips, start=1):
        metrics, real_xy, simulated_xy = motion_comparison(
            clip.tip_vector_m,
            simulations[clip.path],
            sample_rate_hz=clip.sample_rate_hz,
            warmup_s=0.5 * clip.time_s[-1],
            rotation=rotation,
        )
        traces[clip.path] = real_xy, simulated_xy
        rows.append({
            "file": clip.path.name,
            "topology": clip.topology,
            "waveform": clip.waveform,
            "charge_psi": clip.charge_psi,
            "max_psi": clip.max_psi,
            "used_for_fit": clip.condition in FIT_CONDITIONS,
            "motion": asdict(metrics),
        })
        print(
            f"[calibration] validated {index:2d}/{len(clips)} "
            f"{clip.path.name}: lateral={metrics.lateral_rmse_mm:.2f} mm"
        )
    return rows, traces, rotation


def aggregate_metrics(rows: list[dict], key: str, *, fit_value: bool | None) -> dict:
    selected = [
        row for row in rows
        if fit_value is None or row["used_for_fit"] is fit_value
    ]
    return {
        "conditions": len(selected),
        "lateral_rmse_mm": float(np.mean([
            row[key]["lateral_rmse_mm"] for row in selected
        ])),
        "axial_rmse_mm": float(np.mean([
            row[key]["axial_rmse_mm"] for row in selected
        ])),
        "median_lateral_amplitude_ratio": float(np.median([
            row[key]["lateral_amplitude_ratio"] for row in selected
        ])),
        "mean_score": float(np.mean([row[key]["score"] for row in selected])),
    }


def evaluate_end_to_end(
    clips: list[CalibrationClip],
    calibration: RobotCalibration,
    rotation: np.ndarray,
) -> tuple[list[dict], dict[Path, np.ndarray]]:
    rows: list[dict] = []
    traces: dict[Path, np.ndarray] = {}
    for index, clip in enumerate(clips, start=1):
        replay = replay_commands(clip, calibration)
        motion, _, simulated_xy = motion_comparison(
            clip.tip_vector_m,
            replay.tip_vector_m,
            sample_rate_hz=clip.sample_rate_hz,
            warmup_s=0.5 * clip.time_s[-1],
            rotation=rotation,
        )
        scored = slice(len(clip.time_s) // 2, None)
        actuator_rmse = float(np.sqrt(np.mean(
            (replay.actuator_pressures_psi[scored]
             - clip.measured_pressures_psi[scored]) ** 2
        )))
        reservoir_rmse = float(np.sqrt(np.mean(
            (replay.reservoir_pressures_psi[scored]
             - clip.reservoir_pressures_psi[scored]) ** 2
        )))
        traces[clip.path] = simulated_xy
        rows.append({
            "file": clip.path.name,
            "topology": clip.topology,
            "waveform": clip.waveform,
            "charge_psi": clip.charge_psi,
            "max_psi": clip.max_psi,
            "used_for_mechanics_fit": clip.condition in FIT_CONDITIONS,
            "motion": asdict(motion),
            "actuator_pressure_rmse_psi": actuator_rmse,
            "reservoir_pressure_rmse_psi": reservoir_rmse,
        })
        print(
            f"[calibration] end-to-end {index:2d}/{len(clips)} "
            f"{clip.path.name}: lateral={motion.lateral_rmse_mm:.2f} mm, "
            f"active pressure={actuator_rmse:.3f} psi"
        )
    return rows, traces


def aggregate_end_to_end(rows: list[dict]) -> dict:
    return {
        "conditions": len(rows),
        "lateral_rmse_mm": float(np.mean([
            row["motion"]["lateral_rmse_mm"] for row in rows
        ])),
        "axial_rmse_mm": float(np.mean([
            row["motion"]["axial_rmse_mm"] for row in rows
        ])),
        "median_lateral_amplitude_ratio": float(np.median([
            row["motion"]["lateral_amplitude_ratio"] for row in rows
        ])),
        "actuator_pressure_rmse_psi": float(np.mean([
            row["actuator_pressure_rmse_psi"] for row in rows
        ])),
        "reservoir_pressure_rmse_psi": float(np.mean([
            row["reservoir_pressure_rmse_psi"] for row in rows
        ])),
    }


def save_validation_plot(
    path: str | Path,
    clips: list[CalibrationClip],
    traces: dict[Path, tuple[np.ndarray, np.ndarray]],
    end_to_end_rows: list[dict],
    end_to_end_traces: dict[Path, np.ndarray],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    representatives = []
    for waveform in ("axial", "circular", "triangular"):
        candidates = [
            clip for clip in clips
            if clip.topology == "parallel"
            and clip.waveform == waveform
            and clip.charge_psi == 3.0
            and clip.max_psi in (5.0, 10.0)
        ]
        representatives.append(max(candidates, key=lambda clip: clip.max_psi))

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for axis, clip in zip(axes, representatives):
        real, measured_pressure_replay = traces[clip.path]
        end_to_end = end_to_end_traces[clip.path]
        axis.plot(real[:, 0] * 1000, real[:, 1] * 1000,
                  color="black", lw=2.0, label="robot")
        axis.plot(
            measured_pressure_replay[:, 0] * 1000,
            measured_pressure_replay[:, 1] * 1000,
            color="tab:cyan",
            lw=1.2,
            alpha=0.9,
            linestyle="--",
            label="measured-pressure replay",
        )
        axis.plot(end_to_end[:, 0] * 1000, end_to_end[:, 1] * 1000,
                  color="tab:blue", lw=1.5, label="command replay")
        row = next(
            item for item in end_to_end_rows
            if item["file"] == clip.path.name
        )
        axis.set_title(
            f"{clip.waveform.capitalize()}\n"
            f"end-to-end lateral RMSE "
            f"{row['motion']['lateral_rmse_mm']:.1f} mm"
        )
        axis.set_xlabel("transverse 1 [mm]")
        axis.set_ylabel("transverse 2 [mm]")
        axis.axis("equal")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle("Physical robot vs MuJoCo — end-to-end held cycle")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--report", type=Path,
                        default=Path("output/robot_calibration_metrics.json"))
    parser.add_argument("--plot", type=Path,
                        default=Path("output/robot_calibration.png"))
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--clip-start", type=float, default=50.0)
    parser.add_argument("--clip-duration", type=float, default=20.0)
    parser.add_argument("--max-interpolation-gap", type=float, default=0.5)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument(
        "--write-calibration",
        type=Path,
        default=None,
        help="write optimized ArmConfig values into a copy of the calibration JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    calibration = RobotCalibration.load(args.calibration)
    paths = discover_experiments(args.data_dir)
    clips = [
        load_clip(
            path,
            sample_rate_hz=args.sample_rate,
            clip_start_s=args.clip_start,
            clip_duration_s=args.clip_duration,
            max_interpolation_gap_s=args.max_interpolation_gap,
        )
        for path in paths
    ]
    calibrated_config = calibration.make_arm_config()
    optimization = None
    if args.optimize:
        calibrated_config, optimization = fit_mechanics(
            clips,
            calibrated_config,
            seed=args.seed,
            maxiter=args.maxiter,
            popsize=args.popsize,
        )

    evaluation_config = replace(calibrated_config, timestep=0.002)
    runtime_calibration = replace(
        calibration,
        arm_config=asdict(calibrated_config),
    )
    rows, traces, frame_rotation = evaluate_mechanics(
        clips, evaluation_config
    )
    aggregate = {
        "all_conditions": aggregate_metrics(rows, "motion", fit_value=None),
        "fit_conditions": aggregate_metrics(
            rows, "motion", fit_value=True
        ),
        "unselected_conditions": aggregate_metrics(
            rows, "motion", fit_value=False
        ),
    }
    end_to_end_rows, end_to_end_traces = evaluate_end_to_end(
        clips, runtime_calibration, frame_rotation
    )
    end_to_end_aggregate = aggregate_end_to_end(end_to_end_rows)
    report = {
        "source": dict(calibration.source),
        "calibration_file": str(args.calibration),
        "mechanics_parameters": {
            name: float(getattr(calibrated_config, name))
            for name in PARAMETER_NAMES
        },
        "sample_rate_hz": args.sample_rate,
        "clip_start_s": args.clip_start,
        "clip_duration_s": args.clip_duration,
        "score_window_s": [
            args.clip_start + 0.5 * args.clip_duration,
            args.clip_start + args.clip_duration,
        ],
        "metric_definition": {
            "motion": "per-trial mean-centered held-cycle displacement",
            "frame": "one proper 3D rotation fitted on six mechanics conditions",
            "absolute_pose_scored": False,
            "trial_specific_rotation_or_reflection": False,
        },
        "simulation_to_robot_rotation": frame_rotation.tolist(),
        "simulation_to_robot_rotation_determinant": float(
            np.linalg.det(frame_rotation)
        ),
        "optimization": None if optimization is None else {
            "seed": args.seed,
            "evaluations": int(optimization.nfev),
            "iterations": int(optimization.nit),
            "objective": float(optimization.fun),
        },
        "aggregate": aggregate,
        "trials": rows,
        "end_to_end_aggregate": end_to_end_aggregate,
        "end_to_end_trials": end_to_end_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as stream:
        json.dump(report, stream, indent=2)
    save_validation_plot(
        args.plot,
        clips,
        traces,
        end_to_end_rows,
        end_to_end_traces,
    )

    if args.write_calibration is not None:
        with args.calibration.open() as stream:
            payload = json.load(stream)
        payload["arm_config"].update({
            name: float(getattr(calibrated_config, name))
            for name in PARAMETER_NAMES
        })
        payload["validation"] = {
            "mechanics_optimizer": report["optimization"],
            "mechanics_replay": aggregate,
            "end_to_end": end_to_end_aggregate,
        }
        args.write_calibration.parent.mkdir(parents=True, exist_ok=True)
        with args.write_calibration.open("w") as stream:
            json.dump(payload, stream, indent=2)

    mechanics = aggregate["all_conditions"]
    print(
        "[calibration] measured-pressure replay lateral/axial RMSE: "
        f"{mechanics['lateral_rmse_mm']:.3f} / "
        f"{mechanics['axial_rmse_mm']:.3f} mm"
    )
    print(
        "[calibration] end-to-end lateral/axial RMSE: "
        f"{end_to_end_aggregate['lateral_rmse_mm']:.3f} / "
        f"{end_to_end_aggregate['axial_rmse_mm']:.3f} mm"
    )
    print(f"[calibration] report: {args.report}")
    print(f"[calibration] plot:   {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
