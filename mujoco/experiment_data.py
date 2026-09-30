"""Load the physical-arm experiment CSVs into uniformly sampled arrays.

The files under ``data/_data_extract`` contain a roughly 100 Hz control log,
but OptiTrack updates at roughly 50 Hz. Several adjacent log rows therefore
carry the same ``mocap_time_rel_s`` and pose. Control and pressure signals are
interpolated on the logger's ``time`` clock; pose is independently deduplicated
and interpolated on the mocap clock. Both clocks are then sampled at the same
uniform timestamps, avoiding the artificial pressure delay produced by pairing
a later control row with an earlier held mocap frame.

The active experiment begins when Segment 1 is isolated: its desired pressure
drops from the filename's charge/pre-inflation pressure to approximately zero.
Only the following 180 seconds are returned by default.

Rigid-body roles in the recorded setup are:

* Rigid body 1: stationary fixture and base-frame orientation.
* Rigid body 2: top of the soft arm.
* Rigid body 3: tip of the soft arm.

Consequently, ``tip_vector_m`` is RB2 -> RB3, rotated into the RB1 frame.  The
optional ``fixture_to_tip_vector_m`` is RB1 -> RB3 in the same frame and is not
used for arm length or displacement calculations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
import re

import numpy as np


CSV_COLUMNS = (
    "step_id",
    "time",
    "Desired_pressure_segment_1",
    "Desired_pressure_segment_2",
    "Desired_pressure_segment_3",
    "Desired_pressure_segment_4",
    "Measured_pressure_Segment_1_pouch_1",
    "Measured_pressure_Segment_1_pouch_2",
    "Measured_pressure_Segment_1_pouch_3",
    "Measured_pressure_Segment_1_pouch_4",
    "Measured_pressure_Segment_1_pouch_5",
    "Measured_pressure_Segment_2",
    "Measured_pressure_Segment_3",
    "Measured_pressure_Segment_4",
    "Rigid_body_1_x",
    "Rigid_body_1_y",
    "Rigid_body_1_z",
    "Rigid_body_1_qx",
    "Rigid_body_1_qy",
    "Rigid_body_1_qz",
    "Rigid_body_1_qw",
    "Rigid_body_2_x",
    "Rigid_body_2_y",
    "Rigid_body_2_z",
    "Rigid_body_2_qx",
    "Rigid_body_2_qy",
    "Rigid_body_2_qz",
    "Rigid_body_2_qw",
    "Rigid_body_3_x",
    "Rigid_body_3_y",
    "Rigid_body_3_z",
    "Rigid_body_3_qx",
    "Rigid_body_3_qy",
    "Rigid_body_3_qz",
    "Rigid_body_3_qw",
    "mocap_time_rel_s",
)

_FILENAME_RE = re.compile(
    r"^(?P<waveform>axial|circular|triangular|triangle)_"
    r"(?P<charge>\d+(?:\.\d+)?)-(?P<maximum>\d+(?:\.\d+)?)_"
    r"(?P<topology>coupled|parallel)\.csv$",
    re.IGNORECASE,
)


class RobotExperimentDataError(ValueError):
    """Raised when a robot experiment cannot be interpreted safely."""


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata encoded by an experiment filename plus load-time details."""

    source_path: Path
    topology: str
    waveform: str
    charge_psi: float
    max_psi: float
    active_start_s: float | None = None
    active_duration_s: float | None = None
    sample_rate_hz: float | None = None
    raw_row_count: int | None = None
    unique_mocap_samples: int | None = None

    @property
    def pre_inflation_psi(self) -> float:
        """Alias describing the Segment-1 charge used before isolation."""

        return self.charge_psi

    @property
    def max_pressure_psi(self) -> float:
        """Alias for the waveform's filename-encoded maximum pressure."""

        return self.max_psi


@dataclass(frozen=True)
class RobotExperimentData:
    """Uniform, active-window signals from one physical-arm experiment.

    Shapes are ``(N,)`` for time and axial displacement, ``(N, 3)`` for
    commands, active pressure measurements, and tip vectors, ``(N, 5)`` for
    reservoir pouch measurements, and ``(N, 2)`` for transverse displacement.
    Segment order for the three-channel arrays is always S2, S3, S4.
    """

    time_s: np.ndarray
    commands_psi: np.ndarray
    measured_pressures_psi: np.ndarray
    reservoir_pressures_psi: np.ndarray
    tip_vector_m: np.ndarray
    axial_displacement_m: np.ndarray
    transverse_displacement_m: np.ndarray
    initial_axis: np.ndarray
    transverse_basis: np.ndarray
    metadata: ExperimentMetadata
    fixture_to_tip_vector_m: np.ndarray | None = None

    @property
    def commands_s2_s4_psi(self) -> np.ndarray:
        return self.commands_psi

    @property
    def measured_active_pressures_psi(self) -> np.ndarray:
        return self.measured_pressures_psi

    @property
    def reservoir_pouches_psi(self) -> np.ndarray:
        return self.reservoir_pressures_psi

    @property
    def arm_vector_m(self) -> np.ndarray:
        """Alias emphasizing that ``tip_vector_m`` is RB2 -> RB3."""

        return self.tip_vector_m


def parse_experiment_metadata(path: str | Path) -> ExperimentMetadata:
    """Parse ``<waveform>_<charge>-<max>_<topology>.csv`` metadata.

    ``triangle`` and ``triangular`` are both normalized to ``triangular``.
    If the file is inside a directory named ``coupled`` or ``parallel``, that
    directory must agree with the filename.
    """

    source = Path(path)
    match = _FILENAME_RE.fullmatch(source.name)
    if match is None:
        raise RobotExperimentDataError(
            "experiment filename must match "
            "'<axial|circular|triangular>_<charge>-<max>_"
            "<coupled|parallel>.csv'; got "
            f"{source.name!r}"
        )

    waveform = match.group("waveform").lower()
    if waveform == "triangle":
        waveform = "triangular"
    topology = match.group("topology").lower()
    charge = float(match.group("charge"))
    maximum = float(match.group("maximum"))
    if charge <= 0.0 or maximum <= 0.0:
        raise RobotExperimentDataError(
            f"filename pressures must be positive; got {charge:g}-{maximum:g} psi"
        )
    if maximum < charge:
        raise RobotExperimentDataError(
            "filename maximum pressure must not be below charge pressure; "
            f"got {charge:g}-{maximum:g} psi"
        )

    parent_topology = source.parent.name.lower()
    if parent_topology in {"coupled", "parallel"} and parent_topology != topology:
        raise RobotExperimentDataError(
            f"filename topology {topology!r} disagrees with parent directory "
            f"{parent_topology!r}"
        )

    return ExperimentMetadata(
        source_path=source,
        topology=topology,
        waveform=waveform,
        charge_psi=charge,
        max_psi=maximum,
    )


def load_robot_experiment_csv(
    path: str | Path,
    *,
    sample_rate_hz: float = 100.0,
    active_duration_s: float = 180.0,
    zero_threshold_psi: float = 0.1,
    max_interpolation_gap_s: float | None = 0.25,
) -> RobotExperimentData:
    """Load, synchronize, and resample one robot experiment CSV.

    Blank or nonfinite signal values are linearly interpolated only when the
    requested window remains bracketed by finite observations and the finite
    support does not contain a gap larger than ``max_interpolation_gap_s``.
    Malformed non-numeric cells, missing columns, unbracketed boundaries, long
    gaps, invalid quaternions, and absent Segment-1 isolation transitions raise
    :class:`RobotExperimentDataError` with a signal-specific message.
    """

    metadata = parse_experiment_metadata(path)
    sample_rate_hz = _positive_finite(sample_rate_hz, "sample_rate_hz")
    active_duration_s = _positive_finite(active_duration_s, "active_duration_s")
    zero_threshold_psi = _nonnegative_finite(
        zero_threshold_psi, "zero_threshold_psi"
    )
    if max_interpolation_gap_s is not None:
        max_interpolation_gap_s = _positive_finite(
            max_interpolation_gap_s, "max_interpolation_gap_s"
        )

    columns, raw_row_count = _read_numeric_csv(metadata.source_path)
    logger_time = columns["time"]
    mocap_time = columns["mocap_time_rel_s"]
    desired_s1 = columns["Desired_pressure_segment_1"]
    start_row = _find_active_start(
        desired_s1,
        metadata.charge_psi,
        zero_threshold_psi,
    )
    if not np.isfinite(logger_time[start_row]):
        raise RobotExperimentDataError(
            "the Segment-1 isolation row has a nonfinite logger time"
        )
    active_start_s = float(logger_time[start_row])

    arm_vector_raw, fixture_vector_raw = _base_frame_vectors(columns)
    control_signals = np.column_stack(
        [
            *(columns[f"Desired_pressure_segment_{segment}"] for segment in (2, 3, 4)),
            *(columns[f"Measured_pressure_Segment_{segment}"] for segment in (2, 3, 4)),
            *(
                columns[f"Measured_pressure_Segment_1_pouch_{pouch}"]
                for pouch in range(1, 6)
            ),
        ]
    )
    control_signal_names = (
        "desired pressure S2",
        "desired pressure S3",
        "desired pressure S4",
        "measured pressure S2",
        "measured pressure S3",
        "measured pressure S4",
        "reservoir pouch 1",
        "reservoir pouch 2",
        "reservoir pouch 3",
        "reservoir pouch 4",
        "reservoir pouch 5",
    )
    pose_signals = np.column_stack((arm_vector_raw, fixture_vector_raw))
    pose_signal_names = (
        "arm vector x",
        "arm vector y",
        "arm vector z",
        "fixture-to-tip vector x",
        "fixture-to-tip vector y",
        "fixture-to-tip vector z",
    )

    control_time, unique_control = _deduplicate_last_finite(
        logger_time, control_signals
    )
    pose_time, unique_pose = _deduplicate_last_finite(mocap_time, pose_signals)
    interval_count_float = active_duration_s * sample_rate_hz
    interval_count = int(round(interval_count_float))
    if not np.isclose(interval_count_float, interval_count, rtol=0.0, atol=1e-8):
        raise RobotExperimentDataError(
            "active_duration_s * sample_rate_hz must be an integer so the "
            "uniform grid includes both window boundaries"
        )
    relative_time = np.arange(interval_count + 1, dtype=float) / sample_rate_hz
    target_time = active_start_s + relative_time

    resampled_control = np.empty(
        (relative_time.size, unique_control.shape[1]), dtype=float
    )
    for column_index, signal_name in enumerate(control_signal_names):
        resampled_control[:, column_index] = _interpolate_bracketed(
            control_time,
            unique_control[:, column_index],
            target_time,
            signal_name,
            max_interpolation_gap_s,
        )
    resampled_pose = np.empty(
        (relative_time.size, unique_pose.shape[1]), dtype=float
    )
    for column_index, signal_name in enumerate(pose_signal_names):
        resampled_pose[:, column_index] = _interpolate_bracketed(
            pose_time,
            unique_pose[:, column_index],
            target_time,
            signal_name,
            max_interpolation_gap_s,
        )

    commands = resampled_control[:, 0:3]
    measured = resampled_control[:, 3:6]
    reservoir = resampled_control[:, 6:11]
    tip_vector = resampled_pose[:, 0:3]
    fixture_vector = resampled_pose[:, 3:6]
    initial_axis, transverse_basis = _stable_basis(tip_vector[0])
    displacement = tip_vector - tip_vector[0]
    axial_displacement = displacement @ initial_axis
    transverse_displacement = displacement @ transverse_basis.T

    metadata = replace(
        metadata,
        active_start_s=active_start_s,
        active_duration_s=active_duration_s,
        sample_rate_hz=sample_rate_hz,
        raw_row_count=raw_row_count,
        unique_mocap_samples=int(pose_time.size),
    )
    return RobotExperimentData(
        time_s=relative_time,
        commands_psi=commands,
        measured_pressures_psi=measured,
        reservoir_pressures_psi=reservoir,
        tip_vector_m=tip_vector,
        axial_displacement_m=axial_displacement,
        transverse_displacement_m=transverse_displacement,
        initial_axis=initial_axis,
        transverse_basis=transverse_basis,
        metadata=metadata,
        fixture_to_tip_vector_m=fixture_vector,
    )


def load_robot_experiment(
    path: str | Path,
    **kwargs,
) -> RobotExperimentData:
    """Short alias for :func:`load_robot_experiment_csv`."""

    return load_robot_experiment_csv(path, **kwargs)


def _read_numeric_csv(path: Path) -> tuple[dict[str, np.ndarray], int]:
    if not path.is_file():
        raise RobotExperimentDataError(f"experiment CSV does not exist: {path}")

    values: dict[str, list[float]] = {name: [] for name in CSV_COLUMNS}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = tuple(reader.fieldnames or ())
        missing = [name for name in CSV_COLUMNS if name not in header]
        unexpected = [name for name in header if name not in CSV_COLUMNS]
        if len(header) != len(CSV_COLUMNS) or missing or unexpected:
            details = []
            if len(header) != len(CSV_COLUMNS):
                details.append(f"expected 36 columns, found {len(header)}")
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            raise RobotExperimentDataError(
                "invalid experiment CSV header (" + "; ".join(details) + ")"
            )

        row_count = 0
        for csv_row_number, row in enumerate(reader, start=2):
            if None in row:
                raise RobotExperimentDataError(
                    f"row {csv_row_number} has more values than the 36-column header"
                )
            row_count += 1
            for name in CSV_COLUMNS:
                text = (row.get(name) or "").strip()
                if not text:
                    values[name].append(np.nan)
                    continue
                try:
                    value = float(text)
                except ValueError as exc:
                    raise RobotExperimentDataError(
                        f"row {csv_row_number}, column {name!r} is not numeric: {text!r}"
                    ) from exc
                values[name].append(value if np.isfinite(value) else np.nan)

    if row_count < 2:
        raise RobotExperimentDataError(
            f"experiment CSV must contain at least two data rows; found {row_count}"
        )
    return {name: np.asarray(column, dtype=float) for name, column in values.items()}, row_count


def _find_active_start(
    desired_s1: np.ndarray,
    charge_psi: float,
    zero_threshold_psi: float,
) -> int:
    prefill_threshold = max(0.5 * charge_psi, 2.0 * zero_threshold_psi)
    saw_prefill = False
    for index, pressure in enumerate(desired_s1):
        if not np.isfinite(pressure):
            continue
        if pressure >= prefill_threshold:
            saw_prefill = True
        elif saw_prefill and pressure <= zero_threshold_psi:
            return index
    raise RobotExperimentDataError(
        "could not find the active-window start: Segment-1 desired pressure "
        f"never dropped to <= {zero_threshold_psi:g} psi after reaching its "
        f"{charge_psi:g} psi prefill"
    )


def _base_frame_vectors(
    columns: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rb1_pos = _stack_xyz(columns, 1)
    rb2_pos = _stack_xyz(columns, 2)
    rb3_pos = _stack_xyz(columns, 3)
    rb1_quat = np.column_stack(
        [columns[f"Rigid_body_1_q{component}"] for component in "xyzw"]
    )

    arm_world = rb3_pos - rb2_pos
    fixture_world = rb3_pos - rb1_pos
    arm_base = np.full_like(arm_world, np.nan)
    fixture_base = np.full_like(fixture_world, np.nan)
    for index in range(arm_world.shape[0]):
        if not (
            np.all(np.isfinite(arm_world[index]))
            and np.all(np.isfinite(fixture_world[index]))
            and np.all(np.isfinite(rb1_quat[index]))
        ):
            continue
        rotation = _quaternion_xyzw_to_rotation(rb1_quat[index])
        if rotation is None:
            continue
        # RB1 quaternion maps base-frame vectors into the mocap world frame.
        arm_base[index] = rotation.T @ arm_world[index]
        fixture_base[index] = rotation.T @ fixture_world[index]
    return arm_base, fixture_base


def _stack_xyz(columns: dict[str, np.ndarray], rigid_body: int) -> np.ndarray:
    return np.column_stack(
        [columns[f"Rigid_body_{rigid_body}_{component}"] for component in "xyz"]
    )


def _quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _deduplicate_last_finite(
    timestamps: np.ndarray,
    signals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_time = np.isfinite(timestamps)
    if np.count_nonzero(valid_time) < 2:
        raise RobotExperimentDataError(
            "mocap_time_rel_s must contain at least two finite observations"
        )
    time = timestamps[valid_time]
    data = signals[valid_time]
    order = np.argsort(time, kind="stable")
    time = time[order]
    data = data[order]
    unique_time, first, counts = np.unique(time, return_index=True, return_counts=True)
    deduplicated = np.full((unique_time.size, data.shape[1]), np.nan)
    for group_index, (start, count) in enumerate(zip(first, counts)):
        group = data[start : start + count]
        for column_index in range(data.shape[1]):
            finite = np.flatnonzero(np.isfinite(group[:, column_index]))
            if finite.size:
                deduplicated[group_index, column_index] = group[finite[-1], column_index]
    return unique_time, deduplicated


def _interpolate_bracketed(
    source_time: np.ndarray,
    source_value: np.ndarray,
    target_time: np.ndarray,
    signal_name: str,
    max_gap_s: float | None,
) -> np.ndarray:
    finite = np.isfinite(source_value)
    finite_time = source_time[finite]
    finite_value = source_value[finite]
    if finite_time.size < 2:
        raise RobotExperimentDataError(
            f"{signal_name} has fewer than two finite timestamped observations"
        )
    start = float(target_time[0])
    end = float(target_time[-1])
    tolerance = 1e-9
    if finite_time[0] > start + tolerance or finite_time[-1] < end - tolerance:
        raise RobotExperimentDataError(
            f"{signal_name} does not bracket the active window "
            f"[{start:.6g}, {end:.6g}] s"
        )

    if max_gap_s is not None:
        left = max(0, int(np.searchsorted(finite_time, start, side="right")) - 1)
        right = min(
            finite_time.size - 1,
            int(np.searchsorted(finite_time, end, side="left")),
        )
        support = finite_time[left : right + 1]
        if support.size >= 2:
            largest_gap = float(np.max(np.diff(support)))
            if largest_gap > max_gap_s + tolerance:
                raise RobotExperimentDataError(
                    f"{signal_name} has a {largest_gap:.6g} s missing-data gap, "
                    f"above the allowed {max_gap_s:.6g} s"
                )
    return np.interp(target_time, finite_time, finite_value)


def _stable_basis(initial_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    initial_length = float(np.linalg.norm(initial_vector))
    if not np.isfinite(initial_length) or initial_length <= 1e-9:
        raise RobotExperimentDataError(
            "initial RB2-to-RB3 arm vector is nonfinite or too short to define an axis"
        )
    axis = initial_vector / initial_length
    # Prefer the same base-frame coordinate axis across experiments.  Merely
    # choosing the mathematically least-aligned axis is sensitive to tiny
    # mocap noise when two candidates are nearly tied.
    reference = None
    for candidate in np.eye(3):
        if abs(float(candidate @ axis)) < 0.9:
            reference = candidate
            break
    if reference is None:  # Defensive only: at least two Cartesian axes qualify.
        raise RobotExperimentDataError("could not construct a transverse basis")
    transverse_1 = reference - axis * float(reference @ axis)
    transverse_1 /= np.linalg.norm(transverse_1)
    transverse_2 = np.cross(axis, transverse_1)
    transverse_2 /= np.linalg.norm(transverse_2)
    return axis, np.vstack((transverse_1, transverse_2))


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RobotExperimentDataError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise RobotExperimentDataError(f"{name} must be nonnegative and finite")
    return result


__all__ = [
    "CSV_COLUMNS",
    "ExperimentMetadata",
    "RobotExperimentData",
    "RobotExperimentDataError",
    "load_robot_experiment",
    "load_robot_experiment_csv",
    "parse_experiment_metadata",
]
