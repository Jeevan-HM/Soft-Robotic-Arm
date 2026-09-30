"""Train and demonstrate the PRC controller in the MuJoCo soft-arm model.

This is a simulation-only commissioning scaffold.  It mirrors the proposal's
four training stages without the optional LLM residual:

1. charge and seal Segment 1 (column 0), then safely excite Segments 2--4;
2. fit a delay-aware forward model that is nonlinear in measured state and
   affine in a candidate three-pressure command;
3. solve bounded, slew-aware inverse problems to make pressure labels and fit
   the causal PRC readout; and
4. run student-only closed-loop step, sine, and multisine trajectories.

The default controlled coordinate is the home-relative rotation about MuJoCo
+Y.  That plane uses the sealed Segment-1 column as an informative reservoir;
the middle active command (Segment 3 / column 2) supplies the opposing moment.
For this scalar commissioning task, Segments 2 and 4 are regulated back to the
balanced x-psi bias, removing their unobservable orthogonal-plane nullspace.
A future planar controller can replace that constraint with two bend outputs.

Example
-------
    uv run python run_prc.py --x 3.0

The generated metrics demonstrate software integration in the digital twin;
they are not evidence of physical-arm validation or stability.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
import tempfile
from time import perf_counter, sleep
from typing import Iterable

import numpy as np

from prc import (
    PRCConfig,
    PRCController,
    PRCFeatureBuilder,
    PressureProjector,
    fit_ridge_readout,
    signed_bend_angle_deg,
)
from calibration import RobotCalibration
from simulator import SoftArmSim


CONTROL_HZ = 100.0
DT = 1.0 / CONTROL_HZ
RESERVOIR_COLUMN = 0
ACTUATOR_COLUMNS = np.array([1, 2, 3], dtype=int)
DEFAULT_BEND_AXIS_XY = np.array([0.0, 1.0])
ROBOT_CALIBRATION = RobotCalibration.load()
CALIBRATED_ACTUATOR_DELAY_S = float(ROBOT_CALIBRATION.actuator["delay_s"])
CALIBRATED_PRESSURE_TAU_S = float(
    ROBOT_CALIBRATION.arm_config["tau_pneumatic"]
)
CALIBRATED_PRESSURE_CEILING_PSI = float(
    ROBOT_CALIBRATION.arm_config["p_max"]
)
RECORDED_COMMAND_CEILING_PSI = float(
    ROBOT_CALIBRATION.source["pressure_range_psi"][1]
)
RECORDED_EXCITATION_HZ = float(
    ROBOT_CALIBRATION.source["excitation_frequency_hz"]
)
DEFAULT_DELAY_STEPS = int(round(CALIBRATED_ACTUATOR_DELAY_S * CONTROL_HZ))
DEFAULT_SETTLE_SECONDS = 4.0
DEFAULT_FREQUENCY_SCALE = 1.0


def calibrated_plant_metadata() -> dict:
    """Return the immutable identity of the plant used for controller runs."""
    return {
        "model": "robot_calibration.json",
        "source_url": str(ROBOT_CALIBRATION.source["url"]),
        "source_commit": str(ROBOT_CALIBRATION.source["commit"]),
        "reservoir_topology": "parallel",
        "actuator_delay_s": CALIBRATED_ACTUATOR_DELAY_S,
        "pressure_time_constant_s": CALIBRATED_PRESSURE_TAU_S,
        "simulator_pressure_ceiling_psi": CALIBRATED_PRESSURE_CEILING_PSI,
        "recorded_command_ceiling_psi": RECORDED_COMMAND_CEILING_PSI,
        "recorded_excitation_frequency_hz": RECORDED_EXCITATION_HZ,
    }


def _as_three(value: float | Iterable[float], name: str) -> np.ndarray:
    """Return a finite three-vector, accepting a scalar for convenience."""
    out = np.asarray(value, dtype=float)
    if out.ndim == 0:
        out = np.full(3, float(out))
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite and scalar or shape (3,)")
    return out


@dataclass(frozen=True)
class SimulationData:
    """Synchronized samples from a three-actuator reservoir-mode rollout."""

    time_s: np.ndarray
    reservoir_psi: np.ndarray
    bend_deg: np.ndarray
    measured_pressure_psi: np.ndarray
    command_psi: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.time_s)
        expected = {
            "time_s": (n,),
            "reservoir_psi": (n, 5),
            "bend_deg": (n,),
            "measured_pressure_psi": (n, 3),
            "command_psi": (n, 3),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains a nonfinite value")


@dataclass(frozen=True)
class PredictorReport:
    validation_rmse_deg: float
    persistence_rmse_deg: float
    train_samples: int
    validation_samples: int


@dataclass(frozen=True)
class DelayAffinePredictor:
    """Fixed-basis forward predictor ``y[k+d] = g(state[k]) + B p[k]``.

    ``g`` uses linear, squared, and tanh features of a standardized causal
    measured-state vector.  The candidate command occurs only in a separate
    linear block, so inverse-label generation remains a convex quadratic
    problem in the three pressures.
    """

    delay_steps: int
    history_length: int
    state_mean: np.ndarray
    state_scale: np.ndarray
    command_mean: np.ndarray
    command_scale: np.ndarray
    coefficients: np.ndarray

    @property
    def state_dimension(self) -> int:
        return int(self.state_mean.size)

    @property
    def basis_dimension(self) -> int:
        return 1 + 3 * self.state_dimension

    def _basis(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state.shape[-1] != self.state_dimension:
            raise ValueError(
                f"predictor state must end in {self.state_dimension} values"
            )
        q = (state - self.state_mean) / self.state_scale
        ones = np.ones((*q.shape[:-1], 1), dtype=float)
        return np.concatenate((ones, q, q * q, np.tanh(q)), axis=-1)

    def state_offset_and_command_gain(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``g(state)`` and physical-units command gain ``B``."""
        state_part = self._basis(state) @ self.coefficients[: self.basis_dimension]
        command_coeff = self.coefficients[self.basis_dimension :]
        gain = command_coeff / self.command_scale
        offset = state_part - gain @ self.command_mean
        return np.asarray(offset), gain.copy()

    def predict(self, state: np.ndarray, command_psi: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        command = np.asarray(command_psi, dtype=float)
        if command.shape[-1] != 3:
            raise ValueError("candidate command must end in three pressures")
        offset, gain = self.state_offset_and_command_gain(state)
        return offset + command @ gain


@dataclass(frozen=True)
class ClosedLoopData:
    time_s: np.ndarray
    profile: np.ndarray
    reference_deg: np.ndarray
    preview_reference_deg: np.ndarray
    reservoir_psi: np.ndarray
    bend_deg: np.ndarray
    measured_pressure_psi: np.ndarray
    command_psi: np.ndarray
    raw_command_psi: np.ndarray
    projected: np.ndarray
    fallback: np.ndarray


@dataclass(frozen=True)
class EvaluationScenario:
    """Immutable held-out reference shared by controller rollouts."""

    control_hz: float
    reference_deg: np.ndarray
    preview_reference_deg: np.ndarray
    profile: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("scenario control_hz must be finite and positive")
        reference = np.asarray(self.reference_deg, dtype=float).copy()
        preview = np.asarray(self.preview_reference_deg, dtype=float).copy()
        profile = np.asarray(self.profile, dtype=str).copy()
        if reference.ndim != 1:
            raise ValueError("scenario reference must be one-dimensional")
        n = reference.size
        if preview.shape != (n,) or profile.shape != (n,):
            raise ValueError("scenario arrays must be one-dimensional and aligned")
        if n < 1 or not np.all(np.isfinite(reference)) or not np.all(
            np.isfinite(preview)
        ):
            raise ValueError("scenario reference arrays must be finite and nonempty")
        for values in (reference, preview, profile):
            values.setflags(write=False)
        object.__setattr__(self, "reference_deg", reference)
        object.__setattr__(self, "preview_reference_deg", preview)
        object.__setattr__(self, "profile", profile)

    @property
    def time_s(self) -> np.ndarray:
        return np.arange(len(self.reference_deg)) / self.control_hz


class _PRCVideoDashboard:
    """Reusable live tracking panel rendered beside MuJoCo video frames."""

    def __init__(self, time_s: np.ndarray, profile: np.ndarray,
                 reference_deg: np.ndarray, pressure_min_psi: np.ndarray,
                 pressure_max_psi: np.ndarray):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.time_s = np.asarray(time_s)
        self.profile = np.asarray(profile)
        self.reference_deg = np.asarray(reference_deg)
        self.figure = Figure(figsize=(4.8, 4.8), dpi=100, facecolor="white")
        self.canvas = FigureCanvasAgg(self.figure)
        self.angle_axis, self.pressure_axis = self.figure.subplots(
            2, 1, sharex=True
        )
        self.status = self.figure.text(
            0.5, 0.965, "", ha="center", va="top", fontsize=10
        )

        self.angle_axis.plot(
            self.time_s,
            self.reference_deg,
            "k--",
            lw=1.25,
            label="reference",
        )
        (self.angle_line,) = self.angle_axis.plot(
            [], [], color="tab:purple", lw=1.7, label="simulated bend"
        )
        (self.angle_marker,) = self.angle_axis.plot(
            [], [], "o", color="tab:purple", ms=4
        )
        self.angle_cursor = self.angle_axis.axvline(
            0.0, color="0.45", lw=0.9, ls=":"
        )
        angle_limit = max(
            0.5, 1.25 * float(np.max(np.abs(self.reference_deg)))
        )
        self.angle_axis.set_ylim(-angle_limit, angle_limit)
        self.angle_axis.set_ylabel("bend [deg]")
        self.angle_axis.legend(loc="upper left", fontsize=7, ncol=2)
        self.angle_axis.grid(alpha=0.25)

        self.pressure_lines = []
        colors = ("tab:green", "tab:red", "tab:blue")
        for index, color in enumerate(colors):
            (line,) = self.pressure_axis.plot(
                [], [], color=color, lw=1.4, label=f"Segment {index + 2}"
            )
            self.pressure_lines.append(line)
        self.pressure_cursor = self.pressure_axis.axvline(
            0.0, color="0.45", lw=0.9, ls=":"
        )
        low = max(0.0, float(np.min(pressure_min_psi)) - 0.25)
        high = float(np.max(pressure_max_psi)) + 0.25
        self.pressure_axis.set_ylim(low, high)
        self.pressure_axis.set_ylabel("command [psi]")
        self.pressure_axis.set_xlabel("time [s]")
        self.pressure_axis.legend(loc="upper left", fontsize=7, ncol=3)
        self.pressure_axis.grid(alpha=0.25)

        stop = max(0.01, float(self.time_s[-1]))
        self.pressure_axis.set_xlim(0.0, stop)
        changes = np.flatnonzero(self.profile[1:] != self.profile[:-1]) + 1
        for boundary in changes:
            for axis in (self.angle_axis, self.pressure_axis):
                axis.axvline(
                    self.time_s[boundary], color="0.75", lw=0.7, ls="--"
                )
        self.figure.subplots_adjust(
            left=0.14, right=0.97, bottom=0.11, top=0.89, hspace=0.18
        )

    def render(self, index: int, bend_deg: np.ndarray,
               command_psi: np.ndarray) -> np.ndarray:
        """Render the dashboard through ``index`` as an RGB array."""
        history = slice(0, index + 1)
        time = self.time_s[history]
        self.angle_line.set_data(time, bend_deg[history])
        self.angle_marker.set_data([self.time_s[index]], [bend_deg[index]])
        self.angle_cursor.set_xdata([self.time_s[index], self.time_s[index]])
        for channel, line in enumerate(self.pressure_lines):
            line.set_data(time, command_psi[history, channel])
        self.pressure_cursor.set_xdata(
            [self.time_s[index], self.time_s[index]]
        )

        observed_limit = 1.15 * float(np.max(np.abs(bend_deg[history])))
        current_limit = max(abs(value) for value in self.angle_axis.get_ylim())
        if observed_limit > current_limit:
            self.angle_axis.set_ylim(-observed_limit, observed_limit)
        self.status.set_text(
            f"{self.profile[index]}   t={self.time_s[index]:.2f} s   "
            f"reference={self.reference_deg[index]:+.2f} deg   "
            f"bend={bend_deg[index]:+.2f} deg"
        )

        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()

    def close(self) -> None:
        self.figure.clear()


class _PRCLiveWindow:
    """Interactive window for the composed MuJoCo/dashboard frames."""

    def __init__(self, frame: np.ndarray):
        import matplotlib
        import matplotlib.pyplot as plt

        backend = str(matplotlib.get_backend()).lower()
        try:
            # Matplotlib 3.9+ exposes backend discovery through the registry.
            from matplotlib.backends import BackendFilter, backend_registry

            interactive_backends = {
                str(name).lower()
                for name in backend_registry.list_builtin(
                    BackendFilter.INTERACTIVE
                )
            }
        except (AttributeError, ImportError):
            # Compatibility with older Matplotlib releases.
            from matplotlib import rcsetup

            interactive_backends = {
                str(name).lower() for name in rcsetup.interactive_bk
            }
        if backend not in interactive_backends:
            raise RuntimeError(
                f"Matplotlib backend {matplotlib.get_backend()!r} is not "
                "interactive. Run from a graphical desktop session or pass "
                "--no-live."
            )
        self.plt = plt
        self._interactive_was_on = self.plt.isinteractive()
        self._interaction_restored = False
        try:
            self.plt.ion()
            height, width = frame.shape[:2]
            self.figure = self.plt.figure(
                figsize=(width / 100.0, height / 100.0), dpi=100
            )
            self.axis = self.figure.add_axes((0.0, 0.0, 1.0, 1.0))
            self.axis.set_axis_off()
            self.image = self.axis.imshow(frame)
            manager = self.figure.canvas.manager
            if hasattr(manager, "set_window_title"):
                manager.set_window_title("MuJoCo PRC live controller")
            self.figure.show()
            self.update(frame)
        except BaseException:
            self._restore_interactive_mode()
            raise

    def _restore_interactive_mode(self) -> None:
        if self._interaction_restored:
            return
        if self._interactive_was_on:
            self.plt.ion()
        else:
            self.plt.ioff()
        self._interaction_restored = True

    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.figure.number)

    def update(self, frame: np.ndarray) -> bool:
        if not self.is_open():
            return False
        self.image.set_data(frame)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)
        return self.is_open()

    def hold_until_closed(self) -> None:
        if not self.is_open():
            return
        manager = self.figure.canvas.manager
        if hasattr(manager, "set_window_title"):
            manager.set_window_title(
                "MuJoCo PRC rollout complete — close this window to finish"
            )
        self.plt.show(block=True)

    def close(self) -> None:
        if self.is_open():
            self.plt.close(self.figure)
        self._restore_interactive_mode()


def measured_actuator_pressures(obs: dict) -> np.ndarray:
    """Average the five simulated sensor readings for active Segments 2--4."""
    columns = np.asarray(obs.get("actuator_columns", ACTUATOR_COLUMNS), dtype=int)
    if not np.array_equal(columns, ACTUATOR_COLUMNS):
        raise ValueError(
            "PRC simulation requires Segment 1 as reservoir and Segments 2--4 "
            "as actuators"
        )
    return np.asarray(obs["pouch_pressures"], dtype=float)[columns].mean(axis=1)


def settle_home(
    sim: SoftArmSim,
    reservoir_charge_psi: float,
    settle_seconds: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Reset and settle at balanced ``[x, x, x]`` active pressure."""
    baseline = np.full(3, reservoir_charge_psi, dtype=float)
    obs = sim.reset()
    for _ in range(max(1, int(round(settle_seconds * CONTROL_HZ)))):
        # A three-vector addresses only actuator_columns in reservoir mode.
        obs = sim.step(baseline)
    return obs, np.asarray(obs["tip_quat"]).copy(), baseline


def collect_safe_excitation(
    sim: SoftArmSim,
    home_quat: np.ndarray,
    config: PRCConfig,
    duration_s: float,
    seed: int,
    amplitude_psi: float,
    initial_command_psi: np.ndarray,
    bend_axis_xy: np.ndarray = DEFAULT_BEND_AXIS_XY,
) -> SimulationData:
    """Run reproducible projected ramps/multisines on only three actuators."""
    n = int(round(duration_s * config.control_hz))
    if n <= config.history_length + 2:
        raise ValueError("excitation duration is too short for PRC history")
    rng = np.random.default_rng(seed)
    projector = PressureProjector(config)
    command = _as_three(initial_command_psi, "initial command").copy()

    # Segment 3 (local actuator index 1) opposes sealed Segment 1 in the +Y
    # controlled plane.  The other two channels receive lower-amplitude,
    # independent excitation so all three command gains remain identifiable.
    channel_scale = np.array([0.42, 1.0, 0.42])
    # The measured calibration used 0.1 Hz excitation. Keep the default
    # identification bank inside that demonstrated frequency envelope.
    frequencies = rng.uniform(0.020, 0.090, size=(3, 3))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(3, 3))
    harmonic_weights = np.array([0.58, 0.29, 0.13])
    hold = np.zeros(3)
    hold_ticks = max(100, int(round(1.5 * config.control_hz)))

    time = np.empty(n)
    reservoir = np.empty((n, 5))
    angle = np.empty(n)
    measured = np.empty((n, 3))
    commands = np.empty((n, 3))
    obs = sim.observe()

    for k in range(n):
        t = k * config.dt
        if k % hold_ticks == 0:
            hold = rng.uniform(-0.45, 0.45, size=3)
        waves = np.sum(
            harmonic_weights[None, :]
            * np.sin(2.0 * np.pi * frequencies * t + phases),
            axis=1,
        )
        raw = initial_command_psi + amplitude_psi * channel_scale * (
            0.72 * waves + 0.28 * hold
        )
        projected = projector.project(raw, command, config.dt)
        command = projected.command_psi

        # Store state_k before command_k is applied.  This makes the forward
        # identification rows causal: (state_k, command_k) -> y[k + delay].
        time[k] = t
        reservoir[k] = obs["reservoir_pressures"]
        angle[k] = signed_bend_angle_deg(
            obs["tip_quat"], home_quat, bend_axis_xy
        )
        measured[k] = measured_actuator_pressures(obs)
        commands[k] = command
        obs = sim.step(command)

    return SimulationData(time, reservoir, angle, measured, commands)


def build_predictor_states(
    data: SimulationData, history_length: int, dt: float = DT
) -> np.ndarray:
    """Build causal state vectors with newest-to-oldest reservoir history."""
    if history_length < 1:
        raise ValueError("history_length must be positive")
    n = len(data.time_s)
    dangle = np.zeros(n)
    dangle[1:] = np.diff(data.bend_deg) / dt
    states = np.empty((n, history_length * 5 + 2 + 3))
    for k in range(n):
        history = [data.reservoir_psi[max(0, k - lag)]
                   for lag in range(history_length)]
        states[k] = np.concatenate(
            (np.concatenate(history), [data.bend_deg[k], dangle[k]],
             data.measured_pressure_psi[k])
        )
    return states


def fit_delay_aware_predictor(
    data: SimulationData,
    delay_steps: int,
    history_length: int = 6,
    ridge: float = 2e-2,
    validation_fraction: float = 0.2,
) -> tuple[DelayAffinePredictor, PredictorReport, np.ndarray]:
    """Fit the nonlinear-state/affine-command delayed forward model."""
    if delay_steps < 1:
        raise ValueError("delay_steps must be at least one")
    if not 0.05 <= validation_fraction <= 0.45:
        raise ValueError("validation_fraction must lie in [0.05, 0.45]")
    states = build_predictor_states(data, history_length)
    valid = np.arange(history_length - 1, len(data.time_s) - delay_steps)
    if valid.size < 40:
        raise ValueError("not enough delayed samples to fit and validate predictor")
    split = int(np.floor(valid.size * (1.0 - validation_fraction)))
    split = int(np.clip(split, 20, valid.size - 10))
    train_idx, val_idx = valid[:split], valid[split:]

    state_mean = states[train_idx].mean(axis=0)
    state_scale = states[train_idx].std(axis=0)
    state_scale = np.where(state_scale < 1e-8, 1.0, state_scale)
    command_mean = data.command_psi[train_idx].mean(axis=0)
    command_scale = data.command_psi[train_idx].std(axis=0)
    command_scale = np.where(command_scale < 1e-8, 1.0, command_scale)

    def design(indices: np.ndarray) -> np.ndarray:
        q = (states[indices] - state_mean) / state_scale
        basis = np.concatenate(
            (np.ones((len(indices), 1)), q, q * q, np.tanh(q)), axis=1
        )
        command = (data.command_psi[indices] - command_mean) / command_scale
        return np.concatenate((basis, command), axis=1)

    x_train = design(train_idx)
    y_train = data.bend_deg[train_idx + delay_steps]
    penalty = np.eye(x_train.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    gram = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_train
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs

    predictor = DelayAffinePredictor(
        delay_steps=delay_steps,
        history_length=history_length,
        state_mean=state_mean,
        state_scale=state_scale,
        command_mean=command_mean,
        command_scale=command_scale,
        coefficients=coefficients,
    )
    predicted = predictor.predict(states[val_idx], data.command_psi[val_idx])
    truth = data.bend_deg[val_idx + delay_steps]
    persistence = data.bend_deg[val_idx]
    report = PredictorReport(
        validation_rmse_deg=float(np.sqrt(np.mean((predicted - truth) ** 2))),
        persistence_rmse_deg=float(np.sqrt(np.mean((persistence - truth) ** 2))),
        train_samples=len(train_idx),
        validation_samples=len(val_idx),
    )
    return predictor, report, states


def _box_qp_inverse_label(
    reference_deg: float,
    state_offset_deg: float,
    command_gain_deg_per_psi: np.ndarray,
    previous_command_psi: np.ndarray,
    lower_psi: np.ndarray,
    upper_psi: np.ndarray,
    tracking_weight: float,
    move_weight: np.ndarray,
) -> np.ndarray:
    """Exactly solve the proposal's three-variable box-constrained label QP."""
    gain = _as_three(command_gain_deg_per_psi, "command gain")
    previous = _as_three(previous_command_psi, "previous command")
    lower = _as_three(lower_psi, "lower bounds")
    upper = _as_three(upper_psi, "upper bounds")
    r = _as_three(move_weight, "move weight")
    if tracking_weight <= 0 or np.any(r <= 0) or np.any(lower > upper):
        raise ValueError("inverse-label QP weights/bounds are invalid")

    # Objective, ignoring constants: p.T H p - 2 c.T p.
    target_delta = float(reference_deg - state_offset_deg)
    hessian = tracking_weight * np.outer(gain, gain) + np.diag(r)
    linear = tracking_weight * target_delta * gain + r * previous
    best = None
    best_cost = np.inf
    tolerance = 1e-8

    # Each coordinate is free (0), at its lower bound (-1), or upper (+1).
    # There are only 27 active sets, making this deterministic and dependency
    # free while still solving the coupled rank-one quadratic exactly.
    for status in product((-1, 0, 1), repeat=3):
        status_array = np.asarray(status)
        free = np.flatnonzero(status_array == 0)
        fixed = np.flatnonzero(status_array != 0)
        candidate = np.empty(3)
        if fixed.size:
            candidate[fixed] = np.where(
                status_array[fixed] < 0, lower[fixed], upper[fixed]
            )
        if free.size:
            rhs = linear[free]
            if fixed.size:
                rhs = rhs - hessian[np.ix_(free, fixed)] @ candidate[fixed]
            try:
                candidate[free] = np.linalg.solve(
                    hessian[np.ix_(free, free)], rhs
                )
            except np.linalg.LinAlgError:
                continue
        if np.any(candidate < lower - tolerance) or np.any(candidate > upper + tolerance):
            continue
        gradient = hessian @ candidate - linear
        low = status_array < 0
        high = status_array > 0
        if np.any(gradient[low] < -tolerance):
            continue
        if np.any(gradient[high] > tolerance):
            continue
        error = reference_deg - (state_offset_deg + gain @ candidate)
        cost = (tracking_weight * error * error
                + np.sum(r * (candidate - previous) ** 2))
        if cost < best_cost:
            best_cost = float(cost)
            best = candidate.copy()

    if best is None:  # Numerical guard; strict convexity should prevent this.
        best = np.clip(np.linalg.solve(hessian, linear), lower, upper)
    return np.clip(best, lower, upper)


def generate_inverse_control_labels(
    predictor: DelayAffinePredictor,
    states: np.ndarray,
    preview_reference_deg: np.ndarray,
    previous_command_psi: np.ndarray,
    config: PRCConfig,
    tracking_weight: float = 1.0,
    move_weight: float | Iterable[float] = 0.08,
    fixed_plane_bias_psi: float | None = None,
    recursive_previous: bool = False,
) -> np.ndarray:
    """Create offline pressure labels using hard bounds and per-tick slew.

    For the default scalar +Y task, ``fixed_plane_bias_psi`` makes the two
    orthogonal-plane channels slew back toward their balanced bias while the
    middle channel performs the inverse bend optimization.  All three labels
    remain valid setpoints; this simply removes a physically unobservable
    nullspace from the one-degree-of-freedom commissioning problem.
    With ``recursive_previous=True``, each target is slew-bounded from the
    preceding inverse target rather than from the unrelated exploration
    command.  This yields a self-consistent teacher command trajectory.
    """
    if config.max_total_pressure_psi is not None:
        raise ValueError(
            "offline labels do not implement max_total_pressure_psi; leave the "
            "coupled cap unset or add it to the label QP before training"
        )
    states = np.asarray(states, dtype=float)
    reference = np.asarray(preview_reference_deg, dtype=float)
    previous = np.asarray(previous_command_psi, dtype=float)
    if states.ndim != 2 or states.shape[0] != len(reference):
        raise ValueError("states and references must have matching rows")
    if previous.shape != (len(reference), 3):
        raise ValueError("previous commands must have shape (samples, 3)")
    move = _as_three(move_weight, "move weight")
    p_min = _as_three(config.pressure_min_psi, "pressure_min_psi")
    p_max = _as_three(config.pressure_max_psi, "pressure_max_psi")
    slew = _as_three(config.slew_rate_psi_s, "slew_rate_psi_s")
    delta = slew * config.dt
    gain = predictor.coefficients[predictor.basis_dimension :] / predictor.command_scale
    if fixed_plane_bias_psi is not None:
        fixed_plane_bias_psi = float(fixed_plane_bias_psi)
        if not np.isfinite(fixed_plane_bias_psi):
            raise ValueError("fixed-plane bias must be finite")

    labels = np.empty_like(previous)
    for i in range(len(reference)):
        previous_i = labels[i - 1] if recursive_previous and i > 0 else previous[i]
        offset, _ = predictor.state_offset_and_command_gain(states[i])
        lower = np.maximum(p_min, previous_i - delta)
        upper = np.minimum(p_max, previous_i + delta)
        if fixed_plane_bias_psi is not None:
            fixed = np.clip(fixed_plane_bias_psi, lower[[0, 2]], upper[[0, 2]])
            lower[[0, 2]] = fixed
            upper[[0, 2]] = fixed
        labels[i] = _box_qp_inverse_label(
            reference[i], float(offset), gain, previous_i, lower, upper,
            tracking_weight, move,
        )
    return labels


def reachable_reference_limit(bend_deg: np.ndarray) -> float:
    """Choose a conservative symmetric envelope observed during excitation."""
    low, high = np.percentile(np.asarray(bend_deg, dtype=float), [3.0, 97.0])
    if not low < 0.0 < high:
        return 0.0
    # Never enlarge the reference beyond what the safe excitation actually
    # demonstrated. The 0.82 factor retains margin for model/tracking error
    # while making the physical motion easier to see in the dashboard.
    return float(min(0.82 * min(-low, high), 14.0))


def make_training_references(
    n: int, dt: float, limit_deg: float, count: int, seed: int
) -> list[np.ndarray]:
    """Make preplanned references that remain inside the observed envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    references: list[np.ndarray] = []
    for j in range(count):
        frequencies = (
            np.array([0.025, 0.050, 0.075]) + 0.004 * (j % 6)
        )
        phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
        waveform = (
            0.56 * np.sin(2.0 * np.pi * frequencies[0] * t + phases[0])
            + 0.29 * np.sin(2.0 * np.pi * frequencies[1] * t + phases[1])
            + 0.15 * np.sin(2.0 * np.pi * frequencies[2] * t + phases[2])
        )
        peak = max(1.0, float(np.max(np.abs(waveform))))
        references.append(0.90 * limit_deg * waveform / peak)
    return references


def _preview(reference: np.ndarray, delay_steps: int) -> np.ndarray:
    index = np.minimum(np.arange(len(reference)) + delay_steps, len(reference) - 1)
    return np.asarray(reference, dtype=float)[index]


def build_causal_prc_training_set(
    data: SimulationData,
    states: np.ndarray,
    predictor: DelayAffinePredictor,
    references: list[np.ndarray],
    config: PRCConfig,
    move_weight: float,
    fixed_plane_bias_psi: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build proposal features and inverse labels without future measurements."""
    stop = len(data.time_s) - predictor.delay_steps
    if stop <= config.history_length:
        raise ValueError("training rollout is too short after delay alignment")
    # The inverse teacher rolls out its own previous target.  Only row zero is
    # consumed; tiling the known balanced start makes that separation from the
    # unrelated exploration commands explicit.
    previous = np.tile(
        _as_three(config.initial_command_psi, "initial command"), (stop, 1)
    )
    feature_batches = []
    label_batches = []
    first_labels = None

    for reference in references:
        if len(reference) != len(data.time_s):
            raise ValueError("each training reference must match excitation length")
        preview = _preview(reference, predictor.delay_steps)
        builder = PRCFeatureBuilder(config)
        builder.reset(data.reservoir_psi[0])
        features = np.empty((stop, config.feature_size))
        for k in range(stop):
            features[k] = builder.update(
                data.reservoir_psi[k],
                float(reference[k]),
                float(preview[k]),
                float(data.bend_deg[k]),
                data.measured_pressure_psi[k],
                config.dt,
            )
        labels = generate_inverse_control_labels(
            predictor,
            states[:stop],
            preview[:stop],
            previous,
            config,
            move_weight=move_weight,
            fixed_plane_bias_psi=fixed_plane_bias_psi,
            recursive_previous=True,
        )
        feature_batches.append(features)
        label_batches.append(labels)
        if first_labels is None:
            first_labels = labels

    return (
        np.concatenate(feature_batches, axis=0),
        np.concatenate(label_batches, axis=0),
        np.asarray(first_labels),
    )


def make_demo_reference(
    seconds_per_profile: float,
    delay_steps: int,
    limit_deg: float,
    control_hz: float = CONTROL_HZ,
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return held-out step, sine, and multisine references and preview."""
    if not np.isfinite(frequency_scale) or frequency_scale <= 0.0:
        raise ValueError("frequency_scale must be finite and positive")
    n = max(80, int(round(seconds_per_profile * control_hz)))
    t = np.arange(n) / control_hz

    step = np.zeros(n)
    q1, q2, q3 = n // 5, 2 * n // 5, 4 * n // 5
    step[q1:q2] = 0.70 * limit_deg
    step[q2:q3] = -0.55 * limit_deg

    # Frequencies differ from the controller-label reference bank.
    sine = 0.72 * limit_deg * np.sin(
        2.0 * np.pi * (0.060 * frequency_scale) * t
    )
    multisine = limit_deg * (
        0.48 * np.sin(2.0 * np.pi * (0.025 * frequency_scale) * t + 0.3)
        + 0.27 * np.sin(2.0 * np.pi * (0.055 * frequency_scale) * t + 1.4)
        + 0.13 * np.sin(2.0 * np.pi * (0.090 * frequency_scale) * t + 2.2)
    )
    reference = np.concatenate((step, sine, multisine))
    profile = np.repeat(np.array(["step", "sine", "multisine"]), n)
    return reference, _preview(reference, delay_steps), profile


def make_evaluation_scenario(
    seconds_per_profile: float,
    delay_steps: int,
    limit_deg: float,
    control_hz: float = CONTROL_HZ,
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
) -> EvaluationScenario:
    """Freeze one reference so multiple controllers see identical samples."""
    reference, preview, profile = make_demo_reference(
        seconds_per_profile,
        delay_steps,
        limit_deg,
        control_hz,
        frequency_scale,
    )
    return EvaluationScenario(control_hz, reference, preview, profile)


def run_closed_loop_demo(
    controller: PRCController,
    reservoir_charge_psi: float,
    seed: int,
    settle_seconds: float,
    seconds_per_profile: float,
    delay_steps: int,
    reference_limit_deg: float,
    bend_axis_xy: np.ndarray = DEFAULT_BEND_AXIS_XY,
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
    video_out: str | Path | None = None,
    video_fps: float = 30.0,
    live: bool = False,
    live_fps: float = 15.0,
    live_hold: bool = True,
    scenario: EvaluationScenario | None = None,
) -> ClosedLoopData:
    """Run the trained readout alone in a fresh seeded simulator rollout.

    When ``video_out`` is supplied, MuJoCo frames are streamed to an MP4 while
    the same controller rollout used for the reported metrics is running.
    """
    if (not np.isfinite(video_fps)
            or not 0.0 < video_fps <= controller.config.control_hz):
        raise ValueError("video_fps must be finite and in (0, control_hz]")
    if (not np.isfinite(live_fps)
            or not 0.0 < live_fps <= controller.config.control_hz):
        raise ValueError("live_fps must be finite and in (0, control_hz]")
    video_path = None if video_out is None else Path(video_out)
    if video_path is not None and video_path.suffix.lower() != ".mp4":
        raise ValueError("video_out must use an .mp4 extension")

    sim = ROBOT_CALIBRATION.make_sim(
        topology="parallel",
        control_hz=controller.config.control_hz,
        seed=seed,
        reservoir_pressure_psi=reservoir_charge_psi,
    )
    obs, home_quat, baseline = settle_home(
        sim, reservoir_charge_psi, settle_seconds
    )
    controller.reset(baseline, obs["reservoir_pressures"])
    if scenario is None:
        scenario = make_evaluation_scenario(
            seconds_per_profile,
            delay_steps,
            reference_limit_deg,
            controller.config.control_hz,
            frequency_scale,
        )
    elif not np.isclose(scenario.control_hz, controller.config.control_hz):
        raise ValueError("scenario and controller control rates must match")
    reference = scenario.reference_deg
    preview = scenario.preview_reference_deg
    profile = scenario.profile
    n = len(reference)
    reservoir = np.empty((n, 5))
    angle = np.empty(n)
    measured = np.empty((n, 3))
    command = np.empty((n, 3))
    raw = np.empty((n, 3))
    projected = np.empty(n, dtype=bool)
    fallback = np.empty(n, dtype=bool)

    writer = None
    dashboard = None
    live_window = None
    live_active = bool(live)
    temporary_video_path = None
    next_video_tick = 0.0
    next_live_tick = 0.0
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        live_started = None
        for k in range(n):
            y = signed_bend_angle_deg(obs["tip_quat"], home_quat, bend_axis_xy)
            p_measured = measured_actuator_pressures(obs)
            step = controller.compute(
                float(reference[k]),
                y,
                obs["reservoir_pressures"],
                p_measured,
                preview_reference_deg=float(preview[k]),
                dt=controller.config.dt,
            )
            reservoir[k] = obs["reservoir_pressures"]
            angle[k] = y
            measured[k] = p_measured
            command[k] = step.command_psi
            raw[k] = step.raw_command_psi
            projected[k] = step.projected
            fallback[k] = step.fallback

            capture_video = (
                video_path is not None and k + 1e-12 >= next_video_tick
            )
            capture_live = (
                live_active and k + 1e-12 >= next_live_tick
            )
            if capture_video or capture_live:
                try:
                    frame = sim.render_frame()
                except Exception as exc:
                    raise RuntimeError(
                        "MuJoCo dashboard rendering is unavailable in this "
                        "process. On macOS, run from a logged-in graphical "
                        "session (using mjpython if required). On Linux, "
                        "select a working EGL or OSMesa backend before Python "
                        "starts. For a numeric-only run, pass --no-live and "
                        "do not request --video-out."
                    ) from exc
                if dashboard is None:
                    dashboard = _PRCVideoDashboard(
                        scenario.time_s,
                        profile,
                        reference,
                        controller.projector.p_min,
                        controller.projector.p_max,
                    )
                panel = dashboard.render(k, angle, command)
                frame = np.concatenate((frame, panel), axis=1)
                if capture_live:
                    if live_window is None:
                        live_window = _PRCLiveWindow(frame)
                        # Do not count one-time renderer, font, and window
                        # initialization against the real-time schedule.
                        live_started = (
                            perf_counter() - k * controller.config.dt
                        )
                    else:
                        live_active = live_window.update(frame)
                    next_live_tick += controller.config.control_hz / live_fps
                if capture_video and writer is None:
                    import imageio.v2 as imageio

                    temporary = tempfile.NamedTemporaryFile(
                        prefix=f".{video_path.stem}-",
                        suffix=".mp4",
                        dir=video_path.parent,
                        delete=False,
                    )
                    temporary_video_path = Path(temporary.name)
                    temporary.close()
                    writer = imageio.get_writer(
                        temporary_video_path,
                        fps=video_fps,
                        codec="libx264",
                        quality=8,
                        pixelformat="yuv420p",
                        macro_block_size=16,
                    )
                if capture_video:
                    writer.append_data(frame)
                    next_video_tick += (
                        controller.config.control_hz / video_fps
                    )

            obs = sim.step(step.command_psi)
            if live_active and live_started is not None:
                remaining = (
                    live_started + (k + 1) * controller.config.dt
                    - perf_counter()
                )
                if remaining > 0.0:
                    sleep(remaining)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary_video_path, video_path)
            temporary_video_path = None
        if live_window is not None and live_hold:
            live_window.hold_until_closed()
    except BaseException:
        if writer is not None:
            writer.close()
        if temporary_video_path is not None:
            temporary_video_path.unlink(missing_ok=True)
        raise
    finally:
        if live_window is not None:
            live_window.close()
        if dashboard is not None:
            dashboard.close()
        sim.close()

    return ClosedLoopData(
        time_s=scenario.time_s,
        profile=profile,
        reference_deg=reference,
        preview_reference_deg=preview,
        reservoir_psi=reservoir,
        bend_deg=angle,
        measured_pressure_psi=measured,
        command_psi=command,
        raw_command_psi=raw,
        projected=projected,
        fallback=fallback,
    )


def tracking_metrics(demo: ClosedLoopData) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name in ("step", "sine", "multisine", "all"):
        mask = np.ones(len(demo.time_s), dtype=bool) if name == "all" else demo.profile == name
        error = demo.reference_deg[mask] - demo.bend_deg[mask]
        command = demo.command_psi[mask]
        total_variation = float(np.abs(np.diff(command, axis=0)).sum())
        metrics[name] = {
            "rmse_deg": float(np.sqrt(np.mean(error * error))),
            "zero_bend_baseline_rmse_deg": float(
                np.sqrt(np.mean(demo.reference_deg[mask] ** 2))
            ),
            "max_error_deg": float(np.max(np.abs(error))),
            "pressure_tv_psi": total_variation,
            "projected_fraction": float(np.mean(demo.projected[mask])),
            "fallback_count": float(np.count_nonzero(demo.fallback[mask])),
        }
    return metrics


def evaluate_student_acceptance(
    demo: ClosedLoopData,
    metrics: dict[str, dict[str, float]],
    predictor_report: PredictorReport,
    config: PRCConfig,
    initial_command_psi: np.ndarray,
) -> tuple[bool, tuple[str, ...]]:
    """Apply frozen simulation gates before publishing a controller artifact.

    The inverse-label persistence metric is diagnostic, not a deployment gate:
    it uses the previous teacher label, which the proposal intentionally omits
    from the learned readout. Acceptance is based on the actual student-only
    rollout plus the independently held-out forward-model check.
    """
    failures: list[str] = []
    if (not np.isfinite(predictor_report.validation_rmse_deg)
            or predictor_report.validation_rmse_deg
            >= predictor_report.persistence_rmse_deg):
        failures.append("delayed predictor did not beat angle persistence")

    for name in ("step", "sine", "multisine"):
        metric = metrics[name]
        if (not np.isfinite(metric["rmse_deg"])
                or metric["rmse_deg"]
                >= metric["zero_bend_baseline_rmse_deg"]):
            failures.append(f"{name} tracking did not beat the zero-bend baseline")

    if np.any(demo.fallback):
        failures.append("student rollout invoked a controller fallback")
    if not all(np.all(np.isfinite(values)) for values in (
        demo.bend_deg,
        demo.reservoir_psi,
        demo.measured_pressure_psi,
        demo.command_psi,
        demo.raw_command_psi,
    )):
        failures.append("student rollout contains nonfinite data")

    p_min = _as_three(config.pressure_min_psi, "pressure_min_psi")
    p_max = _as_three(config.pressure_max_psi, "pressure_max_psi")
    tolerance = 1e-9
    if (np.any(demo.command_psi < p_min - tolerance)
            or np.any(demo.command_psi > p_max + tolerance)):
        failures.append("student commands violated pressure bounds")
    if config.max_total_pressure_psi is not None and np.any(
        demo.command_psi.sum(axis=1) > config.max_total_pressure_psi + tolerance
    ):
        failures.append("student commands violated the coupled pressure cap")

    trajectory = np.vstack((
        _as_three(initial_command_psi, "initial command"),
        demo.command_psi,
    ))
    allowed_delta = _as_three(config.slew_rate_psi_s, "slew rate") * config.dt
    if np.any(np.abs(np.diff(trajectory, axis=0)) > allowed_delta + tolerance):
        failures.append("student commands violated slew limits")
    if float(np.ptp(demo.command_psi[:, 1])) < 0.1:
        failures.append("controlled Segment 3 command did not vary by 0.1 psi")

    return not failures, tuple(failures)


def save_combined_log(
    path: str | Path,
    excitation: SimulationData,
    training_reference: np.ndarray,
    training_preview: np.ndarray,
    inverse_labels: np.ndarray,
    delay_steps: int,
    demo: ClosedLoopData,
) -> Path:
    """Write synchronized excitation/training and closed-loop signals to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "phase", "profile", "time_s", "reference_deg",
        "preview_reference_deg", "bend_deg",
        *[f"reservoir_s1_pouch{i}_psi" for i in range(1, 6)],
        *[f"measured_pressure_s{i}_psi" for i in range(2, 5)],
        *[f"command_s{i}_psi" for i in range(2, 5)],
        *[f"inverse_label_s{i}_psi" for i in range(2, 5)],
        *[f"raw_command_s{i}_psi" for i in range(2, 5)],
        "projected", "fallback",
    ]

    def values(items: np.ndarray) -> list[str]:
        return [f"{float(value):.8g}" for value in items]

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        label_stop = len(inverse_labels)
        for k in range(len(excitation.time_s)):
            label = inverse_labels[k] if k < label_stop else np.full(3, np.nan)
            writer.writerow([
                "training", "safe_excitation", f"{excitation.time_s[k]:.6f}",
                f"{training_reference[k]:.8g}", f"{training_preview[k]:.8g}",
                f"{excitation.bend_deg[k]:.8g}",
                *values(excitation.reservoir_psi[k]),
                *values(excitation.measured_pressure_psi[k]),
                *values(excitation.command_psi[k]),
                *values(label),
                "", "", "", "", "",
            ])
        for k in range(len(demo.time_s)):
            writer.writerow([
                "closed_loop", str(demo.profile[k]), f"{demo.time_s[k]:.6f}",
                f"{demo.reference_deg[k]:.8g}",
                f"{demo.preview_reference_deg[k]:.8g}",
                f"{demo.bend_deg[k]:.8g}",
                *values(demo.reservoir_psi[k]),
                *values(demo.measured_pressure_psi[k]),
                *values(demo.command_psi[k]),
                "", "", "",
                *values(demo.raw_command_psi[k]),
                int(demo.projected[k]), int(demo.fallback[k]),
            ])
    return path


def save_demo_plot(
    path: str | Path,
    demo: ClosedLoopData,
    predictor_report: PredictorReport,
    metrics: dict[str, dict[str, float]],
) -> Path:
    """Save a compact simulation-only training and tracking summary."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(11.0, 8.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    ax_angle, ax_pressure, ax_reservoir = axes
    ax_angle.plot(demo.time_s, demo.reference_deg, "k--", lw=1.2, label="reference")
    ax_angle.plot(demo.time_s, demo.bend_deg, color="tab:purple", lw=1.35,
                  label="simulated bend")
    ax_angle.set_ylabel("bend [deg]")
    ax_angle.legend(loc="upper right", ncol=2)
    ax_angle.grid(alpha=0.25)

    segment_colors = ("tab:green", "tab:red", "tab:blue")
    for i, color in enumerate(segment_colors):
        ax_pressure.plot(demo.time_s, demo.command_psi[:, i], color=color,
                         lw=1.0, label=f"Segment {i + 2}")
    ax_pressure.set_ylabel("command [psi]")
    ax_pressure.legend(loc="upper right", ncol=3)
    ax_pressure.grid(alpha=0.25)

    for i in range(5):
        ax_reservoir.plot(demo.time_s, demo.reservoir_psi[:, i], lw=0.9,
                          label=f"pouch {i + 1}")
    ax_reservoir.set_xlabel("time [s]")
    ax_reservoir.set_ylabel("sealed S1 [psi]")
    ax_reservoir.legend(loc="upper right", ncol=5, fontsize=8)
    ax_reservoir.grid(alpha=0.25)

    changes = np.flatnonzero(demo.profile[1:] != demo.profile[:-1]) + 1
    for boundary in changes:
        for axis in axes:
            axis.axvline(demo.time_s[boundary], color="0.55", lw=0.8, ls=":")
    for name in ("step", "sine", "multisine"):
        indices = np.flatnonzero(demo.profile == name)
        center = demo.time_s[indices[len(indices) // 2]]
        ax_angle.text(center, 1.02, name, transform=ax_angle.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "MuJoCo PRC demonstration (simulation only)\n"
        f"overall RMSE {metrics['all']['rmse_deg']:.2f} deg "
        f"(zero-bend baseline {metrics['all']['zero_bend_baseline_rmse_deg']:.2f}); "
        f"forward-model validation RMSE {predictor_report.validation_rmse_deg:.2f} deg"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    fig.clear()
    return path


def run_pipeline(
    args: argparse.Namespace,
    *,
    persist_artifacts: bool = True,
) -> dict:
    """Execute training and a held-out simulation.

    ``persist_artifacts=False`` lets an embedded evaluation, such as the
    paired PRC/PID comparison, avoid replacing the standalone PRC model, log,
    or plot.
    """
    if not 0.0 <= args.reservoir_charge <= args.pressure_max:
        raise ValueError("reservoir charge x must be within [0, pressure-max]")
    if args.pressure_max > CALIBRATED_PRESSURE_CEILING_PSI:
        raise ValueError(
            "pressure-max cannot exceed the calibrated simulator ceiling of "
            f"{CALIBRATED_PRESSURE_CEILING_PSI:g} psi"
        )
    if args.amplitude <= 0.0:
        raise ValueError("excitation amplitude must be positive")
    if args.train_seconds <= 0.0 or args.profile_seconds <= 0.0:
        raise ValueError("training and profile durations must be positive")
    if args.settle_seconds < 0.0:
        raise ValueError("settle duration cannot be negative")
    if args.delay_steps < 1:
        raise ValueError("delay-steps must be at least one")
    if args.reference_augmentations < 1:
        raise ValueError("reference-augmentations must be at least one")
    if args.predictor_ridge < 0.0 or args.readout_ridge < 0.0:
        raise ValueError("ridge penalties cannot be negative")
    if args.inverse_move_weight <= 0.0:
        raise ValueError("inverse-move-weight must be positive")
    if not np.isfinite(args.frequency_scale) or args.frequency_scale <= 0.0:
        raise ValueError("frequency-scale must be finite and positive")
    if not np.isfinite(args.video_fps) or not 0.0 < args.video_fps <= CONTROL_HZ:
        raise ValueError("video-fps must be finite and in (0, 100]")
    if not np.isfinite(args.live_fps) or not 0.0 < args.live_fps <= CONTROL_HZ:
        raise ValueError("live-fps must be finite and in (0, 100]")

    baseline = np.full(3, args.reservoir_charge)
    config = PRCConfig(
        control_hz=CONTROL_HZ,
        history_length=args.history_length,
        pressure_min_psi=0.0,
        pressure_max_psi=args.pressure_max,
        slew_rate_psi_s=args.slew_rate,
        initial_command_psi=tuple(baseline),
        pressure_trip_psi=CALIBRATED_PRESSURE_CEILING_PSI,
        max_tick_s=0.008,
        fallback_mode="hold",
    )

    sim = ROBOT_CALIBRATION.make_sim(
        topology="parallel",
        control_hz=config.control_hz,
        seed=args.seed,
        reservoir_pressure_psi=args.reservoir_charge,
    )
    obs, home_quat, baseline = settle_home(
        sim, args.reservoir_charge, args.settle_seconds
    )
    if not np.array_equal(obs["actuator_columns"], ACTUATOR_COLUMNS):
        raise RuntimeError("unexpected simulator actuator mapping")
    excitation = collect_safe_excitation(
        sim,
        home_quat,
        config,
        args.train_seconds,
        args.seed,
        args.amplitude,
        baseline,
    )

    predictor, predictor_report, states = fit_delay_aware_predictor(
        excitation,
        args.delay_steps,
        history_length=args.predictor_history,
        ridge=args.predictor_ridge,
    )
    reference_limit = reachable_reference_limit(excitation.bend_deg)
    if reference_limit < 0.25:
        raise RuntimeError(
            "safe excitation did not establish at least +/-0.25 deg of "
            "bidirectional authority; choose x inside the actuator range, "
            "increase only the validated excitation amplitude, or implement "
            "an asymmetric reference generator"
        )
    references = make_training_references(
        len(excitation.time_s), config.dt, reference_limit,
        args.reference_augmentations, args.seed + 19,
    )
    features, labels, first_labels = build_causal_prc_training_set(
        excitation,
        states,
        predictor,
        references,
        config,
        args.inverse_move_weight,
        fixed_plane_bias_psi=args.reservoir_charge,
    )
    weights, normalizer = fit_ridge_readout(features, labels, args.readout_ridge)
    # The scalar +Y commissioning task has no information with which to choose
    # motion in the orthogonal +X/-X actuator pair.  Encode the planar
    # constraint exactly in Wc: Segments 2 and 4 remain at the balanced bias,
    # while the ridge-fitted middle row controls the requested bend.
    weights[[0, 2], :] = 0.0
    weights[[0, 2], 0] = args.reservoir_charge
    controller = PRCController(
        weights,
        config,
        normalizer,
        metadata={
            "controller": "physical_reservoir_computing",
            "simulation_only": True,
            "reservoir_column": int(RESERVOIR_COLUMN),
            "actuator_columns": ACTUATOR_COLUMNS.tolist(),
            "reservoir_charge_psi": float(args.reservoir_charge),
            "bend_axis_xy": DEFAULT_BEND_AXIS_XY.tolist(),
            "home_tip_quaternion_wxyz": home_quat.tolist(),
            "reference_preview_steps": int(args.delay_steps),
            "reference_preview_seconds": float(
                args.delay_steps / config.control_hz
            ),
            "demo_frequency_scale": float(args.frequency_scale),
            "maximum_demo_frequency_hz": float(
                0.090 * args.frequency_scale
            ),
            "plant_calibration": calibrated_plant_metadata(),
            "training_seed": int(args.seed),
            "command_units": "psi_gauge",
            "angle_units": "degrees",
            "reservoir_signal_order": [
                f"segment_1_pouch_{i}" for i in range(1, 6)
            ],
            "actuator_output_order": ["segment_2", "segment_3", "segment_4"],
        },
    )

    scaled_features = normalizer.transform(features)
    fitted_labels = scaled_features @ weights.T
    label_rmse = float(np.sqrt(np.mean((fitted_labels - labels) ** 2)))
    controlled_label_rmse = float(
        np.sqrt(np.mean((fitted_labels[:, 1] - labels[:, 1]) ** 2))
    )
    persistence_batches = []
    batch_size = len(first_labels)
    for start in range(0, len(labels), batch_size):
        batch = labels[start:start + batch_size]
        persistence_batches.append(np.vstack((baseline, batch[:-1])))
    persistence = np.concatenate(persistence_batches, axis=0)
    persistence_rmse = float(np.sqrt(np.mean((persistence - labels) ** 2)))
    controlled_persistence_rmse = float(
        np.sqrt(np.mean((persistence[:, 1] - labels[:, 1]) ** 2))
    )

    scenario = make_evaluation_scenario(
        args.profile_seconds,
        args.delay_steps,
        reference_limit,
        config.control_hz,
        args.frequency_scale,
    )
    demo = run_closed_loop_demo(
        controller,
        args.reservoir_charge,
        args.seed + 1000,
        args.settle_seconds,
        args.profile_seconds,
        args.delay_steps,
        reference_limit,
        frequency_scale=args.frequency_scale,
        video_out=args.video_out,
        video_fps=args.video_fps,
        live=args.live,
        live_fps=args.live_fps,
        live_hold=args.live_hold,
        scenario=scenario,
    )
    metrics = tracking_metrics(demo)
    accepted, acceptance_failures = evaluate_student_acceptance(
        demo, metrics, predictor_report, config, baseline
    )
    controller.metadata["simulation_acceptance"] = {
        "passed": bool(accepted),
        "predictor_validation_rmse_deg": float(
            predictor_report.validation_rmse_deg
        ),
        "predictor_persistence_rmse_deg": float(
            predictor_report.persistence_rmse_deg
        ),
        "profile_rmse_deg": {
            name: float(metrics[name]["rmse_deg"])
            for name in ("step", "sine", "multisine", "all")
        },
        "profile_zero_bend_baseline_rmse_deg": {
            name: float(metrics[name]["zero_bend_baseline_rmse_deg"])
            for name in ("step", "sine", "multisine", "all")
        },
        "fallback_count": int(np.count_nonzero(demo.fallback)),
    }
    model_existed_before = bool(
        persist_artifacts and Path(args.model_out).is_file()
    )
    model_path = None
    log_path = None
    plot_path = None
    if persist_artifacts:
        model_path = controller.save(args.model_out) if accepted else None
        first_preview = _preview(references[0], args.delay_steps)
        log_path = save_combined_log(
            args.log_out,
            excitation,
            references[0],
            first_preview,
            first_labels,
            args.delay_steps,
            demo,
        )
        plot_path = save_demo_plot(
            args.plot_out, demo, predictor_report, metrics
        )

    return {
        "model_path": model_path,
        "log_path": log_path,
        "plot_path": plot_path,
        "video_path": None if args.video_out is None else Path(args.video_out),
        "predictor_report": predictor_report,
        "reference_limit_deg": reference_limit,
        "label_rmse_psi": label_rmse,
        "persistence_label_rmse_psi": persistence_rmse,
        "controlled_label_rmse_psi": controlled_label_rmse,
        "controlled_persistence_label_rmse_psi": controlled_persistence_rmse,
        "accepted": accepted,
        "acceptance_failures": acceptance_failures,
        "existing_model_preserved": bool(not accepted and model_existed_before),
        "metrics": metrics,
        "excitation_samples": len(excitation.time_s),
        "controller_training_samples": len(features),
        "controller": controller,
        "config": config,
        "demo": demo,
        "scenario": scenario,
    }


def build_argument_parser(*, live_default: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and demonstrate a simulation-only PRC controller with "
            "Segment 1 sealed and Segments 2--4 actuated."
        )
    )
    parser.add_argument(
        "--reservoir-charge", "--x", dest="reservoir_charge", type=float,
        default=2.0, help="one-time sealed Segment-1 charge x [psi] (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-out", type=Path,
        default=Path("output/calibrated_prc_model.npz"))
    parser.add_argument("--log-out", type=Path,
        default=Path("output/calibrated_prc_simulation_log.csv"))
    parser.add_argument("--plot-out", type=Path,
        default=Path("output/calibrated_prc_simulation.png"))
    parser.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="optional MP4 path for the held-out controller rollout",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="encoded video frame rate in (0, 100] (default: 30)",
    )
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=live_default,
        help=(
            "show the model and tracking dashboard live "
            f"(default: {'enabled' if live_default else 'disabled'})"
        ),
    )
    parser.add_argument(
        "--live-fps",
        type=float,
        default=15.0,
        help="live dashboard refresh rate in (0, 100] (default: 15)",
    )
    parser.add_argument(
        "--live-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the completed dashboard open until it is closed",
    )
    parser.add_argument("--train-seconds", type=float, default=60.0,
                        help="excitation rollout duration [s] (default: 60)")
    parser.add_argument(
        "--profile-seconds",
        type=float,
        default=20.0,
        help="duration of each held-out profile [s] (default: 20)",
    )
    parser.add_argument(
        "--frequency-scale",
        type=float,
        default=DEFAULT_FREQUENCY_SCALE,
        help=(
            "demo sine/multisine frequency multiplier; the default keeps all "
            "components at or below the measured 0.1 Hz envelope (default: 1)"
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=DEFAULT_SETTLE_SECONDS,
        help="home-settling duration [s] for the calibrated lag (default: 4)",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=DEFAULT_DELAY_STEPS,
        help="calibrated command delay at 100 Hz (default: 50 / 0.5 s)",
    )
    parser.add_argument("--history-length", type=int, default=50,
                        help="PRC reservoir history window (default: 50)")
    parser.add_argument("--predictor-history", type=int, default=50,
                        help="delay-aware predictor history (default: 50)")
    parser.add_argument("--reference-augmentations", type=int, default=6,
                        help="number of training reference waveforms (default: 6)")
    parser.add_argument(
        "--pressure-max",
        type=float,
        default=RECORDED_COMMAND_CEILING_PSI,
        help=(
            "command ceiling [psi]; default 10 follows the recorded command "
            "range (calibrated simulator hard ceiling: 11)"
        ),
    )
    parser.add_argument("--slew-rate", type=float, default=6.0,
                        help="per-channel pressure slew limit [psi/s] (default: 6)")
    parser.add_argument("--amplitude", type=float, default=2.8,
                        help="safe excitation scale [psi] (default: 2.8)")
    parser.add_argument("--predictor-ridge", type=float, default=1e-2,
                        help="ridge penalty for forward predictor (default: 0.01)")
    parser.add_argument("--readout-ridge", type=float, default=0.10,
                        help="ridge penalty for PRC readout (default: 0.10)")
    parser.add_argument("--inverse-move-weight", type=float, default=0.04,
                        help="inverse-label slew penalty (default: 0.04)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_pipeline(args)
    report: PredictorReport = result["predictor_report"]
    print("PRC simulation pipeline complete (simulation evidence only).")
    print(
        "  delayed predictor validation RMSE: "
        f"{report.validation_rmse_deg:.3f} deg "
        f"(angle-persistence baseline {report.persistence_rmse_deg:.3f} deg)"
    )
    print(
        "  inverse-label readout RMSE (all channels): "
        f"{result['label_rmse_psi']:.3f} psi "
        f"(command-persistence baseline {result['persistence_label_rmse_psi']:.3f} psi)"
    )
    print(
        "  inverse-label readout RMSE (controlled Segment 3): "
        f"{result['controlled_label_rmse_psi']:.3f} psi "
        "(command-persistence baseline "
        f"{result['controlled_persistence_label_rmse_psi']:.3f} psi)"
    )
    for name in ("step", "sine", "multisine", "all"):
        metric = result["metrics"][name]
        print(
            f"  {name:10s} RMSE {metric['rmse_deg']:.3f} deg, "
            f"zero-bend baseline {metric['zero_bend_baseline_rmse_deg']:.3f} deg, "
            f"max |error| {metric['max_error_deg']:.3f} deg, "
            f"fallbacks {int(metric['fallback_count'])}"
        )
    if result["accepted"]:
        print("  simulation acceptance gates: PASSED")
        print(f"  model: {result['model_path']}")
    else:
        print("  simulation acceptance gates: REJECTED")
        for failure in result["acceptance_failures"]:
            print(f"    - {failure}")
        if result["existing_model_preserved"]:
            print(f"  model: rejected candidate not published; existing {args.model_out} left unchanged")
        else:
            print("  model: rejected candidate not published")
    print(f"  log:   {result['log_path']}")
    print(f"  plot:  {result['plot_path']}")
    if result["video_path"] is not None:
        print(f"  video: {result['video_path']}")
    print("No physical-arm performance or stability claim is made by this run.")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
