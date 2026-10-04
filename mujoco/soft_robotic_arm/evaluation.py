"""Deterministic circular-reference evaluation for student controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Controller(Protocol):
    """Controller interface used by :func:`evaluate_controller`."""

    def compute(
        self,
        t: float,
        obs: Mapping[str, Any],
        ref: NDArray[np.float64],
    ) -> ArrayLike:
        """Return S2--S4 setpoints; ``t`` equals ``obs['time']`` in seconds."""


@dataclass(frozen=True)
class CircularTrackingTask:
    """Shared trajectory and scoring settings for controller comparison.

    ``baseline_psi`` may be one scalar or three values for Segments 2--4. The
    plant settles at that command before its trajectory centre is measured.
    """

    control_hz: float = 100.0
    settle_s: float = 5.0
    track_s: float = 30.0
    radius_m: float = 0.007
    frequency_hz: float = 0.1
    baseline_psi: float | tuple[float, float, float] = (4.5, 4.5, 4.5)
    command_limit_psi: float = 9.0

    def __post_init__(self) -> None:
        positive = {
            "control_hz": self.control_hz,
            "track_s": self.track_s,
            "frequency_hz": self.frequency_hz,
            "command_limit_psi": self.command_limit_psi,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.settle_s) or self.settle_s < 0.0:
            raise ValueError("settle_s must be nonnegative and finite")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        baseline = np.asarray(self.baseline_psi, dtype=float)
        if baseline.ndim > 1 or baseline.size not in (1, 3):
            raise ValueError("baseline_psi must be scalar or shape (3,)")
        if not np.all(np.isfinite(baseline)) or np.any(baseline < 0.0):
            raise ValueError("baseline_psi must be finite and nonnegative")

    def baseline_command(self, max_pressure: float) -> NDArray[np.float64]:
        """Return the validated three-pressure baseline command."""
        baseline = np.asarray(self.baseline_psi, dtype=float)
        if baseline.ndim == 0:
            baseline = np.full(3, float(baseline))
        else:
            baseline = baseline.reshape(3).copy()
        if np.any(baseline > max_pressure):
            raise ValueError(
                f"baseline_psi must not exceed the model limit {max_pressure:g} psi"
            )
        return baseline

    @property
    def settle_steps(self) -> int:
        return int(round(self.settle_s * self.control_hz))

    @property
    def track_steps(self) -> int:
        return int(round(self.track_s * self.control_hz))


def make_reference(
    home: ArrayLike,
    t: float,
    task: CircularTrackingTask | None = None,
) -> NDArray[np.float64]:
    """Return the task's circular x-y reference with constant z."""
    task = task or CircularTrackingTask()
    centre = np.asarray(home, dtype=float)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("home must be a finite vector with shape (3,)")
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    # Start on the -x side to preserve the physical experiment convention.
    angle = 2.0 * np.pi * task.frequency_hz * float(t) + np.pi
    ref = centre.copy()
    ref[0] += task.radius_m * np.cos(angle)
    ref[1] += task.radius_m * np.sin(angle)
    return ref


@dataclass(frozen=True)
class TrackingResult:
    """Metrics and sampled signals from one tracking evaluation."""

    rmse_mm: float
    max_error_mm: float
    phase_lag_s: float
    effort_psi: float
    home: NDArray[np.float64]
    times: NDArray[np.float64]
    tips: NDArray[np.float64]
    references: NDArray[np.float64]
    commands: NDArray[np.float64]
    task: CircularTrackingTask

    def as_dict(self) -> dict[str, Any]:
        """Return metrics plus copies of sampled arrays."""
        return {
            "rmse_mm": self.rmse_mm,
            "max_error_mm": self.max_error_mm,
            "phase_lag_s": self.phase_lag_s,
            "effort_psi": self.effort_psi,
            "home": self.home.copy(),
            "times": self.times.copy(),
            "tips": self.tips.copy(),
            "references": self.references.copy(),
            "commands": self.commands.copy(),
        }

    def plot(self, label: str = "Controller"):
        """Plot path, error, and commands; imports Matplotlib only on demand."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "plotting requires `pip install soft-robotic-arm[coursework]`"
            ) from exc

        lateral_error = np.linalg.norm(
            self.tips[:, :2] - self.references[:, :2], axis=1
        ) * 1000.0
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            f"{label} | RMSE {self.rmse_mm:.1f} mm · "
            f"lag {self.phase_lag_s:.2f} s · effort {self.effort_psi:.2f} psi"
        )

        ax = axes[0]
        ax.plot(
            self.references[:, 0] * 100.0,
            self.references[:, 1] * 100.0,
            "--",
            color="gray",
            lw=1,
            label="Reference",
        )
        ax.plot(self.tips[:, 0] * 100.0, self.tips[:, 1] * 100.0, label="Tip")
        ax.plot(self.home[0] * 100.0, self.home[1] * 100.0, "k+", ms=10)
        ax.set(xlabel="x [cm]", ylabel="y [cm]", title="Tip path in x-y")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(self.times, lateral_error, lw=1)
        ax.axhline(
            self.rmse_mm,
            color="tab:orange",
            ls="--",
            label=f"RMSE {self.rmse_mm:.1f} mm",
        )
        ax.set(
            xlabel="Tracking time [s]",
            ylabel="Lateral error [mm]",
            title="Lateral tracking error",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        for index, name in enumerate(("S2", "S3", "S4")):
            ax.plot(self.times, self.commands[:, index], lw=1, label=name)
        baseline = self.task.baseline_command(np.inf)
        for value in np.unique(baseline):
            ax.axhline(value, color="gray", ls=":", lw=1)
        ax.set(
            xlabel="Tracking time [s]",
            ylabel="Pressure [psi]",
            title="Commanded pressures",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, axes


def _controller_command(
    controller: Controller,
    t: float,
    obs: Mapping[str, Any],
    ref: NDArray[np.float64],
    max_pressure: float,
) -> NDArray[np.float64]:
    command = np.asarray(controller.compute(t, obs, ref), dtype=float)
    if command.shape != (3,):
        raise ValueError(
            "controller.compute(t, obs, ref) must return shape (3,) "
            "for Segments 2, 3, and 4"
        )
    if not np.all(np.isfinite(command)):
        raise ValueError("controller command must contain only finite values")
    return np.clip(command, 0.0, max_pressure)


def _phase_lag_seconds(
    reference: ArrayLike,
    measured: ArrayLike,
    sample_hz: float,
    max_lag_s: float | None = None,
    min_amplitude_ratio: float = 0.01,
) -> float:
    """Estimate lag by cross-correlation; delayed measurements are positive.

    Lag is reported as zero when measured motion is negligible relative to the
    reference, because cross-correlating sensor drift has no physical meaning.
    """
    ref = np.asarray(reference, dtype=float)
    signal = np.asarray(measured, dtype=float)
    if ref.ndim != 1 or signal.shape != ref.shape or ref.size == 0:
        raise ValueError("reference and measured must be nonempty vectors of equal shape")
    if not np.isfinite(sample_hz) or sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive and finite")
    if max_lag_s is not None and (
        not np.isfinite(max_lag_s) or max_lag_s <= 0.0
    ):
        raise ValueError("max_lag_s must be positive and finite")
    if not np.isfinite(min_amplitude_ratio) or min_amplitude_ratio < 0.0:
        raise ValueError("min_amplitude_ratio must be nonnegative and finite")
    ref = ref - np.mean(ref)
    signal = signal - np.mean(signal)
    ref_rms = float(np.sqrt(np.mean(ref**2)))
    signal_rms = float(np.sqrt(np.mean(signal**2)))
    if (
        np.isclose(ref_rms, 0.0)
        or signal_rms < min_amplitude_ratio * ref_rms
    ):
        return 0.0
    correlation = np.correlate(signal, ref, mode="full")
    lags = np.arange(-len(ref) + 1, len(ref))
    if max_lag_s is not None:
        max_lag_samples = int(round(max_lag_s * sample_hz))
        selected = np.abs(lags) <= max_lag_samples
        correlation = correlation[selected]
        lags = lags[selected]
    lag_samples = int(lags[np.argmax(correlation)])
    return float(lag_samples / sample_hz)


def evaluate_controller(
    controller: Controller,
    *,
    task: CircularTrackingTask | None = None,
    sim=None,
) -> TrackingResult:
    """Run the common circular task and return deterministic metrics/signals.

    The simulator first settles at ``task.baseline_psi``. Its resulting tip
    position becomes the circle centre, which avoids scoring initial transient
    motion caused by the charged Segment-1 reservoir. Controller time is the
    monotonic plant clock (and always equals ``obs['time']``); stored result
    times and reference-generation times start at zero for the tracking phase.
    """
    task = task or CircularTrackingTask()
    if sim is None:
        from . import make_sim

        sim = make_sim(control_hz=task.control_hz, seed=0)
    expected_dt = 1.0 / task.control_hz
    if not np.isclose(sim.control_dt, expected_dt, rtol=0.0, atol=1e-12):
        raise ValueError("sim control rate must equal task.control_hz")

    command_limit = min(float(task.command_limit_psi), float(sim.cfg.p_max))
    baseline = task.baseline_command(command_limit)
    obs = sim.reset()
    for _ in range(task.settle_steps):
        obs = sim.step(baseline)
    home = np.asarray(obs["tip_pos"], dtype=float).copy()

    times = np.arange(task.track_steps, dtype=float) / task.control_hz
    tips = np.empty((task.track_steps, 3), dtype=float)
    refs = np.empty_like(tips)
    commands = np.empty((task.track_steps, 3), dtype=float)
    for index, t in enumerate(times):
        ref = make_reference(home, t, task)
        # Controller time matches the monotonic plant clock used on hardware;
        # ``t`` remains tracking-relative for trajectory generation and plots.
        command = _controller_command(
            controller, float(obs["time"]), obs, ref, command_limit
        )
        obs = sim.step(command)
        tips[index] = obs["tip_pos"]
        refs[index] = ref
        commands[index] = command

    lateral_error_mm = np.linalg.norm(tips[:, :2] - refs[:, :2], axis=1) * 1000.0
    rmse_mm = float(np.sqrt(np.mean(lateral_error_mm**2)))
    max_error_mm = float(np.max(lateral_error_mm))
    effort_psi = float(np.sqrt(np.mean((commands - baseline) ** 2)))

    phase_lag_s = _phase_lag_seconds(
        refs[:, 1],
        tips[:, 1],
        task.control_hz,
        max_lag_s=0.5 / task.frequency_hz,
    )

    return TrackingResult(
        rmse_mm=rmse_mm,
        max_error_mm=max_error_mm,
        phase_lag_s=phase_lag_s,
        effort_psi=effort_psi,
        home=home,
        times=times,
        tips=tips,
        references=refs,
        commands=commands,
        task=task,
    )
