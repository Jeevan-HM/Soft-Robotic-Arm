"""Train PRC, run a matched-preview PID, and compare paired rollouts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from pid import PIDController, PIDGains
from run_prc import (
    ClosedLoopData,
    EvaluationScenario,
    build_argument_parser as build_prc_argument_parser,
    calibrated_plant_metadata,
    run_closed_loop_demo,
    run_pipeline,
    tracking_metrics,
)


PROFILE_NAMES = ("step", "sine", "multisine", "all")
COMPARISON_SCENARIO_VERSION = 2


def build_argument_parser() -> argparse.ArgumentParser:
    parser = build_prc_argument_parser(live_default=False)
    parser.description = (
        "Compare PRC and PID on paired MuJoCo soft-arm rollouts."
    )
    parser.set_defaults(video_out=None)
    parser.add_argument("--pid-kp", type=float, default=1.0,
                        help="PID proportional gain [psi/deg] (default: 1.0)")
    parser.add_argument("--pid-ki", type=float, default=0.12,
                        help="PID integral gain [psi/(deg s)] (default: 0.12)")
    parser.add_argument("--pid-kd", type=float, default=0.03,
                        help="PID derivative gain [psi s/deg] (default: 0.03)")
    parser.add_argument(
        "--pid-derivative-cutoff",
        type=float,
        default=2.0,
        help="measurement-derivative low-pass cutoff [Hz] (default: 2)",
    )
    parser.add_argument(
        "--pid-integral-limit",
        type=float,
        default=8.0,
        help="absolute PID integral clamp [deg s] (default: 8)",
    )
    parser.add_argument(
        "--pid-antiwindup-gain",
        type=float,
        default=0.5,
        help="PID back-calculation gain [1/s] (default: 0.5)",
    )
    parser.add_argument(
        "--pid-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the same reference preview as PRC in the proportional term "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--pid-model-out",
        type=Path,
        default=Path("output/calibrated_pid_controller.json"),
    )
    parser.add_argument(
        "--comparison-log-out",
        type=Path,
        default=Path("output/calibrated_prc_vs_pid.csv"),
    )
    parser.add_argument(
        "--comparison-metrics-out",
        type=Path,
        default=Path("output/calibrated_prc_vs_pid_metrics.json"),
    )
    parser.add_argument(
        "--comparison-plot-out",
        type=Path,
        default=Path("output/calibrated_prc_vs_pid.png"),
    )
    return parser


def _validate_pid_arguments(args: argparse.Namespace) -> None:
    gains = np.array([args.pid_kp, args.pid_ki, args.pid_kd], dtype=float)
    if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
        raise ValueError("PID gains must be finite and nonnegative")
    if (not np.isfinite(args.pid_derivative_cutoff)
            or args.pid_derivative_cutoff <= 0.0):
        raise ValueError("pid-derivative-cutoff must be finite and positive")
    if (not np.isfinite(args.pid_integral_limit)
            or args.pid_integral_limit < 0.0):
        raise ValueError("pid-integral-limit must be finite and nonnegative")
    if (not np.isfinite(args.pid_antiwindup_gain)
            or args.pid_antiwindup_gain < 0.0):
        raise ValueError("pid-antiwindup-gain must be finite and nonnegative")


def pid_calibration_metadata(args: argparse.Namespace) -> dict:
    """Describe known provenance without making claims about custom gains."""
    uses_migration_defaults = bool(
        np.allclose(
            [args.pid_kp, args.pid_ki, args.pid_kd],
            [1.0, 0.12, 0.03],
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(args.reservoir_charge, 2.0)
        and np.isclose(args.pid_derivative_cutoff, 2.0)
        and np.isclose(args.pid_integral_limit, 8.0)
        and np.isclose(args.pid_antiwindup_gain, 0.5)
        and args.pid_preview
    )
    if not uses_migration_defaults:
        return {
            "evaluation_scenario_used_for_selection": None,
            "provenance": "unknown_for_custom_settings",
            "note": (
                "custom PID settings are outside the provisional baseline; "
                "whether they used this evaluation scenario is unknown"
            ),
        }
    return {
        "evaluation_scenario_used_for_selection": False,
        "provenance": "pid_calibration_record.json",
        "status": "provisional_migration_baseline",
        "plant_calibration": calibrated_plant_metadata(),
        "note": (
            "safe-frequency calibrated-plant baseline; a fresh multi-seed "
            "gain robustness tune is still required before comparison claims"
        ),
    }


def make_comparison_scenario(
    seconds_per_profile: float,
    delay_steps: int,
    limit_deg: float,
    control_hz: float,
    frequency_scale: float,
) -> EvaluationScenario:
    """Build the comparison-only reference, distinct from PID calibration."""
    if (not np.isfinite(seconds_per_profile) or seconds_per_profile <= 0.0
            or not np.isfinite(limit_deg) or limit_deg <= 0.0
            or not np.isfinite(control_hz) or control_hz <= 0.0
            or not np.isfinite(frequency_scale) or frequency_scale <= 0.0):
        raise ValueError("comparison scenario settings must be finite and positive")
    if delay_steps < 0:
        raise ValueError("delay_steps must be nonnegative")
    n = max(80, int(round(seconds_per_profile * control_hz)))
    t = np.arange(n) / control_hz

    step = np.zeros(n)
    boundaries = (
        int(round(0.18 * n)),
        int(round(0.38 * n)),
        int(round(0.62 * n)),
        int(round(0.80 * n)),
    )
    b1, b2, b3, b4 = boundaries
    step[b1:b2] = 0.57 * limit_deg
    step[b2:b3] = -0.46 * limit_deg
    step[b3:b4] = 0.22 * limit_deg

    sine = 0.66 * limit_deg * np.sin(
        2.0 * np.pi * (0.055 * frequency_scale) * t - 0.25
    )
    multisine = limit_deg * (
        0.40 * np.sin(2.0 * np.pi * (0.020 * frequency_scale) * t + 1.0)
        + 0.25 * np.sin(2.0 * np.pi * (0.050 * frequency_scale) * t + 2.35)
        + 0.14 * np.sin(2.0 * np.pi * (0.085 * frequency_scale) * t + 0.15)
    )
    reference = np.concatenate((step, sine, multisine))
    indices = np.minimum(
        np.arange(len(reference)) + delay_steps,
        len(reference) - 1,
    )
    preview = reference[indices]
    profile = np.repeat(np.array(["step", "sine", "multisine"]), n)
    return EvaluationScenario(control_hz, reference, preview, profile)


def comparison_metrics(
    demo: ClosedLoopData,
    bias_psi: float,
    pressure_min_psi: float,
    pressure_max_psi: float,
) -> dict[str, dict[str, float]]:
    """Return tracking and effort metrics on consistent profile masks."""
    base = tracking_metrics(demo)
    metrics: dict[str, dict[str, float]] = {}
    for name in PROFILE_NAMES:
        mask = (
            np.ones(len(demo.time_s), dtype=bool)
            if name == "all"
            else demo.profile == name
        )
        error = demo.reference_deg[mask] - demo.bend_deg[mask]
        controlled = demo.command_psi[mask, 1]
        saturated = np.isclose(controlled, pressure_min_psi, atol=1e-9) | np.isclose(
            controlled, pressure_max_psi, atol=1e-9
        )
        zero_baseline = base[name]["zero_bend_baseline_rmse_deg"]
        metrics[name] = {
            **base[name],
            "mae_deg": float(np.mean(np.abs(error))),
            "normalized_rmse": float(base[name]["rmse_deg"] / zero_baseline),
            "bend_peak_to_peak_deg": float(np.ptp(demo.bend_deg[mask])),
            "segment3_rms_excursion_psi": float(np.sqrt(np.mean(
                (controlled - bias_psi) ** 2
            ))),
            "saturation_fraction": float(np.mean(saturated)),
        }
    return metrics


def common_safety_failures(
    demo: ClosedLoopData,
    pressure_min_psi: float,
    pressure_max_psi: float,
    slew_rate_psi_s: float,
    initial_pressure_psi: float,
    control_hz: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    arrays = (
        demo.reference_deg,
        demo.bend_deg,
        demo.reservoir_psi,
        demo.measured_pressure_psi,
        demo.command_psi,
        demo.raw_command_psi,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        failures.append("nonfinite rollout data")
    if np.any(demo.fallback):
        failures.append("controller fallback occurred")
    tolerance = 1e-9
    if (np.any(demo.command_psi < pressure_min_psi - tolerance)
            or np.any(demo.command_psi > pressure_max_psi + tolerance)):
        failures.append("pressure bound violation")
    trajectory = np.vstack((
        np.full(3, initial_pressure_psi),
        demo.command_psi,
    ))
    allowed = slew_rate_psi_s / control_hz
    if np.any(np.abs(np.diff(trajectory, axis=0)) > allowed + tolerance):
        failures.append("pressure slew violation")
    return tuple(failures)


def save_comparison_log(
    path: str | Path,
    prc: ClosedLoopData,
    pid: ClosedLoopData,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (
        np.array_equal(prc.time_s, pid.time_s)
        and np.array_equal(prc.profile, pid.profile)
        and np.array_equal(prc.reference_deg, pid.reference_deg)
        and np.array_equal(prc.preview_reference_deg, pid.preview_reference_deg)
    ):
        raise ValueError("PRC and PID rollouts are not paired to one scenario")

    header = [
        "tick", "time_s", "profile", "reference_deg", "preview_reference_deg",
        "prc_bend_deg", "pid_bend_deg", "prc_error_deg", "pid_error_deg",
    ]
    for controller in ("prc", "pid"):
        header.extend(
            f"{controller}_measured_s{segment}_psi" for segment in range(2, 5)
        )
        header.extend(
            f"{controller}_command_s{segment}_psi" for segment in range(2, 5)
        )
        header.extend(
            f"{controller}_raw_s{segment}_psi" for segment in range(2, 5)
        )
        header.extend((f"{controller}_projected", f"{controller}_fallback"))

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for k in range(len(prc.time_s)):
            row: list[object] = [
                k,
                f"{prc.time_s[k]:.6f}",
                str(prc.profile[k]),
                f"{prc.reference_deg[k]:.8g}",
                f"{prc.preview_reference_deg[k]:.8g}",
                f"{prc.bend_deg[k]:.8g}",
                f"{pid.bend_deg[k]:.8g}",
                f"{prc.reference_deg[k] - prc.bend_deg[k]:.8g}",
                f"{pid.reference_deg[k] - pid.bend_deg[k]:.8g}",
            ]
            for demo in (prc, pid):
                row.extend(f"{value:.8g}" for value in demo.measured_pressure_psi[k])
                row.extend(f"{value:.8g}" for value in demo.command_psi[k])
                row.extend(f"{value:.8g}" for value in demo.raw_command_psi[k])
                row.extend((int(demo.projected[k]), int(demo.fallback[k])))
            writer.writerow(row)
    return path


def save_comparison_plot(
    path: str | Path,
    prc: ClosedLoopData,
    pid: ClosedLoopData,
    prc_metrics: dict[str, dict[str, float]],
    pid_metrics: dict[str, dict[str, float]],
    bias_psi: float,
    pressure_min_psi: float,
    pressure_max_psi: float,
    pid_label: str,
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12.0, 10.0))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4, 1, height_ratios=(2.0, 1.25, 1.25, 1.35))
    tracking_axis = figure.add_subplot(grid[0])
    error_axis = figure.add_subplot(grid[1], sharex=tracking_axis)
    command_axis = figure.add_subplot(grid[2], sharex=tracking_axis)
    metric_axis = figure.add_subplot(grid[3])

    tracking_axis.plot(
        prc.time_s, prc.reference_deg, "k--", lw=1.2, label="reference"
    )
    tracking_axis.plot(
        prc.time_s, prc.bend_deg, color="tab:purple", lw=1.4, label="PRC"
    )
    tracking_axis.plot(
        pid.time_s, pid.bend_deg, color="tab:orange", lw=1.25, label=pid_label
    )
    tracking_axis.set_ylabel("bend [deg]")
    tracking_axis.legend(loc="upper right", ncol=3)

    error_axis.plot(
        prc.time_s,
        prc.reference_deg - prc.bend_deg,
        color="tab:purple",
        lw=1.1,
        label="PRC error",
    )
    error_axis.plot(
        pid.time_s,
        pid.reference_deg - pid.bend_deg,
        color="tab:orange",
        lw=1.1,
        label=f"{pid_label} error",
    )
    error_axis.axhline(0.0, color="0.35", lw=0.8)
    error_axis.set_ylabel("error [deg]")
    error_axis.legend(loc="upper right", ncol=2)

    command_axis.plot(
        prc.time_s,
        prc.command_psi[:, 1],
        color="tab:purple",
        lw=1.1,
        label="PRC Segment 3",
    )
    command_axis.plot(
        pid.time_s,
        pid.command_psi[:, 1],
        color="tab:orange",
        lw=1.1,
        label=f"{pid_label} Segment 3",
    )
    command_axis.axhline(bias_psi, color="0.35", lw=0.8, ls=":", label="x bias")
    command_axis.axhline(pressure_min_psi, color="0.7", lw=0.7)
    command_axis.axhline(pressure_max_psi, color="0.7", lw=0.7)
    command_axis.set_ylabel("command [psi]")
    command_axis.set_xlabel("time [s]")
    command_axis.legend(loc="upper right", ncol=3)

    boundaries = np.flatnonzero(prc.profile[1:] != prc.profile[:-1]) + 1
    for boundary in boundaries:
        for axis in (tracking_axis, error_axis, command_axis):
            axis.axvline(prc.time_s[boundary], color="0.6", lw=0.75, ls=":")
    for name in ("step", "sine", "multisine"):
        indices = np.flatnonzero(prc.profile == name)
        center = prc.time_s[indices[len(indices) // 2]]
        tracking_axis.text(
            center,
            1.02,
            name,
            transform=tracking_axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for axis in (tracking_axis, error_axis, command_axis):
        axis.grid(alpha=0.22)

    x = np.arange(len(PROFILE_NAMES))
    width = 0.36
    metric_axis.bar(
        x - width / 2,
        [prc_metrics[name]["rmse_deg"] for name in PROFILE_NAMES],
        width,
        color="tab:purple",
        label="PRC",
    )
    metric_axis.bar(
        x + width / 2,
        [pid_metrics[name]["rmse_deg"] for name in PROFILE_NAMES],
        width,
        color="tab:orange",
        label=pid_label,
    )
    metric_axis.set_xticks(x, PROFILE_NAMES)
    metric_axis.set_ylabel("RMSE [deg]")
    metric_axis.legend(loc="upper right", ncol=2)
    metric_axis.grid(axis="y", alpha=0.22)

    figure.suptitle(
        "MuJoCo controller comparison (paired simulation)\n"
        f"overall RMSE: PRC {prc_metrics['all']['rmse_deg']:.3f} deg, "
        f"{pid_label} {pid_metrics['all']['rmse_deg']:.3f} deg; "
        "Segments 2 and 4 fixed at x"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150)
    figure.clear()
    return path


def run_comparison(args: argparse.Namespace) -> dict:
    _validate_pid_arguments(args)
    prc_result = run_pipeline(args, persist_artifacts=False)
    shared_config = prc_result["config"]
    comparison_scenario = make_comparison_scenario(
        args.profile_seconds,
        args.delay_steps,
        prc_result["reference_limit_deg"],
        shared_config.control_hz,
        args.frequency_scale,
    )
    comparison_seed = args.seed + 2000
    prc_demo = run_closed_loop_demo(
        prc_result["controller"],
        args.reservoir_charge,
        comparison_seed,
        args.settle_seconds,
        args.profile_seconds,
        args.delay_steps,
        prc_result["reference_limit_deg"],
        frequency_scale=args.frequency_scale,
        live=False,
        scenario=comparison_scenario,
    )
    pid_config = replace(
        shared_config,
        derivative_cutoff_hz=args.pid_derivative_cutoff,
        integral_limit_deg_s=args.pid_integral_limit,
        antiwindup_gain=args.pid_antiwindup_gain,
    )
    gains = PIDGains(
        kp_psi_per_deg=args.pid_kp,
        ki_psi_per_deg_s=args.pid_ki,
        kd_psi_s_per_deg=args.pid_kd,
    )
    calibration_metadata = pid_calibration_metadata(args)
    pid = PIDController(
        gains,
        pid_config,
        bias_pressure_psi=np.full(3, args.reservoir_charge),
        use_preview_for_proportional=args.pid_preview,
        metadata={
            "controller": (
                "preview_compensated_pid" if args.pid_preview else "causal_pid"
            ),
            "simulation_only": True,
            "controlled_segment": 3,
            "fixed_segments": [2, 4],
            "reservoir_charge_psi": float(args.reservoir_charge),
            "gain_calibration": calibration_metadata,
            "evaluation_preview_used": bool(args.pid_preview),
            "comparison_scenario_version": COMPARISON_SCENARIO_VERSION,
            "plant_calibration": calibrated_plant_metadata(),
        },
    )
    pid_demo = run_closed_loop_demo(
        pid,
        args.reservoir_charge,
        comparison_seed,
        args.settle_seconds,
        args.profile_seconds,
        args.delay_steps,
        prc_result["reference_limit_deg"],
        frequency_scale=args.frequency_scale,
        live=False,
        scenario=comparison_scenario,
    )

    pressure_min = float(np.min(pid.projector.p_min))
    pressure_max = float(np.max(pid.projector.p_max))
    prc_metrics = comparison_metrics(
        prc_demo, args.reservoir_charge, pressure_min, pressure_max
    )
    pid_metrics = comparison_metrics(
        pid_demo, args.reservoir_charge, pressure_min, pressure_max
    )
    prc_safety = common_safety_failures(
        prc_demo,
        pressure_min,
        pressure_max,
        args.slew_rate,
        args.reservoir_charge,
        shared_config.control_hz,
    )
    pid_safety = common_safety_failures(
        pid_demo,
        pressure_min,
        pressure_max,
        args.slew_rate,
        args.reservoir_charge,
        shared_config.control_hz,
    )

    pid_model_path = pid.save(args.pid_model_out)
    log_path = save_comparison_log(
        args.comparison_log_out, prc_demo, pid_demo
    )
    plot_path = save_comparison_plot(
        args.comparison_plot_out,
        prc_demo,
        pid_demo,
        prc_metrics,
        pid_metrics,
        args.reservoir_charge,
        pressure_min,
        pressure_max,
        "PID (matched preview)" if args.pid_preview else "PID (causal)",
    )
    metrics_path = Path(args.comparison_metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "simulation_only": True,
        "paired_seed": int(comparison_seed),
        "common_reference": True,
        "comparison_scenario_version": COMPARISON_SCENARIO_VERSION,
        "plant_calibration": calibrated_plant_metadata(),
        "prc_reference_preview_steps": int(args.delay_steps),
        "pid_reference_preview_steps": (
            int(args.delay_steps) if args.pid_preview else 0
        ),
        "pid_gains": {
            "kp_psi_per_deg": gains.kp_psi_per_deg,
            "ki_psi_per_deg_s": gains.ki_psi_per_deg_s,
            "kd_psi_s_per_deg": gains.kd_psi_s_per_deg,
        },
        "pid_derivative_cutoff_hz": float(args.pid_derivative_cutoff),
        "pid_integral_limit_deg_s": float(args.pid_integral_limit),
        "pid_antiwindup_gain_s_inv": float(args.pid_antiwindup_gain),
        "pid_gain_calibration": calibration_metadata,
        "prc": prc_metrics,
        "pid": pid_metrics,
        "prc_common_safety_failures": list(prc_safety),
        "pid_common_safety_failures": list(pid_safety),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")
    return {
        "prc_result": prc_result,
        "prc_demo": prc_demo,
        "pid_controller": pid,
        "pid_demo": pid_demo,
        "comparison_scenario": comparison_scenario,
        "prc_metrics": prc_metrics,
        "pid_metrics": pid_metrics,
        "prc_safety_failures": prc_safety,
        "pid_safety_failures": pid_safety,
        "pid_model_path": pid_model_path,
        "log_path": log_path,
        "metrics_path": metrics_path,
        "plot_path": plot_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_comparison(args)
    print("Paired PRC vs PID comparison complete (simulation only).")
    print("  profile       PRC RMSE    PID RMSE    lower RMSE")
    for name in PROFILE_NAMES:
        prc_rmse = result["prc_metrics"][name]["rmse_deg"]
        pid_rmse = result["pid_metrics"][name]["rmse_deg"]
        winner = "PRC" if prc_rmse < pid_rmse else "PID"
        print(f"  {name:10s}  {prc_rmse:8.3f}    {pid_rmse:8.3f}    {winner}")
    if args.pid_preview:
        print("  information: PRC and PID use the same reference preview")
    else:
        print("  information: PRC uses reference preview; PID is causal only")
    for controller, failures in (
        ("PRC", result["prc_safety_failures"]),
        ("PID", result["pid_safety_failures"]),
    ):
        status = "PASSED" if not failures else "FAILED: " + "; ".join(failures)
        print(f"  {controller} common safety checks: {status}")
    print(f"  PID settings: {result['pid_model_path']}")
    print(f"  paired log:   {result['log_path']}")
    print(f"  metrics:      {result['metrics_path']}")
    print(f"  plot:         {result['plot_path']}")
    success = (
        result["prc_result"]["accepted"]
        and not result["prc_safety_failures"]
        and not result["pid_safety_failures"]
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
