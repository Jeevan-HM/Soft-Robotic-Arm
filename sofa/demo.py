"""
demo.py — Motion primitive demos for the SOFA FEM soft arm.

Ports the four demonstrations from the MuJoCo version (mujoco/demo.py):

    1. single_column_bend()  — one column fully pressurised
    2. axial_extension()     — all 4 columns equal → pure extension
    3. circle_demo()         — phase-shifted sinusoids → circular tip path
    4. triangle_demo()       — linear vertex interpolation → triangular tip path

Each demo prints tip trajectory statistics.  Visualisation requires SOFA GUI
(run with `runSofa demo.py` or use `python demo.py --demo circle --plot`).

Usage
-----
    # Run in headless mode (prints stats only)
    python sofa/demo.py --demo bend
    python sofa/demo.py --demo axial
    python sofa/demo.py --demo circle
    python sofa/demo.py --demo triangle

    # Plot tip trajectory (requires matplotlib)
    python sofa/demo.py --demo circle --plot

    # Run all demos
    python sofa/demo.py --demo all --plot
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from arm_config import ArmConfig
from soft_arm_sim import SoftArmSim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_tip_stats(label: str, tips: np.ndarray) -> None:
    """Print range statistics for a tip trajectory array (T, 3)."""
    lateral = np.hypot(tips[:, 0], tips[:, 1]) * 100   # cm
    z_delta = (tips[:, 2] - tips[0, 2]) * 100           # cm
    print(f"\n[{label}]")
    print(f"  Max lateral deflection : {lateral.max():.2f} cm")
    print(f"  Max axial displacement : {abs(z_delta).max():.2f} cm")
    print(f"  Final tip position     : {np.round(tips[-1] * 100, 2)} cm")


def _make_sim(pre_inflation_psi: float = 1.5) -> SoftArmSim:
    sim = SoftArmSim(cfg=ArmConfig())
    sim.set_pre_inflation(pre_inflation_psi)
    sim.reset()
    return sim


# ---------------------------------------------------------------------------
# Demo 1 — Single column bend
# ---------------------------------------------------------------------------

def single_column_bend(col: int = 0, pressure_psi: float = 4.0,
                       duration_s: float = 3.0) -> np.ndarray:
    """Pressurise one column fully; observe bending direction and magnitude.

    Parameters
    ----------
    col : int
        Column index (0=East, 1=North, 2=West, 3=South).
    pressure_psi : float
        Commanded pressure [psi].
    duration_s : float
        Simulation duration [s].

    Returns
    -------
    tips : (T, 3) float
        Tip positions over time [m].
    """
    col_labels = ["East", "North", "West", "South"]
    print(f"\n=== Single Column Bend: col {col} ({col_labels[col]}) @ {pressure_psi} psi ===")

    sim  = _make_sim()
    n    = int(duration_s * 100)   # 100 Hz
    tips = np.zeros((n, 3))

    P = np.zeros((4, 5))
    P[col, :] = pressure_psi

    for i in range(n):
        obs     = sim.step(P)
        tips[i] = obs["tip_pos"]

    _print_tip_stats(f"Bend col {col}", tips)
    return tips


# ---------------------------------------------------------------------------
# Demo 2 — Axial extension
# ---------------------------------------------------------------------------

def axial_extension(pressure_psi: float = 4.0, duration_s: float = 3.0) -> np.ndarray:
    """Pressurise all 4 columns equally — bending cancels, pure axial extension.

    Returns
    -------
    tips : (T, 3) float
    """
    print(f"\n=== Axial Extension: all columns @ {pressure_psi} psi ===")

    sim  = _make_sim()
    n    = int(duration_s * 100)
    tips = np.zeros((n, 3))

    P = np.ones((4, 5)) * pressure_psi

    for i in range(n):
        obs     = sim.step(P)
        tips[i] = obs["tip_pos"]

    lateral = np.hypot(tips[:, 0], tips[:, 1]) * 100
    print(f"  Max lateral drift      : {lateral.max():.3f} cm  (should be ≈ 0)")
    _print_tip_stats("Axial extension", tips)
    return tips


# ---------------------------------------------------------------------------
# Demo 3 — Circular trajectory
# ---------------------------------------------------------------------------

def circle_demo(freq_hz: float = 0.5, amp_psi: float = 3.0,
                duration_s: float = 8.0) -> np.ndarray:
    """Drive the tip in a circle using phase-shifted sinusoidal pressures.

    Each column receives a sinusoid phase-shifted by 90°:
        col 0 (East)  : sin(2π f t)
        col 1 (North) : sin(2π f t + π/2)
        col 2 (West)  : sin(2π f t + π)
        col 3 (South) : sin(2π f t + 3π/2)

    Only the positive half of each sinusoid is applied (pressure ≥ 0).

    Returns
    -------
    tips : (T, 3) float
    """
    print(f"\n=== Circle Demo: freq={freq_hz} Hz, amp={amp_psi} psi, t={duration_s} s ===")

    sim  = _make_sim()
    n    = int(duration_s * 100)
    t    = np.arange(n) / 100.0
    tips = np.zeros((n, 3))

    phase_offsets = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

    for i in range(n):
        P = np.zeros((4, 5))
        for col, phi in enumerate(phase_offsets):
            val = amp_psi * math.sin(2 * math.pi * freq_hz * t[i] + phi)
            P[col, :] = max(0.0, val)
        obs     = sim.step(P)
        tips[i] = obs["tip_pos"]

    lateral = np.hypot(tips[:, 0], tips[:, 1]) * 100
    print(f"  Circle radius (mean)   : {lateral.mean():.2f} cm")
    print(f"  Circle radius (std)    : {lateral.std():.2f} cm")
    _print_tip_stats("Circle", tips)
    return tips


# ---------------------------------------------------------------------------
# Demo 4 — Triangle trajectory
# ---------------------------------------------------------------------------

def triangle_demo(n_vertices: int = 3, t_per_edge_s: float = 2.5,
                  amp_psi: float = 2.2) -> np.ndarray:
    """Drive the tip along a triangle by interpolating between vertex directions.

    The arm tip traces straight edges between 3 vertices at 0°, 120°, 240°
    by linearly interpolating the Cartesian unit vectors of adjacent vertices
    and projecting onto each column's azimuth.

    This matches the reference MuJoCo implementation exactly.

    Returns
    -------
    tips : (T, 3) float
    """
    print(f"\n=== Triangle Demo: {n_vertices} vertices, {t_per_edge_s} s/edge, amp={amp_psi} psi ===")

    sim       = _make_sim()
    cfg       = sim.cfg
    col_phis  = np.radians(cfg.col_azimuths_deg())   # (4,) column azimuths

    vertex_angles = np.radians(np.arange(n_vertices) * 360.0 / n_vertices)
    xy_verts      = np.stack([np.cos(vertex_angles), np.sin(vertex_angles)], axis=1)  # (3, 2)

    n     = int(n_vertices * t_per_edge_s * 100)
    t     = np.arange(n) / 100.0
    tips  = np.zeros((n, 3))

    frac   = (t % t_per_edge_s) / t_per_edge_s
    v_cur  = (t / t_per_edge_s).astype(int) % n_vertices
    v_next = (v_cur + 1) % n_vertices

    v0     = xy_verts[v_cur]                                   # (n, 2)
    v1     = xy_verts[v_next]                                  # (n, 2)
    target = v0 + frac[:, None] * (v1 - v0)                   # (n, 2) — straight edge

    # Project target direction onto each column's axis
    cmd = amp_psi * (
        target[:, 0:1] * np.cos(col_phis)[None, :] +
        target[:, 1:2] * np.sin(col_phis)[None, :]
    )                                                           # (n, 4)
    P_arr = np.clip(cmd, 0, None)[:, :, None] * np.ones((1, 1, 5))  # (n, 4, 5)

    for i, P in enumerate(P_arr):
        obs     = sim.step(P)
        tips[i] = obs["tip_pos"]

    _print_tip_stats("Triangle", tips)
    return tips


# ---------------------------------------------------------------------------
# Optional matplotlib plotting
# ---------------------------------------------------------------------------

def _plot_tip_trajectory(tips_dict: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262d")
    ax.grid(True, color="#21262d", lw=0.5)

    colors = {
        "bend_east":    "#FF6B6B",
        "axial":        "#FFE66D",
        "circle":       "#4ECDC4",
        "triangle":     "#80ED99",
    }

    for label, tips in tips_dict.items():
        c = colors.get(label, "#e6edf3")
        ax.plot(tips[:, 0] * 100, tips[:, 1] * 100,
                color=c, lw=1.5, label=label)
        ax.scatter(tips[0, 0] * 100, tips[0, 1] * 100, color=c, s=20, zorder=5)

    ax.set_xlabel("X [cm]", color="#e6edf3")
    ax.set_ylabel("Y [cm]", color="#e6edf3")
    ax.set_aspect("equal")
    ax.set_title("Tip trajectories — SOFA FEM simulation",
                 color="#e6edf3", fontweight="bold")
    ax.legend(fontsize=9, facecolor="#161b22", labelcolor="#e6edf3")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOFA soft arm motion demos.")
    parser.add_argument(
        "--demo", default="all",
        choices=["bend", "axial", "circle", "triangle", "all"],
        help="Which demo to run (default: all)",
    )
    parser.add_argument("--plot", action="store_true", help="Plot tip trajectories")
    args = parser.parse_args()

    results: dict[str, np.ndarray] = {}

    if args.demo in ("bend", "all"):
        results["bend_east"] = single_column_bend(col=0)

    if args.demo in ("axial", "all"):
        results["axial"] = axial_extension()

    if args.demo in ("circle", "all"):
        results["circle"] = circle_demo()

    if args.demo in ("triangle", "all"):
        results["triangle"] = triangle_demo()

    if args.plot and results:
        _plot_tip_trajectory(results)
