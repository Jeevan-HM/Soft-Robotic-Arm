"""
Validation demo for the soft arm digital twin.

Architecture: 4 segment columns (Seg 1-4) × 5 floor levels.
Control:      P[s, k] = pressure [psi] in segment s at floor k.
              Seg 1 = col index 0, Seg 2 = col index 1, ... Seg 4 = col index 3.
Max pressure: 10.0 psi.

Demos
-----
1. step_response     — Seg 1 step input at two pre-inflation levels.
2. segment_local     — Each segment independently → tip deflection direction map.
3. single_seg_axial  — Seg 4 sine wave, Segs 1-3 off → arm oscillates.
4. circle_demo       — Phase-shifted pressures → circular tip trajectory.
5. triangle_demo     — Linearly swept bending direction → triangular trajectory.
6. prc_sensor_check  — Band-limited noise; verify sensors carry body-state info.

Every plot includes a pressure-input waveform panel.

Run:  uv run demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from soft_arm_sim import SoftArmSim

HZ = 100
OUT = "output"
os.makedirs(OUT, exist_ok=True)

# ── Segment labels / colours (Seg 1-4 = col index 0-3) ──────────────────────
SEG_LABELS = ["Seg 1", "Seg 2", "Seg 3", "Seg 4"]
SEG_COLORS = ["tab:orange", "tab:green", "tab:red", "tab:blue"]


def _plot_inputs(ax, t_hist, P_hist, title="Pressure commands"):
    """Plot the 4-segment mean pressure waveforms on ax."""
    P = np.array(P_hist)   # (T, 4, 5)
    t = np.array(t_hist)
    for s in range(4):
        ax.plot(t, P[:, s, :].mean(axis=1),
                color=SEG_COLORS[s], label=SEG_LABELS[s], lw=1.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("pressure [psi]")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, None)


# ─────────────────────────────────────────────────── 1. Step response ──────
def step_response():
    """Seg 1 (col 0) step at 3.0 psi for two pre-inflation levels.
    Stiffness modulation: higher pre-inflation → stiffer → smaller deflection."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")

    for col_idx, p_pre in enumerate([0.0, 2.2]):
        sim = SoftArmSim()
        sim.set_pre_inflation(p_pre)
        P = np.zeros((4, 5))
        t_hist, tip_hist, P_hist = [], [], []

        for k in range(int(4.0 * HZ)):
            P[:] = 0.0
            if k >= HZ:
                P[0, :] = 3.0   # Seg 1, all 5 floors
            obs = sim.step(P)
            t_hist.append(obs["time"])
            tip_hist.append(obs["tip_pos"].copy())
            P_hist.append(P.copy())

        tip = np.array(tip_hist)
        ax_r = axes[0, col_idx]
        ax_r.plot(t_hist, tip[:, 0] * 100, label="tip x", lw=1.5,
                  color=SEG_COLORS[0])
        ax_r.plot(t_hist, (tip[:, 2] - tip[0, 2]) * 100, label="tip z (rel)",
                  lw=1.5, color="gray", ls="--")
        ax_r.set_title(f"Seg 1 step 3.0 psi  |  pre-inflation = {p_pre:.1f} psi")
        ax_r.set_ylabel("deflection [cm]")
        ax_r.legend(fontsize=8)
        ax_r.grid(alpha=0.3)

        _plot_inputs(axes[1, col_idx], t_hist, P_hist,
                     title=f"Pressure inputs (pre={p_pre:.1f} psi)")

    fig.suptitle("Step Response — Segment 1", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/step_response.png", dpi=130)
    print(f"saved {OUT}/step_response.png")


# ──────────────────────────────────────────── 2. Per-segment direction map ──
def segment_local():
    """Each of the 4 segments independently → tip deflection direction map."""
    sim = SoftArmSim()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_dir, ax_inp = axes

    global_t = 0.0
    t_hist, P_hist = [], []

    for s in range(sim.cfg.n_segments):
        sim.reset()
        sim.set_pre_inflation(1.2)
        P = np.zeros((4, 5))
        P[s, :] = 3.0   # one segment, all 5 floors

        t0 = global_t
        for _ in range(int(3 * HZ)):
            obs = sim.step(P)
            t_hist.append(t0 + obs["time"])
            P_hist.append(P.copy())
        global_t = t0 + obs["time"] + 1.0 / HZ

        d = obs["tip_pos"]
        mag = np.hypot(d[0], d[1]) * 100
        ax_dir.annotate(
            "", xy=(d[0] * 100, d[1] * 100), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=SEG_COLORS[s], lw=2.5))
        ax_dir.scatter(d[0] * 100, d[1] * 100, color=SEG_COLORS[s], s=90, zorder=5,
                       label=f"{SEG_LABELS[s]}: {mag:.1f} cm")
        print(f"  {SEG_LABELS[s]}: tip ({d[0]*100:.1f}, {d[1]*100:.1f}) cm")

    lim = 12
    ax_dir.set_xlim(-lim, lim); ax_dir.set_ylim(-lim, lim)
    ax_dir.set_xlabel("tip x [cm]"); ax_dir.set_ylabel("tip y [cm]")
    ax_dir.set_title("Per-segment bending (3.0 psi, all 5 floors)")
    ax_dir.axhline(0, lw=0.5, c="gray"); ax_dir.axvline(0, lw=0.5, c="gray")
    ax_dir.legend(fontsize=8); ax_dir.grid(alpha=0.3); ax_dir.set_aspect("equal")

    _plot_inputs(ax_inp, t_hist, P_hist, "Pressure: Seg 1→2→3→4 (each 3 s)")
    fig.suptitle("Segment-Local Actuation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/segment_local.png", dpi=130)
    print(f"saved {OUT}/segment_local.png")


# ─────────────────────────────────────────────── 3. Axial — Seg 4 sine ─────
def single_segment_axial(video=True):
    """Sine wave pressure on Segment 4 only; Segments 1-3 stay at zero.

    Seg 4 (col index 3) expands axially with each pressure cycle.
    The opposite Seg 2 (col index 1) is passively compressed by the
    structural coupling, causing the arm to oscillate in the Seg 4–Seg 2 plane.
    """
    sim = SoftArmSim()
    A_offset = 3.6    # [psi] DC offset — keeps pressure >= 0
    A_amp    = 3.6    # [psi] sine amplitude — ranges 0 to 7.2 psi
    f_sine   = 0.5    # [Hz]
    T        = 8.0    # [s] total duration

    frames, tip_hist, t_hist, P_hist = [], [], [], []

    for k in range(int(T * HZ)):
        t = k / HZ
        P = np.zeros((4, 5))
        # Seg 4 = col index 3
        P[3, :] = A_offset + A_amp * np.sin(2 * np.pi * f_sine * t)
        P = np.clip(P, 0, sim.cfg.p_max)
        obs = sim.step(P)
        tip_hist.append(obs["tip_pos"].copy())
        t_hist.append(obs["time"])
        P_hist.append(P.copy())
        if video and k % 3 == 0:
            frames.append(sim.render_frame())

    tip = np.array(tip_hist)
    tip = np.array(tip_hist)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_traj, ax_tip, ax_inp = axes

    # Trajectory in Y-Z plane
    ax_traj.plot(tip[:, 1] * 100, (tip[:, 2] - tip[0, 2]) * 100, lw=1.5, color="tab:blue")
    ax_traj.set_xlabel("tip y [cm]")
    ax_traj.set_ylabel("tip z change [cm]")
    ax_traj.set_title("Tip Trajectory (Y-Z plane)")
    ax_traj.grid(alpha=0.3)

    # Time series of tip components
    ax_tip.plot(t_hist, tip[:, 0] * 100, label="tip x", lw=1.5, color="tab:orange")
    ax_tip.plot(t_hist, tip[:, 1] * 100, label="tip y", lw=1.5, color="tab:green")
    ax_tip.plot(t_hist, (tip[:, 2] - tip[0, 2]) * 100,
                label="tip z (rel)", lw=1.5, color="gray", ls="--")
    ax_tip.set_xlabel("time [s]")
    ax_tip.set_ylabel("tip deflection [cm]")
    ax_tip.set_title(
        f"Axial — Seg 4 sine ({f_sine:.1f} Hz, {A_amp:.1f} psi amp)\n"
        "Seg 4 expands → opp Seg 2 compresses"
    )
    ax_tip.legend(fontsize=8)
    ax_tip.grid(alpha=0.3)

    # Inputs
    _plot_inputs(ax_inp, t_hist, P_hist,
                 f"Pressure input — Segs 1-3 off")

    fig.suptitle("Axial Actuation — Segment 4 Sine Wave", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/axial_extension.png", dpi=130)
    print(f"saved {OUT}/axial_extension.png")

    if video and frames:
        imageio.mimsave(f"{OUT}/axial_demo.mp4", frames, fps=33, quality=8)
        print(f"saved {OUT}/axial_demo.mp4 ({len(frames)} frames)")

    sim.save_pressure_log(f"{OUT}/pressure_axial.csv")


# ────────────────────────────────────────────── 4. Circular trajectory ──────
def circle_demo(video=True):
    """Phase-shifted cosine pressure commands → rotating bending moment → circle."""
    sim = SoftArmSim()
    sim.set_pre_inflation(1.5)
    phis = sim.cfg.col_azimuths()   # (4,) [0, π/2, π, 3π/2]

    frames, tip = [], []
    t_hist, P_hist = [], []
    T, f_orb = 8.0, 0.4

    for k in range(int(T * HZ)):
        t = k / HZ
        cmd_cols = 1.7 + 1.5 * np.cos(2 * np.pi * f_orb * t - phis)   # (4,)
        P = np.tile(np.clip(cmd_cols, 0, None)[:, None], (1, 5))          # (4, 5)
        obs = sim.step(P)
        tip.append(obs["tip_pos"].copy())
        t_hist.append(obs["time"])
        P_hist.append(P.copy())
        if video and k % 3 == 0:
            frames.append(sim.render_frame())

    tip = np.array(tip)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_traj, ax_inp = axes
    ax_traj.plot(tip[:, 0] * 100, tip[:, 1] * 100, lw=1.5)
    ax_traj.set_xlabel("tip x [cm]"); ax_traj.set_ylabel("tip y [cm]")
    ax_traj.set_title("Tip trajectory — circle")
    ax_traj.axis("equal"); ax_traj.grid(alpha=0.3)
    _plot_inputs(ax_inp, t_hist, P_hist, "Pressure commands (Seg 1-4)")
    fig.suptitle("Circular Trajectory — Phase-Shifted Segment Pressures", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/tip_circle.png", dpi=130)
    print(f"saved {OUT}/tip_circle.png")

    if video:
        imageio.mimsave(f"{OUT}/soft_arm_demo.mp4", frames, fps=33, quality=8)
        print(f"saved {OUT}/soft_arm_demo.mp4 ({len(frames)} frames)")

    sim.save_pressure_log(f"{OUT}/pressure_circle.csv")


# ────────────────────────────────────────────── 5. Triangle trajectory ──────
def triangle_demo(video=True):
    """Linearly sweep bending direction between 3 vertices → triangle."""
    sim = SoftArmSim()
    sim.set_pre_inflation(1.5)
    phis = sim.cfg.col_azimuths()   # (4,)

    n_vertices = 3
    T_per_edge = 2.5                # [s] per triangle edge
    A_drive = 2.2                   # [psi] pressure amplitude
    T_total = n_vertices * T_per_edge
    vertex_angles = np.deg2rad(np.arange(n_vertices) * 360.0 / n_vertices)

    frames, tip = [], []
    t_hist, P_hist = [], []

    for k in range(int(T_total * HZ)):
        t = k / HZ
        frac = (t % T_per_edge) / T_per_edge
        v_cur = int(t / T_per_edge) % n_vertices
        v_next = (v_cur + 1) % n_vertices

        # Target Cartesian direction
        x0, y0 = np.cos(vertex_angles[v_cur]), np.sin(vertex_angles[v_cur])
        x1, y1 = np.cos(vertex_angles[v_next]), np.sin(vertex_angles[v_next])

        # Linearly interpolate Cartesian coordinates
        tx = x0 + frac * (x1 - x0)
        ty = y0 + frac * (y1 - y0)

        # Project Cartesian target onto each column's axis
        cmd_cols = A_drive * (tx * np.cos(phis) + ty * np.sin(phis))
        P = np.tile(np.clip(cmd_cols, 0, None)[:, None], (1, 5))
        obs = sim.step(P)
        tip.append(obs["tip_pos"].copy())
        t_hist.append(obs["time"])
        P_hist.append(P.copy())
        if video and k % 3 == 0:
            frames.append(sim.render_frame())

    tip = np.array(tip)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_traj, ax_inp = axes
    ax_traj.plot(tip[:, 0] * 100, tip[:, 1] * 100, lw=1.5, color="tab:orange")
    r_max = np.max(np.hypot(tip[:, 0], tip[:, 1])) * 100
    for i, a in enumerate(vertex_angles):
        ax_traj.plot(r_max * np.cos(a), r_max * np.sin(a),
                     "o", ms=9, color="tab:red", zorder=5)
        ax_traj.annotate(f"V{i}", xy=(r_max * np.cos(a), r_max * np.sin(a)),
                         xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax_traj.set_xlabel("tip x [cm]"); ax_traj.set_ylabel("tip y [cm]")
    ax_traj.set_title("Tip trajectory — triangle")
    ax_traj.axis("equal"); ax_traj.grid(alpha=0.3)
    _plot_inputs(ax_inp, t_hist, P_hist, "Pressure commands (Seg 1-4)")
    fig.suptitle("Triangle Trajectory — Linear Cartesian Interpolation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/tip_triangle.png", dpi=130)
    print(f"saved {OUT}/tip_triangle.png")

    if video and frames:
        imageio.mimsave(f"{OUT}/triangle_demo.mp4", frames, fps=33, quality=8)
        print(f"saved {OUT}/triangle_demo.mp4 ({len(frames)} frames)")

    sim.save_pressure_log(f"{OUT}/pressure_triangle.csv")


# ──────────────────────────────────────────────── 6. PRC sensor check ───────
def prc_sensor_check():
    """Band-limited noise drive; confirm sensors carry body-state information."""
    sim = SoftArmSim()
    sim.set_pre_inflation(1.2)
    rng = np.random.default_rng(1)
    u = np.zeros((4, 5))
    resid = []
    for _ in range(int(10 * HZ)):
        u = 0.98 * u + 0.5 * rng.normal(0, 1, (4, 5))
        obs = sim.step(np.clip(1.5 + 0.7 * u, 0, None))
        resid.append((obs["pouch_pressures"] - obs["p_actual"]).ravel())
    resid = np.array(resid)
    print(f"pouch sensor body-state signal: std per channel "
          f"{resid.std(axis=0).min():.2f}-{resid.std(axis=0).max():.2f} psi "
          f"(nonzero -> reservoir carries body state)")


# ──────────────────────────────────────────────────────────── main ───────────
if __name__ == "__main__":
    step_response()
    segment_local()
    single_segment_axial()
    circle_demo()
    triangle_demo()
    prc_sensor_check()
