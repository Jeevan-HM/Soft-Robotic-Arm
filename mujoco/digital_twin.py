"""
digital_twin.py — MuJoCo-rendered live digital twin of the soft robotic arm.

Receives the three OptiTrack rigid bodies from the ZMQ publisher and drives
the MuJoCo arm model in real time using constant-curvature inverse kinematics
derived from the measured tip displacement.

Rigid body roles
----------------
  RB 001 — plywood mount plate   (world reference)
  RB 002 — arm base              (bottom of the mount disc)
  RB 003 — end-effector / tip    (OptiTrack marker cross at the tip)

Inverse kinematics (constant-curvature)
---------------------------------------
  Given  d = p_tip - p_base  in world frame:
    lateral_r = ‖ [dx, dy] ‖
    φ         = atan2(dy, dx)       bending azimuth
    θ_total   = 2 * arcsin(r / L)  total arc angle (CC model)

  Per-level (5 levels):
    bx_k =  (θ/5) * sin(φ)         rotation about X  → Y displacement
    by_k = -(θ/5) * cos(φ)         rotation about Y  → X displacement
    ext_k = (|d| - L_rest) / 5     axial extension

  Arm length L is auto-calibrated from the first N mocap samples.

Dashboard layout
----------------
  ┌────────────────────┬──────────────────────────────────┐
  │                    │  XY workspace  (arm frame)       │
  │  MuJoCo render     ├──────────────────────────────────┤
  │  (live arm pose)   │  X / Y tip displacement vs time  │
  │                    ├──────────────────────────────────┤
  │                    │  arm length + Z vs time          │
  └────────────────────┴──────────────────────────────────┘

Usage
-----
    python digital_twin.py
    python digital_twin.py --publisher 100.124.65.8
    python digital_twin.py --history 20 --cam-az 160 --cam-el -15
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import threading
import time
from typing import Deque, Dict, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
import zmq

# MuJoCo (offscreen render — macOS uses CGL natively, no env override needed)
import mujoco

# Local sim modules
sys.path.insert(0, os.path.dirname(__file__))
from arm_model import ArmConfig, build_arm_xml

# ──────────────────────────────────────────────── config ─────────────────────
DEFAULT_PUBLISHER_IP = "100.124.65.8"   # rise-testbed-laptop Tailscale IP
DEFAULT_PORT         = 5556

HISTORY_S        = 10.0    # s — trail / time-series window
CALIB_SAMPLES    = 30      # number of still frames used to auto-measure arm length
ARM_LENGTH_NOM   = 0.220   # m — nominal arm length (fallback if calib fails)

# Camera defaults
CAM_AZIMUTH   = 140.0
CAM_ELEVATION = -18.0
CAM_DISTANCE  = 0.80
RENDER_W, RENDER_H = 720, 560

# Visual theme
BG_DARK  = "#0d1117"
BG_PANEL = "#161b22"
GRID_COL = "#21262d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"

C_BASE   = "#FFE66D"
C_TIP    = "#FF6B6B"
C_ARM    = "#80ED99"
C_TRAIL  = "#C77DFF"


# ──────────────────────────────────────────────── coordinate transforms ───────
def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[qx, qy, qz, qw] → 3×3 rotation matrix."""
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])

def opti_to_mj_pos(p: np.ndarray) -> np.ndarray:
    """OptiTrack Y-up to MuJoCo Z-up: X_mj = X_o, Y_mj = -Z_o, Z_mj = Y_o"""
    return np.array([p[0], -p[2], p[1]])

def opti_to_mj_rot(q: np.ndarray) -> np.ndarray:
    """Convert OptiTrack quaternion to a MuJoCo world-frame rotation matrix."""
    R_o = quat_to_rot(q)
    T = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])
    return T @ R_o @ T.T

# ──────────────────────────────────────────────── ZMQ receiver ───────────────
class MocapReceiver:
    def __init__(self, publisher_ip: str, port: int):
        self.address  = f"tcp://{publisher_ip}:{port}"
        self._lock    = threading.Lock()
        self._latest: Dict[int, dict] = {}
        self._msgs    = 0
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get(self) -> Dict[int, dict]:
        with self._lock:
            return dict(self._latest)

    @property
    def msgs(self) -> int:
        return self._msgs

    def _loop(self):
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self.address)
        print(f"[twin] Subscribed to {self.address}")
        while self._running:
            try:
                msg = json.loads(sock.recv_string())
                rb_id = int(msg["id"])
                with self._lock:
                    self._latest[rb_id] = {
                        "pos":  np.array(msg["position"],   dtype=float),
                        "quat": np.array(msg["quaternion"], dtype=float),
                        "name": msg.get("name") or f"RB {rb_id}",
                    }
                self._msgs += 1
            except zmq.Again:
                continue
            except Exception:
                pass
        sock.close()
        ctx.term()


# ──────────────────────────────────────────────── CSV recorder ────────────────
class _Recorder:
    """Thread-safe CSV writer for live mocap data.

    Columns: t, tip_x, tip_y, tip_z, arm_len,
             disp_x, disp_y, disp_z   (tip − base in arm frame)
    """
    HEADER = ["t", "tip_x", "tip_y", "tip_z",
              "arm_len", "disp_x", "disp_y", "disp_z"]

    def __init__(self, path: str):
        self._path = path
        self._fh   = open(path, "w", newline="")
        self._w    = csv.writer(self._fh)
        self._w.writerow(self.HEADER)
        self._t0   = time.monotonic()
        self._n    = 0
        print(f"[twin] Recording to {path}")

    def write(self, tip_world: np.ndarray,
              arm_len: float, disp: np.ndarray) -> None:
        t = time.monotonic() - self._t0
        self._w.writerow([
            f"{t:.4f}",
            f"{tip_world[0]:.5f}", f"{tip_world[1]:.5f}", f"{tip_world[2]:.5f}",
            f"{arm_len:.5f}",
            f"{disp[0]:.5f}", f"{disp[1]:.5f}", f"{disp[2]:.5f}",
        ])
        self._n += 1

    def close(self):
        self._fh.flush()
        self._fh.close()
        print(f"[twin] Saved {self._n} rows → {self._path}")


# ──────────────────────────────────────────────── MuJoCo twin ─────────────────
class ArmTwin:
    """
    Holds the MuJoCo model, applies constant-curvature IK from mocap, renders.
    """

    def __init__(self,
                 arm_length: float,
                 cam_az: float, cam_el: float, cam_dist: float,
                 base_cfg: Optional[ArmConfig] = None):
        self.arm_length = arm_length
        # Use identified params if provided, but always apply the measured arm length
        if base_cfg is not None:
            import dataclasses
            cfg = dataclasses.replace(base_cfg, length=arm_length)
        else:
            cfg = ArmConfig(length=arm_length)
        xml = build_arm_xml(cfg)
        self.model  = mujoco.MjModel.from_xml_string(xml)
        self.data   = mujoco.MjData(self.model)
        self.cfg    = cfg
        self._renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)

        # DOF address lookup
        def dof(name: str) -> int:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_dofadr[j]

        self.ext_dofs  = [dof(f"ext{k}") for k in range(cfg.n_pouches)]
        self.bx_dofs   = [dof(f"bx{k}")  for k in range(cfg.n_pouches)]
        self.by_dofs   = [dof(f"by{k}")  for k in range(cfg.n_pouches)]

        # Camera
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth   = cam_az
        self.cam.elevation = cam_el
        self.cam.distance  = cam_dist
        mount_z = 0.5  # mount body is fixed at z=0.5 in the MuJoCo world
        self.cam.lookat[:] = [0.0, 0.0, mount_z - cfg.length * 0.5]

        mujoco.mj_forward(self.model, self.data)

    def apply_ik(self, disp: np.ndarray, arm_length_meas: float):
        """
        Constant-curvature IK:  set qpos from arm-frame displacement vector.
        Coordinate conventions (arm hangs along -Z in MuJoCo world frame):
          bx_k > 0  → rotates about X  → tip moves +Y
          by_k > 0  → rotates about Y  → tip moves -X  (right-hand rule)
          ext_k > 0 → extends downward
        """
        dx, dy = disp[0], disp[1]
        r   = float(np.hypot(dx, dy)) # lateral displacement magnitude
        phi = float(np.arctan2(dy, dx))  # bending azimuth

        L = self.arm_length

        # Total bending angle from CC model: r = L * sin(θ/2) * 2 / θ
        # Numerically stable small-angle: θ ≈ 2r/L
        if r < 1e-4:
            theta = 0.0
        else:
            # Newton solve: f(θ) = 2L sin(θ/2) / θ - r = 0  →  use arcsin approx
            # For moderate bends, θ = 2*arcsin(r/L) is exact for arc chord
            arg = np.clip(r / L, -1.0, 1.0)
            theta = 2.0 * float(np.arcsin(arg))

        # Extension: measured arm length vs rest length
        total_ext = arm_length_meas - L
        per_level_ext = np.clip(total_ext / self.cfg.n_pouches, -0.005, 0.030)

        # Distribute bending uniformly across 5 levels
        per_level_theta = theta / self.cfg.n_pouches

        qpos = self.data.qpos
        for k in range(self.cfg.n_pouches):
            qpos[self.ext_dofs[k]] = per_level_ext
            qpos[self.bx_dofs[k]] =  per_level_theta * np.sin(phi)
            qpos[self.by_dofs[k]] = -per_level_theta * np.cos(phi)

        mujoco.mj_forward(self.model, self.data)

    def render(self) -> np.ndarray:
        """Render and return (H, W, 3) uint8 RGB frame."""
        self._renderer.update_scene(self.data, camera=self.cam)
        return self._renderer.render()




# ──────────────────────────────────────────────── live dashboard ─────────────
class DigitalTwin:

    def __init__(self, rx: MocapReceiver,
                 cam_az: float, cam_el: float, cam_dist: float,
                 recorder: Optional["_Recorder"] = None,
                 base_cfg: Optional[ArmConfig] = None):
        self.rx       = rx
        self.recorder = recorder
        self.base_cfg = base_cfg          # identified params (or None for defaults)
        self._arm_length: Optional[float] = None   # set on first frame
        self.twin: Optional[ArmTwin] = None
        self.cam_az = cam_az
        self.cam_el = cam_el
        self.cam_dist = cam_dist

        # Ring-buffer: (t, dx, dy, dz, arm_length)
        maxlen = int(HISTORY_S * 120)
        self._trail: Deque[Tuple] = collections.deque(maxlen=maxlen)

        # ── figure ────────────────────────────────────────────────────────────
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(17, 9), facecolor=BG_DARK)
        self.fig.canvas.manager.set_window_title("Soft Arm — Digital Twin")

        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            width_ratios=[1.2, 1],
            hspace=0.45, wspace=0.30,
            left=0.05, right=0.97, top=0.93, bottom=0.08,
        )

        # MuJoCo render — left, full height
        self.ax_mj = self.fig.add_subplot(gs[:, 0])
        self.ax_mj.set_facecolor(BG_DARK)
        self.ax_mj.set_axis_off()
        self.ax_mj.set_title("MuJoCo Model — Live Pose", color=TEXT_COL,
                              fontsize=10, pad=6)
        self._mj_im = self.ax_mj.imshow(
            np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8),
            aspect="auto", interpolation="bilinear",
        )
        self._calib_txt = self.ax_mj.text(
            0.5, 0.5, "Calibrating arm length…",
            color=TEXT_COL, fontsize=13, ha="center", va="center",
            transform=self.ax_mj.transAxes,
        )

        # XY workspace — top right
        self.ax_xy = self.fig.add_subplot(gs[0, 1])
        self._init_xy()

        # X/Y vs time — middle right
        self.ax_xy_t = self.fig.add_subplot(gs[1, 1])
        self._init_ts(self.ax_xy_t, "Tip X / Y displacement  [m]")

        # Z + arm length vs time — bottom right
        self.ax_z_t = self.fig.add_subplot(gs[2, 1])
        self._init_ts(self.ax_z_t, "Arm length + Z disp  [m]")

        self.fig.suptitle(
            "Soft Robotic Arm  ·  MuJoCo Digital Twin  ·  Live OptiTrack",
            color=TEXT_COL, fontsize=13, fontweight="bold", y=0.97,
        )

        # ── 2D artists ────────────────────────────────────────────────────────
        # XY workspace
        self._xy_trail, = self.ax_xy.plot([], [], "-",  color=C_TRAIL, lw=1.0, alpha=0.7)
        self._xy_tip,   = self.ax_xy.plot([], [], "o",  color=C_TIP,   ms=11,  zorder=5)
        self._xy_origin,= self.ax_xy.plot([0], [0], "+", color=C_BASE, ms=14,  mew=2.5)
        theta_c = np.linspace(0, 2*np.pi, 200)
        max_r   = ARM_LENGTH_NOM * np.sin(np.deg2rad(55))
        self._reach_circle, = self.ax_xy.plot(
            max_r * np.cos(theta_c), max_r * np.sin(theta_c),
            "--", color=MUTED, lw=0.8, alpha=0.45,
        )

        # X / Y vs time
        self._ts_x, = self.ax_xy_t.plot([], [], "-", color="#FF6B6B", lw=1.4, label="X")
        self._ts_y, = self.ax_xy_t.plot([], [], "-", color="#4ECDC4", lw=1.4, label="Y")
        self.ax_xy_t.legend(loc="upper left", fontsize=8,
                             facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL)

        # Z + arm length vs time
        self._ts_z,   = self.ax_z_t.plot([], [], "-",  color="#FFE66D", lw=1.4, label="Z")
        self._ts_len, = self.ax_z_t.plot([], [], "--", color=C_ARM,     lw=1.4, label="|tip−base|")
        self.ax_z_t.legend(loc="upper left", fontsize=8,
                            facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL)

        # Status bar
        self._status = self.fig.text(
            0.50, 0.004, "Waiting for data…",
            color=MUTED, fontsize=8.5, ha="center",
        )

    # ── axis init ─────────────────────────────────────────────────────────────
    def _init_xy(self):
        ax = self.ax_xy
        ax.set_facecolor(BG_PANEL)
        ax.set_aspect("equal")
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel("X  [m]  (arm frame)", color=MUTED, fontsize=8)
        ax.set_ylabel("Y  [m]  (arm frame)", color=MUTED, fontsize=8)
        ax.set_title("Tip XY workspace  (top-down)", color=TEXT_COL, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5)
        ax.axvline(0, color=MUTED, lw=0.5, alpha=0.5)
        lim = ARM_LENGTH_NOM * 0.7
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    def _init_ts(self, ax, title: str):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_ylabel(title, color=MUTED, fontsize=7.5)
        ax.set_xlabel("time  [s]", color=MUTED, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.4)
        ax.set_xlim(-HISTORY_S, 0)

    # ── animation ─────────────────────────────────────────────────────────────
    def update(self, _frame):
        bodies_dict = self.rx.get()
        if len(bodies_dict) < 3:
            n = self.rx.msgs
            self._status.set_text(
                "Waiting for all 3 rigid bodies…" if n > 0
                else "Connecting to publisher…"
            )
            return

        # Extract all bodies and convert to MuJoCo Z-up frame
        bodies = list(bodies_dict.values())
        for b in bodies:
            b["mj_pos"] = opti_to_mj_pos(b["pos"])
            
        # Dynamically assign roles based on height (Z axis in MuJoCo frame)
        # The arm hangs downwards, so: highest = mount, middle = base, lowest = tip
        bodies.sort(key=lambda b: b["mj_pos"][2], reverse=True)
        mount, base, tip = bodies[0], bodies[1], bodies[2]

        p_mount = mount["mj_pos"]
        p_base  = base["mj_pos"]
        p_tip   = tip["mj_pos"]

        # ── arm-frame coordinates (mount = origin, mount orientation = axes) ──
        R_mount = opti_to_mj_rot(mount["quat"])
        af_base = R_mount.T @ (p_base  - p_mount)
        af_tip  = R_mount.T @ (p_tip   - p_mount)
        disp    = af_tip - af_base            # tip relative to base, arm frame
        arm_len = float(np.linalg.norm(p_tip - p_base))

        # ── measure arm length on first frame, build twin ─────────────────────
        if self._arm_length is None:
            self._arm_length = float(np.linalg.norm(p_tip - p_base))
            print(f"[twin] Arm length from RB2→RB3: {self._arm_length*100:.1f} cm")

        if self.twin is None:
            L = self._arm_length
            self.twin = ArmTwin(L, self.cam_az, self.cam_el, self.cam_dist,
                                base_cfg=self.base_cfg)
            max_r   = L * np.sin(np.deg2rad(55))
            theta_c = np.linspace(0, 2*np.pi, 200)
            self._reach_circle.set_data(max_r * np.cos(theta_c),
                                        max_r * np.sin(theta_c))
            lim = L * 0.75
            self.ax_xy.set_xlim(-lim, lim)
            self.ax_xy.set_ylim(-lim, lim)
            self._calib_txt.set_text("")
            print(f"[twin] MuJoCo model built  (L={L*100:.1f} cm)")

        # IK → render
        self.twin.apply_ik(disp, arm_len)
        frame = self.twin.render()
        self._mj_im.set_data(frame)

        # ── recorder ─────────────────────────────────────────────────────────────
        if self.recorder is not None:
            self.recorder.write(p_tip, arm_len, disp)

        # ── trail ─────────────────────────────────────────────────────────────
        t_now = time.monotonic()
        self._trail.append((t_now, disp[0], disp[1], disp[2], arm_len))
        trail = np.array(self._trail)
        t_rel = trail[:, 0] - t_now            # seconds ago (negative)
        mask  = t_rel >= -HISTORY_S

        # XY workspace
        if mask.sum() > 1:
            self._xy_trail.set_data(trail[mask, 1], trail[mask, 2])
        self._xy_tip.set_data([disp[0]], [disp[1]])

        r_now = float(np.hypot(disp[0], disp[1]))
        lim   = max(r_now * 1.6, self._arm_length * 0.35, 0.04)
        self.ax_xy.set_xlim(-lim, lim)
        self.ax_xy.set_ylim(-lim, lim)

        # X / Y vs time
        if mask.sum() > 1:
            tr = trail[mask]
            tp = tr[:, 0] - t_now
            self._ts_x.set_data(tp, tr[:, 1])
            self._ts_y.set_data(tp, tr[:, 2])
            self._ts_z.set_data(tp, tr[:, 3])
            self._ts_len.set_data(tp, tr[:, 4])
            self.ax_xy_t.set_xlim(-HISTORY_S, 0)
            self.ax_z_t.set_xlim(-HISTORY_S, 0)

            xy_vals = np.concatenate([tr[:, 1], tr[:, 2]])
            pad_xy  = max(0.005, np.ptp(xy_vals) * 0.2)
            self.ax_xy_t.set_ylim(xy_vals.min() - pad_xy, xy_vals.max() + pad_xy)

            z_vals  = np.concatenate([tr[:, 3], tr[:, 4]])
            pad_z   = max(0.005, np.ptp(z_vals) * 0.2)
            self.ax_z_t.set_ylim(z_vals.min() - pad_z, z_vals.max() + pad_z)

        # ── status bar ────────────────────────────────────────────────────────
        dx, dy, dz = disp
        self._status.set_text(
            f"tip disp from base:  X={dx:+.3f} m   Y={dy:+.3f} m   Z={dz:+.3f} m  ·  "
            f"arm length {arm_len*100:.1f} cm  "
            f"(measured {self._arm_length*100:.1f} cm)  ·  "
            f"{self.rx.msgs} msgs"
        )

    def run(self):
        self._anim = FuncAnimation(
            self.fig, self.update,
            interval=33,
            blit=False, cache_frame_data=False,
        )
        plt.show()


# ──────────────────────────────────────────────── entry point ─────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo digital twin of the soft arm (ZMQ mocap receiver)"
    )
    parser.add_argument("--publisher", default=DEFAULT_PUBLISHER_IP,
                        help=f"Testbed Tailscale IP (default: {DEFAULT_PUBLISHER_IP})")
    parser.add_argument("--port",      type=int, default=DEFAULT_PORT,
                        help=f"ZMQ PUB port (default: {DEFAULT_PORT})")
    parser.add_argument("--history",   type=float, default=10.0,
                        help="Trail window in seconds (default: 10)")
    parser.add_argument("--cam-az",    type=float, default=CAM_AZIMUTH,
                        help=f"Camera azimuth deg (default: {CAM_AZIMUTH})")
    parser.add_argument("--cam-el",    type=float, default=CAM_ELEVATION,
                        help=f"Camera elevation deg (default: {CAM_ELEVATION})")
    parser.add_argument("--cam-dist",  type=float, default=CAM_DISTANCE,
                        help=f"Camera distance m (default: {CAM_DISTANCE})")
    parser.add_argument("--arm-length", type=float, default=None,
                        help="Skip calibration, use this arm length [m]")
    parser.add_argument("--params", default=None, metavar="JSON",
                        help="JSON file from sysid.py (identified_params.json). "
                             "If not specified, auto-detected in the working directory.")
    parser.add_argument("--demo", action="store_true",
                        help="Skip mocap calibration and use the arm length from "
                             "identified_params.json (or ArmConfig default if no JSON).")
    parser.add_argument("--record", default=None, metavar="CSV",
                        help="Stream tip mocap data to this CSV file while running")
    args = parser.parse_args()

    global HISTORY_S
    HISTORY_S = args.history

    print(f"[twin] Publisher : tcp://{args.publisher}:{args.port}")
    print(f"[twin] RB1=mount  RB2=base  RB3=tip")
    print(f"[twin] History   : {HISTORY_S:.0f} s")

    rx = MocapReceiver(publisher_ip=args.publisher, port=args.port)
    rx.start()

    # Load identified physics params (auto-detect if not specified)
    import os
    params_path = args.params
    if params_path is None and os.path.exists("identified_params.json"):
        params_path = "identified_params.json"
    if params_path is not None:
        base_cfg = ArmConfig.from_json(params_path)
        print(f"[twin] Physics params: {params_path}")
        print(f"[twin]   base_stiffness={base_cfg.base_stiffness:.4f}  "
              f"base_damping={base_cfg.base_damping:.4f}  "
              f"pressure_gain={base_cfg.pressure_gain:.4f}")
    else:
        base_cfg = None
        print("[twin] Physics params: default (no identified_params.json found)")

    recorder = _Recorder(args.record) if args.record else None

    twin = DigitalTwin(rx, cam_az=args.cam_az,
                       cam_el=args.cam_el, cam_dist=args.cam_dist,
                       recorder=recorder, base_cfg=base_cfg)

    # Arm length: explicit > --demo (from params/default) > mocap calibration
    if args.arm_length is not None:
        twin._arm_length = args.arm_length
        print(f"[twin] Arm length : {args.arm_length*100:.1f} cm  (explicit --arm-length)")
    elif args.demo:
        demo_length = base_cfg.length if base_cfg is not None else ArmConfig().length
        twin._arm_length = demo_length
        print(f"[twin] Arm length : {demo_length*100:.1f} cm  (--demo, from "
              + (params_path if params_path else "ArmConfig default") + ")")

    try:
        twin.run()
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        if recorder is not None:
            recorder.close()
        print(f"[twin] Stopped. Total messages received: {rx.msgs}")


if __name__ == "__main__":
    main()
