"""
mocap.py — ZMQ subscriber + live plotter for the OptiTrack publisher.

Connects to the ZMQ PUB socket on the testbed laptop (mocap_publisher.py)
and plots every rigid body's position in real time.

Publisher wire format (JSON string per message)
------------------------------------------------
{
  "id":             int,           # Motive streaming ID
  "name":           str | null,    # Motive display name, e.g. "Rigid Body 001"
  "position":       [x, y, z],     # metres, OptiTrack Y-up frame
  "quaternion":     [qx, qy, qz, qw],
  "t_testbed_recv": float,         # wall-clock epoch on testbed when Motive delivered it
  "t_publish":      float          # wall-clock epoch just before ZMQ send
}

Usage
-----
    python mocap.py                            # connect to 100.70.122.84:5556
    python mocap.py --publisher 100.70.122.84  # explicit publisher Tailscale IP
    python mocap.py --publisher 100.70.122.84 --port 5556
    python mocap.py --history 30               # longer trail
"""

from __future__ import annotations

import argparse
import collections
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
import zmq

# ──────────────────────────────────────────────── constants ──────────────────
DEFAULT_PUBLISHER_IP = "100.124.65.8"   # rise-testbed-laptop (ZMQ publisher)
DEFAULT_PORT         = 5556
HISTORY_S            = 10.0      # seconds of trail kept in ring-buffer

COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8DADC",
    "#F77F00", "#80ED99", "#C77DFF", "#FF9EF5",
    "#06D6A0", "#EF476F", "#118AB2", "#FFD166",
    "#CBFF8C", "#FF6FA8", "#6E44FF", "#B5EAD7",
]


# ──────────────────────────────────────────────── data model ─────────────────
@dataclass
class RigidBody:
    rb_id:    int
    name:     str        = ""
    pos:      np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat:     np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    latency:  float      = 0.0    # seconds from t_publish to local receipt


# ──────────────────────────────────────────────── receiver thread ─────────────
class MocapReceiver:
    """
    Subscribes to the ZMQ PUB socket published by mocap_publisher.py and
    fills per-body ring-buffers with (t, x, y, z) tuples.
    """

    def __init__(self, publisher_ip: str, port: int = DEFAULT_PORT,
                 verbose: bool = False):
        self.address  = f"tcp://{publisher_ip}:{port}"
        self.verbose  = verbose

        self._lock    = threading.Lock()
        self._bufs:   Dict[int, collections.deque] = {}   # rb_id -> deque[(t,x,y,z)]
        self._latest: Dict[int, RigidBody]         = {}
        self._msgs_recv = 0
        self._running = False

    # ── public API ────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        t = threading.Thread(target=self._recv_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def snapshot(self) -> Dict[int, RigidBody]:
        with self._lock:
            return dict(self._latest)

    def history(self, rb_id: int) -> np.ndarray:
        """Return (N, 4) array of [t, x, y, z] for rb_id."""
        with self._lock:
            buf = self._bufs.get(rb_id)
            if not buf:
                return np.empty((0, 4))
            return np.array(buf)

    @property
    def msgs_recv(self) -> int:
        return self._msgs_recv

    # ── ZMQ loop ──────────────────────────────────────────────────────────────
    def _recv_loop(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 500)      # ms — lets us check _running
        sock.setsockopt_string(zmq.SUBSCRIBE, "") # subscribe to all messages
        sock.connect(self.address)
        print(f"[mocap] Subscribed to {self.address}")

        while self._running:
            try:
                raw = sock.recv_string()
            except zmq.Again:
                continue    # timeout, check _running
            except zmq.ZMQError:
                break

            try:
                msg = json.loads(raw)
                t_local = time.time()
                rb_id   = int(msg["id"])
                pos     = np.array(msg["position"], dtype=float)
                quat    = np.array(msg["quaternion"], dtype=float)
                name    = msg.get("name") or f"RB {rb_id}"
                latency = t_local - msg.get("t_publish", t_local)

                rb = RigidBody(rb_id=rb_id, name=name, pos=pos,
                               quat=quat, latency=latency)

                t_mono = time.monotonic()
                with self._lock:
                    self._latest[rb_id] = rb
                    if rb_id not in self._bufs:
                        self._bufs[rb_id] = collections.deque()
                    buf = self._bufs[rb_id]
                    buf.append([t_mono, pos[0], pos[1], pos[2]])
                    # trim old samples
                    while buf and (t_mono - buf[0][0]) > HISTORY_S:
                        buf.popleft()

                self._msgs_recv += 1
                if self.verbose or self._msgs_recv <= 3:
                    print(f"[mocap] #{self._msgs_recv}  {name}  "
                          f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
                          f"latency={latency*1000:.1f} ms")

            except Exception as e:
                if self.verbose:
                    print(f"[mocap] parse error: {e}")

        sock.close()
        ctx.term()


# ──────────────────────────────────────────────── live plot ───────────────────
class LivePlot:
    """Dark-themed real-time dashboard: 3D trajectory + X/Y/Z time-series."""

    def __init__(self, receiver: MocapReceiver, publisher_ip: str):
        self.rx = receiver

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9), facecolor="#0d1117")
        self.fig.canvas.manager.set_window_title("OptiTrack — Live Mocap")

        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            hspace=0.45, wspace=0.35,
            left=0.07, right=0.97, top=0.93, bottom=0.08,
        )

        # 3D trajectory — left panel, full height
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self._style_3d(self.ax3d)

        # X / Y / Z time-series — right column
        self.ax_x = self.fig.add_subplot(gs[0, 1])
        self.ax_y = self.fig.add_subplot(gs[1, 1])
        self.ax_z = self.fig.add_subplot(gs[2, 1])
        for ax, label in [(self.ax_x, "X  [m]"),
                          (self.ax_y, "Y  [m]"),
                          (self.ax_z, "Z  [m]")]:
            self._style_ts(ax, label)

        self.fig.suptitle(
            f"OptiTrack  ·  publisher @ {publisher_ip}",
            color="#e6edf3", fontsize=13, fontweight="bold", y=0.97,
        )

        self._trail3d:  Dict[int, object] = {}
        self._head3d:   Dict[int, object] = {}
        self._lines_x:  Dict[int, object] = {}
        self._lines_y:  Dict[int, object] = {}
        self._lines_z:  Dict[int, object] = {}
        self._labels:   Dict[int, object] = {}
        self._body_colors: Dict[int, str] = {}
        self._color_idx = 0

        self._status = self.fig.text(
            0.50, 0.005, "Connecting to publisher…",
            color="#8b949e", fontsize=9, ha="center",
        )

    # ── axis styling ──────────────────────────────────────────────────────────
    def _style_3d(self, ax):
        ax.set_facecolor("#0d1117")
        ax.grid(True, color="#21262d", linewidth=0.5)
        for label, setter in [("X [m]", ax.set_xlabel),
                               ("Y [m]", ax.set_ylabel),
                               ("Z [m]", ax.set_zlabel)]:
            setter(label, color="#8b949e", fontsize=8, labelpad=6)
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.set_title("3D Trajectory", color="#e6edf3", fontsize=10, pad=8)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#21262d")

    def _style_ts(self, ax, ylabel):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=8)
        ax.set_xlabel("time [s]", color="#8b949e", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.5, alpha=0.7)

    # ── per-body artist factory ───────────────────────────────────────────────
    def _color(self, rb_id: int) -> str:
        if rb_id not in self._body_colors:
            self._body_colors[rb_id] = COLORS[self._color_idx % len(COLORS)]
            self._color_idx += 1
        return self._body_colors[rb_id]

    def _ensure_artists(self, rb_id: int, name: str):
        if rb_id in self._trail3d:
            return
        c = self._color(rb_id)
        trail, = self.ax3d.plot([], [], [], "-",  color=c, lw=1.2, alpha=0.7)
        head,  = self.ax3d.plot([], [], [], "o",  color=c, ms=8,   zorder=5)
        self._trail3d[rb_id] = trail
        self._head3d[rb_id]  = head
        self._labels[rb_id]  = self.ax3d.text(
            0, 0, 0, name, color=c, fontsize=7, zorder=6,
        )
        lx, = self.ax_x.plot([], [], "-", color=c, lw=1.2, label=name)
        ly, = self.ax_y.plot([], [], "-", color=c, lw=1.2)
        lz, = self.ax_z.plot([], [], "-", color=c, lw=1.2)
        self._lines_x[rb_id] = lx
        self._lines_y[rb_id] = ly
        self._lines_z[rb_id] = lz
        self.ax_x.legend(
            loc="upper left", fontsize=7,
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
        )

    # ── animation callback ────────────────────────────────────────────────────
    def update(self, _frame):
        snap = self.rx.snapshot()
        now  = time.monotonic()

        if not snap:
            msgs = self.rx.msgs_recv
            if msgs == 0:
                self._status.set_text("Waiting for publisher…")
            return

        all_pos = []
        for rb_id, rb in snap.items():
            self._ensure_artists(rb_id, rb.name)
            hist = self.rx.history(rb_id)   # (N, 4)
            if len(hist) == 0:
                continue

            t_rel = hist[:, 0] - now      # seconds ago (negative)
            xs, ys, zs = hist[:, 1], hist[:, 2], hist[:, 3]

            # 3D trail + current head
            self._trail3d[rb_id].set_data(xs, ys)
            self._trail3d[rb_id].set_3d_properties(zs)
            self._head3d[rb_id].set_data([xs[-1]], [ys[-1]])
            self._head3d[rb_id].set_3d_properties([zs[-1]])
            self._labels[rb_id].set_position((xs[-1], ys[-1]))
            self._labels[rb_id].set_3d_properties(zs[-1], zdir="z")

            # time-series panels
            self._lines_x[rb_id].set_data(t_rel, xs)
            self._lines_y[rb_id].set_data(t_rel, ys)
            self._lines_z[rb_id].set_data(t_rel, zs)

            all_pos.append(rb.pos)

        # ── rescale ──────────────────────────────────────────────────────────
        if all_pos:
            pts = np.vstack(all_pos)
            mn, mx = pts.min(0), pts.max(0)
            pad = np.maximum(0.05, (mx - mn) * 0.3)
            for ax, lo, hi in [(self.ax_x, mn[0]-pad[0], mx[0]+pad[0]),
                               (self.ax_y, mn[1]-pad[1], mx[1]+pad[1]),
                               (self.ax_z, mn[2]-pad[2], mx[2]+pad[2])]:
                ax.set_ylim(lo, hi)
            for ax in (self.ax_x, self.ax_y, self.ax_z):
                ax.set_xlim(-HISTORY_S, 0)

            all_hists = [self.rx.history(i) for i in snap
                         if len(self.rx.history(i)) > 0]
            if all_hists:
                all_h = np.vstack([h[:, 1:4] for h in all_hists])
                mn3, mx3 = all_h.min(0), all_h.max(0)
                pad3 = np.maximum(0.1, (mx3 - mn3) * 0.2)
                self.ax3d.set_xlim(mn3[0]-pad3[0], mx3[0]+pad3[0])
                self.ax3d.set_ylim(mn3[1]-pad3[1], mx3[1]+pad3[1])
                self.ax3d.set_zlim(mn3[2]-pad3[2], mx3[2]+pad3[2])

        # ── status bar ───────────────────────────────────────────────────────
        n = len(snap)
        names = ", ".join(rb.name for rb in snap.values())
        latencies = [rb.latency * 1000 for rb in snap.values()]
        avg_lat = np.mean(latencies) if latencies else 0.0
        self._status.set_text(
            f"{n} rigid bod{'y' if n==1 else 'ies'}: {names}  ·  "
            f"{self.rx.msgs_recv} msgs  ·  "
            f"latency {avg_lat:.1f} ms"
        )

    def run(self):
        self._anim = FuncAnimation(
            self.fig, self.update,
            interval=33,   # ~30 Hz display refresh
            blit=False, cache_frame_data=False,
        )
        plt.show()


# ──────────────────────────────────────────────── entry point ─────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Receive ZMQ mocap stream from mocap_publisher.py and plot live"
    )
    parser.add_argument(
        "--publisher", default=DEFAULT_PUBLISHER_IP,
        help=f"Tailscale IP of the testbed laptop running mocap_publisher.py "
             f"(default: {DEFAULT_PUBLISHER_IP})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"ZMQ PUB port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--history", type=float, default=10.0,
        help="Position trail length in seconds (default: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every received message",
    )
    args = parser.parse_args()

    # Override module-level constant so ring-buffer trimming uses the right window
    import mocap as _self
    _self.HISTORY_S = args.history

    print(f"[mocap] Publisher : tcp://{args.publisher}:{args.port}")
    print(f"[mocap] History   : {args.history:.1f} s")

    rx = MocapReceiver(
        publisher_ip=args.publisher,
        port=args.port,
        verbose=args.verbose,
    )
    rx.start()

    plot = LivePlot(rx, publisher_ip=args.publisher)
    try:
        plot.run()
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        print(f"[mocap] Stopped. Total messages received: {rx.msgs_recv}")


if __name__ == "__main__":
    main()
