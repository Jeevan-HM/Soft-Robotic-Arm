"""
hardware_arm.py — Real-arm driver with the same interface as SoftArmSim.

Combines:
  * Regulator     (ZMQ REQ/REP)  -- command per-column pressure on the Pi
                                     regulator server, mirrors pressure_test.py
  * MocapReceiver (ZMQ SUB)      -- read the 3 OptiTrack rigid bodies,
                                     mirrors digital_twin.py

into a class with the same step()/reset()/set_pre_inflation() interface as
SoftArmSim, so collect_sysid_data.py can run its excitation protocol
UNCHANGED against the physical arm and write a CSV in the exact format it
already writes for the simulator. Feed that CSV straight into:

    uv run sysid.py --data <csv> --arm-length <measured> --out identified_params_real.json

*** THINGS TO VERIFY BEFORE YOUR FIRST RUN ***

1. Column -> regulator mapping (REG_FOR_COLUMN below).
   The simulator's column 0/1/2/3 = East/North/West/South (see arm_model.py).
   If pressurising column 0 bends the *simulated* arm toward +X but
   regulator 0 bends the *real* arm a different physical direction,
   sysid.py will fit garbage without any error -- it has no way to know
   the labels disagree. Run a quick manual check first (e.g.
   `uv run pressure_test.py --psi 3 --regs 0` while watching the live
   digital_twin.py dashboard, or just watching the arm) and fix the
   mapping below if needed.

2. Rigid-body role assignment (`_read_pose`).
   Bodies are told apart by height, exactly like digital_twin.py: highest
   = mount, middle = base, lowest = tip. This assumes the rig is mounted
   the same way (arm hanging down) and that the tip never bends so far
   that it ends up higher than the base RB -- true for the pressures used
   in the default excitation protocol (<=9 psi), but re-check if you
   raise --p-max.

3. IP addresses / ports.
   DEFAULT_REG_IP is the regulator server (Pi). DEFAULT_MOCAP_IP is the
   OptiTrack publisher (testbed laptop). They are two different machines
   -- override with --reg-ip / --mocap-ip if yours differ from the ones
   pressure_test.py / digital_twin.py already use.

Coordinate frame
-----------------
tip_pos is reported as (x, y, MOUNT_Z + z) where (x, y, z) is the tip's
position expressed in the mount rigid body's own frame (origin at the
mount, axes = mount orientation) -- i.e. the same quantity digital_twin.py
calls `af_tip`, just with the fixed MOUNT_Z offset added so it lines up
with SoftArmSim's absolute tip_pos sensor output (mount fixed at z=0.5 in
the MuJoCo model). This lets sysid.py's rollout() compare real vs.
simulated tip_pos directly without any extra transform.

Safety
------
  * Every commanded pressure is clipped to --p-max (default 9.0 psi,
    under the arm's documented 10 psi limit).
  * Regulators are zeroed on connect, and on close()/any exception --
    always wrap a run in try/finally and call sim.close() (collect_sysid_data.py
    --hardware does this for you).
  * This module never sends a pressure command on import. main() below
    (a small standalone smoke test) requires a typed "yes" before doing so.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Dict, Optional

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# Network defaults -- copied from pressure_test.py / digital_twin.py.
# Override on the command line if your rig uses different addresses.
# ---------------------------------------------------------------------------
DEFAULT_REG_IP     = "100.82.152.108"   # raspberrypi-testbed (Tailscale) -- regulator server
DEFAULT_REG_PORT   = 5555               # regulator server REQ/REP
DEFAULT_MOCAP_IP   = "100.124.65.8"     # rise-testbed-laptop (Tailscale) -- OptiTrack publisher
DEFAULT_MOCAP_PORT = 5556               # OptiTrack ZMQ PUB

MOUNT_Z = 0.5   # m -- must match the fixed mount height in arm_model.py / build_arm_xml()


# ---------------------------------------------------------------------------
# Coordinate transforms -- copied from digital_twin.py so this module has no
# matplotlib/mujoco import dependency. Keep these in sync if the rig geometry
# or OptiTrack axis convention changes.
# ---------------------------------------------------------------------------

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[qx, qy, qz, qw] -> 3x3 rotation matrix."""
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])


def opti_to_mj_pos(p: np.ndarray) -> np.ndarray:
    """OptiTrack Y-up to MuJoCo Z-up: X_mj = X_o, Y_mj = -Z_o, Z_mj = Y_o."""
    return np.array([p[0], -p[2], p[1]])


def opti_to_mj_rot(q: np.ndarray) -> np.ndarray:
    """Convert an OptiTrack quaternion to a MuJoCo world-frame rotation matrix."""
    R_o = quat_to_rot(q)
    T = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    return T @ R_o @ T.T


# ---------------------------------------------------------------------------
# Regulator client -- minimal REQ/REP wrapper, mirrors pressure_test.py.
# ---------------------------------------------------------------------------
class Regulator:
    """Thin REQ/REP client. Rebuilds the socket after any failure so a
    timeout can't wedge the strict REQ/REP state machine."""

    def __init__(self, address: str, timeout_ms: int = 3000):
        self.address    = address
        self.timeout_ms = timeout_ms
        self.ctx        = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.address)

    def request(self, msg: dict) -> dict:
        try:
            self.sock.send_json(msg)
            return self.sock.recv_json()
        except Exception as e:                        # noqa: BLE001
            self.sock.close()
            self._connect()
            return {"status": "error", "message": f"comm failure: {e}"}

    def close(self):
        self.sock.close()


# ---------------------------------------------------------------------------
# Mocap receiver -- background SUB thread, mirrors digital_twin.py.
# ---------------------------------------------------------------------------
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
        import json
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self.address)
        print(f"[hw] Subscribed to mocap {self.address}")
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
            except Exception:                          # noqa: BLE001
                continue
        sock.close()
        ctx.term()


# ---------------------------------------------------------------------------
# HardwareArm -- SoftArmSim-compatible driver for the physical robot.
# ---------------------------------------------------------------------------
class HardwareArm:
    """
    Drop-in replacement for SoftArmSim. Implements the subset of the
    interface collect_sysid_data.py actually uses:
        reset()             -> obs dict
        set_pre_inflation(p)
        step(P)              -> obs dict with obs["tip_pos"], obs["time"]
        close()               (not on SoftArmSim -- call manually / via try-finally)
    """

    # column index (sim convention: 0=E,1=N,2=W,3=S) -> regulator index.
    # *** VERIFY against your wiring before the first real run. ***
    REG_FOR_COLUMN = {0: 0, 1: 1, 2: 2, 3: 3}

    def __init__(self, cfg, reg_ip: str = DEFAULT_REG_IP, reg_port: int = DEFAULT_REG_PORT,
                 mocap_ip: str = DEFAULT_MOCAP_IP, mocap_port: int = DEFAULT_MOCAP_PORT,
                 control_hz: float = 100.0, p_max: float = 9.0,
                 mocap_timeout_s: float = 15.0, settle_s: float = 1.0):
        self.cfg = cfg
        self.control_dt = 1.0 / control_hz
        self.p_max = float(min(p_max, cfg.p_max))
        self.p_pre = 0.0
        self.settle_s = settle_s

        self._reg = Regulator(f"tcp://{reg_ip}:{reg_port}")
        self._mocap = MocapReceiver(mocap_ip, mocap_port)
        self._mocap.start()

        print(f"[hw] Waiting for 3 OptiTrack rigid bodies from {mocap_ip}:{mocap_port} ...")
        t0 = time.monotonic()
        while len(self._mocap.get()) < 3:
            if time.monotonic() - t0 > mocap_timeout_s:
                raise RuntimeError(
                    "Timed out waiting for 3 rigid bodies from the mocap publisher. "
                    "Check --mocap-ip/--mocap-port and that OptiTrack streaming is running."
                )
            time.sleep(0.1)
        print("[hw] Mocap connected (3 rigid bodies seen).")

        r = self._reg.request({"cmd": "zero_all"})
        if r.get("status") not in ("ok", "partial"):
            raise RuntimeError(f"Could not reach regulator server at {reg_ip}:{reg_port} -> {r}")
        print("[hw] Regulator server reachable, all regulators zeroed.")

        self._last_cmd: Optional[np.ndarray] = None
        self._t_next: Optional[float] = None
        self._time = 0.0

    # ---- pose -----------------------------------------------------------
    def _read_pose(self):
        """Return (tip_pos[3], arm_len) in the SoftArmSim-compatible frame,
        or None if fewer than 3 rigid bodies are currently visible."""
        bodies_dict = self._mocap.get()
        if len(bodies_dict) < 3:
            return None
        bodies = list(bodies_dict.values())
        for b in bodies:
            b["mj_pos"] = opti_to_mj_pos(b["pos"])
        # Highest = mount (fixed), middle = base (top of arm), lowest = tip.
        bodies.sort(key=lambda b: b["mj_pos"][2], reverse=True)
        mount, base, tip = bodies[0], bodies[1], bodies[2]

        R_mount = opti_to_mj_rot(mount["quat"])
        af_tip  = R_mount.T @ (tip["mj_pos"]  - mount["mj_pos"])
        tip_pos = np.array([af_tip[0], af_tip[1], MOUNT_Z + af_tip[2]])
        arm_len = float(np.linalg.norm(tip["mj_pos"] - base["mj_pos"]))
        return tip_pos, arm_len

    # ---- SoftArmSim-compatible API ---------------------------------------
    def set_pre_inflation(self, p_pre_psi: float) -> None:
        self.p_pre = float(np.clip(p_pre_psi, 0.0, self.p_max))

    def reset(self, clear_log: bool = True) -> dict:
        self._reg.request({"cmd": "zero_all"})
        self._last_cmd = None
        self._t_next = None
        self._time = 0.0
        time.sleep(self.settle_s)   # let the arm settle back to rest
        return self.observe()

    def step(self, p_cmd_psi) -> dict:
        cfg = self.cfg
        p_cmd = np.asarray(p_cmd_psi, dtype=float)

        if p_cmd.ndim == 2:
            p_col = p_cmd.mean(axis=1)          # (n_segments, n_pouches) -> per-column mean
        elif p_cmd.ndim == 1 and p_cmd.size == cfg.n_segments:
            p_col = p_cmd
        else:
            raise ValueError(f"HardwareArm.step: unexpected pressure shape {p_cmd.shape}")

        p_col = np.clip(p_col + self.p_pre, 0.0, self.p_max)

        # Only issue a regulator command when the target actually changes --
        # the excitation protocol holds pressure constant for seconds at a
        # time, so this avoids a ZMQ REQ/REP round trip on every 10 ms tick.
        if self._last_cmd is None or not np.allclose(p_col, self._last_cmd, atol=1e-6):
            values = {str(self.REG_FOR_COLUMN[s]): float(p_col[s]) for s in range(cfg.n_segments)}
            r = self._reg.request({"cmd": "set_many", "values": values})
            if r.get("status") not in ("ok", "partial"):
                print(f"[hw][warn] regulator command failed: {r}")
            self._last_cmd = p_col.copy()

        # Pace this call to control_hz without accumulating drift.
        now = time.monotonic()
        if self._t_next is None:
            self._t_next = now
        self._t_next += self.control_dt
        sleep_for = self._t_next - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._t_next = now      # fell behind -- resync rather than drift further

        self._time += self.control_dt
        return self.observe()

    def observe(self) -> dict:
        pose = self._read_pose()
        if pose is None:
            raise RuntimeError("Lost mocap tracking (fewer than 3 rigid bodies visible).")
        tip_pos, arm_len = pose
        return {"time": self._time, "tip_pos": tip_pos, "arm_len": arm_len}

    def close(self):
        """Zero all regulators and release sockets. Always call this
        (wrap the run in try/finally) -- there is no watchdog on the
        server side, per pressure_test.py."""
        try:
            self._reg.request({"cmd": "zero_all"})
            print("[hw] Regulators zeroed on close.")
        except Exception as e:                         # noqa: BLE001
            print(f"[hw][warn] failed to zero on close: {e}")
        self._reg.close()
        self._mocap.stop()


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Smoke-test HardwareArm: step one column to a low pressure "
                    "and print tip_pos for a few seconds."
    )
    ap.add_argument("--reg-ip",     default=DEFAULT_REG_IP)
    ap.add_argument("--reg-port",   type=int, default=DEFAULT_REG_PORT)
    ap.add_argument("--mocap-ip",   default=DEFAULT_MOCAP_IP)
    ap.add_argument("--mocap-port", type=int, default=DEFAULT_MOCAP_PORT)
    ap.add_argument("--psi",        type=float, default=2.0,
                    help="test pressure, kept low on purpose (default 2.0)")
    ap.add_argument("--col",        type=int, default=0, help="column to step (0-3)")
    ap.add_argument("--seconds",    type=float, default=3.0)
    args = ap.parse_args()

    from arm_model import ArmConfig
    cfg = ArmConfig()

    print(f"About to command {args.psi} psi on column {args.col} for {args.seconds}s")
    print(f"  regulator server : tcp://{args.reg_ip}:{args.reg_port}")
    print(f"  mocap publisher  : tcp://{args.mocap_ip}:{args.mocap_port}")
    if input("Type 'yes' to continue and actuate the real arm: ").strip().lower() != "yes":
        print("Aborted.")
        return

    arm = HardwareArm(cfg, reg_ip=args.reg_ip, reg_port=args.reg_port,
                      mocap_ip=args.mocap_ip, mocap_port=args.mocap_port)
    try:
        obs = arm.reset()
        print("rest tip_pos:", np.round(obs["tip_pos"], 4))
        P = np.zeros((cfg.n_segments, cfg.n_pouches))
        P[args.col, :] = args.psi
        n = int(args.seconds * 100)
        for i in range(n):
            obs = arm.step(P)
            if i % 20 == 0:
                print(f"t={obs['time']:.2f}s  tip_pos={np.round(obs['tip_pos'], 4)}")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
