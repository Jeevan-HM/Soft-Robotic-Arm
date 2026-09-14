"""
pressure_test.py — Command the regulators and live-plot the 8 pressure sensors.

Sends a pressure setpoint to the regulator server (REQ/REP :5555) on the
Raspberry Pi, then subscribes to the pressure monitor (PUB :5556) and plots
the sensor readings in real time until you close the window (or hit Ctrl-C).

  • Top plot:    sensors 1-5   (the first 5 readings)
  • Bottom plot: sensors 6-8   (the next 3 readings)

The script records a short baseline, then steps to the commanded pressure so
you can see the sensors respond (dashed line marks the step). On exit it zeros
all regulators — the server watchdog is disabled, so this is the only thing
that brings pressure back down.

Usage
-----
  uv run pressure_test.py                        # 3 psi on all regulators
  uv run pressure_test.py --psi 5               # 5 psi on all regulators
  uv run pressure_test.py --psi 4 --regs 0,1    # 4 psi on regulators 0 and 1
  uv run pressure_test.py --psi 6 --delay 4     # 4 s baseline, then step to 6
  uv run pressure_test.py --window 40 --resend 2
  uv run pressure_test.py --no-zero-exit        # leave pressure applied on exit
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import zmq

# ──────────────────────────────────────────────────────────── config ─────────
DEFAULT_IP = "100.82.152.108"          # raspberrypi-testbed (Tailscale)
CMD_PORT   = 5555                      # regulator server  (REQ/REP)
MON_PORT   = 5556                      # pressure monitor  (PUB)
TOPIC      = b"pressure"

GROUP_TOP = [1, 2, 3, 4, 5]           # first 5 readings
GROUP_BOT = [6, 7, 8]                 # next 3 readings

COLORS = {
    1: "#FF6B6B", 2: "#FFA94D", 3: "#FFD43B", 4: "#69DB7C", 5: "#38D9A9",
    6: "#4DABF7", 7: "#9775FA", 8: "#F783AC",
}

T0 = time.monotonic()                 # script start — the shared time origin


# ──────────────────────────────────────────────── regulator client (REQ) ─────
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
        except Exception as e:                       # noqa: BLE001
            self.sock.close()
            self._connect()
            return {"status": "error", "message": f"comm failure: {e}"}

    def close(self):
        self.sock.close()


# ──────────────────────────────────────────────── pressure receiver (SUB) ────
class PressureReceiver:
    """Background thread: collects (t, {sensor: psi}) rows from the PUB feed."""

    def __init__(self, address: str, maxlen: int = 20000):
        self.address  = address
        self._lock    = threading.Lock()
        self._hist    = deque(maxlen=maxlen)
        self._running = False
        self.msgs     = 0

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def snapshot(self) -> list:
        with self._lock:
            return list(self._hist)

    def _loop(self):
        ctx  = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        sock.connect(self.address)
        print(f"[pressure-test] subscribed to {self.address}  (topic '{TOPIC.decode()}')")
        while self._running:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:                        # noqa: BLE001
                continue
            try:
                data    = json.loads(parts[-1])
                sensors = data.get("sensors", {})
                t = time.monotonic() - T0
                row = {}
                for k in range(1, 9):
                    v = sensors.get(str(k))
                    row[k] = float(v) if isinstance(v, (int, float)) else np.nan
                with self._lock:
                    self._hist.append((t, row))
                    self.msgs += 1
            except Exception:                        # noqa: BLE001
                continue
        sock.close()


# ──────────────────────────────────────────────────────────── helpers ────────
def parse_regs(spec: str):
    if spec.strip().lower() == "all":
        return "all"
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise argparse.ArgumentTypeError("no regulator indices parsed")
    return out


def build_apply_msg(regs, psi: float) -> dict:
    if regs == "all":
        return {"cmd": "set_all", "psi": psi}
    return {"cmd": "set_many", "values": {str(r): psi for r in regs}}


# ──────────────────────────────────────────────────────────── main ───────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip",        default=DEFAULT_IP, help=f"Pi Tailscale IP (default {DEFAULT_IP})")
    ap.add_argument("--cmd-port",  type=int, default=CMD_PORT)
    ap.add_argument("--mon-port",  type=int, default=MON_PORT)
    ap.add_argument("--psi",       type=float, default=3.0, help="setpoint pressure [psi] (default 3.0)")
    ap.add_argument("--regs",      type=parse_regs, default="all",
                    help="'all' or comma list, e.g. 0,1,2 (default all)")
    ap.add_argument("--delay",     type=float, default=2.0,
                    help="seconds of baseline before applying the setpoint (default 2.0)")
    ap.add_argument("--resend",    type=float, default=0.0,
                    help="re-send the setpoint every N s (0 = send once, default)")
    ap.add_argument("--window",    type=float, default=30.0,
                    help="rolling time window shown [s] (default 30)")
    ap.add_argument("--no-zero-exit", action="store_true",
                    help="do NOT zero regulators when the script exits")
    ap.add_argument("--monitor-only", action="store_true",
                    help="only plot the pressure feed; never touch the regulator server")
    args = ap.parse_args()

    cmd_addr = f"tcp://{args.ip}:{args.cmd_port}"
    mon_addr = f"tcp://{args.ip}:{args.mon_port}"
    zero_msg  = {"cmd": "zero_all"}
    apply_msg = build_apply_msg(args.regs, args.psi)

    print(f"[pressure-test] regulator server : {cmd_addr}")
    print(f"[pressure-test] pressure monitor : {mon_addr}")
    print(f"[pressure-test] setpoint         : {args.psi} psi -> "
          f"{'all regulators' if args.regs == 'all' else args.regs}")

    # ── connect to regulator server, start from a known (zeroed) state ───────
    reg = None
    if not args.monitor_only:
        reg = Regulator(cmd_addr)
        print("[pressure-test] zeroing regulators ...")
        r = reg.request(zero_msg)
        print(f"   reply: {r}")
        if r.get("status") not in ("ok", "partial"):
            print("[pressure-test] cannot reach regulator server — aborting.")
            print("[pressure-test] (start regulator_server.py on the Pi, or "
                  "re-run with --monitor-only to just watch the sensors)")
            reg.close()
            sys.exit(1)
    else:
        print("[pressure-test] --monitor-only: not sending any regulator commands.")

    # ── start the pressure feed ─────────────────────────────────────────────
    rx = PressureReceiver(mon_addr)
    rx.start()

    # ── figure ─────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title("Pressure test — live sensor readings")
    fig.suptitle("Pressure feed — monitor only" if args.monitor_only else
                 f"Regulator setpoint {args.psi} psi   ·   "
                 f"{'all' if args.regs == 'all' else args.regs}", fontsize=12)

    lines: dict[int, plt.Line2D] = {}
    for ax, group, title in ((ax_top, GROUP_TOP, "Sensors 1–5"),
                             (ax_bot, GROUP_BOT, "Sensors 6–8")):
        for sid in group:
            (lines[sid],) = ax.plot([], [], lw=1.6, color=COLORS[sid], label=f"S{sid}")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("pressure [psi]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncol=len(group), fontsize=8)
    ax_bot.set_xlabel("time since start [s]")

    status = fig.text(0.5, 0.01, "waiting for data …", ha="center", fontsize=8, color="#8b949e")

    state = {"applied": False, "last_send": 0.0}

    def update(_frame):
        now = time.monotonic() - T0

        # step: apply the setpoint once the baseline window has elapsed
        if reg is not None:
            if not state["applied"] and now >= args.delay:
                rep = reg.request(apply_msg)
                print(f"[pressure-test] applied setpoint @ t={now:.1f}s -> {rep}")
                state["applied"]   = True
                state["last_send"] = now
                for ax in (ax_top, ax_bot):
                    ax.axvline(now, color="#adb5bd", ls="--", lw=1.0)
            elif state["applied"] and args.resend > 0 and now - state["last_send"] >= args.resend:
                reg.request(apply_msg)
                state["last_send"] = now

        hist = rx.snapshot()
        if hist:
            ts = np.array([h[0] for h in hist])
            m  = ts >= now - args.window
            for sid, line in lines.items():
                ys = np.array([h[1].get(sid, np.nan) for h in hist])
                line.set_data(ts[m], ys[m])

            x0 = max(0.0, now - args.window)
            ax_top.set_xlim(x0, max(now, x0 + 1.0))
            for ax in (ax_top, ax_bot):
                ax.relim()
                ax.autoscale_view(scalex=False)
                lo, hi = ax.get_ylim()
                ax.set_ylim(min(lo, -0.2), max(hi, 1.0))

        if reg is None:
            note = "monitor only"
        elif state["applied"]:
            note = f"setpoint {args.psi} psi APPLIED"
        else:
            note = f"baseline — applying in {max(0.0, args.delay - now):.1f} s"
        status.set_text(f"{rx.msgs} messages   ·   {note}   ·   Ctrl-C or close window to stop")
        return list(lines.values())

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        del anim
        rx.stop()
        if reg is not None:
            if args.no_zero_exit:
                print("[pressure-test] --no-zero-exit set: leaving regulators pressurized.")
            else:
                print("[pressure-test] zeroing all regulators ...")
                print(f"   reply: {reg.request(zero_msg)}")
            reg.close()
        print(f"[pressure-test] done. {rx.msgs} messages received.")


if __name__ == "__main__":
    main()
