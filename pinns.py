"""
Characterization Experiment Suite
Runs a sequence of Triangular and Sine waves with varying pressure ranges and rates.
"""

import concurrent.futures
import datetime
import logging
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
from typing import List, Optional

import numpy as np

# Optional imports
try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import zmq

    HAS_ZMQ = True
except Exception:
    HAS_ZMQ = False

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# =====================================================================
# Configuration
# =====================================================================

PC_ADDRESS = "0.0.0.0"
ARDUINO_IDS = [3, 8]
ARDUINO_PORTS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]

USE_MOCAP = False
MOCAP_PORT = "tcp://127.0.0.1:3885"
MOCAP_DATA_SIZE = 21

USE_LIVE_PLOT = True and HAS_MPL

# Experiment Parameters
RANGES = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Max PSI
RATES = [1.0, 0.5]  # PSI/s
WAVE_TYPES = ["triangular", "sine"]

RUN_DURATION = 120.0  # seconds
COOLDOWN_DURATION = 20.0  # seconds
RAMPDOWN_DURATION = 5.0  # seconds


# =====================================================================
# Logging setup
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Arduino Manager
# =====================================================================


class ArduinoManager:
    """
    Manages TCP connections to multiple Arduinos.
    PC -> Arduino: float32 desired pressure (psi)
    Arduino -> PC: 4 * int16 ADC counts from ADS1115 (big endian)
    """

    def __init__(self) -> None:
        self.server_sockets: List[socket.socket] = []
        self.client_sockets: List[socket.socket] = []
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(ARDUINO_IDS)
        )

    def connect(self) -> bool:
        """Bind/listen on all configured ports and accept Arduino connections."""
        for arduino_id in ARDUINO_IDS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((PC_ADDRESS, ARDUINO_PORTS[arduino_id - 1]))
                sock.listen(1)
                sock.settimeout(10.0)  # Add timeout for safety
                logger.info(f"Waiting for Arduino {arduino_id}...")
                self.server_sockets.append(sock)

                client, addr = sock.accept()
                client.settimeout(None)  # Reset timeout for blocking operations
                self.client_sockets.append(client)
                logger.info(f"Arduino {arduino_id} connected from {addr}")
            except Exception as e:
                logger.error(f"Failed to connect Arduino {arduino_id}: {e}")
                return False
        return True

    def send_pressure(self, idx: int, pressure: float) -> List[float]:
        try:
            self.client_sockets[idx].send(struct.pack("f", float(pressure)))
            data = b""
            while len(data) < 8:
                chunk = self.client_sockets[idx].recv(8 - len(data))
                if not chunk:
                    raise ConnectionError("Arduino disconnected")
                data += chunk

            sensors: List[float] = []
            for counts in struct.unpack(">4h", data):
                volts = counts * (6.144 / 32768.0)
                pressure_psi = 30.0 * (volts - 0.5) / 4.0
                sensors.append(round(pressure_psi, 8))

            return sensors
        except Exception as e:
            logger.error(f"send_pressure failed for Arduino index {idx}: {e}")
            return [0.0] * 4

    def send_all_parallel(self, desired_pressures: List[float]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(ARDUINO_IDS)

        def _worker(i: int, p: float):
            return i, self.send_pressure(i, p)

        futures = [
            self.executor.submit(_worker, i, desired_pressures[i])
            for i in range(len(ARDUINO_IDS))
        ]

        for fut in concurrent.futures.as_completed(futures):
            try:
                i, sensors = fut.result()
                results[i] = sensors
            except Exception:
                pass

        return [r if r is not None else [math.nan] * 4 for r in results]

    def ramp_down(self, seconds: float) -> None:
        if seconds <= 0:
            return
        start = time.perf_counter()
        current_p = 5.0  # Estimate max
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= seconds:
                cmds = [0.0 for _ in ARDUINO_IDS]
            else:
                alpha = 1.0 - elapsed / seconds
                cmds = [alpha * current_p for _ in ARDUINO_IDS]

            self.send_all_parallel(cmds)

            if elapsed >= seconds:
                break
            time.sleep(0.02)

    def cleanup(self) -> None:
        for s in self.client_sockets + self.server_sockets:
            try:
                s.close()
            except Exception:
                pass
        self.client_sockets.clear()
        self.server_sockets.clear()
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass


# =====================================================================
# Mocap Manager
# =====================================================================


class MocapManager:
    def __init__(self) -> None:
        self.data = [math.nan] * MOCAP_DATA_SIZE
        self.last_timestamp_ns: Optional[int] = None
        self.running = False
        self.socket = None
        self.thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        if not (USE_MOCAP and HAS_ZMQ):
            return False
        try:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.SUB)
            sock.connect(MOCAP_PORT)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.CONFLATE, 1)
            self.socket = sock
            logger.info(f"Connected to mocap at {MOCAP_PORT}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to mocap: {e}")
            return False

    def start(self) -> None:
        if not self.socket:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        last_data = None
        while self.running:
            try:
                msg = self.socket.recv(zmq.NOBLOCK)
                t_ns = time.perf_counter_ns()
                arr = np.fromstring(msg.decode("utf-8"), dtype=float, sep=",")
                if len(arr) >= MOCAP_DATA_SIZE:
                    self.data = arr[:MOCAP_DATA_SIZE].tolist()
                    self.last_timestamp_ns = t_ns
                    last_data = self.data
            except zmq.Again:
                pass
            except Exception:
                if last_data is not None:
                    self.data = last_data.copy()
            time.sleep(0.005)

    def get_data(self):
        return self.data.copy(), self.last_timestamp_ns

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# =====================================================================
# Data Logger
# =====================================================================


class DataLogger:
    """Streaming CSV logger with dynamic file management."""

    def __init__(self) -> None:
        self.base_folder = "experiments_characterization"
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_folder, self.session_timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        self.file_handle = None
        self.current_filepath = None
        self.start_time_ns: Optional[int] = None
        self.lock = threading.RLock()
        self.total_samples = 0
        self.columns = []

    def start_new_file(self, filename: str):
        with self.lock:
            self.stop()  # Close existing if any

            self.current_filepath = os.path.join(self.session_dir, filename)
            logger.info(f"Starting new log file: {self.current_filepath}")

            self.start_time_ns = time.perf_counter_ns()
            self.total_samples = 0

            # Build columns
            col_names = ["step_id", "time"]
            for aid in ARDUINO_IDS:
                col_names.append(f"desired_p_{aid}")
            for aid in ARDUINO_IDS:
                for s in range(1, 5):
                    col_names.append(f"measured_p_{aid}_{s}")

            if USE_MOCAP and HAS_ZMQ:
                col_names.append("mocap_time_rel_s")

            self.columns = col_names

            self.file_handle = open(self.current_filepath, "w", newline="")
            self.file_handle.write(",".join(col_names) + "\n")
            self.file_handle.flush()

    def log(
        self,
        step_id: int,
        desired: List[float],
        measured: List[List[float]],
        mocap_tuple=None,
    ) -> None:
        if not self.file_handle:
            return

        now_ns = time.perf_counter_ns()
        t_rel_s = (now_ns - self.start_time_ns) / 1e9

        row = [str(step_id), f"{t_rel_s:.6f}"]
        row.extend(f"{d:.4f}" for d in desired)
        for m in measured:
            row.extend(f"{v:.4f}" for v in m)

        # Simply ignore mocap details for compactness unless requested
        if USE_MOCAP and HAS_ZMQ:
            row.append("")  # Placeholder for now

        line = ",".join(row) + "\n"

        with self.lock:
            if self.file_handle:
                self.file_handle.write(line)
                self.total_samples += 1
                if self.total_samples % 50 == 0:
                    self.file_handle.flush()

    def stop(self):
        with self.lock:
            if self.file_handle:
                self.file_handle.flush()
                self.file_handle.close()
                self.file_handle = None


# =====================================================================
# Controller
# =====================================================================


class Controller:
    def __init__(self) -> None:
        self.arduinos = ArduinoManager()
        self.logger = DataLogger()
        self.mocap: Optional[MocapManager] = None
        self.running = False

        self.data_lock = threading.Lock()
        self.desired = [0.0] * len(ARDUINO_IDS)
        self.measured = [[0.0] * 4 for _ in ARDUINO_IDS]

        # Wave Parameters
        self.current_wave_type = "idle"  # triangular, sine, idle
        self.current_max_psi = 0.0
        self.current_rate = 1.0  # PSI/s
        self.wave_start_time = 0.0

        self.wave_thread = None
        self.log_thread = None

        self.live_plotter = None
        if USE_LIVE_PLOT:
            self._setup_plot()

    def _setup_plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        self.live_plotter = {"fig": fig, "axes": axes}

        axes[0].set_title("Pressure: Ard3 (Sensors) vs Ard8 (Target)")
        axes[0].set_ylabel("PSI")
        axes[0].grid(True)

        axes[1].set_title("Real-time Monitor")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("PSI")
        axes[1].grid(True)

        fig.tight_layout()

    def initialize(self) -> bool:
        if not self.arduinos.connect():
            return False

        if USE_MOCAP and HAS_ZMQ:
            self.mocap = MocapManager()
            if self.mocap.connect():
                self.mocap.start()

        self.running = True

        self.wave_thread = threading.Thread(target=self._wave_loop, daemon=True)
        self.wave_thread.start()

        self.log_thread = threading.Thread(target=self._log_loop, daemon=True)
        self.log_thread.start()

        return True

    def _wave_loop(self):
        """Generates waves based on current settings."""
        last_debug_time = 0.0
        while self.running:
            now = time.perf_counter()
            val = 0.0

            if self.current_wave_type == "idle":
                val = 0.0

            elif self.current_wave_type == "triangular":
                max_p = self.current_max_psi
                rate = self.current_rate
                if rate <= 0:
                    rate = 0.1

                T = 2.0 * max_p / rate
                if T == 0:
                    T = 1.0

                t = now - self.wave_start_time
                t_cycle = t % T

                if t_cycle < (T / 2.0):
                    val = (max_p / (T / 2.0)) * t_cycle
                else:
                    val = max_p * (1.0 - (t_cycle - (T / 2.0)) / (T / 2.0))

            elif self.current_wave_type == "sine":
                max_p = self.current_max_psi
                rate = self.current_rate
                if rate <= 0:
                    rate = 0.1

                # Match period of triangular wave
                T = 2.0 * max_p / rate
                if T == 0:
                    T = 1.0
                f = 1.0 / T

                t = now - self.wave_start_time

                # Shift by -pi/2 to start at 0
                val = (max_p / 2.0) * (
                    1.0 + math.sin(2.0 * math.pi * f * t - 0.5 * math.pi)
                )

            val = max(0.0, min(val, 25.0))

            target = [0.0] * len(ARDUINO_IDS)
            if len(ARDUINO_IDS) > 1:
                target[1] = val  # Ard 8
            else:
                target[0] = val

            # Debug print every 2 seconds
            if now - last_debug_time > 2.0:
                print(
                    f"DEBUG: Wave Type={self.current_wave_type}, Time={now - self.wave_start_time:.1f}s, Val={val:.2f}, Target={target}",
                    flush=True,
                )
                last_debug_time = now

            measured = self.arduinos.send_all_parallel(target)

            with self.data_lock:
                self.desired = target
                self.measured = measured

            time.sleep(0.01)

    def _log_loop(self):
        step = 0
        while self.running:
            mocap_tup = self.mocap.get_data() if self.mocap else None
            with self.data_lock:
                d = list(self.desired)
                m = list(self.measured)

            self.logger.log(step, d, m, mocap_tup)
            step += 1
            time.sleep(0.01)

    def run_experiment(self):
        """Sequencer loop."""
        print(f"Starting Characterization Sequence.")
        print(f"Ranges: {RANGES}")
        print(f"Rates: {RATES}")
        print(f"Types: {WAVE_TYPES}")
        print(f"Config: {RUN_DURATION}s run, {COOLDOWN_DURATION}s cooldown")
        print(f"Saving to: {self.logger.session_dir}")
        print("Press Ctrl+C to stop.")

        start_exp = time.perf_counter()

        configs = []
        for w_type in WAVE_TYPES:
            for rate in RATES:
                for max_p in RANGES:
                    configs.append((w_type, rate, max_p))

        total_runs = len(configs)
        print(
            f"Total runs: {total_runs}. Est duration: {total_runs * (RUN_DURATION + COOLDOWN_DURATION) / 60:.1f} mins"
        )

        for i, (w_type, rate, max_p) in enumerate(configs):
            if not self.running:
                break

            # Setup
            run_name = f"{w_type}_0-{int(max_p)}psi_{rate}psi-s"
            fname = f"{run_name}.csv"

            print(f"[{i + 1}/{total_runs}] Starting: {run_name}")

            # Start Logging
            self.logger.start_new_file(fname)

            # Start Wave
            self.current_max_psi = max_p
            self.current_rate = rate
            self.current_wave_type = w_type
            self.wave_start_time = time.perf_counter()

            # Run
            run_start = time.perf_counter()
            while (time.perf_counter() - run_start) < RUN_DURATION and self.running:
                self._update_plot()
                time.sleep(0.1)

            # Stop Logging
            self.logger.stop()

            # Cooldown
            print(f"[{i + 1}/{total_runs}] Cooldown ({COOLDOWN_DURATION}s)...")
            self.current_wave_type = "idle"  # Go to 0

            cool_start = time.perf_counter()
            while (
                time.perf_counter() - cool_start
            ) < COOLDOWN_DURATION and self.running:
                self._update_plot()
                time.sleep(0.1)

        print("Sequence Complete.")

    def _update_plot(self):
        if not self.live_plotter:
            return

        if not hasattr(self, "_plot_lists"):
            # Store [time], [target], [s1, s2, s3, s4]
            self._plot_lists = {"t": [], "d": [], "sens": []}

        with self.data_lock:
            # Get all 4 sensors from Ard 3 (idx 0)
            curr_val_ard3 = self.measured[0] if self.measured else [0] * 4
            curr_d = self.desired[1] if len(self.desired) > 1 else self.desired[0]

        now = time.perf_counter()
        self._plot_lists["t"].append(now)
        self._plot_lists["d"].append(curr_d)
        self._plot_lists["sens"].append(curr_val_ard3)

        # Keep window
        if len(self._plot_lists["t"]) > 500:  # approx 50s at 10Hz
            self._plot_lists["t"].pop(0)
            self._plot_lists["d"].pop(0)
            self._plot_lists["sens"].pop(0)

        ax0 = self.live_plotter["axes"][0]
        ax0.cla()

        if self._plot_lists["t"]:
            base_t = self._plot_lists["t"][0]
            rel_t = [t - base_t for t in self._plot_lists["t"]]

            # Plot Target
            ax0.plot(rel_t, self._plot_lists["d"], "k--", label="Target")

            # Plot Sensors
            # sens is list of lists: [[s1,s2,s3,s4], ...]
            # Transpose to plot easily
            sens_arr = list(zip(*self._plot_lists["sens"]))

            colors = ["r", "g", "b", "m"]
            for idx, s_vals in enumerate(sens_arr):
                ax0.plot(
                    rel_t,
                    s_vals,
                    label=f"S{idx + 1}",
                    color=colors[idx % 4],
                    linewidth=1,
                )

        ax0.set_ylim(-0.5, 12.0)
        ax0.legend(loc="upper right", fontsize="small", ncol=2)
        ax0.grid(True)

        plt.pause(0.01)

    def stop(self):
        self.running = False
        self.arduinos.ramp_down(RAMPDOWN_DURATION)
        if self.wave_thread:
            self.wave_thread.join()
        if self.log_thread:
            self.log_thread.join()
        if self.mocap:
            self.mocap.stop()
        self.logger.stop()
        self.arduinos.cleanup()
        if self.live_plotter:
            plt.close(self.live_plotter["fig"])


def main():
    print("DEBUG: Starting main...", flush=True)
    ctrl = Controller()
    print("DEBUG: Controller initialized.", flush=True)

    def on_sig(sig, frame):
        print("\nDEBUG: Caught SIGINT! Stopping...", flush=True)
        ctrl.running = False

    signal.signal(signal.SIGINT, on_sig)

    if ctrl.initialize():
        try:
            print("DEBUG: Running experiment...", flush=True)
            ctrl.run_experiment()
        except KeyboardInterrupt:
            pass
        finally:
            ctrl.stop()


if __name__ == "__main__":
    main()
