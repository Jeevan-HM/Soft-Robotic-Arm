"""
Soft Robot Control System - Corrected Axial Logic

Fixes:
1. Arduino 3 now holds base pressure (2.0 PSI) instead of dropping to 0.
2. Arduino 6 & 8 targets are verified at 2.0 PSI.
3. Added debug prints to verify commanded targets vs actuals.
"""

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
import concurrent.futures
from typing import List, Optional

import h5py
import numpy as np

# Optional imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button 

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

try:
    import google.genai as genai

    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# =====================================================================
# Configuration
# =====================================================================

# IP of the PC that Arduinos connect to
PC_ADDRESS = "0.0.0.0"  # bind on all interfaces

# Which Arduino IDs to use (1-based indexing)
ARDUINO_IDS = [3, 6, 7, 8]

# TCP ports
ARDUINO_PORTS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]

# Experiment settings
EXPERIMENT_DURATION = 300.0  # 5 mins for tuning
RAMPDOWN_DURATION = 5.0

# Initial PID Control Settings
TARGET_PRESSURES = [3.0 for _ in ARDUINO_IDS]
KP_INIT = 2.0
KI_INIT = 0.5
KD_INIT = 0.1   

# DERIVATIVE FILTER ALPHA (0.0 = No new data, 1.0 = No filter)
DERIVATIVE_FILTER_ALPHA = 0.1 

# --- SAFETY LIMIT ---
MAX_SAFETY_PRESSURE = 10.0

# Waveform
WAVE_FUNCTION = "axial"      # "axial", "circular", "static"

# Mocap
USE_MOCAP = True
MOCAP_PORT = "tcp://127.0.0.1:3885"
MOCAP_DATA_SIZE = 21

# Plotting
USE_LIVE_PLOT = True and HAS_MPL

# Gemini
USE_GEMINI_AUTO_DESCRIPTION = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# =====================================================================
# Logging setup
# =====================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# Filtered PID Controller Logic
# =====================================================================

class FilteredPID:
    """
    PID Controller with Low-Pass Filter on Derivative Term
    and Derivative-on-Measurement (prevents setpoint kick).
    """
    def __init__(self, kp: float, ki: float, kd: float, min_out: float, max_out: float, d_alpha: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        self.d_alpha = d_alpha # Smoothing factor for derivative (0.0 to 1.0)
        
        self._integral = 0.0
        self._prev_measurement = 0.0 # Store MEASUREMENT, not error
        self._filtered_derivative = 0.0
        self._last_time = None

    def update(self, setpoint: float, measurement: float, current_time: float) -> float:
        if self._last_time is None:
            self._last_time = current_time
            self._prev_measurement = measurement
            dt = 0.0
        else:
            dt = current_time - self._last_time
            self._last_time = current_time

        if dt <= 1e-5: dt = 1e-4 # Avoid divide by zero

        error = setpoint - measurement

        # 1. Proportional Term
        P = self.kp * error

        # 2. Integral Term (with Anti-Windup)
        self._integral += error * dt
        # Clamp I-term to 50% of range
        i_limit = self.max_out * 0.5
        self._integral = max(-i_limit, min(i_limit, self._integral))
        I = self.ki * self._integral

        # 3. Derivative Term (Derivative on Measurement + Low Pass Filter)
        # We calculate slope of MEASUREMENT, not Error. 
        # Note the negative sign: if Meas goes UP, we want D to push DOWN.
        raw_derivative = -(measurement - self._prev_measurement) / dt
        
        # Apply Low Pass Filter: 
        # New_Filtered = Alpha * New_Raw + (1 - Alpha) * Old_Filtered
        self._filtered_derivative = (self.d_alpha * raw_derivative) + ((1.0 - self.d_alpha) * self._filtered_derivative)
        
        D = self.kd * self._filtered_derivative
        self._prev_measurement = measurement

        # Feedforward (Base pressure to hold target)
        feedforward = setpoint

        # Total Output
        output = feedforward + P + I + D
        output = max(self.min_out, min(self.max_out, output))

        return output

    def reset(self):
        self._integral = 0.0
        self._prev_measurement = 0.0
        self._filtered_derivative = 0.0
        self._last_time = None


# =====================================================================
# Arduino Manager
# =====================================================================

class ArduinoManager:
    def __init__(self) -> None:
        self.server_sockets: List[socket.socket] = []
        self.client_sockets: List[socket.socket] = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(ARDUINO_IDS))

    def connect(self) -> None:
        for arduino_id in ARDUINO_IDS:
            port = ARDUINO_PORTS[arduino_id - 1]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((PC_ADDRESS, port))
                sock.listen(1)
                logger.info(f"Waiting for Arduino {arduino_id} on port {port}...")
                self.server_sockets.append(sock)
                client, addr = sock.accept()
                logger.info(f"Arduino {arduino_id} connected from {addr}")
                self.client_sockets.append(client)
            except Exception as e:
                logger.error(f"Failed to bind/connect Arduino {arduino_id}: {e}")
                sys.exit(1)

    def send_pressure(self, idx: int, pressure: float) -> List[float]:
        try:
            # HARDWARE SAFETY CLAMP
            safe_pressure = max(0.0, min(pressure, MAX_SAFETY_PRESSURE))
            
            self.client_sockets[idx].send(struct.pack("f", float(safe_pressure)))
            data = b""
            while len(data) < 8:
                chunk = self.client_sockets[idx].recv(8 - len(data))
                if not chunk: raise ConnectionError("Arduino disconnected")
                data += chunk
            sensors: List[float] = []
            for counts in struct.unpack(">4h", data):
                volts = counts * (6.144 / 32768.0)
                pressure_psi = 30.0 * (volts - 0.5) / 4.0
                sensors.append(pressure_psi)
            return sensors
        except Exception as e:
            logger.error(f"send_pressure failed for Arduino index {idx}: {e}")
            return [float("nan")] * 4

    def send_all_parallel(self, control_signals: List[float]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(ARDUINO_IDS)
        def _worker(i: int, p: float): return i, self.send_pressure(i, p)
        futures = [self.executor.submit(_worker, i, control_signals[i]) for i in range(len(ARDUINO_IDS))]
        for fut in concurrent.futures.as_completed(futures):
            i, sensors = fut.result()
            results[i] = sensors
        return [r if r is not None else [math.nan] * 4 for r in results]

    def ramp_down(self, seconds: float) -> None:
        if seconds <= 0: return
        start = time.perf_counter()
        initial = [3.0] * len(ARDUINO_IDS)
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= seconds: cmds = [0.0 for _ in ARDUINO_IDS]
            else:
                alpha = 1.0 - elapsed / seconds
                cmds = [alpha * p for p in initial]
            self.send_all_parallel(cmds)
            if elapsed >= seconds: break
            time.sleep(0.02)

    def cleanup(self) -> None:
        for s in self.client_sockets + self.server_sockets:
            try: s.close()
            except Exception: pass
        self.client_sockets.clear()
        self.server_sockets.clear()
        try: self.executor.shutdown(wait=False)
        except Exception: pass


# =====================================================================
# Mocap & Data Logger
# =====================================================================

class MocapManager:
    def __init__(self) -> None:
        self.data = [math.nan] * MOCAP_DATA_SIZE
        self.last_timestamp_ns: Optional[int] = None
        self.running = False
        self.socket = None
        self.thread: Optional[threading.Thread] = None
    def connect(self) -> bool:
        if not (USE_MOCAP and HAS_ZMQ): return False
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
        if not self.socket: return
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
            except zmq.Again: pass
            except Exception:
                if last_data is not None: self.data = last_data.copy()
            time.sleep(0.005)
    def get_data(self): return self.data.copy(), self.last_timestamp_ns
    def stop(self) -> None:
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)


class DataLogger:
    def __init__(self) -> None:
        self.hdf5_path: Optional[str] = None
        self.exp_group_name: Optional[str] = None
        self.start_time_ns: Optional[int] = None
        self.last_flush_wall: Optional[float] = None
        self.data_buffer: List[List[float]] = []
        self.lock = threading.Lock()
        self.flush_interval_s = 0.5
        self.flush_batch_size = 100
        self.columns: List[str] = []
        self._initialized_dataset = False
        self.total_samples = 0
        self.current_kp = KP_INIT
        self.current_ki = KI_INIT
        self.current_kd = KD_INIT

    def _build_column_names(self) -> List[str]:
        cols: List[str] = ["step_id", "time"]
        for aid in ARDUINO_IDS: cols.append(f"pd_{aid}")
        for aid in ARDUINO_IDS: cols.append(f"u_{aid}")
        for aid in ARDUINO_IDS: 
            for s in range(1, 5): cols.append(f"pm_{aid}_{s}")
        if USE_MOCAP and HAS_ZMQ:
            for body_num in range(1, 4):
                cols.extend([f"mocap_{body_num}_{ax}" for ax in ["x","y","z","qx","qy","qz","qw"]])
            cols.append("mocap_time_rel_s")
        cols.extend(["kp_log", "ki_log", "kd_log"])
        return cols

    def _ensure_dataset_and_append(self, data_array: np.ndarray) -> None:
        if data_array.size == 0: return
        with h5py.File(self.hdf5_path, "a") as f:
            if not self._initialized_dataset:
                grp = f.create_group(self.exp_group_name)
                grp.create_dataset("data", data=data_array, maxshape=(None, data_array.shape[1]), compression="gzip", compression_opts=4)
                grp.attrs["columns"] = self.columns
                grp.attrs["timestamp"] = datetime.datetime.now().isoformat()
                grp.attrs["wave_function"] = WAVE_FUNCTION
                grp.attrs["max_safety"] = MAX_SAFETY_PRESSURE
                self._initialized_dataset = True
            else:
                grp = f[self.exp_group_name]
                dset = grp["data"]
                old_rows = dset.shape[0]
                new_rows = old_rows + data_array.shape[0]
                dset.resize((new_rows, dset.shape[1]))
                dset[old_rows:new_rows, :] = data_array
            f.flush()

    def _flush_buffer_locked(self) -> None:
        if not self.data_buffer: return
        arr = np.array(self.data_buffer, dtype=np.float64)
        self.data_buffer.clear()
        self._ensure_dataset_and_append(arr)
        self.last_flush_wall = time.time()

    def start(self) -> None:
        now = datetime.datetime.now()
        folder = "experiments"
        os.makedirs(folder, exist_ok=True)
        month_file = now.strftime("%Y_%B.h5")
        self.hdf5_path = os.path.join(folder, month_file)
        exp_num = 1
        if os.path.exists(self.hdf5_path):
            try:
                with h5py.File(self.hdf5_path, "r") as f:
                    existing = [k for k in f.keys() if k.startswith("exp_")]
                    numbers = []
                    for name in existing:
                        try: numbers.append(int(name.split("_")[1]))
                        except: pass
                    if numbers: exp_num = max(numbers) + 1
            except Exception: pass
        date_str = now.strftime("%b%d_%Hh%Mm")
        self.exp_group_name = f"exp_{exp_num:03d}_{WAVE_FUNCTION}_{date_str}"
        self.start_time_ns = time.perf_counter_ns()
        self.last_flush_wall = time.time()
        self.columns = self._build_column_names()
        logger.info(f"Logging to HDF5: {os.path.abspath(self.hdf5_path)} group={self.exp_group_name}")

    def log(self, step_id: int, desired: List[float], commands: List[float], measured: List[List[float]], mocap_tuple=None) -> None:
        if self.hdf5_path is None or self.start_time_ns is None: return
        now_ns = time.perf_counter_ns()
        t_rel_s = (now_ns - self.start_time_ns) / 1e9
        row: List[float] = [float(step_id), float(t_rel_s)]
        row.extend(float(p) for p in desired)
        row.extend(float(c) for c in commands)
        for sensors in measured: row.extend(float(v) for v in sensors)
        if USE_MOCAP and HAS_ZMQ:
            mocap_data = None
            mocap_dt_s = math.nan
            if mocap_tuple is not None:
                mocap_vec, mocap_ts_ns = mocap_tuple
                if mocap_vec is not None and len(mocap_vec) >= MOCAP_DATA_SIZE:
                    mocap_data = mocap_vec[:MOCAP_DATA_SIZE]
                    if mocap_ts_ns is not None: mocap_dt_s = (mocap_ts_ns - self.start_time_ns) / 1e9
            if mocap_data is None: row.extend([math.nan] * MOCAP_DATA_SIZE)
            else: row.extend(float(v) for v in mocap_data)
            row.append(float(mocap_dt_s))
        
        row.extend([self.current_kp, self.current_ki, self.current_kd])

        with self.lock:
            self.data_buffer.append(row)
            self.total_samples += 1
            need_flush = False
            if len(self.data_buffer) >= self.flush_batch_size: need_flush = True
            elif self.last_flush_wall is None or (time.time() - self.last_flush_wall) >= self.flush_interval_s: need_flush = True
            if need_flush: self._flush_buffer_locked()

    def update_pid_log_values(self, kp, ki, kd):
        self.current_kp = kp
        self.current_ki = ki
        self.current_kd = kd

    def stop(self, description: Optional[str] = None) -> None:
        if self.hdf5_path is None or self.exp_group_name is None: return
        with self.lock: self._flush_buffer_locked()
        try:
            with h5py.File(self.hdf5_path, "a") as f:
                if self.exp_group_name in f:
                    grp = f[self.exp_group_name]
                    grp.attrs["sample_count"] = int(self.total_samples)
                    grp.attrs["description"] = description if description else "No description"
                    grp.attrs["final_kp"] = self.current_kp
                    grp.attrs["final_ki"] = self.current_ki
                    grp.attrs["final_kd"] = self.current_kd
        except Exception as e: logger.error(f"Error saving HDF5 metadata: {e}")

    def generate_ai_description(self) -> str:
        if not (GEMINI_AVAILABLE and USE_GEMINI_AUTO_DESCRIPTION and GEMINI_API_KEY): return "No description provided"
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"Describe soft robot exp: {WAVE_FUNCTION}, {int(EXPERIMENT_DURATION)}s, PID Tuned to Kp={self.current_kp:.2f}, Ki={self.current_ki:.2f}, Kd={self.current_kd:.2f}."
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return resp.text.strip()
        except Exception: return "No description provided"


# =====================================================================
# Controller with Live GUI
# =====================================================================

class Controller:
    def __init__(self) -> None:
        self.arduinos = ArduinoManager()
        self.logger = DataLogger()
        self.mocap: Optional[MocapManager] = None
        self.running = False
        self.data_lock = threading.Lock()
        
        # Use Filtered PID
        self.pids = [FilteredPID(KP_INIT, KI_INIT, KD_INIT, 0.0, MAX_SAFETY_PRESSURE, d_alpha=DERIVATIVE_FILTER_ALPHA) 
                     for _ in ARDUINO_IDS]

        self.desired: List[float] = [0.0 for _ in ARDUINO_IDS]
        self.commands: List[float] = [0.0 for _ in ARDUINO_IDS]
        self.measured: List[List[float]] = [[0.0] * 4 for _ in ARDUINO_IDS]

        self.wave_thread: Optional[threading.Thread] = None
        self.log_thread: Optional[threading.Thread] = None
        self.progress_thread: Optional[threading.Thread] = None
        self.live_plotter = None
        
        self.widgets = [] 

        if USE_LIVE_PLOT and HAS_MPL:
            self._setup_plot()

    def _setup_plot(self) -> None:
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3) 

        self.live_plotter = {"fig": fig, "ax": ax}
        ax.set_title(f"Live Tuning (Init: Kp={KP_INIT})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (psi)")
        ax.grid(True)

        ax_kp = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_ki = plt.axes([0.25, 0.10, 0.65, 0.03])
        ax_kd = plt.axes([0.25, 0.05, 0.65, 0.03])

        s_kp = Slider(ax_kp, 'Proportional (Kp)', 0.0, 10.0, valinit=KP_INIT, valstep=0.1)
        s_ki = Slider(ax_ki, 'Integral (Ki)', 0.0, 5.0, valinit=KI_INIT, valstep=0.01)
        # Increased range for Kd since filtered D is often smaller
        s_kd = Slider(ax_kd, 'Derivative (Kd)', 0.0, 2.0, valinit=KD_INIT, valstep=0.01)

        def update_val(val):
            kp_new = s_kp.val
            ki_new = s_ki.val
            kd_new = s_kd.val
            
            for pid in self.pids:
                pid.kp = kp_new
                pid.ki = ki_new
                pid.kd = kd_new
            
            self.logger.update_pid_log_values(kp_new, ki_new, kd_new)
            
            ax.set_title(f"Live Tuning: Kp={kp_new:.2f}, Ki={ki_new:.2f}, Kd={kd_new:.2f}")

        s_kp.on_changed(update_val)
        s_ki.on_changed(update_val)
        s_kd.on_changed(update_val)

        self.widgets = [s_kp, s_ki, s_kd]

    def initialize(self) -> bool:
        logger.info("Connecting Arduinos...")
        self.arduinos.connect()
        logger.info("Arduinos connected.")
        if USE_MOCAP and HAS_ZMQ:
            self.mocap = MocapManager()
            if self.mocap.connect(): self.mocap.start()
            else: self.mocap = None
        self.logger.start()
        self.running = True
        self.wave_thread = threading.Thread(target=self._wave_loop, daemon=True)
        self.wave_thread.start()
        self.log_thread = threading.Thread(target=self._log_loop, daemon=True)
        self.log_thread.start()
        self.progress_thread = threading.Thread(target=self._progress_loop, daemon=True)
        self.progress_thread.start()
        return True

    def _wave_loop(self) -> None:
        start_time = time.perf_counter()
        PREFILL_DURATION = 5.0
        AXIAL_FREQ = 0.1
        AXIAL_CENTER = 5.0
        AXIAL_AMPL = 5.0

        last_print = time.time()

        while self.running:
            now = time.perf_counter()
            t = now - start_time
            desired: List[float] = [0.0] * len(ARDUINO_IDS)

            if WAVE_FUNCTION == "axial":
                base = 2.0
                if t < PREFILL_DURATION: desired = [base] * len(ARDUINO_IDS)
                else:
                    # UPDATED LOGIC: Keep everyone at base pressure (2 PSI)
                    # except the one that is moving.
                    desired[0] = base               # Arduino 3 now holds pressure
                    desired[1] = base               # Arduino 6 should be 2.0
                    desired[3] = base               # Arduino 8 should be 2.0
                    
                    tau = t - PREFILL_DURATION
                    p3 = AXIAL_CENTER + AXIAL_AMPL * math.sin(2.0 * math.pi * AXIAL_FREQ * tau)
                    
                    # Ensure the generated wave ALSO respects the safety limit
                    p3 = max(0.0, min(MAX_SAFETY_PRESSURE, p3))
                    desired[2] = p3
            elif WAVE_FUNCTION == "circular":
                for i, base in enumerate(TARGET_PRESSURES):
                    phase = (2.0 * math.pi / len(TARGET_PRESSURES)) * i
                    val = base + base * 0.5 * math.sin(2.0 * math.pi * 0.1 * t + phase)
                    desired.append(max(0.0, val))
            else: desired = TARGET_PRESSURES.copy()

            # DEBUG PRINT (Once per second)
            if time.time() - last_print > 1.0:
                print(f"[DEBUG] Ard 3 Tgt: {desired[0]:.1f} | Ard 6 Tgt: {desired[1]:.1f} | Ard 7 Tgt: {desired[2]:.1f} | Ard 8 Tgt: {desired[3]:.1f}")
                last_print = time.time()

            commands = []
            with self.data_lock: current_measurements = self.measured
            for i, target in enumerate(desired):
                curr_p = current_measurements[i][0]
                if math.isnan(curr_p): curr_p = 0.0
                u = self.pids[i].update(target, curr_p, now)
                commands.append(u)

            new_measured = self.arduinos.send_all_parallel(commands)
            with self.data_lock:
                self.desired = desired
                self.commands = commands
                self.measured = new_measured
            time.sleep(0.01)

    def _log_loop(self) -> None:
        step_id = 0
        while self.running:
            if self.mocap: mocap_tuple = self.mocap.get_data()
            else: mocap_tuple = None
            with self.data_lock:
                desired_copy = list(self.desired)
                commands_copy = list(self.commands)
                measured_copy = [list(m) for m in self.measured]
            self.logger.log(step_id, desired_copy, commands_copy, measured_copy, mocap_tuple)
            step_id += 1
            time.sleep(0.01)

    def _progress_loop(self) -> None:
        start = time.time()
        total_duration = EXPERIMENT_DURATION
        total_duration_str = f"{int(total_duration)}s"
        while self.running:
            elapsed = time.time() - start
            elapsed = max(0.0, elapsed)
            remaining = max(0.0, total_duration - elapsed)
            progress = min(100.0, (elapsed / total_duration) * 100.0)
            bar_length = 30
            filled = int(bar_length * progress / 100.0)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r[{bar}] {progress:5.1f}% | {int(elapsed)}s/{total_duration_str} | rem: {int(remaining)}s", end="", flush=True)
            time.sleep(1.0)
        print()

    def run(self) -> None:
        if not self.running: return
        logger.info(f"Starting LIVE TUNING Experiment. Duration: {EXPERIMENT_DURATION}s")
        start = time.perf_counter()
        times: List[float] = []
        traces: List[List[float]] = [[] for _ in ARDUINO_IDS]
        target_traces: List[List[float]] = [[] for _ in ARDUINO_IDS]

        while self.running and (time.perf_counter() - start) < EXPERIMENT_DURATION:
            if self.live_plotter:
                now_t = time.perf_counter() - start
                with self.data_lock:
                    current_measured = [m[0] for m in self.measured]
                    current_desired = list(self.desired)
                times.append(now_t)
                for i, p in enumerate(current_measured):
                    traces[i].append(p)
                    target_traces[i].append(current_desired[i])
                if len(times) > 300: 
                    times.pop(0)
                    for t in traces: t.pop(0)
                    for t in target_traces: t.pop(0)
                
                ax = self.live_plotter["ax"]
                ax.cla()
                for i, aid in enumerate(ARDUINO_IDS):
                    ax.plot(times, traces[i], label=f"Meas {aid}")
                    ax.plot(times, target_traces[i], linestyle='--', alpha=0.5)
                ax.legend(loc='upper right', fontsize='small')
                ax.grid(True)
                
                plt.pause(0.1) 
            else: time.sleep(0.1)
        self.stop()

    def stop(self) -> None:
        if not self.running: return
        self.running = False
        try: self.arduinos.ramp_down(RAMPDOWN_DURATION)
        except Exception: pass
        if self.wave_thread: self.wave_thread.join(timeout=1.0)
        if self.log_thread: self.log_thread.join(timeout=1.0)
        if self.progress_thread: self.progress_thread.join(timeout=1.0)
        if self.mocap: self.mocap.stop()

        desc = None
        if USE_GEMINI_AUTO_DESCRIPTION and GEMINI_AVAILABLE and GEMINI_API_KEY:
            print("\nGenerating AI description...")
            desc = self.logger.generate_ai_description()
            print(desc)
        self.logger.stop(desc)
        self.arduinos.cleanup()
        logger.info("Controller stopped.")

def main() -> int:
    controller = Controller()
    def _signal_handler(sig, frame):
        logger.warning("Ctrl+C received, stopping...")
        controller.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        if not controller.initialize(): return 1
        controller.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        controller.stop()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())