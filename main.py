"""
Simplified Soft Robot Pressure Control System

Configuration: Modify the variables below
Run with: python main.py

Optional: For AI auto-descriptions, add to .env file:
  GEMINI_API_KEY=your-api-key-here
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

import h5py
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

# TCP ports (index 0 is Arduino 1, etc.)
ARDUINO_PORTS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]

# Experiment settings
# Experiment settings
EXPERIMENT_DURATION = 0.0  # Will be calculated below
RAMPDOWN_DURATION = 5.0  # seconds to ramp down all pressures to zero

# Desired base pressures (one per Arduino ID, in PSI)
TARGET_PRESSURES = [2.0 for _ in ARDUINO_IDS]

# Waveform to run
WAVE_FUNCTION = "sequence"  # "sequence", "axial", "circular", "triangular", "static"

# Sequence Configuration
SEQ_WAVE_TYPES = ["axial", "circular", "triangular"]
SEQ_SEG1_PRESSURES = [3.0]
SEQ_MAX_PRESSURES = [5.0, 10.0]
SEQ_WAVE_DURATION = 100.0  # Duration for each wave type in the sequence
SEQ_COOLDOWN_DURATION = 5.0  # Duration of 2psi hold between waves
SEQ_SEG1_REFILL_PERIOD = 100.0  # Target period for refilling Segment 1
SEQ_REFILL_ACTION_DURATION = 5.0  # Duration of the refill pause/interruption

# Calculate Experiment Duration
_num_items = len(SEQ_WAVE_TYPES) * len(SEQ_SEG1_PRESSURES) * len(SEQ_MAX_PRESSURES)
_raw_duration = 10.0 + _num_items * (SEQ_WAVE_DURATION + SEQ_COOLDOWN_DURATION)
# Add buffer for refills (approx estimate: 1 refill every 100s taking 5s -> +5%)
# We add 20% buffer to be safe
EXPERIMENT_DURATION = _raw_duration * 1.2

# Mocap settings
USE_MOCAP = True
MOCAP_PORT = "tcp://127.0.0.1:3885"
MOCAP_DATA_SIZE = 21  # 3 rigid bodies * 7 values each

# Plotting
USE_LIVE_PLOT = True and HAS_MPL
ARDUINOS_NO_PLOT = [6, 8]

# Gemini API settings
USE_GEMINI_AUTO_DESCRIPTION = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
            # Implementation hidden
            pass
        return True  # Mocked

    pass


# CALL 1: Constants

MOCAP_PORT = "tcp://127.0.0.1:3885"
MOCAP_DATA_SIZE = 21  # 3 rigid bodies * 7 values each

# Plotting
USE_LIVE_PLOT = True and HAS_MPL
ARDUINOS_NO_PLOT = [6]

# Gemini API settings
USE_GEMINI_AUTO_DESCRIPTION = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
            sock.bind((PC_ADDRESS, ARDUINO_PORTS[arduino_id - 1]))
            sock.listen(1)
            sock.settimeout(10.0)  # Add timeout for safety
            logger.info(f"Waiting for Arduino {arduino_id}...")
            self.server_sockets.append(sock)

            try:
                client, addr = sock.accept()
                client.settimeout(None)  # Reset timeout for blocking operations
                self.client_sockets.append(client)
                logger.info(f"Arduino {arduino_id} connected from {addr}")
            except socket.timeout:
                logger.error(f"Timeout waiting for Arduino {arduino_id}")
                return False
        return True

    def send_pressure(self, idx: int, pressure: float) -> List[float]:
        """
        Send desired pressure (psi) to Arduino idx and read back 4 sensor pressures.

        Arduino sends 4 x int16 from an ADS1115 configured with:
            adc.setGain(GAIN_TWOTHIRDS);  # ±6.144 V full-scale

        ADS1115 LSB = 6.144 / 32768 V ≈ 0.0001875 V per count.
        Pressure sensor: 0.5–4.5 V -> 0–30 psi.
        """
        try:
            # send desired pressure as float32 (little-endian from Arduino side)
            self.client_sockets[idx].send(struct.pack("f", float(pressure)))

            # read 4 * int16 = 8 bytes
            data = b""
            while len(data) < 8:
                chunk = self.client_sockets[idx].recv(8 - len(data))
                if not chunk:
                    raise ConnectionError("Arduino disconnected")
                data += chunk

            sensors: List[float] = []
            for counts in struct.unpack(">4h", data):
                # ADS1115 conversion: counts -> volts (GAIN_TWOTHIRDS, ±6.144 V)
                volts = counts * (6.144 / 32768.0)

                # Sensor transfer: 0.5–4.5 V => 0–30 psi
                pressure_psi = 30.0 * (volts - 0.5) / 4.0

                sensors.append(round(pressure_psi, 8))

            return sensors
        except Exception as e:
            logger.error(f"send_pressure failed for Arduino index {idx}: {e}")
            return [0.0] * 4

    def send_all_parallel(self, desired_pressures: List[float]) -> List[List[float]]:
        """Send pressures to all Arduinos in parallel using a thread pool."""
        results: List[Optional[List[float]]] = [None] * len(ARDUINO_IDS)

        def _worker(i: int, p: float):
            return i, self.send_pressure(i, p)

        futures = [
            self.executor.submit(_worker, i, desired_pressures[i])
            for i in range(len(ARDUINO_IDS))
        ]

        for fut in concurrent.futures.as_completed(futures):
            i, sensors = fut.result()
            results[i] = sensors

        # Replace any None with NaNs
        return [r if r is not None else [math.nan] * 4 for r in results]

    def ramp_down(self, seconds: float) -> None:
        """Linearly ramp all pressures down to zero over given time."""
        if seconds <= 0:
            return
        start = time.perf_counter()
        # capture last commanded pressures if you want; here use TARGET_PRESSURES
        initial = TARGET_PRESSURES.copy()
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= seconds:
                cmds = [0.0 for _ in ARDUINO_IDS]
            else:
                alpha = 1.0 - elapsed / seconds
                cmds = [alpha * p for p in initial]

            self.send_all_parallel(cmds)

            if elapsed >= seconds:
                break
            time.sleep(0.02)

    def cleanup(self) -> None:
        """Close all sockets and shut down executor."""
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
    """ZeroMQ subscriber for mocap data."""

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
            # Only keep the latest sample
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
                # no message ready
                pass
            except Exception:
                # keep last good data if something goes wrong
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
    """
    Streaming HDF5 logger with:
      - step_id (monotonic integer index)
      - time (monotonic high-res seconds since start)
      - periodic flushing to disk to reduce data loss on crash
    """

    def __init__(self) -> None:
        self.hdf5_path: Optional[str] = None
        self.exp_group_name: Optional[str] = None

        self.start_time_ns: Optional[int] = None
        self.last_flush_wall: Optional[float] = None

        self.data_buffer: List[List[float]] = []
        self.wave_type_buffer: List[str] = []
        self.lock = threading.Lock()

        self.flush_interval_s = 0.5
        self.flush_batch_size = 100

        self.columns: List[str] = []
        self._initialized_dataset = False
        self.total_samples = 0

    # ---------------- column & dataset helpers ----------------

    def _build_column_names(self) -> List[str]:
        cols: List[str] = ["step_id", "time"]

        # Desired pressures
        for aid in ARDUINO_IDS:
            cols.append(f"pd_{aid}")

        # Measured pressures: 4 sensors per Arduino
        for aid in ARDUINO_IDS:
            for s in range(1, 5):
                cols.append(f"pm_{aid}_{s}")

        # Mocap columns
        if USE_MOCAP and HAS_ZMQ:
            # 3 bodies * (x,y,z,qx,qy,qz,qw)
            for body_num in range(1, 4):
                cols.extend(
                    [
                        f"mocap_{body_num}_x",
                        f"mocap_{body_num}_y",
                        f"mocap_{body_num}_z",
                        f"mocap_{body_num}_qx",
                        f"mocap_{body_num}_qy",
                        f"mocap_{body_num}_qz",
                        f"mocap_{body_num}_qw",
                    ]
                )
            # plus mocap time offset
            cols.append("mocap_time_rel_s")

        # Sequence columns
        # Note: 'config_wave_type' is now stored in a separate string dataset
        cols.extend(["config_seg1_psi", "config_max_psi"])

        return cols

    def _ensure_dataset_and_append(
        self, data_array: np.ndarray, wave_types: List[str]
    ) -> None:
        if data_array.size == 0:
            return

        with h5py.File(self.hdf5_path, "a") as f:
            if not self._initialized_dataset:
                grp = f.create_group(self.exp_group_name)

                # 1. Numeric Data
                grp.create_dataset(
                    "data",
                    data=data_array,
                    maxshape=(None, data_array.shape[1]),
                    compression="gzip",
                    compression_opts=4,
                )

                # 2. String Data (Wave Types)
                # Use variable-length UTF-8 string dtype
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset(
                    "wave_types",
                    data=np.array(wave_types, dtype=object),
                    maxshape=(None,),
                    dtype=dt,
                    compression="gzip",
                    compression_opts=4,
                )

                grp.attrs["columns"] = self.columns
                grp.attrs["timestamp"] = datetime.datetime.now().isoformat()
                grp.attrs["wave_function"] = WAVE_FUNCTION
                grp.attrs["arduino_ids"] = ARDUINO_IDS
                grp.attrs["target_pressures"] = TARGET_PRESSURES
                grp.attrs["mocap_enabled"] = bool(USE_MOCAP and HAS_ZMQ)

                self._initialized_dataset = True
            else:
                grp = f[self.exp_group_name]

                # Resize and Append Numeric Data
                dset = grp["data"]
                old_rows = dset.shape[0]
                new_rows = old_rows + data_array.shape[0]
                dset.resize((new_rows, dset.shape[1]))
                dset[old_rows:new_rows, :] = data_array

                # Resize and Append String Data
                dset_str = grp["wave_types"]
                dset_str.resize((new_rows,))
                dset_str[old_rows:new_rows] = np.array(wave_types, dtype=object)

            f.flush()

    def _flush_buffer_locked(self) -> None:
        if not self.data_buffer:
            return
        arr = np.array(self.data_buffer, dtype=np.float64)
        waves = list(self.wave_type_buffer)

        self.data_buffer.clear()
        self.wave_type_buffer.clear()

        self._ensure_dataset_and_append(arr, waves)
        self.last_flush_wall = time.time()

    # ---------------- public API ----------------

    def start(self):
        now = datetime.datetime.now()
        folder = "experiments"
        os.makedirs(folder, exist_ok=True)

        # Main monthly HDF5 file (we will copy to this later if saved)
        self.month_file_path = os.path.join(folder, now.strftime("%Y_%B.h5"))

        # Temporary file for current experiment
        self.temp_path = os.path.join(folder, "temp_experiment.h5")
        if os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except OSError:
                pass

        # We log to the temp path
        self.hdf5_path = self.temp_path

        # Determine experiment number (check main file)
        exp_num = 1
        if os.path.exists(self.month_file_path):
            with h5py.File(self.month_file_path, "r") as f:
                existing = [k for k in f.keys() if k.startswith("exp_")]
                if existing:
                    numbers = []
                    for exp_name in existing:
                        try:
                            parts = exp_name.split("_")
                            if len(parts) >= 2:
                                num = int(parts[1])
                                numbers.append(num)
                        except (ValueError, IndexError):
                            continue
                    if numbers:
                        exp_num = max(numbers) + 1

        # Create experiment name
        date_str = now.strftime("%b%d_%Hh%Mm")
        self.exp_group_name = f"exp_{exp_num:03d}_{WAVE_FUNCTION}_{date_str}"
        self.start_time = time.time()
        self.start_time_ns = time.perf_counter_ns()

        logger.info(
            f"Logging to temp file: {os.path.abspath(self.hdf5_path)} -> {self.exp_group_name}"
        )

        # Initialize HDF5 file and dataset
        self._init_hdf5()

    def _init_hdf5(self):
        if not self.hdf5_path:
            return

        try:
            with h5py.File(self.hdf5_path, "a") as f:
                grp = f.create_group(self.exp_group_name)

                # Create column names
                col_names = ["step_id", "time"]
                for aid in ARDUINO_IDS:
                    col_names.append(f"pd_{aid}")
                for aid in ARDUINO_IDS:
                    for s in range(1, 5):
                        col_names.append(f"pm_{aid}_{s}")

                if USE_MOCAP and HAS_ZMQ:
                    for body_num in range(1, 4):
                        col_names.append(f"mocap_{body_num}_x")
                        col_names.append(f"mocap_{body_num}_y")
                        col_names.append(f"mocap_{body_num}_z")
                        col_names.append(f"mocap_{body_num}_qx")
                        col_names.append(f"mocap_{body_num}_qy")
                        col_names.append(f"mocap_{body_num}_qz")
                        col_names.append(f"mocap_{body_num}_qw")

                    col_names.append("mocap_time_rel_s")

                # Sequence columns (Wave type is now in separate dataset)
                col_names.extend(["config_seg1_psi", "config_max_psi"])

                # Create resizable dataset for numeric data
                num_cols = len(col_names)
                grp.create_dataset(
                    "data",
                    shape=(0, num_cols),
                    maxshape=(None, num_cols),
                    dtype="float32",
                    compression="gzip",
                    compression_opts=4,
                )

                # Create resizable dataset for string data (Wave Types)
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset(
                    "wave_types",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dt,
                    compression="gzip",
                    compression_opts=4,
                )

                # Save metadata immediately
                grp.attrs["columns"] = col_names
                grp.attrs["timestamp"] = datetime.datetime.now().isoformat()
                grp.attrs["experiment_type"] = WAVE_FUNCTION
                grp.attrs["duration_seconds"] = EXPERIMENT_DURATION
                grp.attrs["mocap_enabled"] = USE_MOCAP and HAS_ZMQ
                grp.attrs["arduino_ids"] = ARDUINO_IDS
                grp.attrs["target_pressures_psi"] = TARGET_PRESSURES
                grp.attrs["rampdown_duration_seconds"] = RAMPDOWN_DURATION
                grp.attrs["pc_address"] = PC_ADDRESS

                if USE_MOCAP and HAS_ZMQ:
                    grp.attrs["mocap_port"] = MOCAP_PORT
                    grp.attrs["mocap_data_size"] = MOCAP_DATA_SIZE

                # Description will be updated at the end
                grp.attrs["description"] = "Experiment in progress..."

                self._initialized_dataset = True

        except Exception as e:
            logger.error(f"Error initializing HDF5: {e}")

    def finalize(self, save: bool):
        """
        If save is True, copy data from temp file to the main month file.
        Then delete the temp file.
        """
        if not self.hdf5_path or not os.path.exists(self.hdf5_path):
            return

        if save:
            try:
                logger.info(f"Saving data to {self.month_file_path}...")
                with h5py.File(self.hdf5_path, "r") as src:
                    with h5py.File(self.month_file_path, "a") as dst:
                        # Copy group
                        src.copy(src[self.exp_group_name], dst, self.exp_group_name)
                logger.info("Save complete.")
            except Exception as e:
                logger.error(f"Error saving data: {e}")
        else:
            logger.info("Data discarded (not saved).")

        # Clean up temp file
        try:
            os.remove(self.hdf5_path)
        except OSError as e:
            logger.warning(f"Could not remove temp file: {e}")

        self.hdf5_path = None

    def log(
        self,
        step_id: int,
        desired: List[float],
        measured: List[List[float]],
        mocap_tuple=None,
        seq_config=None,
    ) -> None:
        if self.hdf5_path is None or self.start_time_ns is None:
            return

        now_ns = time.perf_counter_ns()
        t_rel_s = (now_ns - self.start_time_ns) / 1e9

        row: List[float] = [float(step_id), float(t_rel_s)]

        # Desired
        row.extend(float(p) for p in desired)

        # Measured
        for sensors in measured:
            row.extend(float(v) for v in sensors)

        if USE_MOCAP and HAS_ZMQ:
            mocap_data = None
            mocap_dt_s = math.nan
            if mocap_tuple is not None:
                mocap_vec, mocap_ts_ns = mocap_tuple
                if mocap_vec is not None and len(mocap_vec) >= MOCAP_DATA_SIZE:
                    mocap_data = mocap_vec[:MOCAP_DATA_SIZE]
                    if mocap_ts_ns is not None:
                        mocap_dt_s = (mocap_ts_ns - self.start_time_ns) / 1e9

            if mocap_data is None:
                row.extend([math.nan] * MOCAP_DATA_SIZE)
            else:
                row.extend(float(v) for v in mocap_data)
            row.append(float(mocap_dt_s))

        # Sequence config
        wave_str = ""
        if seq_config:
            # wave_type (string), seg1_psi, max_psi
            wave_str = seq_config.get("wave", "unknown")
            row.append(float(seq_config.get("seg1", 0.0)))
            row.append(float(seq_config.get("max_p", 0.0)))
        else:
            wave_str = WAVE_FUNCTION
            row.extend([0.0, 0.0])

        with self.lock:
            self.data_buffer.append(row)
            self.wave_type_buffer.append(wave_str)
            self.total_samples += 1

            need_flush = False
            if len(self.data_buffer) >= self.flush_batch_size:
                need_flush = True
            elif (
                self.last_flush_wall is None
                or (time.time() - self.last_flush_wall) >= self.flush_interval_s
            ):
                need_flush = True

            if need_flush:
                self._flush_buffer_locked()

    def stop(self, description: Optional[str] = None) -> None:
        if self.hdf5_path is None or self.exp_group_name is None:
            return

        with self.lock:
            self._flush_buffer_locked()

        try:
            with h5py.File(self.hdf5_path, "a") as f:
                if self.exp_group_name not in f:
                    logger.warning(
                        f"Group {self.exp_group_name} not found in {self.hdf5_path}"
                    )
                    return
                grp = f[self.exp_group_name]
                grp.attrs["sample_count"] = int(self.total_samples)
                if description:
                    grp.attrs["description"] = description
                else:
                    grp.attrs.setdefault("description", "No description provided")

        except Exception as e:
            logger.error(f"Error saving HDF5 metadata: {e}")

    # ---------------- AI description ----------------

    def generate_ai_description(self) -> str:
        if not (GEMINI_AVAILABLE and USE_GEMINI_AUTO_DESCRIPTION and GEMINI_API_KEY):
            return "No description provided"
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            model = "gemini-2.5-flash"
            prompt = f"""
                Generate a 1-sentence experiment description (max 20 words) for:
                Wave type: {WAVE_FUNCTION}
                Duration: {int(EXPERIMENT_DURATION)}s
                Arduinos: {ARDUINO_IDS}
                Target pressures: {TARGET_PRESSURES} PSI
                Mocap enabled: {"Yes" if USE_MOCAP and HAS_ZMQ else "No"}
            """
            if WAVE_FUNCTION == "sequence":
                prompt += f"""
                Sequence Details:
                - Wave Types: {SEQ_WAVE_TYPES}
                - Seg1 Pressures: {SEQ_SEG1_PRESSURES} PSI
                - Max Pressures: {SEQ_MAX_PRESSURES} PSI
                - Wave Duration: {SEQ_WAVE_DURATION}s per wave
                - Total items: {len(SEQ_WAVE_TYPES) * len(SEQ_SEG1_PRESSURES) * len(SEQ_MAX_PRESSURES)}
                The experiment runs through all combinations of these parameters.
                """
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            logger.error(f"Gemini description failed: {e}")
            return "No description provided"


# =====================================================================
# Controller
# =====================================================================


class Controller:
    def __init__(
        self,
        refill_mode: str = "periodic",
        refill_period: float = SEQ_SEG1_REFILL_PERIOD,
    ) -> None:
        self.arduinos = ArduinoManager()
        self.logger = DataLogger()
        self.mocap: Optional[MocapManager] = None

        self.refill_mode = refill_mode
        self.refill_period = refill_period

        self.running = False

        # Shared data
        self.data_lock = threading.Lock()
        self.desired: List[float] = [0.0 for _ in ARDUINO_IDS]
        self.measured: List[List[float]] = [[math.nan] * 4 for _ in ARDUINO_IDS]
        self.current_seq_config = None

        # Threads
        self.wave_thread: Optional[threading.Thread] = None
        self.log_thread: Optional[threading.Thread] = None
        self.progress_thread: Optional[threading.Thread] = None

        # Plotting
        self.live_plotter = None
        if USE_LIVE_PLOT and HAS_MPL:
            self._setup_plot()

    # ---------------- init / teardown ----------------

    def _setup_plot(self) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        self.live_plotter = {"fig": fig, "axes": axes}

        axes[0].set_ylabel("Pressure (PSI)")
        axes[0].set_title("Segment 1 (Pouches 1-5)")

        axes[1].set_ylabel("Pressure (PSI)")
        axes[1].set_title("Segments 2-4")

        axes[2].set_xlabel("X Position")
        axes[2].set_ylabel("Z Position")
        axes[2].set_title("Robot Trajectory (X-Z)")

        for ax in axes:
            ax.grid(True)

        fig.tight_layout()

    def initialize(self) -> bool:
        logger.info("Connecting Arduinos...")
        self.arduinos.connect()
        logger.info("Arduinos connected.")

        if USE_MOCAP and HAS_ZMQ:
            self.mocap = MocapManager()
            if self.mocap.connect():
                self.mocap.start()
            else:
                logger.warning("Mocap requested but unavailable; continuing without.")
                self.mocap = None

        self.logger.start()
        self.running = True

        # Start threads
        self.wave_thread = threading.Thread(target=self._wave_loop, daemon=True)
        self.wave_thread.start()

        self.log_thread = threading.Thread(target=self._log_loop, daemon=True)
        self.log_thread.start()

        self.progress_thread = threading.Thread(target=self._progress_loop, daemon=True)
        self.progress_thread.start()

        return True

    # ---------------- wave & logging loops ----------------

    def _wave_loop(self) -> None:
        """
        Generates desired pressures and communicates with Arduinos.
        Supports "sequence" mode to iterate through combinations of parameters.
        """
        start_time = time.perf_counter()

        # If running a sequence, build the playlist
        playlist = []
        if WAVE_FUNCTION == "sequence":
            # Track time for refill logic
            simulated_time = 0.0
            next_refill_time = (
                self.refill_period
            )  # Target time for next refill (USER CONFIG)

            def add_wave_item(wave_type, seg1_p, max_p, duration):
                nonlocal simulated_time, next_refill_time

                # Check if this item crosses the next refill time
                # We need to loop because a very long wave might need multiple refills (unlikely but possible)
                remaining_duration = duration
                current_wave_offset = 0.0

                # If mode is end_of_wave, we do NOT split. refill_period is inf anyway.
                # If periodic, we split.

                while remaining_duration > 0:
                    time_until_refill = next_refill_time - simulated_time

                    if self.refill_mode == "periodic":
                        if (
                            time_until_refill <= 0
                            or time_until_refill > remaining_duration
                        ):
                            # Case 1: Refill is far off or we are skipping it (if inf)
                            playlist.append(
                                {
                                    "wave": wave_type,
                                    "seg1": seg1_p,
                                    "max_p": max_p,
                                    "duration": remaining_duration,
                                    "offset": current_wave_offset,
                                }
                            )
                            simulated_time += remaining_duration
                            remaining_duration = 0
                        else:
                            # Case 2: Refill happens during this item
                            chunk_dur = time_until_refill

                            if chunk_dur > 0.1:
                                playlist.append(
                                    {
                                        "wave": wave_type,
                                        "seg1": seg1_p,
                                        "max_p": max_p,
                                        "duration": chunk_dur,
                                        "offset": current_wave_offset,
                                    }
                                )
                                simulated_time += chunk_dur
                                current_wave_offset += chunk_dur
                                remaining_duration -= chunk_dur

                            # Add Refill Pause
                            playlist.append(
                                {
                                    "wave": "refill_pause",
                                    "seg1": seg1_p,
                                    "max_p": 0.0,
                                    "duration": SEQ_REFILL_ACTION_DURATION,
                                    "offset": 0.0,
                                }
                            )
                            simulated_time += SEQ_REFILL_ACTION_DURATION
                            next_refill_time += self.refill_period
                    else:  # refill_mode == "end_of_wave" OR "constant_2psi"
                        # No splitting
                        playlist.append(
                            {
                                "wave": wave_type,
                                "seg1": seg1_p,
                                "max_p": max_p,
                                "duration": remaining_duration,
                                "offset": current_wave_offset,
                            }
                        )
                        simulated_time += remaining_duration
                        remaining_duration = 0

            # Initial Prefill
            playlist.append(
                {
                    "wave": "cooldown",  # Re-use cooldown logic (all 2.0 psi)
                    "seg1": 2.0,
                    "max_p": 0.0,
                    "duration": 10.0,  # Initial prefill duration
                    "offset": 0.0,
                }
            )
            simulated_time += 10.0
            # Reset next refill target
            next_refill_time = simulated_time + self.refill_period

            for seg1_p in SEQ_SEG1_PRESSURES:
                for max_p in SEQ_MAX_PRESSURES:
                    for wave_type in SEQ_WAVE_TYPES:
                        # Add wave
                        add_wave_item(wave_type, seg1_p, max_p, SEQ_WAVE_DURATION)

                        # Logic specific: If End of Wave Profile, add refill NOW
                        if self.refill_mode == "end_of_wave":
                            playlist.append(
                                {
                                    "wave": "refill_pause",
                                    "seg1": seg1_p,
                                    "max_p": 0.0,
                                    "duration": SEQ_REFILL_ACTION_DURATION,
                                    "offset": 0.0,
                                }
                            )

                        # Add cooldown after each wave
                        add_wave_item("cooldown", 0.0, 0.0, SEQ_COOLDOWN_DURATION)

        else:
            # Single mode (legacy behavior but using the new loop structure)
            playlist.append(
                {
                    "wave": WAVE_FUNCTION,
                    "seg1": TARGET_PRESSURES[0],  # Default from config
                    "max_p": 10.0,  # Default max
                    "duration": float("inf"),
                    "offset": 0.0,
                }
            )

        current_item_idx = 0
        item_start_time = time.perf_counter()

        next_wake_time = time.perf_counter()

        # Track last pressures for holding during pauses
        last_wave_pressures = [0.0] * len(ARDUINO_IDS)

        while self.running:
            now = time.perf_counter()

            # Check if we need to advance to next item in sequence
            current_item = playlist[current_item_idx]
            time_in_item = now - item_start_time

            if time_in_item >= current_item["duration"]:
                current_item_idx += 1
                if current_item_idx >= len(playlist):
                    logger.info("Sequence complete.")
                    break
                current_item = playlist[current_item_idx]
                item_start_time = now
                time_in_item = 0.0
                logger.info(f"Starting sequence item: {current_item}")

            # Extract parameters for current step
            w_type = current_item["wave"]
            seg1_target = current_item["seg1"]
            p_max = current_item["max_p"]
            time_offset = current_item.get("offset", 0.0)

            effective_time = time_in_item + time_offset  # For phase continuity

            # Store current config for the logger
            self.current_seq_config = current_item

            desired: List[float] = [0.0] * len(ARDUINO_IDS)

            # --- Wave Generation Logic ---

            if w_type == "refill_pause":
                # Pause experiment, refill segment 1
                desired[0] = seg1_target  # Refill
                # Hold others at last known value
                for i in range(1, len(ARDUINO_IDS)):
                    desired[i] = last_wave_pressures[i]

            elif w_type == "cooldown":
                # Regular cooldown, Seg 1 is OFF (0.0) unless it was the split entry?
                # In add_wave_item call above, we passed 0.0 for seg1.
                desired = [2.0] * len(ARDUINO_IDS)
                desired[0] = seg1_target  # This will be 0.0 if we passed 0.0

            elif w_type == "axial":
                # seg1 -> 0.0 (OFF during experiment)
                # seg2 -> constant 2.0
                # seg3 -> sinusoid
                # seg4 -> constant 2.0

                AXIAL_FREQ = 0.1
                center = p_max / 2.0
                ampl = p_max / 2.0

                desired[0] = 0.0  # Turn OFF Segment 1 during wave
                desired[1] = 2.0
                desired[3] = 2.0

                val = center + ampl * math.sin(
                    2.0 * math.pi * AXIAL_FREQ * effective_time
                )
                desired[2] = max(0.0, min(p_max, val))

            elif w_type == "triangular":
                desired[0] = 0.0  # Turn OFF Segment 1 during wave

                # Triangular trajectory for segments 2, 3, 4
                T = 10.0
                t_cycle = effective_time % T
                phase_len = T / 3.0

                idx_seg2 = 1
                idx_seg3 = 2
                idx_seg4 = 3

                if t_cycle < phase_len:
                    u = t_cycle / phase_len
                    desired[idx_seg2] = p_max * (1.0 - u)
                    desired[idx_seg3] = p_max * u
                    desired[idx_seg4] = 0.0
                elif t_cycle < 2 * phase_len:
                    u = (t_cycle - phase_len) / phase_len
                    desired[idx_seg2] = 0.0
                    desired[idx_seg3] = p_max * (1.0 - u)
                    desired[idx_seg4] = p_max * u
                else:
                    u = (t_cycle - 2 * phase_len) / phase_len
                    desired[idx_seg2] = p_max * u
                    desired[idx_seg3] = 0.0
                    desired[idx_seg4] = p_max * (1.0 - u)

            elif w_type == "circular":
                desired[0] = 0.0  # Turn OFF Segment 1 during wave

                indices = [1, 2, 3]  # Segments 2, 3, 4
                freq = 0.1

                for k, idx in enumerate(indices):
                    phase = (2.0 * math.pi / 3.0) * k
                    val = (
                        (math.sin(2.0 * math.pi * freq * effective_time + phase) + 1.0)
                        / 2.0
                        * p_max
                    )
                    desired[idx] = val

            # --- Override for Constant 2 PSI Mode ---
            if self.refill_mode == "constant_2psi":
                desired[0] = 2.0

            # Save state for holding pressure during pause
            if w_type in ["axial", "triangular", "circular", "static"]:
                last_wave_pressures = list(desired)

            # elif w_type == "static":
            #     desired = [seg1_target if i == 0 else 0.0 for i in range(len(ARDUINO_IDS))]
            #     pass
            # Handled by fallback? Wait, static was missing in my large paste?
            elif w_type == "static":
                desired = [
                    seg1_target if i == 0 else 0.0 for i in range(len(ARDUINO_IDS))
                ]

            # --- Override for Constant 2 PSI Mode --- (Redundant check if placed after all types, but good)
            if self.refill_mode == "constant_2psi":
                desired[0] = 2.0

            # Save state for holding pressure during pause (again?)
            if w_type in ["axial", "triangular", "circular", "static"]:
                last_wave_pressures = list(desired)

            # Update shared state for logging
            with self.data_lock:
                self.desired = desired

            measured = self.arduinos.send_all_parallel(desired)
            with self.data_lock:
                self.measured = measured

            # Drift correction for 100Hz
            next_wake_time += 0.01
            sleep_time = next_wake_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _log_loop(self) -> None:
        """Logging thread: builds rows and hands to DataLogger."""
        step_id = 0
        next_wake_time = time.perf_counter()

        while self.running:
            if self.mocap:
                mocap_tuple = self.mocap.get_data()
            else:
                mocap_tuple = None

            with self.data_lock:
                desired_copy = list(self.desired)
                measured_copy = [list(m) for m in self.measured]
                seq_config_copy = self.current_seq_config

            self.logger.log(
                step_id, desired_copy, measured_copy, mocap_tuple, seq_config_copy
            )
            step_id += 1

            # Drift correction for 100Hz
            next_wake_time += 0.01
            sleep_time = next_wake_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _progress_loop(self) -> None:
        """Display experiment progress with elapsed and remaining time."""
        start = time.time()
        total_duration = EXPERIMENT_DURATION
        total_duration_str = f"{int(total_duration)}s"

        while self.running:
            elapsed = time.time() - start
            elapsed = max(0.0, elapsed)
            remaining = max(0.0, total_duration - elapsed)

            elapsed_str = f"{int(elapsed)}s"
            remaining_str = f"{int(remaining)}s"

            progress = min(100.0, (elapsed / total_duration) * 100.0)

            bar_length = 30
            filled = int(bar_length * progress / 100.0)
            bar = "█" * filled + "░" * (bar_length - filled)

            print(
                f"\r[{bar}] {progress:5.1f}% | "
                f"{elapsed_str}/{total_duration_str} | "
                f"remaining: {remaining_str}",
                end="",
                flush=True,
            )

            time.sleep(1.0)

        print()  # newline when finished

    # ---------------- run / stop ----------------

    def run(self) -> None:
        """Run experiment for configured duration, updating live plot if enabled."""
        if not self.running:
            return

        logger.info(f"Starting experiment for {EXPERIMENT_DURATION} seconds...")

        start = time.perf_counter()
        times: List[float] = []
        traces: List[List[float]] = [[] for _ in ARDUINO_IDS]

        mocap_x = []
        mocap_z = []

        while self.running and (time.perf_counter() - start) < EXPERIMENT_DURATION:
            # Capture mocap for plotting
            if self.mocap:
                md, _ = self.mocap.get_data()
                # 0=x, 1=y, 2=z
                if md and len(md) >= 3 and not math.isnan(md[0]):
                    mocap_x.append(md[0])
                    mocap_z.append(md[2])

            if self.live_plotter:
                now_t = time.perf_counter() - start
                with self.data_lock:
                    # Copy all sensor data
                    current_measured = [list(m) for m in self.measured]

                times.append(now_t)

                if not traces[0] or not isinstance(traces[0][0], list):
                    traces = [[[] for _ in range(4)] for _ in ARDUINO_IDS]

                for i, sensors in enumerate(current_measured):
                    for s_idx, val in enumerate(sensors):
                        traces[i][s_idx].append(val)

                axes = self.live_plotter["axes"]

                # Clear all axes
                for ax in axes:
                    ax.cla()
                    ax.grid(True)

                # --- Plot 1: Segment 1 (Pouches 1-5) ---
                # Arduino 3 (All 4 channels) + Arduino 7 (Channel 0)
                idx3 = ARDUINO_IDS.index(3) if 3 in ARDUINO_IDS else -1
                idx7 = ARDUINO_IDS.index(7) if 7 in ARDUINO_IDS else -1

                ax0 = axes[0]
                if idx3 >= 0:
                    for s in range(4):
                        if len(traces[idx3][s]) == len(times):
                            ax0.plot(times, traces[idx3][s], label=f"Pouch {s + 1}")
                if idx7 >= 0:
                    if len(traces[idx7][0]) == len(times):
                        ax0.plot(times, traces[idx7][0], label="Pouch 5")

                ax0.set_ylabel("PSI")
                ax0.set_title("Segment 1")
                ax0.legend(loc="upper right", fontsize="x-small", ncol=2)

                # --- Plot 2: Segments 2-4 ---
                # Arduino 7 (Channels 1, 2, 3)
                ax1 = axes[1]
                if idx7 >= 0:
                    labels = ["Segment 2", "Segment 3", "Segment 4"]
                    for i, lbl in enumerate(labels, start=1):
                        if len(traces[idx7][i]) == len(times):
                            ax1.plot(times, traces[idx7][i], label=lbl)

                ax1.set_ylabel("PSI")
                ax1.set_title("Segments 2-4")
                ax1.legend(loc="upper right", fontsize="small")

                # --- Plot 3: Robot Trajectory (X-Z) ---
                ax2 = axes[2]
                if mocap_x:
                    ax2.plot(mocap_x, mocap_z, "b-")
                    ax2.plot(mocap_x[-1], mocap_z[-1], "ro")  # Current pos

                ax2.set_xlabel("X (m)")
                ax2.set_ylabel("Z (m)")
                ax2.set_title("Trajectory (Top View)")
                ax2.set_aspect("equal", adjustable="datalim")

                plt.pause(0.05)
            else:
                time.sleep(0.1)

            # Check if wave thread has finished (sequence complete)
            if self.wave_thread and not self.wave_thread.is_alive():
                logger.info("Wave sequence finished early.")
                break

        logger.info("Experiment duration reached, ramping down...")
        self.stop()

    def stop(self) -> None:
        if not self.running:
            return

        # Signal threads to stop their loops
        self.running = False

        # Ramp down pressures safely
        try:
            self.arduinos.ramp_down(RAMPDOWN_DURATION)
        except Exception as e:
            logger.error(f"Error during ramp-down: {e}")

        # Join threads
        if self.wave_thread:
            self.wave_thread.join(timeout=1.0)
        if self.log_thread:
            self.log_thread.join(timeout=1.0)
        if self.progress_thread:
            self.progress_thread.join(timeout=1.0)

        if self.mocap:
            self.mocap.stop()

        if self.live_plotter:
            try:
                plt.close(self.live_plotter["fig"])
            except Exception:
                pass
            self.live_plotter = None

        # ---------------- Description        # Prompt for description or use AI
        description = None
        has_api_key = GEMINI_API_KEY and len(GEMINI_API_KEY) > 0

        if USE_GEMINI_AUTO_DESCRIPTION and GEMINI_AVAILABLE and has_api_key:
            # AI is available - generate description and ask for additions
            print("\n" + "=" * 60)
            print("Generating AI description...")
            ai_description = self.logger.generate_ai_description()
            description = ai_description  # Default to AI description
            print("Generated Description: ", ai_description)
            try:
                additional_desc = input(
                    "Enter additional description (optional, press Enter to skip): "
                ).strip()
                if additional_desc:
                    # Combine descriptions
                    description = f"{ai_description}. {additional_desc}"
            except (EOFError, KeyboardInterrupt):
                print("\nSkipping additional description.")
                # `description` is already set to ai_description, so we're good

        else:
            # No AI available or configured, ask for manual description
            try:
                print("\n" + "=" * 60)
                if not has_api_key and USE_GEMINI_AUTO_DESCRIPTION:
                    print("AI descriptions enabled, but GEMINI_API_KEY is not set.")

                description = input(
                    "Enter experiment description (or press Enter to skip): "
                ).strip()
                if not description:
                    description = "No description provided"
            except (EOFError, KeyboardInterrupt):
                description = "Interrupted - no description"

        self.logger.stop(description)

        # Ask to save
        try:
            save_input = input("Save experiment data? [Y/n]: ").strip().lower()
            save = save_input not in ("n", "no")
        except (EOFError, KeyboardInterrupt):
            save = True  # Default to save on interrupt during prompt
            print("\nSaving by default.")

        self.logger.finalize(save)
        self.arduinos.cleanup()
        logger.info("Controller stopped and cleaned up.")


# =====================================================================
# Main entry
# =====================================================================


def main():
    # Prompts for Refill Options
    refill_mode = "periodic"  # "periodic", "end_of_wave", or "constant_2psi"
    refill_period = SEQ_SEG1_REFILL_PERIOD

    try:
        if WAVE_FUNCTION == "sequence":
            print("Select Refill Mode:")
            print("[1] Periodic (default)")
            print("[2] End of Wave Profile")
            print("[3] Constant 2 PSI (Seg 1)")

            ans = input("Choice [1/2/3]: ").strip()
            if ans == "2":
                refill_mode = "end_of_wave"
                print("Mode: End of Wave Profile (Refill after every wave)")
                refill_period = float("inf")  # Disable periodic check
            elif ans == "3":
                refill_mode = "constant_2psi"
                print("Mode: Constant 2 PSI on Segment 1")
                refill_period = float("inf")  # Disable periodic check
            else:
                refill_mode = "periodic"
                # Ask for period
                ans = input(
                    f"Refill period (seconds) [default: {SEQ_SEG1_REFILL_PERIOD}]: "
                ).strip()
                if ans:
                    try:
                        val = float(ans)
                        if val > 0:
                            refill_period = val
                    except ValueError:
                        print(f"Invalid input, using default: {SEQ_SEG1_REFILL_PERIOD}")
                print(f"Mode: Periodic ({refill_period}s)")

    except (EOFError, KeyboardInterrupt):
        pass  # Use defaults

    controller = Controller(refill_mode=refill_mode, refill_period=refill_period)

    def signal_handler(sig, frame):
        print("\nCtrl+C pressed. Stopping gracefully...")
        controller.running = False

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 50)
    logger.info(f"Arduinos: {ARDUINO_IDS}")
    logger.info(f"Pressures: {TARGET_PRESSURES} psi")
    logger.info(f"Wave: {WAVE_FUNCTION}")
    logger.info(f"Refill Mode: {refill_mode}")
    if refill_mode == "periodic":
        logger.info(f"Refill Period: {refill_period}s")
    logger.info(f"Mocap: {'ON' if USE_MOCAP and HAS_ZMQ else 'OFF'}")
    logger.info(
        f"AI Descriptions: {'ON' if USE_GEMINI_AUTO_DESCRIPTION and GEMINI_AVAILABLE and GEMINI_API_KEY else 'OFF'}"
    )
    logger.info(f"Rampdown: {RAMPDOWN_DURATION}s")
    logger.info("=" * 50)

    try:
        if not controller.initialize():
            return 1
        controller.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
