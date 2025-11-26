"""
Calibration script for pressure vs commanded psi.

What it does:
- Connects to Arduinos using same scheme as main.py
- Sweeps commanded pressures (e.g. 0, 2, 4, 6, 8, 10 psi)
- For each commanded pressure, takes N samples from ADS1115
- Logs:
    arduino_id, command_psi, sample_index, sensor_index, raw_counts, psi_est
- Fits a line P_measured = a * P_command + b per Arduino (based on psi_est)
  and prints 'a' and 'b'.

You should:
- Optionally use an external gauge to verify psi_est vs real physical psi
- Use the fitted a,b to correct your control mapping.
"""

import csv
import logging
import math
import os
import socket
import struct
import sys
import time
from typing import List, Optional

import numpy as np

# ---------------- Configuration ----------------

PC_ADDRESS = "0.0.0.0"  # bind address for this PC

# Same IDs and ports as main.py
ARDUINO_IDS = [3, 6, 7, 8]
ARDUINO_PORTS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]

# ADS1115 config (GAIN_TWOTHIRDS => ±6.144 V)
ADS_RANGE_VOLTS = 6.144
ADS_COUNTS = 32768.0
ADS_LSB = ADS_RANGE_VOLTS / ADS_COUNTS

# Sensor mapping ASSUMED during calibration (this is what you're currently using)
SENSOR_V_ZERO = 0.5   # V at 0 psi
SENSOR_V_FULL = 4.5   # V at full scale
SENSOR_P_FULL = 30.0  # psi at full scale

# Command sweep settings
COMMAND_PSI_VALUES = [0, 2, 4, 6, 8, 10]  # psi to command for calibration
SAMPLES_PER_POINT = 50                    # number of readings per command
SAMPLE_PERIOD = 0.05                      # seconds between samples

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------- Arduino Manager (minimal) ----------------

class ArduinoManager:
    def __init__(self):
        self.server_sockets: List[socket.socket] = []
        self.client_sockets: List[socket.socket] = []

    def connect(self):
        for arduino_id in ARDUINO_IDS:
            port = ARDUINO_PORTS[arduino_id - 1]
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((PC_ADDRESS, port))
            s.listen(1)
            logger.info(f"Waiting for Arduino {arduino_id} on port {port}...")
            self.server_sockets.append(s)

            client, addr = s.accept()
            logger.info(f"Arduino {arduino_id} connected from {addr}")
            self.client_sockets.append(client)

    def send_pressure(self, idx: int, pressure_psi: float) -> List[int]:
        """
        Send commanded pressure (psi) to Arduino idx, get back 4x int16 counts.
        Returns raw ADC counts (int) from ADS1115.
        """
        try:
            sock = self.client_sockets[idx]
            sock.send(struct.pack("f", float(pressure_psi)))

            data = b""
            while len(data) < 8:
                chunk = sock.recv(8 - len(data))
                if not chunk:
                    raise ConnectionError("Arduino disconnected")
                data += chunk

            counts = list(struct.unpack(">4h", data))
            return counts
        except Exception as e:
            logger.error(f"send_pressure failed for idx {idx}: {e}")
            return [0, 0, 0, 0]

    def cleanup(self):
        for s in self.client_sockets + self.server_sockets:
            try:
                s.close()
            except:
                pass
        self.client_sockets.clear()
        self.server_sockets.clear()


# ---------------- Helper: counts -> volts -> psi_est ----------------

def counts_to_psi(counts: int) -> float:
    """
    Convert ADS1115 raw counts -> volts -> psi using current assumed mapping.
    This is the mapping you're using now (may be wrong but consistent).
    """
    volts = counts * ADS_LSB
    psi = SENSOR_P_FULL * (volts - SENSOR_V_ZERO) / (SENSOR_V_FULL - SENSOR_V_ZERO)
    return psi


# ---------------- Calibration routine ----------------

def run_calibration(output_csv: str = "calibration_data.csv"):
    """
    For each Arduino and each commanded psi value:
      - Send that psi repeatedly
      - Record SAMPLES_PER_POINT readings
      - Save to CSV
      - Fit P_measured = a * P_command + b per Arduino (using psi_est on sensor 0)
    """
    mgr = ArduinoManager()
    mgr.connect()

    # Prepare CSV
    os.makedirs("calibration_logs", exist_ok=True)
    csv_path = os.path.join("calibration_logs", output_csv)
    logger.info(f"Writing calibration data to {csv_path}")
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "arduino_id",
        "command_psi",
        "sample_idx",
        "sensor_idx",
        "raw_counts",
        "psi_est",
    ])

    # Data for fitting
    fit_data = {aid: {"cmd": [], "meas": []} for aid in ARDUINO_IDS}

    try:
        for idx, arduino_id in enumerate(ARDUINO_IDS):
            logger.info(f"=== Calibrating Arduino {arduino_id} (index {idx}) ===")

            for cmd_psi in COMMAND_PSI_VALUES:
                logger.info(f"Commanding {cmd_psi} psi on Arduino {arduino_id}")

                # Collect samples
                for s_idx in range(SAMPLES_PER_POINT):
                    counts_list = mgr.send_pressure(idx, cmd_psi)
                    # use sensor 0 as the main channel, but log all 4
                    for sensor_idx, c in enumerate(counts_list):
                        psi_est = counts_to_psi(c)
                        writer.writerow([
                            arduino_id,
                            cmd_psi,
                            s_idx,
                            sensor_idx,
                            int(c),
                            float(psi_est),
                        ])
                    # For fitting: use sensor 0 psi-estimate
                    c0 = counts_list[0]
                    psi0 = counts_to_psi(c0)
                    fit_data[arduino_id]["cmd"].append(cmd_psi)
                    fit_data[arduino_id]["meas"].append(psi0)

                    time.sleep(SAMPLE_PERIOD)

                logger.info(f"Finished {SAMPLES_PER_POINT} samples at {cmd_psi} psi.")

        f.flush()
        f.close()
        logger.info("Calibration data collection finished.")

    except KeyboardInterrupt:
        logger.warning("Calibration interrupted by user.")
        f.flush()
        f.close()
    finally:
        mgr.cleanup()

    # ---------------- Fit linear model per Arduino ----------------
    print("\n================= CALIBRATION RESULTS =================")
    print("Fitting P_measured ≈ a * P_command + b using sensor 0 psi_est\n")

    for aid in ARDUINO_IDS:
        cmd = np.array(fit_data[aid]["cmd"], dtype=float)
        meas = np.array(fit_data[aid]["meas"], dtype=float)

        if len(cmd) < 2:
            print(f"Arduino {aid}: not enough data to fit.")
            continue

        # Fit line: meas ≈ a*cmd + b
        a, b = np.polyfit(cmd, meas, 1)
        print(f"Arduino {aid}:")
        print(f"  P_measured ≈ {a:.4f} * P_command + {b:.4f}")
        print(f"  (Use this to compensate mapping or update P_MAX / sensor transfer.)")
        print()

    print("Calibration CSV saved at:", csv_path)
    print("========================================================")


if __name__ == "__main__":
    run_calibration()
