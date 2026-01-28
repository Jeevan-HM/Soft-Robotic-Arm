import logging
import socket
import struct
import threading
import time
from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LEAKAGE_PC_ADDRESS = "10.211.215.251"
LEAKAGE_ARDUINO_PORTS = [
    10001,
    10002,
    10003,
    10004,
    10005,
    10006,
    10007,
    10008,
]
LEAKAGE_STABILIZATION_TIME = 10.0
LEAKAGE_THRESHOLD = 0.5
LEAKAGE_SAMPLE_RATE = 10

logger = logging.getLogger("utilities.leakage")


class ArduinoConnection:
    """Manages connection and communication with an Arduino"""

    def __init__(self, arduino_id: int):
        self.arduino_id = arduino_id
        # Safety check for index
        idx = arduino_id - 1
        if 0 <= idx < len(LEAKAGE_ARDUINO_PORTS):
            self.port = LEAKAGE_ARDUINO_PORTS[idx]
        else:
            self.port = 0
            logger.error(f"Invalid Arduino ID {arduino_id}")
        self.socket = None

    def connect(self) -> bool:
        if self.port == 0:
            return False
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((LEAKAGE_PC_ADDRESS, self.port))
            server.listen(1)
            server.settimeout(5.0)
            logger.info(f"Waiting for Arduino {self.arduino_id} on port {self.port}...")
            self.socket, addr = server.accept()
            self.socket.settimeout(2.0)
            logger.info(f"Arduino {self.arduino_id} connected from {addr}")
            server.close()
            return True
        except Exception as e:
            logger.error(f"Failed to connect Arduino {self.arduino_id}: {e}")
            return False

    def send_pressure_read_sensors(self, pressure: float) -> List[float]:
        if not self.socket:
            return [0.0] * 4
        try:
            self.socket.send(struct.pack("f", pressure))
            data = self._receive_full(8)
            raw_values = struct.unpack(">4h", data)
            return [self._convert_to_psi(v) for v in raw_values]
        except Exception as e:
            logger.error(f"Error with Arduino {self.arduino_id}: {e}")
            return [0.0] * 4

    def _receive_full(self, size: int) -> bytes:
        data = b""
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Connection lost")
            data += chunk
        return data

    def _convert_to_psi(self, raw: int) -> float:
        voltage = raw * (12.288 / 65536.0)
        psi = ((30.0 - 0.0) * (voltage - 0.5)) / 4.0
        return round(psi, 4)

    def close(self):
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


class LeakageTest:
    def __init__(
        self, pump_ids: Union[int, List[int]], sensor_id: int, target_pressure: float
    ):
        self.pump_ids = [pump_ids] if isinstance(pump_ids, int) else pump_ids
        self.sensor_id = sensor_id
        self.target_pressure = target_pressure

        self.pumps = [ArduinoConnection(pid) for pid in self.pump_ids]
        # Identify sensor Arduino
        self.sensor = next((p for p in self.pumps if p.arduino_id == sensor_id), None)
        if self.sensor is None:
            self.sensor = ArduinoConnection(sensor_id)

        self.timestamps = []
        self.sensor_data = [[] for _ in range(4)]
        self.running = False
        self.stabilization_complete = False
        self.baseline_pressure = None
        self.leakage_detected = False
        self.leakage_info = None

    def connect(self) -> bool:
        to_connect = list(self.pumps)
        if self.sensor not in to_connect:
            to_connect.append(self.sensor)
        return all(a.connect() for a in to_connect)

    def run_test(self):
        self.running = True
        start_time = time.time()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST STARTED - Pumps: {self.pump_ids} | Sensor: {self.sensor_id}")
        logger.info(f"Target: {self.target_pressure} PSI")
        logger.info(f"{'=' * 60}\n")

        while self.running:
            elapsed = time.time() - start_time
            sensors = None

            for pump in self.pumps:
                result = pump.send_pressure_read_sensors(self.target_pressure)
                if pump.arduino_id == self.sensor_id:
                    sensors = result

            if sensors is None:
                sensors = self.sensor.send_pressure_read_sensors(0.0)

            self.timestamps.append(elapsed)
            for i, val in enumerate(sensors):
                self.sensor_data[i].append(val)

            if (
                not self.stabilization_complete
                and elapsed >= LEAKAGE_STABILIZATION_TIME
            ):
                self.stabilization_complete = True
                self.baseline_pressure = [
                    sum(s[-10:]) / 10 for s in self.sensor_data if len(s) >= 10
                ]
                logger.info(
                    f"\nStabilized. Baseline: {[f'{p:.2f}' for p in self.baseline_pressure]} PSI"
                )

            if self.stabilization_complete and not self.leakage_detected:
                for i, readings in enumerate(self.sensor_data):
                    if len(readings) >= 10:
                        drop = self.baseline_pressure[i] - (sum(readings[-10:]) / 10)
                        if drop > LEAKAGE_THRESHOLD:
                            self.leakage_detected = True
                            self.leakage_info = (i + 1, elapsed, drop)
                            logger.warning(
                                f"\n⚠️ LEAKAGE on Sensor {i + 1}: {drop:.2f} PSI drop at {elapsed:.1f}s"
                            )

            time.sleep(1.0 / LEAKAGE_SAMPLE_RATE)

        self._print_results()

    def _print_results(self):
        logger.info(f"\n{'=' * 60}")
        logger.info("TEST RESULTS")
        if self.timestamps:
            logger.info(f"Duration: {self.timestamps[-1]:.1f}s")

        if self.leakage_detected:
            sensor_num, leak_time, drop = self.leakage_info
            logger.info(
                f"❌ LEAKAGE DETECTED - Sensor {sensor_num} at {leak_time:.1f}s ({drop:.2f} PSI drop)"
            )
        else:
            logger.info("✓ NO LEAKAGE DETECTED")

        logger.info("\nFinal Readings:")
        for i, readings in enumerate(self.sensor_data):
            if readings:
                final = readings[-1]
                drop = (
                    self.baseline_pressure[i] - final if self.baseline_pressure else 0
                )
                logger.info(f"  Sensor {i + 1}: {final:.2f} PSI (drop: {drop:.2f})")
        logger.info(f"{'=' * 60}\n")

    def start(self):
        threading.Thread(target=self.run_test, daemon=True).start()

    def stop(self):
        self.running = False

    def cleanup(self):
        self.stop()
        time.sleep(0.5)
        for pump in self.pumps:
            try:
                pump.send_pressure_read_sensors(0.0)
            except:
                pass
            pump.close()
        if self.sensor not in self.pumps:
            self.sensor.close()


class RealtimePlotter:
    def __init__(self, test: LeakageTest):
        self.test = test
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.lines = [
            self.ax.plot([], [], label=f"Sensor {i + 1}", linewidth=2)[0]
            for i in range(4)
        ]
        self.threshold_line = None
        self.ani = None

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pressure (PSI)")
        self.ax.set_title(
            f"Leakage Test - Pumps: {test.pump_ids} | Sensor: {test.sensor_id}"
        )
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def update(self, frame):
        if not self.test.timestamps:
            return self.lines

        for i, line in enumerate(self.lines):
            line.set_data(self.test.timestamps, self.test.sensor_data[i])

        self.ax.set_xlim(0, max(10, self.test.timestamps[-1] + 2))
        all_vals = [v for s in self.test.sensor_data for v in s if v > 0]
        if all_vals:
            self.ax.set_ylim(max(0, min(all_vals) - 2), max(all_vals) + 2)

        if (
            self.test.stabilization_complete
            and not self.threshold_line
            and self.test.baseline_pressure
        ):
            threshold_y = min(self.test.baseline_pressure) - LEAKAGE_THRESHOLD
            self.threshold_line = self.ax.axhline(
                y=threshold_y,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Threshold",
            )
            self.ax.legend()

        if self.test.leakage_detected and self.test.leakage_info:
            self.ax.axvline(
                x=self.test.leakage_info[1],
                color="red",
                linestyle=":",
                linewidth=2,
                alpha=0.5,
            )

        return self.lines

    def show(self):
        self.ani = FuncAnimation(
            self.fig, self.update, interval=100, blit=False, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
