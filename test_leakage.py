"""
Simplified Pouch Leakage Test Tool

Tests soft robot pouches for leakage by applying pressure and monitoring sensor readings.
"""

import logging
import signal
import socket
import struct
import sys
import threading
import time
from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
PC_ADDRESS = "10.211.215.251"
ARDUINO_PORTS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]
STABILIZATION_TIME = 10.0  # seconds
LEAKAGE_THRESHOLD = 0.5  # PSI
SAMPLE_RATE = 10  # Hz


class ArduinoConnection:
    """Manages connection and communication with an Arduino"""

    def __init__(self, arduino_id: int):
        self.arduino_id = arduino_id
        self.port = ARDUINO_PORTS[arduino_id - 1]
        self.socket = None

    def connect(self) -> bool:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((PC_ADDRESS, self.port))
            server.listen(1)
            server.settimeout(5.0)  # Timeout for accepting connection
            logger.info(f"Waiting for Arduino {self.arduino_id} on port {self.port}...")
            self.socket, addr = server.accept()
            self.socket.settimeout(2.0)  # Add 2-second timeout
            logger.info(f"Arduino {self.arduino_id} connected from {addr}")
            server.close()
            return True
        except Exception as e:
            logger.error(f"Failed to connect Arduino {self.arduino_id}: {e}")
            return False

    def send_pressure_read_sensors(self, pressure: float) -> List[float]:
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
    """Leakage test controller"""

    def __init__(
        self, pump_ids: Union[int, List[int]], sensor_id: int, target_pressure: float
    ):
        self.pump_ids = [pump_ids] if isinstance(pump_ids, int) else pump_ids
        self.sensor_id = sensor_id
        self.target_pressure = target_pressure

        self.pumps = [ArduinoConnection(pid) for pid in self.pump_ids]
        self.sensor = next(
            (p for p in self.pumps if p.arduino_id == sensor_id),
            ArduinoConnection(sensor_id),
        )

        self.timestamps = []
        self.sensor_data = [[] for _ in range(4)]
        self.running = False
        self.stabilization_complete = False
        self.baseline_pressure = None
        self.leakage_detected = False
        self.leakage_info = None

    def connect(self) -> bool:
        arduinos = set([*self.pumps, self.sensor])
        return all(a.connect() for a in arduinos)

    def run_test(self):
        self.running = True
        start_time = time.time()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST STARTED - Pumps: {self.pump_ids} | Sensor: {self.sensor_id}")
        logger.info(f"Target: {self.target_pressure} PSI")
        logger.info(f"{'=' * 60}\n")

        while self.running:
            elapsed = time.time() - start_time

            # Send pressure commands and read sensors
            sensors = None
            for pump in self.pumps:
                result = pump.send_pressure_read_sensors(self.target_pressure)
                # Save sensor data if this pump is the sensor
                if pump.arduino_id == self.sensor_id:
                    sensors = result

            # If sensor is separate from pumps, read it
            if sensors is None:
                sensors = self.sensor.send_pressure_read_sensors(0.0)

            self.timestamps.append(elapsed)
            for i, val in enumerate(sensors):
                self.sensor_data[i].append(val)

            # Check stabilization
            if not self.stabilization_complete and elapsed >= STABILIZATION_TIME:
                self.stabilization_complete = True
                self.baseline_pressure = [
                    sum(s[-10:]) / 10 for s in self.sensor_data if len(s) >= 10
                ]
                logger.info(
                    f"\nStabilized. Baseline: {[f'{p:.2f}' for p in self.baseline_pressure]} PSI"
                )

            # Check for leakage
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

            time.sleep(1.0 / SAMPLE_RATE)

        self._print_results()

    def _print_results(self):
        logger.info(f"\n{'=' * 60}")
        logger.info("TEST RESULTS")
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
    """Real-time plot of sensor data"""

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

        # Add threshold line after stabilization
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


def main():
    print("\nPOUCH LEAKAGE TEST TOOL")
    print("=" * 60)

    # Configuration (modify as needed)
    pump_ids = [3, 6, 7, 8]
    sensor_id = 7
    target_pressure = 3.0

    test = LeakageTest(pump_ids, sensor_id, target_pressure)

    def signal_handler(sig, frame):
        logger.info("\nCtrl+C pressed. Stopping...")
        test.stop()
        plt.close("all")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        if not test.connect():
            logger.error("Connection failed. Exiting.")
            return 1

        test.start()
        plotter = RealtimePlotter(test)
        plotter.show()

    except KeyboardInterrupt:
        pass  # Handled by signal handler or exit
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return 1
    finally:
        logger.info("Cleaning up...")
        test.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
