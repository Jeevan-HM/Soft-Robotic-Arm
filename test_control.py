"""
Control Verification Tool for Segment 1 Test
Tests Arduinos 1, 2, 3, 4, 5 by applying a pressure step and monitoring response.
"""

import logging
import signal
import socket
import struct
import sys
import threading
import time
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configuration
PC_ADDRESS = "0.0.0.0"  # Listen on all interfaces
ARDUINO_IDS = [1, 2, 3]
# Port mapping: ID 1 -> 10001, etc.
ARDUINO_PORTS = {
    1: 10001,
    2: 10002,
    3: 10003,
}

TEST_DURATION = 20.0  # Total test time
STEP_PRESSURE = 3.0  # Pressure to apply
STEP_START = 3.0  # Time to start applying pressure
STEP_END = 15.0  # Time to stop applying pressure


class ArduinoConnection:
    """Manages connection and communication with an Arduino"""

    def __init__(self, arduino_id: int):
        self.arduino_id = arduino_id
        self.port = ARDUINO_PORTS[arduino_id]
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((PC_ADDRESS, self.port))
            server.listen(1)
            server.settimeout(5.0)
            logger.info(f"Waiting for Arduino {self.arduino_id} on port {self.port}...")

            self.socket, addr = server.accept()
            self.socket.settimeout(2.0)
            self.connected = True
            logger.info(f"Arduino {self.arduino_id} connected from {addr}")
            server.close()
            return True
        except Exception as e:
            logger.error(f"Failed to connect Arduino {self.arduino_id}: {e}")
            return False

    def send_pressure_read_sensors(self, pressure: float) -> List[float]:
        if not self.socket or not self.connected:
            return [0.0] * 4
        try:
            self.socket.send(struct.pack("f", float(pressure)))

            # Expect 8 bytes (4 * int16)
            data = b""
            while len(data) < 8:
                chunk = self.socket.recv(8 - len(data))
                if not chunk:
                    raise ConnectionError("Connection lost")
                data += chunk

            raw_values = struct.unpack(">4h", data)
            # Use same conversion as main.py/test_leakage.py
            # ADS1115 GAIN_TWOTHIRDS: +/- 6.144V range.
            # 1 bit = 6.144 / 32768 = 0.0001875 V
            # Pressure Sensor: 0.5V=0psi, 4.5V=30psi.
            # psi = (volts - 0.5) * 30 / 4

            results = []
            for raw in raw_values:
                voltage = raw * (6.144 / 32768.0)
                psi = ((30.0 - 0.0) * (voltage - 0.5)) / 4.0
                results.append(round(psi, 3))
            return results

        except Exception as e:
            logger.error(f"Error with Arduino {self.arduino_id}: {e}")
            self.connected = False
            return [0.0] * 4

    def close(self):
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False


class ControlTest:
    def __init__(self):
        self.arduinos = [ArduinoConnection(aid) for aid in ARDUINO_IDS]
        self.running = False

    def setup(self):
        logger.info("Setting up connections...")
        # Sequential connect (could be parallel but seq is safer for debugging)
        for a in self.arduinos:
            if not a.connect():
                logger.error("Aborting test due to connection failure.")
                return False
        return True

    def run(self):
        self.running = True
        start_time = time.time()

        logger.info("\nSTARTING CONTROL TEST")
        logger.info(
            f"Applying {STEP_PRESSURE} PSI from t={STEP_START}s to t={STEP_END}s"
        )
        logger.info("Format: Time | A1_S1 | A2_S1 | A3_S1 | A4_S1 | A5_S1")
        logger.info("-" * 60)

        try:
            while self.running:
                now = time.time()
                elapsed = now - start_time

                if elapsed > TEST_DURATION:
                    break

                # Determine target pressure
                target = 0.0
                if STEP_START <= elapsed <= STEP_END:
                    target = STEP_PRESSURE

                # Send and log
                row_str = f"{elapsed:5.1f}s | "

                for a in self.arduinos:
                    sensors = a.send_pressure_read_sensors(target)
                    # Show first sensor only (assuming it's the feedback one)
                    # Or show all? Space is limited. Let's show S1 only.
                    val = sensors[0] if sensors else 0.0
                    row_str += f"{val:5.2f} | "

                print(f"\r{row_str}", end="", flush=True)

                time.sleep(0.1)  # 10Hz

        except KeyboardInterrupt:
            pass
        finally:
            print()  # Newline
            self.stop()

    def stop(self):
        self.running = False
        logger.info("Stopping...")
        for a in self.arduinos:
            a.send_pressure_read_sensors(0.0)  # Safe state
            a.close()
        logger.info("Cleanup complete.")


def main():
    test = ControlTest()

    def signal_handler(sig, frame):
        print("\nCtrl+C pressed.")
        test.running = False

    signal.signal(signal.SIGINT, signal_handler)

    if test.setup():
        test.run()
    else:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
