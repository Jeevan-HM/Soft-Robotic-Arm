"""Bridge the native pressure-control window to the SOFA actuators."""

from __future__ import annotations

import json
import math
import os
import socket
import time

import Sofa.Core


class PressureControlPanel(Sofa.Core.Controller):
    """Receive a 4 x 5 PSI matrix and apply it to the pouch constraints."""

    COLUMN_LABELS = ("East", "North", "West", "South")
    DEFAULT_PORT = 47631
    COMMAND_TIMEOUT_S = 1.0
    FEEDBACK_INTERVAL_S = 0.05

    def __init__(self, cavity_nodes, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cavity_nodes = cavity_nodes
        self.cfg = cfg
        self._command_psi = [
            [0.0 for _ in range(cfg.n_levels)] for _ in range(cfg.n_cols)
        ]
        self._actual_psi = [
            [0.0 for _ in range(cfg.n_levels)] for _ in range(cfg.n_cols)
        ]
        self._pre_inflation_psi = 0.0
        self._last_command_time = 0.0
        self._feedback_address = None
        self._simulation_time = 0.0
        self._last_feedback_wall_time = 0.0
        self._port = int(os.environ.get("SOFT_ARM_PRESSURE_PORT", self.DEFAULT_PORT))

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._socket.bind(("127.0.0.1", self._port))
            connection_status = f"Listening for the pressure window on port {self._port}"
        except OSError as exc:
            connection_status = f"Pressure window connection failed: {exc}"

        self.addData(
            "instructions",
            type="string",
            value="Use the separate Soft Arm Pressure Control window to enter PSI.",
            help="The bundled SofaImGui cannot edit Python-created numeric Data.",
            group="Pressure panel connection",
        ).setReadOnly(True)
        self.addData(
            "connectionStatus",
            type="string",
            value=connection_status,
            help="Connection between the native pressure window and SOFA.",
            group="Pressure panel connection",
        ).setReadOnly(True)

        for label in self.COLUMN_LABELS[: cfg.n_cols]:
            data = self.addData(
                f"actual{label}Psi",
                type="vector<float>",
                value=[0.0] * cfg.n_levels,
                help=f"Filtered total {label} pressure, levels 0 through 4 [psi].",
                group="Live pressure readback [psi]",
            )
            data.setReadOnly(True)

    def _clamp(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("pressure must be finite")
        return max(0.0, value)

    def _receive_latest_command(self) -> None:
        latest_command = None
        while True:
            try:
                packet, sender = self._socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                return
            try:
                payload = json.loads(packet.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            latest_command = (payload, sender)

        if latest_command is None:
            if (
                self._last_command_time
                and time.monotonic() - self._last_command_time > self.COMMAND_TIMEOUT_S
            ):
                self._command_psi = [
                    [0.0 for _ in range(self.cfg.n_levels)]
                    for _ in range(self.cfg.n_cols)
                ]
                self._pre_inflation_psi = 0.0
            return

        latest_payload, sender = latest_command
        try:
            commands = latest_payload["commands"]
            if len(commands) != self.cfg.n_cols:
                return
            parsed = []
            for column in commands:
                if len(column) != self.cfg.n_levels:
                    return
                parsed.append([self._clamp(value) for value in column])
            pre_inflation = self._clamp(latest_payload["pre_inflation_psi"])
        except (KeyError, TypeError, ValueError):
            return

        self._command_psi = parsed
        self._pre_inflation_psi = pre_inflation
        self._last_command_time = time.monotonic()
        self._feedback_address = sender

    def command_matrix_psi(self) -> list[list[float]]:
        return [column.copy() for column in self._command_psi]

    def actual_matrix_psi(self) -> list[list[float]]:
        return [column.copy() for column in self._actual_psi]

    def _push_pressures_to_sofa(self) -> None:
        pascals_per_psi = 6894.76
        for col, column_nodes in enumerate(self.cavity_nodes):
            for level, (_, actuator) in enumerate(column_nodes):
                actuator.value = [
                    self._actual_psi[col][level]
                    * pascals_per_psi
                    * self.cfg.sofa_pressure_scale
                ]

        for col, label in enumerate(self.COLUMN_LABELS[: self.cfg.n_cols]):
            getattr(self, f"actual{label}Psi").value = self._actual_psi[col]

    def _send_pressure_feedback(self) -> None:
        if self._feedback_address is None:
            return
        now = time.monotonic()
        if (
            self._last_feedback_wall_time
            and now - self._last_feedback_wall_time < self.FEEDBACK_INTERVAL_S
        ):
            return

        packet = json.dumps(
            {
                "simulation_time": self._simulation_time,
                "actual_psi": self._actual_psi,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            self._socket.sendto(packet, self._feedback_address)
        except OSError:
            return
        self._last_feedback_wall_time = now

    def onAnimateBeginEvent(self, event) -> None:
        self._receive_latest_command()
        try:
            dt = float(event.get("dt", self.cfg.timestep))
        except (AttributeError, TypeError, ValueError):
            dt = self.cfg.timestep
        alpha = dt / max(self.cfg.tau_pneumatic, dt)

        for col in range(self.cfg.n_cols):
            for level in range(self.cfg.n_levels):
                target = self._clamp(
                    self._command_psi[col][level] + self._pre_inflation_psi
                )
                actual = self._actual_psi[col][level]
                self._actual_psi[col][level] = actual + alpha * (target - actual)

        self._simulation_time += max(0.0, dt)
        self._push_pressures_to_sofa()
        self._send_pressure_feedback()

    def reset(self) -> None:
        for col in range(self.cfg.n_cols):
            for level in range(self.cfg.n_levels):
                self._actual_psi[col][level] = 0.0
        self._simulation_time = 0.0
        self._last_feedback_wall_time = 0.0
        self._push_pressures_to_sofa()

    def cleanup(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass
