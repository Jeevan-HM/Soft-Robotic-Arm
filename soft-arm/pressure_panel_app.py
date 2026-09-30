"""Editable pressure controls and live pressure plots for the SOFA arm."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import socket
import tempfile
import time
import tkinter as tk
from tkinter import ttk

# Keep Matplotlib's font/cache files out of the user's home directory.  This
# also avoids a first-launch permissions warning when the panel is started by
# run.sh from SOFA's bundled Python environment.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "soft-arm-matplotlib")
)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PressurePanelApp:
    COLUMN_LABELS = ("East", "North", "West", "South")
    N_LEVELS = 5
    INITIAL_PLOT_MAX_PSI = 10.0
    INITIAL_TIME_WINDOW_SECONDS = 1.0
    HISTORY_SECONDS = 20.0
    UPDATE_MS = 50

    def __init__(self, port: int) -> None:
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.setblocking(False)

        self.root = tk.Tk()
        self.root.title("Soft Arm Pressure Control")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.pre_inflation = tk.StringVar(value="1.5")
        self.preset = tk.StringVar(value="3.0")
        self.pressures = [
            [tk.StringVar(value="0.0") for _ in range(self.N_LEVELS)]
            for _ in self.COLUMN_LABELS
        ]
        self.status = tk.StringVar(value="Waiting for live pressure from SOFA")
        self._last_feedback_time = 0.0
        self._latest_simulation_time = 0.0
        self._sample_count = 0
        self._history_time: deque[float] = deque(maxlen=500)
        self._history_pressure = [
            [deque(maxlen=500) for _ in range(self.N_LEVELS)]
            for _ in self.COLUMN_LABELS
        ]

        self._build_ui()
        self.root.after(50, self._update_periodically)

    def _pressure_entry(self, parent, variable, width=7):
        widget = ttk.Entry(
            parent,
            textvariable=variable,
            width=width,
            justify="center",
        )
        widget.bind("<Return>", lambda _event: self.publish())
        widget.bind("<FocusOut>", lambda _event: self.publish())
        return widget

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        controls = ttk.Frame(main, padding=(2, 2, 14, 2))
        controls.grid(row=0, column=0, sticky="n")
        self._build_controls(controls)

        plot_frame = ttk.LabelFrame(main, text="Live pouch pressure [psi]", padding=6)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self._build_plots(plot_frame)

    def _build_controls(self, frame) -> None:
        ttk.Label(frame, text="Soft Arm Pressure Control", font=("Helvetica", 17, "bold")).grid(
            row=0, column=0, columnspan=5, pady=(0, 10)
        )
        ttk.Label(frame, text="Enter non-negative pressure in PSI (no software maximum)").grid(
            row=1, column=0, columnspan=5, pady=(0, 12)
        )

        ttk.Label(frame, text="Pre-inflation").grid(row=2, column=0, sticky="e", padx=(0, 6))
        self._pressure_entry(frame, self.pre_inflation).grid(row=2, column=1, sticky="w")
        ttk.Label(frame, text="Fill preset").grid(row=3, column=0, sticky="e", padx=(0, 6), pady=(6, 0))
        self._pressure_entry(frame, self.preset).grid(row=3, column=1, sticky="w", pady=(6, 0))
        ttk.Button(frame, text="Fill all", command=self.fill_all).grid(row=2, column=2, padx=(12, 4))
        ttk.Button(frame, text="VENT ALL", command=self.vent_all).grid(row=3, column=2, padx=(12, 4), pady=(6, 0))

        ttk.Separator(frame, orient="horizontal").grid(
            row=4, column=0, columnspan=5, sticky="ew", pady=12
        )

        ttk.Label(frame, text="Pouch").grid(row=5, column=0, padx=(0, 10))
        for col, label in enumerate(self.COLUMN_LABELS):
            ttk.Label(frame, text=label, font=("Helvetica", 12, "bold")).grid(
                row=5, column=col + 1, padx=5
            )

        for level in range(self.N_LEVELS):
            location = "mount" if level == 0 else "tip" if level == 4 else ""
            text = f"Level {level}" + (f" ({location})" if location else "")
            ttk.Label(frame, text=text).grid(row=6 + level, column=0, sticky="e", padx=(0, 8), pady=4)
            for col in range(len(self.COLUMN_LABELS)):
                self._pressure_entry(frame, self.pressures[col][level]).grid(
                    row=6 + level, column=col + 1, padx=5, pady=4
                )

        fill_row = 6 + self.N_LEVELS
        ttk.Label(frame, text="Fill column").grid(row=fill_row, column=0, sticky="e", padx=(0, 8), pady=(8, 0))
        for col, label in enumerate(self.COLUMN_LABELS):
            ttk.Button(
                frame,
                text=f"Fill {label}",
                command=lambda c=col: self.fill_column(c),
            ).grid(row=fill_row, column=col + 1, padx=5, pady=(8, 0))

        ttk.Separator(frame, orient="horizontal").grid(
            row=fill_row + 1, column=0, columnspan=5, sticky="ew", pady=(14, 8)
        )
        ttk.Label(frame, textvariable=self.status).grid(
            row=fill_row + 2, column=0, columnspan=5
        )
        ttk.Label(
            frame,
            text="Plots show filtered pressure reported by SOFA.\n"
            "Pre-inflation is added to every pouch command.",
            justify="center",
        ).grid(row=fill_row + 3, column=0, columnspan=5, pady=(4, 0))

    def _build_plots(self, parent) -> None:
        self.figure = Figure(figsize=(7.8, 6.2), dpi=100, constrained_layout=True)
        axes = self.figure.subplots(2, 2, sharex=True, sharey=True)
        self.axes = list(axes.flat)
        self.plot_lines = []

        colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
        for col, (axis, label) in enumerate(zip(self.axes, self.COLUMN_LABELS)):
            axis.set_title(f"{label} column", fontsize=11, fontweight="bold")
            axis.set_xlim(0.0, self.INITIAL_TIME_WINDOW_SECONDS)
            axis.set_ylim(0.0, self.INITIAL_PLOT_MAX_PSI)
            axis.set_ylabel("Pressure [psi]")
            axis.grid(True, alpha=0.28)
            lines = []
            for level in range(self.N_LEVELS):
                line, = axis.plot(
                    [],
                    [],
                    color=colors[level],
                    linewidth=1.8,
                    marker=".",
                    markersize=3,
                    label=f"L{level}",
                )
                lines.append(line)
            axis.legend(loc="upper right", ncol=5, fontsize=7, framealpha=0.85)
            self.plot_lines.append(lines)

        for axis in self.axes[2:]:
            axis.set_xlabel("Simulation time [s]")

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _number(self, variable: tk.StringVar) -> float:
        try:
            value = float(variable.get())
        except ValueError:
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return max(0.0, value)

    def payload(self) -> dict:
        return {
            "pre_inflation_psi": self._number(self.pre_inflation),
            "commands": [
                [self._number(value) for value in column]
                for column in self.pressures
            ],
        }

    def publish(self) -> None:
        packet = json.dumps(self.payload(), separators=(",", ":")).encode("utf-8")
        try:
            self.socket.sendto(packet, ("127.0.0.1", self.port))
        except OSError as exc:
            self.status.set(f"Connection error: {exc}")

    def _receive_feedback(self) -> bool:
        received = False
        while True:
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                return False
            try:
                payload = json.loads(packet.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if self._append_feedback(payload):
                received = True
        return received

    def _append_feedback(self, payload: dict) -> bool:
        try:
            timestamp = float(payload["simulation_time"])
            actual = payload["actual_psi"]
            if len(actual) != len(self.COLUMN_LABELS):
                return False
            parsed = []
            for column in actual:
                if len(column) != self.N_LEVELS:
                    return False
                parsed_column = [float(value) for value in column]
                if not all(math.isfinite(value) for value in parsed_column):
                    return False
                parsed.append([max(0.0, value) for value in parsed_column])
        except (KeyError, TypeError, ValueError):
            return False

        if self._history_time and timestamp < self._history_time[-1]:
            self._clear_history()
        self._history_time.append(timestamp)
        self._latest_simulation_time = timestamp
        for col in range(len(self.COLUMN_LABELS)):
            for level in range(self.N_LEVELS):
                self._history_pressure[col][level].append(parsed[col][level])
        self._sample_count += 1
        self._last_feedback_time = time.monotonic()
        return True

    def _clear_history(self) -> None:
        self._history_time.clear()
        for column in self._history_pressure:
            for level in column:
                level.clear()

    def _refresh_plots(self) -> None:
        if not self._history_time:
            return

        times = list(self._history_time)
        latest_time = times[-1]
        if latest_time < self.HISTORY_SECONDS:
            minimum_time = 0.0
            maximum_time = max(
                self.INITIAL_TIME_WINDOW_SECONDS,
                latest_time * 1.1,
            )
        else:
            minimum_time = latest_time - self.HISTORY_SECONDS
            maximum_time = latest_time
        first_visible = next(
            (index for index, value in enumerate(times) if value >= minimum_time), 0
        )
        visible_times = times[first_visible:]
        visible_peak = 0.0

        for col, axis in enumerate(self.axes):
            axis.set_xlim(minimum_time, maximum_time)
            for level, line in enumerate(self.plot_lines[col]):
                values = list(self._history_pressure[col][level])[first_visible:]
                line.set_data(visible_times, values)
                if values:
                    visible_peak = max(visible_peak, max(values))

        plot_max = max(self.INITIAL_PLOT_MAX_PSI, visible_peak * 1.1)
        for axis in self.axes:
            axis.set_ylim(0.0, plot_max)
        # This callback already runs on Tk's UI thread.  Draw synchronously so
        # every received feedback batch becomes visible immediately.
        self.canvas.draw()

    def _update_periodically(self) -> None:
        self.publish()
        received = self._receive_feedback()
        if received:
            self._refresh_plots()

        if time.monotonic() - self._last_feedback_time < 0.75:
            self.status.set(
                f"LIVE — simulation t={self._latest_simulation_time:.3f} s "
                f"({self._sample_count} samples)"
            )
        else:
            self.status.set("Waiting for SOFA feedback — press Play in the simulation")
        self.root.after(self.UPDATE_MS, self._update_periodically)

    def fill_column(self, column: int) -> None:
        value = f"{self._number(self.preset):g}"
        for pressure in self.pressures[column]:
            pressure.set(value)
        self.publish()

    def fill_all(self) -> None:
        value = f"{self._number(self.preset):g}"
        for column in self.pressures:
            for pressure in column:
                pressure.set(value)
        self.publish()

    def vent_all(self) -> None:
        self.pre_inflation.set("0.0")
        for column in self.pressures:
            for pressure in column:
                pressure.set("0.0")
        self.publish()

    def close(self) -> None:
        self.vent_all()
        self.socket.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=47631)
    args = parser.parse_args()
    PressurePanelApp(args.port).run()


if __name__ == "__main__":
    main()
