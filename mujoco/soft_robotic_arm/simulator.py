"""
SoftArmSim — MuJoCo digital twin of the fabric pneumatic soft arm.

Architecture: one sealed five-pouch Segment-1 reservoir plus three actively
regulated columns (Segments 2--4), across five floor levels.

    P[s, k] = pressure [psi] in column s (0=East, 1=North, 2=West, 3=South)
              at floor level k (0=top, 4=bottom).

Columns are at azimuths [0, 90, 180, 270] deg.  Pressurising column s at
azimuth phi_s elongates that side of the arm, bending the tip toward phi_s.

Usage
-----
    sim = SoftArmSim()                    # calibrated parallel robot, 2 psi charge
    obs = sim.step([3.0, 2.0, 1.0])       # Segment 2, 3, and 4 setpoints
    obs["tip_pos"]           # (3,) tip position [m]
    obs["pouch_pressures"]   # (4, 5) simulated sensor readings [psi]

The packaged calibration supplies all default mechanics, actuator, sensor,
and reservoir parameters. The measured public ``step()`` accepts exactly a
three-vector of absolute pressure setpoints for Segments 2--4. The coursework
simulator accepts either four segment pressures or a 4-by-5 pouch matrix.

Reservoir topology
------------------
The default ``topology="parallel"`` treats the five Segment-1 pouches as
independent sealed reservoirs.  ``topology="coupled"`` selects the calibrated
common-manifold response.  Observations include ``reservoir_pressures`` (5,)
and ``actuator_pressures`` (3,).  The sealed-pouch response is an empirical
deformation/sensor approximation; the calibrated slow charge/leak state may
feed the force law while the instantaneous deformation term is sensor-only.

Design notes
------------
* One control tick = 0.01 s (100 Hz), internally substeps at cfg.timestep (1 ms).
* Pneumatic response: optional static gain/bias and transport delay followed by
  a first-order filter (tau_pneumatic).
* Pressure → wrench, per level k: its 4 columns at azimuths phi_s produce
  bending moment  M[k] = gain * sum_s P[s,k] * (-sin phi_s, cos phi_s)
  applied to level k's bending joints, plus axial force
  f[k] = ext_gain * sum_s P[s,k]  on the level's slide DOF.
* Pre-inflation: scales joint stiffness/damping (symmetric → stiffness modulation).
* Sensor model (parallel topology): measured pressure = actual + curvature
  coupling (arm bending toward col s squeezes its pouches) + extension + noise.
"""

import csv
from collections import deque
from pathlib import Path

import numpy as np
import mujoco

from .model import ArmConfig, build_arm_xml


class SoftArmSim:
    def __init__(self, cfg: ArmConfig | None = None, control_hz: float = 100.0,
                 sensor_noise_psi: float | None = None,
                 curvature_coupling: float | np.ndarray | None = None,
                 extension_coupling: float | np.ndarray | None = None,
                 seed: int | None = 0, topology: str = "parallel",
                 reservoir_column: int | None = None,
                 reservoir_pressure_psi: float | np.ndarray | None = None,
                 actuator_delay_s: float | None = None,
                 actuator_pressure_gain: float | np.ndarray | None = None,
                 actuator_pressure_bias_psi: float | np.ndarray | None = None,
                 reservoir_charge_gain: float | np.ndarray | None = None,
                 reservoir_charge_bias_psi: float | np.ndarray | None = None,
                 reservoir_leak_tau_s: float | np.ndarray | None = None,
                 reservoir_fast_relaxation_fraction: float | np.ndarray | None = None,
                 reservoir_fast_relaxation_tau_s: float | np.ndarray | None = None,
                 reservoir_relaxation_delay_s: float | None = None,
                 reservoir_equalization: float | None = None,
                 reservoir_response_tau_s: float | None = None,
                 reservoir_force_feedback: bool | None = None,
                 strict_segment_commands: bool = False):
        topology = str(topology).lower()
        if topology not in ("parallel", "coupled"):
            raise ValueError("topology must be 'parallel' or 'coupled'")

        # Supplying no ArmConfig means "the physical robot": load every
        # fitted parameter and seal Segment 1.  Explicit ArmConfig instances
        # are reserved for the low-level mechanics fitter and focused tests.
        canonical_runtime = cfg is None
        calibrated: dict = {}
        if canonical_runtime:
            from .calibration import RobotCalibration

            calibration = RobotCalibration.load()
            cfg = calibration.make_arm_config()
            if reservoir_pressure_psi is None:
                reservoir_pressure_psi = 2.0
            if reservoir_column is None:
                reservoir_column = 0
            if reservoir_column != 0:
                raise ValueError(
                    "the robot calibration requires sealed Segment 1 "
                    "(reservoir_column=0)"
                )
            calibrated = calibration.simulator_kwargs(
                topology, reservoir_pressure_psi
            )
        elif reservoir_pressure_psi is None:
            reservoir_pressure_psi = 0.0

        def resolved(value, key: str, raw_default):
            if value is not None:
                return value
            return calibrated.get(key, raw_default)

        sensor_noise_psi = resolved(sensor_noise_psi, "sensor_noise_psi", 0.0)
        curvature_coupling = resolved(
            curvature_coupling, "curvature_coupling", 0.0
        )
        extension_coupling = resolved(
            extension_coupling, "extension_coupling", 0.0
        )
        actuator_delay_s = resolved(actuator_delay_s, "actuator_delay_s", 0.0)
        actuator_pressure_gain = resolved(
            actuator_pressure_gain, "actuator_pressure_gain", 1.0
        )
        actuator_pressure_bias_psi = resolved(
            actuator_pressure_bias_psi, "actuator_pressure_bias_psi", 0.0
        )
        reservoir_charge_gain = resolved(
            reservoir_charge_gain, "reservoir_charge_gain", 1.0
        )
        reservoir_charge_bias_psi = resolved(
            reservoir_charge_bias_psi, "reservoir_charge_bias_psi", 0.0
        )
        reservoir_leak_tau_s = resolved(
            reservoir_leak_tau_s, "reservoir_leak_tau_s", np.inf
        )
        reservoir_fast_relaxation_fraction = resolved(
            reservoir_fast_relaxation_fraction,
            "reservoir_fast_relaxation_fraction",
            0.0,
        )
        reservoir_fast_relaxation_tau_s = resolved(
            reservoir_fast_relaxation_tau_s,
            "reservoir_fast_relaxation_tau_s",
            1.0,
        )
        reservoir_relaxation_delay_s = resolved(
            reservoir_relaxation_delay_s,
            "reservoir_relaxation_delay_s",
            0.0,
        )
        reservoir_equalization = resolved(
            reservoir_equalization, "reservoir_equalization", 0.0
        )
        reservoir_response_tau_s = resolved(
            reservoir_response_tau_s, "reservoir_response_tau_s", 0.0
        )
        reservoir_force_feedback = resolved(
            reservoir_force_feedback, "reservoir_force_feedback", False
        )

        self.cfg = cfg
        self.topology = topology
        self.uses_robot_calibration = canonical_runtime
        self.strict_segment_commands = bool(strict_segment_commands)
        if self.strict_segment_commands and reservoir_column is not None:
            raise ValueError(
                "strict four-segment commands require reservoir_column=None"
            )
        self.control_dt = 1.0 / control_hz
        self.n_sub_steps = max(1, round(self.control_dt / self.cfg.timestep))
        self.sensor_noise = float(sensor_noise_psi)
        if not np.isfinite(self.sensor_noise) or self.sensor_noise < 0.0:
            raise ValueError("sensor_noise_psi must be finite and nonnegative")
        self.curvature_coupling = self._per_pouch_parameter(
            curvature_coupling, "curvature_coupling"
        )
        self.extension_coupling = self._per_pouch_parameter(
            extension_coupling, "extension_coupling"
        )
        self.rng = np.random.default_rng(seed)

        delay = float(actuator_delay_s)
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError("actuator_delay_s must be finite and nonnegative")
        self.actuator_delay_s = delay
        self.actuator_delay_steps = int(round(delay / self.control_dt))
        self.actuator_pressure_gain = self._per_column_parameter(
            actuator_pressure_gain,
            "actuator_pressure_gain",
            reservoir_column,
            fill=1.0,
        )
        if np.any(self.actuator_pressure_gain <= 0.0):
            raise ValueError("actuator_pressure_gain must be positive")
        self.actuator_pressure_bias = self._per_column_parameter(
            actuator_pressure_bias_psi,
            "actuator_pressure_bias_psi",
            reservoir_column,
            fill=0.0,
        )
        self._actuator_delay_queue: deque[np.ndarray] = deque()
        for _ in range(self.actuator_delay_steps):
            self._actuator_delay_queue.append(
                np.zeros((self.cfg.n_segments, self.cfg.n_pouches))
            )

        self.reservoir_charge_gain = self._per_pouch_parameter(
            reservoir_charge_gain, "reservoir_charge_gain"
        )
        self.reservoir_charge_bias = self._per_pouch_parameter(
            reservoir_charge_bias_psi, "reservoir_charge_bias_psi"
        )
        if np.any(self.reservoir_charge_gain <= 0.0):
            raise ValueError("reservoir_charge_gain must be positive")
        self.reservoir_leak_tau = self._per_pouch_parameter(
            reservoir_leak_tau_s,
            "reservoir_leak_tau_s",
            allow_infinite=True,
        )
        if np.any(self.reservoir_leak_tau <= 0.0):
            raise ValueError("reservoir_leak_tau_s must be positive")
        self.reservoir_fast_relaxation_fraction = self._per_pouch_parameter(
            reservoir_fast_relaxation_fraction,
            "reservoir_fast_relaxation_fraction",
        )
        if np.any((self.reservoir_fast_relaxation_fraction < 0.0)
                  | (self.reservoir_fast_relaxation_fraction > 1.0)):
            raise ValueError(
                "reservoir_fast_relaxation_fraction must lie in [0, 1]"
            )
        self.reservoir_fast_relaxation_tau = self._per_pouch_parameter(
            reservoir_fast_relaxation_tau_s,
            "reservoir_fast_relaxation_tau_s",
            allow_infinite=True,
        )
        if np.any(self.reservoir_fast_relaxation_tau <= 0.0):
            raise ValueError("reservoir_fast_relaxation_tau_s must be positive")
        self.reservoir_relaxation_delay = float(reservoir_relaxation_delay_s)
        if (not np.isfinite(self.reservoir_relaxation_delay)
                or self.reservoir_relaxation_delay < 0.0):
            raise ValueError(
                "reservoir_relaxation_delay_s must be finite and nonnegative"
            )
        self.reservoir_equalization = float(reservoir_equalization)
        if (not np.isfinite(self.reservoir_equalization)
                or not 0.0 <= self.reservoir_equalization <= 1.0):
            raise ValueError("reservoir_equalization must lie in [0, 1]")
        self.reservoir_response_tau = float(reservoir_response_tau_s)
        if (not np.isfinite(self.reservoir_response_tau)
                or self.reservoir_response_tau < 0.0):
            raise ValueError(
                "reservoir_response_tau_s must be finite and nonnegative"
            )
        self.reservoir_force_feedback = bool(reservoir_force_feedback)

        self.model = mujoco.MjModel.from_xml_string(build_arm_xml(self.cfg))
        self.data = mujoco.MjData(self.model)

        cfg = self.cfg

        # ── DOF lookup ────────────────────────────────────────────────────
        def dof(name):
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_dofadr[j]

        # One slide DOF per floor level (ext0 … ext4)
        self.ext_dofs = np.array([dof(f"ext{k}") for k in range(cfg.n_pouches)])

        # One (bx, by) hinge pair per floor level (bx0/by0 … bx4/by4)
        self.bend_dofs = np.array(
            [(dof(f"bx{k}"), dof(f"by{k}")) for k in range(cfg.n_pouches)]
        )  # shape (n_pouches, 2)

        self._base_stiffness = self.model.jnt_stiffness.copy()
        self._base_damping = self.model.dof_damping.copy()

        # ── Column bending axes ───────────────────────────────────────────
        # Column s at azimuth phi_s bends the arm about (-sin phi_s, cos phi_s).
        # Pressurising column s produces bending in that direction.
        phis = cfg.col_azimuths()  # (n_segments,)
        self.col_axes = np.stack([-np.sin(phis), np.cos(phis)], axis=1)  # (n_seg, 2)
        if cfg.hang_down:
            self.col_axes = -self.col_axes  # invert bending direction for downward arm

        # p_actual[s, k] = current pouch pressure in column s, floor k [psi]
        self.p_actual = np.zeros((cfg.n_segments, cfg.n_pouches))
        self.p_pre = 0.0

        # The calibrated topology charges one five-pouch column once and seals
        # it while the remaining three columns are actively regulated.
        self.reservoir_column: int | None = None
        self.reservoir_nominal_charge = np.zeros(cfg.n_pouches)
        self.reservoir_charge = np.zeros(cfg.n_pouches)
        self.reservoir_pressure = np.zeros(cfg.n_pouches)
        self._reservoir_reference_offset = np.zeros(cfg.n_pouches)
        self._reservoir_seal_time = 0.0
        if reservoir_column is not None:
            self.set_reservoir_pressure(reservoir_pressure_psi,
                                        column=reservoir_column)
        if canonical_runtime:
            self.set_pre_inflation(
                float(np.mean(np.asarray(reservoir_pressure_psi)))
            )
        mujoco.mj_forward(self.model, self.data)

        # ── Pressure logging ──────────────────────────────────────────────
        self._pressure_log: list[dict] = []   # [{time, p_cmd, p_actual}, ...]
        self._renderer = None

    def _per_pouch_parameter(self, value, name: str,
                             allow_infinite: bool = False) -> np.ndarray:
        """Return a scalar-or-five-value parameter as a validated array."""
        out = np.asarray(value, dtype=float)
        if out.ndim == 0:
            out = np.full(self.cfg.n_pouches, float(out))
        if out.shape != (self.cfg.n_pouches,):
            raise ValueError(
                f"{name} must be scalar or shape ({self.cfg.n_pouches},)"
            )
        valid = ~np.isnan(out) if allow_infinite else np.isfinite(out)
        if not np.all(valid):
            qualifier = "non-NaN" if allow_infinite else "finite"
            raise ValueError(f"{name} must contain only {qualifier} values")
        return out.copy()

    def _per_column_parameter(self, value, name: str,
                              reservoir_column: int | None,
                              fill: float) -> np.ndarray:
        """Expand scalar, four-column, or three-active-column parameters."""
        out = np.asarray(value, dtype=float)
        if out.ndim == 0:
            out = np.full(self.cfg.n_segments, float(out))
        elif out.shape == (self.cfg.n_segments - 1,) and reservoir_column is not None:
            if not 0 <= int(reservoir_column) < self.cfg.n_segments:
                raise ValueError(
                    f"reservoir_column must be in [0, {self.cfg.n_segments - 1}]"
                )
            expanded = np.full(self.cfg.n_segments, fill, dtype=float)
            active = np.arange(self.cfg.n_segments) != int(reservoir_column)
            expanded[active] = out
            out = expanded
        if out.shape != (self.cfg.n_segments,) or not np.all(np.isfinite(out)):
            raise ValueError(
                f"{name} must be finite and scalar, shape "
                f"({self.cfg.n_segments},), or shape ({self.cfg.n_segments - 1},) "
                "when a reservoir column is configured"
            )
        return out.copy()

    # ──────────────────────────────────────────────────────────── API ──────
    def reset(self, clear_log: bool = True) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.p_actual[:] = 0.0
        self._actuator_delay_queue.clear()
        for _ in range(self.actuator_delay_steps):
            self._actuator_delay_queue.append(
                np.zeros((self.cfg.n_segments, self.cfg.n_pouches))
            )
        if self.reservoir_column is not None:
            self._reservoir_seal_time = 0.0
            self.p_actual[self.reservoir_column] = self.reservoir_charge
            self.reservoir_pressure[:] = self.reservoir_charge
            self._reservoir_reference_offset = (
                self._deformation_pressure_offset()[self.reservoir_column]
            )
        mujoco.mj_forward(self.model, self.data)
        if clear_log:
            self._pressure_log.clear()
        return self.observe()

    def set_pre_inflation(self, p_pre_psi: float) -> None:
        """Stiffness-modulating pre-inflation applied to every pouch."""
        self.p_pre = float(np.clip(p_pre_psi, 0.0, self.cfg.p_max))
        scale = 1.0 + self.cfg.stiffness_per_psi * self.p_pre
        self.model.jnt_stiffness[:] = self._base_stiffness * scale
        self.model.dof_damping[:] = self._base_damping * np.sqrt(scale)

    @property
    def actuator_columns(self) -> np.ndarray:
        """Indices of actively commanded columns."""
        if self.reservoir_column is None:
            return np.arange(self.cfg.n_segments, dtype=int)
        return np.array([i for i in range(self.cfg.n_segments)
                         if i != self.reservoir_column], dtype=int)

    def set_reservoir_pressure(self, pressure_psi, column: int = 0) -> None:
        """Charge a five-pouch reservoir column and isolate it.

        ``pressure_psi`` may be a scalar or one value per pouch. Once set,
        commands for this column are ignored. Its pouch pressures vary with
        local deformation, approximating a fixed-gas sealed reservoir rather
        than a pressure source that clamps the column at the charge pressure.
        """
        if not 0 <= int(column) < self.cfg.n_segments:
            raise ValueError(
                f"reservoir column must be in [0, {self.cfg.n_segments - 1}]"
            )
        pressure = self._per_pouch_parameter(
            pressure_psi, "reservoir pressure"
        )
        self.reservoir_column = int(column)
        self._reservoir_seal_time = float(self.data.time)
        self.reservoir_nominal_charge = np.clip(
            pressure, 0.0, self.cfg.p_max
        )
        self.reservoir_charge = np.clip(
            self.reservoir_charge_gain * self.reservoir_nominal_charge
            + self.reservoir_charge_bias,
            0.0,
            self.cfg.p_max,
        )
        self._reservoir_reference_offset = (
            self._deformation_pressure_offset()[self.reservoir_column]
        )
        self.p_actual[self.reservoir_column] = self.reservoir_charge
        self.reservoir_pressure[:] = self.reservoir_charge

    def step(self, p_cmd_psi) -> dict:
        """Advance one control tick (100 Hz).

        The coursework interface accepts either one pressure per segment with
        shape ``(4,)`` or one pressure per pouch with shape ``(4, 5)``. Rows
        are Segments S1--S4 and columns are Pouches P1--P5. The measured
        calibration interface requires ``[S2, S3, S4]`` because S1 is sealed.
        """
        cfg = self.cfg
        p_cmd = np.asarray(p_cmd_psi, dtype=float)
        if not np.all(np.isfinite(p_cmd)):
            raise ValueError("pressure command must contain only finite values")

        if self.strict_segment_commands:
            segment_shape = (cfg.n_segments,)
            pouch_shape = (cfg.n_segments, cfg.n_pouches)
            if p_cmd.shape == segment_shape:
                p_cmd = np.tile(p_cmd[:, None], (1, cfg.n_pouches))
            elif p_cmd.shape != pouch_shape:
                raise ValueError(
                    "coursework pressure command must have shape (4,) for "
                    "S1--S4 or (4, 5) for every segment and pouch; "
                    f"got {p_cmd.shape}"
                )
        elif self.uses_robot_calibration:
            expected = (cfg.n_segments - 1,)
            if p_cmd.shape != expected:
                raise ValueError(
                    "calibrated pressure command must have shape (3,) for "
                    f"Segments 2, 3, and 4; got {p_cmd.shape}"
                )
            full = np.zeros((cfg.n_segments, cfg.n_pouches))
            full[self.actuator_columns] = p_cmd[:, None]
            p_cmd = full
        elif p_cmd.ndim == 1:
            if p_cmd.size == cfg.n_channels:
                # flat → reshape
                p_cmd = p_cmd.reshape(cfg.n_segments, cfg.n_pouches)
            elif (self.reservoir_column is not None
                  and p_cmd.size == cfg.n_segments - 1):
                # Three controller outputs -> the non-reservoir columns.
                full = np.zeros((cfg.n_segments, cfg.n_pouches))
                full[self.actuator_columns] = p_cmd[:, None]
                p_cmd = full
            elif p_cmd.size == cfg.n_segments:
                # one value per column → broadcast to all floors
                p_cmd = np.tile(p_cmd[:, None], (1, cfg.n_pouches))
            elif p_cmd.size == cfg.n_pouches:
                # one value per floor → broadcast to all columns
                p_cmd = np.tile(p_cmd[None, :], (cfg.n_segments, 1))

        p_cmd = p_cmd.reshape(cfg.n_segments, cfg.n_pouches)
        if self.reservoir_column is None:
            p_cmd = np.clip(p_cmd + self.p_pre, 0.0, cfg.p_max)
        else:
            p_cmd = np.clip(p_cmd, 0.0, cfg.p_max)

        # Segment 1 is isolated after charging. It is not an
        # actuator and never tracks a tick-by-tick pressure command.
        if self.reservoir_column is not None:
            p_cmd[self.reservoir_column] = self.reservoir_nominal_charge

        # The experiment's desired setpoints passed through regulator gain,
        # offset, transport delay, and first-order chamber dynamics.  The
        # raw calibration paths can explicitly select identity/no-delay.
        p_target = p_cmd.copy()
        active = self.actuator_columns
        active_cmd = p_cmd[active]
        p_target[active] = np.clip(
            self.actuator_pressure_gain[active, None] * active_cmd
            + self.actuator_pressure_bias[active, None] * (active_cmd > 1e-9),
            0.0,
            cfg.p_max,
        )
        if self.actuator_delay_steps:
            self._actuator_delay_queue.append(p_target.copy())
            p_target = self._actuator_delay_queue.popleft()

        alpha = cfg.timestep / max(cfg.tau_pneumatic, cfg.timestep)
        for _ in range(self.n_sub_steps):
            if self.reservoir_column is None:
                self.p_actual += alpha * (p_target - self.p_actual)
            else:
                active = self.actuator_columns
                self.p_actual[active] += alpha * (
                    p_target[active] - self.p_actual[active]
                )
                self._update_sealed_reservoir()
            self._apply_pressure_wrench()
            mujoco.mj_step(self.model, self.data)

        # Synchronize the sealed pressure sample with the state returned by
        # observe(); the last substep advanced q after the in-loop update.
        if self.reservoir_column is not None:
            self._update_sealed_reservoir()

        # Log the commanded and actual pressures for this tick
        logged_cmd = p_cmd.copy()
        if self.reservoir_column is not None:
            # NaN is deliberate: a sealed pouch has no runtime command and a
            # full log must never be replayed as a regulator setpoint for it.
            logged_cmd[self.reservoir_column] = np.nan
        self._pressure_log.append({
            "time": self.data.time,
            "p_cmd": logged_cmd,
            "actuator_cmd": p_cmd[self.actuator_columns].mean(axis=1).copy(),
            "p_actual": self.p_actual.copy(),  # (4, 5) actual after pneumatic lag
        })

        return self.observe()

    def observe(self) -> dict:
        s = self.data.sensordata
        obs = {
            "time": self.data.time,
            "tip_pos": s[0:3].copy(),
            "tip_quat": s[3:7].copy(),
            "tip_vel": s[7:10].copy(),
            "pouch_pressures": self._pouch_sensors(),   # (n_seg, n_pouch)
            "p_actual": self.p_actual.copy(),
            "q": self.data.qpos.copy(),
        }
        obs["segment_pressures"] = obs["pouch_pressures"].mean(axis=1).copy()
        if self.reservoir_column is not None:
            obs["reservoir_pressures"] = (
                obs["pouch_pressures"][self.reservoir_column].copy()
            )
            obs["actuator_pressures"] = (
                obs["pouch_pressures"][self.actuator_columns].mean(axis=1).copy()
            )
            obs["actuator_columns"] = self.actuator_columns.copy()
        return obs

    # ──────────────────────────────────────────── pressure log access ──────
    def clear_pressure_log(self) -> None:
        """Discard recorded commands without changing the simulation state."""
        self._pressure_log.clear()

    def get_pressure_log(self) -> dict:
        """Return the pressure log as numpy arrays.

        Returns dict with:
          time      — (N,)       timestamps [s]
          p_cmd     — (N, 4, 5)  commanded pressures [psi]; the sealed row is
                                  NaN because it has no runtime command
          actuator_cmd — (N, 3) for replayable Segment 2--4 setpoints
          p_actual  — (N, 4, 5)  actual pouch pressures after pneumatic lag [psi]
        """
        if not self._pressure_log:
            empty = lambda *s: np.empty((0, *s))
            return {"time": np.array([]),
                    "p_cmd": empty(self.cfg.n_segments, self.cfg.n_pouches),
                    "actuator_cmd": empty(len(self.actuator_columns)),
                    "p_actual": empty(self.cfg.n_segments, self.cfg.n_pouches)}
        return {
            "time": np.array([r["time"] for r in self._pressure_log]),
            "p_cmd": np.array([r["p_cmd"] for r in self._pressure_log]),
            "actuator_cmd": np.array(
                [r["actuator_cmd"] for r in self._pressure_log]
            ),
            "p_actual": np.array([r["p_actual"] for r in self._pressure_log]),
        }

    def save_pressure_log(self, path: str | Path = "pressure_log.csv",
                          which: str = "p_cmd",
                          relative_time: bool = False) -> Path:
        """Export the pressure log as a flat CSV file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        which : str
            'p_cmd' for per-pouch commanded pressures (default),
            'actuator_cmd' for one setpoint per active column, or 'p_actual'
            for pressures after pneumatic lag. With a reservoir a 'p_cmd' export
            omits sealed Segment 1 so it cannot be replayed as an air command.
        relative_time : bool
            If true, write the first retained sample at t=0. This is useful
            after ``clear_pressure_log()`` has removed a settling interval.

        CSV columns
        -----------
        Calibrated runtime: time plus Segments 2--4 (15 per-pouch values), or
        three values for ``which='actuator_cmd'``. Unsealed mechanics replay
        exports all 20 internal column-level values.
        """
        log = self.get_pressure_log()
        data = log[which]
        times = log["time"].copy()     # (N,)
        n = len(times)
        if relative_time and n:
            times -= times[0]
        header = ["time"]
        if which == "actuator_cmd":
            flat = data.reshape(n, -1)
            header.extend(f"col{s}" for s in self.actuator_columns)
        else:
            if which == "p_cmd" and self.reservoir_column is not None:
                # Export only replayable actuator setpoints.
                data = data[:, self.actuator_columns, :]
                columns = self.actuator_columns
            else:
                columns = np.arange(self.cfg.n_segments)
            flat = data.reshape(n, -1)
            for s in columns:
                for k in range(self.cfg.n_pouches):
                    header.append(f"col{s}_lv{k}")

        path = Path(path)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(n):
                writer.writerow([f"{times[i]:.6f}"] +
                                [f"{v:.4f}" for v in flat[i]])
        print(f"Saved {n} pressure records to {path}")
        return path

    # ────────────────────────────────────────────────────── internals ──────
    def _apply_pressure_wrench(self) -> None:
        """Map (4×5) pouch pressures to generalised forces on 5 level joints."""
        cfg = self.cfg
        qf = self.data.qfrc_applied
        qf[:] = 0.0
        gain = cfg.pressure_gain * cfg.moment_arm

        # m[k] = gain × Σ_s p_actual[s,k] × col_axes[s]  →  shape (n_levels, 2)
        # p_actual.T: (n_pouches, n_segments)  ×  col_axes: (n_segments, 2)
        m = gain * (self.p_actual.T @ self.col_axes)   # (n_pouches, 2)

        for k in range(cfg.n_pouches):
            bx, by = self.bend_dofs[k]
            qf[bx] += m[k, 0]
            qf[by] += m[k, 1]
            # Symmetric inflation → axial extension of this floor level
            qf[self.ext_dofs[k]] += cfg.extension_gain * self.p_actual[:, k].sum()

    def _deformation_pressure_offset(self) -> np.ndarray:
        """Empirical local pressure change caused by pouch deformation."""
        bend = self._level_curvatures()            # (n_levels, 2)
        kappa = (bend @ self.col_axes.T).T         # (n_segments, n_pouches)
        ext = self.data.qpos[self.ext_dofs][None, :]
        return (self.curvature_coupling * kappa
                + self.extension_coupling * ext)

    def _update_sealed_reservoir(self) -> None:
        """Update five isolated reservoir pressures with no regulator input.

        The rigid-link model has no pouch-volume states, so its calibrated
        deformation/pressure coupling is used as a sealed-gas compliance
        approximation. Each pouch remains an independent sensor state. Only
        the slow, dissipative calibrated charge/leak state can optionally feed
        back into qfrc_applied; the instantaneous deformation term cannot
        inject artificial energy into the rigid-link model.
        """
        if self.reservoir_column is None:
            return
        offset = (
            self._deformation_pressure_offset()[self.reservoir_column]
            - self._reservoir_reference_offset
        )
        elapsed = max(
            0.0,
            float(self.data.time)
            - self._reservoir_seal_time
            - self.reservoir_relaxation_delay,
        )
        slow_fraction = np.exp(-elapsed / self.reservoir_leak_tau)
        fast_fraction = np.exp(
            -elapsed / self.reservoir_fast_relaxation_tau
        )
        relaxation = (
            (1.0 - self.reservoir_fast_relaxation_fraction) * slow_fraction
            + self.reservoir_fast_relaxation_fraction * fast_fraction
        )
        leaked_charge = self.reservoir_charge * relaxation
        target = leaked_charge + offset
        if self.reservoir_equalization:
            target = (
                (1.0 - self.reservoir_equalization) * target
                + self.reservoir_equalization * np.mean(target)
            )
        target = np.clip(target, 0.0, self.cfg.p_max)
        if self.reservoir_response_tau <= self.cfg.timestep:
            self.reservoir_pressure[:] = target
        else:
            alpha = self.cfg.timestep / self.reservoir_response_tau
            self.reservoir_pressure += alpha * (
                target - self.reservoir_pressure
            )

        if self.reservoir_force_feedback:
            # Feed the slow, dissipative leak state back into mechanics.  The
            # instantaneous empirical deformation term stays sensor-only so
            # it cannot inject energy into the rigid-link model.
            force_pressure = leaked_charge
            if self.reservoir_equalization:
                force_pressure = (
                    (1.0 - self.reservoir_equalization) * force_pressure
                    + self.reservoir_equalization * np.mean(force_pressure)
                )
            self.p_actual[self.reservoir_column] = np.clip(
                force_pressure, 0.0, self.cfg.p_max
            )

    def _level_curvatures(self) -> np.ndarray:
        """Per-level bending angle about (x, y), shape (n_pouches, 2)."""
        q = self.data.qpos
        out = np.zeros((self.cfg.n_pouches, 2))
        for k in range(self.cfg.n_pouches):
            bx, by = self.bend_dofs[k]
            out[k, 0] = q[bx]
            out[k, 1] = q[by]
        return out

    def _pouch_sensors(self) -> np.ndarray:
        """Simulated pouch pressures, shape (n_segments, n_pouches).

        Unsealed mechanics replay reads actual pressure plus empirical
        curvature and extension terms. The calibrated runtime returns its
        separately updated sealed-pouch signal and chamber pressure for active
        columns.
        """
        if self.reservoir_column is None:
            # Low-level unsealed mechanics/sensor replay.
            p_meas = self.p_actual + self._deformation_pressure_offset()
        else:
            # Reservoir pressures already include deformation. The other
            # three signals are chamber-pressure measurements.
            p_meas = self.p_actual.copy()
            p_meas[self.reservoir_column] = self.reservoir_pressure
        if self.sensor_noise > 0:
            p_meas = p_meas + self.rng.normal(0, self.sensor_noise, p_meas.shape)
        return p_meas

    # ────────────────────────────────────────────────────── rendering ──────
    def render_frame(self, cam_azimuth: float = 135, cam_elevation: float = -20,
                     cam_distance: float = 0.80, width: int = 640, height: int = 480):
        """Render a frame from the given camera pose.

        Defaults give a diagonal view (135°) that shows all 4 column colours,
        at a distance appropriate for the calibrated ~30 cm arm.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
        cfg = self.cfg
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = cam_azimuth, cam_elevation, cam_distance
        mount_z = cfg.length + 0.28 if cfg.hang_down else 0.05
        arm_mid_z = mount_z - cfg.length * 0.5 if cfg.hang_down else cfg.length * 0.5
        cam.lookat[:] = [0.0, 0.0, arm_mid_z]
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self) -> None:
        """Release the optional offscreen renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    sim = SoftArmSim()
    obs = sim.reset()
    print("tip at rest:", np.round(obs["tip_pos"], 4))

    # Segment 2 step while Segment 1 remains sealed at the 2 psi default.
    for _ in range(200):
        obs = sim.step([3.0, 0.0, 0.0])
    print("tip after Segment 2 step:", np.round(obs["tip_pos"], 4))
    print("reservoir sensors [psi]:", np.round(obs["reservoir_pressures"], 2))

    # Equal active commands produce the recorded axial-style response.
    sim.reset()
    for _ in range(200):
        obs = sim.step([3.0, 3.0, 3.0])
    print("tip after symmetric (axial):", np.round(obs["tip_pos"], 4))

    # Save pressure log for replay on physical robot
    log_path = sim.save_pressure_log("output/pressure_log.csv")
    log = sim.get_pressure_log()
    print(f"Logged {len(log['time'])} ticks, "
          f"duration {log['time'][-1]:.2f} s, "
          f"shape p_cmd={log['p_cmd'].shape}")
