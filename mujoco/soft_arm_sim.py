"""
SoftArmSim — MuJoCo digital twin of the fabric pneumatic soft arm.

Architecture: 4 segment columns × 5 floor levels = 20 pressure channels.

    P[s, k] = pressure [psi] in column s (0=East, 1=North, 2=West, 3=South)
              at floor level k (0=top, 4=bottom).

Columns are at azimuths [0, 90, 180, 270] deg.  Pressurising column s at
azimuth phi_s elongates that side of the arm, bending the tip toward phi_s.

Usage
-----
    sim = SoftArmSim()
    sim.set_pre_inflation(1.5)           # psi, stiffness modulation
    P = np.zeros((4, 5))
    P[0, :] = 3.0                        # East column, all 5 floors
    obs = sim.step(P)
    obs["tip_pos"]           # (3,) tip position [m]
    obs["pouch_pressures"]   # (4, 5) simulated sensor readings [psi]

step() accepts:
    (n_segments, n_pouches)  — full per-pouch command
    (n_segments,)            — broadcast same pressure to all floor levels
    (n_pouches,)             — broadcast same floor pattern to all columns
    flat (n_channels,)       — reshape to (n_segments, n_pouches)

Design notes
------------
* One control tick = 0.01 s (100 Hz), internally substeps at cfg.timestep (1 ms).
* Pneumatic lag: each pouch follows its command via first-order filter (tau_pneumatic).
* Pressure → wrench, per level k: its 4 columns at azimuths phi_s produce
  bending moment  M[k] = gain * sum_s P[s,k] * (-sin phi_s, cos phi_s)
  applied to level k's bending joints, plus axial force
  f[k] = ext_gain * sum_s P[s,k]  on the level's slide DOF.
* Pre-inflation: scales joint stiffness/damping (symmetric → stiffness modulation).
* Sensor model (parallel topology): measured pressure = actual + curvature
  coupling (arm bending toward col s squeezes its pouches) + extension + noise.
"""

import csv
from pathlib import Path

import numpy as np
import mujoco

from arm_model import ArmConfig, build_arm_xml


class SoftArmSim:
    def __init__(self, cfg: ArmConfig | None = None, control_hz: float = 100.0,
                 sensor_noise_psi: float = 0.007, curvature_coupling: float = 4.0,
                 extension_coupling: float = 30.0, seed: int | None = 0):
        self.cfg = cfg or ArmConfig()
        self.control_dt = 1.0 / control_hz
        self.n_sub_steps = max(1, round(self.control_dt / self.cfg.timestep))
        self.sensor_noise = sensor_noise_psi
        self.curvature_coupling = curvature_coupling
        self.extension_coupling = extension_coupling
        self.rng = np.random.default_rng(seed)

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
        mujoco.mj_forward(self.model, self.data)

        # ── Pressure logging ──────────────────────────────────────────────
        self._pressure_log: list[dict] = []   # [{time, p_cmd, p_actual}, ...]
        self._renderer = None

    # ──────────────────────────────────────────────────────────── API ──────
    def reset(self, clear_log: bool = True) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.p_actual[:] = 0.0
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

    def step(self, p_cmd_psi) -> dict:
        """Advance one control tick (100 Hz).

        p_cmd_psi shapes accepted:
          (n_segments, n_pouches) — full per-pouch command
          (n_segments,)           — broadcast to all floor levels
          (n_pouches,)            — broadcast to all columns
          flat (n_channels,)      — reshape to (n_segments, n_pouches)

        Commands are on top of pre-inflation; total is clipped to p_max.
        """
        cfg = self.cfg
        p_cmd = np.asarray(p_cmd_psi, dtype=float)

        if p_cmd.ndim == 1:
            if p_cmd.size == cfg.n_channels:
                # flat → reshape
                p_cmd = p_cmd.reshape(cfg.n_segments, cfg.n_pouches)
            elif p_cmd.size == cfg.n_segments:
                # one value per column → broadcast to all floors
                p_cmd = np.tile(p_cmd[:, None], (1, cfg.n_pouches))
            elif p_cmd.size == cfg.n_pouches:
                # one value per floor → broadcast to all columns
                p_cmd = np.tile(p_cmd[None, :], (cfg.n_segments, 1))

        p_cmd = p_cmd.reshape(cfg.n_segments, cfg.n_pouches)
        p_cmd = np.clip(p_cmd + self.p_pre, 0.0, cfg.p_max)

        alpha = cfg.timestep / max(cfg.tau_pneumatic, cfg.timestep)
        for _ in range(self.n_sub_steps):
            self.p_actual += alpha * (p_cmd - self.p_actual)
            self._apply_pressure_wrench()
            mujoco.mj_step(self.model, self.data)

        # Log the commanded and actual pressures for this tick
        self._pressure_log.append({
            "time": self.data.time,
            "p_cmd": p_cmd.copy(),        # (4, 5) final clipped command incl. pre-inflation
            "p_actual": self.p_actual.copy(),  # (4, 5) actual after pneumatic lag
        })

        return self.observe()

    def observe(self) -> dict:
        s = self.data.sensordata
        return {
            "time": self.data.time,
            "tip_pos": s[0:3].copy(),
            "tip_quat": s[3:7].copy(),
            "tip_vel": s[7:10].copy(),
            "pouch_pressures": self._pouch_sensors(),   # (n_seg, n_pouch)
            "p_actual": self.p_actual.copy(),
            "q": self.data.qpos.copy(),
        }

    # ──────────────────────────────────────────── pressure log access ──────
    def get_pressure_log(self) -> dict:
        """Return the pressure log as numpy arrays.

        Returns dict with:
          time      — (N,)       timestamps [s]
          p_cmd     — (N, 4, 5)  commanded pressures [psi] (incl. pre-inflation)
          p_actual  — (N, 4, 5)  actual pouch pressures after pneumatic lag [psi]
        """
        if not self._pressure_log:
            empty = lambda *s: np.empty((0, *s))
            return {"time": np.array([]),
                    "p_cmd": empty(self.cfg.n_segments, self.cfg.n_pouches),
                    "p_actual": empty(self.cfg.n_segments, self.cfg.n_pouches)}
        return {
            "time": np.array([r["time"] for r in self._pressure_log]),
            "p_cmd": np.array([r["p_cmd"] for r in self._pressure_log]),
            "p_actual": np.array([r["p_actual"] for r in self._pressure_log]),
        }

    def save_pressure_log(self, path: str | Path = "pressure_log.csv",
                          which: str = "p_cmd") -> Path:
        """Export the pressure log as a flat CSV file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        which : str
            'p_cmd' for commanded pressures (default — use this to replay on
            the physical robot), or 'p_actual' for actual pressures after
            pneumatic lag.

        CSV columns
        -----------
        time, col0_lv0, col0_lv1, …, col0_lv4, col1_lv0, …, col3_lv4
        (1 + 20 = 21 columns)
        """
        log = self.get_pressure_log()
        data = log[which]              # (N, 4, 5)
        times = log["time"]            # (N,)
        n = len(times)
        flat = data.reshape(n, -1)     # (N, 20)

        header = ["time"]
        for s in range(self.cfg.n_segments):
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

        Parallel topology: each column-level sensor reads actual pressure
        plus a curvature-coupling term (bending toward col s squeezes its
        pouches) and an extension-coupling term.
        """
        bend = self._level_curvatures()            # (n_levels, 2)
        # kappa[s, k] = bend at level k projected onto col s's direction
        # bend @ col_axes.T  →  (n_levels, n_segments)  →  .T  →  (n_seg, n_lev)
        kappa = (bend @ self.col_axes.T).T         # (n_segments, n_pouches)
        ext = self.data.qpos[self.ext_dofs][None, :]  # (1, n_pouches) → broadcast
        p_meas = (self.p_actual
                  + self.curvature_coupling * kappa
                  + self.extension_coupling * ext)
        if self.sensor_noise > 0:
            p_meas = p_meas + self.rng.normal(0, self.sensor_noise, p_meas.shape)
        return p_meas

    # ────────────────────────────────────────────────────── rendering ──────
    def render_frame(self, cam_azimuth: float = 135, cam_elevation: float = -20,
                     cam_distance: float = 0.80, width: int = 640, height: int = 480):
        """Render a frame from the given camera pose.

        Defaults give a diagonal view (135°) that shows all 4 column colours,
        at a distance appropriate for the compact ~22 cm arm.
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


if __name__ == "__main__":
    sim = SoftArmSim()
    sim.set_pre_inflation(1.5)
    obs = sim.reset()
    print("tip at rest:", np.round(obs["tip_pos"], 4))

    # East column, all 5 floors
    P = np.zeros((4, 5))
    P[0, :] = 3.0
    for _ in range(200):
        obs = sim.step(P)
    print("tip after East col step:", np.round(obs["tip_pos"], 4))
    print("col0 sensors [psi]:", np.round(obs["pouch_pressures"][0], 2))

    # All 4 columns equal → pure axial extension
    sim.reset()
    P[:] = 3.0
    for _ in range(200):
        obs = sim.step(P)
    print("tip after symmetric (axial):", np.round(obs["tip_pos"], 4))

    # Save pressure log for replay on physical robot
    log_path = sim.save_pressure_log("output/pressure_log.csv")
    log = sim.get_pressure_log()
    print(f"Logged {len(log['time'])} ticks, "
          f"duration {log['time'][-1]:.2f} s, "
          f"shape p_cmd={log['p_cmd'].shape}")
