"""
soft_arm_sim.py — SOFA-based digital twin of the fabric pneumatic soft arm.

API
---
This class mirrors the MuJoCo SoftArmSim interface (mujoco/soft_arm_sim.py)
so that existing control code requires minimal changes when switching backends.

    sim = SoftArmSim()
    sim.set_pre_inflation(1.5)          # psi
    P = np.zeros((4, 5))               # (n_cols, n_pouches) — uniform per column
    P[0, :] = 3.0                       # East column, all levels
    obs = sim.step(P)
    obs["tip_pos"]                      # (3,) tip position [m]
    obs["p_actual"]                     # (4, 5) actual cavity pressures [psi]

Each of the 20 MuJoCo pressure channels maps one-to-one onto an independent
SOFA ``SurfacePressureConstraint``. Pressure is converted psi → Pa internally,
then calibrated for the effective fabric continuum before it is passed to
SOFA. ``render_frame()`` remains unavailable; use
``runSofa`` for visualisation.

Pneumatic lag
-------------
The same first-order filter as the MuJoCo model is applied in Python before
the pressure is sent to SOFA's SurfacePressureConstraint:

    p_actual += α × (p_cmd − p_actual),   α = dt / τ_pneumatic

This faithfully reproduces the fill/vent dynamics (τ = 0.12 s).

SOFA runtime dependency
-----------------------
Requires SOFA with SoftRobots plugin. See soft-arm/README.md for installation.
If SOFA is not available, import will raise ImportError with a helpful message.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from arm_config import ArmConfig

# ---------------------------------------------------------------------------
# Lazy SOFA import — give a clear error if SOFA is not installed
# ---------------------------------------------------------------------------

def _require_sofa():
    try:
        import Sofa
        import Sofa.Core
        import Sofa.Simulation
        return Sofa
    except ImportError as exc:
        raise ImportError(
            "\n\nSOFA is not installed or not on PYTHONPATH.\n"
            "See soft-arm/README.md → 'Installation' for setup instructions.\n\n"
            "Quick start:\n"
            "  1. Download SOFA binaries from https://www.sofa-framework.org/\n"
            "  2. Add the Python bindings to your path:\n"
            "       export PYTHONPATH=/path/to/sofa/lib/python3/site-packages:$PYTHONPATH\n"
        ) from exc


# ---------------------------------------------------------------------------
# SoftArmSim
# ---------------------------------------------------------------------------

class SoftArmSim:
    """SOFA FEM digital twin of the fabric pneumatic soft arm.

    Parameters
    ----------
    cfg : ArmConfig, optional
        Physical and simulation parameters.  Defaults to ``ArmConfig()``.
    control_hz : float
        Control loop frequency [Hz].  Each call to ``step()`` advances the
        simulation by ``1 / control_hz`` seconds.
    """

    def __init__(
        self,
        cfg: Optional[ArmConfig] = None,
        control_hz: float = 100.0,
    ) -> None:
        Sofa = _require_sofa()
        import Sofa.Core
        import Sofa.Simulation

        self.cfg        = cfg or ArmConfig()
        self.control_dt = 1.0 / control_hz
        self._time      = 0.0
        self._step_count = 0

        # Pneumatic lag state: one pressure per independent pouch.
        self.p_actual_pa = np.zeros((self.cfg.n_cols, self.cfg.n_levels))
        self.p_pre_pa    = 0.0

        # ── Build SOFA scene ──────────────────────────────────────────────
        self._root = Sofa.Core.Node("root")

        from soft_arm_scene import createScene
        self._arm_node, self._cavity_nodes, self._tip_node = createScene(
            self._root, self.cfg, enable_pressure_panel=False
        )

        Sofa.Simulation.init(self._root)

        # Number of SOFA sub-steps per control tick
        sofa_dt = float(self.cfg.timestep)
        self._n_substeps = max(1, round(self.control_dt / sofa_dt))

    # ── Public API (matches MuJoCo SoftArmSim) ────────────────────────────

    def reset(self) -> dict:
        """Reset simulation to the rest configuration."""
        import Sofa.Simulation
        Sofa.Simulation.reset(self._root)
        self._time       = 0.0
        self._step_count = 0
        self.p_actual_pa[:] = self.p_pre_pa   # pre-inflation at rest
        self._push_pressures_to_sofa()
        return self.observe()

    def set_pre_inflation(self, p_pre_psi: float) -> None:
        """Set the resting pre-inflation pressure [psi]."""
        p_pre_psi = float(p_pre_psi)
        if not np.isfinite(p_pre_psi):
            raise ValueError("pre-inflation pressure must be finite")
        p_pre_psi = max(0.0, p_pre_psi)
        self.p_pre_pa = self.cfg.psi_to_pa(p_pre_psi)

    def step(self, p_cmd_psi) -> dict:
        """Advance one control tick.

        Parameters
        ----------
        p_cmd_psi : array-like
            Commanded pressure [psi].  Accepted shapes:
              (n_cols, n_levels) — full per-pouch command
              (n_cols,)          — one value per column, broadcast over levels
              (n_levels,)        — one level pattern, broadcast over columns
              flat (n_cols*n_levels,) — reshaped to the full command

        Returns
        -------
        dict with keys:
            "time"      float        simulation time [s]
            "tip_pos"   (3,) float   tip position [m]
            "p_actual"  (4, 5) float actual pouch pressures [psi]
        """
        import Sofa.Simulation

        cfg = self.cfg
        p_cmd = self._normalise_pressure_command(p_cmd_psi)

        # Add pre-inflation.  Negative totals are treated as zero pressure;
        # there is intentionally no finite upper software limit.
        p_total_psi = np.maximum(p_cmd + self.p_pre_pa / 6894.76, 0.0)
        p_total_pa  = self.cfg.psi_to_pa(p_total_psi)   # (n_cols, n_levels)

        # First-order pneumatic lag, per control sub-step
        alpha = cfg.timestep / max(cfg.tau_pneumatic, cfg.timestep)
        for _ in range(self._n_substeps):
            self.p_actual_pa += alpha * (p_total_pa - self.p_actual_pa)
            self._push_pressures_to_sofa()
            Sofa.Simulation.animate(self._root, cfg.timestep)

        self._time       += self.control_dt
        self._step_count += 1
        return self.observe()

    def observe(self) -> dict:
        """Return the current simulation state."""
        tip_pos = self._read_tip_position()
        return {
            "time":     self._time,
            "tip_pos":  tip_pos,
            "p_actual": (self.p_actual_pa / 6894.76).copy(),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _normalise_pressure_command(self, p_cmd_psi) -> np.ndarray:
        """Return a pressure command with shape ``(n_cols, n_levels)``."""
        cfg = self.cfg
        p_cmd = np.asarray(p_cmd_psi, dtype=float)
        if not np.all(np.isfinite(p_cmd)):
            raise ValueError("pressure command values must be finite")
        target_shape = (cfg.n_cols, cfg.n_levels)

        if p_cmd.shape == target_shape:
            return p_cmd.copy()
        if p_cmd.ndim == 1:
            if p_cmd.size == cfg.n_cols * cfg.n_levels:
                return p_cmd.reshape(target_shape).copy()
            if p_cmd.size == cfg.n_cols:
                return np.tile(p_cmd[:, None], (1, cfg.n_levels))
            if p_cmd.size == cfg.n_levels:
                return np.tile(p_cmd[None, :], (cfg.n_cols, 1))

        raise ValueError(
            "pressure command must have shape "
            f"{target_shape}, ({cfg.n_cols},), ({cfg.n_levels},), or contain "
            f"{cfg.n_cols * cfg.n_levels} flat values; got {p_cmd.shape}"
        )

    def _push_pressures_to_sofa(self) -> None:
        """Write calibrated pressure into each SurfacePressureConstraint.

        ``p_actual_pa`` remains the physical pressure reported through the API.
        The scale accounts for the fact that the isotropic continuum stands in
        for a much stiffer assembly of fabric, heat seals, carriers, and plates.
        """
        for col, column_nodes in enumerate(self._cavity_nodes):
            for lvl, (_, actuator) in enumerate(column_nodes):
                # SOFA stores this scalar pressure in a one-element vector.
                actuator.value = [
                    float(
                        self.p_actual_pa[col, lvl]
                        * self.cfg.sofa_pressure_scale
                    )
                ]

    def _read_tip_position(self) -> np.ndarray:
        """Read the current tip position from the SOFA scene."""
        try:
            tip_dofs = self._tip_node.getObject("tipDofs")
            pos = np.array(tip_dofs.position.value[0], dtype=float)
            return pos
        except Exception:
            return np.zeros(3)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Initialising SOFA simulation …")
    sim = SoftArmSim()
    sim.set_pre_inflation(1.5)
    obs = sim.reset()
    print(f"Tip at rest: {np.round(obs['tip_pos'] * 100, 2)} cm")

    P = np.zeros((4, 5))
    P[0, :] = 3.0   # East column
    for _ in range(50):
        obs = sim.step(P)
    print(f"Tip after East col @ 3 psi (0.5 s): {np.round(obs['tip_pos'] * 100, 2)} cm")
