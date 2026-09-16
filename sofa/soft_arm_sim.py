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

Key differences from MuJoCo wrapper
-------------------------------------
- `n_pouches` (floor levels) has no direct equivalent in SOFA FEM: each column
  has a single continuous cavity.  `step()` accepts (4, 5) for API compatibility
  but only the *column mean* pressure is used (averaged over the 5 level slots).
- `render_frame()` is not implemented (use runSofa for visualisation).
- Pressure is converted psi → Pa internally before passing to SOFA.

Pneumatic lag
-------------
The same first-order filter as the MuJoCo model is applied in Python before
the pressure is sent to SOFA's SurfacePressureConstraint:

    p_actual += α × (p_cmd − p_actual),   α = dt / τ_pneumatic

This faithfully reproduces the fill/vent dynamics (τ = 0.12 s).

SOFA runtime dependency
-----------------------
Requires SOFA with SoftRobots plugin.  See sofa/README.md for installation.
If SOFA is not available, import will raise ImportError with a helpful message.
"""

from __future__ import annotations

from pathlib import Path
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
            "See sofa/README.md → 'Installation' for setup instructions.\n\n"
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

        # Pneumatic lag state: actual pressures in Pa (4 cols)
        self.p_actual_pa = np.zeros(self.cfg.n_cols)
        self.p_pre_pa    = 0.0

        # ── Build SOFA scene ──────────────────────────────────────────────
        self._root = Sofa.Core.Node("root")

        from soft_arm_scene import createScene
        self._arm_node, self._cavity_nodes, self._tip_node = createScene(
            self._root, self.cfg
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
        p_pre_psi = float(np.clip(p_pre_psi, 0.0, self.cfg.p_max_psi))
        self.p_pre_pa = self.cfg.psi_to_pa(p_pre_psi)

    def step(self, p_cmd_psi) -> dict:
        """Advance one control tick.

        Parameters
        ----------
        p_cmd_psi : array-like
            Commanded pressure [psi].  Accepted shapes:
              (n_cols, n_pouches) — full per-level command (levels are averaged)
              (n_cols,)           — one value per column
              (n_pouches,)        — broadcast same level pattern to all columns
              flat (n_cols*n_pouches,) — reshaped and column-averaged

        Returns
        -------
        dict with keys:
            "time"      float        simulation time [s]
            "tip_pos"   (3,) float   tip position [m]
            "p_actual"  (4, 5) float actual cavity pressures [psi] (broadcast)
        """
        import Sofa.Simulation

        cfg = self.cfg
        p_cmd = np.asarray(p_cmd_psi, dtype=float)

        # Normalise to (n_cols,) — same pressure to all pouches within a col
        if p_cmd.ndim == 2:
            # (n_cols, n_pouches) → mean over pouches
            if p_cmd.shape == (cfg.n_cols, 5) or p_cmd.shape[1] > 1:
                p_cmd = p_cmd.mean(axis=1)   # (n_cols,)
        elif p_cmd.ndim == 1:
            if p_cmd.size == cfg.n_cols * 5:
                p_cmd = p_cmd.reshape(cfg.n_cols, 5).mean(axis=1)
            elif p_cmd.size == 5:
                # per-pouch broadcast → mean, then tile to all cols
                p_cmd = np.full(cfg.n_cols, p_cmd.mean())
        p_cmd = p_cmd.flatten()[:cfg.n_cols]

        # Add pre-inflation and clip
        p_total_psi = np.clip(p_cmd + self.p_pre_pa / 6894.76, 0.0, cfg.p_max_psi)
        p_total_pa  = self.cfg.psi_to_pa(p_total_psi)   # (n_cols,)

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
        # Broadcast (n_cols,) → (n_cols, 5) for API compatibility
        p_psi   = self.p_actual_pa / 6894.76
        p_out   = np.tile(p_psi[:, None], (1, 5))   # (4, 5)
        return {
            "time":     self._time,
            "tip_pos":  tip_pos,
            "p_actual": p_out,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _push_pressures_to_sofa(self) -> None:
        """Write current p_actual_pa into each SurfacePressureConstraint."""
        for i, (_, actuator) in enumerate(self._cavity_nodes):
            actuator.value = float(self.p_actual_pa[i])

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
