"""Physical and numerical parameters for the fabric pneumatic soft arm."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ArmConfig:
    """All physical and numerical parameters for the soft arm model.

    All physical values are in SI units (m, kg, s) unless noted otherwise.
    Modify fields here to explore different arm configurations — the MJCF
    model and simulator adapt automatically.
    """

    # ── Structure ──────────────────────────────────────────────────────────
    n_segments: int = 4   # four radial pouch columns
    n_pouches: int = 5    # five axial pouch levels

    # ── Geometry and mass ──────────────────────────────────────────────────
    length: float = 0.296         # total arm rest length [m]
    spacer_length: float = 0.118  # visual spacer between mount and arm [m]
    col_offset: float = 0.028     # column axis distance from arm centre [m]
    col_radius: float = 0.018     # column capsule radius [m]
    mass: float = 0.35            # arm fabric + fittings mass [kg]
    tip_mass: float = 0.08        # OptiTrack marker frame mass [kg]
    tip_arm: float = 0.07         # half-length of the cross-shaped tip frame [m]
    hang_down: bool = True        # True = arm hangs down from the mount

    # ── Pressure → wrench mapping ──────────────────────────────────────────
    moment_arm: float = 0.028            # lever arm for bending moment [m]
    pressure_gain: float = 0.1273999973  # psi → bending force [N/psi]
    extension_gain: float = 0.0918534967 # psi → axial force [N/psi]

    # ── Passive mechanics (MuJoCo joint springs and dampers) ───────────────
    base_stiffness: float = 0.4168449461   # bending stiffness [N·m/rad]
    base_damping: float = 0.3629226883     # bending damping [N·m·s/rad]
    axial_stiffness: float = 934.2301025   # axial stiffness [N/m]
    axial_damping: float = 61.50955915     # axial damping [N·s/m]

    # ── Pneumatic dynamics and simulation timestep ─────────────────────────
    p_max: float = 10.0          # maximum command pressure [psi]
    pressure_delay: float = 0.5  # transport delay [s]
    tau_pneumatic: float = 0.6   # first-order pneumatic lag time constant [s]
    timestep: float = 0.001      # MuJoCo physics timestep [s]

    # ── Derived quantities ─────────────────────────────────────────────────
    @property
    def n_channels(self) -> int:
        """Total number of independent pouch pressure inputs (n_segments × n_pouches)."""
        return self.n_segments * self.n_pouches

    def level_height(self) -> float:
        """Resting length allocated to each of the five body levels [m]."""
        return self.length / self.n_pouches

    def col_azimuths(self) -> np.ndarray:
        """Column azimuths in radians: East=0°, North=90°, West=180°, South=270°."""
        return np.deg2rad([0.0, 90.0, 180.0, 270.0])

    @property
    def mount_z(self) -> float:
        """Mount height so the free tip starts ~0.40 m above the visual floor [m]."""
        if self.hang_down:
            return self.length + self.spacer_length + 0.40
        return 0.05
