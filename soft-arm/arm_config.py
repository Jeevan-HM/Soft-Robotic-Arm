"""
arm_config.py — Physical parameters for the SOFA FEM soft arm simulation.

Unlike the MuJoCo version (which used spring-damper constants), this config
holds FEM continuum-mechanics parameters for the heat-sealable, fabric-pouch
construction of the physical arm.

Parameter basis
---------------
Young's modulus (E)
    The FEM body is an effective continuum standing in for TPU-coated fabric,
    heat-sealed seams, blue carriers, fittings, and separator plates.  Its
    effective E = 3 MPa is deliberately higher than bare silicone: it matches
    the pressure/bending scale of the MuJoCo model and prevents the numerical
    body from stretching like an unreinforced rubber tube.

Poisson ratio (ν)
    Silicone elastomers are nearly incompressible (ν → 0.5).  ν = 0.45 avoids
    locking in FEM while remaining physically accurate.  Values in the range
    0.45–0.499 are typical for silicone in SOFA simulations.

Mass
    The arm fabric/fittings mass is 0.35 kg and the external marker frame is
    0.08 kg, matching the MuJoCo construction model.

Pressure
    Pressure commands have no finite software ceiling.  SOFA uses SI units
    (Pa).  Pre-inflation is modelled as a uniform low baseline pressure applied
    to all 20 pouch surfaces simultaneously at the start of the simulation.

Pneumatic time constant
    The same first-order lag model as MuJoCo is implemented in the Python wrapper
    (soft_arm_sim.py).  τ = 0.12 s.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


@dataclass
class ArmConfig:
    """Physical and simulation parameters for the SOFA FEM soft arm.

    All physical values are in SI units (m, kg, Pa, s).
    """

    # ── Geometry ──────────────────────────────────────────────────────────
    arm_length:    float = 0.220   # [m]  total arm length at rest
    arm_radius:    float = 0.046   # [m]  outer body radius (col_offset + col_radius)
    col_offset:    float = 0.028   # [m]  column axis distance from arm centre
    col_radius:    float = 0.015   # [m]  cavity radius
    n_cols:        int   = 4       # number of pneumatic columns
    n_levels:      int   = 5       # number of pouches per column (levels/floors)
    level_height:  float = field(init=False)  # [m] arm_length / n_levels
    pouch_gap:     float = 0.004   # [m] wall between adjacent pouches
    col_azimuth0:  float = 0.0     # [deg] azimuth of column 0 (East)
    hang_down:     bool  = True    # local +Z runs from plywood mount toward the tip
    tip_offset:    float = 0.012   # [m] marker frame below the free-end plate
    tip_arm:       float = 0.070   # [m] marker cross half-length

    # ── Material (FEM) ────────────────────────────────────────────────────────
    # Effective composite for TPU fabric, sealed seams, and carrier structure.
    young_modulus:  float = 3_000_000.0 # [Pa]  E = 3 MPa
    poisson_ratio:  float = 0.45        # [—]   near-incompressible
    density:        float = 1100.0      # [kg/m³] retained for derived studies
    arm_mass:       float = 0.35        # [kg] fabric arm + fittings
    tip_mass:       float = 0.08        # [kg] external OptiTrack marker frame

    # ── Pneumatics ────────────────────────────────────────────────────────────
    p_max_psi:      float | None = None # retained for config compatibility; no ceiling
    p_max_pa:       float | None = field(init=False)
    tau_pneumatic:  float = 0.12        # [s]   first-order fill/vent time constant
    pre_inflation_psi: float = 1.5      # [psi] resting pre-inflation
    sofa_pressure_scale: float = 0.08   # calibrated physical → continuum pressure
    max_pouch_volume_growth_ratio: float = 1.0
    # Heat-sealed fabric limits growth; 1.0 permits one initial cavity volume.

    # ── Simulation ────────────────────────────────────────────────────────────
    # SOFA solver parameters
    timestep:       float = 0.001       # [s]  physics step, matching MuJoCo
    constraint_solver_tolerance: float = 1e-9
    constraint_solver_max_iter:  int   = 1000
    ode_solver_rayleigh_stiffness: float = 0.1   # Rayleigh damping β (stiffness)
    ode_solver_rayleigh_mass:      float = 0.1   # Rayleigh damping α (mass)

    # Mesh file (relative to the soft-arm directory)
    mesh_file:      str  = "mesh/soft_arm.msh"

    # Physical group tags (must match generate_mesh.py)
    tag_body:    int = 1
    tag_base:    int = 2   # fixed (mount plate)
    tag_tip:     int = 3   # free tip
    tag_outer:   int = 4
    tag_pouch_base: int = 10

    def __post_init__(self) -> None:
        if self.n_cols <= 0 or self.n_levels <= 0:
            raise ValueError("n_cols and n_levels must both be positive")
        self.level_height = self.arm_length / self.n_levels
        if not 0.0 < self.pouch_gap < self.level_height:
            raise ValueError("pouch_gap must be between zero and level_height")
        self.p_max_pa = (
            None if self.p_max_psi is None else self.p_max_psi * 6894.76
        )

    # ── Derived helpers ───────────────────────────────────────────────────
    @property
    def pouch_height(self) -> float:
        """Height of one individual pouch [m] = level_height - pouch_gap."""
        return self.level_height - self.pouch_gap

    @property
    def pouch_initial_volume(self) -> float:
        """Nominal cylindrical cavity volume for one pouch [m³]."""
        import math
        return math.pi * self.col_radius**2 * self.pouch_height

    def psi_to_pa(self, psi: float) -> float:
        """Convert pressure from psi to Pa."""
        return psi * 6894.76

    def col_azimuths_deg(self) -> list[float]:
        """Azimuth of each column [deg], starting from col_azimuth0."""
        return [self.col_azimuth0 + 90.0 * i for i in range(self.n_cols)]

    def pouch_cavity_tag(self, col: int, level: int) -> int:
        """Physical group tag for one pouch (col 0-3, level 0-4)."""
        if not 0 <= col < self.n_cols:
            raise IndexError(f"column index {col} is outside [0, {self.n_cols})")
        if not 0 <= level < self.n_levels:
            raise IndexError(f"level index {level} is outside [0, {self.n_levels})")
        return self.tag_pouch_base + col * self.n_levels + level

    def pouch_cavity_tags(self):
        """(n_cols, n_levels) array of mesh physical group tags."""
        import numpy as np
        tags = np.zeros((self.n_cols, self.n_levels), dtype=int)
        for c in range(self.n_cols):
            for l in range(self.n_levels):
                tags[c, l] = self.pouch_cavity_tag(c, l)
        return tags

    @property
    def pre_inflation_pa(self) -> float:
        """Pre-inflation pressure in Pa."""
        return self.psi_to_pa(self.pre_inflation_psi)

    @property
    def wall_thickness(self) -> float:
        """Approximate minimum wall thickness [m] between cavity and outer surface."""
        return self.arm_radius - self.col_offset - self.col_radius

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_json(self, path: str | Path) -> None:
        """Save all config fields to a JSON file."""
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "ArmConfig":
        """Load an ArmConfig from a JSON file.  Unknown keys are ignored."""
        with open(path) as f:
            data = json.load(f)
        import dataclasses
        valid = {fd.name for fd in dataclasses.fields(cls) if fd.init}
        return cls(**{k: v for k, v in data.items() if k in valid})


if __name__ == "__main__":
    cfg = ArmConfig()
    print("=== SOFA FEM ArmConfig ===")
    print(f"  Arm length         : {cfg.arm_length * 100:.0f} cm")
    print(f"  Arm radius         : {cfg.arm_radius * 1000:.0f} mm")
    print(f"  Column cavity r    : {cfg.col_radius * 1000:.0f} mm")
    print(f"  Level height       : {cfg.level_height * 1000:.0f} mm")
    print(f"  Pouch height       : {cfg.pouch_height * 1000:.0f} mm")
    print(f"  Inter-pouch wall   : {cfg.pouch_gap * 1000:.0f} mm")
    print(f"  Wall thickness     : {cfg.wall_thickness * 1000:.1f} mm")
    print()
    print("=== Material (FEM) ===")
    print(f"  Young's modulus    : {cfg.young_modulus / 1e3:.0f} kPa  ({cfg.young_modulus / 1e6:.2f} MPa)")
    print(f"  Poisson ratio      : {cfg.poisson_ratio}")
    print(f"  Density            : {cfg.density} kg/m³")
    print()
    print("=== Pneumatics ===")
    if cfg.p_max_psi is None:
        print("  Software PSI limit : none")
    else:
        print(f"  Max pressure       : {cfg.p_max_psi} psi  ({cfg.p_max_pa / 1000:.1f} kPa)")
    print(f"  Pre-inflation      : {cfg.pre_inflation_psi} psi  ({cfg.pre_inflation_pa / 1000:.2f} kPa)")
    print(f"  Pneumatic τ        : {cfg.tau_pneumatic} s")
    print()
    print("=== Column azimuths ===")
    labels = ["East", "North", "West", "South"]
    tags = cfg.pouch_cavity_tags()
    for i, (label, phi) in enumerate(zip(labels, cfg.col_azimuths_deg())):
        print(f"  col {i} ({label:5s}) : {phi:.0f}°  →  mesh tags {tags[i].tolist()}")
