"""
arm_config.py — Physical parameters for the SOFA FEM soft arm simulation.

Unlike the MuJoCo version (which used spring-damper constants), this config
holds FEM continuum-mechanics parameters derived from the material properties
of the physical arm (silicone-coated fabric, similar to Dragon Skin 10).

Parameter basis
---------------
Young's modulus (E)
    Dragon Skin 10 (Shore A 10) silicone: E ≈ 150–250 kPa in uniaxial tension.
    The arm fabric (TPU-laminated nylon) adds a strain-limiting layer that raises
    the effective bending stiffness.  A composite value of E = 0.3 MPa is used,
    consistent with published measurements of fabric-reinforced silicone actuators
    (Polygerinos et al., 2015; Mosadegh et al., 2014) and calibrated against the
    MuJoCo-identified parameters (base_stiffness = 0.459 N·m/rad at 1.5 psi
    pre-inflation, over a 44 mm lever × 5 levels ≈ effective E ~0.25–0.35 MPa).

Poisson ratio (ν)
    Silicone elastomers are nearly incompressible (ν → 0.5).  ν = 0.45 avoids
    locking in FEM while remaining physically accurate.  Values in the range
    0.45–0.499 are typical for silicone in SOFA simulations.

Density (ρ)
    Dragon Skin / Ecoflex silicone: 1070–1200 kg/m³.  Fabric adds ~5 % mass.
    Total 0.369 kg (sysid result) over a 220 mm × 46 mm cylinder volume of
    ≈ 7.3 × 10⁻⁴ m³ gives effective ρ ≈ 505 kg/m³ (much of the volume is
    air in the cavities and the arm is hollow).  We use the bulk silicone density
    and let the mesh volume produce the correct total mass automatically.

Pressure
    Max operating pressure: 10 psi = 68.9 kPa.  SOFA uses SI units (Pa).
    Pre-inflation is modelled as a uniform low baseline pressure applied to all
    4 cavity surfaces simultaneously at the start of the simulation.

Pneumatic time constant
    The same first-order lag model as MuJoCo is implemented in the Python wrapper
    (soft_arm_sim.py).  τ = 0.12 s.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import math


@dataclass
class ArmConfig:
    """Physical and simulation parameters for the SOFA FEM soft arm.

    All physical values are in SI units (m, kg, Pa, s).
    """

    # ── Geometry ──────────────────────────────────────────────────────────────
    arm_length:    float = 0.220   # [m]  total arm length at rest
    arm_radius:    float = 0.046   # [m]  outer body radius (col_offset + col_radius)
    col_offset:    float = 0.028   # [m]  column axis distance from arm centre
    col_radius:    float = 0.015   # [m]  cavity radius (slightly < hardware for wall thickness)
    n_cols:        int   = 4       # number of pneumatic columns
    col_azimuth0:  float = 0.0     # [deg] azimuth of column 0 (East)
    hang_down:     bool  = True    # arm hangs downward from mount plate (Z points down)

    # ── Material (FEM) ────────────────────────────────────────────────────────
    # Dragon Skin 10 silicone, fabric-reinforced (strain-limiting layer).
    # Source: Polygerinos et al. 2015, calibrated against sysid results.
    young_modulus:  float = 300_000.0   # [Pa]  E = 0.3 MPa
    poisson_ratio:  float = 0.45        # [—]   near-incompressible
    density:        float = 1100.0      # [kg/m³] bulk silicone (Dragon Skin: 1070–1200)

    # ── Pneumatics ────────────────────────────────────────────────────────────
    p_max_psi:      float = 10.0        # [psi] max operating pressure
    p_max_pa:       float = field(init=False)  # derived: p_max_psi × 6894.76
    tau_pneumatic:  float = 0.12        # [s]   first-order fill/vent time constant
    pre_inflation_psi: float = 1.5      # [psi] resting pre-inflation

    # ── Simulation ────────────────────────────────────────────────────────────
    # SOFA solver parameters
    timestep:       float = 0.01        # [s]  SOFA animation step (10 ms = 100 Hz)
    constraint_solver_tolerance: float = 1e-9
    constraint_solver_max_iter:  int   = 1000
    ode_solver_rayleigh_stiffness: float = 0.1   # Rayleigh damping β (stiffness)
    ode_solver_rayleigh_mass:      float = 0.1   # Rayleigh damping α (mass)

    # Mesh file (relative to sofa/ directory)
    mesh_file:      str  = "mesh/soft_arm.msh"

    # Physical group tags (must match generate_mesh.py)
    tag_body:    int = 1
    tag_base:    int = 2   # fixed (mount plate)
    tag_tip:     int = 3   # free tip
    tag_outer:   int = 4
    tag_col0:    int = 10  # East  cavity
    tag_col1:    int = 11  # North cavity
    tag_col2:    int = 12  # West  cavity
    tag_col3:    int = 13  # South cavity

    def __post_init__(self) -> None:
        self.p_max_pa = self.p_max_psi * 6894.76

    # ── Derived helpers ───────────────────────────────────────────────────────
    def psi_to_pa(self, psi: float) -> float:
        """Convert pressure from psi to Pa."""
        return psi * 6894.76

    def col_azimuths_deg(self) -> list[float]:
        """Azimuth of each column [deg], starting from col_azimuth0."""
        return [self.col_azimuth0 + 90.0 * i for i in range(self.n_cols)]

    def col_cavity_tags(self) -> list[int]:
        """Physical group tags for the 4 column cavities (in column order)."""
        return [self.tag_col0, self.tag_col1, self.tag_col2, self.tag_col3]

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
    print(f"  Wall thickness     : {cfg.wall_thickness * 1000:.1f} mm")
    print()
    print("=== Material (FEM) ===")
    print(f"  Young's modulus    : {cfg.young_modulus / 1e3:.0f} kPa  ({cfg.young_modulus / 1e6:.2f} MPa)")
    print(f"  Poisson ratio      : {cfg.poisson_ratio}")
    print(f"  Density            : {cfg.density} kg/m³")
    print()
    print("=== Pneumatics ===")
    print(f"  Max pressure       : {cfg.p_max_psi} psi  ({cfg.p_max_pa / 1000:.1f} kPa)")
    print(f"  Pre-inflation      : {cfg.pre_inflation_psi} psi  ({cfg.pre_inflation_pa / 1000:.2f} kPa)")
    print(f"  Pneumatic τ        : {cfg.tau_pneumatic} s")
    print()
    print("=== Column azimuths ===")
    labels = ["East", "North", "West", "South"]
    for i, (label, phi) in enumerate(zip(labels, cfg.col_azimuths_deg())):
        tag = cfg.col_cavity_tags()[i]
        print(f"  col {i} ({label:5s}) : {phi:.0f}°  →  mesh tag {tag}")
