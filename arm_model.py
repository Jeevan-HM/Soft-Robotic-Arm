"""
MJCF model generator for the fabric pneumatic soft robotic arm.

Architecture
------------
Think of it as 4 buildings arranged at the corners of a square,
each building having 5 floors:

    N (col 1)
    |
W --+-- E (col 0)   <-- square cross-section, viewed from above
    |
    S (col 3)

  col 0: East  (+x, phi=  0 deg)
  col 1: North (+y, phi= 90 deg)
  col 2: West  (-x, phi=180 deg)
  col 3: South (-y, phi=270 deg)

Each column has 5 floor levels (pouches) stacked vertically.
Pressurising a column elongates that side -> arm bends toward that
column's direction.

Control input: P has shape (n_segments=4, n_pouches=5)
               P[s, k] = pressure [psi] in column s at floor k.
Max pressure:  10.0 psi.

Kinematic chain: 5 nested MuJoCo bodies (one per floor level), each with
  - slide joint  (ext{k})  : axial extension/compression of that level
  - hinge joint  (bx{k})   : bending about X
  - hinge joint  (by{k})   : bending about Y
Total DOF: 15.

The wrapper (soft_arm_sim.SoftArmSim) maps P -> per-level bending
moments + axial forces written to data.qfrc_applied every physics step.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ArmConfig:
    # --- structure ---
    n_segments: int = 4           # pneumatic columns (the "buildings")
    n_pouches: int = 5            # floor levels per column (the "floors")
    seg_azimuth0_deg: float = 0.0 # azimuth of column 0 [deg]

    # --- geometry (estimated from physical arm photos) ---
    length: float = 0.22          # total arm length at rest [m]
    spacer_length: float = 0.118  # distance from plywood mount to arm base [m]
    col_offset: float = 0.028     # column axis distance from arm centre [m]
    col_radius: float = 0.018     # column outer radius [m]
    mass: float = 0.35            # total moving mass: fabric + fittings [kg]
    tip_mass: float = 0.08        # tip OptiTrack marker frame [kg]
    tip_arm: float = 0.07         # marker cross half-length [m]
    hang_down: bool = True        # arm hangs downward from mount plate

    # --- pressure -> wrench mapping ---
    # Bending moment at level k from column s:
    #   M[k] = pressure_gain * moment_arm * P[s,k] * col_direction[s]
    moment_arm: float = 0.028     # = col_offset [m]
    pressure_gain: float = 0.552  # [N/psi] bending force per column per level
    extension_gain: float = 0.034 # [N/psi] axial force per column per level

    # --- passive mechanical properties (at zero pre-inflation) ---
    base_stiffness: float = 0.30  # [N*m/rad] per bending joint
    base_damping: float = 0.08    # [N*m*s/rad]
    axial_stiffness: float = 600.0  # [N/m] per level slide
    axial_damping: float = 30.0     # [N*s/m]
    # stiffness scales with pre-inflation: k = k0 * (1 + c*P_pre)
    stiffness_per_psi: float = 0.414

    # --- pneumatics ---
    p_max: float = 10.0           # [psi] max pouch pressure
    tau_pneumatic: float = 0.12   # [s] first-order fill/vent time constant

    # --- simulation ---
    timestep: float = 0.001       # [s] physics dt

    # --- derived ---------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return self.n_segments * self.n_pouches

    def level_height(self) -> float:
        """Height of each floor level [m]."""
        return self.length / self.n_pouches

    def col_azimuths(self) -> np.ndarray:
        """Azimuth of each column [rad], shape (n_segments,).
        User requested: Mirror horizontally and vertically (180 deg rotation).
        col 0: -Y (270 deg)
        col 1: -X (180 deg)
        col 2: +Y (90 deg)
        col 3: +X (0 deg)
        """
        angles_deg = [270.0, 180.0, 90.0, 0.0]
        return np.deg2rad(angles_deg)

    def azimuths(self) -> np.ndarray:
        """Alias for col_azimuths() — API compatibility."""
        return self.col_azimuths()
        
    @property
    def mount_z(self) -> float:
        """World Z coordinate of the mount plate."""
        # 0.40m clearance below the arm tip to prevent clipping the floor
        return self.length + self.spacer_length + 0.40 if self.hang_down else 0.05


def build_arm_xml(cfg: ArmConfig) -> str:
    """Return an MJCF string for the 4-column × 5-level soft arm."""
    h = cfg.level_height()
    n_levels = cfg.n_pouches           # 5 vertical bodies
    link_mass = cfg.mass / n_levels    # mass per level body
    zdir = -1.0 if cfg.hang_down else 1.0

    col_phis = cfg.col_azimuths()      # (4,)
    # One color per column so it's visually obvious which column is which
    col_colors = [
        "0.82 0.18 0.08 1",  # col 0 : red (swapped from orange)
        "0.08 0.28 0.90 1",  # col 1 : blue (swapped from green)
        "0.92 0.55 0.08 1",  # col 2 : orange (swapped from red)
        "0.10 0.68 0.18 1",  # col 3 : green (swapped from blue)
    ]
    disc_r = cfg.col_offset + cfg.col_radius + 0.003  # connector ring radius

    body_xml = ""
    indent = "      "   # 6 spaces, inside mount body

    for k in range(n_levels):
        #   level 0 is at the end of the spacer posts
        #   every subsequent level is one step further down (inside previous body)
        offset = -cfg.spacer_length if k == 0 and cfg.hang_down else (0.0 if k == 0 else zdir * h)
        body_xml += f'{indent}<body name="level{k}" pos="0 0 {offset:.6f}">\n'

        # Axial slide DOF
        body_xml += (
            f'{indent}  <joint name="ext{k}" type="slide" axis="0 0 {zdir:g}" '
            f'stiffness="{cfg.axial_stiffness}" damping="{cfg.axial_damping}" '
            f'springref="0" range="-0.005 0.030"/>\n'
        )
        # 2-DOF bending (universal joint)
        body_xml += (
            f'{indent}  <joint name="bx{k}" type="hinge" axis="1 0 0" '
            f'damping="{cfg.base_damping}" stiffness="{cfg.base_stiffness}" springref="0"/>\n'
            f'{indent}  <joint name="by{k}" type="hinge" axis="0 1 0" '
            f'damping="{cfg.base_damping}" stiffness="{cfg.base_stiffness}" springref="0"/>\n'
        )

        # Rigid connector ring at the top of this level (between levels)
        body_xml += (
            f'{indent}  <geom name="ring{k}" type="cylinder" pos="0 0 0" '
            f'size="{disc_r:.4f} 0.0025" mass="0.006" '
            f'rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>\n'
        )

        # 4 column capsules (coloured by segment)
        col_mass = link_mass * 0.85 / cfg.n_segments
        for s, phi in enumerate(col_phis):
            cx = cfg.col_offset * np.cos(phi)
            cy = cfg.col_offset * np.sin(phi)
            body_xml += (
                f'{indent}  <geom name="col{s}_l{k}" type="capsule" '
                f'fromto="{cx:.5f} {cy:.5f} 0 {cx:.5f} {cy:.5f} {zdir * h:.5f}" '
                f'size="{cfg.col_radius:.4f}" mass="{col_mass:.6f}" '
                f'rgba="{col_colors[s]}" contype="0" conaffinity="0"/>\n'
            )

        # Thin central structural core
        body_xml += (
            f'{indent}  <geom name="core{k}" type="cylinder" '
            f'pos="0 0 {zdir * h * 0.5:.5f}" '
            f'size="0.005 {h * 0.48:.5f}" mass="{link_mass * 0.15:.6f}" '
            f'rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0"/>\n'
        )

        indent += "  "   # next body will be nested inside this one

    # ── Tip disc + OptiTrack marker frame ─────────────────────────────────
    a = cfg.tip_arm
    body_xml += (
        f'{indent}<body name="tip_disc" pos="0 0 {zdir * h:.6f}">\n'
        f'{indent}  <geom name="tip_ring" type="cylinder" pos="0 0 0" '
        f'size="{disc_r:.4f} 0.0025" mass="0.010" '
        f'rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>\n'
        f'{indent}  <body name="tip_frame" pos="0 0 {zdir * 0.012:.6f}">\n'
        f'{indent}    <geom name="tip_bar_x" type="capsule" '
        f'fromto="-{a} 0 0 {a} 0 0" size="0.004" '
        f'mass="{cfg.tip_mass * 0.30:.4f}" rgba="0.12 0.12 0.14 1" '
        f'contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_bar_y" type="capsule" '
        f'fromto="0 -{a} 0 0 {a} 0" size="0.004" '
        f'mass="{cfg.tip_mass * 0.30:.4f}" rgba="0.12 0.12 0.14 1" '
        f'contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_c" type="sphere" pos="0 0 0" '
        f'size="0.007" mass="{cfg.tip_mass * 0.40:.4f}" '
        f'rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_px" type="sphere" pos="{a} 0 0" '
        f'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_nx" type="sphere" pos="-{a} 0 0" '
        f'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_py" type="sphere" pos="0 {a} 0" '
        f'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_ny" type="sphere" pos="0 -{a} 0" '
        f'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <site name="tip" pos="0 0 0" size="0.007" rgba="0.1 0.9 0.3 1"/>\n'
        f'{indent}  </body>\n'
        f'{indent}</body>\n'
    )

    # Close all level bodies (deepest first)
    for k in range(n_levels, 0, -1):
        ind = "      " + "  " * (k - 1)
        body_xml += f"{ind}</body>\n"

    xml = f"""<mujoco model="fabric_soft_arm">
  <option timestep="{cfg.timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.18 0.20 0.24" rgb2="0.24 0.26 0.30"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.08"/>
    <material name="mount_wood" rgba="0.62 0.48 0.30 1"/>
    <material name="post_metal" rgba="0.14 0.14 0.16 1"/>
  </asset>
  <worldbody>
    <light pos="0.0 -0.8 1.8" dir="0.0 0.5 -1" diffuse="0.6 0.6 0.65" specular="0.2 0.2 0.2"/>
    <light pos="0.6 0.4 1.5" dir="-0.4 -0.3 -1" diffuse="0.4 0.4 0.45" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="2.0 2.0 0.05" material="grid"/>
    <body name="mount" pos="0 0 {cfg.mount_z:.4f}">
      <!-- Plywood mount plate (~15x15 cm) -->
      <geom name="mount_plate" type="box" size="0.150 0.150 0.009"
            material="mount_wood" contype="0" conaffinity="0"/>
      
      <!-- 4 black 3D-printed spacer posts hanging down -->
      <geom name="spacer_fl" type="box" fromto="-0.035 -0.035 0 -0.035 -0.035 -{cfg.spacer_length:.4f}"
            size="0.006 0.006 0.006" material="post_metal" contype="0" conaffinity="0"/>
      <geom name="spacer_fr" type="box" fromto=" 0.035 -0.035 0  0.035 -0.035 -{cfg.spacer_length:.4f}"
            size="0.006 0.006 0.006" material="post_metal" contype="0" conaffinity="0"/>
      <geom name="spacer_bl" type="box" fromto="-0.035  0.035 0 -0.035  0.035 -{cfg.spacer_length:.4f}"
            size="0.006 0.006 0.006" material="post_metal" contype="0" conaffinity="0"/>
      <geom name="spacer_br" type="box" fromto=" 0.035  0.035 0  0.035  0.035 -{cfg.spacer_length:.4f}"
            size="0.006 0.006 0.006" material="post_metal" contype="0" conaffinity="0"/>
            
      <!-- Bottom base disc (mount -> arm) -->
      <geom name="mount_disc" type="box" pos="0 0 -{cfg.spacer_length:.4f}"
            size="0.045 0.045 0.003" mass="0.010"
            rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
{body_xml}    </body>
  </worldbody>
  <sensor>
    <framepos name="tip_pos" objtype="site" objname="tip"/>
    <framequat name="tip_quat" objtype="site" objname="tip"/>
    <framelinvel name="tip_vel" objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""
    return xml


if __name__ == "__main__":
    import mujoco

    cfg = ArmConfig()
    xml = build_arm_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    print(f"Model OK: {model.nq} DoF, {model.nbody} bodies, "
          f"{cfg.n_channels} pressure channels ({cfg.n_segments} cols × {cfg.n_pouches} levels)")
    print(f"Arm: length={cfg.length*100:.0f} cm, "
          f"col_offset={cfg.col_offset*100:.1f} cm, "
          f"col_radius={cfg.col_radius*100:.1f} cm, "
          f"p_max={cfg.p_max:.1f} psi")
