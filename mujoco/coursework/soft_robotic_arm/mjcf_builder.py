"""Build the MuJoCo MJCF XML string for the fabric pneumatic soft arm."""

import numpy as np

from .arm_config import ArmConfig


def build_arm_xml(cfg: ArmConfig) -> str:
    """Return the complete arm MJCF as a string.

    Nested bodies make motion at one level affect every level below it.
    Pass the result directly to ``mujoco.MjModel.from_xml_string()``.

    Parameters
    ----------
    cfg:
        Physical and numerical parameters for the arm.  Use the defaults
        for the calibrated physical arm, or modify fields to explore
        different configurations.

    Returns
    -------
    str
        A valid MuJoCo XML string ready for ``MjModel.from_xml_string()``.

    Examples
    --------
    >>> import mujoco
    >>> from soft_robotic_arm import ArmConfig, build_arm_xml
    >>> model = mujoco.MjModel.from_xml_string(build_arm_xml(ArmConfig()))
    """
    # Divide the measured arm length and mass across five model levels.
    h = cfg.level_height()
    n_levels = cfg.n_pouches
    link_mass = cfg.mass / n_levels
    # Negative z places the arm below the mounting plate.
    zdir = -1.0 if cfg.hang_down else 1.0
    col_phis = cfg.col_azimuths()
    # Segment colors are reused at every level for visual orientation.
    col_colors = [
        "0.92 0.55 0.08 1",  # Segment 1 / East
        "0.10 0.68 0.18 1",  # Segment 2 / North
        "0.82 0.18 0.08 1",  # Segment 3 / West
        "0.08 0.28 0.90 1",  # Segment 4 / South
    ]
    disc_r = cfg.col_offset + cfg.col_radius + 0.003

    # Generate one nested body for each pouch level.
    body_xml = ""
    indent = "      "
    for k in range(n_levels):
        # Level 0 begins below the 118 mm visual spacer.
        offset = zdir * cfg.spacer_length if k == 0 else zdir * h
        body_xml += f'{indent}<body name="level{k}" pos="0 0 {offset:.6f}">\n'
        # One slide gives axial compliance at this level.
        body_xml += (
            f'{indent}  <joint name="ext{k}" type="slide" axis="0 0 {zdir:g}" '
            f'stiffness="{cfg.axial_stiffness}" damping="{cfg.axial_damping}" '
            'springref="0" range="-0.005 0.030"/>\n'
        )
        # Two perpendicular hinges allow bending in any horizontal direction.
        body_xml += (
            f'{indent}  <joint name="bx{k}" type="hinge" axis="1 0 0" '
            f'damping="{cfg.base_damping}" stiffness="{cfg.base_stiffness}" springref="0"/>\n'
            f'{indent}  <joint name="by{k}" type="hinge" axis="0 1 0" '
            f'damping="{cfg.base_damping}" stiffness="{cfg.base_stiffness}" springref="0"/>\n'
        )
        body_xml += (
            f'{indent}  <geom name="ring{k}" type="cylinder" pos="0 0 0" '
            f'size="{disc_r:.4f} 0.0025" mass="0.006" '
            'rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>\n'
        )
        # Colored capsules show segment locations and carry distributed mass.
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
        # The central core carries the remaining mass at this level.
        body_xml += (
            f'{indent}  <geom name="core{k}" type="cylinder" '
            f'pos="0 0 {zdir * h * 0.5:.5f}" '
            f'size="0.005 {h * 0.48:.5f}" mass="{link_mass * 0.15:.6f}" '
            'rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0"/>\n'
        )
        indent += "  "

    # Approximate the connector disc and OptiTrack marker frame at the tip.
    a = cfg.tip_arm
    body_xml += (
        f'{indent}<body name="tip_disc" pos="0 0 {zdir * h:.6f}">\n'
        f'{indent}  <geom name="tip_ring" type="cylinder" pos="0 0 0" '
        f'size="{disc_r:.4f} 0.0025" mass="0.010" '
        'rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>\n'
        f'{indent}  <body name="tip_frame" pos="0 0 {zdir * 0.012:.6f}">\n'
        f'{indent}    <geom name="tip_bar_x" type="capsule" '
        f'fromto="-{a} 0 0 {a} 0 0" size="0.004" '
        f'mass="{cfg.tip_mass * 0.30:.4f}" rgba="0.12 0.12 0.14 1" '
        'contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_bar_y" type="capsule" '
        f'fromto="0 -{a} 0 0 {a} 0" size="0.004" '
        f'mass="{cfg.tip_mass * 0.30:.4f}" rgba="0.12 0.12 0.14 1" '
        'contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_c" type="sphere" pos="0 0 0" '
        f'size="0.007" mass="{cfg.tip_mass * 0.40:.4f}" '
        'rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_px" type="sphere" pos="{a} 0 0" '
        'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_nx" type="sphere" pos="-{a} 0 0" '
        'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_py" type="sphere" pos="0 {a} 0" '
        'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <geom name="tip_ball_ny" type="sphere" pos="0 -{a} 0" '
        'size="0.005" mass="0.001" rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>\n'
        f'{indent}    <site name="tip" pos="0 0 0" size="0.007" rgba="0.1 0.9 0.3 1"/>\n'
        f'{indent}  </body>\n'
        f'{indent}</body>\n'
    )
    # Close the five nested level bodies after attaching the tip.
    for k in range(n_levels, 0, -1):
        body_xml += "      " + "  " * (k - 1) + "</body>\n"

    # Assemble the complete MJCF around the generated kinematic chain.
    return f"""<mujoco model="fabric_soft_arm">
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
    <light pos="0.0 -0.8 1.8" dir="0.0 0.5 -1" diffuse="0.6 0.6 0.65"/>
    <light pos="0.6 0.4 1.5" dir="-0.4 -0.3 -1" diffuse="0.4 0.4 0.45"/>
    <geom name="floor" type="plane" size="2.0 2.0 0.05" material="grid"/>
    <body name="mount" pos="0 0 {cfg.mount_z:.4f}">
      <geom name="mount_plate" type="box" size="0.150 0.150 0.009"
            material="mount_wood" contype="0" conaffinity="0"/>
      <geom name="spacer_fl" type="cylinder" fromto="-0.035 -0.035 0 -0.035 -0.035 -{cfg.spacer_length:.4f}"
            size="0.006" material="post_metal" contype="0" conaffinity="0"/>
      <geom name="spacer_fr" type="cylinder" fromto="0.035 -0.035 0 0.035 -0.035 -{cfg.spacer_length:.4f}"
            size="0.006" material="post_metal" contype="0" conaffinity="0"/>
      <geom name="spacer_bl" type="cylinder" fromto="-0.035 0.035 0 -0.035 0.035 -{cfg.spacer_length:.4f}"
            size="0.006" material="post_metal" contype="0" conaffinity="0"/>
      <geom name="spacer_br" type="cylinder" fromto="0.035 0.035 0 0.035 0.035 -{cfg.spacer_length:.4f}"
            size="0.006" material="post_metal" contype="0" conaffinity="0"/>
      <geom name="mount_disc" type="cylinder" pos="0 0 -{cfg.spacer_length:.4f}"
            size="{disc_r:.4f} 0.003" mass="0.010"
            rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
{body_xml}    </body>
  </worldbody>
  <sensor>
    <framepos name="tip_pos" objtype="site" objname="tip"/>
    <framequat name="tip_quat" objtype="site" objname="tip"/>
    <framelinvel name="tip_vel" objtype="site" objname="tip"/>
  </sensor>
</mujoco>"""
