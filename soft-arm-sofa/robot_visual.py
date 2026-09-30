"""Procedural visual skin for the physical 20-pouch soft arm.

The tetrahedral mesh is intentionally kept as the simulation model only.  A
separate, smooth surface is mapped to those FEM degrees of freedom so the
viewer resembles the fabric robot instead of exposing the coarse tet mesh.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from arm_config import ArmConfig


Mesh = tuple[list[list[float]], list[list[int]]]


def _material(
    diffuse: tuple[float, float, float],
    ambient: tuple[float, float, float],
    specular: tuple[float, float, float],
    shininess: float,
    alpha: float = 1.0,
) -> str:
    """Build the material syntax understood by SOFA's OglModel."""
    return (
        "Default "
        f"Diffuse 1 {diffuse[0]} {diffuse[1]} {diffuse[2]} {alpha} "
        f"Ambient 1 {ambient[0]} {ambient[1]} {ambient[2]} {alpha} "
        f"Specular 1 {specular[0]} {specular[1]} {specular[2]} {alpha} "
        "Emissive 0 0 0 0 0 "
        f"Shininess 1 {shininess}"
    )


POUCH_MATERIAL = _material(
    diffuse=(0.14, 0.15, 0.17),
    ambient=(0.045, 0.050, 0.060),
    specular=(0.26, 0.29, 0.34),
    shininess=28.0,
)
BLUE_FABRIC_MATERIAL = _material(
    diffuse=(0.035, 0.14, 0.62),
    ambient=(0.015, 0.045, 0.18),
    specular=(0.08, 0.10, 0.16),
    shininess=10.0,
)
PLATE_MATERIAL = _material(
    diffuse=(0.10, 0.11, 0.13),
    ambient=(0.035, 0.040, 0.050),
    specular=(0.26, 0.28, 0.32),
    shininess=40.0,
)
MARKER_BAR_MATERIAL = _material(
    diffuse=(0.075, 0.080, 0.090),
    ambient=(0.025, 0.028, 0.034),
    specular=(0.18, 0.19, 0.21),
    shininess=24.0,
)
MARKER_BALL_MATERIAL = _material(
    diffuse=(0.70, 0.72, 0.74),
    ambient=(0.15, 0.16, 0.17),
    specular=(0.70, 0.72, 0.75),
    shininess=70.0,
)
MOUNT_MATERIAL = _material(
    diffuse=(0.32, 0.22, 0.13),
    ambient=(0.10, 0.065, 0.035),
    specular=(0.10, 0.08, 0.06),
    shininess=12.0,
)
AIRLINE_MATERIAL = _material(
    diffuse=(0.68, 0.74, 0.78),
    ambient=(0.10, 0.12, 0.14),
    specular=(0.70, 0.75, 0.80),
    shininess=75.0,
    alpha=0.72,
)


def _combine(meshes: Iterable[Mesh]) -> Mesh:
    positions: list[list[float]] = []
    triangles: list[list[int]] = []
    for mesh_positions, mesh_triangles in meshes:
        offset = len(positions)
        positions.extend(mesh_positions)
        triangles.extend(
            [[a + offset, b + offset, c + offset] for a, b, c in mesh_triangles]
        )
    return positions, triangles


def _pouch_radius(u: float, max_radius: float) -> float:
    """Pinched fabric seam at each end with a soft, pillow-like belly."""
    seam_radius = max_radius * 0.34
    bulge = math.sin(math.pi * u) ** 0.58
    return seam_radius + (max_radius - seam_radius) * bulge


def _pillow_mesh(
    cx: float,
    cy: float,
    z0: float,
    z1: float,
    max_radius: float,
    axial_segments: int = 9,
    radial_segments: int = 24,
) -> Mesh:
    positions: list[list[float]] = []
    triangles: list[list[int]] = []

    for iz in range(axial_segments + 1):
        u = iz / axial_segments
        z = z0 + (z1 - z0) * u
        base_radius = _pouch_radius(u, max_radius)
        for ia in range(radial_segments):
            angle = 2.0 * math.pi * ia / radial_segments
            # A very small four-lobed variation keeps the cross-section from
            # reading as a machined tube while remaining visually smooth.
            radius = base_radius * (1.0 + 0.025 * math.cos(4.0 * angle))
            positions.append(
                [cx + radius * math.cos(angle), cy + radius * math.sin(angle), z]
            )

    for iz in range(axial_segments):
        lower = iz * radial_segments
        upper = (iz + 1) * radial_segments
        for ia in range(radial_segments):
            nxt = (ia + 1) % radial_segments
            triangles.append([lower + ia, lower + nxt, upper + nxt])
            triangles.append([lower + ia, upper + nxt, upper + ia])

    bottom_center = len(positions)
    positions.append([cx, cy, z0])
    top_center = len(positions)
    positions.append([cx, cy, z1])
    top_ring = axial_segments * radial_segments
    for ia in range(radial_segments):
        nxt = (ia + 1) % radial_segments
        triangles.append([bottom_center, nxt, ia])
        triangles.append([top_center, top_ring + ia, top_ring + nxt])

    return positions, triangles


def _carrier_panel_mesh(
    cx: float,
    cy: float,
    z0: float,
    z1: float,
    max_radius: float,
    outward_angle: float,
    axial_segments: int = 8,
    angular_segments: int = 6,
) -> Mesh:
    """Two blue fabric cradle panels along the sides of one black pouch."""
    patches: list[Mesh] = []
    panel_half_angle = math.radians(30.0)

    for side in (-1.0, 1.0):
        centre_angle = outward_angle + side * math.pi * 0.5
        positions: list[list[float]] = []
        triangles: list[list[int]] = []

        for iz in range(axial_segments + 1):
            u = iz / axial_segments
            z = z0 + (z1 - z0) * u
            # Sit just above the black fabric to avoid z-fighting.
            radius = _pouch_radius(u, max_radius) + 0.0006
            for ia in range(angular_segments + 1):
                v = ia / angular_segments
                angle = centre_angle - panel_half_angle + 2.0 * panel_half_angle * v
                positions.append(
                    [cx + radius * math.cos(angle), cy + radius * math.sin(angle), z]
                )

        row = angular_segments + 1
        for iz in range(axial_segments):
            lower = iz * row
            upper = (iz + 1) * row
            for ia in range(angular_segments):
                triangles.append([lower + ia, lower + ia + 1, upper + ia + 1])
                triangles.append([lower + ia, upper + ia + 1, upper + ia])

        patches.append((positions, triangles))

    return _combine(patches)


def _annular_disc_mesh(
    z: float,
    inner_radius: float,
    outer_radius: float,
    half_height: float,
    segments: int = 48,
) -> Mesh:
    positions: list[list[float]] = []
    triangles: list[list[int]] = []

    # Four rings: lower/upper × inner/outer.
    for z_offset in (-half_height, half_height):
        for radius in (inner_radius, outer_radius):
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                positions.append(
                    [radius * math.cos(angle), radius * math.sin(angle), z + z_offset]
                )

    low_inner = 0
    low_outer = segments
    high_inner = 2 * segments
    high_outer = 3 * segments
    for i in range(segments):
        nxt = (i + 1) % segments

        # Upper and lower annular faces.
        triangles.extend([
            [high_inner + i, high_outer + i, high_outer + nxt],
            [high_inner + i, high_outer + nxt, high_inner + nxt],
            [low_inner + i, low_outer + nxt, low_outer + i],
            [low_inner + i, low_inner + nxt, low_outer + nxt],
            # Outer rim.
            [low_outer + i, low_outer + nxt, high_outer + nxt],
            [low_outer + i, high_outer + nxt, high_outer + i],
            # Inner rim.
            [low_inner + i, high_inner + nxt, low_inner + nxt],
            [low_inner + i, high_inner + i, high_inner + nxt],
        ])

    return positions, triangles


def _tube_mesh(
    axis: str,
    centre: tuple[float, float, float],
    half_length: float,
    radius: float,
    segments: int = 20,
) -> Mesh:
    positions: list[list[float]] = []
    triangles: list[list[int]] = []
    cx, cy, cz = centre

    for end in (-1.0, 1.0):
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            ca = math.cos(angle)
            sa = math.sin(angle)
            if axis == "x":
                positions.append([cx + end * half_length, cy + radius * ca, cz + radius * sa])
            elif axis == "y":
                positions.append([cx + radius * ca, cy + end * half_length, cz + radius * sa])
            elif axis == "z":
                positions.append([cx + radius * ca, cy + radius * sa, cz + end * half_length])
            else:
                raise ValueError(f"unsupported tube axis: {axis}")

    for i in range(segments):
        nxt = (i + 1) % segments
        triangles.append([i, nxt, segments + nxt])
        triangles.append([i, segments + nxt, segments + i])

    first_cap = len(positions)
    last_cap = first_cap + 1
    if axis == "x":
        positions.extend([
            [cx - half_length, cy, cz],
            [cx + half_length, cy, cz],
        ])
    elif axis == "y":
        positions.extend([
            [cx, cy - half_length, cz],
            [cx, cy + half_length, cz],
        ])
    else:
        positions.extend([
            [cx, cy, cz - half_length],
            [cx, cy, cz + half_length],
        ])
    for i in range(segments):
        nxt = (i + 1) % segments
        triangles.append([first_cap, nxt, i])
        triangles.append([last_cap, segments + i, segments + nxt])

    return positions, triangles


def _sphere_mesh(
    centre: tuple[float, float, float],
    radius: float,
    latitude_segments: int = 8,
    longitude_segments: int = 16,
) -> Mesh:
    positions: list[list[float]] = []
    triangles: list[list[int]] = []
    cx, cy, cz = centre

    positions.append([cx, cy, cz - radius])
    for lat in range(1, latitude_segments):
        polar = math.pi * lat / latitude_segments
        ring_radius = radius * math.sin(polar)
        z = cz - radius * math.cos(polar)
        for lon in range(longitude_segments):
            angle = 2.0 * math.pi * lon / longitude_segments
            positions.append(
                [cx + ring_radius * math.cos(angle), cy + ring_radius * math.sin(angle), z]
            )
    north = len(positions)
    positions.append([cx, cy, cz + radius])

    first_ring = 1
    for lon in range(longitude_segments):
        nxt = (lon + 1) % longitude_segments
        triangles.append([0, first_ring + lon, first_ring + nxt])

    for lat in range(latitude_segments - 2):
        lower = 1 + lat * longitude_segments
        upper = lower + longitude_segments
        for lon in range(longitude_segments):
            nxt = (lon + 1) % longitude_segments
            triangles.append([lower + lon, upper + lon, upper + nxt])
            triangles.append([lower + lon, upper + nxt, lower + nxt])

    last_ring = 1 + (latitude_segments - 2) * longitude_segments
    for lon in range(longitude_segments):
        nxt = (lon + 1) % longitude_segments
        triangles.append([north, last_ring + nxt, last_ring + lon])

    return positions, triangles


def _box_mesh(
    centre: tuple[float, float, float],
    size: tuple[float, float, float],
) -> Mesh:
    cx, cy, cz = centre
    hx, hy, hz = (component * 0.5 for component in size)
    positions = [
        [cx + sx * hx, cy + sy * hy, cz + sz * hz]
        for sz in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sx in (-1.0, 1.0)
    ]
    triangles = [
        [0, 2, 3], [0, 3, 1],  # bottom
        [4, 5, 7], [4, 7, 6],  # top
        [0, 1, 5], [0, 5, 4],
        [2, 6, 7], [2, 7, 3],
        [0, 4, 6], [0, 6, 2],
        [1, 3, 7], [1, 7, 5],
    ]
    return positions, triangles


def _add_deformable_visual(
    arm,
    name: str,
    mesh: Mesh,
    material: str,
):
    positions, triangles = mesh
    node = arm.addChild(name)
    node.addObject(
        "OglModel",
        name="model",
        position=positions,
        triangles=triangles,
        material=material,
        updateNormals=True,
    )
    node.addObject(
        "BarycentricMapping",
        input="@../dofs",
        output="@model",
    )
    return node


def _add_static_visual(root, name: str, mesh: Mesh, material: str):
    positions, triangles = mesh
    node = root.addChild(name)
    node.addObject(
        "OglModel",
        name="model",
        position=positions,
        triangles=triangles,
        material=material,
        updateNormals=False,
    )
    return node


def add_robot_visual(root, arm, cfg: ArmConfig) -> dict[str, object]:
    """Add the photo-matched exterior while leaving the FEM model untouched."""
    pouch_meshes: list[Mesh] = []
    carrier_meshes: list[Mesh] = []
    airline_meshes: list[Mesh] = []

    for azimuth_deg in cfg.col_azimuths_deg():
        outward_angle = math.radians(azimuth_deg)
        cx = cfg.col_offset * math.cos(outward_angle)
        cy = cfg.col_offset * math.sin(outward_angle)
        for level in range(cfg.n_levels):
            z0 = level * cfg.level_height + cfg.pouch_gap * 0.5
            z1 = (level + 1) * cfg.level_height - cfg.pouch_gap * 0.5
            pouch_meshes.append(
                _pillow_mesh(cx, cy, z0, z1, max_radius=cfg.col_radius)
            )
            carrier_meshes.append(
                _carrier_panel_mesh(
                    cx,
                    cy,
                    z0,
                    z1,
                    max_radius=cfg.col_radius,
                    outward_angle=outward_angle,
                )
            )
            # One short clear airline stub per pouch makes the independent
            # pneumatic routing visible without drawing metres of loose hose.
            port_radius = cfg.col_offset + cfg.col_radius + 0.006
            port_centre = (
                port_radius * math.cos(outward_angle),
                port_radius * math.sin(outward_angle),
                0.5 * (z0 + z1),
            )
            port_axis = "x" if abs(math.cos(outward_angle)) > 0.5 else "y"
            airline_meshes.append(
                _tube_mesh(port_axis, port_centre, half_length=0.007, radius=0.00125)
            )

    # Six thin rings: mount, four inter-level separators, and the tip disc.
    ring_mesh = _combine(
        _annular_disc_mesh(
            z=level * cfg.level_height,
            inner_radius=0.006,
            outer_radius=0.045,
            half_height=0.00115,
        )
        for level in range(cfg.n_levels + 1)
    )

    marker_z = cfg.arm_length + cfg.tip_offset
    bar_half_length = cfg.tip_arm
    marker_bars = _combine([
        _tube_mesh(
            "z",
            (0.0, 0.0, cfg.arm_length + cfg.tip_offset * 0.5),
            cfg.tip_offset * 0.5,
            0.004,
        ),
        _tube_mesh("x", (0.0, 0.0, marker_z), bar_half_length, 0.0025),
        _tube_mesh("y", (0.0, 0.0, marker_z), bar_half_length, 0.0025),
    ])
    marker_balls = _combine(
        _sphere_mesh(centre, radius)
        for centre, radius in [
            ((0.0, 0.0, marker_z), 0.0055),
            ((bar_half_length, 0.0, marker_z), 0.0048),
            ((-bar_half_length, 0.0, marker_z), 0.0048),
            ((0.0, bar_half_length, marker_z), 0.0048),
            ((0.0, -bar_half_length, marker_z), 0.0048),
        ]
    )

    visuals = {
        "pouches": _add_deformable_visual(
            arm, "BlackFabricPouches", _combine(pouch_meshes), POUCH_MATERIAL
        ),
        "carriers": _add_deformable_visual(
            arm, "BlueFabricCarriers", _combine(carrier_meshes), BLUE_FABRIC_MATERIAL
        ),
        "airlines": _add_deformable_visual(
            arm, "IndividualAirlinePorts", _combine(airline_meshes), AIRLINE_MATERIAL
        ),
        "rings": _add_deformable_visual(
            arm, "ConnectorRings", ring_mesh, PLATE_MATERIAL
        ),
        "marker_bars": _add_deformable_visual(
            arm, "TipMarkerBars", marker_bars, MARKER_BAR_MATERIAL
        ),
        "marker_balls": _add_deformable_visual(
            arm, "TipMarkerBalls", marker_balls, MARKER_BALL_MATERIAL
        ),
    }

    # The hardware hangs below the plywood on four black mounting standoffs.
    # These are fixed rig components rather than part of the deformable arm.
    mount_z = -0.105
    visuals["mount"] = _add_static_visual(
        root,
        "MountPlate",
        _box_mesh((0.0, 0.0, mount_z), (0.150, 0.150, 0.009)),
        MOUNT_MATERIAL,
    )
    standoff_top = mount_z + 0.0045
    standoff_bottom = -0.001
    standoff_height = standoff_bottom - standoff_top
    standoff_z = 0.5 * (standoff_top + standoff_bottom)
    visuals["mount_standoffs"] = _add_static_visual(
        root,
        "MountStandoffs",
        _combine(
            _box_mesh((x, y, standoff_z), (0.008, 0.008, standoff_height))
            for x in (-0.032, 0.032)
            for y in (-0.032, 0.032)
        ),
        MARKER_BAR_MATERIAL,
    )
    return visuals


__all__ = ["add_robot_visual"]
