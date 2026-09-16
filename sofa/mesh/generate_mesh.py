"""
generate_mesh.py — Gmsh tetrahedral mesh for the fabric pneumatic soft arm.

Geometry
--------
The arm is modelled as a hollow cylinder (the silicone/fabric body) with four
cylindrical longitudinal voids — one per pneumatic column (East / North / West
/ South).  Each void represents the inflatable chamber that, when pressurised,
elongates one side of the arm and causes bending in that direction.

              N (void 1)
              |  r_void = 18 mm
    W --------+-------- E (void 0)    centre-to-void = 28 mm
              |
              S (void 3)

Mesh parameters are exposed as constants at the top of the file so they can be
tuned without touching the Gmsh API calls.

Output
------
    sofa/mesh/soft_arm.msh   (Gmsh MSH4 format — SOFA reads it via GmshReader)

Usage
-----
    python sofa/mesh/generate_mesh.py            # writes soft_arm.msh
    python sofa/mesh/generate_mesh.py --view     # also launches Gmsh GUI

Requirements
------------
    pip install gmsh
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Mesh parameters — edit here, not in the Gmsh API calls below
# ---------------------------------------------------------------------------

# Arm geometry (must match ArmConfig in sofa/arm_config.py)
ARM_LENGTH:    float = 0.220   # [m]  total arm length at rest
ARM_RADIUS:    float = 0.046   # [m]  outer radius of the arm body
                               #      ≈ col_offset (0.028) + col_radius (0.018)
COL_OFFSET:    float = 0.028   # [m]  column axis distance from arm centre
COL_RADIUS:    float = 0.015   # [m]  column cavity radius (slightly < hardware
                               #      col_radius so wall thickness stays ≥ 3 mm)
N_COLS:        int   = 4       # number of columns (cavities)
COL_AZIMUTH0:  float = 0.0     # [deg] azimuth of column 0 (East)

# Mesh element sizes
MESH_SIZE_OUTER: float = 0.008  # [m]  element size on outer surface
MESH_SIZE_INNER: float = 0.005  # [m]  element size on cavity walls
MESH_SIZE_ENDCAP: float = 0.006 # [m]  element size on top/bottom faces

# Output path (relative to repository root)
OUTPUT_PATH = Path(__file__).parent / "soft_arm.msh"

# Physical group tags (used by SOFA scene to reference boundaries)
# These integers are written into the .msh file and referenced in soft_arm_scene.py
TAG_BODY    = 1   # volume: silicone/fabric body
TAG_BASE    = 2   # surface: top face (fixed / mount plate)
TAG_TIP     = 3   # surface: bottom face (free tip)
TAG_OUTER   = 4   # surface: outer lateral surface
TAG_COL0    = 10  # surface: column 0 (East) cavity wall  }
TAG_COL1    = 11  # surface: column 1 (North) cavity wall } pressurised
TAG_COL2    = 12  # surface: column 2 (West)  cavity wall }  in SOFA
TAG_COL3    = 13  # surface: column 3 (South) cavity wall }


def build_mesh(view: bool = False) -> None:
    """Generate the tetrahedral mesh and write it to OUTPUT_PATH."""
    try:
        import gmsh
    except ImportError:
        print(
            "ERROR: gmsh Python package not found.\n"
            "Install with:  pip install gmsh\n"
            "Then re-run:   python sofa/mesh/generate_mesh.py"
        )
        sys.exit(1)

    gmsh.initialize()
    gmsh.model.add("soft_arm")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)   # Frontal-Delaunay 3D

    # ── Outer arm cylinder ─────────────────────────────────────────────────
    outer_tag = gmsh.model.occ.addCylinder(
        0, 0, 0,           # base centre
        0, 0, ARM_LENGTH,  # axis vector (arm points +Z upward, base at Z=0)
        ARM_RADIUS,
    )

    # ── Column cavity cylinders ────────────────────────────────────────────
    col_azimuths_deg = [COL_AZIMUTH0 + 90.0 * i for i in range(N_COLS)]
    col_tags = []
    for phi_deg in col_azimuths_deg:
        phi = math.radians(phi_deg)
        cx = COL_OFFSET * math.cos(phi)
        cy = COL_OFFSET * math.sin(phi)
        tag = gmsh.model.occ.addCylinder(
            cx, cy, -0.001,             # start slightly below base (clean boolean)
            0,  0, ARM_LENGTH + 0.002,  # slightly longer than body
            COL_RADIUS,
        )
        col_tags.append(tag)

    # ── Boolean: subtract cavities from body ──────────────────────────────
    body_dimtag = [(3, outer_tag)]
    cavity_dimtags = [(3, t) for t in col_tags]

    result, tool_map = gmsh.model.occ.cut(
        body_dimtag, cavity_dimtags,
        tag=-1, removeObject=True, removeTool=True,
    )
    gmsh.model.occ.synchronize()

    # ── Physical groups ────────────────────────────────────────────────────
    # Volume
    vol_tags = [t for d, t in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, vol_tags, TAG_BODY, name="ArmBody")

    # Identify surfaces by their centroid Z and radial position
    base_surfs, tip_surfs, outer_surfs = [], [], []
    col_surfs: list[list[int]] = [[] for _ in range(N_COLS)]

    for _, s in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(2, s)
        cx_s, cy_s, cz_s = com

        r_from_axis = math.hypot(cx_s, cy_s)
        z_frac = cz_s / ARM_LENGTH  # 0 = base, 1 = tip

        if z_frac < 0.05:
            base_surfs.append(s)
        elif z_frac > 0.95:
            tip_surfs.append(s)
        elif r_from_axis > ARM_RADIUS * 0.85:
            outer_surfs.append(s)
        else:
            # Cavity wall — identify by azimuth of centroid
            phi_s = math.atan2(cy_s, cx_s)
            phi_deg_s = math.degrees(phi_s) % 360.0
            # Find closest column azimuth
            best_col = min(
                range(N_COLS),
                key=lambda i: abs((phi_deg_s - col_azimuths_deg[i]) % 360.0)
            )
            col_surfs[best_col].append(s)

    gmsh.model.addPhysicalGroup(2, base_surfs,  TAG_BASE,  name="Base")
    gmsh.model.addPhysicalGroup(2, tip_surfs,   TAG_TIP,   name="Tip")
    gmsh.model.addPhysicalGroup(2, outer_surfs, TAG_OUTER, name="OuterSurface")
    for i, surfs in enumerate(col_surfs):
        if surfs:
            gmsh.model.addPhysicalGroup(
                2, surfs, TAG_COL0 + i, name=f"CavityCol{i}"
            )

    # ── Mesh sizing ────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE_INNER * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE_OUTER)

    # Finer mesh on cavity walls
    for i, surfs in enumerate(col_surfs):
        for s in surfs:
            gmsh.model.mesh.setSize(
                gmsh.model.getBoundary([(2, s)], oriented=False), MESH_SIZE_INNER
            )

    # ── Generate and save ──────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(OUTPUT_PATH))

    n_nodes = gmsh.model.mesh.getNodes()[0].shape[0]
    print(f"\n[mesh] Wrote {OUTPUT_PATH}  ({n_nodes} nodes)")
    print(f"[mesh] Physical groups:")
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        print(f"  dim={dim}  tag={tag:3d}  name='{name}'")

    if view:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tetrahedral mesh for soft arm.")
    parser.add_argument("--view", action="store_true", help="Open Gmsh GUI after meshing")
    args = parser.parse_args()
    build_mesh(view=args.view)
