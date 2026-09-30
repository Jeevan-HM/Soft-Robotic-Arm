"""
generate_mesh.py — Gmsh tetrahedral mesh for the 4-column, 5-level soft arm.

Geometry
--------
The arm body is a hollow cylinder with 20 independent pneumatic pouches:
4 columns (East/North/West/South) × 5 levels (0=base … 4=tip).

Each pouch is a short cylinder aligned with the column axis.  Between
consecutive levels within the same column there is a thin silicone wall
(POUCH_GAP).  This matches the MuJoCo architecture exactly.

Cross-section (top view, Z pointing into page):

              N (col 1)
              ○  r_pouch = 15 mm
    W --------+-------- E (col 0)    centre-to-column = 28 mm
              ○
              S (col 3)

Side view (one column):

  Z = ARM_LENGTH  ─── tip (free)
                  ╔═══╗  level 4 (pouch)
                  ╟───╢  4 mm wall
                  ╔═══╗  level 3
                  ╟───╢
                  ╔═══╗  level 2
                  ╟───╢
                  ╔═══╗  level 1
                  ╟───╢
                  ╔═══╗  level 0
  Z = 0.0     ─── base (fixed / mount plate)

Physical group tags
-------------------
Tag scheme: pouch tag = TAG_COL_BASE + col_index * N_LEVELS + level_index

    col 0 (East)  levels 0-4 → tags 10, 11, 12, 13, 14
    col 1 (North) levels 0-4 → tags 15, 16, 17, 18, 19
    col 2 (West)  levels 0-4 → tags 20, 21, 22, 23, 24
    col 3 (South) levels 0-4 → tags 25, 26, 27, 28, 29

Output
------
    soft-arm/mesh/soft_arm.msh   (Gmsh MSH2 ASCII format)

Usage
-----
    python soft-arm/mesh/generate_mesh.py            # write .msh
    python soft-arm/mesh/generate_mesh.py --view     # write + open Gmsh GUI

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

# Arm geometry (matches ArmConfig in arm_config.py)
ARM_LENGTH:    float = 0.220   # [m]  total arm length
ARM_RADIUS:    float = 0.046   # [m]  outer body radius
COL_OFFSET:    float = 0.028   # [m]  column axis distance from arm centre
COL_RADIUS:    float = 0.015   # [m]  individual pouch radius
N_COLS:        int   = 4       # number of columns
N_LEVELS:      int   = 5       # number of pouches per column (levels)
COL_AZIMUTH0:  float = 0.0     # [deg] azimuth of column 0 (East)

# Pouch geometry
LEVEL_HEIGHT:  float = ARM_LENGTH / N_LEVELS          # 0.044 m per level
POUCH_GAP:     float = 0.004                           # [m] silicone wall between levels
POUCH_HEIGHT:  float = LEVEL_HEIGHT - POUCH_GAP        # 0.040 m per pouch
POUCH_Z_OFFSET: float = POUCH_GAP / 2.0               # 2 mm end wall at base/tip

# Mesh element sizes
MESH_SIZE_OUTER:  float = 0.009   # [m] outer surface
MESH_SIZE_INNER:  float = 0.005   # [m] pouch cavity walls
MESH_SIZE_WALL:   float = 0.003   # [m] thin inter-level walls

# Output path
OUTPUT_PATH = Path(__file__).parent / "soft_arm.msh"

# Physical group tags
TAG_BODY      = 1     # volume: silicone/fabric body
TAG_BASE      = 2     # surface: base face (Z ≈ 0, fixed)
TAG_TIP       = 3     # surface: tip face (Z ≈ ARM_LENGTH, free)
TAG_OUTER     = 4     # surface: outer lateral surface
TAG_COL_BASE  = 10    # first pouch tag; tag = TAG_COL_BASE + col*N_LEVELS + level


def pouch_tag(col: int, level: int) -> int:
    """Physical group tag for a specific pouch."""
    return TAG_COL_BASE + col * N_LEVELS + level


def pouch_name(col: int, level: int) -> str:
    return f"CavityCol{col}_Lvl{level}"


def build_mesh(view: bool = False) -> None:
    """Generate the 20-pouch tetrahedral mesh and write it to OUTPUT_PATH."""
    try:
        import gmsh
    except ImportError:
        print(
            "ERROR: gmsh Python package not found.\n"
            "Install with:  pip install gmsh\n"
        )
        sys.exit(1)

    gmsh.initialize()
    gmsh.model.add("soft_arm_20pouches")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay
    # SOFA's MeshGmshLoader and the scene's lightweight physical-group parser
    # both consume the stable, documented MSH2 ASCII layout.
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)

    col_azimuths_deg = [COL_AZIMUTH0 + 90.0 * i for i in range(N_COLS)]

    # ── Outer arm cylinder ─────────────────────────────────────────────────
    outer_tag = gmsh.model.occ.addCylinder(
        0, 0, 0,
        0, 0, ARM_LENGTH,
        ARM_RADIUS,
    )

    # ── 20 pouch cylinders (4 cols × 5 levels) ────────────────────────────
    pouch_tags = []
    for col in range(N_COLS):
        phi = math.radians(col_azimuths_deg[col])
        cx  = COL_OFFSET * math.cos(phi)
        cy  = COL_OFFSET * math.sin(phi)
        for lvl in range(N_LEVELS):
            z_start = lvl * LEVEL_HEIGHT + POUCH_Z_OFFSET
            tag = gmsh.model.occ.addCylinder(
                cx, cy, z_start,
                0, 0, POUCH_HEIGHT,
                COL_RADIUS,
            )
            pouch_tags.append((col, lvl, tag))

    # ── Boolean: cut all pouches from arm body ─────────────────────────────
    cavity_dimtags = [(3, t) for _, _, t in pouch_tags]
    result, _ = gmsh.model.occ.cut(
        [(3, outer_tag)], cavity_dimtags,
        tag=-1, removeObject=True, removeTool=True,
    )
    gmsh.model.occ.synchronize()

    # ── Physical groups ────────────────────────────────────────────────────
    # Volume
    vol_tags = [t for d, t in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, vol_tags, TAG_BODY, name="ArmBody")

    # Classify surfaces
    base_surfs, tip_surfs, outer_surfs = [], [], []
    # bucket surfaces by which (col, level) they belong to
    pouch_surf_map: dict[tuple[int,int], list[int]] = {
        (c, l): [] for c in range(N_COLS) for l in range(N_LEVELS)
    }

    # Pre-compute pouch centroid positions for classification
    pouch_centroids = []
    for col in range(N_COLS):
        phi = math.radians(col_azimuths_deg[col])
        cx  = COL_OFFSET * math.cos(phi)
        cy  = COL_OFFSET * math.sin(phi)
        for lvl in range(N_LEVELS):
            z_mid = lvl * LEVEL_HEIGHT + POUCH_Z_OFFSET + POUCH_HEIGHT * 0.5
            pouch_centroids.append((col, lvl, cx, cy, z_mid))

    surface_tol = 1e-7
    for _, s in gmsh.model.getEntities(2):
        sx, sy, sz = gmsh.model.occ.getCenterOfMass(2, s)
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s)

        # Use the bounding box rather than centroid thresholds so the end caps
        # of level 0/4 are not mistaken for the arm's base/tip surfaces.
        if abs(zmin) <= surface_tol and abs(zmax) <= surface_tol:
            base_surfs.append(s)
        elif (
            abs(zmin - ARM_LENGTH) <= surface_tol
            and abs(zmax - ARM_LENGTH) <= surface_tol
        ):
            tip_surfs.append(s)
        elif max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) >= ARM_RADIUS - surface_tol:
            outer_surfs.append(s)
        else:
            # Cavity surface — find closest pouch by 3D distance to centroid
            best_col, best_lvl = 0, 0
            best_dist = float("inf")
            for col, lvl, cx, cy, z_mid in pouch_centroids:
                d = math.sqrt((sx-cx)**2 + (sy-cy)**2 + (sz-z_mid)**2)
                if d < best_dist:
                    best_dist = d
                    best_col, best_lvl = col, lvl
            pouch_surf_map[(best_col, best_lvl)].append(s)

    missing = [key for key, surfaces in pouch_surf_map.items() if not surfaces]
    if missing:
        raise RuntimeError(f"No cavity surfaces found for pouches: {missing}")

    gmsh.model.addPhysicalGroup(2, base_surfs,  TAG_BASE,  name="Base")
    gmsh.model.addPhysicalGroup(2, tip_surfs,   TAG_TIP,   name="Tip")
    gmsh.model.addPhysicalGroup(2, outer_surfs, TAG_OUTER, name="OuterSurface")

    for (col, lvl), surfs in pouch_surf_map.items():
        if surfs:
            gmsh.model.addPhysicalGroup(
                2, surfs, pouch_tag(col, lvl), name=pouch_name(col, lvl)
            )

    # ── Mesh sizing ────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE_WALL * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE_OUTER)

    # Finer mesh around pouch walls
    for (col, lvl), surfs in pouch_surf_map.items():
        for s in surfs:
            bnd = gmsh.model.getBoundary([(2, s)], oriented=False)
            if bnd:
                gmsh.model.mesh.setSize(bnd, MESH_SIZE_INNER)

    # ── Generate and save ──────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(OUTPUT_PATH))

    n_nodes = gmsh.model.mesh.getNodes()[0].shape[0]
    print(f"\n[mesh] Wrote {OUTPUT_PATH}  ({n_nodes} nodes)")
    print(f"[mesh] Pouch tags:")
    col_labels = ["East", "North", "West", "South"]
    for col in range(N_COLS):
        for lvl in range(N_LEVELS):
            tag = pouch_tag(col, lvl)
            surfs = pouch_surf_map.get((col, lvl), [])
            print(f"  col {col} ({col_labels[col]:5s}) lvl {lvl} → tag {tag:3d}  ({len(surfs)} surfaces)")

    if view:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 20-pouch soft arm mesh.")
    parser.add_argument("--view", action="store_true", help="Open Gmsh GUI after meshing")
    args = parser.parse_args()
    build_mesh(view=args.view)
