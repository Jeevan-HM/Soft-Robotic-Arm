# Soft Robotic Arm — Model Construction Guide

> How the MuJoCo digital twin is built: joints, shapes, bodies, and assembly.
> Source: [`arm_model.py`](../arm_model.py) · [`build_arm_xml()`](../arm_model.py#L100)

---

## Overall Concept — "4 Buildings, 5 Floors Each"

The arm is a **serial chain of 5 nested bodies** (one per floor level), each hanging inside the one above it. Think of it like a telescoping structure — the arm hangs downward from a fixed mount plate, with each floor level 44 mm below the previous one.

```
MOUNT PLATE  (fixed to world, z = 0.50 m)
    │
  level 0  ←── 3 joints + 4 column capsules + ring + core
    │
  level 1
    │
  level 2
    │
  level 3
    │
  level 4
    │
  tip_disc + tip_frame  (OptiTrack marker cross)
```

Each level is a MuJoCo `<body>` nested inside its parent. Their relative position is `zdir × h` (downward, since `hang_down = True`), so each level hangs **44 mm** below the one above.

---

## Joints — 3 DOF per Level (15 Total)

Each floor level has exactly **3 joints**, chained together in sequence:

```
Level k body
  ├── joint: ext{k}   → SLIDE  (axial up/down)
  ├── joint: bx{k}    → HINGE  (bend about X axis)
  └── joint: by{k}    → HINGE  (bend about Y axis)
```

| Joint | Type | Axis | Range | Stiffness | Damping | Purpose |
|-------|------|------|-------|-----------|---------|---------|
| `ext0` … `ext4` | Slide | Z `(0 0 -1)` | −5 mm to +30 mm | 600 N/m | 30 N·s/m | Axial extension / compression |
| `bx0` … `bx4` | Hinge | X `(1 0 0)` | unlimited | 0.30 N·m/rad | 0.08 N·m·s/rad | Bending forward / backward |
| `by0` … `by4` | Hinge | Y `(0 1 0)` | unlimited | 0.30 N·m/rad | 0.08 N·m·s/rad | Bending left / right |

> **Note:** The `bx` + `by` pair together form a **universal joint** — they allow the arm to bend in any direction in 3D space.

All joints have `springref="0"`, meaning they are spring-loaded back to their zero (straight) configuration.

---

## Geometry — 4 Shapes per Level

Each level body contains **4 geometry pieces**. All have collision disabled (`contype="0"`) since this is a soft body without contact physics.

### 1. Connector Ring — `ring{k}` (Cylinder)

```
type  : cylinder
size  : radius = 49 mm,  half-height = 2.5 mm
color : dark grey  (0.10 0.10 0.12)
mass  : 6 g
pos   : (0, 0, 0)  — top of the level
```

A flat disc representing the rigid aluminium ring that connects adjacent floor levels.

---

### 2. Four Column Capsules — `col{s}_l{k}` (Capsule × 4)

```
type     : capsule
size     : radius = 18 mm
fromto   : (cx, cy, 0)  →  (cx, cy, −44 mm)   — full level height
positions: 28 mm from arm centre, at azimuths 0° / 90° / 180° / 270°
```

One capsule per pneumatic column, placed around the centre:

```
         col 1 — North (green)
              │
col 2 ────────┼──────── col 0       ← 28 mm from centre axis
(West, red)   │       (East, orange)
              │
         col 3 — South (blue)
```

| Column | Direction | Colour | RGBA |
|--------|-----------|--------|------|
| 0 | East | Orange | `0.92 0.55 0.08 1` |
| 1 | North | Green | `0.10 0.68 0.18 1` |
| 2 | West | Red | `0.82 0.18 0.08 1` |
| 3 | South | Blue | `0.08 0.28 0.90 1` |

---

### 3. Central Core — `core{k}` (Cylinder)

```
type        : cylinder
size        : radius = 5 mm,  half-height = 21 mm
color       : dark grey  (0.12 0.12 0.14)
mass        : 15% of level mass ≈ 10.5 g
pos         : (0, 0, −22 mm)  — centre of level height
```

A thin structural spine running down the centre of each level. Carries 15% of each level's mass.

---

## Tip — 2 Extra Bodies at the Bottom

Below level 4, two additional bodies represent the **OptiTrack motion-capture marker cross**:

```
tip_disc  — flat cylinder ring  (same radius as connector rings)
  └── tip_frame  (offset 12 mm further down)
        ├── tip_bar_x    — capsule along X axis  (r=4mm)
        ├── tip_bar_y    — capsule along Y axis  (r=4mm)
        ├── tip_ball_c   — sphere at centre  (r=7mm)
        ├── tip_ball_px/nx/py/ny — 4 corner spheres  (r=5mm)
        └── site "tip"   ← sensor measurement point  (green)
```

The `tip` **site** is where all sensor readings are measured — tip position, orientation, and velocity.

---

## Mount — Fixed to World

At the top, a static `mount` body is fixed at `z = 0.50 m`. It has **no joints** and never moves:

```
mount  (fixed to world, z = 0.50 m)
  ├── mount_plate    — plywood box  (150 × 150 × 9 mm)
  ├── post_fl/fr/bl/br — 4 aluminium corner posts  (r=8mm, h=300mm)
  └── mount_disc     — connector disc to arm  (r=49mm, h=3mm)
```

---

## Full Assembly Hierarchy

```
WORLD
  └── mount  (fixed, z = 0.50 m)
       ├── mount_plate, posts, mount_disc
       │
       └── level 0
            ├── joint: ext0  [slide Z,  k=600 N/m,    d=30 N·s/m]
            ├── joint: bx0   [hinge X,  k=0.30 N·m/rad, d=0.08]
            ├── joint: by0   [hinge Y,  k=0.30 N·m/rad, d=0.08]
            ├── geom:  ring0  [cylinder disc]
            ├── geom:  col0_l0 … col3_l0  [4 coloured capsules]
            ├── geom:  core0  [cylinder spine]
            │
            └── level 1
                 └── level 2
                      └── level 3
                           └── level 4
                                └── tip_disc
                                     └── tip_frame
                                          ├── tip_bar_x, tip_bar_y
                                          ├── tip_ball_c, _px, _nx, _py, _ny
                                          └── site "tip"  ←── sensors here
```

---

## Mass Budget

| Component | Count | Unit Mass | Total |
|-----------|-------|-----------|-------|
| Column capsule | 4 × 5 = 20 | ~14.9 g | ~297 g |
| Core cylinder | 5 | ~10.5 g | ~52 g |
| Connector ring | 5 | 6 g | 30 g |
| Mount / tip discs | 2 | 10 g | 20 g |
| Tip bars (X + Y) | 2 | 24 g | 48 g |
| Tip centre ball | 1 | 32 g | 32 g |
| Tip corner balls | 4 | 1 g | 4 g |
| **Moving arm total** | | | **~0.35 kg** |
| **Tip total** | | | **~0.08 kg** |

---

## Simulator Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `integrator` | `implicitfast` | Stable implicit integration for stiff systems |
| `timestep` | 1 ms | Physics step size |
| `gravity` | `(0, 0, −9.81)` m/s² | Standard Earth gravity |
| Collision | disabled | `contype="0"` on all geoms — no contact physics |

---

## Sensors

Three sensors are attached to the `tip` site and read out via `data.sensordata`:

| Indices | Sensor | Type | Output |
|---------|--------|------|--------|
| `[0:3]` | `tip_pos` | `framepos` | Tip position (x, y, z) in world frame [m] |
| `[3:7]` | `tip_quat` | `framequat` | Tip orientation quaternion (w, x, y, z) |
| `[7:10]` | `tip_vel` | `framelinvel` | Tip linear velocity (vx, vy, vz) [m/s] |
