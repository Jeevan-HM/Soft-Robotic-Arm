# Soft Robotic Arm — Parameter Reference

> MuJoCo digital twin of the fabric pneumatic soft arm.
> Source: [`arm_model.py`](file:///Users/g1/Developer/simulation/arm_model.py) · [`soft_arm_sim.py`](file:///Users/g1/Developer/simulation/soft_arm_sim.py)

---

## Architecture Overview

```
4 segment columns × 5 floor levels = 20 pressure channels
Total DOF: 15 (5 slides + 5×2 hinges)
```

```
     N (col 1)
     |  r = 18 mm
W ---+--- E (col 0)     centre-to-column = 28 mm
     |                   overall diameter ≈ 92 mm
     S (col 3)
```

| Column | Direction | Azimuth |
|--------|-----------|---------|
| 0      | East      | 0°      |
| 1      | North     | 90°     |
| 2      | West      | 180°    |
| 3      | South     | 270°    |

Each column has 5 floor levels (pouches) stacked vertically. Pressurising a column elongates that side, bending the arm toward that column's direction.

---

## 1. Structural / Geometric Properties

| Parameter | Field | Value | Unit | Description |
|-----------|-------|-------|------|-------------|
| Arm length | `length` | 0.22 | m | Total arm length at rest (~22 cm) |
| Column offset | `col_offset` | 0.028 | m | Column axis distance from arm centre |
| Column radius | `col_radius` | 0.018 | m | Column outer radius |
| Level height | `level_height()` | 0.044 | m | Derived: `length / n_pouches` (22 cm / 5) |
| Number of segments | `n_segments` | 4 | — | Pneumatic columns (E / N / W / S) |
| Number of pouches | `n_pouches` | 5 | — | Floor levels per column |
| Total channels | `n_channels` | 20 | — | Derived: `n_segments × n_pouches` |
| Column 0 azimuth | `seg_azimuth0_deg` | 0.0 | deg | Azimuth of column 0 |
| Hang direction | `hang_down` | True | — | Arm hangs downward from mount plate |

---

## 2. Mass Properties

| Parameter | Field | Value | Unit | Description |
|-----------|-------|-------|------|-------------|
| Total moving mass | `mass` | 0.35 | kg | Fabric + fittings |
| Mass per level | derived | 0.07 | kg | `mass / n_pouches` |
| Column mass per level | derived | ~0.0149 | kg | `(level_mass × 0.85) / n_segments` |
| Core mass per level | derived | ~0.0105 | kg | `level_mass × 0.15` |
| Tip mass | `tip_mass` | 0.08 | kg | OptiTrack marker frame at tip |
| Tip marker arm | `tip_arm` | 0.07 | m | Marker cross half-length |
| Connector ring mass | hardcoded | 0.006 | kg | Per-level rigid disc between levels |
| Mount disc mass | hardcoded | 0.010 | kg | Top connector disc at mount |
| Tip ring mass | hardcoded | 0.010 | kg | Tip disc |

---

## 3. Passive Mechanical / Elasticity Properties

| Parameter | Field | Value | Unit | Description |
|-----------|-------|-------|------|-------------|
| Bending stiffness | `base_stiffness` | 0.30 | N·m/rad | Per bending hinge joint (at zero pre-inflation) |
| Bending damping | `base_damping` | 0.08 | N·m·s/rad | Per bending hinge joint |
| Axial stiffness | `axial_stiffness` | 600.0 | N/m | Per level slide joint |
| Axial damping | `axial_damping` | 30.0 | N·s/m | Per level slide joint |
| Stiffness scaling | `stiffness_per_psi` | 0.414 | 1/psi | Stiffness scaling factor with pre-inflation |
| Axial slide range | hardcoded | −0.005 to 0.030 | m | Joint limits on slide DOF |

### Pre-Inflation Stiffness Modulation

Pre-inflation (`P_pre`) modulates joint properties symmetrically:

```
scale = 1 + stiffness_per_psi × P_pre
     = 1 + 0.414 × P_pre

Joint stiffness:  k = k₀ × scale
Joint damping:    d = d₀ × √(scale)
```

| P_pre (psi) | Scale Factor | Effective Bending Stiffness (N·m/rad) | Effective Bending Damping (N·m·s/rad) |
|-------------|--------------|---------------------------------------|---------------------------------------|
| 0.0         | 1.000        | 0.300                                 | 0.080                                 |
| 1.0         | 1.414        | 0.424                                 | 0.095                                 |
| 1.5         | 1.621        | 0.486                                 | 0.102                                 |
| 3.0         | 2.242        | 0.673                                 | 0.120                                 |
| 5.0         | 3.070        | 0.921                                 | 0.140                                 |
| 10.0        | 5.140        | 1.542                                 | 0.181                                 |

---

## 4. Pressure → Force Mapping

| Parameter | Field | Value | Unit | Description |
|-----------|-------|-------|------|-------------|
| Pressure gain | `pressure_gain` | 0.552 | N/psi | Bending force per column per level |
| Moment arm | `moment_arm` | 0.028 | m | Lever arm for bending (= `col_offset`) |
| Extension gain | `extension_gain` | 0.034 | N/psi | Axial force per column per level |
| Max pressure | `p_max` | 10.0 | psi | Maximum pouch pressure |

### Bending Moment Equation

Per floor level `k`:

```
M[k] = pressure_gain × moment_arm × Σ_s P[s,k] × (-sin φ_s, cos φ_s)
     = 0.552 × 0.028 × Σ_s P[s,k] × (-sin φ_s, cos φ_s)
     = 0.01546 [N·m/psi] × Σ_s P[s,k] × direction[s]
```

### Axial Force Equation

Per floor level `k`:

```
f[k] = extension_gain × Σ_s P[s,k]
     = 0.034 × Σ_s P[s,k]
```

---

## 5. Pneumatic Properties

| Parameter | Field | Value | Unit | Description |
|-----------|-------|-------|------|-------------|
| Time constant | `tau_pneumatic` | 0.12 | s | First-order fill/vent time constant |
| Max pressure | `p_max` | 10.0 | psi | Absolute maximum pouch pressure |

### First-Order Pressure Dynamics

Each pouch pressure follows its command via a discrete first-order filter:

```
α = timestep / max(τ_pneumatic, timestep)
  = 0.001 / 0.12
  ≈ 0.00833

p_actual += α × (p_cmd − p_actual)    (per physics substep)
```

---

## 6. Sensor Model Properties

| Parameter | Field | Value | Unit | Description |
|-----------|-------|-------|------|-------------|
| Sensor noise | `sensor_noise_psi` | 0.007 | psi | Gaussian noise σ on pressure readings |
| Curvature coupling | `curvature_coupling` | 4.0 | — | Bending-to-pressure coupling gain |
| Extension coupling | `extension_coupling` | 30.0 | — | Extension-to-pressure coupling gain |

### Sensor Equation (Parallel Topology)

```
p_measured = p_actual
           + curvature_coupling × κ        (bending toward col s squeezes its pouches)
           + extension_coupling × extension (axial stretch affects readings)
           + N(0, sensor_noise²)            (Gaussian noise)
```

Where:
- `κ[s, k]` = bending at level `k` projected onto column `s`'s direction
- `extension` = slide joint position at level `k`

---

## 7. Simulation Properties

| Parameter | Field | Value | Unit | Description |
|-----------|-------|-------|------|-------------|
| Physics timestep | `timestep` | 0.001 | s | MuJoCo integration step (1 ms) |
| Control rate | `control_hz` | 100.0 | Hz | Control tick frequency |
| Control dt | derived | 0.01 | s | `1 / control_hz` |
| Substeps per tick | derived | 10 | — | `control_dt / timestep` |
| Integrator | hardcoded | `implicitfast` | — | MuJoCo integrator type |
| Gravity | hardcoded | (0, 0, −9.81) | m/s² | Standard gravity |

---

## 8. Geometry Thickness Values

| Component | Dimension | Value | Unit | Description |
|-----------|-----------|-------|------|-------------|
| Column capsule | radius | 0.018 | m | Outer radius of each pneumatic column |
| Central core | radius | 0.005 | m | Structural core cylinder |
| Central core | half-height | ~0.0211 | m | `level_height × 0.48` |
| Connector ring | half-height | 0.0025 | m | Rigid disc between levels |
| Connector ring | radius | ~0.049 | m | `col_offset + col_radius + 0.003` |
| Mount plate | half-size (XY) | 0.150 | m | Plywood mount plate |
| Mount plate | half-size (Z) | 0.009 | m | Plywood mount plate thickness |
| Mount disc | half-height | 0.003 | m | Top connector disc |
| Support posts | radius | 0.008 | m | Aluminium extrusion |
| Support posts | length | 0.30 | m | Corner posts |
| Tip cross-bar | radius | 0.004 | m | OptiTrack marker bars |
| Tip sphere (centre) | radius | 0.007 | m | Central marker ball |
| Tip sphere (ends) | radius | 0.005 | m | Corner marker balls |

---

## 9. Visual Properties

| Column | Colour | RGBA |
|--------|--------|------|
| 0 (East) | Orange | `0.92 0.55 0.08 1` |
| 1 (North) | Green | `0.10 0.68 0.18 1` |
| 2 (West) | Red | `0.82 0.18 0.08 1` |
| 3 (South) | Blue | `0.08 0.28 0.90 1` |

| Component | Colour | RGBA |
|-----------|--------|------|
| Connector rings | Dark grey | `0.10 0.10 0.12 1` |
| Core cylinder | Dark grey | `0.12 0.12 0.14 1` |
| Tip bars | Dark grey | `0.12 0.12 0.14 1` |
| Tip spheres | Light grey | `0.85 0.85 0.85 1` |
| Tip site | Green | `0.1 0.9 0.3 1` |

---

## 10. Default Rendering Camera

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cam_azimuth` | 135° | Diagonal view showing all 4 columns |
| `cam_elevation` | −20° | Slight downward angle |
| `cam_distance` | 0.80 m | Distance for the ~22 cm arm |
| `width` | 640 px | Frame width |
| `height` | 480 px | Frame height |

---

## 11. Dynamic Model Characteristics

This is a **fully dynamic** model — not static or kinematic. MuJoCo numerically integrates the full equations of motion at each timestep:

```
M(q) q̈  +  c(q, q̇)  +  k·q  +  d·q̇  =  τ(P)
```

Where:
- `M(q)` — mass matrix (from the 0.35 kg distributed across 5 levels + 0.08 kg tip)
- `c(q, q̇)` — Coriolis and gravitational forces (gravity = 9.81 m/s²)
- `k·q` — passive spring restoring forces (bending stiffness, axial stiffness)
- `d·q̇` — viscous damping forces (bending damping, axial damping)
- `τ(P)` — active forces from pressure commands (bending torques + axial forces)

### Dynamic Elements

| Feature | Source | Description |
|---------|--------|-------------|
| Inertial dynamics | `gravity="0 0 -9.81"` | Masses accelerate under gravity; arm exhibits overshoot and oscillation |
| Time integration | `integrator="implicitfast"` | MuJoCo solves `M(q)q̈ + C(q,q̇) = τ` at 1 kHz |
| Mass distribution | `mass=0.35`, `tip_mass=0.08` | Distributed inertia creates realistic coupling between levels |
| Spring-damper joints | `stiffness`, `damping` on every joint | Passive viscoelastic restoring forces |
| Pneumatic lag | `tau_pneumatic=0.12 s` | First-order pressure dynamics — pressure doesn't change instantaneously |
| Velocity-dependent damping | `dof_damping` | Damping forces proportional to joint velocity |
| Tip velocity sensing | `framelinvel` sensor | Measures instantaneous tip velocity (only meaningful in dynamic model) |
| Pre-inflation stiffness scaling | `stiffness_per_psi` | Joint stiffness/damping modulated in real time via pre-inflation |

### Dynamic vs. Static/Kinematic

| Model type | Behaviour | This simulation |
|------------|-----------|-----------------|
| **Static** | `tip_pos = f(P)` — instant equilibrium, no time evolution | ✗ |
| **Kinematic** | Prescribed joint trajectories, no forces | ✗ |
| **Dynamic** | Forces → acceleration → velocity → position; overshoots, oscillates, settles | ✓ |

A step pressure input produces **overshoot and ringing** before settling — characteristic of a second-order dynamic system governed by mass, stiffness, and damping.
