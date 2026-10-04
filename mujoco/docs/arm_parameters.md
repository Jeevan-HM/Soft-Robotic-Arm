# Calibrated Soft Robotic Arm — Parameter Reference

This document describes the single MuJoCo robot model used by the repository.
The machine-readable source of fitted values is
[`calibration.json`](../soft_robotic_arm/data/calibration.json); do not duplicate or tune
values in application code.

The calibration was fitted to 36 recorded robot conditions at 0.1 Hz. It
models four radial pouch columns with five pouches per column. Segment 1
(column 0) is charged and sealed as a pressure reservoir. Segments 2--4
(columns 1--3) accept three commanded pressure setpoints.

## Architecture and command interface

```text
4 columns × 5 pouch levels = 20 simulated pressure states
5 levels × (1 axial + 2 bending joints) = 15 mechanical DOF

          Segment 2 / column 1
                    N
                    |
Segment 3 / col 2 W-+-E Segment 1 / col 0 (sealed)
                    |
                    S
          Segment 4 / column 3
```

| Segment | Column | Azimuth | Role |
|---|---:|---:|---|
| 1 | 0 | 0° | Five-pouch sealed reservoir |
| 2 | 1 | 90° | Commanded actuator `p2` |
| 3 | 2 | 180° | Commanded actuator `p3` |
| 4 | 3 | 270° | Commanded actuator `p4` |

`make_calibrated_sim()` selects the calibrated parallel topology, charges
Segment 1 to 2 psi, and accepts `step([p2, p3, p4])`. To reproduce a
coupled-plumbing run, call
`make_calibrated_sim(topology="coupled", reservoir_pressure_psi=x)`. The
classroom-facing `make_sim()` commands all four segments directly.

## Geometry and mass

| Parameter | `ArmConfig` field | Fitted value | Unit |
|---|---|---:|---|
| Number of columns | `n_segments` | 4 | — |
| Pouches / levels per column | `n_pouches` | 5 | — |
| Rest length | `length` | 0.296 | m |
| Level height | `length / n_pouches` | 0.0592 | m |
| Column centre offset | `col_offset` | 0.028 | m |
| Column radius | `col_radius` | 0.018 | m |
| Moving arm mass | `mass` | 0.35 | kg |
| Tip marker mass | `tip_mass` | 0.08 | kg |
| Tip marker half-span | `tip_arm` | 0.07 | m |
| Pressure moment arm | `moment_arm` | 0.028 | m |
| Hanging orientation | `hang_down` | `true` | — |

The MuJoCo body is a five-level serial chain. Each level has one axial slide
and two orthogonal bending hinges. The four visible column capsules are rigid
geometry attached to each level; pressure is converted into generalized force,
not applied as a deforming surface load.

## Mechanical and pressure-force parameters

| Parameter | `ArmConfig` field | Fitted value | Unit |
|---|---|---:|---|
| Bending force gain | `pressure_gain` | 0.1273999973 | N/psi |
| Axial force gain | `extension_gain` | 0.0918534967 | N/psi |
| Bending stiffness / hinge | `base_stiffness` | 0.4168449461 | N·m/rad |
| Bending damping / hinge | `base_damping` | 0.3629226883 | N·m·s/rad |
| Axial stiffness / level | `axial_stiffness` | 934.2301025 | N/m |
| Axial damping / level | `axial_damping` | 61.50955915 | N·s/m |
| Charge stiffness coefficient | `stiffness_per_psi` | 0.0008948236 | 1/psi |
| Pressure limit | `p_max` | 11.0 | psi |
| Physics timestep | `timestep` | 0.001 | s |

At level `k`, column pressure produces bending around the radial column axis
and all four column pressures contribute to axial force:

```text
M[k] = pressure_gain × moment_arm
       × Σ_s P[s,k] (-sin(phi_s), cos(phi_s))

F_axial[k] = extension_gain × Σ_s P[s,k]
```

The mean Segment-1 charge applies the small calibrated stiffness multiplier:

```text
scale = 1 + stiffness_per_psi × charge_pressure
hinge stiffness = base_stiffness × scale
hinge damping   = base_damping × sqrt(scale)
```

## Commanded actuator dynamics

The three desired Segment 2--4 pressures pass through a measured transport
delay, column calibration, and first-order pneumatic response.

| Parameter | Value |
|---|---:|
| Transport delay | 0.5 s |
| Pneumatic time constant | 0.6 s |
| Segment 2 gain / positive-command bias | 1.11956 / 0.44967 psi |
| Segment 3 gain / positive-command bias | 1.09670 / 0.15251 psi |
| Segment 4 gain / positive-command bias | 1.16407 / 0.29976 psi |

The bias applies only above a zero command. After the delayed desired pressure
is mapped to its physical target, each pouch follows the target with the
calibrated first-order time constant. The internal command target and pressure
states are clamped to the 11 psi model bound. That extra headroom accommodates
the fitted regulator gain and bias; real-arm and coursework controller
commands use the hardware-safe 0--9 psi range.

## Sealed Segment-1 reservoir

The selected topology controls how a requested charge `x` initializes and
evolves the five isolated Segment-1 pouch pressures. For pouch `k`:

```text
P_charge[k] = charge_gain[k] × x + charge_bias[k]
```

The reservoir then changes through its calibrated slow leak and
deformation-to-pressure feedback. This is an empirical representation of the
recorded sensor response, not a thermodynamic pressure-volume law.

### Parallel plumbing (default)

| Quantity | Pouches 1 → 5 |
|---|---|
| Charge gain | `[1.0086, 1.0225, 1.0034, 1.0309, 1.0311]` |
| Charge bias [psi] | `[0.6143, 0.4036, 0.0401, 0.2495, -0.1262]` |
| Leak time [s] | `[3928, 3812, 2531, 3744, 2691]` |
| Curvature coupling [psi/rad] | `[-3.111, -3.333, -1.903, -1.261, 0.228]` |

Parallel pouches do not equalize (`equalization = 0`). Their calibrated sensor
noise standard deviation is 0.002 psi, deformation feedback is enabled, and
the response starts after a 1.5 s isolation delay.

### Coupled plumbing

| Quantity | Pouches 1 → 5 |
|---|---|
| Charge gain | `[1.03746, 1.04121, 1.04095, 1.02600, 1.02687]` |
| Charge bias [psi] | `[-0.23327, -0.24095, -0.23663, -0.23060, -0.23485]` |
| Leak time [s] | `[5780, 5780, 5780, 5780, 5780]` |
| Curvature coupling [psi/rad] | `[-3.332, -3.008, -2.424, -1.887, -1.546]` |

Coupled pouches equalize strongly (`equalization = 0.9`) and include a
charge-dependent fast relaxation component:

| Requested charge | Fast fraction | Fast time constant |
|---:|---:|---:|
| 1 psi | 0.11496 | 12.75 s |
| 2 psi | 0.08465 | 16.61 s |
| 3 psi | 0.05899 | 18.46 s |

As in the parallel model, sensor noise is 0.002 psi, deformation feedback is
enabled, and reservoir relaxation starts after 1.5 s.

## Integration and observations

The default controller rate is 100 Hz. Each control step contains ten 1 ms
MuJoCo physics steps using the `implicitfast` integrator and standard gravity.
Collision is disabled on arm geometry.

The observation dictionary includes tip position, quaternion, linear velocity,
all pouch pressure states, five `reservoir_pressures`, and three
`actuator_pressures`. The tip quantities come from MuJoCo sensors attached to
the marker-frame site.

## Validated operating range

The public calibration set covers:

- parallel and coupled Segment-1 plumbing;
- axial, circular, and triangular inputs;
- Segment-1 charges of 1, 2, and 3 psi;
- active command peaks of 5 and 10 psi; and
- 0.1 Hz excitation.

Across all 36 runs, complete desired-command replay achieved 3.34 mm mean
lateral tip RMSE, 1.17 mm axial RMSE, 0.44 psi active-pressure RMSE, and
0.09 psi Segment-1 pressure RMSE. These are centered held-cycle errors: they
evaluate the motion pattern and amplitude, not absolute mocap placement or a
trial-specific static fabric-sag offset.

Faster motion, charges outside 1--3 psi, command shapes outside the three
recorded waveforms, and active pressures outside the recorded range are
extrapolations and require new robot validation.

The recorded 10 psi peaks describe the historical calibration data. They do
not supersede the current 9 psi hardware/coursework command limit.
