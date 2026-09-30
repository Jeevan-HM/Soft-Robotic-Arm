# Calibrated MuJoCo Arm — Construction Guide

This guide explains how the calibrated robot model is assembled and how a
three-pressure command reaches the MuJoCo joints. See
[`arm_parameters.md`](../arm_parameters.md) for fitted values and
[`calibration.json`](../calibration.json) for the authoritative
machine-readable calibration.

## Physical and pneumatic layout

The simulated arm has four radial pouch columns and five vertical pouch
levels. Segment 1 is not an actuator: it is charged once, isolated, and used as
a five-state pressure reservoir. Segments 2--4 receive the three live pressure
commands.

```text
Top view

             Segment 2 / column 1 (North)
                         |
Segment 3 / column 2 ----+---- Segment 1 / column 0 (sealed)
                         |
             Segment 4 / column 3 (South)
```

The runtime keeps a complete 4 × 5 pressure matrix because all columns still
contribute to mechanics. The public command path is three values:

```python
from simulator import SoftArmSim

sim = SoftArmSim()                    # parallel, Segment 1 charged to 2 psi
obs = sim.step([p2, p3, p4])          # commands for Segments 2, 3, and 4
```

Use `SoftArmSim(topology="coupled", reservoir_pressure_psi=3.0)` to reproduce
a coupled-plumbing condition.

## Five-level mechanical chain

The arm is represented by five nested MuJoCo bodies, one per pouch level. The
calibrated 0.296 m rest length gives a 0.0592 m level height.

```text
mount (fixed)
  └── level0: ext0, bx0, by0
       └── level1: ext1, bx1, by1
            └── level2: ext2, bx2, by2
                 └── level3: ext3, bx3, by3
                      └── level4: ext4, bx4, by4
                           └── tip_disc
                                └── tip_frame and sensor site
```

Each level contributes three degrees of freedom:

| Joint | Type and axis | Range | Calibrated stiffness | Calibrated damping |
|---|---|---|---:|---:|
| `ext0` … `ext4` | Axial slide, downward Z | −5 to +30 mm | 934.2301 N/m | 61.5096 N·s/m |
| `bx0` … `bx4` | Hinge around X | Unbounded | 0.416845 N·m/rad | 0.362923 N·m·s/rad |
| `by0` … `by4` | Hinge around Y | Unbounded | 0.416845 N·m/rad | 0.362923 N·m·s/rad |

Together, each X/Y hinge pair acts as a universal bending joint. All joints
have zero spring reference, so the passive structure returns toward its
straight rest configuration.

## Level geometry

Each level contains:

- one dark connector ring at the top;
- four colored column capsules, 28 mm from the center axis and with an 18 mm
  radius; and
- one thin central structural core.

The column colors and directions are fixed:

| Column | Segment | Direction | Color |
|---:|---:|---|---|
| 0 | 1 | East | Orange |
| 1 | 2 | North | Green |
| 2 | 3 | West | Red |
| 3 | 4 | South | Blue |

The capsules visualize the pneumatic structure, but MuJoCo does not deform
their surfaces. All arm geometry has contact disabled. Pressure effects are
applied to joint generalized forces by `SoftArmSim`.

## Fixed mount and marker frame

The mount is fixed to the world. Its height is derived from arm length so the
hanging arm clears the floor. A plate, four posts, and a connector disc provide
visual context but do not add degrees of freedom.

The bottom `tip_frame` represents the OptiTrack marker cross. Its calibrated
half-span is 70 mm and its nominal mass is 0.08 kg. A site at the frame center
feeds three MuJoCo sensors:

| Sensor | Output |
|---|---|
| `tip_pos` | World position `(x, y, z)` in meters |
| `tip_quat` | World orientation quaternion `(w, x, y, z)` |
| `tip_vel` | World linear velocity in m/s |

## From command to motion

One call to `step([p2, p3, p4])` advances this pipeline:

```text
three desired pressures
  → 0.5 s calibrated transport delay
  → per-actuator pressure gain and positive-command bias
  → 0.6 s first-order pouch response
  → complete 4 × 5 pressure state, including sealed Segment 1
  → bending moments and axial forces at every level
  → ten 1 ms MuJoCo integration steps
  → pose, velocity, actuator-pressure, and reservoir-pressure observations
```

For column azimuth `phi_s` and pouch level `k`, the runtime uses:

```text
M[k] = pressure_gain × moment_arm
       × Σ_s P[s,k] (-sin(phi_s), cos(phi_s))

F_axial[k] = extension_gain × Σ_s P[s,k]
```

Those forces are written to `data.qfrc_applied` before every physics substep.
This makes the model dynamic rather than a static pressure-to-pose map: mass,
gravity, damping, elastic restoring force, actuator lag, and pressure history
all affect the trajectory.

## Sealed Segment-1 behavior

At construction, Segment 1's requested charge is mapped through five
pouch-specific charge gains and biases. The column is then removed from the
live command path. Its pressures evolve through the calibrated leak and
deformation feedback terms.

The two recorded plumbing arrangements use different reservoir models:

- `parallel` is the default. Pouches have individual charge offsets and leak
  rates and do not equalize with one another.
- `coupled` strongly equalizes the five pouch states and adds the measured
  charge-dependent fast relaxation immediately after isolation.

Both topologies expose the five local states as `reservoir_pressures`. This
response is an empirical fit to the sensors; it is not a thermodynamic cavity
or fluid-flow model.

## Numerical model

The MJCF uses:

- the `implicitfast` integrator;
- a 1 ms physics timestep;
- standard gravity `(0, 0, -9.81)` m/s²;
- a 100 Hz default control rate; and
- collision-disabled arm geometry.

The model therefore solves a spring-damper rigid-body approximation of the
arm. It reproduces the measured slow motion patterns over the calibrated
conditions, but it should not be interpreted as a finite-element material
model.

## Calibration boundary

The parameters were fitted and checked on 36 robot runs covering parallel and
coupled plumbing, axial/circular/triangular waveforms, 1--3 psi Segment-1
charge, 5/10 psi command peaks, and 0.1 Hz excitation. Behavior outside those
conditions is extrapolation and should be validated against new robot data.
