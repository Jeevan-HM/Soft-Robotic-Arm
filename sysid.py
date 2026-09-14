"""
sysid.py — System identification: fit ArmConfig physics params to measured data.

Loads a CSV produced by collect_sysid_data.py (or by digital_twin.py --record
paired with the same pressure commands), replays the commanded pressures
through SoftArmSim with a candidate parameter vector, and minimises the RMSE
between simulated and measured tip trajectories using scipy.optimize.

Identified parameters
---------------------
    base_stiffness   [N*m/rad]   bending joint stiffness per level
    base_damping     [N*m*s/rad] bending joint damping per level
    axial_stiffness  [N/m]       axial slide stiffness per level
    axial_damping    [N*s/m]     axial slide damping per level
    pressure_gain    [N/psi]     bending force per column per level
    extension_gain   [N/psi]     axial force per column per level
    mass             [kg]        total moving arm mass

All other ArmConfig fields (geometry, length, etc.) are held fixed.

Usage
-----
    # Quick test with synthetic data (generates its own dataset)
    uv run sysid.py --synthetic

    # Identify from real collected data
    uv run sysid.py --data sysid_data.csv

    # Specify real arm length
    uv run sysid.py --data sysid_data.csv --arm-length 0.296

    # Output file
    uv run sysid.py --data sysid_data.csv --out identified_params.json

    # Resume from a previous result
    uv run sysid.py --data sysid_data.csv --init identified_params.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

from arm_model import ArmConfig
from soft_arm_sim import SoftArmSim


# ---------------------------------------------------------------------------
# Parameter vector <-> ArmConfig mapping
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "base_stiffness",
    "base_damping",
    "axial_stiffness",
    "axial_damping",
    "pressure_gain",
    "extension_gain",
    "mass",
]

# (lower_bound, upper_bound) for each parameter — physically sensible ranges
PARAM_BOUNDS = [
    (0.05,   5.0),    # base_stiffness  N*m/rad
    (0.005,  1.0),    # base_damping    N*m*s/rad
    (50.0,   5000.0), # axial_stiffness N/m
    (1.0,    200.0),  # axial_damping   N*s/m
    (0.05,   5.0),    # pressure_gain   N/psi
    (0.001,  0.5),    # extension_gain  N/psi
    (0.05,   2.0),    # mass            kg
]


def cfg_to_vec(cfg: ArmConfig) -> np.ndarray:
    return np.array([getattr(cfg, n) for n in PARAM_NAMES], dtype=float)


def vec_to_cfg(vec: np.ndarray, base_cfg: ArmConfig) -> ArmConfig:
    """Return a new ArmConfig with identified params replaced from vec."""
    import dataclasses
    d = dataclasses.asdict(base_cfg)
    for name, val in zip(PARAM_NAMES, vec):
        d[name] = float(val)
    return ArmConfig(**d)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str):
    """Load CSV produced by collect_sysid_data.py.

    Returns
    -------
    pressures : ndarray (T, n_segments)   mean pressure per column [psi]
    tips      : ndarray (T, 3)            measured tip position [m]
    times     : ndarray (T,)              timestamps [s]
    """
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    T = len(rows)
    # Detect number of columns from header
    n_seg = sum(1 for k in rows[0] if k.startswith("p_col"))

    pressures = np.zeros((T, n_seg))
    tips      = np.zeros((T, 3))
    times     = np.zeros(T)

    for i, row in enumerate(rows):
        times[i] = float(row["t"])
        for s in range(n_seg):
            pressures[i, s] = float(row[f"p_col{s}"])
        tips[i, 0] = float(row["tip_x"])
        tips[i, 1] = float(row["tip_y"])
        tips[i, 2] = float(row["tip_z"])

    print(f"[sysid] Loaded {T} rows from {csv_path}  "
          f"(duration {times[-1]:.1f} s, {n_seg} columns)")
    return pressures, tips, times


# ---------------------------------------------------------------------------
# Simulation rollout
# ---------------------------------------------------------------------------

def rollout(cfg: ArmConfig,
            pressures: np.ndarray,
            n_pouches: int,
            arm_length: float,
            pre_inflation: float = 1.5) -> np.ndarray:
    """Replay pressure commands through SoftArmSim, return tip positions.

    Parameters
    ----------
    pressures : (T, n_segments)  mean pressure per column [psi]
    Returns   : (T, 3)           simulated tip positions  [m]
    """
    sim = SoftArmSim(cfg=cfg)
    sim.set_pre_inflation(pre_inflation)
    sim.reset()

    T, n_seg = pressures.shape
    tips_sim = np.zeros((T, 3))

    for t in range(T):
        # Broadcast column mean pressure to all pouches
        P = np.tile(pressures[t, :, None], (1, n_pouches))  # (n_seg, n_pou)
        obs = sim.step(P)
        tips_sim[t] = obs["tip_pos"]

    return tips_sim


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

_eval_count    = [0]
_best_rmse     = [np.inf]
_stall_count   = [0]
STALL_PATIENCE = 200   # stop if no improvement for this many evaluations
STALL_TOL      = 1e-5  # minimum improvement to reset stall counter [m]


def make_objective(base_cfg: ArmConfig,
                   pressures: np.ndarray,
                   tips_real: np.ndarray,
                   pre_inflation: float = 1.5,
                   verbose: bool = True):
    """Return a scalar objective f(vec) -> RMSE [m]."""

    def objective(vec: np.ndarray) -> float:
        # Enforce bounds via penalty (for optimisers that don't support bounds)
        for v, (lo, hi) in zip(vec, PARAM_BOUNDS):
            if v < lo or v > hi:
                return 1e6

        cfg = vec_to_cfg(vec, base_cfg)
        try:
            tips_sim = rollout(cfg, pressures, cfg.n_pouches,
                               cfg.length, pre_inflation)
        except Exception:
            return 1e6

        rmse = float(np.sqrt(np.mean((tips_sim - tips_real) ** 2)))

        _eval_count[0] += 1
        if rmse < _best_rmse[0] - STALL_TOL:
            _best_rmse[0]  = rmse
            _stall_count[0] = 0
            if verbose:
                print(f"  [{_eval_count[0]:4d}] RMSE={rmse*1000:.2f} mm  "
                      + "  ".join(f"{n}={v:.4f}"
                                  for n, v in zip(PARAM_NAMES, vec)))
        else:
            _stall_count[0] += 1
            if verbose and _eval_count[0] % 50 == 0:
                print(f"  [{_eval_count[0]:4d}] RMSE={rmse*1000:.2f} mm  "
                      f"(best {_best_rmse[0]*1000:.2f} mm, "
                      f"stall {_stall_count[0]}/{STALL_PATIENCE})")
            if _stall_count[0] >= STALL_PATIENCE:
                print(f"  [early-stop] No improvement for {STALL_PATIENCE} evals. "
                      f"Best RMSE={_best_rmse[0]*1000:.2f} mm")
                raise StopIteration("early_stop")

        return rmse

    return objective


# ---------------------------------------------------------------------------
# Synthetic dataset (for testing without real hardware)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(cfg_true: ArmConfig,
                            n_seg: int = 4,
                            n_pou: int = 5) -> tuple:
    """Generate a synthetic dataset with added Gaussian noise."""
    print("[sysid] Generating synthetic dataset from cfg_true ...")

    # Simple pressure protocol: step each column
    steps_per_col = 500   # timesteps
    T = n_seg * steps_per_col + steps_per_col  # + release phase
    pressures = np.zeros((T, n_seg))

    for s in range(n_seg):
        start = s * steps_per_col
        pressures[start:start + steps_per_col, s] = 6.0

    sim_true = SoftArmSim(cfg=cfg_true)
    sim_true.set_pre_inflation(1.5)
    sim_true.reset()

    tips = np.zeros((T, 3))
    for t in range(T):
        P = np.tile(pressures[t, :, None], (1, n_pou))
        obs = sim_true.step(P)
        tips[t] = obs["tip_pos"]

    # Add realistic noise (1 mm std, same as OptiTrack)
    tips += np.random.normal(0, 0.001, tips.shape)
    times = np.arange(T) * cfg_true.timestep
    print(f"[sysid] Synthetic dataset: {T} rows, duration {times[-1]:.2f} s")
    return pressures, tips, times


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def identify(base_cfg: ArmConfig,
             pressures: np.ndarray,
             tips_real: np.ndarray,
             init_vec: Optional[np.ndarray] = None,
             method: str = "nelder-mead",
             pre_inflation: float = 1.5) -> tuple:
    """Run the optimiser and return (identified_cfg, result)."""

    x0 = init_vec if init_vec is not None else cfg_to_vec(base_cfg)

    # Clamp initial point inside bounds
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        x0[i] = np.clip(x0[i], lo, hi)

    # Reset global tracking state
    _eval_count[0]  = 0
    _best_rmse[0]   = np.inf
    _stall_count[0] = 0
    _best_vec       = [x0.copy()]   # track best param vector for early-stop recovery

    _orig_obj = make_objective(base_cfg, pressures, tips_real, pre_inflation)

    def obj_tracked(vec):
        rmse = _orig_obj(vec)
        if rmse < _best_rmse[0] + 1e-9:
            _best_vec[0] = vec.copy()
        return rmse

    t0 = time.time()
    print(f"\n[sysid] Starting optimisation ({method}) ...")
    print(f"[sysid] Initial RMSE: {_orig_obj(x0)*1000:.2f} mm")
    print(f"[sysid] Parameters to identify: {PARAM_NAMES}")

    result = None
    try:
        if method == "differential_evolution":
            result = differential_evolution(
                obj_tracked, PARAM_BOUNDS,
                maxiter=200, popsize=10, tol=1e-5,
                seed=42, disp=False, workers=1,
            )
        else:
            result = minimize(
                obj_tracked, x0,
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-7,
                         "disp": False, "adaptive": True},
            )
    except StopIteration:
        # Early stopping fired — synthesise a result-like object from best seen
        class _FakeResult:
            x   = _best_vec[0]
            fun = _best_rmse[0]
        result = _FakeResult()

    elapsed = time.time() - t0
    print(f"\n[sysid] Optimisation finished in {elapsed:.1f} s  "
          f"({_eval_count[0]} evaluations)")
    print(f"[sysid] Final RMSE: {result.fun*1000:.2f} mm")

    identified_cfg = vec_to_cfg(result.x, base_cfg)
    return identified_cfg, result



# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(times: np.ndarray,
                    tips_real: np.ndarray,
                    tips_before: np.ndarray,
                    tips_after: np.ndarray,
                    out_path: str = "sysid_comparison.png") -> None:
    """Plot real vs sim tip trajectory before and after identification."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             facecolor="#0d1117")
    labels = ["X [m]", "Y [m]", "Z [m]"]
    for ax, i, label in zip(axes, range(3), labels):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")
        ax.grid(True, color="#21262d", lw=0.5)

        ax.plot(times, tips_real[:, i],   color="#e6edf3",  lw=1.8,
                label="Real / measured", zorder=3)
        ax.plot(times, tips_before[:, i], color="#FF6B6B",  lw=1.2, ls="--",
                label="Sim (default params)", alpha=0.8)
        ax.plot(times, tips_after[:, i],  color="#80ED99",  lw=1.5,
                label="Sim (identified)", alpha=0.9)
        ax.set_ylabel(label, color="#e6edf3", fontsize=9)
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3")

    axes[-1].set_xlabel("Time [s]", color="#e6edf3", fontsize=9)

    rmse_before = np.sqrt(np.mean((tips_before - tips_real)**2)) * 1000
    rmse_after  = np.sqrt(np.mean((tips_after  - tips_real)**2)) * 1000

    fig.suptitle(
        f"System Identification — Tip Trajectory Fit\n"
        f"RMSE  before: {rmse_before:.1f} mm   after: {rmse_after:.1f} mm   "
        f"({(1 - rmse_after/rmse_before)*100:.0f}% reduction)",
        color="#e6edf3", fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[sysid] Saved comparison plot -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Identify ArmConfig physics parameters from tip trajectory data."
    )
    parser.add_argument("--data",         default=None,
                        help="CSV from collect_sysid_data.py")
    parser.add_argument("--synthetic",    action="store_true",
                        help="Generate and use synthetic data (no real arm needed)")
    parser.add_argument("--arm-length",   type=float, default=None,
                        help="Arm length [m] (overrides CSV-derived value)")
    parser.add_argument("--pre-inflation",type=float, default=1.5,
                        help="Pre-inflation pressure [psi] (default: 1.5)")
    parser.add_argument("--out",          default="identified_params.json",
                        help="Output JSON path (default: identified_params.json)")
    parser.add_argument("--init",         default=None,
                        help="JSON from a previous run to warm-start from")
    parser.add_argument("--method",       default="nelder-mead",
                        choices=["nelder-mead", "differential_evolution"],
                        help="Optimiser (default: nelder-mead)")
    parser.add_argument("--no-plot",      action="store_true",
                        help="Skip comparison plot")
    args = parser.parse_args()

    if args.data is None and not args.synthetic:
        parser.error("Provide --data <csv> or use --synthetic")

    # Build base config
    base_cfg = ArmConfig(length=args.arm_length) if args.arm_length else ArmConfig()

    # Load or generate dataset
    if args.synthetic:
        # Perturb the true cfg so identification has something to recover
        import dataclasses
        d = dataclasses.asdict(base_cfg)
        d["base_stiffness"]  *= 1.6
        d["base_damping"]    *= 0.7
        d["pressure_gain"]   *= 1.3
        d["mass"]            *= 1.2
        cfg_true = ArmConfig(**d)
        pressures, tips_real, times = make_synthetic_dataset(
            cfg_true, base_cfg.n_segments, base_cfg.n_pouches)
    else:
        pressures, tips_real, times = load_dataset(args.data)

    # Warm-start from previous result if given
    init_vec = None
    if args.init and Path(args.init).exists():
        init_cfg = ArmConfig.from_json(args.init)
        init_vec = cfg_to_vec(init_cfg)
        print(f"[sysid] Warm-starting from {args.init}")

    # Baseline: rollout with default parameters
    print("[sysid] Computing baseline (default params) rollout ...")
    tips_before = rollout(base_cfg, pressures, base_cfg.n_pouches,
                          base_cfg.length, args.pre_inflation)
    rmse_before = np.sqrt(np.mean((tips_before - tips_real)**2)) * 1000
    print(f"[sysid] Baseline RMSE: {rmse_before:.2f} mm")

    # Run identification
    identified_cfg, result = identify(
        base_cfg, pressures, tips_real,
        init_vec=init_vec,
        method=args.method,
        pre_inflation=args.pre_inflation,
    )

    # Save identified params
    identified_cfg.to_json(args.out)
    print(f"[sysid] Identified params saved -> {args.out}")

    # Print comparison table
    print("\n[sysid] Parameter comparison:")
    print(f"  {'Parameter':<22}  {'Default':>12}  {'Identified':>12}  {'Change':>8}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*8}")
    for name in PARAM_NAMES:
        default_val    = getattr(base_cfg, name)
        identified_val = getattr(identified_cfg, name)
        change = (identified_val / default_val - 1) * 100
        print(f"  {name:<22}  {default_val:>12.5f}  {identified_val:>12.5f}  "
              f"{change:>+7.1f}%")

    # Comparison plot
    if not args.no_plot:
        tips_after = rollout(identified_cfg, pressures, identified_cfg.n_pouches,
                             identified_cfg.length, args.pre_inflation)
        rmse_after = np.sqrt(np.mean((tips_after - tips_real)**2)) * 1000
        print(f"\n[sysid] RMSE before: {rmse_before:.2f} mm")
        print(f"[sysid] RMSE after:  {rmse_after:.2f} mm  "
              f"({(1 - rmse_after/rmse_before)*100:.0f}% reduction)")
        plot_comparison(times, tips_real, tips_before, tips_after)

    print(f"\n[sysid] Done. Load with:  ArmConfig.from_json('{args.out}')")


if __name__ == "__main__":
    main()
