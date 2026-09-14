"""
collect_sysid_data.py — Structured excitation for system identification.

Drives the soft arm through a sequence of pressure profiles designed to
maximally excite identifiable modes, and logs the commanded pressures and
resulting tip position to a CSV that sysid.py can consume.

By default this runs against the MuJoCo simulator (SoftArmSim). Pass
--hardware to run the exact same protocol against the real arm instead,
via hardware_arm.HardwareArm (regulator server + OptiTrack) — the CSV
format is identical either way, so sysid.py doesn't need to know which
one produced it.

Output CSV columns
------------------
    t           : time [s] (sim time, or hardware wall-clock tick count / 100 Hz)
    phase       : string label for the excitation phase
    p_col0..3   : mean pressure across all levels per column [psi]
    tip_x, tip_y, tip_z : tip position in world frame [m]
    arm_len     : measured arm length [m]

Usage
-----
    uv run collect_sysid_data.py                       # simulator, default output
    uv run collect_sysid_data.py --out sysid_data.csv  # custom path
    uv run collect_sysid_data.py --arm-length 0.296    # use real arm length
    uv run collect_sysid_data.py --no-plot             # skip plot

    # Real hardware (regulator server + OptiTrack publisher must be running):
    uv run collect_sysid_data.py --hardware --arm-length 0.296 --out real_sysid_data.csv
    uv run collect_sysid_data.py --hardware --reg-ip <pi-ip> --mocap-ip <laptop-ip>
    uv run collect_sysid_data.py --hardware --yes      # skip the confirmation prompt

Phases
------
  1. STATIC_SWEEP : ramp each column independently 0->3->6->9 psi (3 s each)
  2. STEP_RESPONSE: step each column from 0 -> 6 psi, hold 5 s
  3. AXIAL        : equal pressure on all columns 0->9 psi (pure extension)
  4. RELEASE      : drop all pressures, capture passive return dynamics

Total protocol duration is ~117 s of arm time regardless of --hardware;
against real hardware that's ~117 s of actual wall-clock time since each
step() call is paced to 100 Hz.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from soft_arm_sim import SoftArmSim
from arm_model import ArmConfig

HZ = 100        # simulation / control rate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_pressure(cfg: ArmConfig, col: int, level_p: float) -> np.ndarray:
    """Return (n_segments, n_pouches) pressure array with one column set."""
    P = np.zeros((cfg.n_segments, cfg.n_pouches))
    if 0 <= col < cfg.n_segments:
        P[col, :] = level_p
    return P


def run_phase(sim,
              pressure_fn,       # callable(t_phase) -> (n_seg, n_pou) array
              duration: float,
              phase_name: str,
              rows: list) -> None:
    """Step simulation (or hardware) for `duration` seconds, collecting rows.

    Only requires `sim.step(P) -> obs` with `obs["time"]` and
    `obs["tip_pos"]` — satisfied by both SoftArmSim and HardwareArm.
    """
    n_steps = int(duration * HZ)
    for i in range(n_steps):
        t_phase = i / HZ
        P = pressure_fn(t_phase)
        obs = sim.step(P)
        tip = obs["tip_pos"]
        arm_len = float(np.linalg.norm(tip - np.zeros(3)))  # distance from origin
        p_cols = P.mean(axis=1)                              # mean per column
        rows.append([
            f"{obs['time']:.4f}",
            phase_name,
            *[f"{p:.4f}" for p in p_cols],
            f"{tip[0]:.5f}", f"{tip[1]:.5f}", f"{tip[2]:.5f}",
            f"{arm_len:.5f}",
        ])


# ---------------------------------------------------------------------------
# excitation protocol
# ---------------------------------------------------------------------------

def collect(cfg: ArmConfig, sim, pre_inflation: float = 1.5) -> List[list]:
    """Run all excitation phases against `sim` and return rows (list of lists).

    `sim` must already be constructed (SoftArmSim or HardwareArm) — this
    function only drives it, so the same protocol runs identically against
    simulation or real hardware.
    """
    sim.set_pre_inflation(pre_inflation)
    sim.reset()

    rows: List[list] = []

    # 1. STATIC SWEEP -------------------------------------------------------
    # Ramp each column independently through 0->3->6->9 psi, holding 3 s each.
    # Both bending and cross-coupling stiffness are observable here.
    print("[sysid] Phase 1: static column sweep ...")
    for col in range(cfg.n_segments):
        for p_level in [0.0, 3.0, 6.0, 9.0]:
            def _static(t, _col=col, _p=p_level):
                return make_pressure(cfg, _col, _p)
            run_phase(sim, _static, duration=3.0,
                      phase_name=f"STATIC_col{col}_p{int(p_level)}",
                      rows=rows)
        # Release
        run_phase(sim, lambda t: np.zeros((cfg.n_segments, cfg.n_pouches)),
                  duration=2.0, phase_name=f"RELEASE_col{col}", rows=rows)

    # 2. STEP RESPONSE -------------------------------------------------------
    # Instantaneous step to 6 psi on each column; capture transient.
    # Damping and pneumatic time constant are identifiable here.
    print("[sysid] Phase 2: step response ...")
    for col in range(cfg.n_segments):
        sim.reset()
        sim.set_pre_inflation(pre_inflation)
        def _step(t, _col=col):
            return make_pressure(cfg, _col, 6.0)
        run_phase(sim, _step, duration=5.0,
                  phase_name=f"STEP_col{col}", rows=rows)
        run_phase(sim, lambda t: np.zeros((cfg.n_segments, cfg.n_pouches)),
                  duration=3.0, phase_name=f"RELEASE_step_col{col}", rows=rows)

    # 3. AXIAL EXTENSION -----------------------------------------------------
    # Inflate all columns equally -- pure axial extension, no bending.
    # Identifies extension_gain and axial_stiffness independently.
    print("[sysid] Phase 3: axial extension sweep ...")
    sim.reset()
    sim.set_pre_inflation(pre_inflation)
    for p_level in [0.0, 3.0, 6.0, 9.0, 6.0, 3.0, 0.0]:
        def _axial(t, _p=p_level):
            P = np.full((cfg.n_segments, cfg.n_pouches), _p)
            return P
        run_phase(sim, _axial, duration=3.0,
                  phase_name=f"AXIAL_p{int(p_level)}", rows=rows)

    # 4. PASSIVE RELEASE -----------------------------------------------------
    # Step to max pressure on one column, then vent all -> free oscillation.
    # Identifies damping ratio.
    print("[sysid] Phase 4: passive release dynamics ...")
    sim.reset()
    sim.set_pre_inflation(pre_inflation)
    run_phase(sim, lambda t: make_pressure(cfg, 0, 9.0),
              duration=3.0, phase_name="PRELOAD", rows=rows)
    run_phase(sim, lambda t: np.zeros((cfg.n_segments, cfg.n_pouches)),
              duration=5.0, phase_name="PASSIVE_RELEASE", rows=rows)

    print(f"[sysid] Collected {len(rows)} rows across all phases.")
    return rows


# ---------------------------------------------------------------------------
# protocol summary (used for the pre-flight confirmation on real hardware)
# ---------------------------------------------------------------------------

def estimate_duration_s(cfg: ArmConfig) -> float:
    static = cfg.n_segments * (4 * 3.0 + 2.0)
    step   = cfg.n_segments * (5.0 + 3.0)
    axial  = 7 * 3.0
    release = 3.0 + 5.0
    return static + step + axial + release


def max_commanded_psi() -> float:
    return 9.0   # highest level used anywhere in the protocol above


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_summary(csv_path: str) -> None:
    """Quick summary plot of the collected dataset."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")
        ax.grid(True, color="#21262d", lw=0.5)

    t = df["t"].values
    axes[0].plot(t, df["p_col0"], label="col 0 (E)", color="#FF9F43")
    axes[0].plot(t, df["p_col1"], label="col 1 (N)", color="#80ED99")
    axes[0].plot(t, df["p_col2"], label="col 2 (W)", color="#FF6B6B")
    axes[0].plot(t, df["p_col3"], label="col 3 (S)", color="#4ECDC4")
    axes[0].set_ylabel("Pressure [psi]", color="#e6edf3", fontsize=9)
    axes[0].legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3")

    axes[1].plot(t, df["tip_x"] * 100, label="X", color="#FF6B6B", lw=1.2)
    axes[1].plot(t, df["tip_y"] * 100, label="Y", color="#4ECDC4", lw=1.2)
    axes[1].set_ylabel("Tip X/Y [cm]", color="#e6edf3", fontsize=9)
    axes[1].legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3")

    axes[2].plot(t, df["tip_z"] * 100, label="Z", color="#FFE66D", lw=1.2)
    axes[2].plot(t, df["arm_len"] * 100, label="|tip|", color="#C77DFF",
                 lw=1.2, ls="--")
    axes[2].set_ylabel("Z / arm length [cm]", color="#e6edf3", fontsize=9)
    axes[2].set_xlabel("Time [s]", color="#e6edf3", fontsize=9)
    axes[2].legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3")

    fig.suptitle("SysID Excitation Dataset", color="#e6edf3", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    out_png = Path(csv_path).with_suffix(".png")
    fig.savefig(out_png, dpi=130)
    print(f"[sysid] Saved summary plot -> {out_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect structured excitation data for system identification."
    )
    parser.add_argument("--out",        default="sysid_data.csv",
                        help="Output CSV path (default: sysid_data.csv)")
    parser.add_argument("--arm-length", type=float, default=None,
                        help="Arm length [m] (default: ArmConfig default)")
    parser.add_argument("--pre-inflation", type=float, default=1.5,
                        help="Pre-inflation pressure [psi] (default: 1.5)")
    parser.add_argument("--no-plot",    action="store_true",
                        help="Skip summary plot")

    # -- hardware options ----------------------------------------------------
    parser.add_argument("--hardware",   action="store_true",
                        help="Run against the real arm instead of the simulator "
                             "(requires the regulator server + OptiTrack publisher "
                             "to be running; see hardware_arm.py)")
    parser.add_argument("--reg-ip",     default=None,
                        help="Regulator server IP (default: hardware_arm.DEFAULT_REG_IP)")
    parser.add_argument("--reg-port",   type=int, default=None,
                        help="Regulator server port (default: hardware_arm.DEFAULT_REG_PORT)")
    parser.add_argument("--mocap-ip",   default=None,
                        help="OptiTrack publisher IP (default: hardware_arm.DEFAULT_MOCAP_IP)")
    parser.add_argument("--mocap-port", type=int, default=None,
                        help="OptiTrack publisher port (default: hardware_arm.DEFAULT_MOCAP_PORT)")
    parser.add_argument("--p-max",      type=float, default=9.0,
                        help="Hardware safety clamp on commanded pressure [psi] (default: 9.0)")
    parser.add_argument("--yes",        action="store_true",
                        help="Skip the pre-flight confirmation prompt before actuating "
                             "real hardware (for scripted/unattended runs)")

    args = parser.parse_args()

    cfg = ArmConfig(length=args.arm_length) if args.arm_length else ArmConfig()
    print(f"[sysid] Arm length : {cfg.length*100:.1f} cm")
    print(f"[sysid] Output     : {args.out}")
    print(f"[sysid] Pre-infl.  : {args.pre_inflation} psi")

    sim = None
    try:
        if args.hardware:
            from hardware_arm import (HardwareArm, DEFAULT_REG_IP, DEFAULT_REG_PORT,
                                      DEFAULT_MOCAP_IP, DEFAULT_MOCAP_PORT)
            reg_ip     = args.reg_ip or DEFAULT_REG_IP
            reg_port   = args.reg_port or DEFAULT_REG_PORT
            mocap_ip   = args.mocap_ip or DEFAULT_MOCAP_IP
            mocap_port = args.mocap_port or DEFAULT_MOCAP_PORT

            duration_s = estimate_duration_s(cfg)
            print("\n[sysid] *** HARDWARE MODE ***")
            print(f"  regulator server : tcp://{reg_ip}:{reg_port}")
            print(f"  mocap publisher  : tcp://{mocap_ip}:{mocap_port}")
            print(f"  max pressure     : {min(args.p_max, cfg.p_max):.1f} psi "
                  f"(arm limit {cfg.p_max:.1f} psi)")
            print(f"  estimated time   : ~{duration_s:.0f} s "
                  f"({duration_s/60:.1f} min) of real actuation")
            print("  column->regulator mapping and rigid-body role assignment are "
                  "ASSUMED -- see the header of hardware_arm.py if this is your first run.")
            if not args.yes:
                ans = input("\nType 'yes' to actuate the real arm and start collecting: ")
                if ans.strip().lower() != "yes":
                    print("[sysid] Aborted, nothing sent to the regulators.")
                    return

            sim = HardwareArm(cfg, reg_ip=reg_ip, reg_port=reg_port,
                              mocap_ip=mocap_ip, mocap_port=mocap_port,
                              control_hz=HZ, p_max=args.p_max)
        else:
            sim = SoftArmSim(cfg=cfg)

        rows = collect(cfg, sim, pre_inflation=args.pre_inflation)
    finally:
        if sim is not None and hasattr(sim, "close"):
            sim.close()

    header = (["t", "phase"]
              + [f"p_col{s}" for s in range(cfg.n_segments)]
              + ["tip_x", "tip_y", "tip_z", "arm_len"])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[sysid] Wrote {len(rows)} rows -> {args.out}")

    if not args.no_plot:
        try:
            import pandas  # noqa: F401
            plot_summary(args.out)
        except ImportError:
            print("[sysid] pandas not available -- skipping plot")


if __name__ == "__main__":
    main()
