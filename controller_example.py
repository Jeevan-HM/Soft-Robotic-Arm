"""
Closed-loop example: how to drop a controller into the 100 Hz loop.

Uses a deliberately simple PID on tip position (error decomposed onto
the five pouch-column azimuths, same command broadcast to all four
segments) as a stand-in — the point is the interface. Your
Reservoir-Koopman MPC or PRC inverse controller plugs in the same way:

    while True:
        obs = sim.step(P_cmd)                  # <- plant (was: hardware/UDP)
        y   = obs["tip_pos"]                   # <- mocap analog
        s   = obs["pouch_pressures"]           # <- 4x5 pouch sensors (PRC input)
        P_cmd = my_controller(y_ref, y, s)     # <- your controller, (4,5) kPa

Run:  MUJOCO_GL=osmesa python3 controller_example.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from soft_arm_sim import SoftArmSim

HZ = 100


def main():
    sim = SoftArmSim()
    sim.set_pre_inflation(6.0)
    obs = sim.reset()
    tip0 = obs["tip_pos"].copy()          # rest tip position

    phis = sim.cfg.azimuths()             # (n_pouches,)
    n_pouches = len(phis)
    # column k pulls the tip toward direction d_k in the xy-plane
    col_dirs = np.stack([-np.cos(phis), -np.sin(phis)], axis=1)  # (n_pouches, 2)

    Kp, Ki, Kd = 110.0, 130.0, 14.0         # kPa per m, m*s of tip error
    p_bias = 3.0                          # on top of pre-inflation
    integ = np.zeros(n_pouches)
    cmd_cols = np.full(n_pouches, p_bias)

    # reference: 4 cm-radius circle at 0.10 Hz, after a 1 s hold
    T = 14.0
    log_ref, log_tip = [], []
    for k in range(int(T * HZ)):
        t = k / HZ
        if t < 1.0:
            ref_xy = np.zeros(2)
        else:
            w = 2 * np.pi * 0.10 * (t - 1.0)
            ref_xy = 0.04 * np.array([np.cos(w) - 1.0, np.sin(w)])
        obs = sim.step(cmd_cols)          # (5,) broadcast to all 4 segments
        err_xy = ref_xy - (obs["tip_pos"][:2] - tip0[:2])
        e_col = col_dirs @ err_xy         # project error onto columns
        v_col = col_dirs @ obs["tip_vel"][:2]
        integ = np.clip(integ + e_col / HZ, -0.35, 0.35)   # anti-windup
        cmd_cols = np.clip(p_bias + Kp * e_col + Ki * integ - Kd * v_col,
                           0, sim.cfg.p_max)
        log_ref.append(ref_xy.copy())
        log_tip.append(obs["tip_pos"][:2] - tip0[:2])

    ref, tip = np.array(log_ref), np.array(log_tip)
    rmse = np.sqrt(((ref - tip) ** 2).sum(axis=1)).mean()
    print(f"closed-loop tracking RMSE: {rmse * 1000:.1f} mm")

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(ref[:, 0] * 100, ref[:, 1] * 100, "k--", lw=1.2, label="reference")
    ax.plot(tip[:, 0] * 100, tip[:, 1] * 100, lw=1.5, label="tip")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    ax.set_title(f"PID tip tracking (RMSE {rmse*1000:.1f} mm)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("closed_loop_tracking.png", dpi=130)
    print("saved closed_loop_tracking.png")


if __name__ == "__main__":
    main()
