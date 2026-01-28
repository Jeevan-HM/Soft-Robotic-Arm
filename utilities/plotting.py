import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from . import config

# Optional: for interactive tooltips
try:
    import mplcursors

    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False


def create_plot_window(
    fig_num, plot_configs, data, time, window_title, x_tick_interval=None
):
    """Helper to create a figure window and populate it with 2D plots."""
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(
        num_plots, 1, figsize=(16, 6 * num_plots), num=fig_num, squeeze=False
    )
    axes = axes.flatten()
    fig.suptitle(window_title, fontweight="bold", y=0.995)

    for ax, plot_cfg in zip(axes, plot_configs):
        # Filter for columns that actually exist in the DataFrame
        columns_to_plot = [col for col in plot_cfg["columns"] if col in data]
        if not columns_to_plot:
            print(
                f"Warning: None of {plot_cfg['columns']} found for plot '{plot_cfg['title']}'. Skipping."
            )
            continue

        for i, col_name in enumerate(columns_to_plot):
            ax.plot(
                time,
                data[col_name].values,
                label=plot_cfg["labels"][i % len(plot_cfg["labels"])],
                color=plot_cfg["colors"][i % len(plot_cfg["colors"])],
                linewidth=2.5,
            )

        ax.set_xlabel(plot_cfg.get("xlabel", "Time (s)"))
        ax.set_ylabel(plot_cfg.get("ylabel", "Value"))
        ax.set_title(plot_cfg["title"], pad=12)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", which="major")

        # Set major ticks interval on the x-axis if specified
        if x_tick_interval:
            ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))

    fig.tight_layout(rect=[0, 0, 1, 0.99], h_pad=2.0)

    if HAS_MPLCURSORS:
        for ax in axes:
            mplcursors.cursor(ax.lines, hover=True)


def create_2d_mocap_plot(fig_num, data, window_title):
    """Creates a 2D plot for the mocap trajectory (X-Z Plane)."""

    # Define the X and Z columns to use
    MOCAP_POS_XZ_COLS = [config.MOCAP_POS_COLS[0], config.MOCAP_POS_COLS[2]]

    # Check if all required columns exist
    if not all(col in data.columns for col in MOCAP_POS_XZ_COLS):
        print(
            f"Warning: Missing one or more X-Z mocap position columns ({MOCAP_POS_XZ_COLS}). Skipping 2D X-Z plot."
        )
        return

    # Create a standard 2D figure and axes
    fig = plt.figure(num=fig_num, figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Extract just X and Z data
    x, z = data[MOCAP_POS_XZ_COLS].values.T

    ax.plot(x, z, label="Trajectory", color="orange")
    ax.scatter(x[0], z[0], c="g", s=100, marker="o", label="Start")
    ax.scatter(x[-1], z[-1], c="r", s=100, marker="s", label="End")

    ax.set_xlabel(f"{config.MOCAP_POS_COLS[0]} Position (X)", fontweight="bold")
    ax.set_ylabel(f"{config.MOCAP_POS_COLS[2]} Position (Z)", fontweight="bold")
    ax.set_title(window_title, fontweight="bold", pad=20)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")


def create_3d_mocap_plot(fig_num, data, window_title):
    """Creates a 3D plot for the mocap trajectory."""
    # Check if all required columns exist
    if not all(col in data.columns for col in config.MOCAP_POS_COLS):
        print("Warning: Missing one or more mocap position columns. Skipping 3D plot.")
        return

    fig = plt.figure(num=fig_num, figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = data[config.MOCAP_POS_COLS].values.T

    # Filter out NaNs for plotting and limits
    valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    if not np.any(valid_mask):
        print("Warning: No valid numeric data for 3D plot.")
        return

    x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]

    ax.plot(x_valid, y_valid, z_valid, label="Trajectory", color="orange")

    # Plot start/end only if we have valid points
    if len(x_valid) > 0:
        ax.scatter(
            x_valid[0],
            y_valid[0],
            z_valid[0],
            c="g",
            s=100,
            marker="o",
            label="Start",
            depthshade=False,
        )
        ax.scatter(
            x_valid[-1],
            y_valid[-1],
            z_valid[-1],
            c="r",
            s=100,
            marker="s",
            label="End",
            depthshade=False,
        )

    ax.set_xlabel("X Position ", fontweight="bold")
    ax.set_ylabel("Y Position ", fontweight="bold")
    ax.set_zlabel("Z Position ", fontweight="bold")
    ax.set_title(window_title, fontweight="bold", pad=20)
    ax.legend()
    ax.grid(True)

    # Use valid data for limits
    max_range = (
        np.ptp(np.vstack([x_valid, y_valid, z_valid]), axis=1).max() / 2.0 or 1.0
    )
    mid_x, mid_y, mid_z = np.mean(x_valid), np.mean(y_valid), np.mean(z_valid)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
