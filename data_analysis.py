#!/usr/bin/env python3
"""
Data Analysis Script for Robotic Arm Experiments

This script loads and plots experimental data from HDF5 files or CSV files.

Usage:
    # Auto-select latest HDF5 experiment (default):
    python data_analysis.py

    # Load a specific CSV file:
    python data_analysis.py --csv path/to/file.csv

    # Examples:
    python data_analysis.py --csv exp_040_axial_Nov12_14h48m_data_with_headers.csv
    python data_analysis.py --csv experiments/csv/July-16/Test_1.csv
"""

import argparse
import os
import re
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Optional: for interactive tooltips
try:
    import mplcursors

    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False

# =================================================================================
# ---- UNIVERSAL FONT SIZE CONFIGURATION ----
# =================================================================================
# -- EDIT THESE VALUES TO CONTROL ALL PLOT FONTS --
BASE_FONT_SIZE = 18  # This is your base size
LABEL_PADDING = 24  # Padding for axis labels
plt.rcParams.update(
    {
        # --- Base and Tick Fonts ---
        "font.size": BASE_FONT_SIZE,  # Default font size for non-specified items
        "xtick.labelsize": BASE_FONT_SIZE
        - 10,  # Font size for X-axis tick labels (e.g., 20, 40, 60)
        "ytick.labelsize": BASE_FONT_SIZE
        - 10,  # Font size for Y-axis tick labels (e.g., 2.5, 5.0, 7.5)
        # --- Label Fonts ---
        "axes.labelsize": LABEL_PADDING - 10,  # Controls X and Y axis labels
        "axes.labelpad": LABEL_PADDING,  # <-- NEW: Applies the padding
        # --- Title Fonts ---
        "axes.titlesize": BASE_FONT_SIZE
        + 4,  # Controls the subplot titles (e.g., "Measured Pressures (Segment 1)")
        "figure.titlesize": BASE_FONT_SIZE
        + 6,  # Controls the main figure title (e.g., "Sensor & Control Data...")
        # --- Legend Font ---
        "legend.fontsize": BASE_FONT_SIZE,  # Controls the legend font size
    }
)
# =================================================================================


# =================================================================================
# ---- DATA_MAPPING & SETTINGS (EDITABLE) ----
# =================================================================================
# NOTE: Column mapping now uses header names (strings) instead of fixed indices.

# -- General Settings --
# EXPERIMENTS_BASE_DIR = "/home/g1/Developer/Thesis/experiments"
EXPERIMENTS_BASE_DIR = "/home/jeevan/Developer/Soft-Robotic-Arm/experiments"
START_TIME_OFFSET_SEC = 0  # Time in seconds to skip at the beginning

# -- Column Names --
TIME_COL = "time"


DESIRED_PRESSURE_COLS = ["pd_3", "pd_6", "pd_7", "pd_8"]
MEASURED_PRESSURE_SEGMENT1_COLS = ["pm_3_1", "pm_3_2", "pm_3_3", "pm_3_4", "pm_7_1"]
MEASURED_PRESSURE_SEGMENT2_COLS = ["pm_7_2", "pm_7_3", "pm_7_4", "pm_8_1", "pm_8_2"]
MEASURED_PRESSURE_SEGMENT3_COLS = ["pm_8_3"]
MEASURED_PRESSURE_SEGMENT4_COLS = ["pm_8_4"]
MOCAP_POS_COLS = ["mocap_3_x", "mocap_3_y", "mocap_3_z"]
MOCAP_QUAT_COLS = ["mocap_3_qx", "mocap_3_qy", "mocap_3_qz", "mocap_3_qw"]


# DESIRED_PRESSURE_COLS = [
#     "Desired_pressure_segment_1",
#     "Desired_pressure_segment_2",
#     "Desired_pressure_segment_3",
#     "Desired_pressure_segment_4",
# ]
# MEASURED_PRESSURE_SEGMENT1_COLS = [
#     "Measured_pressure_Segment_1_pouch_1",
#     "Measured_pressure_Segment_1_pouch_2",
#     "Measured_pressure_Segment_1_pouch_3",
#     "Measured_pressure_Segment_1_pouch_4",
#     "Measured_pressure_Segment_1_pouch_5",
# ]
# MEASURED_PRESSURE_SEGMENT2_COLS = [
#     "Measured_pressure_Segment_2_pouch_1",
#     "Measured_pressure_Segment_2_pouch_2",
#     "Measured_pressure_Segment_2_pouch_3",
#     "Measured_pressure_Segment_2_pouch_4",
#     "Measured_pressure_Segment_2_pouch_5",
# ]
# MEASURED_PRESSURE_SEGMENT3_COLS = ["Measured_pressure_Segment_3"]
# MEASURED_PRESSURE_SEGMENT4_COLS = ["Measured_pressure_Segment_4"]
# MOCAP_POS_COLS = ["mocap_rigid_body_x", "mocap_rigid_body_y", "mocap_rigid_body_z"]
# MOCAP_QUAT_COLS = [
#     "mocap_rigid_body_qx",
#     "mocap_rigid_body_qy",
#     "mocap_rigid_body_qz",
#     "mocap_rigid_body_qw",
# ]

# -- Derived Column Names (for internal use) --
YAW_BODY_NAME = "yaw_body"
PITCH_BODY_NAME = "pitch_body"
ROLL_BODY_NAME = "roll_body"
# =================================================================================


# ---- PLOT CONFIGURATION ----
MOCAP_PLOT_CONFIG = [
    {
        "title": "Mocap Position (Body 3 - Trajectory)",
        "xlabel": "Time (s)",
        "ylabel": "Position ",
        "columns": MOCAP_POS_COLS,
        "labels": ["X Position", "Y Position", "Z Position"],
        "colors": ["tab:blue", "tab:orange", "tab:green"],
    },
    {
        "title": "Mocap Yaw Orientation (Body 3 - Trajectory)",
        "xlabel": "Time (s)",
        "ylabel": "Yaw (rad)",
        "columns": [YAW_BODY_NAME],
        "labels": ["Yaw"],
        "colors": ["tab:red"],
    },
    {
        "title": "Mocap Pitch Orientation (Body 3 - Trajectory)",
        "xlabel": "Time (s)",
        "ylabel": "Pitch (rad)",
        "columns": [PITCH_BODY_NAME],
        "labels": ["Pitch"],
        "colors": ["tab:purple"],
    },
]

SENSOR_CONTROL_CONFIG_1 = [
    {
        "title": "Desired Pressures",
        "xlabel": "Time (s)",
        "ylabel": "Desired Pressure (PSI)",
        "columns": DESIRED_PRESSURE_COLS,
        "labels": ["pd_segment_1", "pd_segment_2", "pd_segment_3", "pd_segment_4"],
        "colors": ["tab:red", "tab:orange", "tab:blue", "tab:green"],
    },
    {
        "title": "Measured Pressures (Segment 3)",
        "xlabel": "Time (s)",
        "ylabel": "Sensor Pressure (PSI)",
        "columns": MEASURED_PRESSURE_SEGMENT3_COLS,
        "labels": ["pm_segment_3"],
        "colors": ["tab:blue", "tab:cyan"],
    },
    {
        "title": "Measured Pressures (Segment 4)",
        "xlabel": "Time (s)",
        "ylabel": "Sensor Pressure (PSI)",
        "columns": MEASURED_PRESSURE_SEGMENT4_COLS,
        "labels": ["pm_segment_4"],
        "colors": ["tab:purple"],
    },
]

SENSOR_CONTROL_CONFIG_2 = [
    {
        "title": "Measured Pressures (Segment 1)",
        "xlabel": "Time (s)",
        "ylabel": "Sensor Pressure (PSI)",
        "columns": MEASURED_PRESSURE_SEGMENT1_COLS,
        "labels": [
            "Segment_1_pouch_1",
            "Segment_1_pouch_2",
            "Segment_1_pouch_3",
            "Segment_1_pouch_4",
            "Segment_1_pouch_5",
        ],
        "colors": ["tab:red", "tab:pink", "crimson", "tab:brown", "salmon"],
    },
    {
        "title": "Measured Pressures (Segment 2)",
        "xlabel": "Time (s)",
        "ylabel": "Sensor Pressure (PSI)",
        "columns": MEASURED_PRESSURE_SEGMENT2_COLS,
        "labels": [
            "Segment_2_pouch_1",
            "Segment_2_pouch_2",
            "Segment_2_pouch_3",
            "Segment_2_pouch_4",
            "Segment_2_pouch_5",
        ],
        "colors": ["tab:orange", "tab:olive", "gold", "darkorange", "peru"],
    },
]


# =================================================================================
# ---- HDF5 LOADING FUNCTIONS ----
# =================================================================================
def list_h5_experiments():
    """List all HDF5 files and their experiments."""
    h5_files = [f for f in os.listdir(EXPERIMENTS_BASE_DIR) if f.endswith(".h5")]

    if not h5_files:
        print("No HDF5 files found in experiments directory!")
        return None

    print("\n" + "=" * 80)
    print("Available HDF5 Files:")
    print("=" * 80)

    all_experiments = []

    for h5_file in sorted(h5_files):
        filepath = os.path.join(EXPERIMENTS_BASE_DIR, h5_file)
        print(f"\n📁 {h5_file}")

        with h5py.File(filepath, "r") as f:
            experiments = sorted([k for k in f.keys() if k.startswith("exp_")])

            for exp_name in experiments:
                exp = f[exp_name]
                timestamp = exp.attrs.get("timestamp", "N/A")
                wave = exp.attrs.get("wave_function", "Unknown")
                desc = exp.attrs.get("description", "No description")
                samples = len(exp["data"])

                all_experiments.append((filepath, exp_name))

                print(f"  {len(all_experiments)}. {exp_name}")
                print(f"     Time: {timestamp}")
                print(f"     Wave: {wave}")
                print(f"     Samples: {samples}")
                print(f"     Description: {desc}")

    print("\n" + "=" * 80)
    return all_experiments


def select_experiment():
    """Auto-selects latest experiment by timestamp."""
    h5_files = [f for f in os.listdir(EXPERIMENTS_BASE_DIR) if f.endswith(".h5")]

    if not h5_files:
        print("No HDF5 files found!")
        return None, None

    all_experiments = []
    for h5_file in h5_files:
        filepath = os.path.join(EXPERIMENTS_BASE_DIR, h5_file)
        with h5py.File(filepath, "r") as f:
            for exp_name in f.keys():
                if exp_name.startswith("exp_"):
                    timestamp_str = f[exp_name].attrs.get("timestamp", "N/A")
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    except (ValueError, TypeError):
                        timestamp = datetime.min
                    all_experiments.append((filepath, exp_name, timestamp))

    if not all_experiments:
        print("No experiments found!")
        return None, None

    all_experiments.sort(key=lambda x: x[2], reverse=True)
    latest = all_experiments[0]
    print(f"Loading: {latest[1]} ({latest[2].strftime('%Y-%m-%d %H:%M:%S')})")

    return latest[0], latest[1]


def load_h5_experiment(h5_file, exp_name):
    """Load experiment data from HDF5 and return as DataFrame."""
    with h5py.File(h5_file, "r") as f:
        exp = f[exp_name]

        # Load data and column names
        data_array = exp["data"][:]
        columns = list(exp.attrs["columns"])

        # Get metadata
        metadata = {
            "timestamp": exp.attrs.get("timestamp", "N/A"),
            "wave_function": exp.attrs.get("wave_function", "Unknown"),
            "description": exp.attrs.get("description", "No description"),
            "arduino_ids": list(exp.attrs.get("arduino_ids", [])),
            "target_pressures": list(exp.attrs.get("target_pressures", [])),
        }

        # Create DataFrame
        df = pd.DataFrame(data_array, columns=columns)

        print(f"\nMetadata:")
        print(f"  Timestamp: {metadata['timestamp']}")
        print(f"  Wave Function: {metadata['wave_function']}")
        print(f"  Description: {metadata['description']}")
        print(f"  Arduino IDs: {metadata['arduino_ids']}")
        print(f"  Target Pressures: {metadata['target_pressures']}")
        print(f"  Data Shape: {df.shape}")

        return df


def get_experiment():
    """
    Modified to work with HDF5 files.
    Returns path to load data from - either CSV or HDF5.
    """
    # Check if we should use HDF5
    h5_files = [f for f in os.listdir(EXPERIMENTS_BASE_DIR) if f.endswith(".h5")]

    if h5_files:
        # Use HDF5
        h5_file, exp_name = select_experiment()
        # h5_file, exp_name = (
        #     "/home/g1/Developer/Thesis/experiments/2025_November.h5",
        #     "exp_057_axial_Nov12_17h41m",
        # )  # --- FOR TESTING ONLY ---
        if h5_file and exp_name:
            return ("h5", h5_file, exp_name)

    # Fallback to CSV (original code)
    try:
        if not os.path.exists(EXPERIMENTS_BASE_DIR):
            raise FileNotFoundError
        folder_names = [
            name
            for name in os.listdir(EXPERIMENTS_BASE_DIR)
            if os.path.isdir(os.path.join(EXPERIMENTS_BASE_DIR, name))
            and not name.startswith(".")
        ]
        date_folder_pattern = re.compile(r"^[A-Za-z]+-\d{1,2}$")
        date_folders = [
            name for name in folder_names if date_folder_pattern.match(name)
        ]
        if not date_folders:
            raise RuntimeError("No valid date-named folders found.")

        def folder_to_date(folder):
            try:
                return datetime.strptime(folder, "%B-%d").replace(
                    year=datetime.now().year
                )
            except ValueError:
                return None

        dated_folders = [(folder, folder_to_date(folder)) for folder in date_folders]
        dated_folders = [item for item in dated_folders if item[1] is not None]
        if not dated_folders:
            raise RuntimeError("No valid date-named folders found.")

        current_date = datetime.now()
        dated_folders.sort(
            key=lambda x: (
                x[1].year if x[1].month <= current_date.month else x[1].year - 1,
                x[1].month,
                x[1].day,
            ),
            reverse=True,
        )
        latest_folder = dated_folders[0][0]

        latest_folder_path = os.path.join(EXPERIMENTS_BASE_DIR, latest_folder)
        test_files = [
            f
            for f in os.listdir(latest_folder_path)
            if re.match(r"(Test_\d+_\d+|Experiment_\d+)\.csv$", f)
        ]
        if not test_files:
            raise RuntimeError(f"No valid test files found in '{latest_folder}'.")

        test_nums = []
        for fname in test_files:
            m = re.match(r"Test_\d+_(\d+)\.csv$|Experiment_(\d+)\.csv$", fname)
            if m:
                num = m.group(1) or m.group(2)
                test_nums.append((int(num), fname))
        if not test_nums:
            raise RuntimeError(
                f"No validly named test files found in '{latest_folder}'."
            )

        latest_test_file = max(test_nums, key=lambda x: x[0])[1]
        filename = os.path.join(latest_folder_path, latest_test_file)
        print(f"Latest experiment file: {filename}")
        return ("csv", filename)
    except FileNotFoundError:
        print(f"Error: Directory not found. Check 'EXPERIMENTS_BASE_DIR'.")
        exit()
    except RuntimeError as e:
        print(f"Error finding experiment file: {e}")
        exit()


# =================================================================================


def quaternion_to_roll(qx, qy, qz, qw):
    """Calculates roll (X-axis rotation)."""
    return np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))


def quaternion_to_pitch(qx, qy, qz, qw):
    """Calculates pitch (Y-axis rotation)."""
    arg = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    return np.arcsin(arg)


def quaternion_to_yaw(qx, qy, qz, qw):
    """Calculates yaw (Z-axis rotation)."""
    return -np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))


def create_plot_window(
    fig_num, plot_configs, data, time, window_title, x_tick_interval=None
):
    """Helper to create a figure window and populate it with 2D plots."""
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(
        num_plots, 1, figsize=(16, 6 * num_plots), num=fig_num, squeeze=False
    )
    axes = axes.flatten()
    # Note: Removed fontsize. It will use 'figure.titlesize' from rcParams
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

        # Note: Removed fontsize. Will use 'axes.labelsize' from rcParams
        ax.set_xlabel(plot_cfg.get("xlabel", "Time (s)"))
        ax.set_ylabel(plot_cfg.get("ylabel", "Value"))

        # Note: Removed fontsize. Will use 'axes.titlesize' from rcParams
        ax.set_title(plot_cfg["title"], pad=12)

        # Note: Removed fontsize. Will use 'legend.fontsize' from rcParams
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Note: Removed labelsize. Will use 'xtick.labelsize' and 'ytick.labelsize'
        ax.tick_params(axis="both", which="major")

        # Set major ticks interval on the x-axis if specified
        if x_tick_interval:
            ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))

    # *** FIX: Reduced h_pad from 4.0 to 2.0 to prevent squishing ***
    fig.tight_layout(rect=[0, 0, 1, 0.99], h_pad=2.0)

    if HAS_MPLCURSORS:
        for ax in axes:
            mplcursors.cursor(ax.lines, hover=True)


def create_2d_mocap_plot(fig_num, data, window_title):
    """Creates a 2D plot for the mocap trajectory (X-Z Plane)."""

    # Define the X and Z columns to use
    MOCAP_POS_XZ_COLS = [MOCAP_POS_COLS[0], MOCAP_POS_COLS[2]]

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
    # Note: MOCAP_POS_COLS[0] is X, MOCAP_POS_COLS[2] is Z
    x, z = data[MOCAP_POS_XZ_COLS].values.T

    # *** THIS IS THE MODIFIED LINE ***
    ax.plot(x, z, label="Trajectory", color="orange")
    # *** END OF MODIFICATION ***

    ax.scatter(x[0], z[0], c="g", s=100, marker="o", label="Start")
    ax.scatter(x[-1], z[-1], c="r", s=100, marker="s", label="End")

    # Use 'axes.labelsize' from rcParams
    ax.set_xlabel(f"{MOCAP_POS_COLS[0]} Position (X)", fontweight="bold")
    ax.set_ylabel(f"{MOCAP_POS_COLS[2]} Position (Z)", fontweight="bold")

    # Use 'axes.titlesize' from rcParams
    ax.set_title(window_title, fontweight="bold", pad=20)

    # Use 'legend.fontsize' from rcParams
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Set equal aspect ratio so X and Z are scaled the same
    ax.set_aspect("equal", adjustable="box")


def create_3d_mocap_plot(fig_num, data, window_title):
    """Creates a 3D plot for the mocap trajectory."""
    # Check if all required columns exist
    if not all(col in data.columns for col in MOCAP_POS_COLS):
        print("Warning: Missing one or more mocap position columns. Skipping 3D plot.")
        return

    fig = plt.figure(num=fig_num, figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = data[MOCAP_POS_COLS].values.T

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
            x_valid[0], y_valid[0], z_valid[0], c="g", s=100, marker="o", label="Start", depthshade=False
        )
        ax.scatter(
            x_valid[-1], y_valid[-1], z_valid[-1], c="r", s=100, marker="s", label="End", depthshade=False
        )

    # Note: Removed fontsize. Will use 'axes.labelsize' from rcParams
    ax.set_xlabel("X Position ", fontweight="bold")
    ax.set_ylabel("Y Position ", fontweight="bold")
    ax.set_zlabel("Z Position ", fontweight="bold")

    # Note: Removed fontsize. Will use 'axes.titlesize' from rcParams
    ax.set_title(window_title, fontweight="bold", pad=20)

    # Note: Legend will use default 'legend.fontsize' from rcParams
    ax.legend()
    ax.grid(True)

    # Use valid data for limits
    max_range = np.ptp(np.vstack([x_valid, y_valid, z_valid]), axis=1).max() / 2.0 or 1.0
    mid_x, mid_y, mid_z = np.mean(x_valid), np.mean(y_valid), np.mean(z_valid)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def load_csv_file(csv_path):
    """Load a specific CSV file and return as DataFrame."""
    print(f"\nLoading CSV file: {csv_path}")
    try:
        data = pd.read_csv(csv_path)
        print(f"Data Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def main():
    """Main function to run the data analysis and plotting."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot experimental data from HDF5 or CSV files."
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a specific CSV file to load and plot",
        default=None,
    )
    args = parser.parse_args()

    # Check if user specified a CSV file
    if args.csv:
        # Load the specified CSV file
        csv_path = args.csv
        # Handle relative paths
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(csv_path)

        data = load_csv_file(csv_path)
        if data is None:
            return
        base_title = os.path.basename(csv_path)
    else:
        # Use the original auto-selection logic
        result = get_experiment()
        if not result:
            return

        # Load data based on file type
        if result[0] == "h5":
            _, h5_file, exp_name = result
            # exp_name = "exp_098_axial_Nov24_14h24m"
            print(f"\nLoading HDF5: {exp_name} from {os.path.basename(h5_file)}")
            data = load_h5_experiment(h5_file, exp_name)
            base_title = exp_name
        else:
            _, filename = result
            print(f"\nAnalyzing CSV:\n{filename}\n")
            try:
                data = pd.read_csv(filename)
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found.")
                return
            base_title = os.path.basename(filename)

    if data.empty:
        print("Error: Data file is empty.")
        return

    # Create a new DataFrame for derived data (like yaw, pitch, roll)
    # This avoids modifying the original data DataFrame
    derived_data = pd.DataFrame()

    # Calculate orientation from quaternions if columns exist
    if all(col in data.columns for col in MOCAP_QUAT_COLS):
        try:
            quat_body = data[MOCAP_QUAT_COLS].astype(float).values
            qx, qy, qz, qw = quat_body.T
            derived_data[YAW_BODY_NAME] = quaternion_to_yaw(qx, qy, qz, qw)
            derived_data[PITCH_BODY_NAME] = quaternion_to_pitch(qx, qy, qz, qw)
            derived_data[ROLL_BODY_NAME] = quaternion_to_roll(qx, qy, qz, qw)
        except Exception as e:
            print(f"Warning: Could not calculate orientation from quaternions: {e}")

    # Combine original and derived data for plotting
    plot_data = pd.concat([data, derived_data], axis=1)

    # Prepare time vector and trim data if needed
    time = plot_data[TIME_COL].values
    if time[-1] >= START_TIME_OFFSET_SEC:
        print(f"Slicing data to start from {START_TIME_OFFSET_SEC} seconds.")
        start_index = np.argmax(time >= START_TIME_OFFSET_SEC)
        time = time[start_index:]
        plot_data = plot_data.iloc[start_index:].reset_index(drop=True)

    create_plot_window(
        1,
        SENSOR_CONTROL_CONFIG_1,
        plot_data,
        time,
        f"Sensor & Control Data (Desired, Segments 3 & 4): {base_title}",
        # x_tick_interval=3,  # Set x-axis ticks to 1-second intervals
    )

    create_plot_window(
        2,
        SENSOR_CONTROL_CONFIG_2,
        plot_data,
        time,
        f"Sensor & Control Data (Segments 1 & 2): {base_title}",
    )

    create_plot_window(
        3,
        MOCAP_PLOT_CONFIG,
        plot_data,
        time,
        f"Mocap Data (Time Series - Body 3): {base_title}",
    )

    # create_3d_mocap_plot(4, plot_data, f"Mocap 3D Trajectory (Body 3): {base_title}")
    create_3d_mocap_plot(4, plot_data, f"Robot Trajectory")
    # create_2d_mocap_plot(5, plot_data, f"Robot Trajectory")

    plt.show()


if __name__ == "__main__":
    main()
