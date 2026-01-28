import os
import re
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from . import config


def list_h5_experiments():
    """List all HDF5 files and their experiments."""
    if not os.path.exists(config.EXPERIMENTS_BASE_DIR):
        print(f"Error: Directory {config.EXPERIMENTS_BASE_DIR} not found.")
        return None

    h5_files = [f for f in os.listdir(config.EXPERIMENTS_BASE_DIR) if f.endswith(".h5")]

    if not h5_files:
        print("No HDF5 files found in experiments directory!")
        return None

    print("\n" + "=" * 80)
    print("Available HDF5 Files:")
    print("=" * 80)

    all_experiments = []

    for h5_file in sorted(h5_files):
        filepath = os.path.join(config.EXPERIMENTS_BASE_DIR, h5_file)
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
    if not os.path.exists(config.EXPERIMENTS_BASE_DIR):
        print(f"Error: Directory {config.EXPERIMENTS_BASE_DIR} not found.")
        return None, None

    h5_files = [f for f in os.listdir(config.EXPERIMENTS_BASE_DIR) if f.endswith(".h5")]

    if not h5_files:
        print("No HDF5 files found!")
        return None, None

    all_experiments = []
    for h5_file in h5_files:
        filepath = os.path.join(config.EXPERIMENTS_BASE_DIR, h5_file)
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
    Returns path to load data from - either CSV or HDF5.
    """
    if not os.path.exists(config.EXPERIMENTS_BASE_DIR):
        print(f"Error: Directory {config.EXPERIMENTS_BASE_DIR} not found.")
        return None

    # Check if we should use HDF5
    h5_files = [f for f in os.listdir(config.EXPERIMENTS_BASE_DIR) if f.endswith(".h5")]

    if h5_files:
        # Use HDF5
        h5_file, exp_name = select_experiment()
        if h5_file and exp_name:
            return ("h5", h5_file, exp_name)

    # Fallback to CSV (original code logic)
    try:
        folder_names = [
            name
            for name in os.listdir(config.EXPERIMENTS_BASE_DIR)
            if os.path.isdir(os.path.join(config.EXPERIMENTS_BASE_DIR, name))
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

        latest_folder_path = os.path.join(config.EXPERIMENTS_BASE_DIR, latest_folder)
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
    except Exception as e:
        print(f"Error finding experiment file: {e}")
        return None


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


def update_column_constants(df):
    """
    Check which column set is present in the DataFrame and update global constants in-place.
    """
    cols = set(df.columns)

    # Check for a representative column from the LONG set
    use_long = any(
        c in cols for c in config.COL_SET_LONG["MEASURED_PRESSURE_SEGMENT1_COLS"]
    )

    target_set = config.COL_SET_LONG if use_long else config.COL_SET_SHORT
    print(
        f"\n[INFO] Detected column set: {'LONG (Alternative)' if use_long else 'SHORT (Standard)'}"
    )

    # Update globals in-place so references in config dicts remain valid
    config.DESIRED_PRESSURE_COLS[:] = target_set["DESIRED_PRESSURE_COLS"]
    config.MEASURED_PRESSURE_SEGMENT1_COLS[:] = target_set[
        "MEASURED_PRESSURE_SEGMENT1_COLS"
    ]

    # Handle Segment 2 specifically
    seg2_cols = target_set["MEASURED_PRESSURE_SEGMENT2_COLS"]
    if use_long:
        # Check if any of the expected pouch columns exist
        has_pouches = any(c in cols for c in seg2_cols)
        # If not, but we have the single column, use that instead
        if not has_pouches and "Measured_pressure_Segment_2" in cols:
            seg2_cols = ["Measured_pressure_Segment_2"]

    config.MEASURED_PRESSURE_SEGMENT2_COLS[:] = seg2_cols
    config.MEASURED_PRESSURE_SEGMENT3_COLS[:] = target_set[
        "MEASURED_PRESSURE_SEGMENT3_COLS"
    ]
    config.MEASURED_PRESSURE_SEGMENT4_COLS[:] = target_set[
        "MEASURED_PRESSURE_SEGMENT4_COLS"
    ]
    config.MOCAP_POS_COLS[:] = target_set["MOCAP_POS_COLS"]
    config.MOCAP_QUAT_COLS[:] = target_set["MOCAP_QUAT_COLS"]


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
