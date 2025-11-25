"""
HDF5 to CSV Converter

This script converts experiment data from HDF5 files to CSV format.

Usage:
    # List experiments in latest HDF5 file (auto-detected):
    python h5_to_csv.py --list

    # Convert latest experiment from latest HDF5 file:
    python h5_to_csv.py --exp exp_041_axial_Nov12_15h17m

    # Convert with specific HDF5 file:
    python h5_to_csv.py --h5 2025_November.h5 --exp exp_041_axial_Nov12_15h17m

    # Convert with custom output name:
    python h5_to_csv.py --exp exp_041_axial_Nov12_15h17m --output my_data.csv
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import h5py

# Default experiments directory
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


# Default experiments directory
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_latest_h5_file(directory=None):
    """Find the most recent HDF5 file in the experiments directory."""
    if directory is None:
        directory = EXPERIMENTS_DIR

    h5_files = [f for f in os.listdir(directory) if f.endswith(".h5")]

    if not h5_files:
        print(f"Error: No HDF5 files found in '{directory}'")
        return None

    # Sort by modification time, most recent first
    h5_files_with_time = []
    for f in h5_files:
        filepath = os.path.join(directory, f)
        mtime = os.path.getmtime(filepath)
        h5_files_with_time.append((filepath, mtime, f))

    h5_files_with_time.sort(key=lambda x: x[1], reverse=True)
    latest_file = h5_files_with_time[0][0]
    latest_name = h5_files_with_time[0][2]

    print(f"Auto-selected latest HDF5 file: {latest_name}")
    return latest_file


def list_experiments(h5_file):
    """List all experiments in an HDF5 file."""
    print(f"\nExperiments in '{os.path.basename(h5_file)}':")
    print("=" * 80)

    with h5py.File(h5_file, "r") as f:
        experiments = sorted([k for k in f.keys() if k.startswith("exp_")])

        if not experiments:
            print("No experiments found!")
            return

        for exp_name in experiments:
            exp = f[exp_name]
            timestamp = exp.attrs.get("timestamp", "N/A")
            wave = exp.attrs.get("wave_function", "Unknown")
            desc = exp.attrs.get("description", "No description")

            if "data" in exp:
                samples = len(exp["data"])
            else:
                samples = "N/A"

            print(f"\n  {exp_name}")
            print(f"    Time: {timestamp}")
            print(f"    Wave: {wave}")
            print(f"    Samples: {samples}")
            print(f"    Description: {desc}")

    print("\n" + "=" * 80)


def convert_h5_to_csv(h5_file, experiment_name, csv_file):
    """Convert an HDF5 experiment to CSV format."""
    # Construct paths
    group_path = f"/{experiment_name}"
    dataset_path = f"/{experiment_name}/data"

    print(f"Opening '{h5_file}'...")
    try:
        with h5py.File(h5_file, "r") as f:
            # 1. Access the Group and read the 'columns' attribute
            if group_path not in f:
                print(f"Error: The group '{group_path}' was not found.")
                return False

            group = f[group_path]
            print("Reading 'columns' attribute from group...")

            # The attribute is already a NumPy array, not a string
            column_headers = group.attrs["columns"]

            print(f"Found {len(column_headers)} column headers.")

            # 2. Access the Dataset
            if dataset_path not in f:
                print(f"Error: The dataset '{dataset_path}' was not found.")
                return False

            data = f[dataset_path]
            print(f"Reading dataset '{dataset_path}' with shape {data.shape}...")

            # 3. Verify dimensions
            if len(column_headers) != data.shape[1]:
                print(
                    f"Warning: Mismatch! Found {len(column_headers)} headers but data has {data.shape[1]} columns."
                )

            # 4. Open the CSV and write data
            with open(csv_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write the header row first
                writer.writerow(column_headers)

                # Write the data rows
                writer.writerows(data[()])

            print(f"Successfully converted data with headers to '{csv_file}'")
            return True

    except FileNotFoundError:
        print(f"Error: The file '{h5_file}' was not found.")
        return False
    except KeyError as e:
        print(f"Error: Could not find an attribute or path. {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 experiment data to CSV format."
    )

    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Path to the HDF5 file (default: auto-detect latest file in experiments directory)",
    )

    parser.add_argument(
        "--exp",
        "--experiment",
        type=str,
        help="Name of the experiment to convert (e.g., exp_041_axial_Nov12_15h17m)",
        dest="experiment",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV filename (default: <experiment_name>.csv)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all experiments in the HDF5 file",
    )

    args = parser.parse_args()

    # Handle the H5 file path
    if args.h5:
        # User specified a file
        h5_file = args.h5
        if not os.path.isabs(h5_file):
            h5_file = os.path.abspath(h5_file)

        # Check if file exists
        if not os.path.exists(h5_file):
            print(f"Error: File '{h5_file}' not found.")
            sys.exit(1)
    else:
        # Auto-detect latest HDF5 file
        h5_file = find_latest_h5_file()
        if h5_file is None:
            sys.exit(1)

    # If --list flag is set, list experiments and exit
    if args.list:
        try:
            list_experiments(h5_file)
        except Exception as e:
            print(f"Error listing experiments: {e}")
            sys.exit(1)
        return

    # Otherwise, we need an experiment name
    if not args.experiment:
        print("Error: --exp argument is required (unless using --list)")
        print("Use --list to see available experiments.")
        sys.exit(1)

    experiment_name = args.experiment

    # Determine output filename
    if args.output:
        csv_file = args.output
    else:
        csv_file = f"{experiment_name}.csv"

    # Convert to absolute path if relative
    if not os.path.isabs(csv_file):
        csv_file = os.path.abspath(csv_file)

    # Perform the conversion
    success = convert_h5_to_csv(h5_file, experiment_name, csv_file)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
