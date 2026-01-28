import argparse
import logging
import os
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure 'utilities' package can be imported when running as a script
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from utilities import config
from utilities.analysis import (
    get_experiment,
    load_csv_file,
    load_h5_experiment,
    quaternion_to_pitch,
    quaternion_to_roll,
    quaternion_to_yaw,
    update_column_constants,
)
from utilities.cleaner import DataCleaner
from utilities.leakage import LeakageTest, RealtimePlotter
from utilities.plotting import (
    create_2d_mocap_plot,
    create_3d_mocap_plot,
    create_plot_window,
)

logger = logging.getLogger("utilities.main")


def run_analysis(csv_path=None):
    # Check if user specified a CSV file
    if csv_path:
        # Load the specified CSV file
        # Handle relative paths
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(csv_path)

        data = load_csv_file(csv_path)
        if data is None:
            return
        base_title = os.path.basename(csv_path)
    else:
        # Use the auto-selection logic
        result = get_experiment()
        if not result:
            return

        # Load data based on file type
        if result[0] == "h5":
            _, h5_file, exp_name = result
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

    # Update column constants based on detected columns
    update_column_constants(data)

    # Create a new DataFrame for derived data (like yaw, pitch, roll)
    derived_data = pd.DataFrame()

    # Calculate orientation from quaternions if columns exist
    if all(col in data.columns for col in config.MOCAP_QUAT_COLS):
        try:
            quat_body = data[config.MOCAP_QUAT_COLS].astype(float).values
            qx, qy, qz, qw = quat_body.T
            derived_data[config.YAW_BODY_NAME] = quaternion_to_yaw(qx, qy, qz, qw)
            derived_data[config.PITCH_BODY_NAME] = quaternion_to_pitch(qx, qy, qz, qw)
            derived_data[config.ROLL_BODY_NAME] = quaternion_to_roll(qx, qy, qz, qw)
        except Exception as e:
            print(f"Warning: Could not calculate orientation from quaternions: {e}")

    # Combine original and derived data for plotting
    plot_data = pd.concat([data, derived_data], axis=1)

    # Prepare time vector and trim data if needed
    time = plot_data[config.TIME_COL].values
    if time[-1] >= config.START_TIME_OFFSET_SEC:
        print(f"Slicing data to start from {config.START_TIME_OFFSET_SEC} seconds.")
        start_index = np.argmax(time >= config.START_TIME_OFFSET_SEC)
        time = time[start_index:]
        plot_data = plot_data.iloc[start_index:].reset_index(drop=True)

    configs = config.get_plot_configs()

    create_plot_window(
        1,
        configs["sensor1"],
        plot_data,
        time,
        f"Sensor Data (Seg 3,4): {base_title}",
    )

    create_plot_window(
        2,
        configs["sensor2"],
        plot_data,
        time,
        f"Sensor Data (Seg 1,2): {base_title}",
    )

    create_plot_window(
        3,
        configs["mocap"],
        plot_data,
        time,
        f"Mocap Data: {base_title}",
    )

    create_2d_mocap_plot(5, plot_data, "Robot Trajectory (2D X-Z)")
    create_3d_mocap_plot(4, plot_data, "Robot Trajectory (3D)")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Soft Robotic Arm Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: clean
    parser_clean = subparsers.add_parser("clean", help="Clean/Rename CSV data columns")
    parser_clean.add_argument(
        "--folder", type=str, help="Folder containing CSVs (optional)"
    )

    # Command: leak-test
    parser_leak = subparsers.add_parser("leak-test", help="Run pouch leakage test")
    parser_leak.add_argument(
        "--pumps",
        type=int,
        nargs="+",
        default=[3, 6, 7, 8],
        help="Pump Arduino IDs (default: 3 6 7 8)",
    )
    parser_leak.add_argument(
        "--sensor", type=int, default=7, help="Sensor Arduino ID (default: 7)"
    )
    parser_leak.add_argument(
        "--target", type=float, default=3.0, help="Target Pressure PSI (default: 3.0)"
    )

    # Command: analyze
    parser_ana = subparsers.add_parser("analyze", help="Plot experimental data")
    parser_ana.add_argument(
        "--csv", type=str, help="Specific CSV file to analyze (optional)"
    )

    args = parser.parse_args()

    if args.command == "clean":
        DataCleaner.run_cleaning(args.folder)

    elif args.command == "leak-test":
        print("\nPOUCH LEAKAGE TEST TOOL")
        print("=" * 60)
        test = LeakageTest(args.pumps, args.sensor, args.target)

        def sig_h(sig, frame):
            logger.info("\nStopping...")
            test.stop()
            plt.close("all")

        signal.signal(signal.SIGINT, sig_h)

        if not test.connect():
            logger.error("Connection failed.")
            return

        try:
            test.start()
            plotter = RealtimePlotter(test)
            plotter.show()
        except KeyboardInterrupt:
            pass
        finally:
            test.cleanup()

    elif args.command == "analyze":
        run_analysis(args.csv)

    else:
        # Default behavior: Print help
        parser.print_help()


if __name__ == "__main__":
    main()
