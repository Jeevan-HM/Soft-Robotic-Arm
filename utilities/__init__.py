"""
Utilities Package for Soft Robotic Arm
======================================

This package bundles data cleaning, leakage testing, and data analysis tools.
You can run these tools via the command line using: `uv run utilities <command>` or `python -m utilities <command>`.

Available Commands:
-------------------

1. Data Cleaning
   -------------
   Cleans and renames columns in CSV files within a specified folder.

   Command:
   uv run utilities clean --folder <path/to/folder>

   Example:
   uv run utilities clean --folder experiments/raw_data

2. Leakage Testing
   ---------------
   Runs a pressure hold test on specified pumps and sensors to detect leaks.

   Command:
   uv run utilities leak-test --pumps <ids> --sensor <id> --target <psi>

   Example:
   uv run utilities leak-test --pumps 3 6 7 8 --sensor 7 --target 3.0

3. Data Analysis
   -------------
   Loads experimental data (HDF5 or CSV), computes orientation, and plots results.

   Command:
   uv run utilities analyze --csv <path/to/file.csv>     # Analyze specific CSV
   uv run utilities analyze                              # Auto-select latest HDF5 experiment

   Example:
   uv run utilities analyze --csv files_for_submission/two_psi_isolated.csv

"""

from . import config
from .analysis import (
    get_experiment,
    load_csv_file,
    load_h5_experiment,
    quaternion_to_pitch,
    quaternion_to_roll,
    quaternion_to_yaw,
    update_column_constants,
)
from .cleaner import DataCleaner
from .leakage import LeakageTest, RealtimePlotter
from .plotting import (
    create_2d_mocap_plot,
    create_3d_mocap_plot,
    create_plot_window,
)

__all__ = [
    "DataCleaner",
    "LeakageTest",
    "RealtimePlotter",
    "get_experiment",
    "load_csv_file",
    "load_h5_experiment",
    "update_column_constants",
    "create_plot_window",
    "create_2d_mocap_plot",
    "create_3d_mocap_plot",
    "config",
]
