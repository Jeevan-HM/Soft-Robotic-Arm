# Soft Robotic Arm Utilities

This repository contains a unified utility script (`utilities.py`) for managing, cleaning, testing, and analyzing data from soft robotic arm experiments.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage Overview

The `utilities.py` script provides three main subcommands:

1. **`clean`**: Renames columns and removes unwanted data from raw CSVs.
2. **`leak-test`**: Runs a pressure hold test to detect leaks in the soft robot pouches.
3. **`analyze`**: Plots and visualizes experimental data (sensors, pressures, mocap).

Run the help command to see all options:

```bash
python utilities.py --help
```

---

## 1. Clean Data

Renames columns to standard naming conventions and drops unused columns.

### Usage

```bash
# Interactive mode (prompts for folder)
python utilities.py clean

# Specify a folder directly
python utilities.py clean --folder path/to/raw_csvs
```

**What it does:**
- Scans the folder for `.csv` files.
- Creates a `cleaned_data` subdirectory.
- Maps short column names (e.g., `pd_3`, `mocap_1_x`) to descriptive names (`Desired_pressure_segment_1`, `Rigid_body_1_x`).
- Removes temporary/unused columns (e.g., `pm_6_1`).

---

## 2. Leakage Test

Performs a pressure hold test on specified pumps and sensors.

### Usage

```bash
# Run with default settings (Pumps: 3,6,7,8 | Sensor: 7 | Target: 3.0 PSI)
python utilities.py leak-test

# Custom configuration
python utilities.py leak-test --pumps 3 6 --sensor 3 --target 2.5
```

**Arguments:**
- `--pumps`: List of Arduino IDs controlling the pumps (e.g., `3 6 7 8`).
- `--sensor`: Arduino ID of the sensor to monitor.
- `--target`: Target pressure in PSI to hold.

**Key Features:**
- Visualizes pressure in real-time.
- Automatically detects leakage based on pressure drop after stabilization.

---

## 3. Data Analysis

Loads and visualizes experimental data from HDF5 or CSV files.

### Usage

#### Analyze Latest Experiment (Auto-Select)
Automatically loads the most recent HDF5 experiment found in the `experiments/` directory.

```bash
python utilities.py analyze
```

#### Analyze a Specific CSV File
To load and plot a specific CSV file:

```bash
python utilities.py analyze --csv path/to/your/file.csv
```

**Features:**
- **Plots**:
  - Desired vs. Measured Pressures.
  - Mocap Trajectory (3D).
  - Orientation (Yaw, Pitch, Roll).
- **Auto-Detection**: Automatically detects whether the file uses "Short" or "Long" column names.
- **Derived Metrics**: Calculates Euler angles (Roll, Pitch, Yaw) from Quaternion data.

---

## Examples

```bash
# Clean raw data in a specific submission folder
python utilities.py clean --folder files_for_submission/clean

# Check for leaks on a specific segment (e.g., connected to Arduino 3) at 4 PSI
python utilities.py leak-test --pumps 3 --sensor 3 --target 4.0

# Analyze a specific consolidated experiment file
python utilities.py analyze --csv files_for_submission/clean/cleaned_data/experiment.csv
```
