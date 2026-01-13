# Thesis Data Analysis

This repository contains tools for analyzing experimental data from robotic arm experiments.

## Data Analysis Script

The `data_analysis.py` script can load and plot experimental data from both HDF5 files and CSV files.

### Usage

#### Auto-select Latest Experiment (Default)
By default, the script will automatically load the most recent experiment from the HDF5 files:

```bash
python data_analysis.py
```

#### Load a Specific CSV File
To load and plot a specific CSV file, use the `--csv` flag:

```bash
python data_analysis.py --csv path/to/your/file.csv
```

Examples:
```bash
# Load CSV from root directory
python data_analysis.py --csv exp_040_axial_Nov12_14h48m_data_with_headers.csv

# Load CSV with absolute path
python data_analysis.py --csv /home/g1/Developer/Thesis/experiments/csv/July-16/Test_1.csv

# Load CSV with relative path
python data_analysis.py --csv experiments/csv/June-20/Test_3.csv
```

### Features

- **Automatic Experiment Selection**: Loads the most recent HDF5 experiment by default
- **CSV File Support**: Load any CSV file with the `--csv` argument
- **Multiple Plot Types**: 
  - Sensor & Control Data (Desired Pressures, Measured Pressures)
  - Motion Capture Data (Position and Orientation)
  - 2D Trajectory Plot (X-Z plane)
  - 3D Trajectory Plot
- **Configurable Font Sizes**: Easy to adjust all plot fonts through the configuration section
- **Data Trimming**: Automatically skips the first 10 seconds of data (configurable)

### Requirements

See `requirements.txt` for Python dependencies.
