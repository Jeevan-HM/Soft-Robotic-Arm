import logging
import os

import matplotlib.pyplot as plt

# =================================================================================
# ---- LOGGING SETUP ----
# =================================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("utilities")

# =================================================================================
# ---- PATHS & FOLDERS ----
# =================================================================================
EXPERIMENTS_BASE_DIR = "/Users/g1/Developer/Soft-Robotic-Arm/experiments"
CLEAN_OUTPUT_FOLDER_NAME = "cleaned_data"
CLEAN_INPUT_FOLDER_NAME = "clean_data_folder"

# =================================================================================
# ---- CLEANING CONFIGURATION ----
# =================================================================================
COLUMN_RENAME_MAP = {
    # Desired pressures
    "pd_3": "Desired_pressure_segment_1",
    "pd_6": "Desired_pressure_segment_2",
    "pd_7": "Desired_pressure_segment_3",
    "pd_8": "Desired_pressure_segment_4",
    # Segment 1: 5 pouches (Arduino 3 + Arduino 7)
    "pm_3_1": "Measured_pressure_Segment_1_pouch_1",
    "pm_3_2": "Measured_pressure_Segment_1_pouch_2",
    "pm_3_3": "Measured_pressure_Segment_1_pouch_3",
    "pm_3_4": "Measured_pressure_Segment_1_pouch_4",
    "pm_7_1": "Measured_pressure_Segment_1_pouch_5",
    # Segment 2: 1 sensor only (Arduino 7)
    "pm_7_2": "Measured_pressure_Segment_2",
    # Segment 3 and 4: from Arduino 7
    "pm_7_3": "Measured_pressure_Segment_3",
    "pm_7_4": "Measured_pressure_Segment_4",
    # Mocap 1 -> Rigid_body_1
    "mocap_1_x": "Rigid_body_1_x",
    "mocap_1_y": "Rigid_body_1_y",
    "mocap_1_z": "Rigid_body_1_z",
    "mocap_1_qx": "Rigid_body_1_qx",
    "mocap_1_qy": "Rigid_body_1_qy",
    "mocap_1_qz": "Rigid_body_1_qz",
    "mocap_1_qw": "Rigid_body_1_qw",
    # Mocap 2 -> Rigid_body_2
    "mocap_2_x": "Rigid_body_2_x",
    "mocap_2_y": "Rigid_body_2_y",
    "mocap_2_z": "Rigid_body_2_z",
    "mocap_2_qx": "Rigid_body_2_qx",
    "mocap_2_qy": "Rigid_body_2_qy",
    "mocap_2_qz": "Rigid_body_2_qz",
    "mocap_2_qw": "Rigid_body_2_qw",
    # Mocap 3 -> Rigid_body_3
    "mocap_3_x": "Rigid_body_3_x",
    "mocap_3_y": "Rigid_body_3_y",
    "mocap_3_z": "Rigid_body_3_z",
    "mocap_3_qx": "Rigid_body_3_qx",
    "mocap_3_qy": "Rigid_body_3_qy",
    "mocap_3_qz": "Rigid_body_3_qz",
    "mocap_3_qw": "Rigid_body_3_qw",
}

COLUMNS_TO_DROP = [
    # Arduino 6 - not used
    "pm_6_1",
    "pm_6_2",
    "pm_6_3",
    "pm_6_4",
    # Arduino 8 - no longer used
    "pm_8_1",
    "pm_8_2",
    "pm_8_3",
    "pm_8_4",
]

# =================================================================================
# ---- ANALYSIS & PLOTTING CONFIGURATION ----
# =================================================================================
START_TIME_OFFSET_SEC = 0
TIME_COL = "time"

# -- Column Name Sets --
COL_SET_SHORT = {
    "DESIRED_PRESSURE_COLS": ["pd_3", "pd_7"],
    "MEASURED_PRESSURE_SEGMENT1_COLS": [
        "pm_3_1",
        "pm_3_2",
        "pm_3_3",
        "pm_3_4",
        "pm_7_1",
    ],
    "MEASURED_PRESSURE_SEGMENT2_COLS": ["pm_7_2"],
    "MEASURED_PRESSURE_SEGMENT3_COLS": ["pm_7_3"],
    "MEASURED_PRESSURE_SEGMENT4_COLS": ["pm_7_4"],
    "MOCAP_POS_COLS": ["mocap_3_x", "mocap_3_y", "mocap_3_z"],
    "MOCAP_QUAT_COLS": ["mocap_3_qx", "mocap_3_qy", "mocap_3_qz", "mocap_3_qw"],
}

COL_SET_LONG = {
    "DESIRED_PRESSURE_COLS": [
        "Desired_pressure_segment_1",
        "Desired_pressure_segment_2",
        "Desired_pressure_segment_3",
        "Desired_pressure_segment_4",
    ],
    "MEASURED_PRESSURE_SEGMENT1_COLS": [
        "Measured_pressure_Segment_1_pouch_1",
        "Measured_pressure_Segment_1_pouch_2",
        "Measured_pressure_Segment_1_pouch_3",
        "Measured_pressure_Segment_1_pouch_4",
        "Measured_pressure_Segment_1_pouch_5",
    ],
    "MEASURED_PRESSURE_SEGMENT2_COLS": [
        "Measured_pressure_Segment_2_pouch_1",
        "Measured_pressure_Segment_2_pouch_2",
        "Measured_pressure_Segment_2_pouch_3",
        "Measured_pressure_Segment_2_pouch_4",
        "Measured_pressure_Segment_2_pouch_5",
    ],
    "MEASURED_PRESSURE_SEGMENT3_COLS": ["Measured_pressure_Segment_3"],
    "MEASURED_PRESSURE_SEGMENT4_COLS": ["Measured_pressure_Segment_4"],
    "MOCAP_POS_COLS": [
        "Rigid_body_3_x",
        "Rigid_body_3_y",
        "Rigid_body_3_z",
    ],
    "MOCAP_QUAT_COLS": [
        "Rigid_body_3_qx",
        "Rigid_body_3_qy",
        "Rigid_body_3_qz",
        "Rigid_body_3_qw",
    ],
}

# Globals to be updated (Mutable Lists)
DESIRED_PRESSURE_COLS = list(COL_SET_SHORT["DESIRED_PRESSURE_COLS"])
MEASURED_PRESSURE_SEGMENT1_COLS = list(COL_SET_SHORT["MEASURED_PRESSURE_SEGMENT1_COLS"])
MEASURED_PRESSURE_SEGMENT2_COLS = list(COL_SET_SHORT["MEASURED_PRESSURE_SEGMENT2_COLS"])
MEASURED_PRESSURE_SEGMENT3_COLS = list(COL_SET_SHORT["MEASURED_PRESSURE_SEGMENT3_COLS"])
MEASURED_PRESSURE_SEGMENT4_COLS = list(COL_SET_SHORT["MEASURED_PRESSURE_SEGMENT4_COLS"])
MOCAP_POS_COLS = list(COL_SET_SHORT["MOCAP_POS_COLS"])
MOCAP_QUAT_COLS = list(COL_SET_SHORT["MOCAP_QUAT_COLS"])

YAW_BODY_NAME = "yaw_body"
PITCH_BODY_NAME = "pitch_body"
ROLL_BODY_NAME = "roll_body"

# -- Plot Styling --
BASE_FONT_SIZE = 18
LABEL_PADDING = 24
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 10,
        "ytick.labelsize": BASE_FONT_SIZE - 10,
        "axes.labelsize": LABEL_PADDING - 10,
        "axes.labelpad": LABEL_PADDING,
        "axes.titlesize": BASE_FONT_SIZE + 4,
        "figure.titlesize": BASE_FONT_SIZE + 6,
        "legend.fontsize": BASE_FONT_SIZE,
    }
)


def get_plot_configs():
    # Helper to return current config based on global columns
    return {
        "mocap": [
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
        ],
        "sensor1": [
            {
                "title": "Desired Pressures",
                "xlabel": "Time (s)",
                "ylabel": "Desired Pressure (PSI)",
                "columns": DESIRED_PRESSURE_COLS,
                "labels": [
                    "pd_segment_1",
                    "pd_segment_2",
                    "pd_segment_3",
                    "pd_segment_4",
                ],
                "colors": ["tab:red", "tab:orange", "tab:blue", "tab:green"],
            },
            {
                "title": "Measured Pressures (Segment 3)",
                "xlabel": "Time (s)",
                "ylabel": "Sensor Pressure (PSI)",
                "columns": MEASURED_PRESSURE_SEGMENT3_COLS,
                "labels": ["pm_segment_3"],
                "colors": ["tab:blue"],
            },
            {
                "title": "Measured Pressures (Segment 4)",
                "xlabel": "Time (s)",
                "ylabel": "Sensor Pressure (PSI)",
                "columns": MEASURED_PRESSURE_SEGMENT4_COLS,
                "labels": ["pm_segment_4"],
                "colors": ["tab:purple"],
            },
        ],
        "sensor2": [
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
                ],
                "colors": ["tab:orange"],
            },
        ],
    }
