import csv
import os

# ---- CONFIGURATION ----

OUTPUT_FOLDER_NAME = "cleaned_data"
INPUT_FODLER_NAME = "files_for_submission"
# This dictionary defines all the renaming operations.
# 'old_name': 'new_name'
COLUMN_RENAME_MAP = {
    "pd_3": "Desired_pressure_segment_1",
    "pd_6": "Desired_pressure_segment_2",
    "pd_7": "Desired_pressure_segment_3",
    "pd_8": "Desired_pressure_segment_4",
    "pm_3_1": "Measured_pressure_Segment_1_pouch_1",
    "pm_3_2": "Measured_pressure_Segment_1_pouch_2",
    "pm_3_3": "Measured_pressure_Segment_1_pouch_3",
    "pm_3_4": "Measured_pressure_Segment_1_pouch_4",
    "pm_7_1": "Measured_pressure_Segment_1_pouch_5",
    "pm_7_2": "Measured_pressure_Segment_2_pouch_1",
    "pm_7_3": "Measured_pressure_Segment_2_pouch_2",
    "pm_7_4": "Measured_pressure_Segment_2_pouch_3",
    "pm_8_1": "Measured_pressure_Segment_2_pouch_4",
    "pm_8_2": "Measured_pressure_Segment_2_pouch_5",
    "pm_8_4": "Measured_pressure_Segment_3",
    "pm_8_3": "Measured_pressure_Segment_4",
    # rename mocap3 columns to the new generic rigid body names
    "mocap_3_x": "mocap_rigid_body_x",
    "mocap_3_y": "mocap_rigid_body_y",
    "mocap_3_z": "mocap_rigid_body_z",
    "mocap_3_qx": "mocap_rigid_body_qx",
    "mocap_3_qy": "mocap_rigid_body_qy",
    "mocap_3_qz": "mocap_rigid_body_qz",
    "mocap_3_qw": "mocap_rigid_body_qw",
}

# This list contains all columns that should be completely removed.
COLUMNS_TO_DROP = [
    "mocap_1_x",
    "mocap_1_y",
    "mocap_1_z",
    "mocap_1_qx",
    "mocap_1_qy",
    "mocap_1_qz",
    "mocap_1_qw",
    "mocap_2_x",
    "mocap_2_y",
    "mocap_2_z",
    "mocap_2_qx",
    "mocap_2_qy",
    "mocap_2_qz",
    "mocap_2_qw",
    "pm_6_1",
    "pm_6_2",
    "pm_6_3",
    "pm_6_4",
]

# ---- END CONFIGURATION ----


def process_and_clean_csv(input_filepath, output_dir):
    """
    Reads a single CSV, renames and drops columns, and saves it to the output directory.
    """
    base_filename = os.path.basename(input_filepath)
    try:
        print(f"--- Processing: {base_filename} ---")

        # Read the original CSV file
        with open(input_filepath, "r", newline="") as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        if not rows:
            print("Error: Empty CSV file")
            return

        headers = rows[0]
        data_rows = rows[1:]

        # --- Column Transformation ---

        # 1. Find columns to drop and keep
        cols_to_drop_indices = []
        existing_cols_to_drop = []
        for i, header in enumerate(headers):
            if header in COLUMNS_TO_DROP:
                cols_to_drop_indices.append(i)
                existing_cols_to_drop.append(header)

        print(f"Dropped {len(existing_cols_to_drop)} columns.")

        # 2. Keep only the columns we want and rename them
        new_headers = []
        kept_indices = []
        for i, header in enumerate(headers):
            if i not in cols_to_drop_indices:
                # Rename if in rename map, otherwise keep original name
                new_name = COLUMN_RENAME_MAP.get(header, header)
                new_headers.append(new_name)
                kept_indices.append(i)

        print("Renamed columns.")

        # Filter data rows to keep only the desired columns
        filtered_data_rows = []
        for row in data_rows:
            filtered_row = [row[i] if i < len(row) else "" for i in kept_indices]
            filtered_data_rows.append(filtered_row)

        # --- Saving the new file ---
        output_filepath = os.path.join(output_dir, base_filename)
        with open(output_filepath, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(new_headers)
            writer.writerows(filtered_data_rows)

        print(f"Successfully saved cleaned file to: {output_filepath}\n")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing {base_filename}: {e}")


def main():
    """
    Main function to ask for a folder path and process all CSV files within it.
    """
    # Loop until the user provides a valid directory path
    while True:
        target_folder = INPUT_FODLER_NAME
        if os.path.isdir(target_folder):
            break
        else:
            print("Invalid path. Please enter a valid folder path.")

    # --- Prepare Output Directory ---
    # Create the 'cleaned_data' directory inside the user-specified folder
    output_dir = os.path.join(target_folder, OUTPUT_FOLDER_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print("-" * 50)
    print(f"Cleaned files will be saved in: {output_dir}")
    print("-" * 50)

    # --- Find and Process CSV Files ---
    csv_files = [f for f in os.listdir(target_folder) if f.lower().endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in '{target_folder}'.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process.\n")

    # Loop through each found CSV file and process it
    for filename in csv_files:
        full_input_path = os.path.join(target_folder, filename)
        process_and_clean_csv(full_input_path, output_dir)

    print("=" * 50)
    print("All files processed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
