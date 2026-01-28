import csv
import os

from . import config


class DataCleaner:
    @staticmethod
    def process_and_clean_csv(input_filepath, output_dir):
        base_filename = os.path.basename(input_filepath)
        try:
            print(f"--- Processing: {base_filename} ---")
            with open(input_filepath, "r", newline="") as infile:
                reader = csv.reader(infile)
                rows = list(reader)

            if not rows:
                print("Error: Empty CSV file")
                return

            headers = rows[0]
            data_rows = rows[1:]

            cols_to_drop_indices = []
            existing_cols_to_drop = []
            for i, header in enumerate(headers):
                if header in config.COLUMNS_TO_DROP:
                    cols_to_drop_indices.append(i)
                    existing_cols_to_drop.append(header)

            print(f"Dropped {len(existing_cols_to_drop)} columns.")

            new_headers = []
            kept_indices = []
            for i, header in enumerate(headers):
                if i not in cols_to_drop_indices:
                    new_name = config.COLUMN_RENAME_MAP.get(header, header)
                    new_headers.append(new_name)
                    kept_indices.append(i)

            print("Renamed columns.")

            filtered_data_rows = []
            for row in data_rows:
                filtered_row = [row[i] if i < len(row) else "" for i in kept_indices]
                filtered_data_rows.append(filtered_row)

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

    @staticmethod
    def run_cleaning(target_folder=None):
        if not target_folder:
            # Interactive mode if no folder provided
            while True:
                target_folder = config.CLEAN_INPUT_FOLDER_NAME
                if os.path.isdir(target_folder):
                    break
                else:
                    ans = input("Default folder not found. Enter path: ").strip()
                    if os.path.isdir(ans):
                        target_folder = ans
                        break
                    print("Invalid path.")

        output_dir = os.path.join(target_folder, config.CLEAN_OUTPUT_FOLDER_NAME)
        os.makedirs(output_dir, exist_ok=True)
        print("-" * 50)
        print(f"Cleaned files will be saved in: {output_dir}")
        print("-" * 50)

        csv_files = [f for f in os.listdir(target_folder) if f.lower().endswith(".csv")]

        if not csv_files:
            print(f"No CSV files found in '{target_folder}'.")
            return

        print(f"Found {len(csv_files)} CSV file(s) to process.\n")

        for filename in csv_files:
            full_input_path = os.path.join(target_folder, filename)
            DataCleaner.process_and_clean_csv(full_input_path, output_dir)

        print("=" * 50)
        print("All files processed.")
        print("=" * 50)
