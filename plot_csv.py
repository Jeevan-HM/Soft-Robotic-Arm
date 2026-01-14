import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def plot_csv(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Loaded {filepath}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # clean columns
    df.columns = [c.strip() for c in df.columns]

    # auto-detect time column
    time_col = None
    possible_time_names = [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "t",
        "Seconds",
        "seconds",
    ]
    for col in df.columns:
        if col.lower() in possible_time_names:
            time_col = col
            break

    # Plotting
    plt.figure(figsize=(12, 8))

    # If time column exists, use it as x-axis
    if time_col:
        x = df[time_col]
        # Drop time from y-axis candidates
        y_cols = [c for c in df.columns if c != time_col]
        plt.xlabel(time_col)
    else:
        x = df.index
        y_cols = list(df.columns)
        plt.xlabel("Index")

    # Filter out non-numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    y_cols = [c for c in y_cols if c in numeric_cols]

    if not y_cols:
        print("No numeric data to plot.")
        return

    for col in y_cols:
        plt.plot(x, df[col], label=col)

    plt.title(f"Data from {os.path.basename(filepath)}")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CSV file")
    parser.add_argument("file", nargs="?", help="Path to CSV file")
    args = parser.parse_args()

    if args.file:
        plot_csv(args.file)
    else:
        print("Usage: python simple_plot_csv.py <path_to_csv>")
        # Optional: list recent CSVs?
