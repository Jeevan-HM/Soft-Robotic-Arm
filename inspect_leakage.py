import numpy as np
import pandas as pd

print("Starting inspection...", flush=True)

try:
    csv_path = (
        "files_for_submission/cleaned_data/experiment-refil-end-of-experiment.csv"
    )
    print(f"Reading {csv_path}...", flush=True)
    df = pd.read_csv(csv_path)

    # Clean NaNs to match PRC logic
    measured_cols = [c for c in df.columns if "Measured" in c]
    df = df.dropna(subset=measured_cols).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:]

    col0 = "Desired_pressure_segment_1"
    if col0 in df.columns:
        print(f"Test Set (last 20%) Value Counts for {col0}:")
        counts = test[col0].value_counts()
        print(counts)

        y_true = test[col0]
        variance = y_true.var()
        print(f"Test Set Variance for {col0}: {variance}")

        if variance < 1e-9:
            print(
                "WARNING: Target is effectively constant in Test Set. R2 is not a valid metric."
            )

        # Check Measured stats in test set to see if it's drifting
        m_col = "Measured_pressure_Segment_1_pouch_1"
        if m_col in df.columns:
            m_data = test[m_col]
            print(
                f"Test Set Measured Seg1 Pouch1 - Mean: {m_data.mean():.4f}, Std: {m_data.std():.4f}"
            )
            print(f"Range: {m_data.min():.4f} to {m_data.max():.4f}")
    else:
        print(f"Column {col0} not found.")

except Exception as e:
    print(f"Error: {e}")
