"""
Physical Reservoir Computing (PRC) - Bending Angle Prediction
==============================================================

Predicts bending angle of a soft robotic arm using:
- Segment 1 (5 pouches): Reservoir
- Segments 2,3,4 (7 sensors): Actuation input
- Tapped delay line (15 taps) for temporal features
- Ridge regression for linear readout

Interactive plots (zoom, pan, reset via toolbar)
"""

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Sensor columns (Updated configuration)
# Segment 1: 5 pouches (reservoir)
SEGMENT_1_COLS = [f"Measured_pressure_Segment_1_pouch_{i}" for i in range(1, 6)]
# Segment 2: 1 sensor only (no longer 5 pouches)
SEGMENT_2_COLS = ["Measured_pressure_Segment_2"]
# Segments 3-4: 1 sensor each
SEGMENT_34_COLS = ["Measured_pressure_Segment_3", "Measured_pressure_Segment_4"]

# PRC Architecture:
# - Reservoir: Segment 1 (5 pouches) - the physical body whose dynamics we exploit
# - Input/Context: Segments 2-4 (3 sensors) - actuation signals driving the reservoir
RESERVOIR_COLS = SEGMENT_1_COLS  # Segment 1 as reservoir
INPUT_COLS = SEGMENT_2_COLS + SEGMENT_34_COLS  # Segments 2-4 as input/context

# Training hyperparameters
TAPS = 15  # Number of temporal taps for delay line
RIDGE_ALPHA = 0.01  # Ridge regression regularization
TRAIN_FRACTION = 0.5  # Train/test split ratio
TRIM_SECONDS = 10  # Seconds to trim from start/end


# =============================================================================
# CORE PRC FUNCTIONS
# =============================================================================


def build_tapped_delay_line(X, taps):
    """Create [x(t), x(t-1), ..., x(t-taps+1)] features."""
    T, n_features = X.shape
    X_tapped = np.zeros((T, n_features * taps))
    for k in range(taps):
        X_tapped[k:, k * n_features : (k + 1) * n_features] = X[: T - k, :]
    X_tapped[: taps - 1, :] = np.nan
    return X_tapped


def compute_bending_angle(df, top_prefix, tip_prefix="Rigid_body_3"):
    """Bending angle from mocap: θ = arctan(horizontal / vertical)."""
    dx = df[f"{tip_prefix}_x"].values - df[f"{top_prefix}_x"].values
    dy = df[f"{tip_prefix}_y"].values - df[f"{top_prefix}_y"].values
    dz = df[f"{tip_prefix}_z"].values - df[f"{top_prefix}_z"].values
    return np.arctan2(np.sqrt(dx**2 + dz**2), np.abs(dy)) * 180 / np.pi


def load_and_preprocess(filepath, top_marker_prefix):
    """Load and clean data."""
    df = pd.read_csv(filepath)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mocap_cols = [f"{top_marker_prefix}_{c}" for c in ["x", "y", "z"]] + [
        f"Rigid_body_3_{c}" for c in ["x", "y", "z"]
    ]
    sensor_cols = SEGMENT_1_COLS + SEGMENT_2_COLS + SEGMENT_34_COLS

    df = df.dropna(subset=sensor_cols + mocap_cols).reset_index(drop=True)
    df = df[df[f"{top_marker_prefix}_x"] != 0].reset_index(drop=True)

    t0 = df["time"].iloc[0]
    df["t_rel"] = df["time"] - t0
    t_max = df["t_rel"].iloc[-1]
    df = df[
        (df["t_rel"] >= TRIM_SECONDS) & (df["t_rel"] <= t_max - TRIM_SECONDS)
    ].reset_index(drop=True)

    return df


def train_prc_model(X_reservoir, X_actuation, y):
    """Train PRC: tapped delay + Ridge regression."""
    X_res_tapped = build_tapped_delay_line(X_reservoir, TAPS)
    X_act_tapped = build_tapped_delay_line(X_actuation, TAPS)
    X_combined = np.hstack([X_res_tapped, X_act_tapped])

    valid = ~np.isnan(X_combined).any(axis=1)
    X_combined, y_valid = X_combined[valid], y[valid]

    n_train = int(len(y_valid) * TRAIN_FRACTION)
    X_train, X_test = X_combined[:n_train], X_combined[n_train:]
    y_train, y_test = y_valid[:n_train], y_valid[n_train:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    error_pct = (mae / np.ptp(y_test)) * 100

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "R2": r2,
        "MAE": mae,
        "Error_Percent": error_pct,
    }


# =============================================================================
# PLOTTING - ESSENTIAL ONLY
# =============================================================================


def plot_results(all_results, sample_rate=100):
    """
    Single figure with time series and scatter for both configurations.
    This is the most informative view of PRC performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (name, res) in enumerate(all_results.items()):
        y_test, y_pred = res["y_test"], res["y_pred"]
        t = np.arange(len(y_test)) / sample_rate
        step = max(1, len(t) // 10000)

        # Time series (left column)
        ax = axes[i, 0]
        ax.plot(t[::step], y_test[::step], "b-", label="Ground Truth", linewidth=1.2)
        ax.plot(
            t[::step],
            y_pred[::step],
            "r-",
            label="PRC Prediction",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Bending Angle (°)")
        ax.set_title(
            f"{name}\nR² = {res['R2']:.3f} | Error = {res['Error_Percent']:.1f}%",
            fontweight="bold",
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Scatter (right column)
        ax = axes[i, 1]
        ax.scatter(y_test[::step], y_pred[::step], alpha=0.3, s=10, c="steelblue")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect")
        ax.set_xlabel("Actual (°)")
        ax.set_ylabel("Predicted (°)")
        ax.set_title(f"Predicted vs Actual (R² = {res['R2']:.3f})", fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle("PRC Bending Angle Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRC - BENDING ANGLE PREDICTION")
    print("=" * 60)
    print("\nArchitecture:")
    print(f"  Reservoir (Segment 1): {len(RESERVOIR_COLS)} sensors")
    print(f"  Input/Context (Segments 2-4): {len(INPUT_COLS)} sensors")
    print(f"  Temporal taps: {TAPS}")

    datasets = [
        ("No Valve", "files_for_submission/experiment.csv", "Rigid_body_1"),
        ("With Valve", "files_for_submission/segment_5_valve.csv", "Rigid_body_2"),
    ]

    all_results = {}

    for name, filepath, top_marker in datasets:
        print(f"\n{name}:")
        df = load_and_preprocess(filepath, top_marker)
        y = compute_bending_angle(df, top_marker)
        # Reservoir: Segment 1 only
        X_res = df[RESERVOIR_COLS].values
        # Input/Context: Segments 2-4
        X_act = df[INPUT_COLS].values

        res = train_prc_model(X_res, X_act, y)
        all_results[name] = res

        print(
            f"  R² = {res['R2']:.4f}, MAE = {res['MAE']:.3f}°, Error = {res['Error_Percent']:.2f}%"
        )

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Config':<12} {'R²':>8} {'MAE':>8} {'Error %':>10}")
    print("-" * 40)
    for name, res in all_results.items():
        print(
            f"{name:<12} {res['R2']:>8.4f} {res['MAE']:>8.3f} {res['Error_Percent']:>10.2f}"
        )
    print("=" * 60)

    # Plot
    print("\nDisplaying results (use toolbar to zoom/pan)...")
    fig = plot_results(all_results)
    plt.show()

    print("\nDone!")
