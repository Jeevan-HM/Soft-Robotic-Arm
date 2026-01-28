"""
Physical Reservoir Computing (PRC) and Limit Cycle Analysis
for Soft Robotic Arm - Version 2 (Corrected)

Based on:
1. Wang et al. (2024) - "Proprioceptive and Exteroceptive Information Perception
   in a Fabric Soft Robotic Arm via Physical Reservoir Computing"
2. Eder et al. (2018) - "Morphological computation-based control of a modular,
   pneumatically driven, soft robotic arm"

Key Changes from v1:
- Tapped delay line for reservoir (Segment 1) only
- Actuation signals (Segments 2-4) used as context without taps
- Robust mocap validity checks
- Low-pass filtering for bending angle
- 70/30 train-test split
- Proper limit cycle identification (not control)

Usage:
    python prc_limit_cycle_v2.py --data path/to/data.csv --mode [prc|limit_cycle|both]

Author: Generated for soft robot research
"""

import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("TkAgg")  # Interactive display
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Configuration parameters for PRC analysis."""

    # Sensor columns - CORRECTED based on actual data structure (segment_5_valve_2)
    # Segment 1: Reservoir (sensing column) - 5 pouches
    SEGMENT_1_COLS = [f"Measured_pressure_Segment_1_pouch_{i}" for i in range(1, 6)]

    # Segments 2, 3, 4: Single pressure readings (actuation columns)
    # NOTE: In this dataset, only Segment 1 has pouch-resolved sensors
    SEGMENT_2_COLS = ["Measured_pressure_Segment_2"]
    SEGMENT_3_COLS = ["Measured_pressure_Segment_3"]
    SEGMENT_4_COLS = ["Measured_pressure_Segment_4"]

    # Combined actuation columns
    ACTUATION_COLS = SEGMENT_2_COLS + SEGMENT_3_COLS + SEGMENT_4_COLS

    # Motion capture columns
    MOCAP_TIP = "Rigid_body_3"  # End effector
    MOCAP_TOP = "Rigid_body_2"  # Top of robot (for bending angle calculation)

    # Training parameters - CORRECTED: 70/30 split
    TRAIN_FRACTION = 0.7
    RIDGE_ALPHA = 0.01
    TRIM_SECONDS = 10  # Trim from start/end to remove transients
    WASHOUT = 250  # Samples to discard at start (for limit cycle)

    # Tapped Delay Line parameters
    DEFAULT_TAPS = 20  # Number of time delays for reservoir

    # Signal processing
    SAMPLE_RATE = 100.0  # Hz
    LOWPASS_CUTOFF = 3.0  # Hz for bending angle filter

    # Wave type mapping
    WAVE_TYPES = {1.0: "axial", 2.0: "circular", 3.0: "triangular", -1.0: "cooldown"}


# =============================================================================
# SIGNAL PROCESSING UTILITIES
# =============================================================================


def lowpass_filter(
    data: np.ndarray, cutoff: float = 3.0, fs: float = 100.0, order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to reduce noise.

    Args:
        data: Input signal (1D or 2D with samples on axis 0)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order

    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq

    # Ensure cutoff is valid
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    b, a = signal.butter(order, normalized_cutoff, btype="low")

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        # Filter each column
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])
        return filtered


def build_tapped_delay_line(X: np.ndarray, taps: int) -> np.ndarray:
    """
    Build tapped delay line (TDL) for reservoir computing.

    Creates time-delayed copies of input signals to provide memory.

    For signal x(t), creates: [x(t), x(t-1), x(t-2), ..., x(t-taps+1)]

    Args:
        X: Input array of shape (n_samples, n_features)
        taps: Number of time delays (including current)

    Returns:
        Tapped array of shape (n_samples - taps + 1, n_features * taps)

    Example:
        X with 5 sensors and 10 taps -> 50 features
    """
    if taps < 1:
        # taps=0 means no memory, just use current values (equivalent to taps=1)
        taps = 1

    n_samples, n_features = X.shape
    n_valid = n_samples - taps + 1

    if n_valid <= 0:
        raise ValueError(f"Not enough samples ({n_samples}) for {taps} taps")

    # Build tapped delay line
    X_tapped = np.zeros((n_valid, n_features * taps))

    for t in range(taps):
        start_idx = taps - 1 - t
        end_idx = start_idx + n_valid
        X_tapped[:, t * n_features : (t + 1) * n_features] = X[start_idx:end_idx]

    return X_tapped


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def load_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the experimental data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Raw samples: {len(df)}")
    return df


def validate_mocap(
    df: pd.DataFrame, top_marker: str = "Rigid_body_2", tip_marker: str = "Rigid_body_3"
) -> pd.DataFrame:
    """
    Robust motion capture validity check.

    Removes rows where tracking was lost (all zeros or NaN).
    """
    eps = 1e-6

    # Check all mocap columns
    mocap_cols = [
        f"{top_marker}_x",
        f"{top_marker}_y",
        f"{top_marker}_z",
        f"{tip_marker}_x",
        f"{tip_marker}_y",
        f"{tip_marker}_z",
    ]

    # Validity mask: sum of absolute values > eps (not all zeros)
    mask = df[mocap_cols].abs().sum(axis=1) > eps

    # Also check for NaN
    mask &= ~df[mocap_cols].isna().any(axis=1)

    df_valid = df[mask].reset_index(drop=True)

    n_removed = len(df) - len(df_valid)
    if n_removed > 0:
        print(f"  Removed {n_removed} samples with invalid mocap data")

    return df_valid


def preprocess_data(df: pd.DataFrame, top_marker: str = "Rigid_body_2") -> pd.DataFrame:
    """Clean data and compute derived features."""

    # Define required columns
    sensor_cols = Config.SEGMENT_1_COLS + Config.ACTUATION_COLS

    # Check which columns actually exist
    existing_sensor_cols = [c for c in sensor_cols if c in df.columns]
    missing_cols = set(sensor_cols) - set(existing_sensor_cols)
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")

    # Drop rows with NaN in essential columns
    df = df.dropna(subset=existing_sensor_cols).reset_index(drop=True)

    # Robust mocap validation
    df = validate_mocap(df, top_marker, Config.MOCAP_TIP)

    # Compute relative time
    if "time" in df.columns:
        t0 = df["time"].iloc[0]
        df["t_rel"] = df["time"] - t0
        t_max = df["t_rel"].iloc[-1]

        # Trim start and end
        df = df[
            (df["t_rel"] >= Config.TRIM_SECONDS)
            & (df["t_rel"] <= t_max - Config.TRIM_SECONDS)
        ].reset_index(drop=True)

    print(f"  Cleaned samples: {len(df)}")
    return df


def compute_bending_angle(
    df: pd.DataFrame, top_marker: str = "Rigid_body_2", apply_filter: bool = True
) -> np.ndarray:
    """
    Compute bending angle from motion capture data.

    Bending angle = arctan2(horizontal_displacement, vertical_displacement)

    Args:
        df: DataFrame with mocap data
        top_marker: Reference marker name
        apply_filter: Whether to apply low-pass filter

    Returns:
        Bending angle in degrees
    """
    tip = Config.MOCAP_TIP

    dx = df[f"{tip}_x"].values - df[f"{top_marker}_x"].values
    dy = df[f"{tip}_y"].values - df[f"{top_marker}_y"].values
    dz = df[f"{tip}_z"].values - df[f"{top_marker}_z"].values

    # Bending angle in degrees
    bending_angle = np.arctan2(np.sqrt(dx**2 + dz**2), np.abs(dy)) * 180 / np.pi

    # Apply low-pass filter to reduce noise
    if apply_filter and len(bending_angle) > 20:
        bending_angle = lowpass_filter(
            bending_angle, cutoff=Config.LOWPASS_CUTOFF, fs=Config.SAMPLE_RATE
        )

    return bending_angle


# =============================================================================
# PHYSICAL RESERVOIR COMPUTING (PRC) - Wang et al. Method
# =============================================================================


class PhysicalReservoirComputer:
    """
    Physical Reservoir Computing for soft robot proprioception.

    Based on Wang et al. (2024): Uses pressure sensor readings as reservoir
    states and learns a linear readout to predict bending angle.

    CORRECTED Implementation:
    - Segment 1 (reservoir): Tapped delay line for memory
    - Segments 2-4 (actuation): Current values only as context

    Key equation:
        y(t) = w0 + Σ wi * si(t)

    where si(t) are (tapped) sensor readings and wi are learned weights.
    """

    def __init__(self, taps: int = 20, ridge_alpha: float = 0.01):
        """
        Initialize PRC.

        Args:
            taps: Number of time delays for reservoir (Segment 1)
            ridge_alpha: Regularization strength
        """
        self.taps = taps
        self.ridge_alpha = ridge_alpha
        self.weights = None
        self.scaler_mean = None
        self.scaler_std = None
        self.is_fitted = False
        self.n_reservoir_features = None
        self.n_context_features = None

    def _prepare_features(
        self, X_reservoir: np.ndarray, X_context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Prepare features with tapped delay line for reservoir only.

        Args:
            X_reservoir: Reservoir (Segment 1) readings (n_samples, n_reservoir_sensors)
            X_context: Context (actuation) readings (n_samples, n_context_sensors)

        Returns:
            Combined feature array with reservoir tapped and context current
        """
        # Apply tapped delay line to reservoir
        X_reservoir_tapped = build_tapped_delay_line(X_reservoir, self.taps)
        n_valid = len(X_reservoir_tapped)

        if X_context is not None:
            # Align context with tapped reservoir (drop first taps-1 samples)
            X_context_aligned = X_context[self.taps - 1 :]

            # Combine: [reservoir_tapped, context_current]
            X_combined = np.hstack([X_reservoir_tapped, X_context_aligned])
        else:
            X_combined = X_reservoir_tapped

        return X_combined

    def fit(
        self,
        X_reservoir: np.ndarray,
        y: np.ndarray,
        X_context: Optional[np.ndarray] = None,
    ) -> "PhysicalReservoirComputer":
        """
        Train the readout weights using ridge regression.

        Args:
            X_reservoir: Reservoir sensor readings (n_samples, n_reservoir_sensors)
            y: Target output (n_samples,)
            X_context: Optional context/actuation readings (n_samples, n_context_sensors)

        Returns:
            self
        """
        # Store feature dimensions
        self.n_reservoir_features = X_reservoir.shape[1]
        self.n_context_features = X_context.shape[1] if X_context is not None else 0

        # Prepare features
        X = self._prepare_features(X_reservoir, X_context)

        # Align target with tapped features
        y_aligned = y[self.taps - 1 :]

        # Standardize inputs
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        # Add bias term
        X_bias = np.hstack([np.ones((len(X), 1)), X_scaled])

        # Ridge regression
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        model.fit(X_bias, y_aligned)
        self.weights = model.coef_

        self.is_fitted = True
        return self

    def predict(
        self, X_reservoir: np.ndarray, X_context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict output using learned readout weights.

        Args:
            X_reservoir: Reservoir sensor readings
            X_context: Optional context/actuation readings

        Returns:
            Predicted output
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(X_reservoir, X_context)
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_bias = np.hstack([np.ones((len(X), 1)), X_scaled])

        return X_bias @ self.weights

    def evaluate(
        self,
        X_reservoir: np.ndarray,
        y: np.ndarray,
        X_context: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        y_pred = self.predict(X_reservoir, X_context)

        # Align target
        y_aligned = y[self.taps - 1 :]

        mae = mean_absolute_error(y_aligned, y_pred)
        rmse = np.sqrt(mean_squared_error(y_aligned, y_pred))
        r2 = r2_score(y_aligned, y_pred)

        # Error percentage (normalized by range)
        y_range = np.ptp(y_aligned)
        error_pct = (mae / y_range) * 100 if y_range > 0 else 0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "error_pct": error_pct,
            "y_pred": y_pred,
            "y_actual": y_aligned,
        }


def run_prc_analysis(
    df: pd.DataFrame,
    reservoir_cols: List[str],
    context_cols: Optional[List[str]] = None,
    taps: int = 20,
    top_marker: str = "Rigid_body_2",
) -> Dict:
    """
    Run PRC analysis with proper reservoir structure.

    Args:
        df: Preprocessed dataframe
        reservoir_cols: Segment 1 sensor columns (will be tapped)
        context_cols: Actuation sensor columns (no tapping)
        taps: Number of time delays for reservoir
        top_marker: Motion capture marker for bending angle calculation

    Returns:
        Dictionary with results and metrics
    """
    print("\n" + "=" * 60)
    print("PHYSICAL RESERVOIR COMPUTING ANALYSIS")
    print("=" * 60)

    # Filter columns that exist
    reservoir_cols = [c for c in reservoir_cols if c in df.columns]
    if context_cols:
        context_cols = [c for c in context_cols if c in df.columns]

    # Extract features
    X_reservoir = df[reservoir_cols].values
    X_context = df[context_cols].values if context_cols else None
    y = compute_bending_angle(df, top_marker, apply_filter=True)

    print(f"Reservoir sensors (Segment 1): {len(reservoir_cols)}")
    print(f"Context sensors (Actuation): {len(context_cols) if context_cols else 0}")
    print(f"Taps: {taps}")
    print(
        f"Total features: {len(reservoir_cols) * taps + (len(context_cols) if context_cols else 0)}"
    )
    print(f"Samples: {len(X_reservoir)}")
    print(f"Bending angle range: {y.min():.2f}° to {y.max():.2f}°")

    # Train/test split (70/30)
    n_train = int(len(X_reservoir) * Config.TRAIN_FRACTION)

    X_res_train, X_res_test = X_reservoir[:n_train], X_reservoir[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    if X_context is not None:
        X_ctx_train, X_ctx_test = X_context[:n_train], X_context[n_train:]
    else:
        X_ctx_train, X_ctx_test = None, None

    # Train PRC
    prc = PhysicalReservoirComputer(taps=taps, ridge_alpha=Config.RIDGE_ALPHA)
    prc.fit(X_res_train, y_train, X_ctx_train)

    # Evaluate
    train_metrics = prc.evaluate(X_res_train, y_train, X_ctx_train)
    test_metrics = prc.evaluate(X_res_test, y_test, X_ctx_test)

    print(
        f"\nTraining: MAE={train_metrics['mae']:.3f}°, R²={train_metrics['r2']:.3f}, Error={train_metrics['error_pct']:.1f}%"
    )
    print(
        f"Testing:  MAE={test_metrics['mae']:.3f}°, R²={test_metrics['r2']:.3f}, Error={test_metrics['error_pct']:.1f}%"
    )

    return {
        "model": prc,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "taps": taps,
    }


def run_prc_taps_comparison(
    df: pd.DataFrame,
    reservoir_cols: List[str],
    context_cols: Optional[List[str]] = None,
    tap_values: List[int] = [0, 1, 5, 10, 15, 20, 25, 30],
    top_marker: str = "Rigid_body_2",
) -> pd.DataFrame:
    """
    Compare PRC performance across different tap values.

    Returns:
        DataFrame with results for each tap value
    """
    print("\n" + "=" * 60)
    print("PRC TAPS COMPARISON")
    print("=" * 60)

    # Filter columns
    reservoir_cols = [c for c in reservoir_cols if c in df.columns]
    if context_cols:
        context_cols = [c for c in context_cols if c in df.columns]

    X_reservoir = df[reservoir_cols].values
    X_context = df[context_cols].values if context_cols else None
    y = compute_bending_angle(df, top_marker, apply_filter=True)

    n_train = int(len(X_reservoir) * Config.TRAIN_FRACTION)

    results = []

    for taps in tap_values:
        print(f"\nTaps = {taps}...")

        X_res_train = X_reservoir[:n_train]
        X_res_test = X_reservoir[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        X_ctx_train = X_context[:n_train] if X_context is not None else None
        X_ctx_test = X_context[n_train:] if X_context is not None else None

        try:
            prc = PhysicalReservoirComputer(taps=taps, ridge_alpha=Config.RIDGE_ALPHA)
            prc.fit(X_res_train, y_train, X_ctx_train)

            test_metrics = prc.evaluate(X_res_test, y_test, X_ctx_test)

            results.append(
                {
                    "taps": taps,
                    "n_features": len(reservoir_cols) * taps
                    + (len(context_cols) if context_cols else 0),
                    "mae": test_metrics["mae"],
                    "rmse": test_metrics["rmse"],
                    "r2": test_metrics["r2"],
                    "error_pct": test_metrics["error_pct"],
                }
            )

            print(
                f"  Error: {test_metrics['error_pct']:.1f}%, R²: {test_metrics['r2']:.3f}"
            )

        except Exception as e:
            print(f"  Failed: {e}")

    return pd.DataFrame(results)


def run_prc_condition_analysis(
    df: pd.DataFrame,
    reservoir_cols: List[str],
    context_cols: Optional[List[str]] = None,
    taps: int = 20,
    top_marker: str = "Rigid_body_2",
) -> Tuple[np.ndarray, List[str]]:
    """
    Run PRC analysis across different input conditions (Wang et al. Figure 3 style).

    Returns:
        error_matrix: (n_train_conditions, n_test_conditions) array of errors
        condition_names: List of condition names
    """
    print("\n" + "=" * 60)
    print("PRC CONDITION ANALYSIS (Wang et al. Figure 3 style)")
    print("=" * 60)

    # Filter columns
    reservoir_cols = [c for c in reservoir_cols if c in df.columns]
    if context_cols:
        context_cols = [c for c in context_cols if c in df.columns]

    # Filter out cooldown periods
    df_active = df[df["config_wave_type"] > 0].copy()

    # Create condition labels
    df_active["condition"] = (
        df_active["config_wave_type"].astype(int).astype(str)
        + "_"
        + df_active["config_seg1_psi"].astype(int).astype(str)
        + "_"
        + df_active["config_max_psi"].astype(int).astype(str)
    )

    conditions = sorted(df_active["condition"].unique())
    n_cond = len(conditions)

    print(f"Found {n_cond} input conditions")
    print(f"Using {taps} taps for reservoir")

    # Prepare data for each condition
    condition_data = {}
    for cond in conditions:
        df_cond = df_active[df_active["condition"] == cond].copy()

        # Take middle 80%
        n = len(df_cond)
        start, end = int(n * 0.1), int(n * 0.9)
        df_cond = df_cond.iloc[start:end].reset_index(drop=True)

        # Split 70/30
        split_idx = int(len(df_cond) * 0.7)

        X_res = df_cond[reservoir_cols].values
        X_ctx = df_cond[context_cols].values if context_cols else None
        y = compute_bending_angle(df_cond, top_marker, apply_filter=True)

        condition_data[cond] = {
            "X_res_train": X_res[:split_idx],
            "X_res_test": X_res[split_idx:],
            "X_ctx_train": X_ctx[:split_idx] if X_ctx is not None else None,
            "X_ctx_test": X_ctx[split_idx:] if X_ctx is not None else None,
            "y_train": y[:split_idx],
            "y_test": y[split_idx:],
            "y_range": np.ptp(y),
        }

    # Build error matrix
    def get_training_subset(n_train, all_conds):
        if n_train >= len(all_conds):
            return all_conds
        indices = np.linspace(0, len(all_conds) - 1, n_train).astype(int)
        return [all_conds[i] for i in indices]

    error_matrix = np.zeros((n_cond, n_cond))

    for row_idx in range(1, n_cond + 1):
        train_conditions = get_training_subset(row_idx, conditions)

        # Compile training data
        X_res_train = np.vstack(
            [condition_data[c]["X_res_train"] for c in train_conditions]
        )
        y_train = np.hstack([condition_data[c]["y_train"] for c in train_conditions])

        if context_cols:
            X_ctx_train = np.vstack(
                [condition_data[c]["X_ctx_train"] for c in train_conditions]
            )
        else:
            X_ctx_train = None

        # Train
        prc = PhysicalReservoirComputer(taps=taps, ridge_alpha=Config.RIDGE_ALPHA)
        prc.fit(X_res_train, y_train, X_ctx_train)

        # Test on all conditions
        for col_idx, cond in enumerate(conditions):
            metrics = prc.evaluate(
                condition_data[cond]["X_res_test"],
                condition_data[cond]["y_test"],
                condition_data[cond]["X_ctx_test"],
            )
            error_matrix[row_idx - 1, col_idx] = metrics["error_pct"]

    # Print summary
    e_avg = error_matrix.mean(axis=1)
    best_row = np.argmin(e_avg)
    print(
        f"\nBest average error: {e_avg[best_row]:.1f}% using {best_row + 1} training conditions"
    )

    return error_matrix, conditions


# =============================================================================
# LIMIT CYCLE IDENTIFICATION (Eder et al. Method) - REFRAMED
# =============================================================================


class PeriodicTrajectoryReadout:
    """
    Periodic Trajectory Readout using Morphological Computation.

    REFRAMED from "LimitCycleController":
    This performs offline trajectory identification/imitation,
    NOT closed-loop limit cycle control.

    Based on Eder et al. (2018): Uses the robot body as a reservoir
    to learn mappings from sensor states to periodic trajectories.

    Key equations:
        w* = L_clean^+ @ y_hat  (Equation 1)
        y(t) = Σ wi * si(t)     (Linear readout)
    """

    def __init__(self, taps: int = 1):
        """
        Initialize trajectory readout.

        Args:
            taps: Number of time delays (1 = no memory, instantaneous only)
        """
        self.taps = taps
        self.w_x = None
        self.w_y = None
        self.scaler_mean = None
        self.scaler_std = None
        self.is_fitted = False

    def fit(
        self,
        S: np.ndarray,
        x_target: np.ndarray,
        y_target: np.ndarray,
        washout: int = 250,
    ) -> "PeriodicTrajectoryReadout":
        """
        Learn readout weights for trajectory mapping.

        Args:
            S: Sensor readings (n_samples, n_sensors)
            x_target: Target x trajectory (n_samples,)
            y_target: Target y trajectory (n_samples,)
            washout: Number of initial samples to discard

        Returns:
            self
        """
        # Apply tapped delay line if taps > 1
        if self.taps > 1:
            S_tapped = build_tapped_delay_line(S, self.taps)
            # Align targets
            x_clean = x_target[washout + self.taps - 1 :]
            y_clean = y_target[washout + self.taps - 1 :]
            S_clean = S_tapped[washout:]
        else:
            S_clean = S[washout:]
            x_clean = x_target[washout:]
            y_clean = y_target[washout:]

        # Standardize
        self.scaler_mean = S_clean.mean(axis=0)
        self.scaler_std = S_clean.std(axis=0) + 1e-8
        S_scaled = (S_clean - self.scaler_mean) / self.scaler_std

        # Add bias term
        S_bias = np.hstack([np.ones((len(S_scaled), 1)), S_scaled])

        # Compute optimal weights via pseudo-inverse
        self.w_x, _, _, _ = np.linalg.lstsq(S_bias, x_clean, rcond=None)
        self.w_y, _, _, _ = np.linalg.lstsq(S_bias, y_clean, rcond=None)

        self.is_fitted = True
        return self

    def predict(self, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict trajectory from sensor readings."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.taps > 1:
            S = build_tapped_delay_line(S, self.taps)

        S_scaled = (S - self.scaler_mean) / self.scaler_std
        S_bias = np.hstack([np.ones((len(S), 1)), S_scaled])

        x_pred = S_bias @ self.w_x
        y_pred = S_bias @ self.w_y

        return x_pred, y_pred

    def evaluate(
        self, S: np.ndarray, x_target: np.ndarray, y_target: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        x_pred, y_pred = self.predict(S)

        # Align targets if using taps
        if self.taps > 1:
            x_target = x_target[self.taps - 1 :]
            y_target = y_target[self.taps - 1 :]

        # Ensure lengths match
        n = min(len(x_pred), len(x_target))
        x_pred, x_target = x_pred[:n], x_target[:n]
        y_pred, y_target = y_pred[:n], y_target[:n]

        rmse_x = np.sqrt(mean_squared_error(x_target, x_pred))
        rmse_y = np.sqrt(mean_squared_error(y_target, y_pred))

        return {
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "x_target": x_target,
            "y_target": y_target,
        }


def identify_limit_cycles(df: pd.DataFrame, wave_type: float = 2.0) -> Dict:
    """
    Identify limit cycles from measured data using Poincaré section analysis.

    This is the correct way to find and verify limit cycles:
    1. Detect cycles from actuation pressure or phase variable
    2. Build Poincaré section on [θ, θ̇]
    3. Demonstrate cycle-to-cycle convergence

    Args:
        df: DataFrame with sensor and mocap data
        wave_type: Wave type to analyze

    Returns:
        Dictionary with limit cycle characteristics
    """
    print("\n" + "=" * 60)
    print(f"LIMIT CYCLE IDENTIFICATION - Wave Type {int(wave_type)}")
    print("=" * 60)

    df_wave = df[df["config_wave_type"] == wave_type].copy().reset_index(drop=True)

    if len(df_wave) < 1000:
        print("Not enough data!")
        return None

    # Get bending angle and its derivative
    theta = compute_bending_angle(df_wave, apply_filter=True)

    # Compute angular velocity (derivative)
    dt = 1.0 / Config.SAMPLE_RATE
    theta_dot = np.gradient(theta, dt)

    # Also filter velocity
    theta_dot = lowpass_filter(
        theta_dot, cutoff=Config.LOWPASS_CUTOFF, fs=Config.SAMPLE_RATE
    )

    # Find zero crossings of theta_dot (Poincaré section at θ̇ = 0)
    zero_crossings = np.where(np.diff(np.sign(theta_dot)) > 0)[0]

    print(f"Found {len(zero_crossings)} Poincaré crossings (θ̇ = 0, increasing)")

    if len(zero_crossings) < 3:
        print("Not enough crossings to identify limit cycle")
        return None

    # Extract θ values at crossings (fixed points on Poincaré map)
    theta_at_crossings = theta[zero_crossings]

    # Check convergence: difference between consecutive crossings
    crossing_diffs = np.diff(theta_at_crossings)

    # Cycle periods (time between crossings)
    periods = np.diff(zero_crossings) * dt

    print(f"\nPoincaré Section Analysis:")
    print(
        f"  Mean θ at crossing: {theta_at_crossings.mean():.3f}° ± {theta_at_crossings.std():.3f}°"
    )
    print(f"  Mean period: {periods.mean():.2f}s ± {periods.std():.2f}s")
    print(f"  Crossing variability: {theta_at_crossings.std():.4f}°")

    # A stable limit cycle should have low variability
    is_stable = theta_at_crossings.std() < 0.5  # Threshold in degrees
    print(f"  Limit cycle stable: {'Yes' if is_stable else 'No (high variability)'}")

    return {
        "theta": theta,
        "theta_dot": theta_dot,
        "zero_crossings": zero_crossings,
        "theta_at_crossings": theta_at_crossings,
        "periods": periods,
        "is_stable": is_stable,
        "mean_period": periods.mean(),
        "period_std": periods.std(),
    }


def run_limit_cycle_analysis(
    df: pd.DataFrame, sensor_cols: List[str], wave_type: float = 2.0, taps: int = 1
) -> Dict:
    """
    Run limit cycle trajectory analysis for a specific wave type.

    Args:
        df: Preprocessed dataframe
        sensor_cols: List of sensor column names
        wave_type: Wave type code (1=axial, 2=circular, 3=triangular)
        taps: Number of time delays for readout

    Returns:
        Dictionary with results and metrics
    """
    wave_name = Config.WAVE_TYPES.get(wave_type, "unknown")

    print(f"\n" + "=" * 60)
    print(f"PERIODIC TRAJECTORY ANALYSIS - {wave_name.upper()}")
    print("=" * 60)

    # Filter for specific wave type
    df_wave = df[df["config_wave_type"] == wave_type].copy().reset_index(drop=True)
    print(f"Samples: {len(df_wave)}")

    if len(df_wave) < 1000:
        print("Not enough data!")
        return None

    # Filter columns that exist
    sensor_cols = [c for c in sensor_cols if c in df_wave.columns]

    # Get sensor readings
    S = df_wave[sensor_cols].values

    # Get actual end-effector position
    x_actual = df_wave[f"{Config.MOCAP_TIP}_x"].values
    z_actual = df_wave[f"{Config.MOCAP_TIP}_z"].values

    # Apply low-pass filter to position
    x_actual = lowpass_filter(
        x_actual, cutoff=Config.LOWPASS_CUTOFF, fs=Config.SAMPLE_RATE
    )
    z_actual = lowpass_filter(
        z_actual, cutoff=Config.LOWPASS_CUTOFF, fs=Config.SAMPLE_RATE
    )

    # Normalize positions
    x_norm = (x_actual - x_actual.mean()) / (x_actual.std() * 2 + 1e-8)
    z_norm = (z_actual - z_actual.mean()) / (z_actual.std() * 2 + 1e-8)

    # Generate target trajectory (for comparison only)
    t = np.arange(len(df_wave)) / Config.SAMPLE_RATE
    period = 10.0  # seconds

    if wave_name == "circular":
        x_target = -np.sin(2 * np.pi * t / period)
        y_target = np.cos(2 * np.pi * t / period)
    elif wave_name == "axial":
        x_target = -np.sin(2 * np.pi * t / period)
        y_target = np.zeros_like(t)
    else:  # triangular/oval
        x_target = -np.sin(2 * np.pi * t / period)
        y_target = 0.5 * np.cos(2 * np.pi * t / period)

    # Train/test split (70/30)
    n_train = int(len(df_wave) * Config.TRAIN_FRACTION)

    # Train trajectory readout
    readout = PeriodicTrajectoryReadout(taps=taps)
    readout.fit(
        S[:n_train], x_target[:n_train], y_target[:n_train], washout=Config.WASHOUT
    )

    # Also train for proprioception (predicting actual position)
    proprio_readout = PeriodicTrajectoryReadout(taps=taps)
    proprio_readout.fit(
        S[:n_train], x_norm[:n_train], z_norm[:n_train], washout=Config.WASHOUT
    )

    # Evaluate on test set
    S_test = S[n_train:]
    target_metrics = readout.evaluate(S_test, x_target[n_train:], y_target[n_train:])
    proprio_metrics = proprio_readout.evaluate(
        S_test, x_norm[n_train:], z_norm[n_train:]
    )

    print(
        f"\nTarget Trajectory RMSE: X={target_metrics['rmse_x']:.4f}, Y={target_metrics['rmse_y']:.4f}"
    )
    print(
        f"Proprioception RMSE: X={proprio_metrics['rmse_x']:.4f}, Z={proprio_metrics['rmse_y']:.4f}"
    )

    # Also run limit cycle identification
    lc_results = identify_limit_cycles(df, wave_type)

    return {
        "wave_type": wave_name,
        "readout": readout,
        "proprio_readout": proprio_readout,
        "target_metrics": target_metrics,
        "proprio_metrics": proprio_metrics,
        "x_target": x_target,
        "y_target": y_target,
        "x_actual": x_norm,
        "z_actual": z_norm,
        "n_train": n_train,
        "limit_cycle": lc_results,
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_prc_results(results: Dict, save_path: str = None):
    """Plot PRC analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    y_train = results["train_metrics"]["y_actual"]
    y_test = results["test_metrics"]["y_actual"]
    y_pred_train = results["train_metrics"]["y_pred"]
    y_pred_test = results["test_metrics"]["y_pred"]

    # Time series - Training
    ax = axes[0, 0]
    t = np.arange(len(y_train)) / Config.SAMPLE_RATE
    step = max(1, len(t) // 5000)
    ax.plot(t[::step], y_train[::step], "b-", alpha=0.7, label="Ground Truth")
    ax.plot(t[::step], y_pred_train[::step], "r-", alpha=0.7, label="PRC Prediction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bending Angle (°)")
    ax.set_title(
        f"Training (R²={results['train_metrics']['r2']:.3f}, Taps={results['taps']})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time series - Testing
    ax = axes[0, 1]
    t = np.arange(len(y_test)) / Config.SAMPLE_RATE
    step = max(1, len(t) // 5000)
    ax.plot(t[::step], y_test[::step], "b-", alpha=0.7, label="Ground Truth")
    ax.plot(t[::step], y_pred_test[::step], "r-", alpha=0.7, label="PRC Prediction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bending Angle (°)")
    ax.set_title(
        f"Testing (R²={results['test_metrics']['r2']:.3f}, Error={results['test_metrics']['error_pct']:.1f}%)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter - Training
    ax = axes[1, 0]
    ax.scatter(y_train[::step], y_pred_train[::step], alpha=0.3, s=10)
    lims = [
        min(y_train.min(), y_pred_train.min()),
        max(y_train.max(), y_pred_train.max()),
    ]
    ax.plot(lims, lims, "r--", label="Perfect")
    ax.set_xlabel("Actual (°)")
    ax.set_ylabel("Predicted (°)")
    ax.set_title("Training: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter - Testing
    ax = axes[1, 1]
    ax.scatter(y_test[::step], y_pred_test[::step], alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", label="Perfect")
    ax.set_xlabel("Actual (°)")
    ax.set_ylabel("Predicted (°)")
    ax.set_title("Testing: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Physical Reservoir Computing - Bending Angle Prediction",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return fig


def plot_taps_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Plot comparison of different tap values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Error vs Taps
    ax = axes[0]
    ax.plot(results_df["taps"], results_df["error_pct"], "bo-", markersize=8)
    ax.set_xlabel("Number of Taps", fontsize=12)
    ax.set_ylabel("Error (%)", fontsize=12)
    ax.set_title("Prediction Error vs Taps")
    ax.grid(True, alpha=0.3)

    # R² vs Taps
    ax = axes[1]
    ax.plot(results_df["taps"], results_df["r2"], "go-", markersize=8)
    ax.set_xlabel("Number of Taps", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("R² Score vs Taps")
    ax.grid(True, alpha=0.3)

    # Mark best
    best_idx = results_df["error_pct"].idxmin()
    best_taps = results_df.loc[best_idx, "taps"]
    best_error = results_df.loc[best_idx, "error_pct"]
    axes[0].annotate(
        f"Best: {best_taps} taps\n{best_error:.1f}%",
        xy=(best_taps, best_error),
        xytext=(best_taps + 3, best_error + 1),
        fontsize=10,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    plt.suptitle("PRC Performance vs Number of Taps", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig


def plot_prc_error_table(
    error_matrix: np.ndarray, conditions: List[str], save_path: str = None
):
    """Plot PRC error matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Add e_avg column
    e_avg = error_matrix.mean(axis=1)
    matrix_display = np.hstack([error_matrix, e_avg.reshape(-1, 1)])

    # Create short condition names
    wave_map = {"1": "Ax", "2": "Ci", "3": "Tr"}
    short_names = []
    for c in conditions:
        parts = c.split("_")
        short_names.append(f"{wave_map.get(parts[0], parts[0])}{parts[1]}/{parts[2]}")

    col_labels = short_names + ["$e_{avg}$"]
    row_labels = [str(i + 1) for i in range(len(conditions))]

    # Heatmap
    sns.heatmap(
        matrix_display,
        annot=True,
        fmt=".1f",
        cmap="RdYlBu_r",
        ax=ax,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 9},
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Prediction Error (%)"},
    )

    # Highlight best in each row (orange triangle)
    for row_idx in range(len(conditions)):
        min_col = np.argmin(error_matrix[row_idx])
        triangle = plt.Polygon(
            [
                [min_col + 0.7, row_idx],
                [min_col + 1, row_idx],
                [min_col + 1, row_idx + 0.3],
            ],
            color="orange",
            transform=ax.transData,
        )
        ax.add_patch(triangle)

    # Highlight GLOBAL best error in entire matrix (green star)
    global_min_val = error_matrix.min()
    global_min_pos = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
    global_min_row, global_min_col = global_min_pos

    # Draw green star at the top-left corner of the cell (not center)
    ax.plot(
        global_min_col + 0.15,  # Top-left corner
        global_min_row + 0.15,  # Top-left corner
        marker="*",
        markersize=15,
        color="lime",
        markeredgecolor="darkgreen",
        markeredgewidth=1.5,
        zorder=10,
    )

    ax.set_xlabel("Test Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Training Conditions", fontsize=12, fontweight="bold")
    ax.set_title(
        "PRC Prediction Error (%) with Tapped Delay Line\n"
        f"★ Global Best: {global_min_val:.1f}% (Row {global_min_row + 1}, {short_names[global_min_col]})",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        plt.Polygon([[0, 0], [1, 0], [1, 0.5]], color="orange", label="Best in row"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="lime",
            markeredgecolor="darkgreen",
            markersize=15,
            label="Global best error",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()

    return fig


def plot_limit_cycle_results(results: Dict, save_path: str = None):
    """Plot limit cycle analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    n_train = results["n_train"]
    target_metrics = results["target_metrics"]
    proprio_metrics = results["proprio_metrics"]

    # Row 1: Target trajectory analysis
    # Time series
    ax = axes[0, 0]
    n_plot = min(500, len(target_metrics["x_pred"]))
    t = np.arange(n_plot) / Config.SAMPLE_RATE
    ax.plot(t, target_metrics["x_target"][:n_plot], "k--", label="Target X", alpha=0.7)
    ax.plot(t, target_metrics["x_pred"][:n_plot], "r-", label="Output X", alpha=0.7)
    ax.plot(t, target_metrics["y_target"][:n_plot], "b--", label="Target Y", alpha=0.7)
    ax.plot(t, target_metrics["y_pred"][:n_plot], "g-", label="Output Y", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trajectory")
    ax.set_title(f"{results['wave_type'].capitalize()} - Time Series")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2D trajectory
    ax = axes[0, 1]
    ax.plot(
        target_metrics["x_target"],
        target_metrics["y_target"],
        "k--",
        lw=2,
        label="Target",
    )
    ax.plot(
        target_metrics["x_pred"],
        target_metrics["y_pred"],
        "r-",
        lw=1,
        alpha=0.7,
        label="Output",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"Target (RMSE: X={target_metrics['rmse_x']:.3f}, Y={target_metrics['rmse_y']:.3f})"
    )
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Proprioception 2D
    ax = axes[0, 2]
    ax.plot(
        proprio_metrics["x_target"],
        proprio_metrics["y_target"],
        "k--",
        lw=2,
        label="Actual",
    )
    ax.plot(
        proprio_metrics["x_pred"],
        proprio_metrics["y_pred"],
        "g-",
        lw=1,
        alpha=0.7,
        label="Predicted",
    )
    ax.set_xlabel("X (norm)")
    ax.set_ylabel("Z (norm)")
    ax.set_title(
        f"Proprioception (RMSE: X={proprio_metrics['rmse_x']:.3f}, Z={proprio_metrics['rmse_y']:.3f})"
    )
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Row 2: Limit cycle identification
    lc = results.get("limit_cycle")

    if lc is not None:
        # Phase portrait
        ax = axes[1, 0]
        ax.plot(lc["theta"], lc["theta_dot"], "b-", alpha=0.5, lw=0.5)
        ax.scatter(
            lc["theta"][lc["zero_crossings"]],
            lc["theta_dot"][lc["zero_crossings"]],
            c="red",
            s=20,
            zorder=5,
            label="Poincaré crossings",
        )
        ax.set_xlabel("θ (°)")
        ax.set_ylabel("θ̇ (°/s)")
        ax.set_title("Phase Portrait")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Poincaré map
        ax = axes[1, 1]
        theta_n = lc["theta_at_crossings"][:-1]
        theta_n1 = lc["theta_at_crossings"][1:]
        ax.scatter(theta_n, theta_n1, c="blue", s=20, alpha=0.7)
        lims = [min(theta_n.min(), theta_n1.min()), max(theta_n.max(), theta_n1.max())]
        ax.plot(lims, lims, "r--", label="θ(n+1) = θ(n)")
        ax.set_xlabel("θ(n) at crossing")
        ax.set_ylabel("θ(n+1) at crossing")
        ax.set_title(f"Poincaré Map (σ={lc['theta_at_crossings'].std():.3f}°)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Period histogram
        ax = axes[1, 2]
        ax.hist(lc["periods"], bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(
            lc["mean_period"],
            color="r",
            linestyle="--",
            label=f"Mean: {lc['mean_period']:.2f}s",
        )
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"Cycle Periods (σ={lc['period_std']:.3f}s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes[1, :]:
            ax.text(
                0.5,
                0.5,
                "Limit cycle analysis\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    plt.suptitle(
        f"Limit Cycle Analysis - {results['wave_type'].capitalize()} Motion\n(Eder et al. Style)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="PRC and Limit Cycle Analysis v2")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["prc", "limit_cycle", "both"],
        help="Analysis mode",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--taps", type=int, default=20, help="Number of taps for reservoir"
    )
    parser.add_argument(
        "--top_marker",
        type=str,
        default="Rigid_body_2",
        help="Top marker for bending angle calculation",
    )

    args = parser.parse_args()

    # Load and preprocess data
    df = load_data(args.data)
    df = preprocess_data(df, args.top_marker)

    # Define sensor columns
    reservoir_cols = Config.SEGMENT_1_COLS
    context_cols = Config.ACTUATION_COLS
    all_sensor_cols = reservoir_cols + context_cols

    if args.mode in ["prc", "both"]:
        # Run PRC analysis with tapped reservoir
        print("\n" + "=" * 70)
        print("RUNNING PRC WITH TAPPED DELAY LINE")
        print("=" * 70)

        prc_results = run_prc_analysis(
            df, reservoir_cols, context_cols, taps=args.taps, top_marker=args.top_marker
        )
        plot_prc_results(prc_results)

        # Run taps comparison
        taps_results = run_prc_taps_comparison(
            df,
            reservoir_cols,
            context_cols,
            tap_values=[0, 1, 5, 10, 15, 20, 25, 30],
            top_marker=args.top_marker,
        )
        plot_taps_comparison(taps_results)

        # Run condition analysis
        if "config_wave_type" in df.columns:
            error_matrix, conditions = run_prc_condition_analysis(
                df,
                reservoir_cols,
                context_cols,
                taps=args.taps,
                top_marker=args.top_marker,
            )
            plot_prc_error_table(error_matrix, conditions)

    if args.mode in ["limit_cycle", "both"]:
        # Run limit cycle analysis for each wave type
        print("\n" + "=" * 70)
        print("RUNNING LIMIT CYCLE IDENTIFICATION")
        print("=" * 70)

        for wave_type in [1.0, 2.0, 3.0]:
            if "config_wave_type" in df.columns:
                results = run_limit_cycle_analysis(
                    df, all_sensor_cols, wave_type, taps=1
                )
                if results:
                    plot_limit_cycle_results(results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
