import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Configuration
CSV_PATH = "files_for_submission/cleaned_data/experiment-refil-end-of-experiment.csv"
LAG_STEPS = [1, 5, 10, 20, 50]  # Lag steps to include as features
ALPHA = 1.0  # Ridge regularization strength

# 2. Load Data
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Rename/Select columns (Adapting from prc.py logic)
# Check columns first
# Assuming same structure as prc.py since it's the same project
desired_cols = [
    "Desired_pressure_segment_1",
    "Desired_pressure_segment_2",
    "Desired_pressure_segment_3",
    "Desired_pressure_segment_4",
]
measured_cols = (
    [f"Measured_pressure_Segment_1_pouch_{i}" for i in range(1, 6)]
    + [f"Measured_pressure_Segment_2_pouch_{i}" for i in range(1, 6)]
    + ["Measured_pressure_Segment_4", "Measured_pressure_Segment_3"]
)

# Identify available measured columns in this specific CSV
available_measured = [c for c in measured_cols if c in df.columns]
if len(available_measured) < 12:
    print("Warning: Some expected measured columns are missing. Using available ones.")
    print(f"Found: {len(available_measured)}")

df_working = df[["time"] + desired_cols + available_measured].copy()

# Rename for simplicity
# s1..sN
sensor_names = [f"s{i + 1}" for i in range(len(available_measured))]
rename_map = {old: new for old, new in zip(available_measured, sensor_names)}
rename_map.update({old: f"y{i + 1}" for i, old in enumerate(desired_cols)})
df_working = df_working.rename(columns=rename_map)

# Coerce and Drop NaNs
for c in df_working.columns:
    df_working[c] = pd.to_numeric(df_working[c], errors="coerce")

df_working = df_working.dropna().sort_values("time").reset_index(drop=True)

# 3. Create Lagged Features (The "Compensation" part)
print("Creating lagged features...")
feature_cols = sensor_names.copy()
# Create lags
for lag in LAG_STEPS:
    for s in sensor_names:
        col_name = f"{s}_lag{lag}"
        df_working[col_name] = df_working[s].shift(lag)
        feature_cols.append(col_name)

# Add Time as a feature to compensate for drift/leakage
df_working["time_norm"] = df_working["time"] - df_working["time"].iloc[0]
feature_cols.append("time_norm")

# Drop rows with NaNs created by shifting
df_working = df_working.dropna().reset_index(drop=True)

print(f"Total samples: {len(df_working)}")
print(f"Features: {len(feature_cols)}")

# 4. Train/Test Split (Sequential)
# Using 80/20 split
split_idx = int(len(df_working) * 0.8)
train = df_working.iloc[:split_idx].copy()
test = df_working.iloc[split_idx:].copy()

# Exclude y1 (Segment 1) from targets as it is a sensing column (User Input)
# USER REQUESTED RE-INCLUSION OF SEGMENT 1
target_cols = ["y1", "y2", "y3", "y4"]

X_train = train[feature_cols].values
Y_train = train[target_cols].values
X_test = test[feature_cols].values
Y_test = test[target_cols].values

# 5. Model Training (Ridge)
print("Training Ridge Regression model...")
model = Ridge(alpha=ALPHA)
model.fit(X_train, Y_train)

# 6. Prediction
print("Predicting on test set...")
Yhat_test = model.predict(X_test)

# 7. Evaluation
r2 = r2_score(Y_test, Yhat_test, multioutput="raw_values")
mae = mean_absolute_error(Y_test, Yhat_test, multioutput="raw_values")

print("-" * 30)
print("Model Performance (Leakage Compensated + Time Drift)")
print("-" * 30)
print(f"R2 Scores (y1, y2, y3, y4): {np.round(r2, 4)}")
print(f"MAE       (y1, y2, y3, y4): {np.round(mae, 4)}")
print(f"Average R2: {np.mean(r2):.4f}")
print("-" * 30)

# 8. Visualization
t_plot = test["time"].values
# Normalize time for plotting
t_plot = t_plot - t_plot[0]

num_targets = Y_test.shape[1]
fig, axes = plt.subplots(num_targets, 1, figsize=(12, 3 * num_targets), sharex=True)

# Map index to segment name
segment_map = ["Segment 1", "Segment 2", "Segment 3", "Segment 4"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

if num_targets == 1:
    axes = [axes]

for i in range(num_targets):
    ax = axes[i]
    ax.plot(
        t_plot,
        Y_test[:, i],
        label=f"True {segment_map[i]}",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        t_plot,
        Yhat_test[:, i],
        label=f"Predicted {segment_map[i]}",
        color=colors[i],
        linewidth=1.5,
        alpha=0.9,
    )
    ax.set_ylabel("Pressure (psi)")
    ax.set_title(segment_map[i])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("leakage_compensation_results.png")
print("Results plot saved to leakage_compensation_results.png")
