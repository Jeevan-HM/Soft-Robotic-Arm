import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# CSV_PATH = "Test_1.csv"  # old dataset
CSV_PATH = (
    "experiments/October-25/cleaned_data/circular_motion_5_psi_peak.csv"  # new dataset
)
# CSV_PATH = "experiments/October-25/cleaned_data/axial_motion.csv"  # new dataset
# CSV_PATH = "experiments/October-25/cleaned_data/Experiment_14.csv"  # Circular

# 1) Load with header and extract relevant columns
df = pd.read_csv(CSV_PATH)

# Rename columns for easier access
# Desired pressures: segments 1-4
# Measured pressures: Segment 1 (5 pouches), Segment 2 (5 pouches), Segments 3 & 4 (1 each)
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

# Select time, desired pressures (targets), and measured pressures (features)
df_working = df[["time"] + desired_cols + measured_cols].copy()

# Simplify column names
df_working.columns = ["time", "y1", "y2", "y3", "y4"] + [f"s{i}" for i in range(1, 13)]

# 2) Coerce numeric, drop bad rows, sort by time
for c in df_working.columns:
    df_working[c] = pd.to_numeric(df_working[c], errors="coerce")
df_working = df_working.dropna().sort_values("time").reset_index(drop=True)

# 3) Trim first/last 10 s by relative time
t0 = df_working["time"].iloc[0]
df_working["t_rel"] = df_working["time"] - t0
tmax = df_working["t_rel"].iloc[-1]

if tmax <= 20:
    print(f"Warning: Duration {tmax:.3f}s is too short to drop 10s at start/end.")
    print("Using 10% trim from start and end instead.")
    trim_duration = tmax * 0.1
    df_working = df_working[
        (df_working["t_rel"] >= trim_duration)
        & (df_working["t_rel"] <= (tmax - trim_duration))
    ].reset_index(drop=True)
else:
    df_working = df_working[
        (df_working["t_rel"] >= 10.0) & (df_working["t_rel"] <= (tmax - 10.0))
    ].reset_index(drop=True)

if len(df_working) < 4:
    raise ValueError("Not enough samples after trimming.")

# 4) Split 50/50 by time order
mid = len(df_working) // 2
train = df_working.iloc[:mid].copy()
test = df_working.iloc[mid:].copy()
if len(train) == 0 or len(test) == 0:
    raise ValueError("Train/test split empty — check input length.")

# 5) Prepare features/targets (now with 4 targets: y1, y2, y3, y4)
start_sensor = 1  # Use all sensors s1-s12
X_train = train[[f"s{i}" for i in range(start_sensor, 13)]].values
Y_train = train[["y1", "y2", "y3", "y4"]].values
X_test = test[[f"s{i}" for i in range(start_sensor, 13)]].values
Y_test = test[["y1", "y2", "y3", "y4"]].values

# 6) Fit linear readout and predict
model = LinearRegression()
model.fit(X_train, Y_train)
Yhat_test = model.predict(X_test)

# 7) Quick metrics
r2_each = r2_score(Y_test, Yhat_test, multioutput="raw_values")
mae_each = mean_absolute_error(Y_test, Yhat_test, multioutput="raw_values")
print("R2 per target  [y1, y2, y3, y4]:", np.round(r2_each, 4))
print("MAE per target [y1, y2, y3, y4]:", np.round(mae_each, 4))

# 8) Plot test emulation (4 panels for 4 segments)
t_plot = test["time"].values - test["time"].iloc[0]
titles = [
    "Segment 1 emulation (test)",
    "Segment 2 emulation (test)",
    "Segment 3 emulation (test)",
    "Segment 4 emulation (test)",
]

# Bold, distinct colors for true vs predicted
true_color = "#0066CC"  # Bold blue
pred_color = "#FF3333"  # Bold red

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for i in range(4):
    axes[i].plot(
        t_plot,
        Y_test[:, i],
        label=f"Seg{i + 1} true",
        linewidth=2.5,
        color=true_color,
        alpha=0.9,
    )
    axes[i].plot(
        t_plot,
        Yhat_test[:, i],
        "--",
        label=f"Seg{i + 1} pred",
        linewidth=2.5,
        color=pred_color,
        alpha=0.9,
    )
    axes[i].set_ylabel("Pressure", fontsize=16, fontweight="bold")
    axes[i].set_title(titles[i], fontsize=12, fontweight="bold")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(loc="upper right", frameon=True, fontsize=26)

axes[-1].set_xlabel("time (s)", fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()

# 9) Plot sensor time-series (test split) in 3 subplots:
#    (1) s1–s5 (Segment 1 pouches), (2) s6–s10 (Segment 2 pouches), (3) s11–s12 (Segments 3&4)
groups = [
    list(range(1, 6)),  # s1-s5: Segment 1 pouches
    list(range(6, 11)),  # s6-s10: Segment 2 pouches
    list(range(11, 13)),  # s11-s12: Segments 4 and 3
]

# Distinct color palettes for each group
color_palettes = [
    [
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
    ],  # Segment 1: Bold primary colors
    ["#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62"],  # Segment 2: Warm tones
    ["#8DD3C7", "#FFED6F"],  # Segments 3&4: Bright contrast colors
]

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
group_titles = [
    "Segment 1 Pouches (s1–s5)",
    "Segment 2 Pouches (s6–s10)",
    "Segments 3 & 4 (s11–s12)",
]

for ax, g, title, colors in zip(axes2, groups, group_titles, color_palettes):
    for k, color in zip(g, colors):
        ax.plot(
            t_plot,
            test[f"s{k}"].values,
            label=f"s{k}",
            linewidth=2.2,
            color=color,
            alpha=0.9,
        )
    ax.set_ylabel("Pressure", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=len(g), frameon=True, fontsize=9)

axes2[-1].set_xlabel("time (s)", fontsize=11, fontweight="bold")
fig2.tight_layout()
plt.show()
