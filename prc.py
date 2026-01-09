import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# CSV_PATH = "Test_1.csv"  # old dataset
CSV_PATH = (
    # "files_for_submission/cleaned_data/experiment-constant-2psi.csv"  # new dataset
    "files_for_submission/cleaned_data/experiment-refil-end-of-experiment.csv"  # new dataset
    # "files_for_submission/cleaned_data/experiment-refil-100s.csv"  # new dataset
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
    "mocap_rigid_body_x",
    "mocap_rigid_body_y",
    "mocap_rigid_body_z",
]
measured_cols = (
    [f"Measured_pressure_Segment_1_pouch_{i}" for i in range(1, 6)]
    + [f"Measured_pressure_Segment_2_pouch_{i}" for i in range(1, 6)]
    + ["Measured_pressure_Segment_4", "Measured_pressure_Segment_3"]
)

# Select time, desired pressures (targets), and measured pressures (features)
df_working = df[["time"] + desired_cols + measured_cols].copy()

# Simplify column names
df_working.columns = ["time", "y1", "y2", "y3", "y4", "x", "y", "z"] + [
    f"s{i}" for i in range(1, 13)
]

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

# 4) Split 80/20 by time order
mid = int(len(df_working) * 0.8)
train = df_working.iloc[:mid].copy()
test = df_working.iloc[mid:].copy()
if len(train) == 0 or len(test) == 0:
    raise ValueError("Train/test split empty — check input length.")

# 5) Prepare features/targets (Pressures + X, Y, Z coordinates)
start_sensor = 1  # Use all sensors s1-s12
X_train = train[[f"s{i}" for i in range(start_sensor, 13)]].values
# Targets: 4 pressures + 3 trajectory coordinates = 7 targets
Y_train = train[["y1", "y2", "y3", "y4", "x", "y", "z"]].values
X_test = test[[f"s{i}" for i in range(start_sensor, 13)]].values
Y_test = test[["y1", "y2", "y3", "y4", "x", "y", "z"]].values

# 6) Fit linear readout and predict
model = LinearRegression()
model.fit(X_train, Y_train)
Yhat_test = model.predict(X_test)

# 7) Quick metrics
r2_each = r2_score(Y_test, Yhat_test, multioutput="raw_values")
mae_each = mean_absolute_error(Y_test, Yhat_test, multioutput="raw_values")
print("R2 [y1..y4, x,y,z]:", np.round(r2_each, 4))
print("MAE [y1..y4, x,y,z]:", np.round(mae_each, 4))

# 8) Split Plotting into Two Figures

t_plot = test["time"].values - test["time"].iloc[0]

# Bold, distinct colors for true vs predicted
true_color = "#0066CC"  # Bold blue
pred_color = "#FF3333"  # Bold red

# --- FIGURE 1: Pressures and Sensors ---
fig1 = plt.figure(figsize=(14, 12))
gs1 = fig1.add_gridspec(4, 2)

# Col 0: Pressure Predictions (Seg 1-4)
pressure_titles = [
    "Seg 1 Pressure",
    "Seg 2 Pressure",
    "Seg 3 Pressure",
    "Seg 4 Pressure",
]
ax_p = []
for i in range(4):
    ax = fig1.add_subplot(gs1[i, 0])
    ax.plot(
        t_plot,
        Y_test[:, i],
        label="True",
        linewidth=2.0,
        color=true_color,
        alpha=0.9,
    )
    ax.plot(
        t_plot,
        Yhat_test[:, i],
        "--",
        label="Pred",
        linewidth=2.0,
        color=pred_color,
        alpha=0.9,
    )
    ax.set_ylabel("Pressure (PSI)", fontweight="bold")
    ax.set_title(pressure_titles[i], fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)
    if i < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("time (s)", fontweight="bold")
    ax_p.append(ax)

# Link x-axes for pressure
for ax in ax_p[1:]:
    ax.sharex(ax_p[0])

# Col 1: Sensor Inputs (Rows 0-2)
groups = [
    list(range(1, 6)),  # s1-s5
    list(range(6, 11)),  # s6-s10
    list(range(11, 13)),  # s11-s12
]
# Distinct color palettes for sensor lines
color_palettes = [
    ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"],
    ["#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62"],
    ["#8DD3C7", "#FFED6F"],
]
group_titles = [
    "Seg 1 Sensors (s1–s5)",
    "Seg 2 Sensors (s6–s10)",
    "Seg 3/4 Sensors (s11–s12)",
]

ax_s = []
for i in range(3):
    ax = fig1.add_subplot(gs1[i, 1])
    g = groups[i]
    colors = color_palettes[i]
    for k, color in zip(g, colors):
        ax.plot(
            t_plot,
            test[f"s{k}"].values,
            label=f"s{k}",
            linewidth=1.5,
            color=color,
            alpha=0.8,
        )
    ax.set_ylabel("Input (PSI)", fontweight="bold")
    ax.set_title(group_titles[i], fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3, fontsize=6)
    if i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("time (s)", fontweight="bold")
    ax_s.append(ax)

# Link x-axes for sensors
for ax in ax_s:
    ax.sharex(ax_p[0])

# Col 1, Row 3: Metrics Text Box
ax_text = fig1.add_subplot(gs1[3, 1])
ax_text.axis("off")
metrics_text = (
    "Model Results (Test Split)\n"
    "--------------------------\n"
    f"Samples: {len(test)}\n\n"
    "Pressure R2:\n"
    f"  Seg1: {r2_each[0]:.4f}\n"
    f"  Seg2: {r2_each[1]:.4f}\n"
    f"  Seg3: {r2_each[2]:.4f}\n"
    f"  Seg4: {r2_each[3]:.4f}\n\n"
    "Trajectory R2:\n"
    f"  X:    {r2_each[4]:.4f}\n"
    f"  Y:    {r2_each[5]:.4f}\n"
    f"  Z:    {r2_each[6]:.4f}\n"
)
ax_text.text(
    0.05,
    0.95,
    metrics_text,
    transform=ax_text.transAxes,
    fontsize=16,
    verticalalignment="top",
    fontfamily="monospace",
)

plt.tight_layout()
plt.show()


# --- FIGURE 2: Trajectories (X, Y, Z, 3D) ---
fig2 = plt.figure(figsize=(12, 10))
gs2 = fig2.add_gridspec(2, 2)
# (0,0): X, (0,1): Y
# (1,0): Z, (1,1): 3D

coord_names = ["X", "Y", "Z"]
positions = [(0, 0), (0, 1), (1, 0)]  # Grid positions for X, Y, Z

ax_t = []
for i in range(3):
    idx = 4 + i
    r, c = positions[i]
    ax = fig2.add_subplot(gs2[r, c])
    ax.plot(
        t_plot,
        Y_test[:, idx],
        label="True",
        linewidth=2.0,
        color=true_color,
        alpha=0.9,
    )
    ax.plot(
        t_plot,
        Yhat_test[:, idx],
        "--",
        label="Pred",
        linewidth=2.0,
        color=pred_color,
        alpha=0.9,
    )
    ax.set_ylabel(f"Pos {coord_names[i]} (m)", fontweight="bold")
    ax.set_title(f"Trajectory {coord_names[i]}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("time (s)", fontweight="bold")
    ax_t.append(ax)

# Link x-axes for 1D trajectories
for ax in ax_t[1:]:
    ax.sharex(ax_t[0])

# Subplot (1,1): 3D Trajectory
ax3d = fig2.add_subplot(gs2[1, 1], projection="3d")
ax3d.plot(
    Y_test[:, 4],
    Y_test[:, 5],
    Y_test[:, 6],
    label="True",
    color=true_color,
    linewidth=2,
    alpha=0.8,
)
ax3d.plot(
    Yhat_test[:, 4],
    Yhat_test[:, 5],
    Yhat_test[:, 6],
    label="Pred",
    color=pred_color,
    linestyle="--",
    linewidth=2,
    alpha=0.8,
)
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_title("3D Trajectory", fontsize=12, fontweight="bold")
ax3d.legend(fontsize=8)

plt.tight_layout()
plt.show()
