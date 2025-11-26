import pandas as pd
import numpy as np

# Load the CSV
csv_path = "/home/g1/Developer/Thesis/exp_105_axial_Nov25_13h11m.csv"
df = pd.read_csv(csv_path)

# Columns of interest
# Segment 4 measured: pm_8_4
# Segment 4 desired: pd_8
# Segment 3 measured: pm_8_3 (to see if it correlates with drops)

print(f"Total rows: {len(df)}")

# Check Desired Pressure
pd_8_mean = df['pd_8'].mean()
pd_8_std = df['pd_8'].std()
print(f"\nDesired Pressure (pd_8):")
print(f"  Mean: {pd_8_mean:.4f}")
print(f"  Std Dev: {pd_8_std:.4f}")
print(f"  Min: {df['pd_8'].min():.4f}")
print(f"  Max: {df['pd_8'].max():.4f}")

# Check Measured Pressure
pm_8_4 = df['pm_8_4']
print(f"\nMeasured Pressure (pm_8_4):")
print(f"  Mean: {pm_8_4.mean():.4f}")
print(f"  Min: {pm_8_4.min():.4f}")
print(f"  Max: {pm_8_4.max():.4f}")

# Check for exact zeros (Communication loss)
zeros = df[df['pm_8_4'] == 0.0]
print(f"\nExact Zeros in pm_8_4: {len(zeros)}")
if len(zeros) > 0:
    print("  Timestamps of zeros:")
    print(zeros['time'].head().values)

# Check for significant drops (Physical sag)
# Assuming target is around 2.0, let's look for values < 1.8 that are NOT 0.0
drops = df[(df['pm_8_4'] < 1.8) & (df['pm_8_4'] > 0.001)]
print(f"\nSignificant Drops (< 1.8 but > 0.001) in pm_8_4: {len(drops)}")
if len(drops) > 0:
    print(f"  Min value in drops: {drops['pm_8_4'].min():.4f}")
    print("  Timestamps of drops:")
    print(drops['time'].head().values)

# Correlation with Segment 3?
# Segment 3 (pm_8_3) is oscillating. Let's see if drops in 4 correlate with peaks in 3.
if len(drops) > 0:
    print("\nCorrelation check:")
    print("  Avg pm_8_3 during drops in pm_8_4:", df.loc[drops.index, 'pm_8_3'].mean())
    print("  Overall Avg pm_8_3:", df['pm_8_3'].mean())
