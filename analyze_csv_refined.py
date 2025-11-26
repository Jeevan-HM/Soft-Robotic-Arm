import pandas as pd
import numpy as np

csv_path = "/home/g1/Developer/Thesis/exp_105_axial_Nov25_13h11m.csv"
df = pd.read_csv(csv_path)

# Focus on the window around the drop
window = df[(df['time'] > 180) & (df['time'] < 240)]

print(f"Data in window 180s - 240s: {len(window)} rows")

# Check Desired Pressure in this window
print(f"\nDesired Pressure (pd_8) in window:")
print(f"  Mean: {window['pd_8'].mean():.4f}")
print(f"  Min: {window['pd_8'].min():.4f}")
print(f"  Max: {window['pd_8'].max():.4f}")

# Check Measured Pressure stats before and after t=210
before = window[window['time'] < 210]['pm_8_4']
after = window[window['time'] >= 210]['pm_8_4']

print(f"\nMeasured Pressure (pm_8_4):")
print(f"  Mean Before t=210: {before.mean():.4f}")
print(f"  Mean After t=210:  {after.mean():.4f}")
print(f"  Drop magnitude: {before.mean() - after.mean():.4f} PSI")

# Check oscillation correlation
# We suspect pm_8_4 oscillates with pm_7_1 (Arduino 7)
# Let's check correlation coefficient
corr = window['pm_8_4'].corr(window['pm_7_1'])
print(f"\nCorrelation between pm_8_4 (Seg 4) and pm_7_1 (Seg 3/Ard7): {corr:.4f}")

# Check if pm_7_1 changed behavior at t=210
before_7 = window[window['time'] < 210]['pm_7_1']
after_7 = window[window['time'] >= 210]['pm_7_1']
print(f"\nArduino 7 (pm_7_1) stats:")
print(f"  Mean Before t=210: {before_7.mean():.4f}")
print(f"  Mean After t=210:  {after_7.mean():.4f}")
