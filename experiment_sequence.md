# Experiment Sequence Documentation

This document details the automated sequence of experiments configured in `main.py`.

## Overview

The system will execute a series of experiments by iterating through all combinations of the following parameters:

*   **Segment 1 Base Pressure**: `[2.0, 3.0]` PSI
*   **Max Actuation Pressure**: `[5.0, 10.0]` PSI
*   **Wave Type**: `["axial", "circular", "triangular"]`

### Timing
*   **Wave Duration**: 200.0 seconds per experiment
*   **Cooldown Duration**: 5.0 seconds (between experiments)
*   **Total Experiments**: 12
*   **Total Estimated Duration**: ~41 minutes (2460 seconds)

## Sequence Table

The experiments will run in the following order:

| ID | Seg 1 (PSI) | Max Pressure (PSI) | Wave Type | Duration (s) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 2.0 | 5.0 | Axial | 200 |
| 2 | 2.0 | 5.0 | Circular | 200 |
| 3 | 2.0 | 5.0 | Triangular | 200 |
| 4 | 2.0 | 10.0 | Axial | 200 |
| 5 | 2.0 | 10.0 | Circular | 200 |
| 6 | 2.0 | 10.0 | Triangular | 200 |
| 7 | 3.0 | 5.0 | Axial | 200 |
| 8 | 3.0 | 5.0 | Circular | 200 |
| 9 | 3.0 | 5.0 | Triangular | 200 |
| 10 | 3.0 | 10.0 | Axial | 200 |
| 11 | 3.0 | 10.0 | Circular | 200 |
| 12 | 3.0 | 10.0 | Triangular | 200 |

*Note: A 5-second cooldown (all segments at 2.0 PSI) occurs after each experiment.*

## Data Logging

The HDF5 data file will include the following metadata columns for each sample to identify the active experiment:

*   `config_wave_type`: Enum (0=Static, 1=Axial, 2=Circular, 3=Triangular)
*   `config_seg1_psi`: The target pressure for Segment 1
*   `config_max_psi`: The maximum pressure for the active wave
