#!/usr/bin/env python3
"""
Simple experiment manager for HDF5 files
"""

import os

import h5py

EXPERIMENTS_DIR = "experiments"


def list_experiments():
    """List all experiments in all HDF5 files"""
    h5_files = sorted(
        [f for f in os.listdir(EXPERIMENTS_DIR) if f.endswith(".h5")], reverse=True
    )

    if not h5_files:
        print("No HDF5 files found!")
        return []

    all_experiments = []

    for h5_file in h5_files:
        filepath = os.path.join(EXPERIMENTS_DIR, h5_file)
        print(f"\n📁 {h5_file}")

        with h5py.File(filepath, "r") as f:
            experiments = sorted([k for k in f.keys() if k.startswith("exp_")])

            for exp_name in experiments:
                exp = f[exp_name]
                wave = exp.attrs.get("wave_function", "Unknown")
                desc = exp.attrs.get("description", "No description")
                timestamp = exp.attrs.get("timestamp", "N/A")

                all_experiments.append((filepath, exp_name))

                print(f"  [{len(all_experiments)}] {exp_name}")
                print(f"      Wave: {wave} | Time: {timestamp}")
                print(f"      Description: {desc}")

    return all_experiments


def print_experiment(filepath, exp_name):
    """Print detailed info about an experiment"""
    with h5py.File(filepath, "r") as f:
        exp = f[exp_name]

        print(f"\n{'=' * 70}")
        print(f"Experiment: {exp_name}")
        print(f"File: {os.path.basename(filepath)}")
        print(f"{'=' * 70}")

        print("\nMetadata:")
        for key, value in exp.attrs.items():
            print(f"  {key}: {value}")

        print(f"\nData shape: {exp['data'].shape}")
        print(f"Columns: {list(exp.attrs.get('columns', []))}")
        print(f"{'=' * 70}\n")


def delete_experiment(filepath, exp_name):
    """Delete an experiment"""
    # confirm = input(f"Really delete '{exp_name}'? (yes/no): ")
    confirm = "yes"

    if confirm.lower() == "yes":
        with h5py.File(filepath, "a") as f:
            del f[exp_name]
        print(f"✓ Deleted: {exp_name}")
    else:
        print("Cancelled")


def rename_experiment(filepath, old_name):
    """Rename an experiment"""
    new_name = input(f"Enter new name for '{old_name}': ").strip()

    if not new_name:
        print("Cancelled - empty name")
        return

    with h5py.File(filepath, "a") as f:
        if new_name in f:
            print(f"Error: '{new_name}' already exists!")
            return

        # Copy to new name
        f.copy(old_name, new_name)
        # Delete old
        del f[old_name]

    print(f"✓ Renamed: {old_name} → {new_name}")


def edit_description(filepath, exp_name):
    """Edit experiment description"""
    with h5py.File(filepath, "r") as f:
        exp = f[exp_name]
        current_desc = exp.attrs.get("description", "No description")

    print(f"\nCurrent description: {current_desc}")
    new_desc = input("Enter new description (or press Enter to cancel): ").strip()

    if not new_desc:
        print("Cancelled - no changes made")
        return

    with h5py.File(filepath, "a") as f:
        f[exp_name].attrs["description"] = new_desc

    print(f"✓ Updated description: {new_desc}")


def main():
    """Main menu"""
    while True:
        print("\n" + "=" * 70)
        print("EXPERIMENT MANAGER")
        print("=" * 70)

        experiments = list_experiments()

        if not experiments:
            return

        print("\n" + "=" * 70)
        print("Commands:")
        print("  [number] p - Print experiment details")
        print("  [number] d - Delete experiment")
        print("  [number] r - Rename experiment")
        print("  [number] e - Edit description")
        print("  l p/d/r/e - Work with latest experiment")
        print("  q - Quit")
        print("=" * 70)

        choice = input("\nEnter command: ").strip().lower()

        if choice == "q":
            break

        # Parse command
        parts = choice.split()
        if len(parts) != 2:
            print("Invalid command! Use: [number/latest] [p/d/r]")
            continue

        target, action = parts

        # Get experiment
        if target == "l":
            filepath, exp_name = experiments[-1]
        else:
            try:
                idx = int(target) - 1
                if idx < 0 or idx >= len(experiments):
                    print("Invalid number!")
                    continue
                filepath, exp_name = experiments[idx]
            except ValueError:
                print("Invalid number!")
                continue

        # Execute action
        if action == "p":
            print_experiment(filepath, exp_name)
        elif action == "d":
            delete_experiment(filepath, exp_name)
        elif action == "r":
            rename_experiment(filepath, exp_name)
        elif action == "e":
            edit_description(filepath, exp_name)
        else:
            print("Invalid action! Use p (print), d (delete), r (rename), or e (edit)")


if __name__ == "__main__":
    main()
