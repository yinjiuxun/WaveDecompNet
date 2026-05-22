#!/usr/bin/env python3
"""
Generate training data for WaveDecompNet using real STEAD dataset.

Reads from /workspace/data/STEAD/merged.hdf5 (92GB) and merged.csv metadata:
1. Select earthquake events within 100km distance
2. Select real noise events from the dataset
3. Mix earthquake + noise at target SNR
4. Apply STEAD-style normalization
5. Save to HDF5

Usage:
    python generate_training_data.py
"""

import os
import numpy as np
import h5py
import pandas as pd
from pathlib import Path

# --- Configuration ---
STEAD_HDF5 = "../../data/STEAD/merged.hdf5"
STEAD_CSV = "../../data/STEAD/merged.csv"
OUTPUT_DIR = "training_datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_datasets_stead_snr40.hdf5")
N_EVENTS = 500              # Number of training samples to generate
SAMPLE_RATE = 100           # Hz (STEAD standard)
WINDOW_LENGTH = 60          # Seconds (STEAD standard)
SNR_RATIO = 5             # Amplitude ratio for SNR=40dB (10^(40/20))
MAX_DISTANCE_KM = 100       # Maximum source distance in km
RANDOM_SEED = 42            # For reproducibility


def load_stead_metadata():
    """
    Load CSV metadata and filter earthquake events within MAX_DISTANCE_KM.
    Also collect all noise event names.

    Returns:
        eq_events: list of dicts with trace_name, source_distance_km, etc.
        noise_names: list of noise trace names for random selection
    """
    print(f"[1/4] Loading STEAD metadata from {STEAD_CSV}...")

    # Read the full CSV (handles inconsistent rows better than usecols)
    print("  Reading earthquake events...")
    eq_df = pd.read_csv(STEAD_CSV, engine="c", low_memory=False)

    # Filter: earthquake events within 100km
    eq_events = eq_df[
        (eq_df["trace_category"] == "earthquake_local") &
        (eq_df["source_distance_km"] <= MAX_DISTANCE_KM) &
        (eq_df["source_distance_km"].notna())
    ]
    eq_events = eq_events.reset_index(drop=True)

    # Collect noise event names
    print("  Reading noise events...")
    noise_names = eq_df[
        eq_df["trace_category"] == "noise"
    ]["trace_name"].tolist()

    print(f"  Found {len(eq_events)} earthquake events within {MAX_DISTANCE_KM}km")
    print(f"  Found {len(noise_names)} noise events")

    return eq_events, noise_names


def read_waveform(hdf5_file, trace_name):
    """
    Read a single waveform from the STEAD HDF5 file.
    Each event is stored as data/<trace_name> with shape (6000, 3).

    Args:
        hdf5_file: Open h5py.File handle
        trace_name: Name of the event in the HDF5 file

    Returns:
        waveform: numpy array of shape (3, 6000) — (channels, time)
    """
    dataset = hdf5_file.get(f"data/{trace_name}")
    if dataset is None:
        return None
    return np.array(dataset).astype(np.float64)  # shape: (6000, 3) -> we return (3, 6000)


def normalize_per_channel(signal):
    """
    Normalize signal per channel: zero-mean, unit-variance.
    signal shape: (channels, time)

    Returns normalized signal.
    """
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True) + 1e-12
    return (signal - mean) / std


def mix_event(eq_waveform, noise_waveform, snr_ratio):
    """
    Mix earthquake and noise waveforms at target SNR.

    STEAD-style pipeline:
    1. Normalize earthquake per channel
    2. Scale earthquake by SNR ratio
    3. Normalize noise per channel
    4. Stack: mixed = scaled_eq + noise
    5. Re-normalize the mixed signal

    Args:
        eq_waveform: (3, 6000) earthquake waveform
        noise_waveform: (3, 6000) noise waveform
        snr_ratio: amplitude scaling factor (100 for 40dB)

    Returns:
        X: mixed (noisy) signal, shape (3, 6000)
        Y: clean earthquake signal, shape (3, 6000)
    """
    # Step 1: Normalize earthquake
    eq_norm = normalize_per_channel(eq_waveform)

    # Step 2: Scale earthquake by SNR ratio
    eq_scaled = snr_ratio * eq_norm

    # Step 3: Normalize noise
    noise_norm = normalize_per_channel(noise_waveform)

    # Step 4: Mix
    mixed = eq_scaled + noise_norm

    # Step 5: Re-normalize the mixed signal (this becomes X)
    X = normalize_per_channel(mixed)

    # Y is the earthquake part, also re-normalized by the same factor
    Y = eq_scaled / (np.std(mixed, axis=1, keepdims=True) + 1e-12)

    return X, Y


def generate_training_dataset():
    """Main function: generate training dataset from STEAD data."""
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Load metadata ---
    eq_events, noise_names = load_stead_metadata()

    if len(eq_events) == 0:
        raise RuntimeError("No earthquake events found within the specified distance.")

    # --- Prepare HDF5 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[2/4] Generating {N_EVENTS} training samples...")

    # Pre-allocate output arrays: (N, channels, time)
    X_all = np.zeros((N_EVENTS, 3, WINDOW_LENGTH * SAMPLE_RATE), dtype=np.float32)
    Y_all = np.zeros((N_EVENTS, 3, WINDOW_LENGTH * SAMPLE_RATE), dtype=np.float32)

    # Open STEAD HDF5 for reading (kept open throughout)
    with h5py.File(STEAD_HDF5, "r") as stead_file:
        for i in range(N_EVENTS):
            # Pick random earthquake event
            eq_idx = rng.integers(0, len(eq_events))
            eq_name = eq_events.iloc[eq_idx]["trace_name"]

            # Pick random noise event
            noise_idx = rng.integers(0, len(noise_names))
            noise_name = noise_names[noise_idx]

            # Read waveforms from HDF5
            eq_data = read_waveform(stead_file, eq_name)
            noise_data = read_waveform(stead_file, noise_name)

            if eq_data is None or noise_data is None:
                print(f"  Warning: Skipping {eq_name} or {noise_name} — not found in HDF5")
                # Fill with zeros and continue
                continue

            # Transpose to (channels, time)
            eq_data = eq_data.T  # (6000, 3) -> (3, 6000)
            noise_data = noise_data.T  # (6000, 3) -> (3, 6000)

            # Mix at target SNR
            X, Y = mix_event(eq_data, noise_data, SNR_RATIO)

            X_all[i] = X
            Y_all[i] = Y

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1} / {N_EVENTS}")

    # --- Handle NaNs ---
    X_all[np.isnan(X_all)] = 0.0
    Y_all[np.isnan(Y_all)] = 0.0

    # --- Save to HDF5 ---
    print(f"\n[3/4] Saving to {OUTPUT_FILE}...")
    print(f"  X_train shape: {X_all.shape}")
    print(f"  Y_train shape: {Y_all.shape}")

    time_new = np.arange(0, WINDOW_LENGTH, 1.0 / SAMPLE_RATE)

    with h5py.File(OUTPUT_FILE, "w") as f:
        f.create_dataset("time", data=time_new)
        f.create_dataset("time_new", data=time_new)
        f.create_dataset("X_train", data=X_all, compression="gzip", chunks=True)
        f.create_dataset("Y_train", data=Y_all, compression="gzip", chunks=True)

    print(f"\n[4/4] Done! Generated {N_EVENTS} training samples.")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")


if __name__ == "__main__":
    generate_training_dataset()
