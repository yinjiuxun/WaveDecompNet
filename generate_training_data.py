#!/usr/bin/env python3
"""
Generate training data for WaveDecompNet using the STEAD-style pipeline.
Replicates the logic from WaveDecompNet-paper/step2b_prepare_STEAD_waveforms.py:
1. Generate realistic earthquake-like wavelets and background noise.
2. Normalize, apply random time shifts, mix with fixed SNR=40dB (amplitude ratio 100).
3. Re-scale stacked signals and save to HDF5.
"""

import os
import numpy as np
import h5py
from scipy.interpolate import interp1d
import random

# --- Configuration ---
OUTPUT_DIR = "training_datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_datasets_all_snr_40_unshuffled.hdf5")
N_EVENTS = 500          # Target number of events to generate
SAMPLE_RATE = 100       # Hz (STEAD standard)
WINDOW_LENGTH = 60      # Seconds (STEAD standard)
SNR_RATIO = 100         # Amplitude ratio for SNR=40dB (10^(40/20))

def generate_earthquake_waveform():
    """Generate a realistic earthquake-like waveform (sum of damped sinusoids)."""
    t = np.linspace(0, WINDOW_LENGTH, WINDOW_LENGTH * SAMPLE_RATE)
    signal = np.zeros_like(t)
    
    # P-wave arrival
    p_arrival = np.random.uniform(10, 30)
    p_freq = np.random.uniform(1, 5)
    signal += np.sin(2 * np.pi * p_freq * (t - p_arrival)) * np.exp(-((t - p_arrival) / 2)**2) * (t > p_arrival)
    
    # S-wave arrival
    s_arrival = p_arrival + np.random.uniform(2, 10)
    s_freq = np.random.uniform(0.5, 2)
    signal += 2 * np.sin(2 * np.pi * s_freq * (t - s_arrival)) * np.exp(-((t - s_arrival) / 5)**2) * (t > s_arrival)
    
    # Coda waves
    coda_arrival = s_arrival + np.random.uniform(5, 15)
    coda_freq = np.random.uniform(0.2, 1)
    signal += 0.5 * np.sin(2 * np.pi * coda_freq * (t - coda_arrival)) * np.exp(-((t - coda_arrival) / 10)**2) * (t > coda_arrival)
    
    # Add slight randomness
    signal += 0.1 * np.random.randn(len(t))
    
    return signal

def generate_noise_waveform():
    """Generate realistic background noise (colored noise)."""
    t = np.linspace(0, WINDOW_LENGTH, WINDOW_LENGTH * SAMPLE_RATE)
    # 1/f noise approximation
    noise = np.random.randn(len(t))
    noise = np.convolve(noise, np.exp(-np.arange(50)/10), mode='same')
    noise += 0.2 * np.random.randn(len(t))
    return noise

def process_waveform_3d(eq_sig, noise_sig):
    """Process 1D signals into 3D (E, N, Z) arrays following STEAD normalization."""
    # Create 3 components with slight variations
    eq_data = np.array([
        eq_sig * (1 + 0.1 * np.random.randn(len(eq_sig))),
        eq_sig * (1 + 0.1 * np.random.randn(len(eq_sig))),
        eq_sig * (1 + 0.2 * np.random.randn(len(eq_sig))) # Z often has different amplitude
    ])
    
    noise_data = np.array([
        noise_sig * (1 + 0.1 * np.random.randn(len(noise_sig))),
        noise_sig * (1 + 0.1 * np.random.randn(len(noise_sig))),
        noise_sig * (1 + 0.1 * np.random.randn(len(noise_sig)))
    ])
    
    # STEAD-style normalization: zero-mean, unit-variance per channel (axis=1 is time)
    eq_data = (eq_data - np.mean(eq_data, axis=1, keepdims=True)) / (np.std(eq_data, axis=1, keepdims=True) + 1e-12)
    noise_data = (noise_data - np.mean(noise_data, axis=1, keepdims=True)) / (np.std(noise_data, axis=1, keepdims=True) + 1e-12)
    
    return eq_data, noise_data

def generate_stead_style_dataset():
    """Generate dataset following step2b_prepare_STEAD_waveforms.py logic."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[1/3] Generating {N_EVENTS} STEAD-style earthquake and noise waveforms...")
    X_train, Y_train = [], []
    time_new = np.arange(0, WINDOW_LENGTH, 1/SAMPLE_RATE)
    
    # Pre-generate random shifts
    rng_shift = np.random.default_rng(103)
    shifts = rng_shift.uniform(-30, 60, N_EVENTS)
    
    for i in range(N_EVENTS):
        if i % 100 == 0:
            print(f"  Processing {i} / {N_EVENTS}...")
            
        # 1. Generate base waveforms
        eq_sig = generate_earthquake_waveform()
        noise_sig = generate_noise_waveform()
        
        eq_data, noise_data = process_waveform_3d(eq_sig, noise_sig)
        
        # 2. STEAD-style mixing:
        # a. Random shift earthquake signal
        shift_func = interp1d(
            time_new + shifts[i], 
            eq_data, 
            axis=1, # Time is along axis 1 in (channels, time) shape
            kind='nearest', 
            bounds_error=False, 
            fill_value=0.
        )
        shifted_eq = shift_func(time_new)
        
        # b. Scale earthquake by SNR ratio (100 for 40dB)
        scaled_eq = SNR_RATIO * shifted_eq
        
        # c. Stack with noise
        stacked = scaled_eq + noise_data
        
        # d. Re-scale by stacked standard deviation (STEAD style)
        scaling_std = np.std(stacked, axis=1, keepdims=True)
        stacked = stacked / scaling_std
        scaled_eq = scaled_eq / scaling_std
        
        X_train.append(stacked)
        Y_train.append(scaled_eq)
        
    print(f"Total events generated: {len(X_train)}")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # Handle NaNs
    X_train[np.isnan(X_train)] = 0
    Y_train[np.isnan(Y_train)] = 0
    
    print(f"[2/3] Saving to {OUTPUT_FILE}...")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Y_train shape: {Y_train.shape}")
    
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('time', data=time_new)
        f.create_dataset('time_new', data=time_new)
        f.create_dataset('X_train', data=X_train, compression="gzip", chunks=True)
        f.create_dataset('Y_train', data=Y_train, compression="gzip", chunks=True)
        
    print("[3/3] Done! Training dataset generated successfully.")

if __name__ == "__main__":
    generate_stead_style_dataset()
