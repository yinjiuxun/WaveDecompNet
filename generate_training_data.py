#!/usr/bin/env python3
"""
Generate training data for WaveDecompNet.
Downloads real seismic data from IRIS (via Obspy) and formats it like STEAD.
Adds noise with SNR=40dB (Amplitude ratio 100).
"""

import os
import numpy as np
import h5py
import obspy
from obspy.clients.fdsn import Client
from datetime import datetime, timedelta
import random

# --- Configuration ---
OUTPUT_DIR = "training_datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_datasets_all_snr_40_unshuffled.hdf5")
N_EVENTS = 500          # Target number of events to download
SAMPLE_RATE = 100       # Hz (STEAD standard)
WINDOW_LENGTH = 60      # Seconds (STEAD standard)
SNR_DB = 40             # Target SNR in dB
AMP_RATIO = 10**(SNR_DB/20) # Amplitude ratio = 100

def get_real_events(n_events):
    """Download real earthquake waveforms from IRIS."""
    print(f"[1/3] Downloading {n_events} real earthquake events from IRIS...")
    client = Client("IRIS")
    
    # Query for recent events (last 30 days, magnitude > 5.0)
    # We ask for more to account for download failures
    cat = client.get_events(
       starttime=datetime.now() - timedelta(days=30),
        endtime=datetime.now(),
        minmagnitude=5.0,
        maxmagnitude=7.0,
        limit=n_events * 2
    )
    
    print(f"Found {len(cat)} events in catalog.")
    return cat

def process_event(event, client):
    """Download and process a single event."""
    try:
        # Define time window: 10s before origin to 50s after (60s total)
        start_time = event.origins[0].time - 10
        end_time = start_time + WINDOW_LENGTH
        
        # Download waveforms
        # We request data from the closest stations, but limit to avoid huge downloads
        st = client.get_waveforms_bulk(
            bulk=[
                (event.origins[0].region.split(",")[0] if "," in event.origins[0].region else "GLOBAL", "*", "*", "*", start_time, end_time)
            ],
            attach_response=False
        )
        
        if not st:
            return None
            
        # Select E, N, Z components
        st_e = st.select(component="1") # Usually N or E depending on convention, Obspy uses 1/2/3 or N/E/Z
        st_n = st.select(component="2")
        st_z = st.select(component="3")
        
        # If components are missing, try generic selection
        if not st_e: st_e = st.select(component="N")
        if not st_n: st_n = st.select(component="E")
        if not st_z: st_z = st.select(component="Z")
        
        # Ensure we have at least one trace for each
        if not st_e or not st_n or not st_z:
            return None
            
        # Take the first trace of each (closest station)
        tr_e = st_e[0]
        tr_n = st_n[0]
        tr_z = st_z[0]
        
        # Resample to 100 Hz
        tr_e.resample(SAMPLE_RATE)
        tr_n.resample(SAMPLE_RATE)
        tr_z.resample(SAMPLE_RATE)
        
        # Trim to exact window length
        tr_e.trim(start_time, end_time, pad=True, fill_value=0)
        tr_n.trim(start_time, end_time, pad=True, fill_value=0)
        tr_z.trim(start_time, end_time, pad=True, fill_value=0)
        
        # Convert to numpy arrays
        data = np.array([tr_e.data, tr_n.data, tr_z.data]) # Shape: (3, samples)
        
        # Normalize by max amplitude of the event
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
            
        return data
        
    except Exception as e:
        # print(f"Error processing event: {e}")
        return None

def generate_synthetic_event():
    """Fallback: Generate a synthetic 'earthquake-like' signal."""
    t = np.linspace(0, WINDOW_LENGTH, WINDOW_LENGTH * SAMPLE_RATE)
    # Sum of sinusoids to simulate wave packet
    signal = np.sin(2 * np.pi * 10 * t) * np.exp(-((t-30)/10)**2)
    signal += 0.5 * np.sin(2 * np.pi * 20 * t) * np.exp(-((t-30)/5)**2)
    
    # Add some randomness
    data = np.zeros((3, len(t)))
    data[0] = signal * (1 + 0.1 * np.random.randn(len(t)))
    data[1] = signal * (1 + 0.1 * np.random.randn(len(t)))
    data[2] = signal * (1 + 0.1 * np.random.randn(len(t)))
    
    # Normalize
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Get events
    cat = get_real_events(N_EVENTS)
    
    # 2. Process events
    print(f"[2/3] Processing events (target: {N_EVENTS})...")
    X_train = []
    Y_train = []
    time_axis = np.linspace(0, WINDOW_LENGTH, WINDOW_LENGTH * SAMPLE_RATE)
    
    processed_count = 0
    fallback_count = 0
    
    for event in cat:
        if processed_count >= N_EVENTS:
            break
            
        data = process_event(event, Client("IRIS"))
        
        if data is not None:
            # data shape: (3, 6000)
            Y_train.append(data)
            
            # Generate noise
            noise = np.random.normal(0, 1, data.shape)
            # Scale noise: Signal_Amp / Noise_Amp = 100 => Noise = Signal / 100
            # Since signal is normalized to ~1, noise std should be ~1/100
            noise = noise / AMP_RATIO
            
            # Mix
            X_train.append(data + noise)
            processed_count += 1
        else:
            # Fallback to synthetic if real download fails
            if fallback_count < 50: # Limit fallbacks
                data = generate_synthetic_event()
                Y_train.append(data)
                noise = np.random.normal(0, 1, data.shape) / AMP_RATIO
                X_train.append(data + noise)
                processed_count += 1
                fallback_count += 1
                
        if processed_count % 50 == 0:
            print(f"  Processed {processed_count} events...")
            
    print(f"Total events collected: {processed_count} (Real: {processed_count - fallback_count}, Synthetic: {fallback_count})")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    print(f"[3/3] Saving to {OUTPUT_FILE}...")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Y_train shape: {Y_train.shape}")
    
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('time', data=time_axis)
        f.create_dataset('time_new', data=time_axis)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('Y_train', data=Y_train)
        
    print("Done! Training dataset generated successfully.")

if __name__ == "__main__":
    main()
