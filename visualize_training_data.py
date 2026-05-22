#!/usr/bin/env python3
"""
Visualize random samples from the generated training dataset.

Reads HDF5 training data and plots waveforms for inspection.

Usage:
    python visualize_training_data.py                    # default: 5 samples
    python visualize_training_data.py --n 10             # 10 samples
    python visualize_training_data.py --file path/to/file.hdf5
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_dataset(hdf5_path: str):
    """Load X_train and Y_train from the HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        X = f["X_train"][:]   # (N, 3, 6000)
        Y = f["Y_train"][:]   # (N, 3, 6000)
        time = f["time"][:]   # (6000,)
    return X, Y, time


def plot_samples(X, Y, time, indices, output_path, n_channels=3):
    """Plot waveform samples side by side."""
    channel_names = ["N", "E", "Z"]
    n_samples = len(indices)

    # Layout: n_samples rows, 4 columns per row
    # Col 0: X channel 0, Col 1: X channel 1, Col 2: X channel 2, Col 3: Y channel 0 (reference)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 2.5 * n_samples), sharex=True)
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        # Row title: sample index
        for ch in range(n_channels):
            ax = axes[i, ch]
            ax.plot(time, X[idx, ch, :], color=f"C{ch}", linewidth=0.5)
            ax.set_ylabel(f"X\n{channel_names[ch]}")
            ax.set_ylim(-3.5, 3.5)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.text(0.02, 0.95, f"Sample #{idx}", transform=ax.transAxes,
                        fontsize=8, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

        # Column 3: Y channel 0 (clean reference)
        ax_ref = axes[i, 3]
        ax_ref.plot(time, Y[idx, 0, :], color="red", linewidth=0.8, label="Y (clean)")
        ax_ref.plot(time, X[idx, 0, :], color="blue", linewidth=0.5, alpha=0.6, label="X (noisy)")
        ax_ref.set_ylabel("Y / X\n(ch 0)")
        ax_ref.set_ylim(-3.5, 3.5)
        ax_ref.grid(alpha=0.3)
        if i == 0:
            ax_ref.legend(fontsize=7, loc="upper right")
            ax_ref.text(0.02, 0.95, "Ref: X vs Y", transform=ax_ref.transAxes,
                        fontsize=8, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    axes[-1, 3].set_xlabel("Time (s)")

    fig.suptitle(f"Training Data Samples (n={n_samples})", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training data waveforms")
    parser.add_argument("--file", "-f", default=None,
                        help="Path to training HDF5 file (default: auto-detect in training_datasets/)")
    parser.add_argument("--n", "-n", type=int, default=5,
                        help="Number of random samples to plot (default: 5)")
    parser.add_argument("--seed", type=int, default=12,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: auto-generated)")
    args = parser.parse_args()

    # Auto-detect HDF5 file
    if args.file is None:
        default_dir = Path(__file__).parent / "training_datasets"
        candidates = list(default_dir.glob("*.hdf5")) + list(default_dir.glob("*.h5"))
        if not candidates:
            print("ERROR: No .hdf5/.h5 file found in training_datasets/")
            print("Use --file to specify the path.")
            return
        args.file = str(candidates[-1])  # pick latest
        print(f"Auto-detected: {args.file}")

    X, Y, time = load_dataset(args.file)
    n_samples = X.shape[0]
    print(f"Dataset: {n_samples} samples, shape X={X.shape}, Y={Y.shape}")
    print(f"Time axis: {len(time)} points, dt={time[1]-time[0] if len(time)>1 else 'N/A'}s")

    # Random selection
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_samples, size=min(args.n, n_samples), replace=False).tolist()
    indices.sort()
    print(f"Selected samples: {indices}")

    # Output path
    if args.output is None:
        base = Path(args.file).stem
        args.output = f"training_samples_n{args.n}.png"

    plot_samples(X, Y, time, indices, args.output)
    print(f"Done. Inspect: {args.output}")


if __name__ == "__main__":
    main()
