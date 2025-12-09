import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# Smoothing
# ============================================================
def smooth(values, window=50):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


# ============================================================
# Plot: raw curve
# ============================================================
def plot_scalar(name, steps, values, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, linewidth=2)
    plt.title(name, fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"{name.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved RAW: {save_path}")


# ============================================================
# Plot: smoothed curve
# ============================================================
def plot_scalar_smoothed(name, steps, values, output_dir, window=50):
    values = np.array(values)
    steps = np.array(steps)

    smoothed_vals = smooth(values, window)
    smoothed_steps = steps[: len(smoothed_vals)]

    # Variance shading
    std_vals = np.zeros_like(smoothed_vals)
    for i in range(len(smoothed_vals)):
        std_vals[i] = np.std(values[max(0, i-window):i+1])

    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_steps, smoothed_vals, color="orange", linewidth=2)

    # Shade confidence band
    plt.fill_between(
        smoothed_steps,
        smoothed_vals - std_vals,
        smoothed_vals + std_vals,
        color="orange",
        alpha=0.25
    )

    plt.title(name + " (Smoothed)", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"{name.replace('/', '_')}_smooth.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved SMOOTH: {save_path}")


# ============================================================
# Load events
# ============================================================
def load_events(event_file):
    ea = EventAccumulator(event_file)
    ea.Reload()
    return ea


# ============================================================
# Main
# ============================================================
def main(run_dir):

    # TensorBoard dir
    tensorboard_dir = os.path.join(run_dir, "tensorboard", "PPO_1")

    # Find events.out.tfevents.*
    event_files = [
        f for f in os.listdir(tensorboard_dir)
        if f.startswith("events.out.tfevents")
    ]

    if len(event_files) == 0:
        raise FileNotFoundError(f"No event file inside {tensorboard_dir}")

    event_file = os.path.join(tensorboard_dir, event_files[0])
    print(f"\nUsing TensorBoard file:\n{event_file}\n")

    # Output directory: run_dir/tensorboard/metrics
    output_dir = os.path.join(run_dir, "tensorboard", "metrics")
    os.makedirs(output_dir, exist_ok=True)

    # Load event data
    ea = load_events(event_file)

    # Scalar tags
    tags = ea.Tags()["scalars"]
    print("\nFound scalars:")
    for t in tags:
        print(" -", t)

    # Generate all plots
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # RAW plot
        plot_scalar(tag, steps, values, output_dir)

        # SMOOTHED plot
        plot_scalar_smoothed(tag, steps, values, output_dir)

    print("\nAll RAW + SMOOTHED metric plots saved!")


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the training run directory: logs/<timestamp>"
    )

    args = parser.parse_args()
    main(args.run_dir)
