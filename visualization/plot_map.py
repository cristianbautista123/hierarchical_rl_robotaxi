import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.frenet import FrenetFrame

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

MAIN_CL = os.path.join(CONFIG_DIR, "main_centerline_fixed.npy")
OTHER_CL = os.path.join(CONFIG_DIR, "other_centerline_fixed.npy")
YAML_FILE = os.path.join(CONFIG_DIR, "obstacles.yaml")
OUTPUT_PNG = os.path.join(CONFIG_DIR, "map_visualization.png")


def load_obstacles():
    """Load obstacles.yaml if exists."""
    if not os.path.exists(YAML_FILE):
        return []

    with open(YAML_FILE, "r") as f:
        data = yaml.safe_load(f)

    return data.get("road_obstacles", [])


def main():
    # --------------------------------------------------------
    # Load centerlines
    # --------------------------------------------------------
    cl_main = np.load(MAIN_CL)
    cl_other = np.load(OTHER_CL)

    # Build Frenet frame to plot s,d obstacles
    frenet = FrenetFrame.from_centerline(cl_main)

    # Start and End
    start_xy = cl_main[0]
    end_xy = cl_main[-1]

    # Load obstacles
    obstacles = load_obstacles()

    # --------------------------------------------------------
    # Plot figure
    # --------------------------------------------------------
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)

    # Plot main & other centerlines
    ax.plot(cl_main[:, 0], cl_main[:, 1], "k-", linewidth=2, label="Main Lane")
    ax.plot(cl_other[:, 0], cl_other[:, 1], "b--", linewidth=1.5, label="Other Lane")

    # Draw start & end markers
    ax.scatter(start_xy[0], start_xy[1], c="green", s=80, marker="o", label="Start")
    ax.scatter(end_xy[0], end_xy[1], c="red", s=80, marker="X", label="End")

    # --------------------------------------------------------
    # Draw obstacles
    # --------------------------------------------------------
    d_list = []
    for p in cl_other:
        _, d_i = frenet.xy_to_sd(p)
        d_list.append(d_i)
    lane_offsets = [0.0, float(np.mean(d_list))]

    # Draw obstacles
    for obs in obstacles:
        s = float(obs["s"])
        radius = float(obs["radius"])
        obst_type = obs.get("type", "unknown")

        # Interpret lane OR d
        if "lane" in obs:
            lane_id = int(obs["lane"])
            d = float(lane_offsets[lane_id])
        else:
            d = float(obs.get("d", 0.0))

        # Convert Frenet → XY
        x, y = frenet.sd_to_xy(s, d)

        # Draw circle
        circle = plt.Circle((x, y), radius, color="orange", alpha=0.6)
        ax.add_patch(circle)

        # Label
        ax.text(
            x, y + radius + 0.5,
            obst_type,
            fontsize=9,
            ha="center",
            va="bottom",
            color="darkred"
        )

    ax.set_title("Map Visualization with Obstacles")
    ax.legend()

    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nMap saved at: {OUTPUT_PNG}\n")

    plt.show()


if __name__ == "__main__":
    main()
