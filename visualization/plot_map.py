import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse

# --------------------------------------------------------
# Path setup
# --------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.frenet import FrenetFrame

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
MAIN_CL = os.path.join(CONFIG_DIR, "main_centerline_fixed.npy")
OTHER_CL = os.path.join(CONFIG_DIR, "other_centerline_fixed.npy")
YAML_FILE = os.path.join(CONFIG_DIR, "obstacles.yaml")


def load_yaml():
    """Load full YAML structure."""
    if not os.path.exists(YAML_FILE):
        return {}, [], []

    with open(YAML_FILE, "r") as f:
        data = yaml.safe_load(f)

    return (
        data.get("road_obstacles", []),
        data.get("roadside_objects", []),
        data.get("construction_zone", []),
    )


def compute_lane_offsets(cl_main, cl_other):
    """Compute Frenet-based lateral lane offsets."""
    frenet = FrenetFrame.from_centerline(cl_main)
    d_list = [frenet.xy_to_sd(p)[1] for p in cl_other]
    return [0.0, float(np.mean(d_list))], frenet


def draw_circle(ax, x, y, radius, color, label=None):
    circ = plt.Circle((x, y), radius, color=color, alpha=0.6)
    ax.add_patch(circ)
    if label:
        ax.text(x, y + radius + 0.3, label, ha="center", fontsize=9)


def main(output_dir):
    output_png = os.path.join(output_dir, "map_visualization.png")

    # --------------------------------------------------------
    # Load map
    # --------------------------------------------------------
    cl_main = np.load(MAIN_CL)
    cl_other = np.load(OTHER_CL)

    lane_offsets, frenet = compute_lane_offsets(cl_main, cl_other)

    # --------------------------------------------------------
    # Load ALL YAML elements
    # --------------------------------------------------------
    road_obstacles, roadside_objects, construction_clusters = load_yaml()

    # --------------------------------------------------------
    # Plot figure
    # --------------------------------------------------------
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)

    # Lanes
    ax.plot(cl_main[:, 0], cl_main[:, 1], "k-", linewidth=2, label="Main Lane")
    ax.plot(cl_other[:, 0], cl_other[:, 1], "b--", linewidth=2, label="Other Lane")

    # Start & End
    ax.scatter(cl_main[0, 0], cl_main[0, 1], c="green", s=80, label="Start")
    ax.scatter(cl_main[-1, 0], cl_main[-1, 1], c="red", s=80, marker="X", label="End")

    # --------------------------------------------------------
    # 1) Road obstacles
    # --------------------------------------------------------
    for obs in road_obstacles:
        s = float(obs["s"])
        radius = float(obs["radius"])
        t = obs.get("type", "obj")

        if "lane" in obs:
            d = lane_offsets[int(obs["lane"])]
        else:
            d = float(obs.get("d", 0.0))

        x, y = frenet.sd_to_xy(s, d)

        draw_circle(ax, x, y, radius,
                    color="orange",
                    label=t)

    # --------------------------------------------------------
    # 2) Roadside objects (stop signs, traffic lights)
    # --------------------------------------------------------
    for obj in roadside_objects:
        s = float(obj["s"])
        d = float(obj.get("d", 0.0))
        radius = float(obj.get("radius", 0.5))
        t = obj.get("type", "unknown")

        x, y = frenet.sd_to_xy(s, d)

        color = "red" if t == "stop_sign" else "purple"
        if t == "traffic_light":
            color = "green"  # initial (actual logic handled in env)

        draw_circle(ax, x, y, radius, color=color, label=t)

    # --------------------------------------------------------
    # 3) Construction clusters
    # --------------------------------------------------------
    for cluster in construction_clusters:
        for cone in cluster:
            s = float(cone["s"])
            d = float(cone["d"])
            radius = float(cone.get("radius", 0.3))

            x, y = frenet.sd_to_xy(s, d)
            draw_circle(ax, x, y, radius, color="yellow", label="cone")

    ax.set_title("Map Visualization (All Obstacles)")
    ax.legend()

    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"\nMap saved at: {output_png}\n")

    #plt.show()  # optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG_DIR,
        help="Directory where map_visualization.png will be saved"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args.output_dir)
