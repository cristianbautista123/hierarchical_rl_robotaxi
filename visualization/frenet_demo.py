import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.frenet import FrenetFrame

# Ajusta estos paths según cómo guardaste los últimos npy
MAIN_CENTERLINE_FILE = "/home/cris/hierarchical_rl_robotaxi/config/main_centerline_fixed.npy"
OTHER_CENTERLINE_FILE = "/home/cris/hierarchical_rl_robotaxi/config/other_centerline_fixed.npy"

def main():
    cl_main = np.load(MAIN_CENTERLINE_FILE)  # shape (N, 2), en metros
    cl_other = np.load(OTHER_CENTERLINE_FILE)

    fr = FrenetFrame.from_centerline(cl_main)

    # Tomamos algunos puntos sobre el centerline y los perturbamos lateralmente
    num_pts = 6
    s_samples = np.linspace(0.1 * fr.s_accum[-1], 0.9 * fr.s_accum[-1], num_pts)
    d_samples = np.linspace(-2.0, 2.0, num_pts)  # offsets laterales de prueba

    test_points = []
    for s, d in zip(s_samples, d_samples):
        xy = fr.sd_to_xy(s, d)
        test_points.append((s, d, xy))

    # Ahora probamos xy -> (s,d) de vuelta
    print("Pruebas de Frenet (s,d) -> (x,y) -> (s',d'):")
    reconverted = []
    for s, d, xy in test_points:
        s2, d2 = fr.xy_to_sd(xy)
        reconverted.append((s2, d2))
        print(f"Original: s={s:.2f}, d={d:.2f}  |  Reconstruido: s'={s2:.2f}, d'={d2:.2f}")

    # Visualización
    plt.figure(figsize=(10, 6))
    plt.plot(cl_main[:,0], cl_main[:,1], "k-", label="centerline main")
    plt.plot(cl_other[:,0], cl_other[:,1], "b--", alpha=0.5, label="other lane")

    # puntos de prueba
    xs = [p[2][0] for p in test_points]
    ys = [p[2][1] for p in test_points]
    plt.scatter(xs, ys, c="red", label="test (s,d) points")

    for (s, d, xy), (s2, d2) in zip(test_points, reconverted):
        plt.text(xy[0], xy[1], f"s={s:.1f}\nd={d:.1f}", fontsize=8)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Frenet demo sobre centerline principal")
    plt.show()

if __name__ == "__main__":
    main()
