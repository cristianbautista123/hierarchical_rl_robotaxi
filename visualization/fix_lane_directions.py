import numpy as np

MAIN_FILE  = "/home/cris/hierarchical_rl_robotaxi/config/main_centerline_local.npy"
OTHER_FILE = "/home/cris/hierarchical_rl_robotaxi/config/other_centerline_local.npy"

OUT_MAIN  = "/home/cris/hierarchical_rl_robotaxi/config/main_centerline_fixed.npy"
OUT_OTHER = "/home/cris/hierarchical_rl_robotaxi/config/other_centerline_fixed.npy"

def compute_tangents(cl):
    v = np.diff(cl, axis=0)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def average_cosine(main_cl, other_cl):
    tan_main  = compute_tangents(main_cl)
    tan_other = compute_tangents(other_cl)
    n = min(len(tan_main), len(tan_other))
    cosines = np.sum(tan_main[:n] * tan_other[:n], axis=1)
    return cosines.mean()

def main():
    cl_main  = np.load(MAIN_FILE)
    cl_other = np.load(OTHER_FILE)

    print("Manually inverting MAIN lane...")
    cl_main = cl_main[::-1].copy()     #  ★★★★★ ESTE ES EL CAMBIO IMPORTANTE ★★★★★

    # Ahora corregimos OTHER según MAIN
    cos_before = average_cosine(cl_main, cl_other)
    print(f"Average cosine BEFORE fix: {cos_before:.3f}")

    if cos_before < 0:
        print("Inverting OTHER lane to match MAIN direction...")
        cl_other = cl_other[::-1].copy()

    cos_after = average_cosine(cl_main, cl_other)
    print(f"Average cosine AFTER fix: {cos_after:.3f}")

    np.save(OUT_MAIN, cl_main)
    np.save(OUT_OTHER, cl_other)

    print("\nDONE. Saved:")
    print("  ", OUT_MAIN)
    print("  ", OUT_OTHER)

if __name__ == "__main__":
    main()
