import numpy as np
import matplotlib.pyplot as plt

MAIN_CENTERLINE_FILE = "/home/cris/hierarchical_rl_robotaxi/config/lanelet_2363_centerline.npy"
OTHER_CENTERLINE_FILE = "/home/cris/hierarchical_rl_robotaxi/config/lanelet_5109_centerline.npy"

OUT_MAIN = "/home/cris/hierarchical_rl_robotaxi/config/main_centerline_local.npy"
OUT_OTHER = "/home/cris/hierarchical_rl_robotaxi/config/other_centerline_local.npy"

def gps_to_meters(centerline):
    """
    Convierte lon/lat a coordenadas en metros SIN cambiar el origen.
    No altera forma, orden ni dirección del path.
    """
    DEG2M = 111111.0

    lon = centerline[:, 0]
    lat = centerline[:, 1]

    # Conversión directa a metros (sin restar origen)
    x = lon * np.cos(lat.mean()) * DEG2M
    y = lat * DEG2M

    return np.stack([x, y], axis=1)

def main():
    # Carga los centerlines originales (lon, lat)
    cl_main = np.load(MAIN_CENTERLINE_FILE)
    cl_other = np.load(OTHER_CENTERLINE_FILE)

    # Conversión a ENU en metros (SIN mover el origen)
    cl_main_m = gps_to_meters(cl_main)
    cl_other_m = gps_to_meters(cl_other)

    print("Main lane meters start:", cl_main_m[0])
    print("Main lane meters end:", cl_main_m[-1])

    # Guardar
    np.save(OUT_MAIN, cl_main_m)
    np.save(OUT_OTHER, cl_other_m)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cl_main_m[:,0], cl_main_m[:,1], "r", label="main (meters)")
    plt.plot(cl_other_m[:,0], cl_other_m[:,1], "b", label="other (meters)")
    plt.title("Centerlines Converted to Meters (No Origin Shift)")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()