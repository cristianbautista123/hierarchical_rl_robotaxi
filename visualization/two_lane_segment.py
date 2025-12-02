import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

LANELET_IDS = [5109, 2363]  # nuestros dos carriles

def load_lanelet_map(path):
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = {}
    ways = {}
    lanelets = {}

    # Nodos
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[node_id] = (lon, lat)

    # Ways
    for way in root.findall("way"):
        wid = int(way.get("id"))
        pts = []
        for nd in way.findall("nd"):
            ref = int(nd.get("ref"))
            if ref in nodes:
                pts.append(nodes[ref])
        ways[wid] = np.asarray(pts)

    # Lanelets
    for rel in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in rel.findall("tag")}
        if tags.get("type") != "lanelet":
            continue

        lid = int(rel.get("id"))
        left, right = None, None
        for m in rel.findall("member"):
            role = m.get("role")
            ref = int(m.get("ref"))
            if role == "left":
                left = ref
            elif role == "right":
                right = ref
        lanelets[lid] = {"left": left, "right": right}

    return nodes, ways, lanelets


def generate_centerline(left_pts, right_pts):
    """Promedia left/right con re-muestreo para tener mismo número de puntos."""
    L, R = len(left_pts), len(right_pts)
    n = max(L, R)

    # interpolate separately x and y
    t_left = np.linspace(0, 1, L)
    t_right = np.linspace(0, 1, R)
    t_common = np.linspace(0, 1, n)

    left_rs_x = np.interp(t_common, t_left, left_pts[:, 0])
    left_rs_y = np.interp(t_common, t_left, left_pts[:, 1])
    right_rs_x = np.interp(t_common, t_right, right_pts[:, 0])
    right_rs_y = np.interp(t_common, t_right, right_pts[:, 1])

    center_x = 0.5 * (left_rs_x + right_rs_x)
    center_y = 0.5 * (left_rs_y + right_rs_y)

    return np.stack([center_x, center_y], axis=1)


def maybe_flip_direction(ref_center, other_center):
    """
    Asegura que 'other_center' vaya en la misma dirección que 'ref_center':
    si la punta del ref está más cerca del final de other que del inicio, invertimos.
    """
    start_ref, end_ref = ref_center[0], ref_center[-1]
    start_other, end_other = other_center[0], other_center[-1]

    dist_start = np.linalg.norm(end_ref - start_other)
    dist_end = np.linalg.norm(end_ref - end_other)

    if dist_end < dist_start:
        # el final de other está más cerca del final de ref: invertimos
        return other_center[::-1]
    return other_center


def main():
    nodes, ways, lanelets = load_lanelet_map("/home/cris/hierarchical_rl_robotaxi/config/BuckeyeLotMap_v2.osm")

    # Comprobar que los lanelets que queremos existen
    for lid in LANELET_IDS:
        if lid not in lanelets:
            raise ValueError(f"Lanelet {lid} no existe en el mapa.")

    # Generar centerlines para cada uno
    centerlines = {}
    boundaries = {}

    for lid in LANELET_IDS:
        ll = lanelets[lid]
        left_pts = ways[ll["left"]]
        right_pts = ways[ll["right"]]
        cl = generate_centerline(left_pts, right_pts)
        centerlines[lid] = cl
        boundaries[lid] = {"left": left_pts, "right": right_pts}

    # Alinear dirección: tomamos el primero como referencia
    ref_id = LANELET_IDS[0]
    ref_cl = centerlines[ref_id]

    for lid in LANELET_IDS[1:]:
        centerlines[lid] = maybe_flip_direction(ref_cl, centerlines[lid])

    # Plot
    plt.figure(figsize=(10, 6))

    colors = {LANELET_IDS[0]: "r", LANELET_IDS[1]: "m"}  # centerlines
    for lid in LANELET_IDS:
        left_pts = boundaries[lid]["left"]
        right_pts = boundaries[lid]["right"]
        cl = centerlines[lid]

        # Boundaries
        plt.plot(left_pts[:,0], left_pts[:,1], "b--", alpha=0.5)
        plt.plot(right_pts[:,0], right_pts[:,1], "g--", alpha=0.5)

        # Centerline
        plt.plot(cl[:,0], cl[:,1], colors[lid], linewidth=2,
                 label=f"centerline lanelet {lid}")

    plt.title("Two-lane segment from lanelets 5109 and 2363\nCenterlines (red/magenta), Left (blue), Right (green)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Opcional: guardar los centerlines para usarlos luego en el entorno RL
    # Cada fila: [x, y]
    np.save("/home/cris/hierarchical_rl_robotaxi/config/lanelet_5109_centerline.npy", centerlines[5109])
    np.save("/home/cris/hierarchical_rl_robotaxi/config/lanelet_2363_centerline.npy", centerlines[2363])
    print("Saved centerlines to config/ as .npy files.")


if __name__ == "__main__":
    main()
