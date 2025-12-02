import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def load_lanelet_osm(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Diccionarios:
    nodes = {}
    ways = []

    # 1) Leer nodos
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[node_id] = (lon, lat)   # lon = x, lat = y para visualizar

    # 2) Leer ways (secuencias de nodos)
    for way in root.findall("way"):
        nds = []
        for nd in way.findall("nd"):
            ref = int(nd.get("ref"))
            if ref in nodes:
                nds.append(nodes[ref])
        if len(nds) >= 2:
            ways.append(nds)

    return nodes, ways


def plot_lanelet_map(nodes, ways):
    plt.figure(figsize=(8, 8))

    # Dibujar ways
    for way in ways:
        xs = [p[0] for p in way]
        ys = [p[1] for p in way]
        plt.plot(xs, ys, "-k", linewidth=1)

    # Dibujar los nodos individuales (opcional)
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    plt.scatter(xs, ys, s=10, c="red")

    plt.title("Lanelet2 Map")
    plt.xlabel("Longitude (x)")
    plt.ylabel("Latitude (y)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    path = "/home/cris/hierarchical_rl_robotaxi/config/BuckeyeLotMap_v2.osm"   # Ajusta el nombre de tu archivo
    nodes, ways = load_lanelet_osm(path)
    print(f"Loaded {len(nodes)} nodes")
    print(f"Loaded {len(ways)} ways")
    plot_lanelet_map(nodes, ways)
