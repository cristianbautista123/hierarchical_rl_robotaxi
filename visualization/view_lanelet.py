import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def load_lanelet_map(path):
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = {}
    ways = {}
    lanelets = {}

    # NODES
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[node_id] = (lon, lat)

    # WAYS
    for way in root.findall("way"):
        way_id = int(way.get("id"))
        nds = []
        for nd in way.findall("nd"):
            ref = int(nd.get("ref"))
            if ref in nodes:
                nds.append(nodes[ref])
        ways[way_id] = nds

    # LANELETS
    for rel in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in rel.findall("tag")}
        if tags.get("type") != "lanelet":
            continue

        lid = int(rel.get("id"))
        left, right, center = None, None, None

        for m in rel.findall("member"):
            role = m.get("role")
            ref = int(m.get("ref"))
            if role == "left":
                left = ref
            elif role == "right":
                right = ref
            elif role == "centerline":
                center = ref

        lanelets[lid] = {"left": left, "right": right, "center": center}

    return nodes, ways, lanelets


def plot_lanelet(lanelet_id, nodes, ways, lanelets):
    ll = lanelets[lanelet_id]

    plt.figure(figsize=(8, 8))

    def plot_way(way_id, color, label):
        if way_id is None or way_id not in ways:
            return
        pts = ways[way_id]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, color, linewidth=3, label=label)

    plot_way(ll["left"], "-b", "left boundary")
    plot_way(ll["right"], "-g", "right boundary")
    plot_way(ll["center"], "-r", "centerline")

    plt.scatter(
        [p[0] for p in nodes.values()],
        [p[1] for p in nodes.values()],
        s=30, c="gray", alpha=0.3
    )

    plt.title(f"Lanelet {lanelet_id}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    path = "/home/cris/hierarchical_rl_robotaxi/config/BuckeyeLotMap_v2.osm"

    nodes, ways, lanelets = load_lanelet_map(path)

    print("Lanelets available:", len(lanelets))
    lanelet_id = int(input("Enter lanelet ID to visualize: "))

    plot_lanelet(lanelet_id, nodes, ways, lanelets)
