import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def load_lanelet_map(path):
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = {}
    ways = {}
    lanelets = {}

    # ---- LOAD NODES ----
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[node_id] = (lon, lat)

    # ---- LOAD WAYS ----
    for way in root.findall("way"):
        way_id = int(way.get("id"))
        nds = []
        for nd in way.findall("nd"):
            ref = int(nd.get("ref"))
            if ref in nodes:
                nds.append(nodes[ref])
            else:
                print(f"⚠️ Way {way_id} references missing node {ref}")
        ways[way_id] = nds

    # ---- LOAD LANELETS ----
    for rel in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in rel.findall("tag")}
        if tags.get("type") != "lanelet":
            continue

        lanelet_id = int(rel.get("id"))
        left, right, center = None, None, None
        successors, predecessors = [], []

        # Member extraction
        for m in rel.findall("member"):
            role = m.get("role")
            ref = int(m.get("ref"))
            if role == "left":
                left = ref
            elif role == "right":
                right = ref
            elif role == "centerline":
                center = ref

        # Successors / predecessors
        if "successor" in tags:
            successors = [int(tags["successor"])]
        if "predecessor" in tags:
            predecessors = [int(tags["predecessor"])]

        lanelets[lanelet_id] = {
            "left": left,
            "right": right,
            "center": center,
            "successors": successors,
            "predecessors": predecessors,
        }

    return nodes, ways, lanelets


def plot_lanelet_map(nodes, ways, lanelets):
    plt.figure(figsize=(10, 10))

    # Plot ways (default)
    for way_id, points in ways.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, "-k", linewidth=1, alpha=0.4)

    # Highlight lanelet centerlines
    for lid, ll in lanelets.items():
        if ll["center"] in ways:
            pts = ways[ll["center"]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, "-r", linewidth=2, label="centerline" if lid == list(lanelets)[0] else "")

    # Highlight left boundaries
    for lid, ll in lanelets.items():
        if ll["left"] in ways:
            pts = ways[ll["left"]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, "-b", linewidth=1.5, label="left boundary" if lid == list(lanelets)[0] else "")

    # Highlight right boundaries
    for lid, ll in lanelets.items():
        if ll["right"] in ways:
            pts = ways[ll["right"]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, "-g", linewidth=1.5, label="right boundary" if lid == list(lanelets)[0] else "")

    plt.title("Lanelet2 Map — Centerlines (red), Left (blue), Right (green)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def report_inconsistencies(nodes, ways, lanelets):
    print("\n===== MAP CONSISTENCY REPORT =====\n")

    # Missing boundaries
    for lid, ll in lanelets.items():
        if ll["left"] is None:
            print(f"⚠️ Lanelet {lid} has NO left boundary")
        if ll["right"] is None:
            print(f"⚠️ Lanelet {lid} has NO right boundary")
        if ll["center"] is None:
            print(f"⚠️ Lanelet {lid} has NO centerline")

    # Boundaries that are not ways
    for lid, ll in lanelets.items():
        for role in ["left", "right", "center"]:
            wid = ll[role]
            if wid is not None and wid not in ways:
                print(f"❌ Lanelet {lid}: {role} refers to missing way {wid}")

    # Successor checks
    lanelet_ids = set(lanelets.keys())
    for lid, ll in lanelets.items():
        for suc in ll["successors"]:
            if suc not in lanelet_ids:
                print(f"❌ Lanelet {lid} successor {suc} does not exist")

    print("\n==================================\n")


if __name__ == "__main__":
    path = "/home/cris/hierarchical_rl_robotaxi/config/BuckeyeLotMap_v2.osm"    # Ajustar si tu archivo tiene otro nombre

    nodes, ways, lanelets = load_lanelet_map(path)

    print(f"Loaded {len(nodes)} nodes")
    print(f"Loaded {len(ways)} ways")
    print(f"Loaded {len(lanelets)} lanelets")

    print(f"Loaded {len(lanelets)} lanelets")
    print("Lanelet IDs:", list(lanelets.keys()))
    report_inconsistencies(nodes, ways, lanelets)
    plot_lanelet_map(nodes, ways, lanelets)
