import xml.etree.ElementTree as ET

def inspect_osm(path):
    tree = ET.parse(path)
    root = tree.getroot()

    print("\n========== RELATIONS FOUND ==========\n")

    for rel in root.findall("relation"):
        rid = rel.get("id")
        print(f"Relation ID = {rid}")

        # Print all tags
        print("  TAGS:")
        for tag in rel.findall("tag"):
            print(f"    - {tag.get('k')} = {tag.get('v')}")

        # Print all members
        print("  MEMBERS:")
        for m in rel.findall("member"):
            print(f"    - type={m.get('type')}  role={m.get('role')}  ref={m.get('ref')}")

        print("---------------------------------------")

    print("\n============== DONE ==================\n")


if __name__ == "__main__":
    inspect_osm("/home/cris/hierarchical_rl_robotaxi/config/BuckeyeLotMap_v2.osm")
