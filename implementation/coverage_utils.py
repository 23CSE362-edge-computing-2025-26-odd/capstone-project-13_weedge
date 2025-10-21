import json
import math


def infer_coverage_from_dataset(dataset_path="./dataset.json", coverage_radius=25):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Maps base station id to coordinates and coverage radius
    basestations = {}
    for node in data.get("BaseStation", []):
        bs_id = node["attributes"]["id"]
        coords = node["attributes"].get("coordinates", [0, 0])
        # Add coverage radius attribute if missing
        cr = node["attributes"].get("coverageradius", coverage_radius)
        basestations[bs_id] = (coords, cr)

    # Maps edge server id to base station id.
    # Prefer explicit mapping from EdgeServer objects (they often include a 'base_station' relationship).
    edgeservers_bs = {}
    for es in data.get("EdgeServer", []):
        es_id = es.get("attributes", {}).get("id")
        # relationship may be under 'relationships' -> 'base_station'
        bs_rel = es.get("relationships", {}).get("base_station") or es.get("attributes", {}).get("basestation")
        bs_id = None
        if isinstance(bs_rel, dict):
            bs_id = bs_rel.get("id")
        elif isinstance(bs_rel, (int, str)):
            bs_id = bs_rel
        if es_id is not None and bs_id is not None:
            edgeservers_bs[es_id] = bs_id

    # Fallback: if no mapping found, try to infer from NetworkSwitch objects (older dataset variants).
    if not edgeservers_bs:
        for network in data.get("NetworkSwitch", []):
            for es in network.get("relationships", {}).get("edge_servers", []):
                es_id = es.get("id") if isinstance(es, dict) else None
                # NetworkSwitch may be attached to a BaseStation via relationships or attributes; try both
                bs_id = None
                # prefer explicit base_station relationship on the switch
                bs_rel = network.get("relationships", {}).get("base_station")
                if isinstance(bs_rel, dict):
                    bs_id = bs_rel.get("id")
                if bs_id is None:
                    bs_id = network.get("attributes", {}).get("id")
                if es_id is not None and bs_id is not None:
                    edgeservers_bs[es_id] = bs_id

    # Tasks associated with users (User coordinates)
    tasks_coords = {}
    for user in data.get("User", []):
        task_id = user["attributes"]["id"]
        coords = user["attributes"].get("coordinates", [None, None])
        tasks_coords[task_id] = coords

    # Compute coverage dict for each task: edge server id -> 0 or 1
    coverage_maps = {}
    for task_id, t_coords in tasks_coords.items():
        coverage = {}
        if None in t_coords:
            # If no coordinates for task, assume no coverage
            for es_id in edgeservers_bs.keys():
                coverage[es_id] = 0
        else:
            for es_id, bs_id in edgeservers_bs.items():
                bs_coords, cr = basestations.get(
                    bs_id, ([0, 0], coverage_radius))
                dist = math.sqrt(
                    (t_coords[0] - bs_coords[0])**2 + (t_coords[1] - bs_coords[1])**2)
                coverage[es_id] = 1 if dist <= cr else 0
        coverage_maps[task_id] = coverage

    return coverage_maps
