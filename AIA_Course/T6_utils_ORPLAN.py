#importing the required libraries
import json
import os
import math
import re
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from shapely.geometry.polygon import orient

SIMPLIFY_TOL = 0.15
SIMPLIFY_TOL_DOOR = 0.01
SNAP_IOU_THRESHOLD = 0.90
EDGE_ADJ_DIST_TOL = 2.0
PARALLEL_TOL = 0.60
MARGIN = 40
EDGE_ADJ_DIST_TOL_ENTRANCE = 6.0
PARALLEL_TOL_ENTRANCE = 0.50

def polygon_iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union else 0.0


def simplify_and_snap(poly, zone_type):
    tol = SIMPLIFY_TOL_DOOR if zone_type in [15, 17] else SIMPLIFY_TOL
    simp = poly.simplify(tol, preserve_topology=True)
    oobb = simp.minimum_rotated_rectangle
    iou = polygon_iou(simp, oobb)
    return oobb if iou > SNAP_IOU_THRESHOLD else simp


def normalize_polygons(polygons):
    all_coords = np.vstack([np.array(p.exterior.coords) for p in polygons])
    minx, miny = all_coords.min(axis=0)
    maxx, maxy = all_coords.max(axis=0)
    scale = (255 - 2 * MARGIN) / max(maxx - minx, maxy - miny)

    normed = []
    for p in polygons:
        coords = [((x - minx) * scale + MARGIN, (y - miny) * scale + MARGIN)
                  for x, y in p.exterior.coords]
        normed.append(Polygon(coords))
    return normed


def extract_edges(polygons):
    edges = []
    for idx, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            edges.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), idx])
    return edges


def _rotate_ring_to_canonical_start(coords):
    ring = coords[:-1]
    start_i = min(range(len(ring)), key=lambda i: (round(ring[i][1], 1), round(ring[i][0], 1)))
    rotated = ring[start_i:] + ring[:start_i]
    return rotated + [rotated[0]]


def canonicalize_polygon(poly, clockwise=True):
    poly2 = orient(poly, sign=-1.0 if clockwise else 1.0)
    ext = _rotate_ring_to_canonical_start(list(poly2.exterior.coords))
    return Polygon(ext)


def assign_ed_rm(zone_types, edges,
                 dist_tol=EDGE_ADJ_DIST_TOL,
                 parallel_tol=PARALLEL_TOL):
    ed_rm = [[e[4]] for e in edges]
    edge_lines = [LineString([(e[0], e[1]), (e[2], e[3])]) for e in edges]

    door_idxs = [i for i, zt in enumerate(zone_types) if zt in [15, 17]]
    zone_idxs = [i for i, zt in enumerate(zone_types) if zt not in [15, 17]]

    for di in door_idxs:
        door_edges = [ei for ei, e in enumerate(edges) if e[4] == di]
        if len(door_edges) < 4:
            continue

        lengths = [edge_lines[ei].length for ei in door_edges]
        long_edge_ids = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)[:2]
        long_edges = [door_edges[i] for i in long_edge_ids]

        connected_zones = []

        for ei in long_edges:
            door_line = edge_lines[ei]
            dx_door = edges[ei][2] - edges[ei][0]
            dy_door = edges[ei][3] - edges[ei][1]

            closest_zone = None
            min_dist = float("inf")

            for zei, e2 in enumerate(edges):
                zj = e2[4]
                if zj not in zone_idxs:
                    continue

                zl = edge_lines[zei]

                if zone_types[di] == 15:
                    dist_tol_use = EDGE_ADJ_DIST_TOL_ENTRANCE
                    parallel_tol_use = PARALLEL_TOL_ENTRANCE
                else:
                    dist_tol_use = dist_tol
                    parallel_tol_use = parallel_tol

                dx2, dy2 = e2[2] - e2[0], e2[3] - e2[1]
                norm1, norm2 = math.hypot(dx_door, dy_door), math.hypot(dx2, dy2)
                if norm1 == 0 or norm2 == 0:
                    continue

                cos_angle = abs((dx_door * dx2 + dy_door * dy2) / (norm1 * norm2))
                if cos_angle < parallel_tol_use:
                    continue

                d = door_line.distance(zl)
                if d < min_dist and d < dist_tol_use:
                    min_dist = d
                    closest_zone = (zj, zei)

            if closest_zone:
                zone_id, zone_edge_idx = closest_zone
                ed_rm[ei] = [di, zone_id]
                connected_zones.append((zone_id, zone_edge_idx))

        if len(connected_zones) == 2:
            (z1, e1), (z2, e2) = connected_zones
            ed_rm[e1] = [z1, z2]
            ed_rm[e2] = [z2, z1]

        if zone_types[di] == 15 and len(long_edges) == 2:
            ed_rm[long_edges[1]] = [di]

    return ed_rm


from shapely import affinity

def rotate_polygons(polygons, angle_degrees, origin="centroid"):

    if not polygons or angle_degrees % 360 == 0:
        return polygons

    if origin == "centroid":
        all_centroids = np.array([p.centroid.coords[0] for p in polygons])
        cx, cy = all_centroids.mean(axis=0)
        origin_use = (cx, cy)
    else:
        origin_use = origin

    return [
        affinity.rotate(p, angle_degrees, origin=origin_use, use_radians=False)
        for p in polygons
    ]
def canonical_sort_indices(polygons, zone_types):
    centroids = np.array([p.centroid.coords[0] for p in polygons])

    def prio(zt):
        if zt == 15:
            return 2
        if zt == 17:
            return 1
        return 0

    idxs = list(range(len(polygons)))
    idxs.sort(key=lambda i: (prio(zone_types[i]), -centroids[i][1], centroids[i][0]))
    return idxs


def apply_reorder(polygons, zone_types, idxs):
    return [polygons[i] for i in idxs], [zone_types[i] for i in idxs]





def compute_orient_rooms_only(polygons, zone_types):
    centroids = np.array([p.centroid.coords[0] for p in polygons])
    room_idxs = [i for i, zt in enumerate(zone_types) if zt not in [15, 17]]
    overall_c = centroids[room_idxs].mean(axis=0) if room_idxs else centroids.mean(axis=0)

    boxes = [[round(x, 1) for x in p.bounds] for p in polygons]
    orient_vals = [0] * len(polygons)

    for i in room_idxs:
        xmin, ymin, xmax, ymax = boxes[i]
        cx, cy = centroids[i]
        w = xmax - xmin
        h = ymax - ymin

        if w >= h:
            orient_vals[i] = 19 if cy > overall_c[1] else 21
        else:
            orient_vals[i] = 20 if cx > overall_c[0] else 18

    return orient_vals, boxes


def row_to_zone_type(row, zoning_col="zoning"):
    zoning = str(row.get(zoning_col, "")).strip().lower()

    zoning_map = {
        "zone01": 1,
        "zone02": 2,
        "zone03": 3,
        "zone04": 4,
        "zone05": 5, #this zone is only available when using the OHD model pre-trained on ORPLAN
        "door": 17,
        "entrance_door": 15, #this zone is available only if you could distinguish the entrance_door from the rest (internal doors)
    }

    if zoning in zoning_map:
        return zoning_map[zoning]

    m = re.search(r"(\d{2})", zoning)
    if m and m.group(1) in {"01", "02", "03", "04"}:
        return int(m.group(1))

    return 0
def filter_internal_doors_without_two_sided_room_adjacency(polygons, zone_types,
                                                           dist_tol=EDGE_ADJ_DIST_TOL,
                                                           parallel_tol=PARALLEL_TOL):
    if not polygons:
        return polygons, zone_types

    edges_full = extract_edges(polygons)
    ed_rm = assign_ed_rm(zone_types, edges_full, dist_tol=dist_tol, parallel_tol=parallel_tol)
    edge_lines = [LineString([(e[0], e[1]), (e[2], e[3])]) for e in edges_full]

    door_idxs = [i for i, zt in enumerate(zone_types) if zt == 17]
    room_set = set(i for i, zt in enumerate(zone_types) if zt not in [15, 17])

    keep = [True] * len(polygons)

    for di in door_idxs:
        door_edge_ids = [ei for ei, e in enumerate(edges_full) if e[4] == di]
        if len(door_edge_ids) < 4:
            keep[di] = False
            continue

        lengths = [edge_lines[ei].length for ei in door_edge_ids]
        long_local = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)[:2]
        long_edges = [door_edge_ids[k] for k in long_local]

        hits = 0
        for ei in long_edges:
            rm = ed_rm[ei]
            if len(rm) == 2 and rm[1] in room_set:
                hits += 1

        if hits < 2:
            keep[di] = False

    idxs = [i for i, ok in enumerate(keep) if ok]
    return [polygons[i] for i in idxs], [zone_types[i] for i in idxs]

def plot_with_polygons(polygons, edges, ed_rm, zone_types, orient_vals, title="Unit JSON geometry"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.grid(alpha=0.2)

    zone_color_map = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
        4: "#9467bd",
        15: "#8c564b",
        17: "#d62728",
        0: "#e0e0e0",
    }
    orient_labels = {18: "W", 19: "N", 20: "E", 21: "S", 0: "0"}

    for i, poly in enumerate(polygons):
        zt = zone_types[i]
        x, y = poly.exterior.xy
        ax.fill(x, y, color=zone_color_map.get(zt, "#e0e0e0"), alpha=0.5,
                edgecolor="black", linewidth=0.8)

        cx, cy = poly.centroid.coords[0]
        ax.text(
            cx, cy,
            f"{i}\nzt:{zt}\nori:{orient_labels.get(orient_vals[i], '?')}",
            fontsize=7, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.25")
        )

    for e, rm in zip(edges, ed_rm):
        x1, y1, x2, y2 = e[:4]
        color = "red" if len(rm) == 2 else "#bbbbbb"
        width = 2.2 if len(rm) == 2 else 1.0
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.9)

    plt.tight_layout()
    plt.show()


)

def _find_column(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def unit_csv_to_json(
    csv_path,
    show_plot=True,
    return_data=False,
    geometry_col=None,
    zoning_col="zoning",
    is_unit_polygon_col=None,
    rotation_degrees=0,
):
    df = pd.read_csv(csv_path)

    # --- normalize column names ---
    df.columns = df.columns.str.lower()

    # --- resolve geometry column ---
    geom_candidates = ["geometry", "geometry_wkt", "geom", "wkt"]
    if geometry_col:
        geometry_col = geometry_col.lower()

    if not geometry_col or geometry_col not in df.columns:
        geometry_col = _find_column(df, geom_candidates)

    if geometry_col is None:
        raise ValueError(f"No valid geometry column found in {csv_path}")

    print(f"  Using geometry column: {geometry_col}")

    # --- resolve optional columns ---
    if is_unit_polygon_col:
        is_unit_polygon_col = is_unit_polygon_col.lower()

    if zoning_col:
        zoning_col = zoning_col.lower()

    if is_unit_polygon_col not in df.columns:
        print("  is_unit_polygon column not found → skipping filter")
        is_unit_polygon_col = None

    if zoning_col not in df.columns:
        print("  zoning column not found → zone type fallback may apply")
        zoning_col = None

    # --- filter unit polygons if column exists ---
    if is_unit_polygon_col:
        df = df[df[is_unit_polygon_col] != True].copy()

    df = df.reset_index(drop=True)

    polygons = []
    zone_types = []

    for _, row in df.iterrows():
        try:
            geom = wkt.loads(row[geometry_col])
        except Exception:
            continue

        if geom.geom_type == "MultiPolygon":
            geom = max(list(geom.geoms), key=lambda g: g.area)

        if geom.geom_type != "Polygon":
            continue

        zt = row_to_zone_type(row, zoning_col=zoning_col) if zoning_col else 1
        if zt == 0:
            continue

        polygons.append(geom)
        zone_types.append(zt)

    if not polygons:
        raise ValueError(f"No valid polygons found in {csv_path}")

    # --- processing pipeline ---
    polygons = [simplify_and_snap(p, zt) for p, zt in zip(polygons, zone_types)]
    polygons = rotate_polygons(polygons, rotation_degrees, origin="centroid")
    polygons = normalize_polygons(polygons)

    before = sum(1 for z in zone_types if z == 17)
    polygons, zone_types = filter_internal_doors_without_two_sided_room_adjacency(
        polygons, zone_types
    )
    after = sum(1 for z in zone_types if z == 17)
    print(f"  Removed {before - after} internal doors lacking two-sided room adjacency")

    idxs = canonical_sort_indices(polygons, zone_types)
    polygons, zone_types = apply_reorder(polygons, zone_types, idxs)

    orient_vals, boxes = compute_orient_rooms_only(polygons, zone_types)

    polygons = [canonicalize_polygon(p, clockwise=True) for p in polygons]
    edges_full = extract_edges(polygons)
    ed_rm = assign_ed_rm(zone_types, edges_full)
    edges = [[e[0], e[1], e[2], e[3]] for e in edges_full]

    data = {
        "room_type": zone_types,
        "boxes": boxes,
        "edges": edges,
        "ed_rm": ed_rm,
        "orient": orient_vals,
        "rotation_degrees": rotation_degrees,
    }

    if show_plot:
        plot_with_polygons(
            polygons,
            edges_full,
            ed_rm,
            zone_types,
            orient_vals,
            title=f"{os.path.basename(csv_path)} | rot={rotation_degrees}°",
        )

    if return_data:
        return data

    out_path = os.path.splitext(csv_path)[0] + ".json"
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(", ", ": "))

    print(f"  Saved JSON -> {out_path}")
    return data


def batch_convert_unit_csvs_with_plots(
    csv_dir,
    output_dir,
    make_plots=True,
    geometry_col=None,
    zoning_col="zoning",
    is_unit_polygon_col=None,
    rotation_degrees=0,
):
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(csv_dir)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(csv_dir, f))
    )

    if not csv_files:
        print("❌ No CSV files found.")
        return

    print(f"📂 Found {len(csv_files)} CSV files")

    for fname in csv_files:
        csv_path = os.path.join(csv_dir, fname)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".json")

        print(f"\n🔄 Processing {fname} ...")

        try:
            data = unit_csv_to_json(
                csv_path=csv_path,
                show_plot=make_plots,
                return_data=True,
                geometry_col=geometry_col,
                zoning_col=zoning_col,
                is_unit_polygon_col=is_unit_polygon_col,
                rotation_degrees=rotation_degrees,
            )

            with open(out_path, "w") as f:
                json.dump(data, f, separators=(", ", ": "))

            print(f"  ✔ Saved -> {out_path}")

        except Exception as e:
            print(f"  ⚠ Error in {fname}: {e}")

    print("\n Batch conversion complete.")
