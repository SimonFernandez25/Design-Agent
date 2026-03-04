"""
endpoint_generator_v2.py - Per-Primitive Local Parameter Endpoint Generator

Converts enriched_reduced.json (output of manufacturing_agent.py) into scan paths.

KEY CHANGES FROM V1:
- Each primitive is sliced independently using its own local_parameters
  (hatch_spacing_um, slice_spacing_um, solid_contour_count)
- Z layers are quantized to a global grid (Z_BIN = 0.05 um) to prevent
  an explosion of near-duplicate Z keys across primitives
- Toolpaths from different primitives at the same quantized Z key are merged
- Exposure parameters (laser power / scan speed) remain global from PrintParameters.txt
- Simple transition buffer: overlapping primitives with hatch diff > 0.1 use
  the denser (smaller) hatch for the overlap region
"""

import json
import math
import copy
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from shapely.geometry import Polygon, Point, LineString, box as shapely_box
from shapely.ops import unary_union
from shapely import affinity


# ======================================================
# CONFIGURATION
# ======================================================

Z_BIN = 0.05  # um - global Z quantization step


# ======================================================
# PRINT PARAMETERS
# ======================================================

def load_print_parameters(param_file: Path) -> Dict[str, float]:
    """Load global print parameters from file. Used for exposure settings only."""
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    params[key.strip()] = float(value.strip())
                except ValueError:
                    params[key.strip()] = value.strip()
    return params


# ======================================================
# PRIMITIVE -> SHAPELY FOOTPRINT
# ======================================================

def get_primitive_footprint(primitive: Dict) -> Optional[Polygon]:
    """
    Convert a primitive to its XY-plane Shapely footprint (cross-section).
    """
    prim_type = primitive['type']
    center = primitive['center']
    dims = primitive['dimensions']
    rot_z = primitive.get('rotation_z_deg', 0.0)
    cx, cy = center[0], center[1]

    if prim_type == 'box':
        dx = dims.get('x_um', dims.get('length_um', dims.get('width_um', 0.0)))
        dy = dims.get('y_um', dims.get('depth_um', 0.0))
        poly = shapely_box(-dx / 2, -dy / 2, dx / 2, dy / 2)
        if rot_z != 0:
            poly = affinity.rotate(poly, rot_z, origin=(0, 0))
        poly = affinity.translate(poly, cx, cy)
        return poly

    elif prim_type == 'cylinder':
        diameter = dims.get('diameter_um', dims.get('radius_um', 0.0) * 2)
        radius = diameter / 2.0
        poly = Point(cx, cy).buffer(radius, resolution=16)
        return poly

    return None


def primitive_z_range(primitive: Dict) -> Tuple[float, float]:
    """Return (z_min, z_max) for a primitive."""
    center_z = primitive['center'][2]
    dims = primitive['dimensions']
    prim_type = primitive.get('type', 'box')

    if prim_type == 'cylinder':
        height = dims.get('height_um', dims.get('z_um', 0.0))
    else:
        height = dims.get('z_um', dims.get('height_um', 0.0))

    return center_z - height / 2.0, center_z + height / 2.0


# ======================================================
# HATCH GENERATION
# ======================================================

def generate_hatch_lines(polygon: Polygon, hatch_distance: float) -> List[Tuple]:
    """Generate horizontal hatch lines for a Shapely polygon."""
    if polygon.is_empty or hatch_distance <= 0:
        return []

    minx, miny, maxx, maxy = polygon.bounds
    y_start = math.ceil(miny / hatch_distance) * hatch_distance
    segments = []
    y = y_start

    while y <= maxy:
        scan_line = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        intersection = polygon.intersection(scan_line)
        if not intersection.is_empty:
            if intersection.geom_type == 'LineString':
                coords = list(intersection.coords)
                segments.append((coords[0][0], coords[0][1],
                                 coords[-1][0], coords[-1][1]))
            elif intersection.geom_type == 'MultiLineString':
                for line in intersection.geoms:
                    coords = list(line.coords)
                    segments.append((coords[0][0], coords[0][1],
                                     coords[-1][0], coords[-1][1]))
        y += hatch_distance

    return segments


def generate_contour_lines(polygon: Polygon, hatch_distance: float, contour_count: int) -> List[Tuple]:
    """Generate contour offset lines around a polygon boundary."""
    segments = []
    for i in range(1, contour_count + 1):
        # Inward offset
        offset = i * hatch_distance
        contour_geom = polygon.buffer(-offset)
        if contour_geom.is_empty:
            break
        # Extract exterior coords
        if contour_geom.geom_type == 'Polygon':
            coords = list(contour_geom.exterior.coords)
        elif contour_geom.geom_type == 'MultiPolygon':
            coords = []
            for part in contour_geom.geoms:
                coords.extend(list(part.exterior.coords))
        else:
            continue
        for j in range(len(coords) - 1):
            segments.append((coords[j][0], coords[j][1],
                             coords[j + 1][0], coords[j + 1][1]))
    return segments


# ======================================================
# TRANSITION BUFFER
# ======================================================

def _get_bbox(primitive: Dict) -> Optional[Polygon]:
    """Get the XY bounding box of a primitive as a shapely box."""
    footprint = get_primitive_footprint(primitive)
    if footprint is None or footprint.is_empty:
        return None
    minx, miny, maxx, maxy = footprint.bounds
    return shapely_box(minx, miny, maxx, maxy)


def apply_transition_buffer(primitives: List[Dict]) -> List[Dict]:
    """
    Conservative transition buffer:
    For pairs of primitives whose XY bounding boxes overlap (or are within 1 um)
    AND whose hatch values differ by more than 0.1 um:
      -> set both hatch values to min(hatch_A, hatch_B) so the overlap
         region is always printed with the denser pass first.

    Mutates 'local_parameters' in-place (on a deep copy passed in).
    """
    n = len(primitives)
    for i in range(n):
        lp_i = primitives[i].get("local_parameters")
        if not lp_i:
            continue
        bbox_i = _get_bbox(primitives[i])
        if bbox_i is None:
            continue

        for j in range(i + 1, n):
            lp_j = primitives[j].get("local_parameters")
            if not lp_j:
                continue
            bbox_j = _get_bbox(primitives[j])
            if bbox_j is None:
                continue

            # Check if Z ranges overlap
            zi_min, zi_max = primitive_z_range(primitives[i])
            zj_min, zj_max = primitive_z_range(primitives[j])
            if zi_max < zj_min or zj_max < zi_min:
                continue  # no Z overlap

            # Check if XY bboxes are within 1 um of each other
            if bbox_i.distance(bbox_j) > 1.0:
                continue

            hatch_i = lp_i["hatch_spacing_um"]
            hatch_j = lp_j["hatch_spacing_um"]

            if abs(hatch_i - hatch_j) > 0.1:
                denser_hatch = min(hatch_i, hatch_j)
                lp_i["hatch_spacing_um"] = denser_hatch
                lp_j["hatch_spacing_um"] = denser_hatch

    return primitives


# ======================================================
# PER-PRIMITIVE LAYER GENERATION
# ======================================================

def process_single_primitive(primitive: Dict) -> Dict[float, List[Dict]]:
    """
    Slice and hatch a single primitive using its local_parameters.

    Returns: dict mapping quantized z_key -> list of segment dicts
    """
    local_params = primitive.get("local_parameters", {})
    slice_spacing = local_params.get("slice_spacing_um", 0.60)
    hatch_spacing = local_params.get("hatch_spacing_um", 0.35)
    contour_count = int(local_params.get("solid_contour_count", 2))

    z_min, z_max = primitive_z_range(primitive)
    footprint = get_primitive_footprint(primitive)

    if footprint is None or footprint.is_empty:
        return {}

    # Small buffer to ensure touching edges connect properly
    footprint = footprint.buffer(0.01)

    layers: Dict[float, List[Dict]] = {}

    # Snap start Z to the primitive's Z grid
    z_start = math.floor(z_min / slice_spacing) * slice_spacing
    z = z_start

    while z <= z_max + 1e-9:
        # Only emit layers that are within this primitive's extent
        if z < z_min - 1e-9:
            z += slice_spacing
            continue

        # Quantize Z to global grid to allow merging across primitives
        z_key = round(round(z / Z_BIN) * Z_BIN, 6)

        segments: List[Dict] = []

        # Contour lines (shell)
        if contour_count > 0:
            contour_segs = generate_contour_lines(footprint, hatch_spacing, contour_count)
            for x1, y1, x2, y2 in contour_segs:
                segments.append({
                    "start": [round(x1, 6), round(y1, 6)],
                    "end":   [round(x2, 6), round(y2, 6)],
                    "type":  "contour",
                })

        # Hatch fill (core)
        # Erode by contour shell thickness before hatching interior
        interior = footprint.buffer(-contour_count * hatch_spacing) if contour_count > 0 else footprint
        if interior is not None and not interior.is_empty:
            hatch_segs = generate_hatch_lines(interior, hatch_spacing)
            for x1, y1, x2, y2 in hatch_segs:
                segments.append({
                    "start": [round(x1, 6), round(y1, 6)],
                    "end":   [round(x2, 6), round(y2, 6)],
                    "type":  "hatch",
                })

        if segments:
            if z_key not in layers:
                layers[z_key] = []
            layers[z_key].extend(segments)

        z += slice_spacing

    return layers


# ======================================================
# MAIN V2 API
# ======================================================

def generate_endpoint_json_v2(enriched_data: Dict, print_params: Dict) -> Dict:
    """
    Generate endpoint JSON from enriched_reduced.json.

    Args:
        enriched_data: Output from manufacturing_agent.enrich_primitives()
                       Each primitive must have local_parameters.
        print_params:  Global print parameters (used for exposure metadata only).

    Returns:
        Endpoint dict with structure:
        {
            "job_name": str,
            "layers": [{"z_um": float, "segments": [...]}]
        }
    """
    print("[Endpoint Generator V2 - Per-Primitive Local Parameters]")

    primitives = enriched_data.get("primitives", [])
    print(f"  Processing {len(primitives)} primitives...")

    # Apply transition buffer (in-place on a copy)
    primitives_buffered = apply_transition_buffer(copy.deepcopy(primitives))

    # Accumulate all layers across all primitives
    all_layers: Dict[float, List[Dict]] = {}

    for idx, primitive in enumerate(primitives_buffered):
        prim_layers = process_single_primitive(primitive)
        for z_key, segs in prim_layers.items():
            if z_key not in all_layers:
                all_layers[z_key] = []
            all_layers[z_key].extend(segs)

    # Sort each layer's segments for determinism
    for z_key in all_layers:
        all_layers[z_key].sort(
            key=lambda s: (s["start"][1], s["start"][0])
        )

    # Format output
    layers_list = [
        {"z_um": z, "segments": segs}
        for z, segs in sorted(all_layers.items())
    ]

    result = {
        "job_name": enriched_data.get("job_name", "unnamed"),
        "layers": layers_list,
        "metadata": {
            "z_bin_um": Z_BIN,
            "num_layers": len(layers_list),
            "num_primitives_processed": len(primitives),
        }
    }

    print(f"  [OK] Generated {len(layers_list)} layers across {len(primitives)} primitives")
    return result


if __name__ == "__main__":
    print("endpoint_generator_v2 - Per-Primitive Mode")
    print("Use via NamedObjectAgent or manufacturing_agent pipeline.")
