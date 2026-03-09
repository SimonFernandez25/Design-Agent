"""
extract_expected_features.py - Expected Feature Extractor

Reads enriched_reduced.json (already produced by the fabrication pipeline)
and generates a simplified structural description of the expected geometry.

INPUT:  <iter_folder>/enriched_reduced.json   (read-only, never modified)
OUTPUT: <iter_folder>/analysis/expected_features.json

This module performs NO geometry inference, NO CAD reconstruction,
and ignores all manufacturing metadata (risk, local_parameters, etc.).
It only extracts the geometric primitives already present in the file.

Supported primitive types:
    cylinder  -> mapped to expected_features "objects" list
    box       -> mapped to expected_features "objects" list
    cone      -> detected as cylindrical with tapered geometry
    pyramid   -> detected as box-type with tapered geometry

Holes are inferred from objects whose geometry implies a void
(e.g., hollowed cylinders flagged via the 'hole_axis' annotation if present).
In V2 the pipeline does not explicitly tag holes, so the holes list will
be populated only when a primitive explicitly carries hole metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


# ======================================================
# PRIMITIVE MAPPINGS
# ======================================================

# Map enriched_reduced primitive types to SEM feature primitive labels
_CYLINDER_TYPES = {"cylinder", "cone"}
_BOX_TYPES      = {"box", "pyramid"}


def _extract_objects(primitives: List[Dict]) -> List[Dict]:
    """
    Convert flat primitive list from enriched_reduced.json into the
    expected_features 'objects' schema.

    Only uses: type, center, dimensions.
    Ignores:   metrics, risk, risk_factors, local_parameters.
    """
    objects = []

    for idx, prim in enumerate(primitives):
        ptype  = prim.get("type", "").lower()
        center = prim.get("center", [0.0, 0.0, 0.0])
        dims   = prim.get("dimensions", {})

        if ptype in _CYLINDER_TYPES:
            # Cylinder or cone: use diameter/radius and height
            diameter = dims.get("diameter_um", dims.get("radius_um", 0.0) * 2)
            radius   = diameter / 2.0
            height   = dims.get("height_um", dims.get("z_um", 0.0))

            objects.append({
                "object_id": idx,
                "primitive": "cylinder",
                "center": [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                ],
                "radius": float(radius),
                "height": float(height),
            })

        elif ptype in _BOX_TYPES:
            # Box or pyramid: use explicit x/y/z dimensions
            # The reduction engine normalises to x_um, y_um, z_um
            width  = dims.get("x_um",        dims.get("base_width_um", 0.0))
            depth  = dims.get("y_um",        dims.get("base_width_um", 0.0))
            height = dims.get("z_um",        dims.get("height_um",     0.0))

            objects.append({
                "object_id": idx,
                "primitive": "box",
                "center": [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                ],
                "width": float(width),
                "depth": float(depth),
                "height": float(height),
            })

        else:
            # Unknown primitive type: record it verbatim so nothing is lost
            objects.append({
                "object_id": idx,
                "primitive": ptype or "unknown",
                "center": [
                    float(center[0]) if len(center) > 0 else 0.0,
                    float(center[1]) if len(center) > 1 else 0.0,
                    float(center[2]) if len(center) > 2 else 0.0,
                ],
                "dimensions_raw": dims,
            })

    return objects


def _extract_holes(primitives: List[Dict]) -> List[Dict]:
    """
    Populate the 'holes' list from primitives that carry explicit hole metadata.

    The V2 fabrication pipeline does not currently emit hole annotations,
    so this list will be empty for standard pipeline outputs.
    If a future stage adds a 'hole_axis' or 'is_void' field, it is handled here.
    """
    holes = []
    hole_id = 0

    for prim in primitives:
        # Only act on explicit void/hole annotations; never infer
        if not prim.get("is_void", False) and "hole_axis" not in prim:
            continue

        ptype = prim.get("type", "").lower()
        dims  = prim.get("dimensions", {})
        axis  = prim.get("hole_axis", [0.0, 0.0, 1.0])

        if ptype in _CYLINDER_TYPES:
            diameter = dims.get("diameter_um", dims.get("radius_um", 0.0) * 2)
            holes.append({
                "hole_id":  f"h{hole_id}",
                "type":     "cylindrical",
                "radius":   float(diameter / 2.0),
                "axis":     [float(v) for v in axis],
            })
            hole_id += 1

        elif ptype in _BOX_TYPES:
            width  = dims.get("x_um", dims.get("base_width_um", 0.0))
            depth  = dims.get("y_um", dims.get("base_width_um", 0.0))
            holes.append({
                "hole_id":  f"h{hole_id}",
                "type":     "rectangular",
                "width":    float(width),
                "depth":    float(depth),
                "axis":     [float(v) for v in axis],
            })
            hole_id += 1

    return holes


# ======================================================
# PUBLIC API
# ======================================================

def extract_expected_features(enriched_reduced: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse enriched_reduced.json content and return expected_features dict.

    Args:
        enriched_reduced: Parsed content of enriched_reduced.json

    Returns:
        expected_features dict conforming to the SEM schema:
        {
            "objects": [...],
            "holes":   [...]
        }
    """
    primitives = enriched_reduced.get("primitives", [])

    if not isinstance(primitives, list):
        raise ValueError(
            f"enriched_reduced.json 'primitives' must be a list, "
            f"got {type(primitives).__name__}"
        )

    objects = _extract_objects(primitives)
    holes   = _extract_holes(primitives)

    return {
        "objects": objects,
        "holes":   holes,
    }


def run(iter_folder: Path) -> Path:
    """
    Full extraction pass for one iteration folder.

    Reads:   <iter_folder>/enriched_reduced.json
    Writes:  <iter_folder>/analysis/expected_features.json

    Returns: Path to the written expected_features.json
    """
    print("[SEM] extracting expected features")

    enriched_path = iter_folder / "enriched_reduced.json"
    if not enriched_path.exists():
        raise FileNotFoundError(
            f"enriched_reduced.json not found at: {enriched_path}\n"
            "Ensure the fabrication pipeline has completed for this iteration."
        )

    with enriched_path.open("r", encoding="utf-8") as fh:
        enriched_reduced = json.load(fh)

    expected = extract_expected_features(enriched_reduced)

    # Write output into analysis/ subfolder
    analysis_dir = iter_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    out_path = analysis_dir / "expected_features.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(expected, fh, indent=2)

    print(f"[SEM] expected features written → {out_path}")
    return out_path


# ======================================================
# CLI ENTRY (for standalone testing)
# ======================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_expected_features.py <iter_folder>")
        sys.exit(1)

    run(Path(sys.argv[1]))
