"""
load_observed_features.py - Observed Feature Loader

Validates and loads inspection/observed_features.json, which is produced by
the deterministic segmentation pipeline in the Render_to_SEM system.

INPUT:  <iter_folder>/inspection/observed_features.json
OUTPUT: validated Python dict (returned in memory, no new file written)

This module contains NO ML logic. It only:
  1. Verifies the file exists
  2. Parses the JSON
  3. Validates required top-level fields and types
  4. Returns the validated dict for downstream use

Expected schema (minimum required fields):
{
  "objects_detected":   <int>   - number of solid objects segmented from SEM image
  "holes_detected":     <int>   - number of hole/void features detected
  "shape_descriptors": {
      "area":           <number> - total segmented area in pixels
      "convexity":      <number> - convex hull area ratio [0, 1]
      "circularity":    <number> - 4π·area/perimeter² [0, 1]
  }
}

Additional fields in observed_features.json are allowed and passed through
unchanged so that future segmentation stages can add richer descriptors
without breaking this loader.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ======================================================
# VALIDATION HELPERS
# ======================================================

def _require_int_field(data: Dict, key: str, errors: List[str]) -> None:
    if key not in data:
        errors.append(f"Missing required field: '{key}'")
    elif not isinstance(data[key], int):
        errors.append(
            f"Field '{key}' must be int, got {type(data[key]).__name__}"
        )
    elif data[key] < 0:
        errors.append(f"Field '{key}' must be >= 0, got {data[key]}")


def _require_numeric_field(data: Dict, key: str, errors: List[str],
                           lo: float = None, hi: float = None) -> None:
    if key not in data:
        errors.append(f"Missing required field in shape_descriptors: '{key}'")
        return
    val = data[key]
    if not isinstance(val, (int, float)):
        errors.append(
            f"shape_descriptors.{key} must be numeric, "
            f"got {type(val).__name__}"
        )
        return
    if lo is not None and val < lo:
        errors.append(
            f"shape_descriptors.{key} must be >= {lo}, got {val}"
        )
    if hi is not None and val > hi:
        errors.append(
            f"shape_descriptors.{key} must be <= {hi}, got {val}"
        )


def _validate_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the observed_features dict against the required schema.

    Returns:
        (is_valid, errors_list)
    """
    errors: List[str] = []

    # Top-level required integer fields
    _require_int_field(data, "objects_detected", errors)
    _require_int_field(data, "holes_detected",   errors)

    # shape_descriptors block
    if "shape_descriptors" not in data:
        errors.append("Missing required block: 'shape_descriptors'")
    elif not isinstance(data["shape_descriptors"], dict):
        errors.append(
            "'shape_descriptors' must be an object, "
            f"got {type(data['shape_descriptors']).__name__}"
        )
    else:
        desc = data["shape_descriptors"]
        _require_numeric_field(desc, "area",        errors, lo=0.0)
        _require_numeric_field(desc, "convexity",   errors, lo=0.0, hi=1.0)
        _require_numeric_field(desc, "circularity", errors, lo=0.0, hi=1.0)

    is_valid = len(errors) == 0
    return is_valid, errors


# ======================================================
# PUBLIC API
# ======================================================

def validate_observed_features(data: Dict[str, Any]) -> None:
    """
    Validate observed_features dict. Raises ValueError on schema violations.

    Args:
        data: Parsed content of observed_features.json

    Raises:
        ValueError: if any required field is missing or has the wrong type/range
    """
    is_valid, errors = _validate_schema(data)
    if not is_valid:
        formatted = "\n  ".join(errors)
        raise ValueError(
            f"observed_features.json failed validation "
            f"({len(errors)} error(s)):\n  {formatted}"
        )


def load_observed_features(observed_path: Path) -> Dict[str, Any]:
    """
    Load and validate observed_features.json from disk.

    Args:
        observed_path: Absolute path to inspection/observed_features.json

    Returns:
        Validated dict containing observed SEM features.

    Raises:
        FileNotFoundError: if the file does not exist
        json.JSONDecodeError: if the file is not valid JSON
        ValueError: if the JSON content fails schema validation
    """
    if not observed_path.exists():
        raise FileNotFoundError(
            f"observed_features.json not found at: {observed_path}\n"
            "This file must be produced by the deterministic segmentation "
            "pipeline before running SEM analysis."
        )

    with observed_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"observed_features.json must contain a JSON object, "
            f"got {type(data).__name__}"
        )

    validate_observed_features(data)
    return data


def run(iter_folder: Path) -> Dict[str, Any]:
    """
    Convenience wrapper: load observed_features from the standard location
    inside an iteration folder.

    Reads:   <iter_folder>/inspection/observed_features.json

    Returns: validated observed features dict
    """
    print("[SEM] loading observed features")

    observed_path = iter_folder / "inspection" / "observed_features.json"
    data = load_observed_features(observed_path)

    print(
        f"[SEM] observed features loaded — "
        f"objects_detected={data['objects_detected']}, "
        f"holes_detected={data['holes_detected']}"
    )
    return data


# ======================================================
# CLI ENTRY (for standalone testing)
# ======================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python load_observed_features.py <iter_folder>")
        sys.exit(1)

    result = run(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
