"""
manufacturing_agent.py - Primitive-Aware Manufacturing Agent

Deterministic, no LLM. Converts reduced.json into enriched_reduced.json by:
  1. Extracting geometric metrics per primitive
  2. Computing a continuous risk score via a deterministic surrogate
  3. Assigning local print parameters (hatch, slice, contours) per primitive
  4. Writing a manufacturing log for future training data

This module is inserted between:
    reduction_engine  ->  [THIS MODULE]  ->  endpoint_generator_v2
"""

import json
import math
import copy
from pathlib import Path
from typing import Dict, List, Any

# ======================================================
# CONSTANTS
# ======================================================

EPS = 1e-6

# Baseline manufacturing assumptions (used only for scoring)
BASE_HATCH_UM = 0.35
BASE_SLICE_UM = 0.60
BASE_CONTOURS = 2

# Scoring weights (must sum to 1.0)
W_XY    = 0.35
W_Z     = 0.25
W_SHELL = 0.20
W_SLEND = 0.20

# Parameter clamp bounds
HATCH_MIN, HATCH_MAX = 0.20, 0.50
SLICE_MIN, SLICE_MAX = 0.40, 0.70
CONTOURS_MIN, CONTOURS_MAX = 1, 4


# ======================================================
# HELPERS
# ======================================================

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ======================================================
# STAGE A: METRIC EXTRACTION
# ======================================================

def _extract_dims(primitive: Dict) -> tuple:
    """
    Return (height_um, width_um, depth_um) from a primitive's dimensions dict.

    Dimension key conventions (after reduction_engine normalization):
      box       -> x_um, y_um, z_um
      cylinder  -> diameter_um, height_um (z_um fallback)
    """
    dims = primitive.get("dimensions", {})
    prim_type = primitive.get("type", "box")

    if prim_type == "cylinder":
        diameter = dims.get("diameter_um", dims.get("radius_um", EPS) * 2)
        height = dims.get("height_um", dims.get("z_um", EPS))
        width = diameter
        depth = diameter
    else:
        # Box (already normalized by reduction_engine to x_um, y_um, z_um)
        width  = dims.get("x_um", dims.get("width_um",  dims.get("length_um", EPS)))
        depth  = dims.get("y_um", dims.get("depth_um",  EPS))
        height = dims.get("z_um", dims.get("height_um", EPS))

    return (
        max(float(height), EPS),
        max(float(width),  EPS),
        max(float(depth),  EPS),
    )


def compute_metrics(primitive: Dict) -> Dict[str, Any]:
    """Compute and return a metrics dict for a single primitive."""
    height_um, width_um, depth_um = _extract_dims(primitive)
    prim_type = primitive.get("type", "box")

    thickness_um  = min(width_um, depth_um)
    max_dim_um    = max(width_um, depth_um, height_um)
    slenderness   = height_um / max(thickness_um, EPS)

    metrics: Dict[str, Any] = {
        "height_um":    round(height_um, 6),
        "width_um":     round(width_um, 6),
        "depth_um":     round(depth_um, 6),
        "thickness_um": round(thickness_um, 6),
        "slenderness":  round(slenderness, 6),
        "max_dim_um":   round(max_dim_um, 6),
    }

    if prim_type == "cylinder":
        radius_um = max(width_um, depth_um) / 2.0
        metrics["radius_um"] = round(radius_um, 6)

    return metrics


# ======================================================
# STAGE B: RISK SCORING
# ======================================================

def compute_risk(metrics: Dict[str, Any]) -> tuple:
    """
    Compute a continuous risk score in [0, 1] and return (risk, risk_factors).

    Higher risk = harder to print reliably = needs denser bonding (smaller hatch/slice).
    """
    thickness = metrics["thickness_um"]
    height    = metrics["height_um"]

    XY_lines   = thickness / BASE_HATCH_UM
    Z_layers   = height    / BASE_SLICE_UM
    Shell_ratio = (BASE_CONTOURS * BASE_HATCH_UM) / max(thickness, EPS)
    Slenderness = height / max(thickness, EPS)

    S = (
        W_XY    * (XY_lines    / 3.0)
      + W_Z     * (Z_layers    / 25.0)
      - W_SHELL * (Shell_ratio / 0.5)
      - W_SLEND * (Slenderness / 12.0)
    )

    risk = 1.0 - _sigmoid(S)

    risk_factors = {
        "XY_lines":    round(XY_lines,    6),
        "Z_layers":    round(Z_layers,    6),
        "Shell_ratio": round(Shell_ratio, 6),
        "Slenderness": round(Slenderness, 6),
        "S_raw":       round(S,           6),
    }

    return round(risk, 6), risk_factors


# ======================================================
# STAGE C: LOCAL PARAMETER ASSIGNMENT
# ======================================================

def assign_local_parameters(risk: float) -> Dict[str, Any]:
    """
    Map a risk score in [0, 1] to local print parameters.

    Higher risk -> denser bonding (smaller hatch, smaller slice, more contours).
    """
    local_hatch    = _clamp(BASE_HATCH_UM * (1.0 - 0.30 * risk), HATCH_MIN, HATCH_MAX)
    local_slice    = _clamp(BASE_SLICE_UM * (1.0 - 0.30 * risk), SLICE_MIN, SLICE_MAX)
    local_contours = int(_clamp(BASE_CONTOURS + round(2.0 * risk), CONTOURS_MIN, CONTOURS_MAX))

    return {
        "hatch_spacing_um":  round(local_hatch, 6),
        "slice_spacing_um":  round(local_slice, 6),
        "solid_contour_count": local_contours,
    }


# ======================================================
# MAIN ENTRY POINT
# ======================================================

def enrich_primitives(
    reduced_data: Dict,
    output_dir: Path = None
) -> Dict:
    """
    Accept a reduced.json dict and return an enriched_reduced.json dict.

    For each primitive, adds:
      - primitive["metrics"]
      - primitive["risk"]
      - primitive["risk_factors"]
      - primitive["local_parameters"]

    Also writes:
      - <output_dir>/enriched_reduced.json
      - <output_dir>/logs/manufacturing_log.json

    Args:
        reduced_data: Output from reduction_engine.reduce_assembly()
        output_dir:   Job output directory (Path). If None, file writing is skipped.

    Returns:
        Enriched copy of reduced_data.
    """
    enriched = copy.deepcopy(reduced_data)
    primitives = enriched.get("primitives", [])

    log_entries = []

    print(f"[Manufacturing Agent] Processing {len(primitives)} primitives...")

    for idx, primitive in enumerate(primitives):
        # A: Metrics
        metrics = compute_metrics(primitive)
        primitive["metrics"] = metrics

        # B: Risk
        risk, risk_factors = compute_risk(metrics)
        primitive["risk"]         = risk
        primitive["risk_factors"] = risk_factors

        # C: Local parameters
        local_params = assign_local_parameters(risk)
        primitive["local_parameters"] = local_params

        # Log entry
        log_entries.append({
            "primitive_id":    idx,
            "type":            primitive.get("type"),
            "center":          primitive.get("center"),
            "metrics":         metrics,
            "risk":            risk,
            "risk_factors":    risk_factors,
            "local_parameters": local_params,
        })

    print(f"[Manufacturing Agent] Enriched {len(primitives)} primitives.")

    # Write outputs if output_dir given
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # enriched_reduced.json
        enriched_path = output_dir / "enriched_reduced.json"
        with open(enriched_path, "w") as f:
            json.dump(enriched, f, indent=2)
        print(f"[Manufacturing Agent] Saved: {enriched_path.name}")

        # logs/manufacturing_log.json
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "manufacturing_log.json"
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)
        print(f"[Manufacturing Agent] Saved: logs/{log_path.name}")

    return enriched


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python manufacturing_agent.py <reduced.json> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent

    with open(input_path) as f:
        reduced = json.load(f)

    enriched = enrich_primitives(reduced, out_dir)

    print(f"\nSample primitive (first):")
    if enriched["primitives"]:
        p = enriched["primitives"][0]
        print(f"  risk        : {p['risk']}")
        print(f"  metrics     : {p['metrics']}")
        print(f"  local_params: {p['local_parameters']}")
