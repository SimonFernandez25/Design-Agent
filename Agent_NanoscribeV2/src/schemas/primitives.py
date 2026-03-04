"""
Primitive Schema - Native Geometry Types for Executor Boundary

Only native primitives (box, cylinder) should exist at the executor boundary.
Derived primitives (pyramid, cone) are expanded during reduction.
"""

from typing import Dict, List

# Native primitive types - the ONLY types allowed at executor boundary
NATIVE_PRIMITIVE_TYPES = ["box", "cylinder"]

# Derived primitive types - expanded to native during reduction
DERIVED_PRIMITIVE_TYPES = ["pyramid", "cone", "tapered_cylinder"]

# All primitive types (used during object definition)
ALL_PRIMITIVE_TYPES = NATIVE_PRIMITIVE_TYPES + DERIVED_PRIMITIVE_TYPES


PRIMITIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ALL_PRIMITIVE_TYPES,
            "description": "Primitive geometry type"
        },
        "center": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
            "description": "Center point [x, y, z] in micrometers"
        },
        "dimensions": {
            "type": "object",
            "description": "Type-specific dimensions in micrometers",
            "additionalProperties": True
        },
        "construction": {
            "type": "object",
            "description": "Optional construction metadata for derived primitives",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["stacked_boxes", "stacked_cylinders"],
                    "description": "How to decompose this primitive"
                },
                "layers": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of layers for decomposition"
                }
            },
            "required": ["method", "layers"]
        }
    },
    "required": ["type", "center", "dimensions"],
    "additionalProperties": False
}


# Box-specific dimensions
BOX_DIMENSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "width_um": {"type": "number", "description": "X dimension"},
        "depth_um": {"type": "number", "description": "Y dimension"},
        "height_um": {"type": "number", "description": "Z dimension"}
    },
    "required": ["width_um", "depth_um", "height_um"]
}


# Cylinder-specific dimensions
CYLINDER_DIMENSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "diameter_um": {"type": "number", "description": "XY diameter"},
        "height_um": {"type": "number", "description": "Z dimension"}
    },
    "required": ["diameter_um", "height_um"]
}


# Pyramid-specific dimensions (derived, will be expanded)
PYRAMID_DIMENSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "base_width_um": {"type": "number", "description": "Base X/Y dimension"},
        "height_um": {"type": "number", "description": "Z dimension"},
        "top_width_um": {"type": "number", "description": "Top X/Y dimension (0 for pointed)"}
    },
    "required": ["base_width_um", "height_um"]
}


# Cone-specific dimensions (derived, will be expanded)
CONE_DIMENSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "base_diameter_um": {"type": "number", "description": "Base diameter"},
        "height_um": {"type": "number", "description": "Z dimension"},
        "top_diameter_um": {"type": "number", "description": "Top diameter (0 for pointed)"}
    },
    "required": ["base_diameter_um", "height_um"]
}


def validate_primitive(primitive: Dict) -> List[str]:
    """
    Validate a primitive against schema rules.
    
    Args:
        primitive: Primitive dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    for field in ["type", "center", "dimensions"]:
        if field not in primitive:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Check type
    if primitive["type"] not in ALL_PRIMITIVE_TYPES:
        errors.append(f"Invalid primitive type: {primitive['type']}")
    
    # Check center
    center = primitive.get("center", [])
    if not isinstance(center, list) or len(center) != 3:
        errors.append("Center must be [x, y, z] array")
    
    # Check derived primitives have construction metadata
    if primitive["type"] in DERIVED_PRIMITIVE_TYPES:
        if "construction" not in primitive:
            errors.append(f"Derived primitive '{primitive['type']}' requires 'construction' metadata")
    
    return errors


def is_native_primitive(primitive: Dict) -> bool:
    """Check if a primitive is a native type (no expansion needed)."""
    return primitive.get("type") in NATIVE_PRIMITIVE_TYPES


def is_derived_primitive(primitive: Dict) -> bool:
    """Check if a primitive is a derived type (needs expansion)."""
    return primitive.get("type") in DERIVED_PRIMITIVE_TYPES
