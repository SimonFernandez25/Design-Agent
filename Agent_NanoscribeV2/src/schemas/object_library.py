"""
Object Library Schema - Named Object Definitions

Named objects are reusable geometric definitions with identity.
They may contain primitives (geometry type) or reference other objects (composite type).

Key Rule:
- Only 'geometry' objects contain primitives
- 'composite' objects reference other objects
- This prevents recursion errors and ambiguity
"""

from typing import Dict, List, Optional
from schemas.primitives import PRIMITIVE_SCHEMA, validate_primitive


# Object types
OBJECT_TYPES = ["geometry", "composite"]


GEOMETRY_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "const": "geometry",
            "description": "Indicates this object contains raw primitives"
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of this object"
        },
        "components": {
            "type": "array",
            "items": PRIMITIVE_SCHEMA,
            "description": "Array of geometric primitives that compose this object"
        }
    },
    "required": ["type", "components"],
    "additionalProperties": False
}


COMPOSITE_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "const": "composite",
            "description": "Indicates this object references another object"
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of this object"
        },
        "uses": {
            "type": "string",
            "description": "Name of the object this composite references"
        },
        "repeat": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "minimum": 1, "default": 1},
                "y": {"type": "integer", "minimum": 1, "default": 1},
                "z": {"type": "integer", "minimum": 1, "default": 1}
            },
            "description": "Number of repetitions in each axis"
        },
        "spacing_um": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Spacing in X axis (micrometers)"},
                "y": {"type": "number", "description": "Spacing in Y axis (micrometers)"},
                "z": {"type": "number", "description": "Spacing in Z axis (micrometers)"}
            },
            "description": "Spacing between repetitions in each axis"
        },
        "transform": {
            "type": "object",
            "properties": {
                "translate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Translation offset [dx, dy, dz]"
                },
                "rotate_z_deg": {
                    "type": "number",
                    "description": "Rotation around Z axis in degrees"
                }
            },
            "description": "Optional transform to apply to the referenced object"
        }
    },
    "required": ["type", "uses"],
    "additionalProperties": False
}


OBJECT_LIBRARY_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    GEOMETRY_OBJECT_SCHEMA,
                    COMPOSITE_OBJECT_SCHEMA
                ]
            },
            "description": "Dictionary of named objects"
        }
    },
    "required": ["objects"],
    "additionalProperties": False
}


def validate_object(name: str, obj: Dict, library: Dict) -> List[str]:
    """
    Validate a single object definition.
    
    Args:
        name: Object name
        obj: Object definition
        library: Full object library (for reference validation)
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check type field
    obj_type = obj.get("type")
    if obj_type not in OBJECT_TYPES:
        errors.append(f"Object '{name}': Invalid type '{obj_type}'")
        return errors
    
    if obj_type == "geometry":
        # Validate components
        components = obj.get("components", [])
        if not components:
            errors.append(f"Object '{name}': geometry object must have at least one component")
        
        for i, comp in enumerate(components):
            comp_errors = validate_primitive(comp)
            for err in comp_errors:
                errors.append(f"Object '{name}', component[{i}]: {err}")
    
    elif obj_type == "composite":
        # Validate reference
        uses = obj.get("uses")
        if not uses:
            errors.append(f"Object '{name}': composite object must specify 'uses'")
        elif uses not in library.get("objects", {}):
            errors.append(f"Object '{name}': references undefined object '{uses}'")
        
        # Validate repeat (if provided)
        repeat = obj.get("repeat", {})
        for axis in ["x", "y", "z"]:
            val = repeat.get(axis, 1)
            if not isinstance(val, int) or val < 1:
                errors.append(f"Object '{name}': repeat.{axis} must be positive integer")
        
        # Validate spacing (required if repeat > 1)
        spacing = obj.get("spacing_um", {})
        for axis in ["x", "y", "z"]:
            if repeat.get(axis, 1) > 1 and axis not in spacing:
                errors.append(f"Object '{name}': spacing_um.{axis} required when repeat.{axis} > 1")
    
    return errors


def validate_object_library(library: Dict) -> List[str]:
    """
    Validate an entire object library.
    
    Args:
        library: Object library dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    objects = library.get("objects", {})
    if not objects:
        errors.append("Object library must contain at least one object")
        return errors
    
    # Check for circular references
    def check_circular(name: str, visited: set) -> Optional[str]:
        if name in visited:
            return f"Circular reference detected: {' -> '.join(visited)} -> {name}"
        
        obj = objects.get(name)
        if not obj:
            return None
        
        if obj.get("type") == "composite":
            uses = obj.get("uses")
            if uses:
                visited.add(name)
                result = check_circular(uses, visited)
                visited.remove(name)
                return result
        
        return None
    
    # Validate each object
    for name, obj in objects.items():
        obj_errors = validate_object(name, obj, library)
        errors.extend(obj_errors)
        
        # Check for circular references
        circular_error = check_circular(name, set())
        if circular_error:
            errors.append(circular_error)
    
    return errors


def get_object_dependency_order(library: Dict) -> List[str]:
    """
    Get objects in dependency order (dependencies first).
    
    This is useful for reduction - ensures referenced objects
    are processed before composites that use them.
    
    Args:
        library: Object library dictionary
        
    Returns:
        List of object names in dependency order
    """
    objects = library.get("objects", {})
    
    # Build dependency graph
    dependencies = {}
    for name, obj in objects.items():
        if obj.get("type") == "composite":
            dependencies[name] = [obj.get("uses")]
        else:
            dependencies[name] = []
    
    # Topological sort
    ordered = []
    visited = set()
    
    def visit(name: str):
        if name in visited:
            return
        visited.add(name)
        for dep in dependencies.get(name, []):
            if dep:
                visit(dep)
        ordered.append(name)
    
    for name in objects:
        visit(name)
    
    return ordered
