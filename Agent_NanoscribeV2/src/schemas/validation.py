"""
V2 Structural Validation - Hard enforcement of named object architecture.

This module provides the first line of defense against legacy v1 outputs.
All validation here runs BEFORE schema validation and fails LOUDLY.

Key guarantee: If any of these checks pass, the output is structurally v2-compliant.
"""

from typing import Dict, List, Set


# Legacy fields that are absolutely forbidden anywhere in the output
FORBIDDEN_FIELDS = {"unit_cell", "global_info"}

# Required top-level fields for v2 compliance
REQUIRED_FIELDS = {"objects", "assembly"}


def v2_structural_gate(design: dict) -> None:
    """
    Fail immediately if legacy fields detected anywhere in the design.
    
    This is a recursive check - legacy fields hidden in nested objects
    will also be caught.
    
    Args:
        design: The LLM output to validate
        
    Raises:
        ValueError: If any v2 violation is detected
    """
    print(f"[GATE] Running v2 structural gate on keys: {list(design.keys())}")
    
    # Check for forbidden fields recursively
    def walk(obj, path="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in FORBIDDEN_FIELDS:
                    print(f"[GATE] CAUGHT LEGACY FIELD: {k} at {path}")
                    raise ValueError(
                        f"FATAL: Legacy v1 field '{k}' detected at {path}"
                    )
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
    
    walk(design)
    
    # Check for forbidden top-level primitives
    if "primitives" in design:
        raise ValueError(
            "FATAL: Top-level 'primitives' field is forbidden in v2. "
            "Primitives must be inside geometry objects."
        )
    
    # Check required fields exist
    missing = REQUIRED_FIELDS - design.keys()
    if missing:
        raise ValueError(f"FATAL: Missing required v2 fields: {missing}")
    
    # Ensure objects is not empty
    if not design.get("objects"):
        raise ValueError("FATAL: 'objects' dictionary cannot be empty")


def validate_assembly_references(design: dict) -> List[str]:
    """
    Validate that all objects referenced in assembly exist in the object library.
    
    Args:
        design: The design to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    defined = set(design.get("objects", {}).keys())
    assembly = design.get("assembly", {})
    
    # Check default_object
    default_obj = assembly.get("default_object")
    if default_obj and default_obj not in defined:
        errors.append(f"Assembly default_object '{default_obj}' not defined in objects")
    
    # Check mapping references
    mapping = assembly.get("mapping", [])
    for row_idx, row in enumerate(mapping):
        for col_idx, obj_name in enumerate(row):
            if obj_name and obj_name not in defined:
                errors.append(
                    f"Assembly mapping[{row_idx}][{col_idx}] references "
                    f"undefined object '{obj_name}'"
                )
    
    # Check explicit placements
    placements = assembly.get("placements", [])
    for idx, placement in enumerate(placements):
        obj_name = placement.get("object")
        if obj_name and obj_name not in defined:
            errors.append(
                f"Assembly placements[{idx}] references undefined object '{obj_name}'"
            )
    
    return errors


def validate_object_references(design: dict) -> List[str]:
    """
    Validate that all 'uses' references in composite objects exist.
    
    Args:
        design: The design to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    objects = design.get("objects", {})
    defined = set(objects.keys())
    
    for name, obj in objects.items():
        if obj.get("type") == "composite":
            uses = obj.get("uses")
            if uses and uses not in defined:
                errors.append(f"Object '{name}' uses undefined object '{uses}'")
    
    return errors


def validate_repetition_multiplication(design: dict) -> List[str]:
    """
    Detect 'Unit Cell Brain' - where 10x10 is repeated in both object and assembly.
    """
    errors = []
    objects = design.get("objects", {})
    assembly = design.get("assembly", {})
    
    # 1. Check assembly grid size
    grid = assembly.get("grid", {"x": 1, "y": 1, "z": 1})
    grid_total = grid.get("x", 1) * grid.get("y", 1) * grid.get("z", 1)
    
    if grid_total <= 1:
        return [] # No global repetition, no problem
        
    # 2. Check if any referenced object is ALSO a high-count repeat
    relevant_objects = set()
    if assembly.get("type") == "grid":
        rel = assembly.get("default_object")
        if rel: relevant_objects.add(rel)
        for row in assembly.get("mapping", []):
            for obj in row:
                if obj: relevant_objects.add(obj)
    else:
        for p in assembly.get("placements", []):
            rel = p.get("object")
            if rel: relevant_objects.add(rel)
            
    for obj_name in relevant_objects:
        obj = objects.get(obj_name)
        if not obj or obj.get("type") != "composite":
            continue
            
        repeat = obj.get("repeat", {"x": 1, "y": 1, "z": 1})
        obj_total = repeat.get("x", 1) * repeat.get("y", 1) * repeat.get("z", 1)
        
        # Threshold: if both are > 2, it's a red flag for unit cell brain
        # unless it's a very small composite (forming a single unit)
        if grid_total > 4 and obj_total > 4:
            errors.append(
                f"REPETITION MULTIPLICATION DETECTED: Object '{obj_name}' repeats {obj_total} times "
                f"INSIDE a grid of {grid_total}. This causes primitive explosion. "
                f"FIX: Define '{obj_name}' as a single component and put the 10x10 repetition ONLY in the assembly."
            )
            
    return errors


def validate_v2_design(design: dict) -> List[str]:
    """
    Full v2 validation: structural gate + semantic checks.
    
    This is the main entry point for validation.
    Call this after LLM output is parsed.
    
    Args:
        design: The design to validate
        
    Returns:
        List of validation errors (empty if valid)
        
    Raises:
        ValueError: If structural gate fails (blocking errors)
    """
    # Structural gate - fails immediately on violation
    v2_structural_gate(design)
    
    # Semantic validation - collects all errors
    errors = []
    errors.extend(validate_assembly_references(design))
    errors.extend(validate_object_references(design))
    errors.extend(validate_repetition_multiplication(design))
    
    return errors
