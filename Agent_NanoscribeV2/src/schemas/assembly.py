"""
Assembly Schema - Heterogeneous Layout Definition

The assembly defines WHERE objects go, not HOW they are built.
This enables:
- Multiple meta-atom types
- Checkerboards
- Gradients
- Defects
- Intentional disorder

Important: The executor never interprets this - it is semantic layout only.
The reduction pass converts assembly + objects to flat primitives.
"""

from typing import Dict, List, Optional


ASSEMBLY_SCHEMA = {
    "type": "object",
    "properties": {
        "assembly": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["grid", "explicit"],
                    "description": "Assembly type: 'grid' for regular patterns, 'explicit' for arbitrary placement"
                },
                "grid": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "minimum": 1},
                        "y": {"type": "integer", "minimum": 1},
                        "z": {"type": "integer", "minimum": 1, "default": 1}
                    },
                    "description": "Grid dimensions (number of cells in each axis)"
                },
                "spacing_um": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Cell spacing in X (micrometers)"},
                        "y": {"type": "number", "description": "Cell spacing in Y (micrometers)"},
                        "z": {"type": "number", "description": "Cell spacing in Z (micrometers)", "default": 0}
                    },
                    "description": "Spacing between grid cells"
                },
                "mapping": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": "2D array of object names. Row-major order (mapping[y][x])."
                },
                "default_object": {
                    "type": "string",
                    "description": "Default object name for cells not explicitly specified in mapping"
                },
                "placements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "object": {"type": "string", "description": "Object name to place"},
                            "position": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                                "description": "Absolute position [x, y, z] in micrometers"
                            },
                            "rotate_z_deg": {
                                "type": "number",
                                "description": "Optional rotation around Z axis"
                            }
                        },
                        "required": ["object", "position"]
                    },
                    "description": "Explicit object placements (for 'explicit' type)"
                }
            },
            "required": ["type"]
        }
    },
    "required": ["assembly"]
}


def validate_assembly(assembly_def: Dict, object_names: List[str]) -> List[str]:
    """
    Validate an assembly definition.
    
    Args:
        assembly_def: Assembly definition dictionary
        object_names: List of valid object names from the library
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    assembly = assembly_def.get("assembly")
    if not assembly:
        errors.append("Missing 'assembly' field")
        return errors
    
    assembly_type = assembly.get("type")
    if assembly_type not in ["grid", "explicit"]:
        errors.append(f"Invalid assembly type: {assembly_type}")
        return errors
    
    if assembly_type == "grid":
        errors.extend(_validate_grid_assembly(assembly, object_names))
    elif assembly_type == "explicit":
        errors.extend(_validate_explicit_assembly(assembly, object_names))
    
    return errors


def _validate_grid_assembly(assembly: Dict, object_names: List[str]) -> List[str]:
    """Validate a grid-type assembly."""
    errors = []
    
    # Check grid dimensions
    grid = assembly.get("grid", {})
    if not grid:
        errors.append("Grid assembly requires 'grid' field")
    else:
        for axis in ["x", "y"]:
            if axis not in grid:
                errors.append(f"Grid missing required dimension: {axis}")
            elif not isinstance(grid[axis], int) or grid[axis] < 1:
                errors.append(f"Grid.{axis} must be positive integer")
    
    # Check spacing
    spacing = assembly.get("spacing_um", {})
    if not spacing:
        errors.append("Grid assembly requires 'spacing_um' field")
    else:
        for axis in ["x", "y"]:
            if axis not in spacing:
                errors.append(f"spacing_um missing required axis: {axis}")
    
    # Check mapping or default_object
    mapping = assembly.get("mapping")
    default_object = assembly.get("default_object")
    
    if not mapping and not default_object:
        errors.append("Grid assembly requires either 'mapping' or 'default_object'")
    
    if default_object and default_object not in object_names:
        errors.append(f"default_object '{default_object}' not found in object library")
    
    if mapping:
        # Validate mapping dimensions
        grid_y = grid.get("y", 1)
        grid_x = grid.get("x", 1)
        
        if len(mapping) > grid_y:
            errors.append(f"Mapping has {len(mapping)} rows but grid.y is {grid_y}")
        
        for row_idx, row in enumerate(mapping):
            if len(row) > grid_x:
                errors.append(f"Mapping row {row_idx} has {len(row)} columns but grid.x is {grid_x}")
            
            for col_idx, obj_name in enumerate(row):
                if obj_name and obj_name not in object_names:
                    errors.append(f"Mapping[{row_idx}][{col_idx}]: unknown object '{obj_name}'")
    
    return errors


def _validate_explicit_assembly(assembly: Dict, object_names: List[str]) -> List[str]:
    """Validate an explicit-type assembly."""
    errors = []
    
    placements = assembly.get("placements", [])
    if not placements:
        errors.append("Explicit assembly requires 'placements' field")
        return errors
    
    for idx, placement in enumerate(placements):
        obj_name = placement.get("object")
        if not obj_name:
            errors.append(f"Placement[{idx}]: missing 'object' field")
        elif obj_name not in object_names:
            errors.append(f"Placement[{idx}]: unknown object '{obj_name}'")
        
        position = placement.get("position")
        if not position:
            errors.append(f"Placement[{idx}]: missing 'position' field")
        elif not isinstance(position, list) or len(position) != 3:
            errors.append(f"Placement[{idx}]: position must be [x, y, z] array")
    
    return errors


def get_grid_cell_positions(assembly: Dict) -> List[Dict]:
    """
    Generate all cell positions for a grid assembly.
    
    Args:
        assembly: Assembly definition (must be grid type)
        
    Returns:
        List of {'object': str, 'position': [x, y, z]} for each cell
    """
    grid = assembly.get("grid", {})
    spacing = assembly.get("spacing_um", {})
    mapping = assembly.get("mapping", [])
    default_object = assembly.get("default_object")
    
    grid_x = grid.get("x", 1)
    grid_y = grid.get("y", 1)
    grid_z = grid.get("z", 1)
    
    spacing_x = spacing.get("x", 0)
    spacing_y = spacing.get("y", 0)
    spacing_z = spacing.get("z", 0)
    
    positions = []
    
    for iz in range(grid_z):
        for iy in range(grid_y):
            for ix in range(grid_x):
                # Get object from mapping or default
                obj_name = None
                if mapping and iy < len(mapping) and ix < len(mapping[iy]):
                    obj_name = mapping[iy][ix]
                if not obj_name:
                    obj_name = default_object
                
                if obj_name:
                    positions.append({
                        "object": obj_name,
                        "position": [
                            ix * spacing_x,
                            iy * spacing_y,
                            iz * spacing_z
                        ],
                        "grid_index": [ix, iy, iz]
                    })
    
    return positions
