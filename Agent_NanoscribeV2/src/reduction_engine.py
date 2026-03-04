"""
Reduction Engine - Deterministic Named Object to Flat Primitive Conversion

This is the CRITICAL bridge between agent space and executor space.

After reduction:
- No named objects remain
- No hierarchy remains
- No repetition remains
- Only absolute primitives with final positions

The reducer is DETERMINISTIC - same input always produces same output.
No intelligence, no heuristics, just mechanical expansion.
"""

import copy
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from schemas.primitives import (
    is_native_primitive,
    is_derived_primitive,
    NATIVE_PRIMITIVE_TYPES
)

# Minimum allowed vertical resolution in micrometers
# Prevents unintentional primitive explosion (e.g. 1000 layers for a 10um object)
MIN_Z_RESOLUTION = 0.1


@dataclass
class Transform:
    """Accumulated transform for positioning primitives."""
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rotate_z_deg: float = 0.0
    
    def compose(self, other: 'Transform') -> 'Transform':
        """Compose two transforms (apply other on top of self)."""
        # Apply rotation to the translation offset
        rad = math.radians(self.rotate_z_deg)
        rotated_dx = other.dx * math.cos(rad) - other.dy * math.sin(rad)
        rotated_dy = other.dx * math.sin(rad) + other.dy * math.cos(rad)
        
        return Transform(
            dx=self.dx + rotated_dx,
            dy=self.dy + rotated_dy,
            dz=self.dz + other.dz,
            rotate_z_deg=self.rotate_z_deg + other.rotate_z_deg
        )


def apply_transform(primitive: Dict, transform: Transform) -> Dict:
    """
    Apply a transform to a primitive, producing an absolute primitive.
    
    Args:
        primitive: Primitive with center and dimensions
        transform: Accumulated transform to apply
        
    Returns:
        New primitive with transformed center and accumulated rotation
    """
    result = copy.deepcopy(primitive)
    center = result["center"]
    
    # Apply rotation to center coordinates
    if transform.rotate_z_deg != 0:
        rad = math.radians(transform.rotate_z_deg)
        x, y = center[0], center[1]
        center[0] = x * math.cos(rad) - y * math.sin(rad)
        center[1] = x * math.sin(rad) + y * math.cos(rad)
    
    # Apply translation
    center[0] += transform.dx
    center[1] += transform.dy
    center[2] += transform.dz
    
    # PERSIST ROTATION
    # Accumulate rotation in the primitive itself
    current_rot = result.get("rotation_z_deg", 0.0)
    result["rotation_z_deg"] = (current_rot + transform.rotate_z_deg) % 360.0
    
    result["center"] = center
    return result


def normalize_primitive_dimensions(primitive: Dict) -> Dict:
    """
    Normalize primitive dimensions to explicit axis-mapped keys.
    
    Converts ambiguous 'length', 'width', 'depth' to 'x_um', 'y_um', 'z_um'.
    This removes silent inference fallback in downstream consumers.
    
    Mapping logic for Box:
    - If x_um, y_um, z_um exist -> keep
    - If length_um, width_um, height_um:
        - length -> x
        - width -> y
        - height -> z
    - If width_um, depth_um, height_um:
        - width -> x
        - depth -> y
        - height -> z
        
    Returns:
        Primitive with normalized dimensions dict
    """
    if primitive['type'] != 'box':
        return primitive
        
    dims = primitive['dimensions']
    new_dims = dims.copy()
    
    # Already normalized?
    if 'x_um' in dims and 'y_um' in dims and 'z_um' in dims:
        return primitive
        
    # Height is always Z
    z_val = dims.get('height_um', dims.get('z_um', 0))
    new_dims['z_um'] = z_val
    if 'height_um' in new_dims: del new_dims['height_um']
    
    # Handle XY mapping
    # Case 1: Length/Width (Length=X)
    if 'length_um' in dims:
        new_dims['x_um'] = dims['length_um']
        if 'length_um' in new_dims: del new_dims['length_um']
        
        # If length is present, 'width' usually means Y
        if 'width_um' in dims:
            new_dims['y_um'] = dims['width_um']
            if 'width_um' in new_dims: del new_dims['width_um']
            
    # Case 2: Width/Depth (Width=X, Depth=Y)
    elif 'width_um' in dims:
        new_dims['x_um'] = dims['width_um']
        if 'width_um' in new_dims: del new_dims['width_um']
        
        if 'depth_um' in dims:
            new_dims['y_um'] = dims['depth_um']
            if 'depth_um' in new_dims: del new_dims['depth_um']
            
    # Ensure keys exist
    if 'x_um' not in new_dims: new_dims['x_um'] = 0.0
    if 'y_um' not in new_dims: new_dims['y_um'] = 0.0
    if 'z_um' not in new_dims: new_dims['z_um'] = 0.0
    
    primitive['dimensions'] = new_dims
    return primitive


def expand_derived_primitive(primitive: Dict) -> List[Dict]:
    """
    Expand a derived primitive (pyramid, cone) into native primitives.
    
    This is the lowering pass that ensures only native primitives
    reach the executor.
    
    Args:
        primitive: Derived primitive with 'construction' metadata
        
    Returns:
        List of native primitives (boxes or cylinders)
    """
    prim_type = primitive["type"]
    construction = primitive.get("construction", {})
    method = construction.get("method")
    num_layers = construction.get("layers", 20)
    
    if prim_type == "pyramid":
        return _expand_pyramid(primitive, num_layers)
    elif prim_type == "cone":
        return _expand_cone(primitive, num_layers)
    elif prim_type == "tapered_cylinder":
        return _expand_tapered_cylinder(primitive, num_layers)
    else:
        # Unknown derived type - cannot expand
        raise ValueError(f"Cannot expand derived primitive type: {prim_type}")


def _expand_pyramid(pyramid: Dict, num_layers: int) -> List[Dict]:
    """Expand pyramid into stacked boxes."""
    # 2️⃣ Verify Lowering Happens Only Once
    assert pyramid.get("type") == "pyramid", f"Expected type 'pyramid', got '{pyramid.get('type')}'"
    
    center = pyramid["center"]
    dims = pyramid["dimensions"]
    
    base_width = dims.get("base_width_um", dims.get("width_um", 10))
    height = dims.get("height_um", 10)
    
    # 3️⃣ Inspect Construction Metadata Source
    construction = pyramid.get("construction", {})
    layers_attr = construction.get("layers")
    print(f"[DEBUG] LOWERING PYRAMID: height={height}um, layers={layers_attr}")
    
    # 🛠 Required Fix (Vertical Resolution Gate)
    vertical_resolution = height / num_layers
    print(f"[DEBUG]   -> Vertical resolution: {vertical_resolution:.4f} um")
    if vertical_resolution < MIN_Z_RESOLUTION:
        raise ValueError(
            f"Vertical resolution {vertical_resolution:.4f}um is below minimum {MIN_Z_RESOLUTION}um. "
            f"Increase height ({height}) or decrease layers ({num_layers}) to maintain structural integrity."
        )
    
    # Check both dims and construction for top_width
    construction = pyramid.get("construction", {})
    top_width = dims.get("top_width_um", construction.get("top_width_um", 0))
    
    # Pyramid center is at centroid - calculate base z
    base_z = center[2] - height / 2
    layer_height = height / num_layers
    
    boxes = []
    for i in range(num_layers):
        # Linear interpolation from base to top
        t = (i + 0.5) / num_layers  # Center of this layer
        layer_width = base_width + (top_width - base_width) * t
        layer_z = base_z + (i + 0.5) * layer_height
        
        if layer_width > 0.01:  # Skip degenerate layers
            boxes.append({
                "type": "box",
                "center": [center[0], center[1], layer_z],
                "dimensions": {
                    "width_um": layer_width,
                    "depth_um": layer_width,
                    "height_um": layer_height
                }
            })
    
    return boxes


def _expand_cone(cone: Dict, num_layers: int) -> List[Dict]:
    """Expand cone into stacked cylinders."""
    # 2️⃣ Verify Lowering Happens Only Once
    assert cone.get("type") == "cone", f"Expected type 'cone', got '{cone.get('type')}'"
    
    center = cone["center"]
    dims = cone["dimensions"]
    
    base_diameter = dims.get("base_diameter_um", dims.get("diameter_um", 10))
    height = dims.get("height_um", 10)
    
    # 3️⃣ Inspect Construction Metadata Source
    construction = cone.get("construction", {})
    layers_attr = construction.get("layers")
    print(f"[DEBUG] LOWERING CONE: height={height}um, layers={layers_attr}")
    
    # 🛠 Required Fix (Vertical Resolution Gate)
    vertical_resolution = height / num_layers
    print(f"[DEBUG]   -> Vertical resolution: {vertical_resolution:.4f} um")
    if vertical_resolution < MIN_Z_RESOLUTION:
         raise ValueError(f"Vertical resolution {vertical_resolution:.4f}um is below minimum {MIN_Z_RESOLUTION}um.")
    
    # Check both dims and construction for top_diameter
    construction = cone.get("construction", {})
    top_diameter = dims.get("top_diameter_um", construction.get("top_diameter_um", 0))
    
    base_z = center[2] - height / 2
    layer_height = height / num_layers
    
    cylinders = []
    for i in range(num_layers):
        t = (i + 0.5) / num_layers
        layer_diameter = base_diameter + (top_diameter - base_diameter) * t
        layer_z = base_z + (i + 0.5) * layer_height
        
        if layer_diameter > 0.01:
            cylinders.append({
                "type": "cylinder",
                "center": [center[0], center[1], layer_z],
                "dimensions": {
                    "diameter_um": layer_diameter,
                    "height_um": layer_height
                }
            })
    
    return cylinders


def _expand_tapered_cylinder(cyl: Dict, num_layers: int) -> List[Dict]:
    """Expand tapered cylinder into stacked cylinders."""
    center = cyl["center"]
    dims = cyl["dimensions"]
    
    base_diameter = dims.get("base_diameter_um", dims.get("diameter_um", 10))
    top_diameter = dims.get("top_diameter_um", base_diameter)
    height = dims.get("height_um", 10)
    
    base_z = center[2] - height / 2
    layer_height = height / num_layers
    
    cylinders = []
    for i in range(num_layers):
        t = (i + 0.5) / num_layers
        layer_diameter = base_diameter + (top_diameter - base_diameter) * t
        layer_z = base_z + (i + 0.5) * layer_height
        
        cylinders.append({
            "type": "cylinder",
            "center": [center[0], center[1], layer_z],
            "dimensions": {
                "diameter_um": layer_diameter,
                "height_um": layer_height
            }
        })
    
    return cylinders


def reduce_object(
    object_name: str,
    objects: Dict,
    transform: Transform = None
) -> List[Dict]:
    """
    Recursively reduce a named object to flat primitives.
    
    This is the core reduction algorithm:
    - geometry objects: emit transformed primitives
    - composite objects: iterate over repetitions, recursively reduce
    
    Args:
        object_name: Name of object in library
        objects: Object library dictionary
        transform: Accumulated transform (defaults to identity)
        
    Returns:
        List of absolute primitives (no references, no hierarchy)
    """
    if transform is None:
        transform = Transform()
    
    if object_name not in objects:
        raise ValueError(f"Object not found in library: {object_name}")
    
    obj = objects[object_name]
    obj_type = obj["type"]
    primitives = []
    
    if obj_type == "geometry":
        # Emit primitives with transform applied
        for component in obj["components"]:
            if is_derived_primitive(component):
                # Expand derived primitives first
                expanded = expand_derived_primitive(component)
                for prim in expanded:
                    transformed = apply_transform(prim, transform)
                    primitives.append(normalize_primitive_dimensions(transformed))
            else:
                # Native primitive - just transform
                transformed = apply_transform(component, transform)
                primitives.append(normalize_primitive_dimensions(transformed))
    
    elif obj_type == "composite":
        # Get referenced object
        uses = obj["uses"]
        
        # Get repetition parameters
        repeat = obj.get("repeat", {"x": 1, "y": 1, "z": 1})
        # 4️⃣ Check for Nested Repeat Multiplication
        print(f"[DEBUG] COMPOSITE '{object_name}' -> '{uses}' | Repeat: {repeat}")
        
        spacing = obj.get("spacing_um", {"x": 0, "y": 0, "z": 0})
        
        # Get optional base transform for the composite
        obj_transform = obj.get("transform", {})
        base_translate = obj_transform.get("translate", [0, 0, 0])
        base_rotate = obj_transform.get("rotate_z_deg", 0)
        
        base_transform = Transform(
            dx=base_translate[0],
            dy=base_translate[1],
            dz=base_translate[2],
            rotate_z_deg=base_rotate
        )
        
        # Iterate over all repetitions
        for iz in range(repeat.get("z", 1)):
            for iy in range(repeat.get("y", 1)):
                for ix in range(repeat.get("x", 1)):
                    # Calculate offset for this instance
                    instance_transform = Transform(
                        dx=ix * spacing.get("x", 0),
                        dy=iy * spacing.get("y", 0),
                        dz=iz * spacing.get("z", 0)
                    )
                    
                    # Compose: parent transform -> base transform -> instance transform
                    composed = transform.compose(base_transform).compose(instance_transform)
                    
                    # Recursively reduce the referenced object
                    child_primitives = reduce_object(uses, objects, composed)
                    primitives.extend(child_primitives)
    
    else:
        raise ValueError(f"Unknown object type: {obj_type}")
    
    # Track diagnostic summary for pyramids
    pyramid_count = sum(1 for p in primitives if p.get("type") == "pyramid")
    if pyramid_count > 0:
        print(f"[DEBUG] REDUCE OBJECT '{object_name}': Found {pyramid_count} unlowered pyramids")

    return primitives


def count_pyramids_recursive(obj_name: str, objects: Dict) -> int:
    """Recursively count pyramids in a named object before lowering."""
    obj = objects.get(obj_name)
    if not obj:
        return 0
        
    if obj["type"] == "geometry":
        return sum(1 for p in obj.get("components", []) if p.get("type") == "pyramid")
    elif obj["type"] == "composite":
        uses = obj.get("uses")
        repeat = obj.get("repeat", {"x": 1, "y": 1, "z": 1})
        rep_count = repeat.get("x", 1) * repeat.get("y", 1) * repeat.get("z", 1)
        return rep_count * count_pyramids_recursive(uses, objects)
    return 0


def reduce_assembly(design: Dict) -> Dict:
    """
    Reduce entire design (object library + assembly) to flat primitives.
    
    This is the main entry point for reduction.
    
    Args:
        design: Full design dict with 'objects' and 'assembly' keys
        
    Returns:
        {"primitives": [...]} - No references, no names, no hierarchy
    """
    objects = design.get("objects", {})
    assembly = design.get("assembly", {})
    
    if not objects:
        raise ValueError("Design must contain 'objects' library")
    
    primitives = []
    
    assembly_type = assembly.get("type", "grid")
    
    # Count pyramids before lowering
    pyr_count = 0
    if assembly_type == "grid":
        grid = assembly.get("grid", {"x": 1, "y": 1, "z": 1})
        grid_count = grid.get("x", 1) * grid.get("y", 1) * grid.get("z", 1)
        def_obj = assembly.get("default_object")
        if def_obj:
            pyr_count = grid_count * count_pyramids_recursive(def_obj, objects)
    elif assembly_type == "explicit":
        for placement in assembly.get("placements", []):
            pyr_count += count_pyramids_recursive(placement.get("object"), objects)

    if assembly_type == "grid":
        primitives.extend(_reduce_grid_assembly(assembly, objects))
    elif assembly_type == "explicit":
        primitives.extend(_reduce_explicit_assembly(assembly, objects))
    else:
        # No assembly - just reduce the first/main object if specified
        main_object = design.get("main_object")
        if main_object and main_object in objects:
            primitives.extend(reduce_object(main_object, objects))
    
    # 1️⃣ Diagnostic Summary
    print("-" * 40)
    print("REDUCTION DIAGNOSTIC SUMMARY")
    print(f"Pyramid count before lowering: {pyr_count}")
    print(f"Total primitives after lowering: {len(primitives)}")
    print("-" * 40)
    
    return {
        "job_name": design.get("job_name", "unnamed"),
        "primitives": primitives,
        "metadata": {
            "num_primitives": len(primitives),
            "source_objects": list(objects.keys()),
            "assembly_type": assembly_type
        }
    }


def _reduce_grid_assembly(assembly: Dict, objects: Dict) -> List[Dict]:
    """Reduce a grid-type assembly to primitives."""
    primitives = []
    
    grid = assembly.get("grid", {"x": 1, "y": 1})
    # 4️⃣ Check for Nested Repeat Multiplication
    print(f"[DEBUG] ASSEMBLY GRID: {grid}")
    
    spacing = assembly.get("spacing_um", {"x": 0, "y": 0, "z": 0})
    mapping = assembly.get("mapping", [])
    default_object = assembly.get("default_object")
    
    grid_x = grid.get("x", 1)
    grid_y = grid.get("y", 1)
    grid_z = grid.get("z", 1)
    
    for iz in range(grid_z):
        for iy in range(grid_y):
            for ix in range(grid_x):
                # Get object name from mapping or default
                obj_name = None
                if mapping and iy < len(mapping) and ix < len(mapping[iy]):
                    obj_name = mapping[iy][ix]
                if not obj_name:
                    obj_name = default_object
                
                if obj_name and obj_name in objects:
                    # Calculate position
                    transform = Transform(
                        dx=ix * spacing.get("x", 0),
                        dy=iy * spacing.get("y", 0),
                        dz=iz * spacing.get("z", 0)
                    )
                    
                    # Reduce this object at this position
                    cell_primitives = reduce_object(obj_name, objects, transform)
                    primitives.extend(cell_primitives)
    
    return primitives


def _reduce_explicit_assembly(assembly: Dict, objects: Dict) -> List[Dict]:
    """Reduce an explicit-type assembly to primitives."""
    primitives = []
    
    placements = assembly.get("placements", [])
    
    for placement in placements:
        obj_name = placement.get("object")
        position = placement.get("position", [0, 0, 0])
        rotate = placement.get("rotate_z_deg", 0)
        
        if obj_name and obj_name in objects:
            transform = Transform(
                dx=position[0],
                dy=position[1],
                dz=position[2],
                rotate_z_deg=rotate
            )
            
            cell_primitives = reduce_object(obj_name, objects, transform)
            primitives.extend(cell_primitives)
    
    return primitives


def reduce_for_rendering(
    design: Dict,
    target_object: Optional[str] = None
) -> Dict:
    """
    Reduce design for rendering purposes.
    
    Can reduce just a specific object (for object-level renders)
    or the full assembly (for final assembly render).
    
    Args:
        design: Full design dict
        target_object: Optional specific object to reduce (None = full assembly)
        
    Returns:
        Reduced primitive dict
    """
    if target_object:
        objects = design.get("objects", {})
        if target_object not in objects:
            raise ValueError(f"Object not found: {target_object}")
        
        primitives = reduce_object(target_object, objects)
        return {
            "job_name": target_object,
            "primitives": primitives,
            "metadata": {
                "num_primitives": len(primitives),
                "render_scope": "object",
                "object_name": target_object
            }
        }
    else:
        return reduce_assembly(design)


# ======================================================
# VALIDATION AND TESTING
# ======================================================

def validate_reduced_output(reduced: Dict) -> List[str]:
    """
    Validate that reduction output contains only native primitives.
    
    Args:
        reduced: Output from reduce_assembly or reduce_object
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    primitives = reduced.get("primitives", [])
    for idx, prim in enumerate(primitives):
        prim_type = prim.get("type")
        if prim_type not in NATIVE_PRIMITIVE_TYPES:
            errors.append(f"Primitive[{idx}]: non-native type '{prim_type}' in reduced output")
        
        center = prim.get("center", [])
        if len(center) != 3:
            errors.append(f"Primitive[{idx}]: invalid center {center}")
    
    return errors


if __name__ == "__main__":
    # Test the reducer with a sample design
    print("Testing Reduction Engine...")
    
    test_design = {
        "job_name": "test_reduction",
        "objects": {
            "unit_cell_frame": {
                "type": "geometry",
                "components": [
                    {
                        "type": "box",
                        "center": [0, 0, 5],
                        "dimensions": {"width_um": 10, "depth_um": 10, "height_um": 10}
                    }
                ]
            },
            "meta_atom_A": {
                "type": "composite",
                "uses": "unit_cell_frame",
                "repeat": {"x": 2, "y": 2, "z": 1},
                "spacing_um": {"x": 15, "y": 15, "z": 0}
            }
        },
        "assembly": {
            "type": "grid",
            "grid": {"x": 3, "y": 3},
            "spacing_um": {"x": 50, "y": 50},
            "default_object": "meta_atom_A"
        }
    }
    
    reduced = reduce_assembly(test_design)
    
    print(f"[OK] Reduced to {len(reduced['primitives'])} primitives")
    print(f"  Expected: 3x3 grid * 2x2 repeat = 36 primitives")
    
    errors = validate_reduced_output(reduced)
    if errors:
        print(f"[FAIL] Validation errors: {errors}")
    else:
        print("[OK] All primitives are native types")
    
    # Test single object reduction
    obj_reduced = reduce_for_rendering(test_design, "meta_atom_A")
    print(f"[OK] Object 'meta_atom_A' reduced to {len(obj_reduced['primitives'])} primitives")
