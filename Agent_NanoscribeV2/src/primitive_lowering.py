"""
Primitive Lowering Module

Decomposes derived geometric primitives (pyramids, cones, tapered cylinders) 
into stacked native primitives (boxes, cylinders) for fabrication-aware 
endpoint generation.

Pipeline Position: Called within endpoint_generator.py (Step 1)
Input: Unit cell JSON with potential derived primitives
Output: Unit cell JSON with only native primitives (box, cylinder)

Purpose:
    The Nanoscribe system can only fabricate primitives with constant 
    cross-section per Z layer. Derived shapes like pyramids and cones
    must be approximated as stacks of constant-cross-section primitives.

Algorithm:
    - Pyramids: Stack of boxes with linearly decreasing width
    - Cones: Stack of cylinders with linearly decreasing diameter
    - Tapered cylinders: Stack of cylinders with varying diameter

Author: Nanoscribe Design Agent Team
"""

from typing import Dict, List
import copy


def expand_pyramid(component: Dict) -> List[Dict]:
    """
    Expand a pyramid into stacked boxes with linearly decreasing width.
    
    A pyramid is approximated by N horizontal box slices, where N equals
    the 'layers' value in the construction metadata. Each slice has a
    width that linearly interpolates from base_width to top_width.
    
    Example:
        Input pyramid: base_width=10um, height=20um, top_width=0um, layers=20
        Output: 20 boxes, widths ranging from ~9.75um (bottom) to ~0.25um (top)
    
    Args:
        component: Pyramid component with 'construction' metadata
        
    Returns:
        List of box components with linearly tapered widths
        
    Raises:
        ValueError: If component is not a pyramid or lacks construction field
    """
    if component['type'] != 'pyramid':
        raise ValueError(f"Expected pyramid, got {component['type']}")
    
    if 'construction' not in component:
        raise ValueError("Pyramid must have 'construction' field for lowering")
    
    construction = component['construction']
    if construction.get('method') != 'stacked_boxes':
        raise ValueError(f"Pyramid construction method must be 'stacked_boxes', got {construction.get('method')}")
    
    # Extract parameters
    center = component['center']
    dims = component['dimensions']
    base_width = dims.get('base_width_um', dims.get('width_um', 0))
    height = dims.get('height_um', 0)
    layers = construction['layers']
    top_width = construction.get('top_width_um', 0)
    
    if layers == 0:
        return []
    
    # Calculate Z boundaries
    # Pyramid is centered at component['center'][2], so z_min is center - height/2
    z_min = center[2] - height / 2.0
    layer_height = height / layers
    
    boxes = []
    for i in range(layers):
        # Calculate position in taper (0 = bottom, 1 = top)
        # Using center of layer for interpolation
        taper_ratio = (i + 0.5) / layers
        
        # Linear interpolation from base to top
        width_at_layer = base_width * (1 - taper_ratio) + top_width * taper_ratio
        
        # Skip very small layers (below fabrication resolution)
        if width_at_layer < 0.01:
            continue
        
        # Calculate Z center for this layer
        z_center = z_min + (i + 0.5) * layer_height
        
        # Create box component (square cross-section for pyramids)
        box = {
            'type': 'box',
            'center': [center[0], center[1], z_center],
            'dimensions': {
                'width_um': width_at_layer,
                'depth_um': width_at_layer,
                'height_um': layer_height
            }
        }
        boxes.append(box)
    
    return boxes


def expand_cone(component: Dict) -> List[Dict]:
    """
    Expand a cone into stacked cylinders with linearly decreasing diameter.
    
    Similar to pyramid expansion, but uses circular cross-sections.
    
    Args:
        component: Cone component with 'construction' metadata
        
    Returns:
        List of cylinder components with linearly tapered diameters
        
    Raises:
        ValueError: If component is not a cone or lacks construction field
    """
    if component['type'] != 'cone':
        raise ValueError(f"Expected cone, got {component['type']}")
    
    if 'construction' not in component:
        raise ValueError("Cone must have 'construction' field for lowering")
    
    construction = component['construction']
    if construction.get('method') != 'stacked_cylinders':
        raise ValueError(f"Cone construction method must be 'stacked_cylinders', got {construction.get('method')}")
    
    # Extract parameters
    center = component['center']
    dims = component['dimensions']
    base_diameter = dims.get('base_diameter_um', dims.get('diameter_um', 0))
    height = dims.get('height_um', 0)
    layers = construction['layers']
    top_diameter = construction.get('top_diameter_um', 0)
    
    if layers == 0:
        return []
    
    z_min = center[2] - height / 2.0
    layer_height = height / layers
    
    cylinders = []
    for i in range(layers):
        taper_ratio = (i + 0.5) / layers
        diameter_at_layer = base_diameter * (1 - taper_ratio) + top_diameter * taper_ratio
        
        if diameter_at_layer < 0.01:
            continue
        
        z_center = z_min + (i + 0.5) * layer_height
        
        cylinder = {
            'type': 'cylinder',
            'center': [center[0], center[1], z_center],
            'dimensions': {
                'diameter_um': diameter_at_layer,
                'height_um': layer_height
            }
        }
        cylinders.append(cylinder)
    
    return cylinders


def expand_tapered_cylinder(component: Dict) -> List[Dict]:
    """
    Expand a tapered cylinder into stacked cylinders with varying diameter.
    
    Used for cylinders that have different diameters at top vs bottom,
    such as frustums or tapered posts.
    
    Args:
        component: Cylinder component with 'construction' metadata indicating taper
        
    Returns:
        List of cylinder components with linearly varying diameters
        
    Raises:
        ValueError: If component is not a cylinder or lacks construction field
    """
    if component['type'] != 'cylinder':
        raise ValueError(f"Expected cylinder, got {component['type']}")
    
    if 'construction' not in component:
        raise ValueError("Tapered cylinder must have 'construction' field")
    
    construction = component['construction']
    if construction.get('method') != 'stacked_cylinders':
        raise ValueError(f"Tapered cylinder construction method must be 'stacked_cylinders', got {construction.get('method')}")
    
    # Extract parameters
    center = component['center']
    dims = component['dimensions']
    base_diameter = dims.get('diameter_um', 0)
    height = dims.get('height_um', 0)
    layers = construction['layers']
    top_diameter = construction.get('top_diameter_um', base_diameter)
    
    if layers == 0:
        return []
    
    z_min = center[2] - height / 2.0
    layer_height = height / layers
    
    cylinders = []
    for i in range(layers):
        taper_ratio = (i + 0.5) / layers
        diameter_at_layer = base_diameter * (1 - taper_ratio) + top_diameter * taper_ratio
        
        z_center = z_min + (i + 0.5) * layer_height
        
        cylinder = {
            'type': 'cylinder',
            'center': [center[0], center[1], z_center],
            'dimensions': {
                'diameter_um': diameter_at_layer,
                'height_um': layer_height
            }
        }
        cylinders.append(cylinder)
    
    return cylinders


def lower_constructed_primitives(unit_cell_data: Dict) -> Dict:
    """
    Lower all constructed primitives in a unit cell to native primitives.
    
    This is the main orchestrator function. It iterates through all components
    and expands any that have construction metadata.
    
    Processing rules:
        - Pyramids with stacked_boxes -> list of boxes
        - Cones with stacked_cylinders -> list of cylinders
        - Tapered cylinders with stacked_cylinders -> list of cylinders
        - Native primitives (box, cylinder without construction) -> unchanged
    
    Args:
        unit_cell_data: Complete unit cell JSON from geometry agent
        
    Returns:
        Modified unit cell with all constructed primitives expanded
        
    Raises:
        ValueError: If derived primitives found without construction metadata
    """
    # Deep copy to avoid modifying original
    unit_cell_data = copy.deepcopy(unit_cell_data)
    
    components = unit_cell_data['unit_cell']['components']
    expanded_components = []
    
    for comp in components:
        comp_type = comp['type']
        has_construction = 'construction' in comp
        
        # Validate: derived primitives MUST have construction metadata
        if comp_type in ['pyramid', 'cone'] and not has_construction:
            raise ValueError(
                f"Found {comp_type} without 'construction' field. "
                f"Derived primitives must include construction metadata. "
                f"Please regenerate unit cell with updated geometry agent."
            )
        
        # Expand constructed primitives
        if has_construction:
            method = comp['construction'].get('method')
            
            if comp_type == 'pyramid' and method == 'stacked_boxes':
                print(f"  Lowering pyramid -> {comp['construction']['layers']} boxes")
                expanded = expand_pyramid(comp)
                expanded_components.extend(expanded)
            
            elif comp_type == 'cone' and method == 'stacked_cylinders':
                print(f"  Lowering cone -> {comp['construction']['layers']} cylinders")
                expanded = expand_cone(comp)
                expanded_components.extend(expanded)
            
            elif comp_type == 'cylinder' and method == 'stacked_cylinders':
                print(f"  Lowering tapered cylinder -> {comp['construction']['layers']} cylinders")
                expanded = expand_tapered_cylinder(comp)
                expanded_components.extend(expanded)
            
            else:
                print(f"  Warning: Unrecognized construction method '{method}' for {comp_type}, keeping original")
                expanded_components.append(comp)
        else:
            # Native primitive without construction -> keep as is
            expanded_components.append(comp)
    
    # Replace components with expanded version
    unit_cell_data['unit_cell']['components'] = expanded_components
    
    # Validate: ensure only native primitives remain
    remaining_types = set(c['type'] for c in expanded_components)
    derived_types = remaining_types & {'pyramid', 'cone'}
    
    if derived_types:
        raise ValueError(
            f"Lowering failed: derived primitives still present after expansion: {derived_types}"
        )
    
    print(f"[OK] Lowering complete: {len(components)} -> {len(expanded_components)} components")
    
    return unit_cell_data


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("Testing pyramid expansion...")
    
    test_pyramid = {
        'type': 'pyramid',
        'center': [0, 0, 10],
        'dimensions': {
            'base_width_um': 10,
            'height_um': 20
        },
        'construction': {
            'method': 'stacked_boxes',
            'layers': 20,
            'top_width_um': 0
        }
    }
    
    boxes = expand_pyramid(test_pyramid)
    print(f"[OK] Generated {len(boxes)} boxes")
    print(f"  Bottom layer: width={boxes[0]['dimensions']['width_um']:.2f}um at z={boxes[0]['center'][2]:.2f}um")
    print(f"  Top layer: width={boxes[-1]['dimensions']['width_um']:.2f}um at z={boxes[-1]['center'][2]:.2f}um")
    
    # Verify linear taper
    assert boxes[0]['dimensions']['width_um'] > boxes[-1]['dimensions']['width_um'], "Should taper"
    assert abs(boxes[0]['center'][2] - 0.5) < 0.1, "First layer should be near z=0.5"
    assert abs(boxes[-1]['center'][2] - 19.5) < 0.1, "Last layer should be near z=19.5"
    
    print("\n[OK] All tests passed!")
