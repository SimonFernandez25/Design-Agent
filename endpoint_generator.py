"""
Endpoint Generator - Deterministic Unit Cell to Scan Endpoints Converter

Converts unit cell geometric components into layer-by-layer scan endpoints for GWL fabrication.
Pure deterministic geometry processing - no LLM, no randomness.

Input: Unit cell JSON with geometric components + print parameters
Output: Layer-grouped scan endpoint JSON ready for GWL serialization
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import math
from primitive_lowering import lower_constructed_primitives

# ======================================================
# CONFIGURATION
# ======================================================

def load_print_parameters(param_file: Path) -> Dict[str, float]:
    """Load print parameters from file"""
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                params[key.strip()] = float(value.strip())
    return params

# ======================================================
# VOXEL-AWARE PATH CONSOLIDATION
# ======================================================

def consolidate_segments(segments: List[Tuple], voxel_xy: float) -> List[Tuple]:
    """
    Consolidate hatch segments by merging paths that expose the same voxel volume.
    
    Groups segments by Y-coordinate (within voxel tolerance), then merges colinear
    segments whose gaps are smaller than the voxel size.
    
    Args:
        segments: List of (x_start, y, x_end, y) tuples
        voxel_xy: Voxel size in micrometers for grouping tolerance
        
    Returns:
        Consolidated list of (x_start, y, x_end, y) tuples
    """
    if not segments:
        return []
    
    # Group segments by Y coordinate (within voxel tolerance)
    y_groups = {}
    for seg in segments:
        y = seg[1]  # Y coordinate (same for start and end in horizontal hatch)
        
        # Find existing group within voxel tolerance
        found_group = False
        for group_y in y_groups.keys():
            if abs(y - group_y) <= voxel_xy:
                y_groups[group_y].append(seg)
                found_group = True
                break
        
        if not found_group:
            y_groups[y] = [seg]
    
    # Consolidate each Y group
    consolidated = []
    for group_y, group_segs in y_groups.items():
        # Sort segments left-to-right by x_start
        sorted_segs = sorted(group_segs, key=lambda s: s[0])
        
        # Merge colinear segments with gaps <= voxel_xy
        merged = []
        current_start = sorted_segs[0][0]
        current_end = sorted_segs[0][2]
        current_y = group_y  # Use group centroid Y
        
        for seg in sorted_segs[1:]:
            seg_start = seg[0]
            seg_end = seg[2]
            
            # Check if this segment is within voxel distance of current merged segment
            gap = seg_start - current_end
            
            if gap <= voxel_xy:
                # Merge by extending current segment
                current_end = max(current_end, seg_end)
            else:
                # Save current merged segment and start new one
                merged.append((current_start, current_y, current_end, current_y))
                current_start = seg_start
                current_end = seg_end
        
        # Add final segment
        merged.append((current_start, current_y, current_end, current_y))
        consolidated.extend(merged)
    
    return consolidated


# ======================================================
# GEOMETRY CROSS-SECTION FUNCTIONS
# ======================================================

def get_cylinder_cross_section(center_xy: Tuple[float, float], diameter: float) -> Dict:
    """
    Get 2D circle cross-section of a cylinder
    Returns: {type: 'circle', center: (x, y), radius: r}
    """
    return {
        'type': 'circle',
        'center': center_xy,
        'radius': diameter / 2.0
    }

def get_box_cross_section(center_xy: Tuple[float, float], width: float, depth: float) -> Dict:
    """
    Get 2D rectangle cross-section of a box
    Returns: {type: 'rectangle', center: (x, y), width: w, depth: d}
    """
    return {
        'type': 'rectangle',
        'center': center_xy,
        'width': width,
        'depth': depth
    }

# ======================================================
# HATCH LINE GENERATION
# ======================================================

def generate_circle_hatch_lines(center: Tuple[float, float], radius: float, hatch_distance: float) -> List[Tuple]:
    """
    Generate horizontal hatch lines for a circle
    Returns list of (x_start, y, x_end, y) tuples
    Deterministic ordering: Y increasing, X left-to-right
    """
    cx, cy = center
    segments = []
    
    # Generate Y positions for hatch lines
    y_min = cy - radius
    y_max = cy + radius
    
    # Start from bottom, step by hatch_distance
    y = y_min
    while y <= y_max:
        # Calculate X intersection points at this Y
        dy = abs(y - cy)
        if dy < radius:
            dx = math.sqrt(radius**2 - dy**2)
            x_start = cx - dx
            x_end = cx + dx
            segments.append((x_start, y, x_end, y))
        y += hatch_distance
    
    return segments

def generate_rectangle_hatch_lines(center: Tuple[float, float], width: float, depth: float, hatch_distance: float) -> List[Tuple]:
    """
    Generate horizontal hatch lines for a rectangle
    Returns list of (x_start, y, x_end, y) tuples
    """
    cx, cy = center
    segments = []
    
    x_min = cx - width / 2.0
    x_max = cx + width / 2.0
    y_min = cy - depth / 2.0
    y_max = cy + depth / 2.0
    
    # Generate horizontal lines from bottom to top
    y = y_min
    while y <= y_max:
        segments.append((x_min, y, x_max, y))
        y += hatch_distance
    
    return segments

def generate_hatch_segments(cross_section: Dict, hatch_distance: float) -> List[Tuple]:
    """
    Generate hatch line segments for a 2D cross-section
    Returns list of (x_start, y, x_end, y) tuples
    """
    if cross_section['type'] == 'circle':
        return generate_circle_hatch_lines(
            cross_section['center'],
            cross_section['radius'],
            hatch_distance
        )
    elif cross_section['type'] == 'rectangle':
        return generate_rectangle_hatch_lines(
            cross_section['center'],
            cross_section['width'],
            cross_section['depth'],
            hatch_distance
        )
    else:
        return []

# ======================================================
# LAYER GENERATION
# ======================================================

def component_active_at_z(component: Dict, z_layer: float) -> bool:
    """
    Check if a component is active (has geometry) at a given Z layer
    """
    center_z = component['center'][2]
    height = component['dimensions'].get('height_um', 0)
    
    z_min = center_z - height / 2.0
    z_max = center_z + height / 2.0
    
    return z_min <= z_layer <= z_max

def get_component_cross_section(component: Dict) -> Dict:
    """
    Get 2D cross-section of a component
    
    NOTE: After lowering pass, only native primitives (box, cylinder) should reach this function.
    Derived primitives (pyramid, cone) must be expanded before endpoint generation.
    """
    comp_type = component['type']
    
    # Validate: only native primitives should exist after lowering
    if comp_type not in ['box', 'cylinder']:
        raise ValueError(
            f"Unexpected primitive type '{comp_type}' in endpoint generation. "
            f"Derived primitives (pyramid, cone) should be lowered before this step. "
            f"This indicates a bug in the lowering pass."
        )
    
    center_xy = (component['center'][0], component['center'][1])
    dims = component['dimensions']
    
    if comp_type == 'cylinder':
        diameter = dims.get('diameter_um', dims.get('radius_um', 0) * 2)
        return get_cylinder_cross_section(center_xy, diameter)
    elif comp_type == 'box':
        width = dims.get('width_um', 0)
        depth = dims.get('depth_um', 0)
        return get_box_cross_section(center_xy, width, depth)

def generate_unit_cell_layers(unit_cell: Dict, slice_distance: float, hatch_distance: float, voxel_xy: float = 0.5) -> Dict[float, List[Tuple]]:
    """
    Generate all layers for a single unit cell
    Returns: {z_layer: [(x_start, y, x_end, y), ...]}
    """
    components = unit_cell['components']
    
    # Determine Z range
    z_min = 0.0
    z_max = 0.0
    for comp in components:
        center_z = comp['center'][2]
        height = comp['dimensions'].get('height_um', 0)
        comp_z_max = center_z + height / 2.0
        z_max = max(z_max, comp_z_max)
    
    # Generate Z layers
    layers = {}
    z = z_min
    while z <= z_max + 1e-9:  # Small epsilon for floating point
        z_round = round(z, 6)  # Fixed precision
        layer_segments = []
        
        # Check each component at this Z
        for comp in components:
            if component_active_at_z(comp, z):
                cross_section = get_component_cross_section(comp)
                segments = generate_hatch_segments(cross_section, hatch_distance)
                layer_segments.extend(segments)
        
        # Apply voxel-aware consolidation
        if layer_segments:
            consolidated_segments = consolidate_segments(layer_segments, voxel_xy)
            layers[z_round] = consolidated_segments
        
        z += slice_distance
    
    return layers

# ======================================================
# ARRAY TRANSLATION
# ======================================================

def generate_array_offsets(global_info: Dict) -> List[Tuple[float, float]]:
    """
    Generate XY offsets for each unit cell instance in the array
    Returns list of (dx, dy) tuples
    """
    reps = global_info['repetitions']
    spacing = global_info['spacing']
    
    nx = reps['x']
    ny = reps['y']
    spacing_x = spacing['x_um']
    spacing_y = spacing['y_um']
    
    offsets = []
    
    # Center the array around origin
    x_start = -(nx - 1) * spacing_x / 2.0
    y_start = -(ny - 1) * spacing_y / 2.0
    
    for iy in range(ny):
        for ix in range(nx):
            dx = x_start + ix * spacing_x
            dy = y_start + iy * spacing_y
            offsets.append((dx, dy))
    
    return offsets

def translate_segment(segment: Tuple, dx: float, dy: float) -> Tuple[List[float], List[float]]:
    """
    Translate a segment by (dx, dy)
    Returns: ([x_start, y_start], [x_end, y_end])
    """
    x_start, y, x_end, _ = segment
    return (
        [round(x_start + dx, 6), round(y + dy, 6)],
        [round(x_end + dx, 6), round(y + dy, 6)]
    )

def translate_layers_across_array(unit_cell_layers: Dict[float, List[Tuple]], global_info: Dict) -> Dict[float, List[Dict]]:
    """
    Translate unit cell layers across the full array
    Returns: {z_layer: [{"start": [x, y], "end": [x, y]}, ...]}
    """
    offsets = generate_array_offsets(global_info)
    array_layers = {}
    
    # For each Z layer
    for z, segments in unit_cell_layers.items():
        all_segments = []
        
        # Translate segments for each array position
        for dx, dy in offsets:
            for seg in segments:
                start, end = translate_segment(seg, dx, dy)
                all_segments.append({
                    "start": start,
                    "end": end
                })
        
        # Sort segments for deterministic ordering: Y then X
        all_segments.sort(key=lambda s: (s["start"][1], s["start"][0]))
        
        array_layers[z] = all_segments
    
    return array_layers

# ======================================================
# OUTPUT GENERATION
# ======================================================

def generate_endpoint_json(unit_cell_data: Dict, print_params: Dict) -> Dict:
    """
    Generate complete endpoint JSON from unit cell data
    
    This is the main entry point for endpoint generation. It performs:
    1. Lowering pass: expand derived primitives (pyramids, cones) to native primitives
    2. Layer generation: discretize into Z slices
    3. Array translation: replicate across the pattern
    """
    print("[Endpoint Generator]")
    
    # CRITICAL: Lower constructed primitives BEFORE processing
    print("  Step 1: Lowering constructed primitives...")
    unit_cell_data = lower_constructed_primitives(unit_cell_data)
    
    # Extract parameters
    print("  Step 2: Generating unit cell layers...")
    slice_distance = print_params['slice_distance_um']
    hatch_distance = print_params['hatch_distance_um']
    voxel_xy = print_params.get('voxel_xy_um', 0.5)
    
    # Generate layers for single unit cell
    unit_cell_layers = generate_unit_cell_layers(
        unit_cell_data['unit_cell'],
        slice_distance,
        hatch_distance,
        voxel_xy
    )
    
    # Translate across array
    print("  Step 3: Translating across array...")
    array_layers = translate_layers_across_array(
        unit_cell_layers,
        unit_cell_data['global_info']
    )
    
    # Format output
    print("  Step 4: Formatting output...")
    layers_list = [
        {
            "z_um": z,
            "segments": segments
        }
        for z, segments in sorted(array_layers.items())
    ]
    
    result = {
        "job_name": unit_cell_data.get('job_name', 'unknown'),
        "layers": layers_list
    }
    
    print(f"  [OK] Generated {len(layers_list)} layers")
    return result

# ======================================================
# MAIN
# ======================================================

def main(unit_cell_json_path: str, print_params_path: str, output_path: str):
    """
    Main entry point for endpoint generation
    """
    # Load inputs
    with open(unit_cell_json_path, 'r') as f:
        unit_cell_data = json.load(f)
    
    print_params = load_print_parameters(Path(print_params_path))
    
    # Generate endpoints
    endpoint_data = generate_endpoint_json(unit_cell_data, print_params)
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(endpoint_data, f, indent=2)
    
    # Summary
    total_layers = len(endpoint_data['layers'])
    total_segments = sum(len(layer['segments']) for layer in endpoint_data['layers'])
    
    print(f"[ENDPOINT GENERATOR]")
    print(f"  Input: {Path(unit_cell_json_path).name}")
    print(f"  Print params: slice={print_params['slice_distance_um']}um, hatch={print_params['hatch_distance_um']}um")
    print(f"  Output: {total_layers} layers, {total_segments} total segments")
    print(f"  Saved: {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python endpoint_generator.py <unit_cell.json> <print_params.txt> <output.json>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])
