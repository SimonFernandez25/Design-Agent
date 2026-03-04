"""
Render Generator - Create Multi-View Voxel Visualizations

Generates visualization renders showing the fabricated structure from 
multiple viewing angles. Useful for design verification before fabrication.

Pipeline Position: 4 of 5
Input: Endpoint JSON + PrintParameters
Output: PNG renders (global array, unit cell, component views)

Render Types:
    1. Global Array - Complete fabricated structure
    2. Unit Cell - Single repeating unit
    3. Components - Individual primitives in isolation

View Angles:
    - Top-down (XY plane)
    - Side XZ (looking along Y axis)
    - Side YZ (looking along X axis)

Dependencies:
    - numpy: Numerical operations
    - matplotlib: Plotting and image generation
    - endpoint_generator: For unit cell layer generation

Author: Nanoscribe Design Agent Team
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from pathlib import Path
from typing import List, Dict, Tuple


def load_voxel_params(param_file: Path) -> Dict:
    """
    Load voxel dimensions from PrintParameters.txt.
    
    Voxel parameters define the physical resolution of the fabrication:
        - voxel_xy_um: Lateral (XY) voxel dimension
        - voxel_z_um: Axial (Z) voxel dimension
        - slice_distance_um: Z spacing between layers
        - hatch_distance_um: XY spacing between scan lines
    
    Args:
        param_file: Path to PrintParameters.txt
        
    Returns:
        Dict with parameter values
    """
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    try:
                        params[key.strip()] = float(value.strip())
                    except ValueError:
                        params[key.strip()] = value.strip()
    return params


def segment_to_voxel_boxes(segment: Dict, z: float, voxel_xy: float, voxel_z: float) -> List[Dict]:
    """
    Convert a 2D line segment into a series of 3D voxel boxes.
    
    Each voxel box represents the physical volume exposed by the laser
    during one voxel-sized step along the scan path.
    
    Args:
        segment: Dict with 'start' and 'end' [x, y] coordinates
        z: Z coordinate of this layer
        voxel_xy: Voxel size in XY (micrometers)
        voxel_z: Voxel size in Z (micrometers)
        
    Returns:
        List of voxel dicts with 'center' and 'size' fields
    """
    start = np.array(segment['start'])
    end = np.array(segment['end'])
    
    length = np.linalg.norm(end - start)
    num_voxels = max(1, int(np.ceil(length / voxel_xy)))
    
    voxels = []
    for i in range(num_voxels):
        t = (i + 0.5) / num_voxels
        center_2d = start + t * (end - start)
        
        voxel = {
            'center': [float(center_2d[0]), float(center_2d[1]), float(z)],
            'size': [voxel_xy, voxel_xy, voxel_z]
        }
        voxels.append(voxel)
    
    return voxels


# ==============================================================================
# RENDERING FUNCTIONS
# ==============================================================================

def render_top_down_view(voxels: List[Dict], voxel_xy: float, title: str, 
                         output_path: Path) -> None:
    """
    Render and save a top-down view showing voxel footprints in the XY plane.
    
    Color mapping indicates Z height (blue = low, yellow = high).
    
    Args:
        voxels: List of voxel dicts with center and size
        voxel_xy: Voxel size for margin calculation
        title: Plot title
        output_path: Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if not voxels:
        ax.text(0.5, 0.5, 'No voxels', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    z_values = [v['center'][2] for v in voxels]
    z_min, z_max = min(z_values), max(z_values)
    z_range = z_max - z_min if z_max > z_min else 1
    
    patches = []
    colors = []
    
    for voxel in voxels:
        cx, cy, cz = voxel['center']
        sx, sy, sz = voxel['size']
        
        rect = Rectangle((cx - sx/2, cy - sy/2), sx, sy)
        patches.append(rect)
        
        z_norm = (cz - z_min) / z_range
        colors.append(z_norm)
    
    collection = PatchCollection(patches, cmap='viridis', alpha=0.6, edgecolors='none')
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    
    x_coords = [v['center'][0] for v in voxels]
    y_coords = [v['center'][1] for v in voxels]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    margin = voxel_xy * 2
    if 'component' in title.lower():
        ax.set_xlim(-margin, x_max - x_min + margin)
        ax.set_ylim(-margin, y_max - y_min + margin)
    else:
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (um)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (um)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(collection, ax=ax, label='Z Height (um)')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'{z_min:.1f}', f'{(z_min+z_max)/2:.1f}', f'{z_max:.1f}'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def render_side_view_xz(voxels: List[Dict], title: str, output_path: Path) -> None:
    """
    Render and save a side view in the XZ plane (looking along Y axis).
    
    Args:
        voxels: List of voxel dicts
        title: Plot title
        output_path: Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if not voxels:
        ax.text(0.5, 0.5, 'No voxels', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    patches = []
    
    for voxel in voxels:
        cx, cy, cz = voxel['center']
        sx, sy, sz = voxel['size']
        
        rect = Rectangle((cx - sx/2, cz - sz/2), sx, sz)
        patches.append(rect)
    
    collection = PatchCollection(patches, facecolors='steelblue', alpha=0.5, 
                                edgecolors='darkblue', linewidths=0.2)
    ax.add_collection(collection)
    
    x_coords = [v['center'][0] for v in voxels]
    z_coords = [v['center'][2] for v in voxels]
    x_min, x_max = min(x_coords), max(x_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    
    sx_max = max([v['size'][0] for v in voxels])
    sz_max = max([v['size'][2] for v in voxels])
    
    if 'component' in title.lower():
        ax.set_xlim(-sx_max, x_max - x_min + sx_max)
        ax.set_ylim(-sz_max, z_max - z_min + sz_max)
    else:
        ax.set_xlim(x_min - sx_max, x_max + sx_max)
        ax.set_ylim(z_min - sz_max, z_max + sz_max)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (um)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (um)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def render_side_view_yz(voxels: List[Dict], title: str, output_path: Path) -> None:
    """
    Render and save a side view in the YZ plane (looking along X axis).
    
    Args:
        voxels: List of voxel dicts
        title: Plot title
        output_path: Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if not voxels:
        ax.text(0.5, 0.5, 'No voxels', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    patches = []
    
    for voxel in voxels:
        cx, cy, cz = voxel['center']
        sx, sy, sz = voxel['size']
        
        rect = Rectangle((cy - sy/2, cz - sz/2), sy, sz)
        patches.append(rect)
    
    collection = PatchCollection(patches, facecolors='coral', alpha=0.5, 
                                edgecolors='darkred', linewidths=0.2)
    ax.add_collection(collection)
    
    y_coords = [v['center'][1] for v in voxels]
    z_coords = [v['center'][2] for v in voxels]
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    
    sy_max = max([v['size'][1] for v in voxels])
    sz_max = max([v['size'][2] for v in voxels])
    
    if 'component' in title.lower():
        ax.set_xlim(-sy_max, y_max - y_min + sy_max)
        ax.set_ylim(-sz_max, z_max - z_min + sz_max)
    else:
        ax.set_xlim(y_min - sy_max, y_max + sy_max)
        ax.set_ylim(z_min - sz_max, z_max + sz_max)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Y (um)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (um)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def filter_voxels_by_component(all_voxels: List[Dict], component: Dict, 
                               unit_cell_data: Dict, print_params: Dict) -> List[Dict]:
    """
    Filter voxels to only those belonging to a specific component.
    
    Regenerates voxels for just this component to get accurate isolation.
    
    Args:
        all_voxels: Full voxel list (unused, kept for interface compatibility)
        component: Component dict to filter for
        unit_cell_data: Full unit cell data
        print_params: Print parameters
        
    Returns:
        Voxels belonging only to the specified component
    """
    from endpoint_generator import (
        get_component_cross_section,
        component_active_at_z,
        generate_hatch_segments
    )
    
    voxel_xy = print_params.get('voxel_xy_um', 0.5)
    voxel_z = print_params.get('voxel_z_um', 1.3)
    hatch_distance = print_params.get('hatch_distance_um', 0.2)
    
    component_voxels = []
    
    # Find Z layers where this component is active
    z_layers = []
    current_z = 0
    while current_z <= 100:
        if component_active_at_z(component, current_z):
            z_layers.append(current_z)
            current_z += print_params.get('slice_distance_um', 0.5)
        else:
            current_z += print_params.get('slice_distance_um', 0.5)
            if current_z > (component['center'][2] + component['dimensions'].get('height_um', 0) / 2 + 5):
                break
    
    # Generate voxels for this component only
    for z in z_layers:
        if component_active_at_z(component, z):
            cross_section = get_component_cross_section(component)
            if cross_section:
                segments = generate_hatch_segments(cross_section, hatch_distance)
                for seg in segments:
                    seg_dict = {'start': list(seg[:2]), 'end': list(seg[2:])}
                    voxels = segment_to_voxel_boxes(seg_dict, z, voxel_xy, voxel_z)
                    component_voxels.extend(voxels)
    
    return component_voxels


# ==============================================================================
# MAIN GENERATION FUNCTION
# ==============================================================================

def generate_all_renders(endpoint_json_path: Path, print_params_path: Path, 
                        output_dir: Path) -> Dict:
    """
    Generate all renders for a design.
    
    Creates renders for:
        1. Global array (complete pattern)
        2. Single unit cell
        3. Each individual component
    
    Each gets top-down, XZ side, and YZ side views.
    
    Args:
        endpoint_json_path: Path to endpoint JSON file
        print_params_path: Path to PrintParameters.txt
        output_dir: Directory to save render images
        
    Returns:
        Dict with generation statistics
    """
    params = load_voxel_params(print_params_path)
    voxel_xy = params.get('voxel_xy_um', 0.5)
    voxel_z = params.get('voxel_z_um', 1.3)
    
    # Print render parameters
    print(f"[RENDER GENERATOR]")
    print(f"  [PARAMETERS]")
    print(f"    voxel_xy_um: {voxel_xy}")
    print(f"    voxel_z_um: {voxel_z}")
    
    with open(endpoint_json_path, 'r') as f:
        endpoint_data = json.load(f)
    
    # Load unit cell data
    unit_cell_path = endpoint_json_path.parent / 'unit_cell.json'
    if not unit_cell_path.exists():
        unit_cell_path = endpoint_json_path.parent / 'unit_cell_redesigned.json'
        
    if not unit_cell_path.exists():
        raise FileNotFoundError(f"Could not find unit_cell.json in {endpoint_json_path.parent}")
        
    with open(unit_cell_path, 'r') as f:
        unit_cell_data = json.load(f)
    
    job_name = endpoint_data['job_name']
    components = unit_cell_data['unit_cell']['components']
    
    print(f"  Job: {job_name}")
    print(f"  Components: {len(components)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. GLOBAL ARRAY RENDERS
    print("  Generating global array voxels...")
    global_voxels = []
    for layer in endpoint_data['layers']:
        z = layer['z_um']
        for segment in layer['segments']:
            voxels = segment_to_voxel_boxes(segment, z, voxel_xy, voxel_z)
            global_voxels.extend(voxels)
    
    print(f"    {len(global_voxels):,} voxels")
    
    print("  Rendering global array views...")
    render_top_down_view(global_voxels, voxel_xy, f"{job_name} - Global Array (Top)", 
                        output_dir / f"{job_name}_global_top_down.png")
    render_side_view_xz(global_voxels, f"{job_name} - Global Array (Side X-Z)", 
                       output_dir / f"{job_name}_global_side_xz.png")
    render_side_view_yz(global_voxels, f"{job_name} - Global Array (Side Y-Z)", 
                       output_dir / f"{job_name}_global_side_yz.png")
    
    # 2. UNIT CELL RENDERS
    print("  Generating unit cell voxels...")
    from endpoint_generator import generate_unit_cell_layers
    
    unit_cell_layers = generate_unit_cell_layers(
        unit_cell_data['unit_cell'],
        params.get('slice_distance_um', 0.5),
        params.get('hatch_distance_um', 0.2),
        voxel_xy
    )
    
    unit_cell_voxels = []
    for z_layer, segments in unit_cell_layers.items():
        for seg in segments:
            seg_dict = {'start': list(seg[:2]), 'end': list(seg[2:])}
            voxels = segment_to_voxel_boxes(seg_dict, z_layer, voxel_xy, voxel_z)
            unit_cell_voxels.extend(voxels)
    
    print(f"    {len(unit_cell_voxels):,} voxels")
    
    print("  Rendering unit cell views...")
    render_top_down_view(unit_cell_voxels, voxel_xy, f"{job_name} - Unit Cell (Top)", 
                        output_dir / f"{job_name}_unit_cell_top_down.png")
    render_side_view_xz(unit_cell_voxels, f"{job_name} - Unit Cell (Side X-Z)", 
                       output_dir / f"{job_name}_unit_cell_side_xz.png")
    render_side_view_yz(unit_cell_voxels, f"{job_name} - Unit Cell (Side Y-Z)", 
                       output_dir / f"{job_name}_unit_cell_side_yz.png")
    
    # 3. INDIVIDUAL COMPONENT RENDERS
    print("  Generating component renders...")
    for i, component in enumerate(components):
        print(f"    Component {i}: {component['type']}")
        
        component_voxels = filter_voxels_by_component(unit_cell_voxels, component, 
                                                       unit_cell_data, params)
        
        print(f"      {len(component_voxels):,} voxels")
        
        comp_name = component['type']
        render_top_down_view(component_voxels, voxel_xy, 
                           f"{job_name} - Component {i} ({comp_name}) - Top", 
                           output_dir / f"{job_name}_component_{i}_top_down.png")
        render_side_view_xz(component_voxels, 
                          f"{job_name} - Component {i} ({comp_name}) - Side X-Z", 
                          output_dir / f"{job_name}_component_{i}_side_xz.png")
        render_side_view_yz(component_voxels, 
                          f"{job_name} - Component {i} ({comp_name}) - Side Y-Z", 
                          output_dir / f"{job_name}_component_{i}_side_yz.png")
    
    print(f"[OK] All renders saved to: {output_dir}")
    
    return {
        'job_name': job_name,
        'num_global_voxels': len(global_voxels),
        'num_unit_cell_voxels': len(unit_cell_voxels),
        'num_components': len(components),
        'voxel_params': {'xy': voxel_xy, 'z': voxel_z},
        'output_dir': output_dir
    }


def main(endpoint_json_path: str, print_params_path: str, output_dir: str):
    """Main entry point for render generation."""
    result = generate_all_renders(
        Path(endpoint_json_path),
        Path(print_params_path),
        Path(output_dir)
    )
    
    print("\n" + "="*60)
    print("RENDER GENERATION COMPLETE")
    print("="*60)
    print(f"Job: {result['job_name']}")
    print(f"Global voxels: {result['num_global_voxels']:,}")
    print(f"Unit cell voxels: {result['num_unit_cell_voxels']:,}")
    print(f"Components: {result['num_components']}")
    print(f"Renders saved to: {result['output_dir']}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python render_generator.py <endpoints.json> <PrintParameters.txt> <output_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])
