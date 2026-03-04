"""
Object-Aware Render Generator - AABB Projection

Renders geometry using axis-aligned bounding box projection.
No assumptions about geometry type - just projects actual extents.

Output structure:
  renders/
      <object_name>/
          top.png
          side_xz.png
          side_yz.png
      final_assembly/
          top.png
          side_xz.png
          side_yz.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from reduction_engine import reduce_object, reduce_assembly, reduce_for_rendering


# ======================================================
# AABB COMPUTATION
# ======================================================

def compute_primitive_aabb(primitive: Dict) -> Dict:
    """
    Compute axis-aligned bounding box for a primitive.
    
    This is the ONLY function that knows about geometry types.
    All rendering is done via the returned AABB, not geometry semantics.
    
    Returns: {
        'x_min', 'x_max',
        'y_min', 'y_max',
        'z_min', 'z_max'
    }
    """
    center = primitive['center']
    dims = primitive['dimensions']
    prim_type = primitive['type']
    
    # Compute half-extents based on geometry type
    if prim_type == 'box':
        # Get dimensions (handle length/width/height mapping + rotation)
        dx = dims.get('length_um', dims.get('width_um', 1)) / 2
        dy = dims.get('width_um', dims.get('depth_um', 1)) / 2
        # Check if we should use depth_um if length_um was used for X
        if 'length_um' in dims and 'width_um' in dims:
            dy = dims.get('width_um') / 2
        elif 'length_um' in dims and 'depth_um' in dims:
             dy = dims.get('depth_um') / 2

        dz = dims.get('height_um', 1) / 2
        
        # Apply rotation to X/Y extents
        rot_deg = primitive.get('rotation_z_deg', 0)
        if rot_deg != 0:
            rad = np.radians(rot_deg)
            cos_a = abs(np.cos(rad))
            sin_a = abs(np.sin(rad))
            
            # New extents of the OBB (Oriented Bounding Box) AABB
            new_dx = dx * cos_a + dy * sin_a
            new_dy = dx * sin_a + dy * cos_a
            dx, dy = new_dx, new_dy
            
    elif prim_type == 'cylinder':
        r = dims.get('diameter_um', 1) / 2
        dx = dy = r
        dz = dims.get('height_um', 1) / 2
    else:
        # Unknown type - use unit cube
        dx = dy = dz = 0.5
    
    return {
        'x_min': center[0] - dx, 'x_max': center[0] + dx,
        'y_min': center[1] - dy, 'y_max': center[1] + dy,
        'z_min': center[2] - dz, 'z_max': center[2] + dz
    }


# ======================================================
# RENDER FUNCTIONS (AABB PROJECTION)
# ======================================================

def render_top_view(primitives: List[Dict], title: str, output_path: Path):
    """
    Render top-down view (XY projection).
    
    Projects X and Y extents, discards Z.
    """
    if not primitives:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    patches = []
    for prim in primitives:
        aabb = compute_primitive_aabb(prim)
        # Project X and Y
        width = aabb['x_max'] - aabb['x_min']
        height = aabb['y_max'] - aabb['y_min']
        rect = Rectangle((aabb['x_min'], aabb['y_min']), width, height)
        patches.append(rect)
    
    if patches:
        collection = PatchCollection(patches, facecolor='steelblue', edgecolor='navy', 
                                      alpha=0.7, linewidth=0.5)
        ax.add_collection(collection)
    
    # Compute bounds from AABBs
    all_aabbs = [compute_primitive_aabb(p) for p in primitives]
    x_min = min(aabb['x_min'] for aabb in all_aabbs)
    x_max = max(aabb['x_max'] for aabb in all_aabbs)
    y_min = min(aabb['y_min'] for aabb in all_aabbs)
    y_max = max(aabb['y_max'] for aabb in all_aabbs)
    
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(f'{title} - Top View (XY)')
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def render_side_xz(primitives: List[Dict], title: str, output_path: Path):
    """
    Render side view (XZ projection).
    
    Projects X and Z extents, discards Y.
    """
    if not primitives:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    patches = []
    for prim in primitives:
        aabb = compute_primitive_aabb(prim)
        # Project X and Z
        width = aabb['x_max'] - aabb['x_min']
        height = aabb['z_max'] - aabb['z_min']
        rect = Rectangle((aabb['x_min'], aabb['z_min']), width, height)
        patches.append(rect)
    
    if patches:
        collection = PatchCollection(patches, facecolor='steelblue', edgecolor='navy', 
                                      alpha=0.7, linewidth=0.5)
        ax.add_collection(collection)
    
    all_aabbs = [compute_primitive_aabb(p) for p in primitives]
    x_min = min(aabb['x_min'] for aabb in all_aabbs)
    x_max = max(aabb['x_max'] for aabb in all_aabbs)
    z_min = min(aabb['z_min'] for aabb in all_aabbs)
    z_max = max(aabb['z_max'] for aabb in all_aabbs)
    
    margin_x = (x_max - x_min) * 0.1
    margin_z = (z_max - z_min) * 0.1
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(z_min - margin_z, z_max + margin_z)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Z (um)')
    ax.set_title(f'{title} - Side View (XZ)')
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def render_side_yz(primitives: List[Dict], title: str, output_path: Path):
    """
    Render side view (YZ projection).
    
    Projects Y and Z extents, discards X.
    """
    if not primitives:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    patches = []
    for prim in primitives:
        aabb = compute_primitive_aabb(prim)
        # Project Y and Z
        width = aabb['y_max'] - aabb['y_min']
        height = aabb['z_max'] - aabb['z_min']
        rect = Rectangle((aabb['y_min'], aabb['z_min']), width, height)
        patches.append(rect)
    
    if patches:
        collection = PatchCollection(patches, facecolor='steelblue', edgecolor='navy', 
                                      alpha=0.7, linewidth=0.5)
        ax.add_collection(collection)
    
    all_aabbs = [compute_primitive_aabb(p) for p in primitives]
    y_min = min(aabb['y_min'] for aabb in all_aabbs)
    y_max = min(aabb['y_max'] for aabb in all_aabbs)
    z_min = min(aabb['z_min'] for aabb in all_aabbs)
    z_max = max(aabb['z_max'] for aabb in all_aabbs)
    
    margin_y = (y_max - y_min) * 0.1
    margin_z = (z_max - z_min) * 0.1
    ax.set_xlim(y_min - margin_y, y_max + margin_y)
    ax.set_ylim(z_min - margin_z, z_max + margin_z)
    
    ax.set_aspect('equal')
    ax.set_xlabel('Y (um)')
    ax.set_ylabel('Z (um)')
    ax.set_title(f'{title} - Side View (YZ)')
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ======================================================
# OBJECT-AWARE RENDERING
# ======================================================

def generate_object_aware_renders(design: Dict, output_dir: Path, print_params_file: Path) -> Dict[str, List[Path]]:
    """
    Generate renders for each object + final assembly.
    
    Uses AABB projection - no geometry-type assumptions.
    
    Returns: {object_name: [render_paths]}
    """
    output_dir = Path(output_dir)
    render_paths = {}
    
    # Get object library
    objects = design.get('objects', {})
    
    # Render each named object individually
    for obj_name, obj_def in objects.items():
        if obj_def.get('type') == 'geometry':
            # Reduce geometry object to primitives
            # reduce_object returns List[Dict] of primitives directly
            primitives = reduce_object(obj_name, objects)
            
            if not primitives:
                continue
            
            # Create object subfolder
            obj_dir = output_dir / obj_name
            obj_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate views
            paths = []
            top_path = obj_dir / "top.png"
            render_top_view(primitives, obj_name, top_path)
            paths.append(top_path)
            
            xz_path = obj_dir / "side_xz.png"
            render_side_xz(primitives, obj_name, xz_path)
            paths.append(xz_path)
            
            yz_path = obj_dir / "side_yz.png"
            render_side_yz(primitives, obj_name, yz_path)
            paths.append(yz_path)
            
            render_paths[obj_name] = paths
            print(f"  [RENDER] {obj_name}: {len(paths)} views")
    
    # Render final assembly
    reduced_assembly = reduce_assembly(design)
    all_primitives = reduced_assembly.get('primitives', [])
    
    if all_primitives:
        assembly_dir = output_dir / "final_assembly"
        assembly_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        top_path = assembly_dir / "top.png"
        render_top_view(all_primitives, "Final Assembly", top_path)
        paths.append(top_path)
        
        xz_path = assembly_dir / "side_xz.png"
        render_side_xz(all_primitives, "Final Assembly", xz_path)
        paths.append(xz_path)
        
        yz_path = assembly_dir / "side_yz.png"
        render_side_yz(all_primitives, "Final Assembly", yz_path)
        paths.append(yz_path)
        
        render_paths['final_assembly'] = paths
        print(f"  [RENDER] final_assembly: {len(paths)} views")
    
    return render_paths


# ======================================================
# COMMAND-LINE INTERFACE
# ======================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python render_generator_v2.py <design.json> <output_dir> [print_params.txt]")
        sys.exit(1)
    
    design_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    print_params = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    with open(design_file) as f:
        design = json.load(f)
    
    results = generate_object_aware_renders(design, output_dir, print_params)
    
    print(f"\nGenerated {sum(len(paths) for paths in results.values())} renders")
    print(f"Output: {output_dir}")
