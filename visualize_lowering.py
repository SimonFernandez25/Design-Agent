"""
Primitive Lowering Visualization Script

This script demonstrates the fabrication-aware decomposition of derived primitives 
(pyramids, cones) into native primitives (boxes, cylinders).

Use this to understand the lowering process before endpoint generation.

Usage:
    python visualize_lowering.py <unit_cell.json>
"""

import json
import sys
import copy
from pathlib import Path
from primitive_lowering import lower_constructed_primitives


def visualize_lowering(unit_cell_path: Path):
    """
    Load a unit cell JSON and visualize the lowering process
    """
    # Load original unit cell
    with open(unit_cell_path, 'r') as f:
        original_unit_cell = json.load(f)
    
    # Remove metadata if present (from batch processing)
    if 'metadata' in original_unit_cell:
        del original_unit_cell['metadata']
    if 'prompt' in original_unit_cell:
        del original_unit_cell['prompt']
    
    original_components = original_unit_cell['unit_cell']['components']
    
    print("=" * 80)
    print("PRIMITIVE LOWERING VISUALIZATION")
    print("=" * 80)
    print(f"\nInput: {unit_cell_path.name}")
    print(f"Job: {original_unit_cell['job_name']}")
    
    # Display original components
    print("\n" + "=" * 80)
    print("ORIGINAL COMPONENTS (from Geometry Agent)")
    print("=" * 80)
    
    for i, comp in enumerate(original_components):
        comp_type = comp['type']
        has_construction = 'construction' in comp
        
        print(f"\n[{i}] {comp_type.upper()}")
        print(f"    Center: {comp['center']}")
        print(f"    Dimensions: {comp['dimensions']}")
        
        if has_construction:
            construction = comp['construction']
            method = construction['method'].replace('_', ' ')
            layers = construction['layers']
            print(f"    [*] Construction: {construction}")
            print(f"    --> Will be decomposed into {layers} {method}")
        else:
            print(f"    [OK] Native primitive (no decomposition needed)")
    
    # Perform lowering
    print("\n" + "=" * 80)
    print("LOWERING PROCESS")
    print("=" * 80)
    print()
    
    lowered_unit_cell = lower_constructed_primitives(copy.deepcopy(original_unit_cell))
    lowered_components = lowered_unit_cell['unit_cell']['components']
    
    # Display results
    print("\n" + "=" * 80)
    print("LOWERED COMPONENTS (after decomposition)")
    print("=" * 80)
    print(f"\nTotal components: {len(original_components)} -> {len(lowered_components)}")
    
    # Count component types
    type_counts = {}
    for comp in lowered_components:
        comp_type = comp['type']
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
    
    print(f"\nComponent types in lowered structure:")
    for comp_type, count in sorted(type_counts.items()):
        print(f"  {comp_type}: {count}")
    
    # Show sample of expanded components
    print(f"\nFirst 10 lowered components (sample):")
    for i, comp in enumerate(lowered_components[:10]):
        z_center = comp['center'][2]
        height = comp['dimensions'].get('height_um', 0)
        z_min = z_center - height / 2
        z_max = z_center + height / 2
        
        if comp['type'] == 'box':
            width = comp['dimensions'].get('width_um', 0)
            print(f"  [{i:3d}] box: width={width:5.2f}um, Z=[{z_min:6.2f}, {z_max:6.2f}]um")
        elif comp['type'] == 'cylinder':
            diameter = comp['dimensions'].get('diameter_um', 0)
            print(f"  [{i:3d}] cylinder: diameter={diameter:5.2f}um, Z=[{z_min:6.2f}, {z_max:6.2f}]um")
    
    if len(lowered_components) > 10:
        print(f"  ... ({len(lowered_components) - 10} more components)")
    
    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    derived_types = {'pyramid', 'cone'}
    remaining_derived = [c['type'] for c in lowered_components if c['type'] in derived_types]
    
    if remaining_derived:
        print(f"[X] ERROR: Derived primitives still present: {set(remaining_derived)}")
        return False
    else:
        print(f"[OK] SUCCESS: All derived primitives decomposed to native primitives")
        print(f"[OK] Only box and cylinder primitives remain")
        print(f"[OK] Ready for endpoint generation")
        
        # Save lowered version
        output_path = unit_cell_path.parent / f"{unit_cell_path.stem}_lowered.json"
        with open(output_path, 'w') as f:
            json.dump(lowered_unit_cell, f, indent=2)
        
        print(f"\n[FILE] Saved lowered unit cell to: {output_path.name}")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_lowering.py <unit_cell.json>")
        print("\nExample:")
        print("  python visualize_lowering.py Outputs/pyramid_array/unit_cell.json")
        sys.exit(1)
    
    unit_cell_path = Path(sys.argv[1])
    
    if not unit_cell_path.exists():
        print(f"Error: File not found: {unit_cell_path}")
        sys.exit(1)
    
    visualize_lowering(unit_cell_path)


if __name__ == "__main__":
    main()
