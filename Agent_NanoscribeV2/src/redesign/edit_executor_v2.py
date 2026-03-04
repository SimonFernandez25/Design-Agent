"""
edit_executor_v2.py - V2 Edit Executor

Applies structured edits to v2 design format (objects + assembly).
"""

import copy
from typing import Dict, Any, List
from edit_schema_v2 import EditPlan, EditOperation


def apply_edit_plan_v2(design_json: Dict[str, Any], edit_plan: EditPlan) -> Dict[str, Any]:
    """
    Apply v2 edit plan to design JSON.
    
    Args:
        design_json: V2 design with 'objects' and 'assembly'
        edit_plan: Structured edit plan
    
    Returns:
        Modified design JSON
    """
    modified = copy.deepcopy(design_json)
    
    print(f"\n[EDIT EXECUTOR V2] Applying {len(edit_plan['edit_plan'])} edits...")
    
    for i, edit in enumerate(edit_plan['edit_plan'], 1):
        operation = edit['operation']
        params = edit['parameters']
        
        print(f"\n[{i}] {operation}")
        print(f"    Reason: {edit['reason']}")
        
        try:
            if operation == "MODIFY_OBJECT_COMPONENT":
                obj_name = params['object_name']
                idx = params['component_index']
                
                if obj_name not in modified['objects']:
                    raise ValueError(f"Object '{obj_name}' not found")
                
                obj = modified['objects'][obj_name]
                if obj.get('type') != 'geometry':
                    raise ValueError(f"Object '{obj_name}' is not a geometry object")
                
                component = obj['components'][idx]
                
                # Update center
                component['center'] = [
                    params['new_center_x'],
                    params['new_center_y'],
                    params['new_center_z']
                ]
                
                # Update dimensions
                component['dimensions'] = params['new_dimensions']
                
                print(f"    Modified {obj_name}.components[{idx}]: center={component['center']}, dims={component['dimensions']}")
            
            elif operation == "ADD_OBJECT_COMPONENT":
                obj_name = params['object_name']
                
                if obj_name not in modified['objects']:
                    raise ValueError(f"Object '{obj_name}' not found")
                
                obj = modified['objects'][obj_name]
                if obj.get('type') != 'geometry':
                    raise ValueError(f"Object '{obj_name}' is not a geometry object")
                
                new_component = {
                    "type": params['component_type'],
                    "center": [params['center_x'], params['center_y'], params['center_z']],
                    "dimensions": params['dimensions']
                }
                
                insert_at = params['insert_at']
                if insert_at == -1:
                    obj['components'].append(new_component)
                else:
                    obj['components'].insert(insert_at, new_component)
                
                print(f"    Added {params['component_type']} to {obj_name} at index {insert_at}")
            
            elif operation == "REMOVE_OBJECT_COMPONENT":
                obj_name = params['object_name']
                idx = params['component_index']
                
                if obj_name not in modified['objects']:
                    raise ValueError(f"Object '{obj_name}' not found")
                
                obj = modified['objects'][obj_name]
                if obj.get('type') != 'geometry':
                    raise ValueError(f"Object '{obj_name}' is not a geometry object")
                
                # Validate index
                if idx < 0 or idx >= len(obj['components']):
                    print(f"    [WARNING] Component index {idx} out of range for {obj_name} (has {len(obj['components'])} components)")
                    print(f"    Skipping this removal")
                    continue
                
                removed = obj['components'].pop(idx)
                print(f"    Removed {removed['type']} from {obj_name} at index {idx}")
            
            elif operation == "MODIFY_ASSEMBLY_GRID":
                assembly = modified.get('assembly', {})
                
                if 'grid_x' in params or 'grid_y' in params:
                    if 'grid' not in assembly:
                        assembly['grid'] = {}
                    if 'grid_x' in params:
                        assembly['grid']['x'] = params['grid_x']
                    if 'grid_y' in params:
                        assembly['grid']['y'] = params['grid_y']
                    print(f"    Set grid: {assembly['grid']}")
                
                if 'spacing_x' in params or 'spacing_y' in params:
                    if 'spacing_um' not in assembly:
                        assembly['spacing_um'] = {}
                    if 'spacing_x' in params:
                        assembly['spacing_um']['x'] = params['spacing_x']
                    if 'spacing_y' in params:
                        assembly['spacing_um']['y'] = params['spacing_y']
                    print(f"    Set spacing: {assembly['spacing_um']}")
                
                modified['assembly'] = assembly
            
            elif operation == "MODIFY_COMPOSITE":
                obj_name = params['object_name']
                
                if obj_name not in modified['objects']:
                    raise ValueError(f"Object '{obj_name}' not found")
                
                obj = modified['objects'][obj_name]
                if obj.get('type') != 'composite':
                    raise ValueError(f"Object '{obj_name}' is not a composite object")
                
                # Update repeat
                if any(k in params for k in ['repeat_x', 'repeat_y', 'repeat_z']):
                    if 'repeat' not in obj:
                        obj['repeat'] = {}
                    if 'repeat_x' in params:
                        obj['repeat']['x'] = params['repeat_x']
                    if 'repeat_y' in params:
                        obj['repeat']['y'] = params['repeat_y']
                    if 'repeat_z' in params:
                        obj['repeat']['z'] = params['repeat_z']
                    print(f"    Set repeat: {obj['repeat']}")
                
                # Update spacing
                if any(k in params for k in ['spacing_x', 'spacing_y', 'spacing_z']):
                    if 'spacing_um' not in obj:
                        obj['spacing_um'] = {}
                    if 'spacing_x' in params:
                        obj['spacing_um']['x'] = params['spacing_x']
                    if 'spacing_y' in params:
                        obj['spacing_um']['y'] = params['spacing_y']
                    if 'spacing_z' in params:
                        obj['spacing_um']['z'] = params['spacing_z']
                    print(f"    Set spacing: {obj['spacing_um']}")
            
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        except Exception as e:
            print(f"    ERROR: {e}")
            raise ValueError(f"Failed to apply edit {i}: {e}")
    
    print("\n[EDIT EXECUTOR V2] All edits applied successfully")
    return modified
