"""
Edit Executor - Apply Rigid Template Edits

Applies structured edits to unit cell JSON. Each operation modifies
the JSON in a specific, predictable way.

Enforces structural invariants for STRUCTURAL scope edits:
    - STRUCTURAL must start with CLEAR_COMPONENTS
    - STRUCTURAL must include at least one ADD_COMPONENT

Author: Nanoscribe Design Agent Team
"""

import copy
from typing import Dict, Any, List
from edit_schema import EditPlan, EditOperation, validate_edit_plan


class StructuralInvariantError(Exception):
    """Raised when a STRUCTURAL edit plan violates invariants."""
    pass


def apply_edit_plan(original_json: Dict[str, Any], edit_plan: EditPlan) -> Dict[str, Any]:
    """
    Apply edit plan to original JSON.
    
    Processes each operation in sequence, modifying a copy of the
    original JSON.
    
    Operations supported:
        - CLEAR_COMPONENTS: Empty the components list
        - MODIFY_COMPONENT: Update center and dimensions
        - ADD_COMPONENT: Insert new component
        - REMOVE_COMPONENT: Remove component by index
        - MODIFY_PATTERN_MODIFIERS: Update pattern modifiers
    
    Args:
        original_json: Original unit cell JSON
        edit_plan: Structured edit plan
    
    Returns:
        Modified JSON
    
    Raises:
        StructuralInvariantError: If STRUCTURAL edit violates invariants
        ValueError: If edit application fails
    """
    # Validate edit plan including structural invariants
    is_valid, errors = validate_edit_plan(edit_plan)
    if not is_valid:
        raise StructuralInvariantError(
            f"Edit plan validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    modified_json = copy.deepcopy(original_json)
    edit_scope = edit_plan.get('edit_scope', 'PARAMETRIC')
    
    print(f"\n[EDIT EXECUTOR] Scope: {edit_scope}")
    print(f"[EDIT EXECUTOR] Applying {len(edit_plan['edit_plan'])} edits...")
    
    for i, edit in enumerate(edit_plan['edit_plan'], 1):
        operation = edit['operation']
        params = edit['parameters']
        
        print(f"\n[{i}] {operation}")
        print(f"    Reason: {edit['reason']}")
        
        try:
            if operation == "CLEAR_COMPONENTS":
                # Clear all existing components
                component_count = len(modified_json['unit_cell']['components'])
                modified_json['unit_cell']['components'] = []
                print(f"    Cleared {component_count} components")
            
            elif operation == "MODIFY_COMPONENT":
                idx = params['component_index']
                component = modified_json['unit_cell']['components'][idx]
                
                # Update center
                component['center'] = [
                    params['new_center_x'],
                    params['new_center_y'],
                    params['new_center_z']
                ]
                
                # Update dimensions
                component['dimensions'] = params['new_dimensions']
                
                print(f"    Modified component {idx}: center={component['center']}, dims={component['dimensions']}")
            
            elif operation == "ADD_COMPONENT":
                new_component = {
                    "type": params['component_type'],
                    "center": [params['center_x'], params['center_y'], params['center_z']],
                    "dimensions": params['dimensions']
                }
                
                insert_at = params['insert_at']
                if insert_at == -1:
                    modified_json['unit_cell']['components'].append(new_component)
                else:
                    modified_json['unit_cell']['components'].insert(insert_at, new_component)
                
                print(f"    Added {params['component_type']} at index {insert_at}")
            
            elif operation == "REMOVE_COMPONENT":
                idx = params['component_index']
                removed = modified_json['unit_cell']['components'].pop(idx)
                print(f"    Removed component {idx}: {removed['type']}")
            
            elif operation == "MODIFY_PATTERN_MODIFIERS":
                # Ensure structure exists
                if "global_info" not in modified_json:
                    modified_json["global_info"] = {}
                    
                if "pattern_modifiers" not in modified_json["global_info"]:
                    modified_json["global_info"]["pattern_modifiers"] = {}
                
                mods = modified_json["global_info"]["pattern_modifiers"]
                
                if params.get("clear_modifiers"):
                    mods.clear()
                    print("    Cleared existing pattern modifiers")
                
                if "rotation" in params:
                    mods["rotation"] = params["rotation"]
                    print(f"    Set rotation: {params['rotation']}")
                    
                if "flip" in params:
                    mods["flip"] = params["flip"]
                    print(f"    Set flip: {params['flip']}")
                    
                if "row_offset" in params:
                    mods["row_offset"] = params["row_offset"]
                    print(f"    Set row_offset: {params['row_offset']}")
            
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            print(f"    ERROR: {e}")
            raise ValueError(f"Failed to apply edit {i}: {e}")
    
    print("\n[EDIT EXECUTOR] All edits applied successfully")
    return modified_json
