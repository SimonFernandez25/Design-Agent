"""
Edit Plan Schema - Object-Targeted Redesign Operations

Edit plans now target NAMED OBJECTS, not raw geometry.
This removes redesign ambiguity by forcing explicit edit targets.

Key changes from v1:
- edit_target field specifies which object to edit
- STRUCTURAL edits replace object definition atomically
- Executor must re-reduce everything downstream after structural edits
"""

from typing import Dict, List
from enum import Enum


class EditScope(Enum):
    """Edit scope determines how the edit affects the system."""
    PARAMETRIC = "PARAMETRIC"  # Modify parameters within existing structure
    STRUCTURAL = "STRUCTURAL"  # Replace object definition atomically


class EditTargetType(Enum):
    """Types of targets that can be edited."""
    OBJECT = "object"          # Named object in library
    ASSEMBLY = "assembly"      # Assembly layout
    GLOBAL = "global"          # Global parameters


EDIT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "edit_scope": {
            "type": "string",
            "enum": ["PARAMETRIC", "STRUCTURAL"],
            "description": "PARAMETRIC: modify within structure. STRUCTURAL: replace atomically."
        },
        "edit_target": {
            "type": "string",
            "pattern": "^(object|assembly|global):[a-zA-Z0-9_]+$",
            "description": "Target specification: 'object:name', 'assembly:main', or 'global:params'"
        },
        "rationale": {
            "type": "string",
            "description": "Explanation of why this edit is proposed"
        },
        "edit_operations": {
            "type": "array",
            "items": {"$ref": "#/definitions/edit_operation"},
            "description": "List of atomic edit operations"
        },
        "expected_effect": {
            "type": "string",
            "description": "Expected outcome of applying these edits"
        }
    },
    "required": ["edit_scope", "edit_target", "edit_operations"],
    "definitions": {
        "edit_operation": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "set_parameter",      # Change a single parameter
                        "scale_parameter",    # Multiply a parameter by a factor
                        "add_component",      # Add a primitive to a geometry object
                        "remove_component",   # Remove a primitive by index
                        "replace_component",  # Replace a primitive entirely
                        "set_repeat",         # Change repetition of composite
                        "set_spacing",        # Change spacing of composite
                        "replace_object",     # Replace entire object definition (STRUCTURAL only)
                        "set_mapping"         # Change assembly mapping
                    ]
                },
                "path": {
                    "type": "string",
                    "description": "JSON path to the parameter, e.g., 'components[0].dimensions.width_um'"
                },
                "value": {
                    "description": "New value for the parameter"
                },
                "factor": {
                    "type": "number",
                    "description": "Scale factor for scale_parameter operation"
                },
                "component": {
                    "type": "object",
                    "description": "Component definition for add/replace operations"
                },
                "index": {
                    "type": "integer",
                    "description": "Component index for remove/replace operations"
                },
                "object_definition": {
                    "type": "object",
                    "description": "Full object definition for replace_object operation"
                }
            },
            "required": ["operation"]
        }
    }
}


def parse_edit_target(target: str) -> Dict:
    """
    Parse an edit target string.
    
    Args:
        target: Target string like 'object:unit_cell_frame'
        
    Returns:
        {'type': 'object', 'name': 'unit_cell_frame'}
    """
    if ":" not in target:
        raise ValueError(f"Invalid edit target format: {target}")
    
    target_type, target_name = target.split(":", 1)
    
    if target_type not in ["object", "assembly", "global"]:
        raise ValueError(f"Unknown target type: {target_type}")
    
    return {
        "type": target_type,
        "name": target_name
    }


def validate_edit_plan(plan: Dict, library: Dict) -> List[str]:
    """
    Validate an edit plan against the object library.
    
    Args:
        plan: Edit plan dictionary
        library: Object library dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    for field in ["edit_scope", "edit_target", "edit_operations"]:
        if field not in plan:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Validate scope
    scope = plan["edit_scope"]
    if scope not in ["PARAMETRIC", "STRUCTURAL"]:
        errors.append(f"Invalid edit_scope: {scope}")
    
    # Validate target
    try:
        target = parse_edit_target(plan["edit_target"])
        
        if target["type"] == "object":
            if target["name"] not in library.get("objects", {}):
                errors.append(f"Edit target object not found: {target['name']}")
    except ValueError as e:
        errors.append(str(e))
    
    # Validate operations
    operations = plan.get("edit_operations", [])
    if not operations:
        errors.append("edit_operations cannot be empty")
    
    for idx, op in enumerate(operations):
        op_errors = _validate_operation(op, scope, idx)
        errors.extend(op_errors)
    
    return errors


def _validate_operation(op: Dict, scope: str, idx: int) -> List[str]:
    """Validate a single edit operation."""
    errors = []
    
    operation = op.get("operation")
    if not operation:
        errors.append(f"Operation[{idx}]: missing 'operation' field")
        return errors
    
    valid_ops = [
        "set_parameter", "scale_parameter", "add_component", 
        "remove_component", "replace_component", "set_repeat",
        "set_spacing", "replace_object", "set_mapping"
    ]
    
    if operation not in valid_ops:
        errors.append(f"Operation[{idx}]: unknown operation '{operation}'")
        return errors
    
    # Structural-only operations
    if operation == "replace_object" and scope != "STRUCTURAL":
        errors.append(f"Operation[{idx}]: 'replace_object' requires STRUCTURAL scope")
    
    # Check required fields per operation type
    if operation == "set_parameter":
        if "path" not in op:
            errors.append(f"Operation[{idx}]: set_parameter requires 'path'")
        if "value" not in op:
            errors.append(f"Operation[{idx}]: set_parameter requires 'value'")
    
    elif operation == "scale_parameter":
        if "path" not in op:
            errors.append(f"Operation[{idx}]: scale_parameter requires 'path'")
        if "factor" not in op:
            errors.append(f"Operation[{idx}]: scale_parameter requires 'factor'")
    
    elif operation == "add_component":
        if "component" not in op:
            errors.append(f"Operation[{idx}]: add_component requires 'component'")
    
    elif operation in ["remove_component", "replace_component"]:
        if "index" not in op:
            errors.append(f"Operation[{idx}]: {operation} requires 'index'")
        if operation == "replace_component" and "component" not in op:
            errors.append(f"Operation[{idx}]: replace_component requires 'component'")
    
    elif operation == "replace_object":
        if "object_definition" not in op:
            errors.append(f"Operation[{idx}]: replace_object requires 'object_definition'")
    
    return errors


def is_structural_edit(plan: Dict) -> bool:
    """Check if an edit plan is structural (requires full re-reduction)."""
    return plan.get("edit_scope") == "STRUCTURAL"
