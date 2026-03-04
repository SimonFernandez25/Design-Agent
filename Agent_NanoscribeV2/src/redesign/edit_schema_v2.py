"""
edit_schema_v2.py - V2 Named Object Edit Schema

Edit operations for v2 architecture (objects + assembly, NOT unit_cell + global_info).
"""

from typing import TypedDict, Any, Dict, List

# ======================================================
# EDIT TYPES
# ======================================================

class EditOperation(TypedDict):
    """Structured edit - LLM only fills in numbers"""
    target: str  # e.g., "objects.post.components[0]"
    operation: str
    parameters: Dict[str, Any]
    reason: str

class EditPlan(TypedDict):
    """Complete edit plan"""
    edit_plan: List[EditOperation]
    summary: str

# ======================================================
# V2 EDIT SCHEMAS
# ======================================================

# Modify a component within a geometry object
MODIFY_OBJECT_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "description": "e.g., objects.post.components[0]"},
        "operation": {"type": "string", "const": "MODIFY_OBJECT_COMPONENT"},
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "Name of the object to modify"},
                "component_index": {"type": "integer", "description": "Index of component within the object"},
                "new_center_x": {"type": "number"},
                "new_center_y": {"type": "number"},
                "new_center_z": {"type": "number"},
                "new_dimensions": {"type": "object"}
            },
            "required": ["object_name", "component_index", "new_center_x", "new_center_y", "new_center_z", "new_dimensions"]
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"]
}

# Add a component to a geometry object
ADD_OBJECT_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "ADD_OBJECT_COMPONENT"},
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "Name of the object to add component to"},
                "component_type": {"type": "string", "enum": ["cylinder", "box", "pyramid", "cone"]},
                "center_x": {"type": "number"},
                "center_y": {"type": "number"},
                "center_z": {"type": "number"},
                "dimensions": {"type": "object"},
                "insert_at": {"type": "integer", "description": "-1 to append"}
            },
            "required": ["object_name", "component_type", "center_x", "center_y", "center_z", "dimensions", "insert_at"]
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"]
}

# Remove a component from a geometry object
REMOVE_OBJECT_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "REMOVE_OBJECT_COMPONENT"},
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "component_index": {"type": "integer"}
            },
            "required": ["object_name", "component_index"]
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"]
}

# Modify assembly grid settings
MODIFY_ASSEMBLY_GRID_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "const": "assembly.grid"},
        "operation": {"type": "string", "const": "MODIFY_ASSEMBLY_GRID"},
        "parameters": {
            "type": "object",
            "properties": {
                "grid_x": {"type": "integer"},
                "grid_y": {"type": "integer"},
                "spacing_x": {"type": "number"},
                "spacing_y": {"type": "number"}
            }
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"]
}

# Modify a composite object's repeat/spacing
MODIFY_COMPOSITE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "MODIFY_COMPOSITE"},
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "Name of the composite object"},
                "repeat_x": {"type": "integer"},
                "repeat_y": {"type": "integer"},
                "repeat_z": {"type": "integer"},
                "spacing_x": {"type": "number"},
                "spacing_y": {"type": "number"},
                "spacing_z": {"type": "number"}
            },
            "required": ["object_name"]
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"]
}

# Combined schema
EDIT_OPERATION_SCHEMA_V2 = {
    "anyOf": [
        MODIFY_OBJECT_COMPONENT_SCHEMA,
        ADD_OBJECT_COMPONENT_SCHEMA,
        REMOVE_OBJECT_COMPONENT_SCHEMA,
        MODIFY_ASSEMBLY_GRID_SCHEMA,
        MODIFY_COMPOSITE_SCHEMA
    ]
}

EDIT_PLAN_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "edit_plan": {
            "type": "array",
            "items": EDIT_OPERATION_SCHEMA_V2
        },
        "summary": {"type": "string"}
    },
    "required": ["edit_plan", "summary"]
}

# ======================================================
# VALIDATION
# ======================================================

def validate_edit_plan_v2(edit_plan: EditPlan) -> tuple:
    """Validate v2 edit plan"""
    errors = []
    
    if "edit_plan" not in edit_plan:
        errors.append("Missing 'edit_plan' field")
        return False, errors
    
    if "summary" not in edit_plan:
        errors.append("Missing 'summary' field")
    
    for i, edit_op in enumerate(edit_plan["edit_plan"]):
        if "operation" not in edit_op:
            errors.append(f"Edit {i}: missing operation")
            continue
        
        op = edit_op["operation"]
        params = edit_op.get("parameters", {})
        
        if op == "MODIFY_OBJECT_COMPONENT":
            required = ["object_name", "component_index", "new_center_x", "new_center_y", "new_center_z", "new_dimensions"]
            for key in required:
                if key not in params:
                    errors.append(f"Edit {i}: MODIFY_OBJECT_COMPONENT missing {key}")
        
        elif op == "ADD_OBJECT_COMPONENT":
            required = ["object_name", "component_type", "center_x", "center_y", "center_z", "dimensions", "insert_at"]
            for key in required:
                if key not in params:
                    errors.append(f"Edit {i}: ADD_OBJECT_COMPONENT missing {key}")
        
        elif op == "REMOVE_OBJECT_COMPONENT":
            for key in ["object_name", "component_index"]:
                if key not in params:
                    errors.append(f"Edit {i}: REMOVE_OBJECT_COMPONENT missing {key}")
        
        elif op == "MODIFY_ASSEMBLY_GRID":
            pass  # All params optional
        
        elif op == "MODIFY_COMPOSITE":
            if "object_name" not in params:
                errors.append(f"Edit {i}: MODIFY_COMPOSITE missing object_name")
        
        else:
            errors.append(f"Edit {i}: Unknown operation {op}")
    
    return len(errors) == 0, errors


def format_edit_plan_for_display(edit_plan: EditPlan) -> str:
    """Format for display"""
    lines = []
    lines.append("=" * 70)
    lines.append("V2 EDIT PLAN")
    lines.append("=" * 70)
    lines.append(f"Summary: {edit_plan.get('summary', 'N/A')}\n")
    
    for i, edit in enumerate(edit_plan.get('edit_plan', []), 1):
        lines.append(f"[{i}] {edit.get('operation', 'UNKNOWN')}")
        lines.append(f"    Target: {edit.get('target', 'N/A')}")
        lines.append(f"    Parameters:")
        for key, value in edit.get('parameters', {}).items():
            lines.append(f"      - {key}: {value}")
        lines.append(f"    Reason: {edit.get('reason', 'N/A')}\n")
    
    lines.append("=" * 70)
    return "\n".join(lines)
