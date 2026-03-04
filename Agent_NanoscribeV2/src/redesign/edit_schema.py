"""
Edit Schema - Structured Edit Operations for Redesign

Defines the rigid template for edit operations. The LLM only changes
numeric values within these fixed structures.

Edit Operations:
    - MODIFY_COMPONENT: Change dimensions/position of existing component
    - ADD_COMPONENT: Add a new geometric primitive
    - REMOVE_COMPONENT: Remove an existing component
    - MODIFY_PATTERN_MODIFIERS: Change rotation, offset, flip
    - CLEAR_COMPONENTS: Remove all components (for structural edits)

Edit Scopes:
    - PARAMETRIC: Numeric changes only, preserves topology
    - STRUCTURAL: Changes component count/connectivity

Author: Nanoscribe Design Agent Team
"""

from typing import TypedDict, Any, Dict, List


# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class EditOperation(TypedDict):
    """
    A single edit operation with fixed structure.
    
    Attributes:
        target: Path to the target element (e.g., "unit_cell.components[0]")
        operation: Operation type (MODIFY_COMPONENT, ADD_COMPONENT, etc.)
        parameters: Operation-specific parameters
        reason: Human-readable explanation
    """
    target: str
    operation: str
    parameters: Dict[str, Any]
    reason: str


class EditPlan(TypedDict):
    """
    Complete edit plan with scope classification.
    
    Attributes:
        edit_scope: "PARAMETRIC" or "STRUCTURAL"
        edit_plan: List of EditOperation dicts
        summary: Human-readable summary of changes
    """
    edit_scope: str
    edit_plan: List[EditOperation]
    summary: str


# ==============================================================================
# JSON SCHEMAS FOR LLM OUTPUT
# ==============================================================================
# These schemas are passed to GPT to enforce structured output.

MODIFY_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "MODIFY_COMPONENT"},
        "parameters": {
            "type": "object",
            "properties": {
                "component_index": {"type": "integer"},
                "new_center_x": {"type": "number"},
                "new_center_y": {"type": "number"},
                "new_center_z": {"type": "number"},
                "new_dimensions": {"type": "object"}
            },
            "required": ["component_index", "new_center_x", "new_center_y", "new_center_z", "new_dimensions"],
            "additionalProperties": False
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"],
    "additionalProperties": False
}

ADD_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "ADD_COMPONENT"},
        "parameters": {
            "type": "object",
            "properties": {
                "component_type": {"type": "string", "enum": ["cylinder", "box", "sphere", "cone", "pyramid"]},
                "center_x": {"type": "number"},
                "center_y": {"type": "number"},
                "center_z": {"type": "number"},
                "dimensions": {"type": "object"},
                "insert_at": {"type": "integer"}
            },
            "required": ["component_type", "center_x", "center_y", "center_z", "dimensions", "insert_at"],
            "additionalProperties": False
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"],
    "additionalProperties": False
}

REMOVE_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "REMOVE_COMPONENT"},
        "parameters": {
            "type": "object",
            "properties": {
                "component_index": {"type": "integer"}
            },
            "required": ["component_index"],
            "additionalProperties": False
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"],
    "additionalProperties": False
}

MODIFY_PATTERN_MODIFIERS_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "MODIFY_PATTERN_MODIFIERS"},
        "parameters": {
            "type": "object",
            "properties": {
                "rotation": {"type": "number"},
                "flip": {"type": "string", "enum": ["x", "y", "xy", "none"]},
                "row_offset": {
                    "type": "object",
                    "properties": {
                        "axis": {"enum": ["x"]},
                        "offset_um": {"type": "number"},
                        "apply_to": {"enum": ["odd_rows", "even_rows"]}
                    },
                    "required": ["axis", "offset_um", "apply_to"]
                },
                "clear_modifiers": {"type": "boolean"}
            },
            "additionalProperties": False
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"],
    "additionalProperties": False
}

CLEAR_COMPONENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "operation": {"type": "string", "const": "CLEAR_COMPONENTS"},
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "reason": {"type": "string"}
    },
    "required": ["target", "operation", "parameters", "reason"],
    "additionalProperties": False
}

EDIT_OPERATION_SCHEMA = {
    "anyOf": [
        MODIFY_COMPONENT_SCHEMA,
        ADD_COMPONENT_SCHEMA,
        REMOVE_COMPONENT_SCHEMA,
        MODIFY_PATTERN_MODIFIERS_SCHEMA,
        CLEAR_COMPONENTS_SCHEMA
    ]
}

EDIT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "edit_scope": {
            "type": "string",
            "enum": ["PARAMETRIC", "STRUCTURAL"],
            "description": "PARAMETRIC for numeric-only edits, STRUCTURAL for topology changes"
        },
        "edit_plan": {
            "type": "array",
            "items": EDIT_OPERATION_SCHEMA
        },
        "summary": {"type": "string"}
    },
    "required": ["edit_scope", "edit_plan", "summary"],
    "additionalProperties": False
}


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_edit_plan(edit_plan: EditPlan) -> tuple[bool, List[str]]:
    """
    Validate edit plan including scope-based invariants.
    
    Checks:
        - Required fields present
        - Valid edit_scope value
        - STRUCTURAL edits start with CLEAR_COMPONENTS
        - STRUCTURAL edits include ADD_COMPONENT
        - Each operation has required parameters
    
    Args:
        edit_plan: The edit plan to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if "edit_plan" not in edit_plan:
        errors.append("Missing 'edit_plan' field")
        return False, errors
    
    if "summary" not in edit_plan:
        errors.append("Missing 'summary' field")
    
    # Validate edit_scope
    edit_scope = edit_plan.get("edit_scope", "PARAMETRIC")
    if edit_scope not in ["PARAMETRIC", "STRUCTURAL"]:
        errors.append(f"Invalid edit_scope: {edit_scope}. Must be PARAMETRIC or STRUCTURAL")
    
    # Structural invariant enforcement
    if edit_scope == "STRUCTURAL":
        ops = edit_plan["edit_plan"]
        
        # Invariant 1: STRUCTURAL edits must start with CLEAR_COMPONENTS
        if not ops or ops[0].get("operation") != "CLEAR_COMPONENTS":
            errors.append("STRUCTURAL edit must start with CLEAR_COMPONENTS operation")
        
        # Invariant 2: STRUCTURAL edits must contain at least one ADD_COMPONENT
        has_add = any(op.get("operation") == "ADD_COMPONENT" for op in ops)
        if not has_add:
            errors.append("STRUCTURAL edit must contain at least one ADD_COMPONENT operation")
    
    # Validate individual operations
    for i, edit_op in enumerate(edit_plan["edit_plan"]):
        if "operation" not in edit_op:
            errors.append(f"Edit {i}: missing operation")
            continue
            
        if edit_op["operation"] == "MODIFY_COMPONENT":
            required = ["component_index", "new_center_x", "new_center_y", "new_center_z", "new_dimensions"]
            for key in required:
                if key not in edit_op["parameters"]:
                    errors.append(f"Edit {i}: MODIFY_COMPONENT missing {key}")
        
        elif edit_op["operation"] == "ADD_COMPONENT":
            required = ["component_type", "center_x", "center_y", "center_z", "dimensions", "insert_at"]
            for key in required:
                if key not in edit_op["parameters"]:
                    errors.append(f"Edit {i}: ADD_COMPONENT missing {key}")
        
        elif edit_op["operation"] == "REMOVE_COMPONENT":
            if "component_index" not in edit_op["parameters"]:
                errors.append(f"Edit {i}: REMOVE_COMPONENT missing component_index")

        elif edit_op["operation"] == "MODIFY_PATTERN_MODIFIERS":
             pass  # Optional params, no strict requirements
        
        elif edit_op["operation"] == "CLEAR_COMPONENTS":
            pass  # No parameters required
    
    return len(errors) == 0, errors


def format_edit_plan_for_display(edit_plan: EditPlan) -> str:
    """
    Format edit plan for human-readable display.
    
    Args:
        edit_plan: The edit plan to format
        
    Returns:
        Multi-line string representation
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EDIT PLAN")
    lines.append("=" * 70)
    lines.append(f"Scope: {edit_plan.get('edit_scope', 'PARAMETRIC')}")
    lines.append(f"Summary: {edit_plan['summary']}\n")
    
    for i, edit in enumerate(edit_plan['edit_plan'], 1):
        lines.append(f"[{i}] {edit['operation']}")
        lines.append(f"    Target: {edit['target']}")
        if edit['parameters']:
            lines.append(f"    Parameters:")
            for key, value in edit['parameters'].items():
                lines.append(f"      - {key}: {value}")
        lines.append(f"    Reason: {edit['reason']}\n")
    
    lines.append("=" * 70)
    return "\n".join(lines)
