"""
Edit Suggestion Agent - LangGraph Implementation

Rigid template-based agent where LLM only changes numbers within
predefined edit operation structures.

Input: Original unit cell JSON + redesign prompt + optional render images
Output: Structured EditPlan with operations to apply

The agent classifies each request as PARAMETRIC or STRUCTURAL and
generates appropriate edit operations.

Author: Nanoscribe Design Agent Team
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, END
from edit_schema import EditPlan, EDIT_PLAN_SCHEMA


# ==============================================================================
# CONFIGURATION
# ==============================================================================

def find_api_key_file(start_dir: Path, max_levels: int = 5) -> Path:
    """Dynamically search for API.txt in Docs folder."""
    current_dir = start_dir
    for _ in range(max_levels):
        docs_dir = current_dir / "Docs"
        api_file = docs_dir / "API.txt"
        if api_file.exists():
            return api_file
        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent
    raise ValueError(f"API key file not found within {max_levels} parent directories")


SCRIPT_DIR = Path(__file__).parent.resolve()
API_FILE = find_api_key_file(SCRIPT_DIR)
OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)


# ==============================================================================
# STATE
# ==============================================================================

class EditSuggestionState(TypedDict):
    """
    State for edit suggestion agent workflow.
    
    Attributes:
        original_unit_cell: The current unit cell JSON
        redesign_prompt: User's redesign request text
        render_images: Optional list of render image paths for vision
        edit_plan: Generated edit plan
        token_usage: Token consumption metrics
        error: Error message if any
    """
    original_unit_cell: Dict[str, Any]
    redesign_prompt: str
    render_images: List[str]
    edit_plan: EditPlan
    token_usage: Dict[str, int]
    error: str


# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================

EDIT_SUGGESTION_SYSTEM_PROMPT = """You are a geometry edit suggestion expert. You analyze redesign requests and output structured edit plans.

CRITICAL: You must classify each redesign as PARAMETRIC or STRUCTURAL.

=============================================================================
EDIT SCOPE CLASSIFICATION (REQUIRED)
=============================================================================

You MUST set "edit_scope" to one of:

PARAMETRIC:
- Numeric edits only
- Preserves component count and topology
- Legal operations: MODIFY_COMPONENT, MODIFY_PATTERN_MODIFIERS
- Examples: change height, rotate, adjust spacing, thicken elements

STRUCTURAL:
- Changes component count or connectivity
- Partial edits are INVALID
- MUST start with CLEAR_COMPONENTS
- MUST include ADD_COMPONENT operations
- Examples: solid to hollow, solid to lattice, shell creation, interior removal

=============================================================================
OPERATIONS
=============================================================================

OPERATION 1: MODIFY_COMPONENT (PARAMETRIC scope)
For existing components - you ONLY modify:
- component_index: which component (integer)
- new_center_x: new X coordinate (number)
- new_center_y: new Y coordinate (number)  
- new_center_z: new Z coordinate (number)
- new_dimensions: object with ALL the component's dimension keys and new values

Example:
{
  "target": "unit_cell.components[0]",
  "operation": "MODIFY_COMPONENT",
  "parameters": {
    "component_index": 0,
    "new_center_x": 0.0,
    "new_center_y": 0.0,
    "new_center_z": 5.5,
    "new_dimensions": {
      "diameter_um": 2.5,
      "height_um": 12.0
    }
  },
  "reason": "Increasing height by 20 percent"
}

OPERATION 2: CLEAR_COMPONENTS (STRUCTURAL scope - MUST be first)
Removes all existing components. Required as first operation for STRUCTURAL edits.
{
  "target": "unit_cell.components",
  "operation": "CLEAR_COMPONENTS",
  "parameters": {},
  "reason": "Clearing existing geometry for structural redesign"
}

OPERATION 3: ADD_COMPONENT (STRUCTURAL scope)
Add a new component. Required after CLEAR_COMPONENTS in STRUCTURAL edits.
{
  "target": "unit_cell.components",
  "operation": "ADD_COMPONENT",
  "parameters": {
    "component_type": "cylinder",
    "center_x": 0.0,
    "center_y": 0.0,
    "center_z": 5.0,
    "dimensions": {"diameter_um": 1.0, "height_um": 10.0},
    "insert_at": -1
  },
  "reason": "Adding vertical strut"
}

OPERATION 4: REMOVE_COMPONENT (use sparingly)
{
  "target": "unit_cell.components[1]",
  "operation": "REMOVE_COMPONENT",
  "parameters": {
    "component_index": 1
  },
  "reason": "Removing component as requested"
}

OPERATION 5: MODIFY_PATTERN_MODIFIERS (PARAMETRIC scope)
{
  "target": "global_info.pattern_modifiers",
  "operation": "MODIFY_PATTERN_MODIFIERS",
  "parameters": {
    "rotation": 45,
    "flip": "x",
    "row_offset": {"axis": "x", "offset_um": 5, "apply_to": "odd_rows"},
    "clear_modifiers": false
  },
  "reason": "Rotating pattern 45 degrees"
}

=============================================================================
STRUCTURAL EDIT INVARIANTS (ENFORCED BY EXECUTOR)
=============================================================================

If edit_scope is STRUCTURAL, the executor will REJECT your plan unless:
1. First operation is CLEAR_COMPONENTS
2. At least one ADD_COMPONENT operation exists

DO NOT try to make a structural change with only MODIFY_COMPONENT - it will be rejected.

=============================================================================
RULES
=============================================================================

1. For MODIFY_COMPONENT:
   - You MUST include ALL center coordinates (x, y, z) even if not changing
   - You MUST include ALL dimension keys from the original component
   - If not changing a value, use the ORIGINAL value

2. For PARAMETRIC: bias towards MODIFY_COMPONENT, minimize edits

3. For STRUCTURAL: fully redefine the unit cell - no partial changes

4. Each edit MUST have a clear reason

CRITICAL: This is a RIGID TEMPLATE. You are ONLY changing numbers, NOT structure."""


# ==============================================================================
# NODES
# ==============================================================================

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string for GPT-4 vision."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_edit_suggestions(state: EditSuggestionState) -> EditSuggestionState:
    """
    LangGraph node: Generate edit suggestions from redesign prompt.
    
    Args:
        state: Current agent state with unit cell and prompt
        
    Returns:
        Updated state with edit_plan and token_usage
    """
    print("[EDIT SUGGESTION AGENT] Analyzing redesign request...")
    print(f"  Prompt: {state['redesign_prompt'][:60]}...")
    
    # Prepare context
    text_context = f"""CURRENT DESIGN:
{json.dumps(state['original_unit_cell'], indent=2)}

USER REDESIGN REQUEST:
{state['redesign_prompt']}

Analyze the request. First determine if this is PARAMETRIC or STRUCTURAL, then generate the appropriate edit plan."""
    
    message_content = [{"type": "text", "text": text_context}]
    
    # Add images if provided
    if state.get('render_images'):
        render_paths = [Path(img) for img in state['render_images']]
        valid_paths = [p for p in render_paths if p.exists()]
        if valid_paths:
            print(f"[EDIT SUGGESTION AGENT] Analyzing {len(valid_paths)} render images...")
            for img_path in valid_paths:
                base64_image = encode_image_to_base64(img_path)
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                })
    
    # Call LLM
    print("[EDIT SUGGESTION AGENT] Calling LLM...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": EDIT_SUGGESTION_SYSTEM_PROMPT},
            {"role": "user", "content": message_content}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "edit_plan",
                "strict": False,
                "schema": EDIT_PLAN_SCHEMA
            }
        }
    )
    
    # Parse response
    raw_content = response.choices[0].message.content
    try:
        edit_plan = json.loads(raw_content)
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {e}"
        print(f"[EDIT SUGGESTION AGENT] {error_msg}")
        return {**state, "error": error_msg}
    
    if 'edit_plan' not in edit_plan:
        error_msg = f"Response missing edit_plan key"
        print(f"[EDIT SUGGESTION AGENT] ERROR: {error_msg}")
        return {**state, "error": error_msg}
    
    if 'summary' not in edit_plan:
        edit_plan['summary'] = "Edits to satisfy user request"
    
    if 'edit_scope' not in edit_plan:
        edit_plan['edit_scope'] = "PARAMETRIC"
    
    print(f"[EDIT SUGGESTION AGENT] Scope: {edit_plan['edit_scope']}")
    print(f"[EDIT SUGGESTION AGENT] Generated {len(edit_plan['edit_plan'])} edits")
    
    return {
        **state,
        "edit_plan": edit_plan,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }


def build_edit_suggestion_graph() -> StateGraph:
    """Build LangGraph workflow for edit suggestion."""
    workflow = StateGraph(EditSuggestionState)
    workflow.add_node("generate", generate_edit_suggestions)
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def analyze_redesign_prompt(
    original_unit_cell: Dict[str, Any],
    redesign_prompt: str,
    render_images: List[str] = None
) -> EditPlan:
    """
    Main entry point: Analyze redesign prompt and generate edit suggestions.
    
    Args:
        original_unit_cell: Current unit cell JSON
        redesign_prompt: User's redesign request
        render_images: Optional list of render image paths for vision
        
    Returns:
        EditPlan with suggested operations
        
    Raises:
        ValueError: If agent fails to generate valid edit plan
    """
    graph = build_edit_suggestion_graph()
    
    initial_state = EditSuggestionState(
        original_unit_cell=original_unit_cell,
        redesign_prompt=redesign_prompt,
        render_images=render_images or [],
        edit_plan={},
        token_usage={},
        error=""
    )
    
    final_state = graph.invoke(initial_state)
    
    if final_state.get("error"):
        raise ValueError(f"Edit suggestion agent failed: {final_state['error']}")
    
    print(f"[EDIT SUGGESTION AGENT] Tokens used: {final_state['token_usage'].get('total_tokens', 0)}")
    
    return final_state["edit_plan"]


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python edit_suggestion_agent.py <unit_cell.json> <prompt> [images...]")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        original_unit_cell = json.load(f)
    
    edit_plan = analyze_redesign_prompt(original_unit_cell, sys.argv[2], sys.argv[3:] if len(sys.argv) > 3 else None)
    
    from edit_schema import format_edit_plan_for_display
    print("\n" + format_edit_plan_for_display(edit_plan))
    
    with open("edit_suggestions.json", 'w') as f:
        json.dump(edit_plan, f, indent=2)
    print(f"\nEdit plan saved to: edit_suggestions.json")
