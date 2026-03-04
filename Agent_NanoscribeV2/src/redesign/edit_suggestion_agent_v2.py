"""
edit_suggestion_agent_v2_final.py - V2 Named Object Edit Suggestion Agent

LangGraph agent that analyzes redesign prompts and generates edit plans
for the v2 Named Object architecture (objects + assembly, NOT unit_cell).
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, END
from edit_schema_v2 import EditPlan, EDIT_PLAN_SCHEMA_V2

# ======================================================
# CONFIG  
# ======================================================

def find_api_key_file(start_dir: Path, max_levels: int = 5) -> Path:
    """Dynamically search for API.txt in Docs folder"""
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

# ======================================================
# STATE
# ======================================================

class EditSuggestionState(TypedDict):
    """State for v2 edit suggestion agent"""
    design_json: Dict[str, Any]  # V2 format with 'objects' and 'assembly'
    redesign_prompt: str
    render_images: List[str]
    edit_plan: EditPlan
    token_usage: Dict[str, int]
    error: str

# ======================================================
# V2 SYSTEM PROMPT
# ======================================================

EDIT_SUGGESTION_SYSTEM_PROMPT_V2 = """You are a geometry edit suggestion expert for the V2 Named Object Architecture.

CRITICAL: You work with V2 format (objects + assembly), NOT the legacy format (unit_cell + global_info).

V2 DESIGN STRUCTURE:
{
  "job_name": "...",
  "objects": {
    "object_name": {
      "type": "geometry",
      "components": [
        {"type": "cylinder", "center": [x, y, z], "dimensions": {"diameter_um": ..., "height_um": ...}}
      ]
    },
    "another_object": {
      "type": "composite",
      "uses": "object_name",
      "repeat": {"x": 3, "y": 3, "z": 3},
      "spacing_um": {"x": 10, "y": 10, "z": 10}
    }
  },
  "assembly": {
    "type": "grid",
    "grid": {"x": 10, "y": 10},
    "spacing_um": {"x": 15, "y": 15},
    "default_object": "object_name"
  }
}

AVAILABLE OPERATIONS:

1. MODIFY_OBJECT_COMPONENT - Modify a component within a geometry object
{
  "target": "objects.post.components[0]",
  "operation": "MODIFY_OBJECT_COMPONENT",
  "parameters": {
    "object_name": "post",
    "component_index": 0,
    "new_center_x": 0.0,
    "new_center_y": 0.0,
    "new_center_z": 6.0,
    "new_dimensions": {"diameter_um": 3.0, "height_um": 12.0}
  },
  "reason": "Increasing height by 50%"
}

2. ADD_OBJECT_COMPONENT - Add a new component to a geometry object
{
  "target": "objects.nailhead.components",
  "operation": "ADD_OBJECT_COMPONENT",
  "parameters": {
    "object_name": "nailhead",
    "component_type": "cylinder",
    "center_x": 0.0,
    "center_y": 0.0,
    "center_z": 10.0,
    "dimensions": {"diameter_um": 5.0, "height_um": 1.0},
    "insert_at": -1
  },
  "reason": "Adding cap on top"
}

3. REMOVE_OBJECT_COMPONENT - Remove a component from a geometry object
{
  "target": "objects.nailhead.components[1]",
  "operation": "REMOVE_OBJECT_COMPONENT",
  "parameters": {
    "object_name": "nailhead",
    "component_index": 1
  },
  "reason": "Removing the top disk"
}

4. MODIFY_ASSEMBLY_GRID - Change assembly grid size or spacing
{
  "target": "assembly.grid",
  "operation": "MODIFY_ASSEMBLY_GRID",
  "parameters": {
    "grid_x": 5,
    "grid_y": 5,
    "spacing_x": 20.0,
    "spacing_y": 20.0
  },
  "reason": "Reducing array size and increasing spacing"
}

5. MODIFY_COMPOSITE - Change a composite object's repeat pattern
{
  "target": "objects.meta_atom",
  "operation": "MODIFY_COMPOSITE",
  "parameters": {
    "object_name": "meta_atom",
    "repeat_x": 5,
    "repeat_y": 5,
    "repeat_z": 5,
    "spacing_x": 10.0,
    "spacing_y": 10.0,
    "spacing_z": 10.0
  },
  "reason": "Expanding the lattice"
}

RULES:
1. For MODIFY_OBJECT_COMPONENT:
   - Include ALL center coordinates even if not changing
   - Include ALL dimension keys from the original component
   - Use ORIGINAL values for unchanged fields

2. For REMOVE_OBJECT_COMPONENT:
   - When removing MULTIPLE components from the SAME object, you MUST remove from HIGHEST index to LOWEST
   - This prevents index shifting errors
   - Example: To remove indices [1, 3, 4], generate edits in order: remove 4, remove 3, remove 1

3. BIAS towards MODIFY_OBJECT_COMPONENT over ADD/REMOVE
4. MINIMIZE number of edits
5. Each edit MUST have a clear reason

FORBIDDEN:
- Do NOT output 'unit_cell' - use 'objects' instead
- Do NOT output 'global_info' - use 'assembly' instead

You are ONLY changing numbers in a RIGID TEMPLATE. Every field listed above is REQUIRED."""

# ======================================================
# NODES
# ======================================================

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_edit_suggestions_v2(state: EditSuggestionState) -> EditSuggestionState:
    """LangGraph node: Generate v2 edit suggestions"""
    print("[EDIT SUGGESTION AGENT V2] Analyzing redesign request...")
    
    # Prepare context
    text_context = f"""CURRENT V2 DESIGN:
{json.dumps(state['design_json'], indent=2)}

USER REDESIGN REQUEST:
{state['redesign_prompt']}

Analyze the request and suggest the MINIMAL set of V2 edits needed.
Remember: Use objects and assembly, NOT unit_cell and global_info."""
    
    message_content = [{"type": "text", "text": text_context}]
    
    # Add images if provided
    if state.get('render_images'):
        render_paths = [Path(img) for img in state['render_images']]
        valid_paths = [p for p in render_paths if p.exists()]
        if valid_paths:
            print(f"[EDIT SUGGESTION AGENT V2] Analyzing {len(valid_paths)} render images...")
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
    print("[EDIT SUGGESTION AGENT V2] Calling LLM...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": EDIT_SUGGESTION_SYSTEM_PROMPT_V2},
            {"role": "user", "content": message_content}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "edit_plan_v2",
                "strict": False,
                "schema": EDIT_PLAN_SCHEMA_V2
            }
        }
    )
    
    # Parse response
    raw_content = response.choices[0].message.content
    try:
        edit_plan = json.loads(raw_content)
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {e}"
        print(f"[EDIT SUGGESTION AGENT V2] {error_msg}")
        return {**state, "error": error_msg}
    
    if 'edit_plan' not in edit_plan:
        error_msg = "Response missing edit_plan key"
        print(f"[EDIT SUGGESTION AGENT V2] ERROR: {error_msg}")
        return {**state, "error": error_msg}
    
    if 'summary' not in edit_plan:
        edit_plan['summary'] = "Edits to satisfy user request"
    
    print(f"[EDIT SUGGESTION AGENT V2] Generated {len(edit_plan['edit_plan'])} edits")
    
    return {
        **state,
        "edit_plan": edit_plan,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }


def build_edit_suggestion_graph_v2() -> StateGraph:
    """Build LangGraph workflow for v2"""
    workflow = StateGraph(EditSuggestionState)
    workflow.add_node("generate", generate_edit_suggestions_v2)
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def analyze_redesign_prompt_v2(
    design_json: Dict[str, Any],
    redesign_prompt: str,
    render_images: List[str] = None
) -> EditPlan:
    """
    Analyze redesign prompt and generate v2 edit suggestions.
    
    Args:
        design_json: V2 design with 'objects' and 'assembly'
        redesign_prompt: User's redesign request
        render_images: Optional list of render image paths
    
    Returns:
        V2 EditPlan with suggested operations
    """
    graph = build_edit_suggestion_graph_v2()
    
    initial_state = EditSuggestionState(
        design_json=design_json,
        redesign_prompt=redesign_prompt,
        render_images=render_images or [],
        edit_plan={},
        token_usage={},
        error=""
    )
    
    final_state = graph.invoke(initial_state)
    
    if final_state.get("error"):
        raise ValueError(f"V2 Edit suggestion agent failed: {final_state['error']}")
    
    print(f"[EDIT SUGGESTION AGENT V2] Tokens used: {final_state['token_usage'].get('total_tokens', 0)}")
    
    return final_state["edit_plan"]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python edit_suggestion_agent_v2_final.py <design.json> <prompt> [images...]")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        design_json = json.load(f)
    
    edit_plan = analyze_redesign_prompt_v2(design_json, sys.argv[2], sys.argv[3:] if len(sys.argv) > 3 else None)
    
    from edit_schema_v2 import format_edit_plan_for_display
    print("\n" + format_edit_plan_for_display(edit_plan))
    
    with open("edit_suggestions_v2.json", 'w') as f:
        json.dump(edit_plan, f, indent=2)
    print(f"\nV2 Edit plan saved to: edit_suggestions_v2.json")
