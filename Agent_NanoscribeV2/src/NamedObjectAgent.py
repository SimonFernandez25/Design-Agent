"""
Named Object Agent - High-Reasoning Geometry Composition Agent

This agent reasons in OBJECT SPACE, not primitive space.
It defines named objects and assemblies that are later reduced to primitives.

Key responsibilities:
- Define named objects (geometry and composite types)
- Decide composition strategy (repetition, spacing)
- Create object library + assembly specification
- Never think in primitives directly unless defining base geometry objects

The agent output is NOT executable directly. It must pass through
the reduction engine to produce flat primitives.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Dict
from openai import OpenAI
from langgraph.graph import StateGraph, END

from schemas import (
    validate_object_library,
    validate_assembly,
    OBJECT_LIBRARY_SCHEMA,
    ASSEMBLY_SCHEMA,
    v2_structural_gate
)
from reduction_engine import reduce_assembly, validate_reduced_output


# ======================================================
# CONFIG
# ======================================================

def find_api_key_file(start_dir: Path, max_levels: int = 5) -> Path:
    """Dynamically search for API.txt in Docs folder."""
    current_dir = start_dir
    
    for _ in range(max_levels):
        docs_dir = current_dir / "Docs"
        api_file = docs_dir / "API.txt"
        
        if api_file.exists():
            print(f"[CONFIG] Found API key at: {api_file}")
            return api_file
        
        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent
    
    raise ValueError(
        f"API key file (Docs/API.txt) not found within {max_levels} parent directories of {start_dir}"
    )


SCRIPT_DIR = Path(__file__).parent.parent.resolve()
API_FILE = find_api_key_file(SCRIPT_DIR)
OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)


# ======================================================
# STATE
# ======================================================

class AgentState(TypedDict):
    prompt: str
    category: str
    design: dict       # Named object library + assembly
    reduced: dict      # Flat primitives after reduction
    output_path: str
    token_usage: dict
    errors: List[str]


# ======================================================
# JSON SCHEMA FOR LLM OUTPUT
# ======================================================

NAMED_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "job_name": {
            "type": "string",
            "description": "Short descriptive name for this design"
        },
        "objects": {
            "type": "object",
            "description": "Named object library. Keys are object names.",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["geometry", "composite"],
                        "description": "geometry = contains primitives, composite = references another object"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description"
                    },
                    "components": {
                        "type": "array",
                        "description": "For geometry type: array of primitives",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["box", "cylinder", "sphere", "pyramid", "cone"]
                                },
                                "center": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3
                                },
                                "dimensions": {
                                    "type": "object",
                                    "description": "Explicit dimensions (box: x_um, y_um, z_um; cylinder: diameter_um, height_um; pyramid: base_width_um, height_um; cone: base_diameter_um, height_um)"
                                },
                                "construction": {
                                    "type": "object",
                                    "properties": {
                                        "method": {"type": "string", "enum": ["stacked_boxes", "stacked_cylinders"]},
                                        "layers": {"type": "integer", "description": "Number of layers to stack during expansion"},
                                        "top_width_um": {"type": "number", "description": "Top width for pyramids (0 for pointed)"},
                                        "top_diameter_um": {"type": "number", "description": "Top diameter for cones (0 for pointed)"}
                                    },
                                    "required": ["method", "layers"]
                                }
                            },
                            "required": ["type", "center", "dimensions"]
                        }
                    },
                    "uses": {
                        "type": "string",
                        "description": "For composite type: name of object to use"
                    },
                    "repeat": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "minimum": 1},
                            "y": {"type": "integer", "minimum": 1},
                            "z": {"type": "integer", "minimum": 1}
                        },
                        "description": "For composite: repetition count in each axis"
                    },
                    "spacing_um": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"}
                        },
                        "description": "For composite: spacing between repetitions"
                    }
                },
                "required": ["type"]
            }
        },
        "assembly": {
            "type": "object",
            "description": "How named objects are arranged into the final design",
            "oneOf": [
                {
                    "properties": {
                        "type": {"const": "grid", "description": "Regular Cartesian grid arrangement"},
                        "grid": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer", "minimum": 1},
                                "y": {"type": "integer", "minimum": 1}
                            },
                            "required": ["x", "y"]
                        },
                        "spacing_um": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"}
                            },
                            "required": ["x", "y"]
                        },
                        "default_object": {"type": "string", "description": "Name of object to place in grid"},
                        "mapping": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                            "description": "Optional row-major mapping for heterogeneous grids"
                        }
                    },
                    "required": ["type", "grid", "spacing_um", "default_object"],
                    "additionalProperties": False
                },
                {
                    "properties": {
                        "type": {"const": "explicit", "description": "Arbitrary placement layout"},
                        "placements": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "object": {"type": "string", "description": "Name of object to place"},
                                    "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "[x, y, z] position"},
                                    "rotate_z_deg": {"type": "number", "description": "Optional rotation around build axis"}
                                },
                                "required": ["object", "position"]
                            }
                        }
                    },
                    "required": ["type", "placements"],
                    "additionalProperties": False
                }
            ]
        }
    },
    "required": ["job_name", "objects", "assembly"],
    "additionalProperties": False
}


# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = """You are a high-level geometry composition expert. You design metamaterial structures using NAMED OBJECTS and ASSEMBLIES.

ABSOLUTE RULES (NON-NEGOTIABLE):
- You must NOT output unit_cell, global_info, or any legacy fields
- All geometry must be defined using NAMED OBJECTS
- Your output must include: objects dictionary AND assembly definition
- DIMENSION RULE: You must use explicit 'x_um', 'y_um', 'z_um' for all box dimensions.
- FORBIDDEN: Do NOT use 'length', 'width', 'depth', 'height'. These are ambiguous and will cause failure.
- These rules are absolute - violating them causes immediate failure

CORE PRINCIPLE: Think in ATOMS and ASSEMBLIES.
- ATOM: The base building block (e.g., a single pyramid or a single pillar).
- ASSEMBLY: The final layout (e.g., a 10x10 array of pyramids).
- RULE: Repetition belongs in the ASSEMBLY.
- COMPOSITES are for INTERNAL structure only (e.g., a single "cell" made of 4 struts).
- REPETITION HYGIENE: NEVER define a global array (e.g., 10x10) as a named object and then repeat it in the assembly. This creates 10,000 primitives and fails the system.

OBJECT TYPES:
1. "geometry" objects: Contain raw primitives (box, cylinder) ONLY
   - Use for base building blocks
   - Primitives have: type, center [x,y,z], dimensions
   - May NOT reference other objects

2. "composite" objects: Reference other objects with repetition
   - uses: name of referenced object
   - repeat: {x, y, z} repetition counts
   - spacing_um: {x, y, z} spacing between repetitions
   - May NOT contain primitives directly

ASSEMBLY:
- Defines WHERE objects are placed (not how they're built)
- type: "grid" for regular patterns
- type: "explicit" for freeform/arbitrary placements (requires "placements" list)
- grid: {x, y} number of cells (for grid)
- spacing_um: {x, y} spacing between cells (for grid)
- default_object: object name for all cells (for grid)
- mapping: [[...], [...]] for heterogeneous layouts (for grid, row-major, mapping[y][x])
- placements: [{"object": "name", "position": [x, y, z], "rotate_z_deg": 0}, ...] (for explicit)

DESIGN RULES:
1. Start with ATOM objects (geometry type) - the smallest physical part.
2. If the atom is complex (multiple parts), group them in a geometry object.
3. Use a COMPOSITE only if you have internal repetition within a single unit (e.g. 4 pillars in one cell).
4. Use ASSEMBLY for the final grid or array.

- THE GOLDEN RULE: If the prompt asks for a "10x10 array of pyramids", you define ONE pyramid in `objects` and set `grid: {x: 10, y: 10}` in `assembly`.
- DO NOT use "repeat: {x: 10, y: 10}" inside an object if that object is going to be in the assembly grid.

FABRICATION RULES (CRITICAL):
- BOX Dimensions: Must be {x_um: <val>, y_um: <val>, z_um: <val>}
- CYLINDER Dimensions: {diameter_um: <val>, height_um: <val>} (Z-axis aligned)
- PYRAMIDS: You must use the pyramid type. Do NOT abstract it into boxes yourself.
  * dimensions: {base_width_um: <val>, height_um: <val>}
  * construction: {method: "stacked_boxes", layers: <val>, top_width_um: <val>}
  * RULE: Use layers = height_um (one layer per micrometer) for standard resolution.
- CONES: You must use the cone type. Do NOT abstract it into cylinders yourself.
  * dimensions: {base_diameter_um: <val>, height_um: <val>}
  * construction: {method: "stacked_cylinders", layers: <val>, top_diameter_um: <val>}
  * RULE: Use layers = height_um (one layer per micrometer) for standard resolution.

COORDINATE SYSTEM:
- All dimensions in micrometers
- Z-axis is build direction (bottom to top)
- Calculate centers correctly (center of geometry, not base)

FORBIDDEN OUTPUTS (AUTOMATIC FAILURE):
Any output that includes:
- "unit_cell" (LEGACY V1 - FORBIDDEN)
- "global_info" (LEGACY V1 - FORBIDDEN)
- "primitives" at the top level
- Nested 10x10 repetition (DUPLICATION ERROR)
- "length", "width", "depth" in dimensions (AMBIGUOUS)

REQUIRED OUTPUT STRUCTURE:
{
  "job_name": "...",
  "objects": { ... },
  "assembly": { ... }
}

EXAMPLE OUTPUT (10x10 Array of Pyramids):
{
  "job_name": "pyramid_array",
  "objects": {
    "pyramid_atom": {
      "type": "geometry",
      "description": "Single sharp pyramid",
      "components": [
        {
          "type": "pyramid", 
          "center": [0, 0, 10], 
          "dimensions": {"base_width_um": 10, "height_um": 20},
          "construction": {"method": "stacked_boxes", "layers": 20, "top_width_um": 0}
        }
      ]
    },
    "double_pillar": {
        "type": "composite",
        "description": "Two pillars spaced together to form one cell",
        "uses": "single_pillar",
        "repeat": {"x": 2, "y": 1, "z": 1},
        "spacing_um": {"x": 5, "y": 0, "z": 0}
    }
  },
  "assembly": {
    "type": "grid",
    "grid": {"x": 10, "y": 10},
    "spacing_um": {"x": 20, "y": 20},
    "default_object": "pyramid_atom"
  }
}

NOTE: Notice how the pyramid_atom does NOT have a 10x10 repeat inside it. The repeat is in the ASSEMBLY.

Output valid JSON matching the schema. Think hierarchically."""


# ======================================================
# NODES
# ======================================================

def design_geometry(state: AgentState) -> AgentState:
    """Call LLM to design named object structure."""
    print(f"[DESIGNING] Processing prompt...")
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["prompt"]}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "named_object_design",
                "strict": False,  # OpenAI strict requires additionalProperties:false everywhere, incompatible with dynamic object names
                "schema": NAMED_OBJECT_SCHEMA
            }
        }
    )
    
    design = json.loads(response.choices[0].message.content)
    
    # DEBUG: Show what the LLM actually output
    print(f"[DEBUG] LLM output keys: {list(design.keys())}")
    if 'unit_cell' in design or 'global_info' in design:
        print(f"[FATAL] LLM OUTPUT HAS LEGACY FIELDS - SCHEMA ENFORCEMENT FAILED!")
        print(f"[FATAL] This should NOT be possible with strict:true and additionalProperties:false")
    
    state["design"] = design
    state["token_usage"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    print(f"[SUCCESS] Design created: {design['job_name']}")
    print(f"  Objects: {list(design['objects'].keys())}")
    print(f"  Tokens: {state['token_usage']['total_tokens']}")
    
    return state


def validate_design(state: AgentState) -> AgentState:
    """Validate the design against v2 schemas. Fail fast on legacy fields."""
    print("[VALIDATING] Running v2 structural gate...")
    
    design = state["design"]
    
    # CRITICAL: Structural gate runs FIRST - fails immediately on legacy fields
    try:
        v2_structural_gate(design)
        print("[OK] Structural gate passed")
    except ValueError as e:
        # This is a hard failure - do not proceed
        print(f"[FATAL] {e}")
        state["errors"] = [str(e)]
        raise  # Re-raise to stop the pipeline
    
    print("[VALIDATING] Checking schema compliance...")
    errors = []
    
    # Validate object library
    lib_errors = validate_object_library({"objects": design.get("objects", {})})
    errors.extend(lib_errors)
    
    # Validate assembly
    object_names = list(design.get("objects", {}).keys())
    asm_errors = validate_assembly(
        {"assembly": design.get("assembly", {})},
        object_names
    )
    errors.extend(asm_errors)
    
    state["errors"] = errors
    
    if errors:
        print(f"[WARNING] Validation issues found:")
        for err in errors[:5]:  # Show first 5
            print(f"  - {err}")
    else:
        print("[OK] Design validated successfully")
    
    return state


def reduce_design(state: AgentState) -> AgentState:
    """Reduce named objects to flat primitives."""
    print("[REDUCING] Converting to flat primitives...")
    
    design = state["design"]
    
    try:
        reduced = reduce_assembly(design)
        
        # Validate reduced output
        reduce_errors = validate_reduced_output(reduced)
        if reduce_errors:
            state["errors"].extend(reduce_errors)
        
        state["reduced"] = reduced
        
        print(f"[OK] Reduced to {reduced['metadata']['num_primitives']} primitives")
        
    except Exception as e:
        error_msg = f"Reduction failed: {str(e)}"
        state["errors"].append(error_msg)
        state["reduced"] = {"primitives": [], "metadata": {"error": str(e)}}
        print(f"[ERROR] {error_msg}")
    
    return state


def save_output(state: AgentState) -> AgentState:
    """Save design and reduced output."""
    job_name = state["design"]["job_name"]
    category = state.get("category", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    base_dir = SCRIPT_DIR / "Outputs"
    if category:
        output_dir = base_dir / f"{category}_{job_name}_{timestamp}"
    else:
        output_dir = base_dir / f"{job_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save named object design
    design_file = output_dir / "design.json"
    design_output = {
        "metadata": {
            "category": category,
            "timestamp": timestamp,
            "model": "openai/gpt-4o-mini",
            "tokens": state["token_usage"],
            "errors": state["errors"]
        },
        "prompt": state["prompt"],
        **state["design"]
    }
    with open(design_file, "w") as f:
        json.dump(design_output, f, indent=2)
    print(f"[SAVED] Design: {design_file}")
    
    # Save reduced primitives (for endpoint generator compatibility)
    reduced_file = output_dir / "reduced.json"
    with open(reduced_file, "w") as f:
        json.dump(state["reduced"], f, indent=2)
    print(f"[SAVED] Reduced: {reduced_file}")
    

    
    state["output_path"] = str(output_dir)
    
    # Run downstream pipeline
    _run_pipeline(output_dir, state)
    
    return state



def _run_pipeline(output_dir: Path, state: AgentState):
    """
    Run downstream fabrication pipeline using V2 modules.
    
    V2 Pipeline:
    1. Read reduced.json (flat primitives from reduction engine)
    2. Enrich primitives via manufacturing_agent (metrics, risk, local params)
    3. Generate endpoints using endpoint_generator_v2 (per-primitive local params)
    4. Generate GWL files using gwl_serializer
    5. Generate renders using render_generator_v2

    NO unit_cell.json or legacy formats involved.
    """
    try:
        print("[PIPELINE V2] Starting automatic generation...")

        import endpoint_generator_v2
        import gwl_serializer
        import render_generator_v2
        import manufacturing_agent
        
        # Find parameters file
        param_file = None
        curr = output_dir
        for _ in range(4):
            check = curr / "PrintParameters.txt"
            if check.exists():
                param_file = check
                break
            curr = curr.parent
        
        if not param_file:
            print("[PIPELINE V2] Warning: PrintParameters.txt not found, skipping.")
            return
        
        # Load reduced primitives (V2 format)
        reduced_file = output_dir / "reduced.json"
        if not reduced_file.exists():
            print(f"[PIPELINE V2] ERROR: reduced.json not found at {reduced_file}")
            return
        
        with open(reduced_file) as f:
            reduced_data = json.load(f)

        print(f"[PIPELINE V2] Loaded {reduced_data['metadata']['num_primitives']} primitives")

        # Enrich primitives with per-primitive manufacturing parameters
        print("[PIPELINE V2] Running manufacturing agent...")
        enriched_data = manufacturing_agent.enrich_primitives(reduced_data, output_dir)
        print(f"[PIPELINE V2] Enrichment complete.")

        # Generate endpoints using local per-primitive parameters
        print_params = endpoint_generator_v2.load_print_parameters(param_file)
        endpoint_data = endpoint_generator_v2.generate_endpoint_json_v2(enriched_data, print_params)
        
        endpoint_file = output_dir / "output.json"
        with open(endpoint_file, 'w') as f:
            json.dump(endpoint_data, f, indent=2)
        print(f"[PIPELINE V2] Generated Endpoints: {endpoint_file.name}")
        
        # Generate GWL
        gwl_dir = output_dir / "GWL"
        gwl_params = gwl_serializer.load_gwl_parameters(param_file)
        gwl_files = gwl_serializer.generate_gwl_files(endpoint_data, gwl_params, gwl_dir)
        
        master_gwl = gwl_dir / f"{state['design']['job_name']}_master.gwl"
        gwl_serializer.generate_master_gwl(gwl_files, gwl_params, master_gwl)
        print(f"[PIPELINE V2] Generated GWL: {gwl_dir.name}/")
        
        # Generate Renders using v2 object-aware renderer
        render_dir = output_dir / "Renders"
        design_file = output_dir / "design.json"
        
        if design_file.exists():
            with open(design_file) as f:
                design = json.load(f)
            render_generator_v2.generate_object_aware_renders(design, render_dir, param_file)
            print(f"[PIPELINE V2] Generated Renders: {render_dir.name}/")
        else:
            print(f"[PIPELINE V2] Warning: design.json not found, skipping renders")
        
    except Exception as e:
        print(f"[PIPELINE V2] ERROR: {e}")
        import traceback
        traceback.print_exc()


# ======================================================
# GRAPH
# ======================================================

def build_graph():
    """Construct LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("design", design_geometry)
    workflow.add_node("validate", validate_design)
    workflow.add_node("reduce", reduce_design)
    workflow.add_node("save", save_output)
    
    workflow.set_entry_point("design")
    workflow.add_edge("design", "validate")
    workflow.add_edge("validate", "reduce")
    workflow.add_edge("reduce", "save")
    workflow.add_edge("save", END)
    
    return workflow.compile()


# ======================================================
# MAIN
# ======================================================

def run_design(prompt: str, category: str = "") -> Dict:
    """Run the named object design agent."""
    graph = build_graph()
    
    initial_state = {
        "prompt": prompt,
        "category": category,
        "design": {},
        "reduced": {},
        "output_path": "",
        "token_usage": {},
        "errors": []
    }
    
    final_state = graph.invoke(initial_state)
    
    return {
        "job_name": final_state["design"]["job_name"],
        "output_path": final_state["output_path"],
        "num_objects": len(final_state["design"]["objects"]),
        "num_primitives": final_state["reduced"]["metadata"]["num_primitives"],
        "tokens": final_state["token_usage"]["total_tokens"],
        "errors": final_state["errors"]
    }


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("NAMED OBJECT GEOMETRY AGENT")
    print("=" * 70)
    
    # Load prompt
    prompt_file = SCRIPT_DIR / "prompt.txt"
    if not prompt_file.exists():
        print(f"ERROR: prompt.txt not found at {prompt_file}")
        sys.exit(1)
    
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    
    # Run
    result = run_design(prompt)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Job Name: {result['job_name']}")
    print(f"Objects: {result['num_objects']}")
    print(f"Primitives: {result['num_primitives']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Output: {result['output_path']}")
    if result['errors']:
        print(f"Errors: {len(result['errors'])}")
    print("=" * 70)
