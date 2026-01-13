
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Dict
from openai import OpenAI
from langgraph.graph import StateGraph, END

# ======================================================
# CONFIG
# ======================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
AGENTS_DESIGN_DIR = SCRIPT_DIR.parent
DOCS_DIR = AGENTS_DESIGN_DIR.parent / "Docs"
API_FILE = DOCS_DIR / "API.txt"

if not API_FILE.exists():
    raise ValueError(f"API key file not found: {API_FILE}")

OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# PROMPT LOADING
# ======================================================

PROMPT_FILE = SCRIPT_DIR / "prompt.txt"
if not PROMPT_FILE.exists():
    raise ValueError(f"Prompt file not found: {PROMPT_FILE}\\nCreate a prompt.txt file in the Unit_Cell_Agent folder.")

USER_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()

# ======================================================
# STATE
# ======================================================

class AgentState(TypedDict):
    prompt: str
    category: str
    result: dict  # LLM response with job_name + unit_cell data
    output_path: str
    token_usage: dict

# ======================================================
# JSON SCHEMA
# ======================================================

UNIT_CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "job_name": {
            "type": "string",
            "description": "Short descriptive name for this job (e.g., 'nailhead_array')"
        },
        "unit_cell": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Reference location for the unit cell (e.g., 'origin')"
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["cylinder", "box", "sphere", "cone", "pyramid"],
                                "description": "Geometric primitive type"
                            },
                            "center": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                                "description": "[x, y, z] center coordinates in micrometers"
                            },
                            "dimensions": {
                                "type": "object",
                                "description": "Dimensions specific to the primitive type (use relevant fields for the geometry)",
                                "additionalProperties": True
                            },
                            "construction": {
                                "type": "object",
                                "description": "Optional construction metadata for derived primitives (pyramid, cone, tapered cylinders)",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["stacked_boxes", "stacked_cylinders"],
                                        "description": "Construction method: stacked_boxes for pyramids, stacked_cylinders for cones/tapers"
                                    },
                                    "layers": {
                                        "type": "integer",
                                        "description": "Number of layers to decompose into (typically matches height in micrometers)"
                                    },
                                    "top_width_um": {
                                        "type": "number",
                                        "description": "Top width for tapered boxes (pyramids). Use 0 for pointed pyramids."
                                    },
                                    "top_diameter_um": {
                                        "type": "number",
                                        "description": "Top diameter for tapered cylinders (cones). Use 0 for pointed cones."
                                    }
                                },
                                "required": ["method", "layers"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["type", "center", "dimensions"],
                        "additionalProperties": False
                    },
                    "description": "Array of geometric primitives that compose the unit cell"
                }
            },
            "required": ["location", "components"],
            "additionalProperties": False
        },
        "global_info": {
            "type": "object",
            "properties": {
                "pattern_type": {
                    "type": "string",
                    "description": "Type of pattern (e.g., '2D array', '3D lattice')"
                },
                "repetitions": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "z": {"type": "integer"}
                    },
                    "required": ["x", "y", "z"],
                    "additionalProperties": False,
                    "description": "How many times the unit cell repeats in each dimension"
                },
                "spacing": {
                    "type": "object",
                    "properties": {
                        "x_um": {"type": "number"},
                        "y_um": {"type": "number"},
                        "z_um": {"type": "number"},
                        "pattern_description": {"type": "string"}
                    },
                    "required": ["x_um", "y_um", "z_um", "pattern_description"],
                    "additionalProperties": False,
                    "description": "Spacing between unit cells"
                },
                "total_dimensions": {
                    "type": "string",
                    "description": "Overall size of the complete pattern"
                }
            },
            "required": ["pattern_type", "repetitions", "spacing", "total_dimensions"],
            "additionalProperties": False
        }
    },
    "required": ["job_name", "unit_cell", "global_info"],
    "additionalProperties": False
}

# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = """You are a geometry decomposition expert specializing in breaking down structures into machine-actionable geometric primitives.

Given a description of a geometric pattern, you must:

1. Identify the SMALLEST repeating unit cell
2. Decompose it into explicit geometric primitives (cylinder, box, sphere, cone, pyramid)
3. For each component, specify:
   - type: the geometric primitive
   - center: [x, y, z] coordinates in micrometers (use origin [0,0,0] as base reference)
   - dimensions: specific dimensions based on type (e.g., height_um, diameter_um for cylinders)
   - construction (REQUIRED for derived primitives): fabrication metadata
4. Determine the global pattern information (how it repeats, spacing, total dimensions)

IMPORTANT:
- Build components from bottom to top (z-axis)
- Use center coordinates, not corner/base positions
- For stacked components, calculate z-center correctly (e.g., if base cylinder is 6um tall centered at z=3, top cylinder starting at z=6 should be centered at z=6 + height/2)
- Be precise with dimensions - use only dimensions relevant to the primitive type
- For cylinders: use height_um and diameter_um
- For boxes: use width_um, depth_um, height_um
- For pyramids/cones: use base_width_um or base_diameter_um and height_um

FABRICATION-AWARE RULES (CRITICAL):
- PYRAMIDS: Must include 'construction' field with:
  * method: "stacked_boxes"
  * layers: height_um (one layer per micrometer)
  * top_width_um: 0 for pointed pyramids, >0 for truncated pyramids
  
- CONES: Must include 'construction' field with:
  * method: "stacked_cylinders"
  * layers: height_um (one layer per micrometer)
  * top_diameter_um: 0 for pointed cones, >0 for truncated cones
  
- TAPERED CYLINDERS: If a cylinder has varying diameter, include 'construction' with:
  * method: "stacked_cylinders"
  * layers: height_um
  * top_diameter_um: diameter at top
  
- BOX and CYLINDER (constant dimensions): No construction field needed - these are native primitives

Output valid JSON matching the provided schema."""

# ======================================================
# NODES
# ======================================================

def identify_unit_cell(state: AgentState) -> AgentState:
    """Single node that calls LLM to identify unit cell"""
    print(f"[ANALYZING] Prompt: {state['prompt'][:60]}...")
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["prompt"]}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "unit_cell_response",
                "strict": False,
                "schema": UNIT_CELL_SCHEMA
            }
        }
    )
    
    result = json.loads(response.choices[0].message.content)
    
    state["result"] = result
    state["token_usage"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    print(f"[SUCCESS] Unit cell identified: {result['job_name']}")
    print(f"  Tokens: {state['token_usage']['total_tokens']}")
    
    return state

def save_output(state: AgentState) -> AgentState:
    """Save JSON output to file"""
    job_name = state["result"]["job_name"]
    category = state.get("category", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    base_dir = Path(__file__).parent / "Outputs"
    if category:
        output_dir = base_dir / f"{category}_{job_name}_{timestamp}"
    else:
        output_dir = base_dir / f"{job_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save unit cell JSON
    output_file = output_dir / "unit_cell.json"
    
    # Add metadata if category exists
    if category:
        output_data = {
            "metadata": {
                "category": category,
                "timestamp": timestamp,
                "model": "gpt-5-mini",
                "tokens": state["token_usage"]
            },
            "prompt": state["prompt"],
            **state["result"]
        }
    else:
        output_data = state["result"]
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    state["output_path"] = str(output_file)
    
    print(f"[SAVED] Output: {output_file}")
    
    return state

# ======================================================
# GRAPH
# ======================================================

def build_graph():
    """Construct minimal LangGraph"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("identify", identify_unit_cell)
    workflow.add_node("save", save_output)
    
    workflow.set_entry_point("identify")
    workflow.add_edge("identify", "save")
    workflow.add_edge("save", END)
    
    return workflow.compile()

# ======================================================
# PROMPT PARSING FOR BATCH MODE
# ======================================================

def parse_prompts(prompt_file: Path) -> List[Dict[str, str]]:
    """Parse prompt.txt file into individual prompts"""
    content = prompt_file.read_text(encoding="utf-8")
    
    # Split by lines starting with category names (Base-, Undergrad-, Grad-, Postdoc-)
    prompts = []
    
    # Pattern to match category labels
    pattern = r'^(Base|Undergrad|Grad|Postdoc)-\s*(.+?)(?=^(?:Base|Undergrad|Grad|Postdoc)-|\Z)'
    
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        category = match.group(1)
        prompt_text = match.group(2).strip()
        
        # Clean up the prompt text
        prompt_text = re.sub(r'\n\s*\n', '\n', prompt_text)  # Remove extra blank lines
        prompt_text = prompt_text.strip()
        
        if prompt_text and prompt_text != '.':
            prompts.append({
                "category": category,
                "prompt": prompt_text
            })
    
    return prompts

def run_evaluation(prompt_data: Dict[str, str], graph) -> Dict:
    """Run single evaluation"""
    category = prompt_data["category"]
    prompt = prompt_data["prompt"]
    
    print(f"\n[EVAL: {category}]")
    print(f"Prompt: {prompt[:80]}...")
    
    initial_state = {
        "prompt": prompt,
        "category": category,
        "result": {},
        "output_path": "",
        "token_usage": {}
    }
    
    final_state = graph.invoke(initial_state)
    
    result = {
        "category": category,
        "job_name": final_state["result"]["job_name"],
        "output_path": final_state["output_path"],
        "tokens": final_state["token_usage"]["total_tokens"]
    }
    
    print(f"  -> Job: {result['job_name']}")
    print(f"  -> Tokens: {result['tokens']}")
    print(f"  -> Saved: {Path(result['output_path']).parent.name}")
    
    return result

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    import sys
    
    # Check if batch mode is requested
    batch_mode = "--batch" in sys.argv
    
    if batch_mode:
        print("=" * 70)
        print("UNIT CELL BASELINE EVALUATION (BATCH MODE)")
        print("=" * 70)
        
        # Parse prompts
        prompts = parse_prompts(PROMPT_FILE)
        
        print(f"\nFound {len(prompts)} evaluation prompts:")
        for p in prompts:
            print(f"  - {p['category']}")
        
        # Build graph
        graph = build_graph()
        
        # Run all evaluations
        results = []
        for prompt_data in prompts:
            try:
                result = run_evaluation(prompt_data, graph)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        total_tokens = sum(r["tokens"] for r in results)
        
        for r in results:
            print(f"{r['category']:12s} | {r['job_name']:30s} | {r['tokens']:4d} tokens")
        
        print("-" * 70)
        print(f"{'TOTAL':12s} | {len(results)} prompts processed | {total_tokens:4d} tokens")
        print("=" * 70)
        
        # Save summary
        summary_file = SCRIPT_DIR / "Outputs" / f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.parent.mkdir(exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSummary saved: {summary_file.name}")
    else:
        # Single prompt mode
        print("=" * 60)
        print("UNIT CELL GEOMETRY AGENT")
        print("=" * 60)
        
        # Build graph
        graph = build_graph()
        
        # Initial state
        initial_state = {
            "prompt": USER_PROMPT.strip(),
            "category": "",
            "result": {},
            "output_path": "",
            "token_usage": {}
        }
        
        # Run
        final_state = graph.invoke(initial_state)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Job Name: {final_state['result']['job_name']}")
        print(f"Output: {final_state['output_path']}")
        print(f"Tokens Used: {final_state['token_usage']['total_tokens']}")
        print("=" * 60)
