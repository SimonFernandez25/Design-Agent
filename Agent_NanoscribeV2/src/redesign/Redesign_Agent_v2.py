"""
Redesign_Agent_v2.py - V2 Stateful Redesign Agent

Operates on v2 Named Object architecture (objects + assembly).
Each redesign step produces a new immutable variant.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent

# Add paths for imports
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# Import v2 modules
# Import v2 modules
from edit_schema_v2 import EditPlan, format_edit_plan_for_display, validate_edit_plan_v2
from edit_suggestion_agent_v2 import analyze_redesign_prompt_v2
from edit_executor_v2 import apply_edit_plan_v2
from schemas.validation import v2_structural_gate

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
            return api_file
        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent
    raise ValueError(f"API key file not found within {max_levels} parent directories")

API_FILE = find_api_key_file(SCRIPT_DIR)
OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

# ======================================================
# V2 DESIGN VARIANT
# ======================================================

class DesignVariantV2:
    """
    V2 Design Variant - immutable step in redesign lineage.
    
    Uses 'design_json' with objects+assembly, NOT 'unit_cell_json'.
    """
    
    def __init__(
        self,
        variant_id: int,
        parent_id: int,
        design_json: Dict[str, Any],
        applied_edit_plan: List[Dict[str, Any]],
        prompt_history: List[str],
        theta: float
    ):
        self.variant_id = variant_id
        self.parent_id = parent_id
        self.design_json = design_json  # V2 format
        self.applied_edit_plan = applied_edit_plan
        self.prompt_history = prompt_history
        self.theta = theta
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "parent_id": self.parent_id,
            "design_json": self.design_json,
            "applied_edit_plan": self.applied_edit_plan,
            "prompt_history": self.prompt_history,
            "theta": self.theta
        }


def compute_theta(prompt_history: List[str], latest_prompt: str, epsilon: float = 0.1) -> float:
    """Compute simple recency score."""
    return len(prompt_history) + 1 + epsilon


def create_initial_variant_v2(design_json: Dict[str, Any], initial_prompt: str = "Initial design") -> DesignVariantV2:
    """Create initial v2 variant from design.json."""
    return DesignVariantV2(
        variant_id=0,
        parent_id=None,
        design_json=design_json,
        applied_edit_plan=[],
        prompt_history=[initial_prompt],
        theta=compute_theta([], initial_prompt)
    )


def create_child_variant_v2(
    parent: DesignVariantV2,
    new_design_json: Dict[str, Any],
    edit_plan: List[Dict[str, Any]],
    current_prompt: str
) -> DesignVariantV2:
    """Create child variant from parent."""
    new_prompt_history = parent.prompt_history + [current_prompt]
    return DesignVariantV2(
        variant_id=parent.variant_id + 1,
        parent_id=parent.variant_id,
        design_json=new_design_json,
        applied_edit_plan=edit_plan,
        prompt_history=new_prompt_history,
        theta=compute_theta(parent.prompt_history, current_prompt)
    )


# ======================================================
# V2 REDESIGN SESSION
# ======================================================

class RedesignSessionV2:
    """
    V2 Stateful Redesign Session.
    
    Works with design.json (objects + assembly), NOT unit_cell.json.
    """
    
    def __init__(self, project_name: str, base_output_dir: Path):
        self.project_name = project_name
        self.output_dir = base_output_dir / project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.design_variants: List[DesignVariantV2] = []
        self.print_params_file: Path = None
        self.root_dir = ROOT_DIR
    
    def initialize_from_design(
        self,
        design_path: Path,
        print_params_path: Path,
        initial_prompt: str = "Initial design"
    ) -> DesignVariantV2:
        """
        Initialize session from a v2 design.json.
        
        This validates the design passes v2 structural gate.
        """
        with open(design_path, 'r') as f:
            design_json = json.load(f)
        
        # V2 structural gate - must not have legacy fields
        v2_structural_gate(design_json)
        
        self.print_params_file = print_params_path
        
        variant_0 = create_initial_variant_v2(design_json, initial_prompt)
        self.design_variants.append(variant_0)
        
        self._save_variant_to_disk(variant_0)
        
        print(f"[SESSION V2] Initialized with variant_0: {design_json.get('job_name', 'unknown')}")
        print(f"[SESSION V2] Objects: {list(design_json.get('objects', {}).keys())}")
        print(f"[SESSION V2] Output directory: {self.output_dir}")
        
        return variant_0
    
    def get_current_variant(self) -> DesignVariantV2:
        """Get the most recent variant."""
        if not self.design_variants:
            raise ValueError("Session not initialized. Call initialize_from_design first.")
        return self.design_variants[-1]
    
    def apply_redesign(
        self,
        redesign_prompt: str,
        render_images: List[str] = None
    ) -> DesignVariantV2:
        """
        Apply a redesign using natural language prompt.
        
        This uses the v2 edit suggestion agent to analyze the prompt
        and generate edit operations.
        """
        parent = self.get_current_variant()
        
        print("=" * 70)
        print(f"V2 REDESIGN: variant_{parent.variant_id} -> variant_{parent.variant_id + 1}")
        print("=" * 70)
        print(f"Prompt: {redesign_prompt}")
        print(f"Parent theta: {parent.theta:.2f}")
        
        # Step 1: Generate v2 edit suggestions
        print("\n[1/4] Generating v2 edit suggestions...")
        edit_plan = analyze_redesign_prompt_v2(
            parent.design_json,
            redesign_prompt,
            render_images
        )
        
        print(f"  Generated {len(edit_plan['edit_plan'])} edits")
        print(f"  Summary: {edit_plan['summary']}")
        
        # Step 2: Apply v2 edits
        print("\n[2/4] Applying v2 edits...")
        modified_json = apply_edit_plan_v2(parent.design_json, edit_plan)
        
        # Verify output passes v2 gate
        v2_structural_gate(modified_json)
        print("  [OK] Modified design passes v2 structural gate")
        
        # Step 3: Create new variant
        print("\n[3/4] Creating new variant...")
        new_variant = create_child_variant_v2(
            parent,
            modified_json,
            edit_plan["edit_plan"],
            redesign_prompt
        )
        
        self.design_variants.append(new_variant)
        
        # Step 4: Save to disk
        print("\n[4/4] Saving to disk...")
        self._save_variant_to_disk(new_variant, edit_plan)
        
        # Generate outputs
        self._generate_outputs(new_variant)
        
        print("\n" + "=" * 70)
        print("V2 REDESIGN COMPLETE")
        print("=" * 70)
        print(f"New variant: variant_{new_variant.variant_id}")
        print(f"New theta: {new_variant.theta:.2f}")
        print(f"Output: {self._get_variant_dir(new_variant)}")
        print("=" * 70)
        
        return new_variant
    
    def _get_variant_dir(self, variant: DesignVariantV2) -> Path:
        return self.output_dir / f"variant_{variant.variant_id}"
    
    def _save_variant_to_disk(self, variant: DesignVariantV2, edit_plan: Dict = None) -> None:
        """Save variant to disk in v2 format."""
        variant_dir = self._get_variant_dir(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        # Save design.json (NOT unit_cell.json)
        design_path = variant_dir / "design.json"
        with open(design_path, 'w') as f:
            json.dump(variant.design_json, f, indent=2)
        print(f"  Saved: {design_path.name}")
        
        # Save edit plan
        if edit_plan:
            edit_plan_path = variant_dir / "edit_plan.json"
            with open(edit_plan_path, 'w') as f:
                json.dump(edit_plan, f, indent=2)
            print(f"  Saved: {edit_plan_path.name}")
        
        # Save prompt
        prompt_path = variant_dir / "prompt.txt"
        current_prompt = variant.prompt_history[-1] if variant.prompt_history else ""
        with open(prompt_path, 'w') as f:
            f.write(current_prompt)
        print(f"  Saved: {prompt_path.name}")
        
        # Save metadata
        metadata = {
            "variant_id": variant.variant_id,
            "parent_id": variant.parent_id,
            "theta": variant.theta,
            "prompt_history": variant.prompt_history,
            "timestamp": datetime.now().isoformat(),
            "format_version": "v2"
        }
        metadata_path = variant_dir / "variant_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {metadata_path.name}")
        
        (variant_dir / "Renders").mkdir(exist_ok=True)
        (variant_dir / "GWL").mkdir(exist_ok=True)
    
    def _generate_outputs(self, variant: DesignVariantV2) -> None:
        """Generate reduced.json, endpoints, renders, and GWL."""
        variant_dir = self._get_variant_dir(variant)
        design_file = variant_dir / "design.json"
        reduced_file = variant_dir / "reduced.json"
        endpoints_file = variant_dir / "endpoints.json"
        renders_dir = variant_dir / "Renders"
        gwl_dir = variant_dir / "GWL"
        
        # Step 1: Run reduction engine
        print("  Running reduction engine...")
        try:
            from reduction_engine import reduce_assembly
            reduced = reduce_assembly(variant.design_json)
            with open(reduced_file, 'w') as f:
                json.dump(reduced, f, indent=2)
            print(f"    [OK] Reduced to {len(reduced['primitives'])} primitives")
        except Exception as e:
            print(f"    [ERROR] Reduction failed: {e}")
            return
        
        # Step 2: Generate endpoints from reduced.json
        print("  Generating endpoints...")
        try:
            from endpoint_generator_v2 import generate_endpoint_json_v2, load_print_parameters
            params = load_print_parameters(self.print_params_file)
            endpoints = generate_endpoint_json_v2(reduced, params)
            with open(endpoints_file, 'w') as f:
                json.dump(endpoints, f, indent=2)
            print(f"    [OK] Endpoints saved ({len(endpoints.get('layers', []))} layers)")
        except Exception as e:
            print(f"    [ERROR] Endpoint generation failed: {e}")
            return
        
        # Step 3: Generate renders using v2 object-aware renderer
        print("  Generating renders...")
        try:
            from render_generator_v2 import generate_object_aware_renders
            render_results = generate_object_aware_renders(variant.design_json, renders_dir, self.print_params_file)
            total_renders = sum(len(paths) for paths in render_results.values())
            print(f"    [OK] Generated {total_renders} renders across {len(render_results)} objects")
        except Exception as e:
            import traceback
            print(f"    [ERROR] Render generation failed!")
            print(f"    Exception: {e}")
            print(f"    Traceback:")
            traceback.print_exc()
        
        # Step 4: Generate GWL
        print("  Generating GWL files...")
        try:
            from gwl_serializer import generate_gwl_files, generate_master_gwl, load_gwl_parameters
            gwl_params = load_gwl_parameters(self.print_params_file)
            gwl_files = generate_gwl_files(endpoints, gwl_params, gwl_dir)
            
            master_gwl = gwl_dir / f"{variant.design_json.get('job_name', 'design')}_master.gwl"
            generate_master_gwl(gwl_files, gwl_params, master_gwl)
            print(f"    [OK] Generated {len(gwl_files)} GWL files + master")
        except Exception as e:
            print(f"    [WARN] GWL generation failed: {e}")
    
    def get_variant_renders(self, variant: DesignVariantV2) -> List[Path]:
        """Get render paths for a variant."""
        variant_dir = self._get_variant_dir(variant)
        renders_dir = variant_dir / "Renders"
        return sorted(renders_dir.glob('*.png'))
    
    def print_history(self) -> None:
        """Print variant history."""
        print("\n" + "=" * 70)
        print("V2 DESIGN VARIANT HISTORY")
        print("=" * 70)
        for v in self.design_variants:
            parent_str = f"parent={v.parent_id}" if v.parent_id is not None else "root"
            print(f"  variant_{v.variant_id}: theta={v.theta:.2f}, {parent_str}")
            print(f"    Prompt: {v.prompt_history[-1][:60]}...")
        print("=" * 70)


# ======================================================
# CONVENIENCE FUNCTION
# ======================================================

def run_redesign_v2(
    project_name: str,
    design_path: str,
    print_params_path: str,
    redesign_prompt: str,
    output_dir: str = None,
    render_images: List[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a single v2 redesign step.
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR / "redesigns_v2"
    else:
        output_dir = Path(output_dir)
    
    session = RedesignSessionV2(project_name, output_dir)
    session.initialize_from_design(
        Path(design_path),
        Path(print_params_path)
    )
    
    new_variant = session.apply_redesign(redesign_prompt, render_images)
    
    return {
        "variant_id": new_variant.variant_id,
        "theta": new_variant.theta,
        "output_dir": str(session._get_variant_dir(new_variant)),
        "prompt_history": new_variant.prompt_history
    }


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python Redesign_Agent_v2.py <project_name> <design.json> <PrintParameters.txt> <prompt>")
        print("\nExample:")
        print('  python Redesign_Agent_v2.py Grad ../Outputs/Grad/design.json ../PrintParameters.txt "make posts taller"')
        sys.exit(1)
    
    project_name = sys.argv[1]
    design_path = sys.argv[2]
    print_params = sys.argv[3]
    prompt = sys.argv[4]
    
    run_redesign_v2(project_name, design_path, print_params, prompt)
