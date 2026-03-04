"""
Redesign Agent - Stateful Redesign Orchestrator

This module manages the complete redesign workflow, coordinating:
    1. Edit suggestion generation (LLM-powered)
    2. Edit execution (deterministic)
    3. Variant tracking (immutable history)
    4. Output generation (endpoints, GWL, renders)

Usage:
    from redesign_agent import RedesignSession
    
    session = RedesignSession("MyProject", output_dir)
    session.initialize_from_unit_cell(unit_cell_path, print_params_path)
    new_variant = session.apply_redesign("make posts taller")

Author: Nanoscribe Design Agent Team
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI

# Import from redesign package
from design_variant import (
    DesignVariant,
    compute_theta,
    create_initial_variant,
    create_child_variant
)
from edit_schema import EditPlan, format_edit_plan_for_display, validate_edit_plan
from edit_suggestion_agent import analyze_redesign_prompt
from edit_executor import apply_edit_plan


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()


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


def load_print_parameters(param_file: Path) -> Dict[str, float]:
    """Load print parameters from file."""
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and ':' in line:
                key, value = line.split(':', 1)
                try:
                    params[key.strip()] = float(value.strip())
                except ValueError:
                    pass
    return params


# ==============================================================================
# REDESIGN SESSION
# ==============================================================================

class RedesignSession:
    """
    Manages a stateful redesign session with variant history.
    
    This class maintains the in-memory design_variants list and ensures
    the filesystem always mirrors this state.
    
    Attributes:
        project_name: Name of the project (e.g., "Grad", "Base")
        output_dir: Directory where variants are saved
        design_variants: List of all variants in this session
        print_params_file: Path to PrintParameters.txt
    """
    
    def __init__(self, project_name: str, base_output_dir: Path):
        """
        Initialize a redesign session.
        
        Args:
            project_name: Name of the project
            base_output_dir: Base directory for output
        """
        self.project_name = project_name
        self.output_dir = base_output_dir / project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.design_variants: List[DesignVariant] = []
        self.print_params_file: Path = None
        
        # Find root directory for pipeline tools
        self.root_dir = SCRIPT_DIR.parent
    
    def initialize_from_unit_cell(
        self,
        unit_cell_path: Path,
        print_params_path: Path,
        initial_prompt: str = "Initial design"
    ) -> DesignVariant:
        """
        Initialize the session with an original unit cell (variant_0).
        
        Args:
            unit_cell_path: Path to the original unit_cell.json
            print_params_path: Path to PrintParameters.txt
            initial_prompt: Description for the initial state
        
        Returns:
            The initial DesignVariant (variant_0)
        """
        with open(unit_cell_path, 'r') as f:
            original_json = json.load(f)
        
        self.print_params_file = print_params_path
        
        # Create initial variant
        variant_0 = create_initial_variant(original_json, initial_prompt)
        self.design_variants.append(variant_0)
        
        # Save to disk immediately (filesystem mirrors memory)
        self._save_variant_to_disk(variant_0)
        
        print(f"[SESSION] Initialized with variant_0: {original_json.get('job_name', 'unknown')}")
        print(f"[SESSION] Output directory: {self.output_dir}")
        
        return variant_0
    
    def get_current_variant(self) -> DesignVariant:
        """Get the most recent variant (parent for next redesign)."""
        if not self.design_variants:
            raise ValueError("Session not initialized. Call initialize_from_unit_cell first.")
        return self.design_variants[-1]
    
    def apply_redesign(
        self,
        redesign_prompt: str,
        render_images: List[str] = None
    ) -> DesignVariant:
        """
        Apply a redesign to the current variant, creating a new child variant.
        
        This is the main entry point for redesign operations.
        
        Args:
            redesign_prompt: User's redesign request
            render_images: Optional list of render image paths for vision
        
        Returns:
            The newly created DesignVariant
        """
        parent = self.get_current_variant()
        
        print("=" * 70)
        print(f"REDESIGN: variant_{parent['variant_id']} -> variant_{parent['variant_id'] + 1}")
        print("=" * 70)
        print(f"Prompt: {redesign_prompt}")
        print(f"Parent theta: {parent['theta']:.2f}")
        
        # Step 1: Generate edit suggestions
        print("\n[1/4] Generating edit suggestions...")
        edit_plan = analyze_redesign_prompt(
            parent["unit_cell_json"],
            redesign_prompt,
            render_images
        )
        
        print(f"  Generated {len(edit_plan['edit_plan'])} edits")
        print(f"  Summary: {edit_plan['summary']}")
        
        # Step 2: Apply edits
        print("\n[2/4] Applying edits...")
        modified_json = apply_edit_plan(parent["unit_cell_json"], edit_plan)
        
        # Step 3: Create new variant
        print("\n[3/4] Creating new variant...")
        new_variant = create_child_variant(
            parent,
            modified_json,
            edit_plan["edit_plan"],
            redesign_prompt
        )
        
        # Append to in-memory list
        self.design_variants.append(new_variant)
        
        # Step 4: Save to disk (filesystem mirrors memory)
        print("\n[4/4] Saving to disk...")
        self._save_variant_to_disk(new_variant, edit_plan)
        
        # Generate outputs (endpoints, renders, GWL)
        self._generate_outputs(new_variant)
        
        print("\n" + "=" * 70)
        print("REDESIGN COMPLETE")
        print("=" * 70)
        print(f"New variant: variant_{new_variant['variant_id']}")
        print(f"New theta: {new_variant['theta']:.2f}")
        print(f"Prompt history length: {len(new_variant['prompt_history'])}")
        print(f"Output: {self._get_variant_dir(new_variant)}")
        print("=" * 70)
        
        return new_variant
    
    def _get_variant_dir(self, variant: DesignVariant) -> Path:
        """Get the directory path for a variant."""
        return self.output_dir / f"variant_{variant['variant_id']}"
    
    def _save_variant_to_disk(self, variant: DesignVariant, edit_plan: Dict = None) -> None:
        """
        Save a variant to disk, ensuring filesystem mirrors memory.
        
        Creates structure:
            variant_N/
                unit_cell.json
                edit_plan.json (if provided)
                prompt.txt
                variant_metadata.json
                Renders/
                GWL/
        """
        variant_dir = self._get_variant_dir(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        # Save unit cell JSON
        unit_cell_path = variant_dir / "unit_cell.json"
        with open(unit_cell_path, 'w') as f:
            json.dump(variant["unit_cell_json"], f, indent=2)
        print(f"  Saved: {unit_cell_path.name}")
        
        # Save edit plan (if provided)
        if edit_plan:
            edit_plan_path = variant_dir / "edit_plan.json"
            with open(edit_plan_path, 'w') as f:
                json.dump(edit_plan, f, indent=2)
            print(f"  Saved: {edit_plan_path.name}")
        
        # Save prompt
        prompt_path = variant_dir / "prompt.txt"
        current_prompt = variant["prompt_history"][-1] if variant["prompt_history"] else ""
        with open(prompt_path, 'w') as f:
            f.write(current_prompt)
        print(f"  Saved: {prompt_path.name}")
        
        # Save variant metadata
        metadata = {
            "variant_id": variant["variant_id"],
            "parent_id": variant["parent_id"],
            "theta": variant["theta"],
            "prompt_history": variant["prompt_history"],
            "timestamp": datetime.now().isoformat()
        }
        metadata_path = variant_dir / "variant_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {metadata_path.name}")
        
        # Create empty directories for outputs
        (variant_dir / "Renders").mkdir(exist_ok=True)
        (variant_dir / "GWL").mkdir(exist_ok=True)
    
    def _generate_outputs(self, variant: DesignVariant) -> None:
        """Generate endpoints, renders, and GWL for a variant."""
        variant_dir = self._get_variant_dir(variant)
        unit_cell_file = variant_dir / "unit_cell.json"
        endpoints_file = variant_dir / "endpoints.json"
        renders_dir = variant_dir / "Renders"
        gwl_dir = variant_dir / "GWL"
        
        # Step 1: Generate endpoints
        print("  Generating endpoints...")
        cmd_endpoints = [
            'python',
            str(self.root_dir / 'endpoint_generator.py'),
            str(unit_cell_file),
            str(self.print_params_file),
            str(endpoints_file)
        ]
        
        result = subprocess.run(cmd_endpoints, cwd=self.root_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    [OK] Endpoints saved: endpoints.json")
        else:
            print(f"    [ERROR] Endpoint generation failed: {result.stderr}")
            return
        
        # Step 2: Generate renders
        print("  Generating renders...")
        cmd_renders = [
            'python',
            str(self.root_dir / 'render_generator.py'),
            str(endpoints_file),
            str(self.print_params_file),
            str(renders_dir)
        ]
        
        result = subprocess.run(cmd_renders, cwd=self.root_dir, capture_output=True, text=True)
        if result.returncode == 0:
            num_renders = len(list(renders_dir.glob('*.png')))
            print(f"    [OK] Generated {num_renders} renders")
        else:
            print(f"    [ERROR] Render generation failed: {result.stderr}")
        
        # Step 3: Generate GWL files
        print("  Generating GWL files...")
        cmd_gwl = [
            'python',
            str(self.root_dir / 'gwl_serializer.py'),
            str(endpoints_file),
            str(self.print_params_file),
            str(gwl_dir)
        ]
        
        result = subprocess.run(cmd_gwl, cwd=self.root_dir, capture_output=True, text=True)
        if result.returncode == 0:
            num_gwl_files = len(list(gwl_dir.glob('*.gwl')))
            print(f"    [OK] Generated {num_gwl_files} GWL files")
        else:
            print(f"    [ERROR] GWL generation failed: {result.stderr}")
    
    def get_variant_renders(self, variant: DesignVariant) -> List[Path]:
        """Get list of render image paths for a variant."""
        variant_dir = self._get_variant_dir(variant)
        renders_dir = variant_dir / "Renders"
        return sorted(renders_dir.glob('*.png'))
    
    def print_history(self) -> None:
        """Print the variant history for this session."""
        print("\n" + "=" * 70)
        print("DESIGN VARIANT HISTORY")
        print("=" * 70)
        for v in self.design_variants:
            parent_str = f"parent={v['parent_id']}" if v['parent_id'] is not None else "root"
            print(f"  variant_{v['variant_id']}: theta={v['theta']:.2f}, {parent_str}")
            print(f"    Prompt: {v['prompt_history'][-1][:60]}...")
        print("=" * 70)


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def run_redesign(
    project_name: str,
    original_unit_cell_path: str,
    print_params_path: str,
    redesign_prompt: str,
    output_dir: str = None,
    render_images: List[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a single redesign step.
    
    For notebook use, prefer creating a RedesignSession directly.
    
    Args:
        project_name: Name of the project
        original_unit_cell_path: Path to original unit_cell.json
        print_params_path: Path to PrintParameters.txt
        redesign_prompt: User's redesign request
        output_dir: Base output directory (default: redesigns/)
        render_images: Optional list of render image paths
    
    Returns:
        Dictionary with result info
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR / "redesigns"
    else:
        output_dir = Path(output_dir)
    
    session = RedesignSession(project_name, output_dir)
    session.initialize_from_unit_cell(
        Path(original_unit_cell_path),
        Path(print_params_path)
    )
    
    new_variant = session.apply_redesign(redesign_prompt, render_images)
    
    return {
        "variant_id": new_variant["variant_id"],
        "theta": new_variant["theta"],
        "output_dir": str(session._get_variant_dir(new_variant)),
        "prompt_history": new_variant["prompt_history"]
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python redesign_agent.py <project_name> <unit_cell.json> <PrintParameters.txt> <prompt>")
        print("\nExample:")
        print('  python redesign_agent.py Grad ../outputs/unit_cell.json ../PrintParameters.txt "make posts taller"')
        sys.exit(1)
    
    project_name = sys.argv[1]
    unit_cell = sys.argv[2]
    print_params = sys.argv[3]
    prompt = sys.argv[4]
    
    run_redesign(project_name, unit_cell, print_params, prompt)
