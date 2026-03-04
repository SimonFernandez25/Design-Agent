"""
supervisor.py - Supervisor Orchestrator

Orchestrates the full V2 fabrication pipeline externally.
Core modules (NamedObjectAgent, reduction_engine, manufacturing_agent,
endpoint_generator_v2) are called via their public functions -- never modified.

Design decisions (per reviewer feedback):
- Iterations are immutable. apply_redesign loads the previous iteration's
  design.json from disk; it never uses job.design_json as its mutation base.
- Overrides are per-call only. They are NOT stored on the job object.
  They ARE recorded in iteration_metadata for each run.
- Validation runs after EVERY design modification (redesign, initial generate).
- estimated_print_time is deterministic: total_endpoints / WRITE_SPEED_EPS
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fabrication_job import FabricationJob
from dot_path_editor import apply_dot_path_edits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Path bootstrap helper
# ---------------------------------------------------------------------------

def _bootstrap_src_paths():
    """
    Ensure src/ and src/supervisor/ are on sys.path so imports work
    regardless of how the supervisor is invoked.
    """
    here = Path(__file__).parent          # src/supervisor/
    src  = here.parent                    # src/
    root = src.parent                     # Agent_NanoscribeV2/

    for p in [str(src), str(root)]:
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_src_paths()

# Deferred imports of core modules (they live in src/)
import reduction_engine
import manufacturing_agent
import endpoint_generator_v2
import gwl_serializer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_param_file(start: Path, max_levels: int = 5) -> Optional[Path]:
    """Search upward for PrintParameters.txt."""
    curr = start
    for _ in range(max_levels):
        candidate = curr / "PrintParameters.txt"
        if candidate.exists():
            return candidate
        parent = curr.parent
        if parent == curr:
            break
        curr = parent
    return None


def _validate_design(design_json: Dict) -> None:
    """
    Run V2 structural gate + schema validation.
    Called after every design generation or modification.

    Raises ValueError if any structural violation is found.
    """
    # Import here to avoid circular path issues at module load time
    from schemas import validate_object_library, validate_assembly, v2_structural_gate

    # 1. Structural gate (rejects legacy fields)
    v2_structural_gate(design_json)

    # 2. Schema validation
    errors: List[str] = []
    errors.extend(validate_object_library({"objects": design_json.get("objects", {})}))
    errors.extend(validate_assembly(
        {"assembly": design_json.get("assembly", {})},
        list(design_json.get("objects", {}).keys())
    ))

    if errors:
        bullet_list = "\n  - ".join(errors)
        raise ValueError(
            f"Design failed schema validation after modification:\n  - {bullet_list}"
        )


def _apply_overrides_to_enriched(
    enriched_json: Dict[str, Any],
    overrides: Dict[str, Any],
    feedback_log: List[Dict],
) -> Dict[str, Any]:
    """
    Apply parameter overrides to a deep copy of enriched_json.

    Override value rules:
    - str starting with '+' or '-': delta (added to existing value)
    - numeric (int/float): absolute replacement
    - dict with key 'primitive_id': only applied to that primitive index

    Returns the adjusted copy. The caller's enriched_json is NOT mutated.
    """
    adjusted = copy.deepcopy(enriched_json)
    primitives: List[Dict] = adjusted.get("primitives", [])

    # Separate per-primitive overrides from global ones
    global_overrides = {
        k: v for k, v in overrides.items()
        if not k.startswith("primitive_")
    }

    applied_log: List[Dict] = []

    for idx, prim in enumerate(primitives):
        lp = prim.get("local_parameters")
        if lp is None:
            continue

        for param, raw_value in global_overrides.items():
            # Check if a per-primitive version overrides the global for this index
            per_prim_key = f"primitive_{idx}.{param}"
            if per_prim_key in overrides:
                raw_value = overrides[per_prim_key]

            if param not in lp:
                continue  # param not in local_parameters; skip silently

            existing = lp[param]

            if isinstance(raw_value, str) and raw_value[:1] in ("+", "-"):
                # Delta override
                delta = float(raw_value)
                new_val = round(float(existing) + delta, 6)
                applied_log.append({
                    "primitive_id": idx, "param": param,
                    "mode": "delta", "delta": delta,
                    "old": existing, "new": new_val
                })
            else:
                # Absolute replacement
                new_val = raw_value
                applied_log.append({
                    "primitive_id": idx, "param": param,
                    "mode": "absolute",
                    "old": existing, "new": new_val
                })

            lp[param] = new_val

    feedback_log.append({
        "event": "OVERRIDES_APPLIED",
        "overrides_dict": overrides,
        "primitive_edits": applied_log,
    })

    print(f"[Supervisor] Overrides applied to {len(applied_log)} primitive parameter(s).")
    return adjusted


def _compute_metrics(
    enriched_json: Dict[str, Any],
    output_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute iteration metrics from enriched_json and output_json.

    Metrics
    -------
    total_primitives : int
    total_layers : int        -- number of unique Z layers in output_json
    max_z : float             -- highest Z layer in um
    bounding_box : dict       -- {x_min, x_max, y_min, y_max, z_min, z_max}
    average_risk : float
    max_risk : float
    mean_slenderness : float
    estimated_endpoint_count : int  -- total scan segments across all layers
    estimated_print_time : float    -- seconds = endpoints / WRITE_SPEED_EPS
    """
    primitives: List[Dict] = enriched_json.get("primitives", [])
    layers: List[Dict] = output_json.get("layers", [])

    # Primitive-level aggregations
    risks = [float(p.get("risk", 0.0)) for p in primitives]
    slendernesses = [
        float(p.get("metrics", {}).get("slenderness", 0.0)) for p in primitives
    ]

    # Bounding box from primitive centers + half-dimensions
    xs, ys, zs = [], [], []
    for prim in primitives:
        cx, cy, cz = prim.get("center", [0, 0, 0])
        dims = prim.get("dimensions", {})
        ptype = prim.get("type", "box")
        if ptype == "cylinder":
            r = dims.get("diameter_um", 0) / 2.0
            h = dims.get("height_um", dims.get("z_um", 0)) / 2.0
            xs.extend([cx - r, cx + r])
            ys.extend([cy - r, cy + r])
            zs.extend([cz - h, cz + h])
        else:
            hx = dims.get("x_um", dims.get("width_um", 0)) / 2.0
            hy = dims.get("y_um", dims.get("depth_um", 0)) / 2.0
            hz = dims.get("z_um", dims.get("height_um", 0)) / 2.0
            xs.extend([cx - hx, cx + hx])
            ys.extend([cy - hy, cy + hy])
            zs.extend([cz - hz, cz + hz])

    bounding_box = {
        "x_min": round(min(xs, default=0.0), 4),
        "x_max": round(max(xs, default=0.0), 4),
        "y_min": round(min(ys, default=0.0), 4),
        "y_max": round(max(ys, default=0.0), 4),
        "z_min": round(min(zs, default=0.0), 4),
        "z_max": round(max(zs, default=0.0), 4),
    }

    # Endpoint-level aggregations
    total_segments = sum(len(layer.get("segments", [])) for layer in layers)
    max_z = max((layer.get("z_um", 0.0) for layer in layers), default=0.0)

    return {
        "total_primitives": len(primitives),
        "total_layers": len(layers),
        "max_z": round(float(max_z), 4),
        "bounding_box": bounding_box,
        "average_risk": round(sum(risks) / len(risks), 6) if risks else 0.0,
        "max_risk": round(max(risks, default=0.0), 6),
        "mean_slenderness": round(
            sum(slendernesses) / len(slendernesses), 6
        ) if slendernesses else 0.0,
        "estimated_endpoint_count": total_segments,
    }


def _save_artifacts(
    iter_dir: Path,
    job: FabricationJob,
    gwl_params: Dict,
) -> Dict[str, str]:
    """
    Write all iteration artifacts to *iter_dir*.

    Returns a dict mapping artifact name -> relative path string.
    """
    iter_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}

    def _write(fname: str, data: Any) -> Path:
        p = iter_dir / fname
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        artifacts[fname] = str(p)
        print(f"[Supervisor] Saved: {fname}")
        return p

    _write("design.json", job.design_json)
    _write("reduced.json", job.reduced_json)
    _write("enriched_reduced.json", job.enriched_json)
    _write("output.json", job.output_json)
    _write("feedback_log.json", job.feedback_log)

    # GWL files
    gwl_dir = iter_dir / "gwl"
    gwl_files = gwl_serializer.generate_gwl_files(job.output_json, gwl_params, gwl_dir)
    job.gwl_path = gwl_dir

    if gwl_files:
        job_name = (job.design_json or {}).get("job_name", job.job_id)
        master_path = gwl_dir / f"{job_name}_master.gwl"
        gwl_serializer.generate_master_gwl(gwl_files, gwl_params, master_path)

    artifacts["gwl/"] = str(gwl_dir)
    return artifacts


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    """
    Orchestrates the V2 fabrication pipeline for a FabricationJob.

    The Supervisor is STATELESS with respect to job data. All state lives
    in the FabricationJob object. The Supervisor only coordinates calls.

    Parameters
    ----------
    outputs_root : Path
        Root directory for all job outputs.
        Defaults to Agent_NanoscribeV2/Outputs/.
    param_file : Path | None
        Path to PrintParameters.txt.
        If None, the Supervisor searches upward from its own location.
    """

    def __init__(
        self,
        outputs_root: Optional[Path] = None,
        param_file: Optional[Path] = None,
    ) -> None:
        here = Path(__file__).parent          # src/supervisor/
        root = here.parent.parent             # Agent_NanoscribeV2/

        self.outputs_root: Path = outputs_root or (root / "Outputs")
        self.outputs_root.mkdir(parents=True, exist_ok=True)

        self.param_file: Path = param_file or _find_param_file(root)
        if self.param_file is None or not self.param_file.exists():
            raise FileNotFoundError(
                "PrintParameters.txt not found. "
                "Pass param_file explicitly or place it in Agent_NanoscribeV2/."
            )

        print(f"[Supervisor] outputs_root: {self.outputs_root}")
        print(f"[Supervisor] param_file:   {self.param_file}")

    # ------------------------------------------------------------------
    # Internal pipeline runner
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        job: FabricationJob,
        overrides: Optional[Dict[str, Any]] = None,
        redesign_edits: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute full pipeline from validated design_json onward.

        Steps:
          1. Validate design
          2. Reduce to primitives
          3. Enrich via manufacturing_agent
          4. Apply overrides (if any) to adjusted copy
          5. Generate endpoint JSON
          6. Save all artifacts
          7. Compute + store metrics
          8. Increment iteration
        """
        iter_dir = job.get_iter_dir(self.outputs_root)
        if iter_dir.exists():
            raise RuntimeError(
                f"Iteration directory already exists: {iter_dir}. "
                f"This should never happen; each iteration gets a fresh directory."
            )

        print(f"\n[Supervisor] === Iteration {job.iteration:03d} | {iter_dir.name} ===")

        # Open structured metadata record for this iteration
        record = job.open_iteration_record(
            redesign_edits=redesign_edits,
            overrides=overrides,
        )

        # 1. Validate
        print("[Supervisor] Validating design...")
        _validate_design(job.design_json)
        print("[Supervisor] Validation OK.")

        # 2. Reduce
        print("[Supervisor] Reducing assembly to flat primitives...")
        job.reduced_json = reduction_engine.reduce_assembly(job.design_json)
        print(f"[Supervisor] Reduced: {job.reduced_json['metadata']['num_primitives']} primitives")

        # 3. Enrich
        print("[Supervisor] Running manufacturing agent...")
        job.enriched_json = manufacturing_agent.enrich_primitives(
            job.reduced_json,
            output_dir=iter_dir,   # MA will write enriched_reduced.json + logs here
        )

        # 4. Apply overrides (post-MA, pre-endpoint)
        enriched_for_endpoint = job.enriched_json
        if overrides:
            print("[Supervisor] Applying parameter overrides...")
            enriched_for_endpoint = _apply_overrides_to_enriched(
                job.enriched_json, overrides, job.feedback_log
            )
        else:
            job.log("OVERRIDES_SKIPPED", {"reason": "No overrides provided for this iteration."})

        # 5. Generate endpoints
        print("[Supervisor] Generating endpoint JSON...")
        print_params = endpoint_generator_v2.load_print_parameters(self.param_file)
        job.output_json = endpoint_generator_v2.generate_endpoint_json_v2(
            enriched_for_endpoint, print_params
        )

        # 6. Save artifacts
        print("[Supervisor] Saving artifacts...")
        gwl_params = gwl_serializer.load_gwl_parameters(self.param_file)
        artifacts = _save_artifacts(iter_dir, job, gwl_params)

        # 7. Metrics
        print("[Supervisor] Computing iteration metrics...")
        metrics = _compute_metrics(job.enriched_json, job.output_json)
        metrics_path = iter_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        artifacts["metrics.json"] = str(metrics_path)
        print(f"[Supervisor] Metrics: {metrics}")

        job.finalize_iteration(record, metrics, artifacts)

        # 8. Increment
        job.iteration += 1
        print(f"[Supervisor] Iteration complete. Next: iter_{job.iteration:03d}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_prompt_job(
        self,
        job: FabricationJob,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run the full pipeline from a prompt.

        1. Call NamedObjectAgent to generate design_json.
        2. Run _run_pipeline (validate -> reduce -> MA -> overrides -> endpoint -> GWL -> save -> metrics).

        Parameters
        ----------
        job : FabricationJob
            Job object. job.iteration must be 0 for first call.
        overrides : dict | None
            Optional parameter overrides for this iteration only.
            NOT stored on the job object; recorded in iteration_metadata.
        """
        print(f"[Supervisor] run_prompt_job | job_id={job.job_id!r}")

        # Import NamedObjectAgent's run_design at call time to avoid circular imports
        import NamedObjectAgent as noa

        # Generate design via agent
        print(f"[Supervisor] Generating design from prompt...")
        result = noa.run_design(job.base_prompt)

        # Load the design from the output path (agent writes it to disk)
        # We read it back to be consistent with the disk-canonical state model
        output_path = Path(result["output_path"])
        design_file = output_path / "design.json"
        with open(design_file, "r") as f:
            design_on_disk = json.load(f)

        # The NamedObjectAgent already ran its own pipeline internally.
        # We capture the design json and re-run WITH supervisor control.
        # Keep only the core schema fields to avoid legacy metadata keys.
        job.design_json = {
            "job_name": design_on_disk.get("job_name", job.job_id),
            "objects": design_on_disk["objects"],
            "assembly": design_on_disk["assembly"],
        }

        job.log("PROMPT_GENERATED", {
            "prompt": job.base_prompt,
            "job_name": job.design_json["job_name"],
            "agent_output_path": str(output_path),
        })

        self._run_pipeline(job, overrides=overrides, redesign_edits=None)

    def apply_redesign(
        self,
        job: FabricationJob,
        edits: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Deterministically modify the previous iteration's design and re-run.

        - Loads design.json from the PREVIOUS iteration's folder on disk.
        - Applies dot-path edits (via dot_path_editor).
        - Validates the modified design.
        - Re-runs full pipeline from the modified design.

        Parameters
        ----------
        job : FabricationJob
        edits : dict
            Dot-path edit dict, e.g.
            {"objects.pyramid.dimensions.base_width_um": 12}
        overrides : dict | None
            Optional per-iteration parameter overrides.
        """
        if job.iteration == 0:
            raise RuntimeError(
                "apply_redesign called before any iteration has run. "
                "Call run_prompt_job first to establish iteration 0."
            )

        print(f"[Supervisor] apply_redesign | iteration {job.iteration}")

        # Load previous design from DISK (immutable iteration contract)
        prev_design = job.get_prev_design_json(self.outputs_root)
        if prev_design is None:
            raise RuntimeError(
                f"Could not load design.json from previous iteration. "
                f"Expected: {job.get_prev_iter_dir(self.outputs_root)}/design.json"
            )

        # Strip metadata wrapper if present (NamedObjectAgent adds extra keys)
        base_design = {
            "job_name": prev_design.get("job_name", job.job_id),
            "objects": prev_design["objects"],
            "assembly": prev_design["assembly"],
        }

        # Apply dot-path edits to a deep copy
        modified_design, change_log = apply_dot_path_edits(base_design, edits)

        # Log changes
        job.log("REDESIGN_APPLIED", {
            "iteration": job.iteration,
            "edits": edits,
            "change_log": change_log,
        })
        for entry in change_log:
            print(f"  {entry}")

        # Set as current design (will be validated inside _run_pipeline)
        job.design_json = modified_design

        # Run full pipeline with validation + re-run
        self._run_pipeline(job, overrides=overrides, redesign_edits=edits)

    def apply_overrides(
        self,
        job: FabricationJob,
        overrides: Dict[str, Any],
    ) -> None:
        """
        Re-run only the post-MA stages with parameter overrides applied.

        This does NOT regenerate the design or re-reduce.
        It uses the CURRENT iteration's enriched_json as the base.

        Useful for quick parameter sweeps without LLM calls.

        Parameters
        ----------
        job : FabricationJob
            Must have enriched_json populated (i.e., at least one run done).
        overrides : dict
        """
        if job.enriched_json is None:
            raise RuntimeError(
                "apply_overrides called but job.enriched_json is None. "
                "Run run_prompt_job or apply_redesign first."
            )

        print(f"[Supervisor] apply_overrides | iteration {job.iteration}")

        # Reuse current design (no LLM, no reduction, no MA)
        # We do NOT call _run_pipeline fully; instead run override-only path.
        iter_dir = job.get_iter_dir(self.outputs_root)
        if iter_dir.exists():
            raise RuntimeError(f"Iteration directory already exists: {iter_dir}")

        record = job.open_iteration_record(overrides=overrides)
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Override enriched (deep copy)
        adjusted = _apply_overrides_to_enriched(
            job.enriched_json, overrides, job.feedback_log
        )

        # Generate endpoints from adjusted
        print_params = endpoint_generator_v2.load_print_parameters(self.param_file)
        job.output_json = endpoint_generator_v2.generate_endpoint_json_v2(
            adjusted, print_params
        )

        # Carry forward design + reduced + enriched from previous iteration
        # (they are semantically the same; only endpoint changes)
        prev_dir = job.get_prev_iter_dir(self.outputs_root)
        for fname in ("design.json", "reduced.json", "enriched_reduced.json"):
            src = prev_dir / fname if prev_dir else None
            if src and src.exists():
                import shutil
                shutil.copy2(src, iter_dir / fname)

        # Save overridden enriched + output + feedback
        with open(iter_dir / "enriched_reduced.json", "w") as f:
            json.dump(adjusted, f, indent=2)
        with open(iter_dir / "output.json", "w") as f:
            json.dump(job.output_json, f, indent=2)
        with open(iter_dir / "feedback_log.json", "w") as f:
            json.dump(job.feedback_log, f, indent=2)

        gwl_params = gwl_serializer.load_gwl_parameters(self.param_file)
        gwl_dir = iter_dir / "gwl"
        gwl_files = gwl_serializer.generate_gwl_files(job.output_json, gwl_params, gwl_dir)
        job.gwl_path = gwl_dir

        if gwl_files:
            job_name = (job.design_json or {}).get("job_name", job.job_id)
            gwl_serializer.generate_master_gwl(
                gwl_files, gwl_params, gwl_dir / f"{job_name}_master.gwl"
            )

        metrics = _compute_metrics(job.enriched_json, job.output_json)
        with open(iter_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        artifacts = {
            "design.json":          str(iter_dir / "design.json"),
            "reduced.json":         str(iter_dir / "reduced.json"),
            "enriched_reduced.json": str(iter_dir / "enriched_reduced.json"),
            "output.json":          str(iter_dir / "output.json"),
            "metrics.json":         str(iter_dir / "metrics.json"),
            "feedback_log.json":    str(iter_dir / "feedback_log.json"),
            "gwl/":                 str(gwl_dir),
        }

        job.finalize_iteration(record, metrics, artifacts)
        job.iteration += 1
        print(f"[Supervisor] Override iteration complete. Next: iter_{job.iteration:03d}")

    def compute_iteration_metrics(self, job: FabricationJob) -> Dict[str, Any]:
        """
        (Re)compute metrics for the current job state.

        Useful for manual inspection without running the full pipeline.
        Does NOT append to metrics_history.

        Returns
        -------
        metrics : dict
        """
        if job.enriched_json is None or job.output_json is None:
            raise RuntimeError("enriched_json or output_json is None. Run pipeline first.")
        return _compute_metrics(job.enriched_json, job.output_json)
