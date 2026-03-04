"""
fabrication_job.py - FabricationJob Abstraction

Tracks all state for a single fabrication job across multiple iterations.

Design decisions (per reviewer feedback):
- Overrides are NOT stored at the job level. They are passed per-call and
  recorded inside each iteration's iteration_metadata entry.
- design_json is never mutated in place. Each iteration reads from the
  previous iteration's design.json on disk (or from initial generation).
- metrics_history and feedback_log are append-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FabricationJob:
    """
    Represents a single fabrication job with full iteration history.

    Attributes
    ----------
    job_id : str
        Unique identifier for this job. Used as the output folder name.
    base_prompt : str
        The original design prompt. Immutable after creation.
    iteration : int
        Current iteration counter (0-indexed). Incremented by Supervisor
        after each completed pipeline run.

    Design state  (updated each iteration, but never mutated in-place)
    ---------------------------------------------------------------
    design_json : dict | None
        The named-object design produced by NamedObjectAgent (or last
        loaded from disk).  Each iteration writes its own copy to disk.
    reduced_json : dict | None
        Flat-primitive output from reduction_engine.
    enriched_json : dict | None
        Enriched output from manufacturing_agent (baseline, NOT overridden).
    output_json : dict | None
        Endpoint JSON from endpoint_generator_v2. May be from the
        overridden enriched copy.
    gwl_path : Path | None
        Path to the GWL output folder for the current iteration.

    Bookkeeping (append-only)
    -------------------------
    feedback_log : list[dict]
        One entry per event: redesign edits applied, overrides applied,
        validation notes, etc.
    metrics_history : list[dict]
        One entry per completed iteration containing metrics + delta.
    iteration_metadata : list[dict]
        Structured per-iteration record:
        {iteration, redesign_edits, overrides, metrics, artifacts}
    """

    job_id: str
    base_prompt: str
    iteration: int = 0

    # Pipeline state (snapshot of latest iteration)
    design_json: Optional[Dict[str, Any]] = field(default=None, repr=False)
    reduced_json: Optional[Dict[str, Any]] = field(default=None, repr=False)
    enriched_json: Optional[Dict[str, Any]] = field(default=None, repr=False)
    output_json: Optional[Dict[str, Any]] = field(default=None, repr=False)
    gwl_path: Optional[Path] = field(default=None)

    # Append-only logs
    feedback_log: List[Dict[str, Any]] = field(default_factory=list)
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    iteration_metadata: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_iter_dir(self, outputs_root: Path) -> Path:
        """
        Return the output directory for the CURRENT iteration.

        Path: <outputs_root>/<job_id>/iter_NNN/

        The directory is NOT created here; the Supervisor is responsible
        for creating it before writing files.
        """
        return outputs_root / self.job_id / f"iter_{self.iteration:03d}"

    def get_prev_iter_dir(self, outputs_root: Path) -> Optional[Path]:
        """
        Return the output directory for the PREVIOUS iteration, or None
        if this is the first iteration.
        """
        if self.iteration == 0:
            return None
        prev = self.iteration - 1
        return outputs_root / self.job_id / f"iter_{prev:03d}"

    def get_prev_design_json(self, outputs_root: Path) -> Optional[Dict[str, Any]]:
        """
        Load and return the design.json from the previous iteration.

        Returns None if no previous iteration exists.
        This is the canonical source of truth for redesign -- never rely on
        in-memory job.design_json as the mutation base.
        """
        prev_dir = self.get_prev_iter_dir(outputs_root)
        if prev_dir is None:
            return None
        design_path = prev_dir / "design.json"
        if not design_path.exists():
            return None
        with open(design_path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Append a typed entry to feedback_log."""
        self.feedback_log.append({"event": event_type, **payload})

    def open_iteration_record(
        self,
        redesign_edits: Optional[Dict] = None,
        overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Create the structured metadata record for the current iteration.
        Called by Supervisor at the start of each pipeline run.
        Returns the record dict so Supervisor can append artifacts/metrics to it.
        """
        record: Dict[str, Any] = {
            "iteration": self.iteration,
            "redesign_edits": redesign_edits or {},
            "overrides": overrides or {},
            "metrics": {},
            "artifacts": {},
        }
        self.iteration_metadata.append(record)
        return record

    def finalize_iteration(
        self,
        record: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Dict[str, str],
    ) -> None:
        """
        Fill in metrics and artifact paths into an open iteration record.
        Also appends the metrics to metrics_history with delta.
        """
        record["metrics"] = metrics
        record["artifacts"] = artifacts

        # Compute delta vs previous iteration
        delta: Dict[str, Any] = {}
        if self.metrics_history:
            prev = self.metrics_history[-1]
            for key, val in metrics.items():
                if key == "bounding_box":
                    continue  # Skip nested dict for simple delta
                try:
                    delta[key] = round(float(val) - float(prev.get(key, val)), 6)
                except (TypeError, ValueError):
                    pass

        self.metrics_history.append({**metrics, "delta": delta})

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"FabricationJob(id={self.job_id!r}, "
            f"iteration={self.iteration}, "
            f"metrics_iterations={len(self.metrics_history)}, "
            f"log_entries={len(self.feedback_log)})"
        )
