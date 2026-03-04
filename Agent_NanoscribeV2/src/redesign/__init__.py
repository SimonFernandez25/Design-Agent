"""
Redesign Package

This package provides the complete redesign workflow for iteratively
modifying Nanoscribe geometries based on natural language prompts.

Components:
    - design_variant: Immutable variant tracking
    - edit_schema: Structured edit operation definitions
    - edit_suggestion_agent: LLM-powered edit generation
    - edit_executor: Deterministic edit application
    - redesign_agent: Main orchestration
"""

from .design_variant import (
    DesignVariant,
    compute_theta,
    create_initial_variant,
    create_child_variant
)

from .edit_schema import (
    EditOperation,
    EditPlan,
    validate_edit_plan,
    format_edit_plan_for_display
)

from .edit_suggestion_agent import analyze_redesign_prompt

from .edit_executor import apply_edit_plan

from .redesign_agent import RedesignSession, run_redesign
