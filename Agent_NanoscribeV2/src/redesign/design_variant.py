"""
Design Variant - Stateful Redesign Variant Tracking

DesignVariant represents one immutable step in a redesign lineage.
Variants are append-only and never modified after creation.

Purpose:
    The redesign system maintains a history of all design iterations.
    Each variant records its parent, enabling full traceability of
    design evolution.

Key Invariants:
    1. Variants are immutable - once created, a variant is never edited
    2. Parent is explicit - every variant records its parent_id
    3. Filesystem mirrors memory - on-disk structure matches in-memory list

Author: Nanoscribe Design Agent Team
"""

from typing import TypedDict, Any, Dict, List


class DesignVariant(TypedDict):
    """
    Represents one immutable step in a redesign lineage.
    
    Attributes:
        variant_id: Unique integer identifier for this variant (0 for initial)
        parent_id: The variant_id of the parent, or None for initial design
        unit_cell_json: The complete unit cell JSON dictionary
        applied_edit_plan: The list of edit operations that produced this variant
        prompt_history: Cumulative list of all prompts leading to this variant
        theta: A simple recency score computed by compute_theta
    """
    variant_id: int
    parent_id: int | None
    unit_cell_json: Dict[str, Any]
    applied_edit_plan: List[Dict[str, Any]]
    prompt_history: List[str]
    theta: float


def compute_theta(prompt_history: List[str], latest_prompt: str, epsilon: float = 0.1) -> float:
    """
    Compute a simple recency score for a design variant.
    
    theta(x) is the weighted sum of all redesign prompts applied to reach 
    this variant. This function is intentionally simple:
        - Every prompt in history contributes +1
        - The latest prompt contributes +1 + epsilon
    
    This tracks "distance from original" for debugging and analysis.
    
    Args:
        prompt_history: List of all prompts from ancestors (excluding latest)
        latest_prompt: The current prompt being applied
        epsilon: Small recency bonus (default 0.1)
    
    Returns:
        The theta score for this variant
    """
    return len(prompt_history) + 1 + epsilon


def create_initial_variant(unit_cell_json: Dict[str, Any], 
                           initial_prompt: str = "Initial design") -> DesignVariant:
    """
    Create the initial variant (variant_0) from an original unit cell.
    
    This is the root of the variant tree. All subsequent variants will
    trace their lineage back to this variant.
    
    Args:
        unit_cell_json: The original unit cell JSON from geometry agent
        initial_prompt: Description for the initial state
    
    Returns:
        The initial DesignVariant with variant_id=0
    """
    return DesignVariant(
        variant_id=0,
        parent_id=None,
        unit_cell_json=unit_cell_json,
        applied_edit_plan=[],
        prompt_history=[initial_prompt],
        theta=compute_theta([], initial_prompt)
    )


def create_child_variant(
    parent: DesignVariant,
    new_unit_cell_json: Dict[str, Any],
    edit_plan: List[Dict[str, Any]],
    current_prompt: str
) -> DesignVariant:
    """
    Create a new child variant from a parent variant.
    
    This enforces the invariant that parent_id is always explicitly set.
    The new variant inherits the parent's prompt history and extends it.
    
    Args:
        parent: The parent DesignVariant
        new_unit_cell_json: The modified unit cell JSON
        edit_plan: The edit plan that was applied (list of operations)
        current_prompt: The user's redesign prompt
    
    Returns:
        A new DesignVariant with incremented variant_id
    """
    new_prompt_history = parent["prompt_history"] + [current_prompt]
    return DesignVariant(
        variant_id=parent["variant_id"] + 1,
        parent_id=parent["variant_id"],
        unit_cell_json=new_unit_cell_json,
        applied_edit_plan=edit_plan,
        prompt_history=new_prompt_history,
        theta=compute_theta(parent["prompt_history"], current_prompt)
    )
